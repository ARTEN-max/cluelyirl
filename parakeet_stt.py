#!/usr/bin/env python3
import math
import sys
import queue
from pathlib import Path
import os
from contextlib import contextmanager

import click
import numpy as np
import sounddevice as sd
import sherpa_onnx as so
from agent_sounds import sound_agent, World, _ts
from audio_player import AudioPlayer

import concurrent.futures
import itertools
import traceback


player = AudioPlayer(fade_out_sec=1.5)
# Point this to your real sounds directory; no scanning is performed.
world = World(
    sounds_dir="sounds",
    log=[],
)

# Use a small thread pool so agent runs don't block stdin
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
task_id_iter = itertools.count(1)


def process_scenario(task_id: int, scenario: str):
    try:
        world.log.append(
            {
                "ts": _ts(),
                "event": "user_request",
                "scenario": scenario,
                "task_id": task_id,
            }
        )
        result = sound_agent.run_sync(scenario, deps=world)

        sel = result.output
        # Try to resolve an absolute file path robustly:
        candidate_path = None
        if sel.selected_file and os.path.isabs(sel.selected_file):
            candidate_path = sel.selected_file
        elif sel.selected_file:
            candidate_path = os.path.abspath(
                os.path.join(world.sounds_dir, sel.selected_file)
            )
        elif sel.rel_path:
            candidate_path = os.path.abspath(
                os.path.join(world.sounds_dir, sel.rel_path)
            )

        # Log + print the selection immediately
        world.log.append(
            {
                "ts": _ts(),
                "event": "agent_output",
                "task_id": task_id,
                "output": sel.model_dump(),
            }
        )
        print(
            f"\n[#{task_id}] Selected: {os.path.basename(candidate_path) if candidate_path else '(no file)'}"
        )
        # print(result.output.model_dump_json(indent=2))
        print("\n---- Selection ----")
        print(f"scenario:{result.output.scenario}")
        print(f"rationale: {result.output.rationale}")
        print(f"confidence: {result.output.confidence}")
        print(f"selected_sound: {result.output.selected_file}")

        # Fire-and-forget playback (5s with fade)
        if candidate_path and os.path.exists(candidate_path):
            try:
                player.play_5s_fade(candidate_path)
                print(f"[#{task_id}] Playing 5s with fade-out… (non-blocking)")
            except Exception as e:
                print(f"[#{task_id}] Could not play audio: {e}")
        else:
            print(f"[#{task_id}] Warning: selected file not found; skipped audio.")

    except Exception as e:
        print(f"[#{task_id}] Error while processing scenario: {e}")
        traceback.print_exc()


print(
    "Type scenarios to test. New inputs are accepted immediately, even while the agent is working.\n"
    "Type 'quit' to exit.\n"
)


@contextmanager
def _chdir(path: Path):
    """Temporarily change working dir so ONNX external weights resolve correctly."""
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


def make_recognizer(
    model_dir: Path, provider: str, num_threads: int
) -> so.OfflineRecognizer:
    enc = model_dir / "encoder.onnx"
    dec = model_dir / "decoder.onnx"
    joi = model_dir / "joiner.onnx"
    tok = model_dir / "tokens.txt"
    for p in (enc, dec, joi, tok):
        if not p.is_file():
            raise FileNotFoundError(f"Missing model file: {p}")

    with _chdir(model_dir):
        return so.OfflineRecognizer.from_transducer(
            tokens=str(tok.name),  # absolute/relative both OK
            encoder=str(enc.name),  # pass basenames while cwd=model_dir
            decoder=str(dec.name),
            joiner=str(joi.name),
            provider=provider,
            num_threads=num_threads,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2"),
    show_default=True,
    help="Folder that contains encoder.onnx / decoder.onnx / joiner.onnx / tokens.txt",
)
@click.option(
    "--device",
    type=int,
    default=None,
    help="Input device id (use --list-devices to see ids)",
)
@click.option("--list-devices", is_flag=True, help="List audio devices and exit")
@click.option(
    "--sample-rate", type=int, default=16000, show_default=True, help="Mic sample rate"
)
@click.option(
    "--silence-ms",
    type=int,
    default=800,
    show_default=True,
    help="Silence duration to trigger a decode",
)
@click.option(
    "--min-speech-ms",
    type=int,
    default=300,
    show_default=True,
    help="Minimum speech length to decode",
)
@click.option(
    "--energy-threshold",
    type=float,
    default=1e-4,
    show_default=True,
    help="Energy threshold for speech",
)
@click.option(
    "--num-threads", type=int, default=2, show_default=True, help="ONNX Runtime threads"
)
@click.option(
    "--provider",
    type=click.Choice(["cpu", "cuda", "coreml"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="Inference provider (must be available in your sherpa-onnx build)",
)
@click.option(
    "--debug", is_flag=True, help="Print a simple audio meter while listening"
)
def cli(
    model_dir: Path,
    device: int | None,
    list_devices: bool,
    sample_rate: int,
    silence_ms: int,
    min_speech_ms: int,
    energy_threshold: float,
    num_threads: int,
    provider: str,
    debug: bool,
):
    """
    Real-time STT from the microphone using sherpa-onnx NeMo Parakeet-TDT 0.6B v2.
    Prints one line per spoken segment. Ctrl+C to exit.
    """
    if list_devices:
        for i, d in enumerate(sd.query_devices()):
            print(
                f"{i:>3}: {d['name']}  ({d['max_input_channels']} in, {d['max_output_channels']} out)"
            )
        return

    recognizer = make_recognizer(model_dir, provider, num_threads)

    q: queue.Queue[np.ndarray] = queue.Queue()
    chunk_ms = 100
    frames_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
    min_frames = int(min_speech_ms / chunk_ms)
    silence_frames_needed = int(silence_ms / chunk_ms)

    def on_audio(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # indata is float32 in [-1, 1], mono (we request channels=1)
        q.put(indata.copy().reshape(-1))

    print("Listening… (Ctrl+C to quit)")
    buf: list[np.ndarray] = []
    silent_frames = 0
    speech_seen = False
    chunks_since_meter = 0
    transcript = ""
    try:
        with sd.InputStream(
            device=device,
            channels=1,
            samplerate=sample_rate,
            dtype="float32",
            blocksize=frames_per_chunk,
            callback=on_audio,
        ):
            while True:
                data = q.get()  # blocking read of the next 100 ms
                buf.append(data)
                # simple energy-based VAD
                e = float(np.mean(np.square(data)))
                if debug:
                    chunks_since_meter += 1
                    if chunks_since_meter >= int(1000 / chunk_ms):
                        rms = math.sqrt(max(e, 1e-12))
                        dbfs = 20 * math.log10(rms + 1e-12)
                        print(f"\rlevel: {dbfs:6.1f} dBFS   ", end="", flush=True)
                        chunks_since_meter = 0
                if e > energy_threshold:
                    silent_frames = 0
                    speech_seen = True
                else:
                    silent_frames += 1

                # endpoint: enough trailing silence and enough speech collected
                if (
                    speech_seen
                    and silent_frames >= silence_frames_needed
                    and len(buf) >= min_frames
                ):
                    # if silent_frames >= silence_frames_needed and len(buf) >= min_frames:
                    samples = np.concatenate(buf).astype(np.float32)
                    buf.clear()
                    silent_frames = 0

                    stream = recognizer.create_stream()
                    stream.accept_waveform(sample_rate, samples)
                    # for offline recognition, feed once then decode
                    recognizer.decode_streams([stream])  # batch of one
                    text = (stream.result.text or "").strip()
                    if text:
                        print("\n" + text) if debug else print(text)

                        transcript += text + "\n"

                        tid = next(task_id_iter)
                        print(f"[#{tid}] Received. Working in background…")
                        executor.submit(process_scenario, tid, transcript)

                        # world.log.append(
                        #     {
                        #         "ts": _ts(),
                        #         "event": "user_request",
                        #         "scenario": transcript,
                        #     }
                        # )
                        #
                        # result = sound_agent.run_sync(
                        #     "Here is a newline delimited transcript of a conversation, the last line is what was most recently said, based on what just happened pick a suitable sound only if something notable happened recently"
                        #     + transcript,
                        #     deps=world,
                        # )
                        #
                        # world.log.append(
                        #     {
                        #         "ts": _ts(),
                        #         "event": "agent_output",
                        #         "output": result.output.model_dump(),
                        #     }
                        # )
                        #

                    speech_seen = False
    except KeyboardInterrupt:
        print(transcript)
    finally:
        executor.shutdown(wait=False)


if __name__ == "__main__":
    cli()
