#!/usr/bin/env python3
import math
import sys
import queue
from pathlib import Path
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import click
import numpy as np
import sounddevice as sd
import sherpa_onnx as so


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


@dataclass
class TextSeg:
    start_s: float
    end_s: float
    text: str


def make_diarizer(
    segmentation_model: Optional[Path],
    embedding_model: Optional[Path],
    provider: str,
    num_threads: int,
    cluster_threshold: Optional[float],
    num_speakers: Optional[int],
    min_on: float,
    min_off: float,
):
    """Create a sherpa-onnx OfflineSpeakerDiarization, if models provided.

    Returns (diarizer_or_None).
    """
    if not segmentation_model or not embedding_model:
        return None

    seg_cfg = so.OfflineSpeakerSegmentationModelConfig(
        pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(segmentation_model)
        ),
        provider=provider,
        num_threads=num_threads,
        debug=False,
    )
    emb_cfg = so.SpeakerEmbeddingExtractorConfig(
        model=str(embedding_model),
        provider=provider,
        num_threads=num_threads,
        debug=False,
    )
    clus_cfg = so.FastClusteringConfig(
        num_clusters=int(num_speakers or 0), threshold=float(cluster_threshold or 0.5)
    )

    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=seg_cfg,
        embedding=emb_cfg,
        clustering=clus_cfg,
        min_duration_on=float(min_on),
        min_duration_off=float(min_off),
    )
    return so.OfflineSpeakerDiarization(cfg)


def _normalize_diarizer_result(res) -> List[Tuple[float, float, str]]:
    """Return list of (start_s, end_s, speaker_label) from various possible return shapes."""
    out: List[Tuple[float, float, str]] = []
    try:
        # Expect an iterable of objects or tuples
        for x in res:
            if isinstance(x, (list, tuple)) and len(x) >= 3:
                start, end, spk = x[0], x[1], x[2]
            else:
                # object with attributes
                start = getattr(x, "start", getattr(x, "start_s", None))
                end = getattr(x, "end", getattr(x, "end_s", None))
                spk = getattr(x, "speaker", getattr(x, "label", "speaker_00"))
            out.append((float(start), float(end), str(spk)))
    except Exception:
        pass
    return out


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
# Diarization options
@click.option(
    "--diarize/--no-diarize",
    default=False,
    show_default=True,
    help="Enable speaker diarization and print a diarized transcript on exit",
)
@click.option(
    "--segmentation-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to speaker segmentation ONNX (e.g. pyannote segmentation).",
)
@click.option(
    "--embedding-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to speaker embedding ONNX (e.g. NeMo Titanet or 3D-Speaker).",
)
@click.option(
    "--diarize-num-speakers",
    type=int,
    default=None,
    help="If known, fix number of speakers for clustering. Otherwise uses --diarize-threshold.",
)
@click.option(
    "--diarize-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Clustering threshold (higher=fewer speakers) when num-speakers not set.",
)
@click.option(
    "--diarize-min-on",
    type=float,
    default=0.3,
    show_default=True,
    help="Minimum active speech duration for diarization (seconds).",
)
@click.option(
    "--diarize-min-off",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum non-speech gap for diarization (seconds).",
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
    diarize: bool,
    segmentation_model: Optional[Path],
    embedding_model: Optional[Path],
    diarize_num_speakers: Optional[int],
    diarize_threshold: float,
    diarize_min_on: float,
    diarize_min_off: float,
):
    """
    Real-time STT from the microphone using sherpa-onnx NeMo Parakeet-TDT 0.6B v2.
    Prints one line per spoken segment. Ctrl+C to exit.

    If --diarize is set, a diarized transcript (SPEAKER_xx:) is printed on exit.
    """
    if list_devices:
        for i, d in enumerate(sd.query_devices()):
            print(
                f"{i:>3}: {d['name']}  ({d['max_input_channels']} in, {d['max_output_channels']} out)"
            )
        return

    if diarize and (not segmentation_model or not embedding_model):
        raise click.UsageError(
            "--diarize requires --segmentation-model and --embedding-model."
        )

    recognizer = make_recognizer(model_dir, provider, num_threads)

    diarizer = None
    if diarize:
        diarizer = make_diarizer(
            segmentation_model,
            embedding_model,
            provider,
            num_threads,
            diarize_threshold,
            diarize_num_speakers,
            diarize_min_on,
            diarize_min_off,
        )

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
    buf: List[np.ndarray] = []
    silent_frames = 0
    speech_seen = False
    chunks_since_meter = 0

    # For diarization on exit
    session_audio_chunks: List[np.ndarray] = []
    utterances: List[TextSeg] = []
    elapsed_chunks = 0

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
                session_audio_chunks.append(data)  # keep full session audio
                elapsed_chunks += 1

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
                    samples = np.concatenate(buf).astype(np.float32)
                    buf.clear()
                    silent_frames = 0

                    stream = recognizer.create_stream()
                    stream.accept_waveform(sample_rate, samples)
                    # for offline recognition, feed once then decode
                    recognizer.decode_streams([stream])  # batch of one
                    text = (stream.result.text or "").strip()

                    # Compute segment times relative to whole session
                    end_s = elapsed_chunks * (chunk_ms / 1000.0)
                    start_s = end_s - (len(samples) / float(sample_rate))

                    if text:
                        print("\n" + text) if debug else print(text)
                        utterances.append(
                            TextSeg(start_s=start_s, end_s=end_s, text=text)
                        )
                    speech_seen = False
    except KeyboardInterrupt:
        pass

    # After exiting, optionally run diarization for the whole session and print a labeled transcript
    if diarize and utterances:
        print("\n=== Diarizing session... ===")
        session_audio = np.concatenate(session_audio_chunks).astype(np.float32)
        # Run diarization on the full audio for consistent speaker labels
        try:
            # Prefer explicit Python API if available
            # Some versions use .diarize(), others .process(); try both.
            diar_res = None
            if hasattr(diarizer, "process"):  # sherpa-onnx >= 1.10.28
                diar_res = diarizer.process(session_audio)
            elif hasattr(diarizer, "diarize"):  # older naming
                diar_res = diarizer.diarize(session_audio)
            else:
                diar_res = diarizer(session_audio)  # callable fallback
            # if hasattr(diarizer, "diarize"):
            #     diar_res = diarizer.diarize(session_audio, sample_rate)
            # elif hasattr(diarizer, "process"):
            #     diar_res = diarizer.process(session_audio, sample_rate)
            # else:
            #     diar_res = diarizer(session_audio, sample_rate)  # __call__ fallback

            segs = _normalize_diarizer_result(diar_res)
        except Exception as ex:
            print(f"(diarization failed: {ex})", file=sys.stderr)
            segs = []

        if segs:
            # Map diarization segments to utterances by overlap, then pretty-print
            def overlap(a0, a1, b0, b1):
                return max(0.0, min(a1, b1) - max(a0, b0))

            # Normalize diarizer labels to SPEAKER_01, SPEAKER_02, ... in order of first appearance
            label_remap: Dict[str, str] = {}
            next_id = 1

            print("\n=== Diarized transcript ===")
            for utt in utterances:
                # choose diarization segment with maximum overlap
                best_label = None
                best_olap = 0.0
                for s0, s1, lab in segs:
                    ol = overlap(utt.start_s, utt.end_s, s0, s1)
                    if ol > best_olap:
                        best_olap = ol
                        best_label = lab
                if best_label is None:
                    spk = "SPEAKER_??"
                else:
                    if best_label not in label_remap:
                        label_remap[best_label] = f"SPEAKER_{next_id:02d}"
                        next_id += 1
                    spk = label_remap[best_label]
                print(f"{spk}: {utt.text}")
        else:
            print("(No diarization segments produced)")

    print("\nBye!")


if __name__ == "__main__":
    cli()
