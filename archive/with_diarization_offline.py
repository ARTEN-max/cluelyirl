import math
import sys
import queue
from pathlib import Path
import os
import time
import json
import wave
from dataclasses import dataclass
from typing import Optional, List
from contextlib import contextmanager

import click
import numpy as np
import sounddevice as sd
import sherpa_onnx as so

# Optional, improves diarization quality if available
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None

try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.metrics import silhouette_score  # type: ignore
except Exception:  # pragma: no cover
    KMeans = None
    silhouette_score = None


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
class Utterance:
    start_s: float
    end_s: float
    text: str
    audio: np.ndarray
    speaker: Optional[int] = None  # 0-based label assigned during diarization


# ---------- Feature extraction & clustering for simple diarization ----------


def _basic_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Return a small feature vector if librosa is unavailable.
    Features: log-energy mean/std, zero-crossing rate, spectral centroid.
    """
    y = y.astype(np.float32)
    eps = 1e-10
    e = np.mean(y * y) + eps
    le_mean = float(np.log(e))
    le_std = float(np.std(np.log(y * y + eps)))

    # Zero-crossing rate
    zc = np.mean(np.abs(np.diff(np.sign(y)))) * 0.5

    # Spectral centroid (single frame over the whole signal)
    mag = np.abs(np.fft.rfft(y, n=min(len(y), 1 << 14))) + eps
    freqs = np.fft.rfftfreq(len(mag) * 2 - 2, d=1.0 / sr)
    sc = float(np.sum(freqs * mag) / np.sum(mag + eps)) / (sr / 2.0)

    return np.array([le_mean, le_std, zc, sc], dtype=np.float32)


def compute_embedding(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute a per-utterance embedding for clustering speakers.
    Uses librosa mel features if available; otherwise falls back to basic features.
    """
    if librosa is not None:
        # 25ms windows, 10ms hop
        n_fft = max(256, int(0.025 * sr))
        hop = max(128, int(0.010 * sr))
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=40
        )
        logS = librosa.power_to_db(S + 1e-10)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)
        cent = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop
        )
        # Mean + std pooling
        emb = np.concatenate(
            [
                logS.mean(axis=1),
                logS.std(axis=1),
                zcr.mean(axis=1),
                cent.mean(axis=1) / (sr / 2.0),
            ]
        ).astype(np.float32)
        return emb
    else:
        return _basic_features(y, sr)


def choose_num_speakers(X: np.ndarray, max_k: int = 4) -> int:
    n = len(X)
    if n <= 1:
        return 1
    if KMeans is None or silhouette_score is None:
        return 2 if n >= 2 else 1

    best_k, best_score = 1, -1.0
    for k in range(2, min(max_k, n) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_speakers(
    embeddings: List[np.ndarray], speakers: Optional[int] = None
) -> List[int]:
    if not embeddings:
        return []
    X = np.stack(embeddings)

    if speakers is None:
        speakers = choose_num_speakers(X)

    if speakers <= 1 or KMeans is None:
        return [0] * len(embeddings)

    km = KMeans(n_clusters=speakers, n_init=20, random_state=0).fit(X)
    return list(map(int, km.labels_))


# ---------- CLI & real-time loop ----------


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
    "--speakers",
    type=int,
    default=None,
    help="If set, fixes number of speakers for diarization; otherwise auto-estimated",
)
@click.option(
    "--save-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write full diarized transcript as JSON",
)
@click.option(
    "--save-srt",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write diarized subtitles (.srt)",
)
@click.option(
    "--save-wav",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write captured audio as a mono WAV",
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
    speakers: Optional[int],
    save_json: Optional[Path],
    save_srt: Optional[Path],
    save_wav: Optional[Path],
    debug: bool,
):
    """
    Real-time STT from the microphone using sherpa-onnx NeMo Parakeet-TDT 0.6B v2.
    Prints one line per spoken segment. Ctrl+C to exit.

    NEW: When you exit, prints a full diarized transcript and optionally saves JSON/SRT/WAV.
    """
    if list_devices:
        for i, d in enumerate(sd.query_devices()):
            print(
                f"{i:>3}: {d['name']}  ({d['max_input_channels']} in, {d['max_output_channels']} out)"
            )
        return

    recognizer = make_recognizer(model_dir, provider, num_threads)

    # Queued audio chunks (100ms each)
    q: queue.Queue[np.ndarray] = queue.Queue()
    chunk_ms = 100
    frames_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
    min_frames = int(min_speech_ms / chunk_ms)
    silence_frames_needed = int(silence_ms / chunk_ms)

    # Accumulators for final pass
    utterances: List[Utterance] = []
    embeddings: List[np.ndarray] = []
    all_audio_chunks: List[np.ndarray] = []

    samples_received = 0  # total samples seen so far
    segment_start_sample: Optional[int] = None

    def on_audio(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # indata is float32 in [-1, 1], mono (we request channels=1)
        q.put(indata.copy().reshape(-1))

    print("Listening… (Ctrl+C to end and print full transcript)")
    buf: list[np.ndarray] = []
    silent_frames = 0
    speech_seen = False
    chunks_since_meter = 0

    def decode_current_buffer(end_is_silence: bool = True):
        nonlocal buf, silent_frames, speech_seen, segment_start_sample
        if not buf:
            return
        # Concatenate and, for diarization embedding, trim trailing endpoint silence
        samples = np.concatenate(buf).astype(np.float32)
        trailing_silence = silent_frames * frames_per_chunk if end_is_silence else 0
        speech_len = max(0, len(samples) - trailing_silence)
        speech_samples = samples[:speech_len] if speech_len > 0 else samples

        # Determine segment boundaries in absolute sample counts
        if segment_start_sample is None:
            start_abs = samples_received - len(samples)
        else:
            start_abs = segment_start_sample
        end_abs = samples_received - (
            silent_frames * frames_per_chunk if end_is_silence else 0
        )

        # Decode
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_streams([stream])  # batch of one
        text = (stream.result.text or "").strip()
        if text:
            print("\n" + text) if debug else print(text)

        # Save utterance & embedding for diarization
        if speech_samples.size == 0:
            # If we trimmed everything, keep some audio to compute features
            speech_samples = samples
        emb = compute_embedding(speech_samples, sample_rate)
        embeddings.append(emb)
        utterances.append(
            Utterance(
                start_s=start_abs / sample_rate,
                end_s=end_abs / sample_rate,
                text=text,
                audio=speech_samples.copy(),
            )
        )

        # Reset state for next utterance
        buf.clear()
        silent_frames = 0
        speech_seen = False
        segment_start_sample = None

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
                all_audio_chunks.append(data)

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
                    if not speech_seen:
                        segment_start_sample = samples_received
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
                    decode_current_buffer(end_is_silence=True)

                # Advance absolute time *after* processing this chunk
                samples_received += len(data)

    except KeyboardInterrupt:
        print("\nFinalizing…")
        # If there's leftover speech in the buffer, decode it as a final utterance
        if speech_seen and len(buf) >= max(1, min_frames // 2):
            decode_current_buffer(end_is_silence=False)

        # Save WAV if requested
        if save_wav is not None and all_audio_chunks:
            y = np.concatenate(all_audio_chunks).astype(np.float32)
            # WAVs are typically int16; we'll scale safely
            pcm = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(str(save_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())
            print(f"Saved audio: {save_wav}")

        # Diarization via clustering utterance embeddings
        if utterances:
            labels = cluster_speakers(embeddings, speakers=speakers)
            # Normalize labels to consecutive 1..K ordering by time-first appearance
            remap = {}
            next_id = 1
            for u, lab in zip(utterances, labels):
                if lab not in remap:
                    remap[lab] = next_id
                    next_id += 1
                u.speaker = remap[lab]

            # Print nice console transcript
            print("\n=== Conversation Transcript (diarized) ===")
            for i, u in enumerate(sorted(utterances, key=lambda x: x.start_s), start=1):
                print(
                    f"[{_fmt_ts(u.start_s)} - {_fmt_ts(u.end_s)}] SPK{u.speaker}: {u.text}"
                )

            # Save JSON if requested
            if save_json is not None:
                payload = {
                    "sample_rate": sample_rate,
                    "num_speakers": len(
                        set(u.speaker for u in utterances if u.speaker is not None)
                    ),
                    "utterances": [
                        {
                            "start": u.start_s,
                            "end": u.end_s,
                            "speaker": int(u.speaker)
                            if u.speaker is not None
                            else None,
                            "text": u.text,
                        }
                        for u in utterances
                    ],
                }
                save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"Saved JSON transcript: {save_json}")

            # Save SRT if requested
            if save_srt is not None:
                srt = _to_srt(utterances)
                save_srt.write_text(srt, encoding="utf-8")
                print(f"Saved SRT: {save_srt}")
        else:
            print("No speech captured.")

        print("Bye!")


def _fmt_ts(t: float) -> str:
    m, s = divmod(max(0.0, t), 60.0)
    return f"{int(m):02d}:{s:06.3f}"


def _fmt_ts_srt(t: float) -> str:
    # SRT format: HH:MM:SS,mmm
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000.0))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _to_srt(utterances: List[Utterance]) -> str:
    lines = []
    for idx, u in enumerate(sorted(utterances, key=lambda x: x.start_s), start=1):
        lines.append(str(idx))
        lines.append(f"{_fmt_ts_srt(u.start_s)} --> {_fmt_ts_srt(u.end_s)}")
        spk = f"SPK{u.speaker}" if u.speaker is not None else "SPK?"
        text = u.text if u.text else "[inaudible]"
        lines.append(f"{spk}: {text}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
