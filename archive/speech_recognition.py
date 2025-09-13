#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first:  pip install sounddevice")
    sys.exit(-1)

import sherpa_onnx


def assert_file_exists(filename: str):
    if not Path(filename).is_file():
        raise FileNotFoundError(
            f"{filename} does not exist!\n"
            "Please see:\n"
            "  https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html"
        )


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Mic → VAD segmenter → sherpa-onnx OfflineRecognizer (Parakeet/NeMo Transducer)",
    )
    # model paths
    p.add_argument("--tokens", type=str, required=True)
    p.add_argument("--encoder", type=str, required=True)
    p.add_argument("--decoder", type=str, required=True)
    p.add_argument("--joiner", type=str, required=True)
    p.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="greedy_search or modified_beam_search",
    )
    p.add_argument("--provider", type=str, default="cpu", help="cpu, cuda, coreml")
    p.add_argument("--hotwords-file", type=str, default="")
    p.add_argument("--hotwords-score", type=float, default=1.5)
    p.add_argument("--blank-penalty", type=float, default=0.0)
    p.add_argument("--hr-dict-dir", type=str, default="")
    p.add_argument("--hr-lexicon", type=str, default="")
    p.add_argument("--hr-rule-fsts", type=str, default="")

    # audio / device
    p.add_argument("--list-devices", action="store_true", help="List devices and exit")
    p.add_argument(
        "--input-device",
        type=str,
        default=None,
        help="Device index or substring of device name",
    )
    p.add_argument(
        "--mic-rate",
        type=int,
        default=48000,
        help="Mic sampling rate (resampled internally)",
    )
    p.add_argument(
        "--channels", type=int, default=2, choices=[1, 2], help="Input channels to open"
    )
    p.add_argument(
        "--mono-mode",
        type=str,
        default="mix",
        choices=["mix", "ch0", "ch1"],
        help="How to collapse multi-channel to mono for VAD/ASR",
    )

    # VAD/segmentation
    p.add_argument(
        "--silence-ms", type=int, default=800, help="Silence required to end a segment"
    )
    p.add_argument(
        "--pre-roll-ms",
        type=int,
        default=400,
        help="Audio kept before first detected speech",
    )
    p.add_argument(
        "--min-utterance-ms",
        type=int,
        default=350,
        help="Discard segments shorter than this",
    )
    p.add_argument(
        "--energy-thresh",
        type=float,
        default=0.003,
        help="Absolute RMS fallback threshold",
    )
    p.add_argument(
        "--dynamic-thresh",
        action="store_true",
        help="Use adaptive threshold (noise_floor × dyn-factor)",
    )
    p.add_argument(
        "--dyn-factor",
        type=float,
        default=3.0,
        help="Multiplier for adaptive threshold",
    )
    p.add_argument("--block-ms", type=int, default=30, help="Audio block size in ms")
    p.add_argument(
        "--calibrate-sec",
        type=float,
        default=0.8,
        help="Seconds to learn noise floor before VAD (0 to skip)",
    )
    p.add_argument("--debug-vu", action="store_true", help="Print a simple VU meter")

    return p.parse_args()


def create_recognizer(args):
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        provider=args.provider,
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
        blank_penalty=args.blank_penalty,
        hr_dict_dir=args.hr_dict_dir,
        hr_rule_fsts=args.hr_rule_fsts,
        hr_lexicon=args.hr_lexicon,
        model_type="nemo_transducer",
    )
    return recognizer


def pick_input_device(name_or_index):
    if name_or_index is None:
        return sd.default.device[0]
    devices = sd.query_devices()
    if name_or_index.isdigit():
        return int(name_or_index)
    # substring match on name
    low = name_or_index.lower()
    for i, d in enumerate(devices):
        if low in d["name"].lower():
            return i
    raise ValueError(f"Input device '{name_or_index}' not found")


def to_mono(block: np.ndarray, mode: str) -> np.ndarray:
    # block shape: (frames,) for mono or (frames, channels) for multi
    if block.ndim == 1:
        return block.astype(np.float32)
    if mode == "ch0":
        return block[:, 0].astype(np.float32)
    if mode == "ch1":
        return block[:, 1].astype(np.float32)
    # mix/average
    return block.mean(axis=1).astype(np.float32)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def main():
    args = get_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # select device
    try:
        in_dev = pick_input_device(
            str(args.input_device) if args.input_device is not None else None
        )
    except Exception as e:
        print(f"Device selection error: {e}")
        print(sd.query_devices())
        return

    devices = sd.query_devices()
    print(f"🎙️ Using device: {devices[in_dev]['name']}")

    rec = create_recognizer(args)
    print("Ready. Start speaking… (Ctrl+C to quit)")

    mic_rate = args.mic_rate
    block_samples = int(mic_rate * args.block_ms / 1000.0)
    silence_s = args.silence_ms / 1000.0
    min_utt_s = args.min_utterance_ms / 1000.0
    pre_roll = int(mic_rate * args.pre_roll_ms / 1000.0)

    # VAD state
    active = False
    last_voice_t = 0.0
    start_t = time.monotonic()

    # Buffers
    preroll_buf = np.zeros(pre_roll, dtype=np.float32)
    seg_bufs = []

    # Adaptive threshold (EMA of noise when inactive)
    noise_ema = 0.003
    ema_alpha = 0.05

    def current_time():
        return time.monotonic() - start_t

    # Optional calibration phase to learn noise floor
    def calibrate(stream, seconds: float):
        nonlocal noise_ema
        if seconds <= 0:
            return
        n_blocks = max(1, int(seconds * mic_rate / block_samples))
        vals = []
        for _ in range(n_blocks):
            data, _ = stream.read(block_samples)
            mono = to_mono(data, args.mono_mode)
            vals.append(rms(mono))
        if vals:
            noise_ema = float(np.median(vals))
            print(f"🔧 Calibrated noise floor: {noise_ema:.6f}")

    try:
        with sd.InputStream(
            device=(in_dev, None),
            channels=args.channels,
            dtype="float32",
            samplerate=mic_rate,
            blocksize=block_samples,
        ) as stream:
            calibrate(stream, args.calibrate_sec)

            seg_idx = 1
            vu_tick = time.monotonic()
            while True:
                data, _ = stream.read(block_samples)  # blocking read
                mono = to_mono(data, args.mono_mode)

                # Update pre-roll ring buffer
                if pre_roll > 0:
                    preroll_buf = np.concatenate([preroll_buf, mono])[-pre_roll:]

                # VAD decision

                e = rms(mono)

                # --- compute trigger (don't let it be exactly 0)
                if args.dynamic_thresh:
                    if not active:
                        # only learn noise when inactive
                        noise_ema = (1 - ema_alpha) * noise_ema + ema_alpha * e
                    trigger = max(1e-7, noise_ema * args.dyn_factor)
                else:
                    trigger = max(1e-7, args.energy_thresh)

                if not active:
                    if e > trigger:
                        active = True
                        last_voice_t = current_time()
                        seg_bufs = [preroll_buf.copy(), mono]
                        print(
                            f"\n▶️  Segment {seg_idx} started (trigger={trigger:.6f}, rms={e:.6f})"
                        )
                else:
                    seg_bufs.append(mono)
                    if e > trigger:
                        last_voice_t = current_time()

                    if current_time() - last_voice_t >= silence_s:
                        wav = np.concatenate(seg_bufs).astype(np.float32)
                        dur = wav.size / mic_rate
                        active = False
                        seg_bufs = []

                        if dur < min_utt_s:
                            print(
                                f"⏹️  Segment {seg_idx} too short ({dur:.2f}s). Discarded."
                            )
                            seg_idx += 1
                            continue

                        print(f"⏹️  Segment {seg_idx} ended ({dur:.2f}s). Decoding…")
                        off = rec.create_stream()
                        off.accept_waveform(mic_rate, wav)
                        rec.decode_stream(off)
                        result = rec.get_result(off)
                        text = getattr(result, "text", "").strip()
                        print("📝", text if text else "(no text)")
                        seg_idx += 1

                # --- VU meter (print the same trigger we actually used)
                if args.debug_vu and (time.monotonic() - vu_tick) > 0.2:
                    bar = "#" * int(min(50, e / (trigger + 1e-12) * 10))
                    print(
                        f"VU rms={e:.6f}  trig={trigger:.6f} [{bar:<50}]\r",
                        end="",
                        flush=True,
                    )
                    vu_tick = time.monotonic()
                else:
                    seg_bufs.append(mono)
                    # refresh last voice time if above trigger
                    trig = (
                        (noise_ema * args.dyn_factor)
                        if args.dynamic_thresh
                        else args.energy_thresh
                    )
                    if e > trig:
                        last_voice_t = current_time()

                    # End if long enough silence
                    if current_time() - last_voice_t >= silence_s:
                        wav = np.concatenate(seg_bufs).astype(np.float32)
                        dur = wav.size / mic_rate
                        active = False
                        seg_bufs = []

                        if dur < min_utt_s:
                            print(
                                f"⏹️  Segment {seg_idx} too short ({dur:.2f}s). Discarded."
                            )
                            seg_idx += 1
                            continue

                        print(f"⏹️  Segment {seg_idx} ended ({dur:.2f}s). Decoding…")

                        # Decode this segment offline
                        off = rec.create_stream()
                        off.accept_waveform(mic_rate, wav)
                        rec.decode_stream(off)
                        result = rec.get_result(off)

                        text = getattr(result, "text", "").strip()
                        print("📝", text if text else "(no text)")
                        seg_idx += 1

                # Optional VU meter / debug
                if args.debug_vu and (time.monotonic() - vu_tick) > 0.2:
                    trig = (
                        (noise_ema * args.dyn_factor)
                        if args.dynamic_thresh
                        else args.energy_thresh
                    )
                    bar = "#" * int(min(50, e / (trig + 1e-12) * 10))
                    print(
                        f"VU rms={e:.6f}  trig~{trig:.6f} [{bar:<50}]\r",
                        end="",
                        flush=True,
                    )
                    vu_tick = time.monotonic()

    except KeyboardInterrupt:
        print("\nBye!")
    except Exception as e:
        print(f"\nStream error: {e}")
        print(
            "Tip: try --channels 2 --mono-mode ch1  (or ch0), or lower --energy-thresh, "
            "or enable --dynamic-thresh and --calibrate-sec 1.0"
        )


if __name__ == "__main__":
    main()
