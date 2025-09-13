# ---------- Non-blocking audio (5s + fade-out) ----------

import shutil
import subprocess
import threading


class AudioPlayer:
    """
    Plays a 5s segment with a tail fade-out without blocking the main thread.
    Prefers ffplay (part of ffmpeg). Falls back to pydub + sounddevice if available.
    """

    def __init__(self, fade_out_sec: float = 1.5):
        self.fade_out_sec = max(0.1, float(fade_out_sec))
        self._has_ffplay = shutil.which("ffplay") is not None
        try:
            import pydub  # noqa: F401
            import sounddevice  # noqa: F401

            self._has_pydub = True
        except Exception:
            self._has_pydub = False

    def play_5s_fade(self, file_path: str) -> None:
        """Fire-and-forget: returns immediately."""
        if self._has_ffplay:
            self._play_with_ffplay(file_path)
        elif self._has_pydub:
            self._play_with_pydub(file_path)
        else:
            raise RuntimeError(
                "No audio backend found. Install ffmpeg (ffplay) "
                "or `pip install pydub sounddevice`."
            )

    # --- backends ---

    def _play_with_ffplay(self, file_path: str) -> None:
        # Play 5s (-t 5) and fade out during the last fade_out_sec seconds.
        st = max(0.0, 5.0 - self.fade_out_sec)
        args = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-hide_banner",
            "-loglevel",
            "error",
            "-t",
            "5",
            "-af",
            f"afade=t=out:st={st}:d={self.fade_out_sec}",
            file_path,
        ]
        # Non-blocking: spawn and return.
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _play_with_pydub(self, file_path: str) -> None:
        # Load/trim/fade with pydub, play asynchronously with sounddevice.
        # (Requires ffmpeg for m4a loading)
        def _worker():
            from pydub import AudioSegment
            import numpy as np
            import sounddevice as sd

            seg = AudioSegment.from_file(file_path)[:5000]  # first 5s
            seg = seg.fade_out(int(self.fade_out_sec * 1000))
            # Convert to numpy and play non-blocking
            samples = np.array(seg.get_array_of_samples()).reshape(-1, seg.channels)
            sd.play(samples, seg.frame_rate, blocking=False)

        threading.Thread(target=_worker, daemon=True).start()
