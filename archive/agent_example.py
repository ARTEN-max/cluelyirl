from __future__ import annotations

import os
import re
import json
import math
import functools
import contextlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


# ---------- Small logging helper ----------


def _ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def log_tool(fn):
    """Decorator to record tool calls + results into ctx.deps.log."""

    @functools.wraps(fn)
    def wrapper(ctx: RunContext["World"], *args, **kwargs):
        ctx.deps.log.append(
            {
                "ts": _ts(),
                "event": "tool_call",
                "tool": fn.__name__,
                "args": list(args),
                "kwargs": kwargs,
            }
        )
        result = fn(ctx, *args, **kwargs)
        ctx.deps.log.append(
            {
                "ts": _ts(),
                "event": "tool_result",
                "tool": fn.__name__,
                "result": result,
            }
        )
        return result

    return wrapper


# ---------- Output models ----------


class SoundCandidate(BaseModel):
    file: str
    rel_path: str
    format: str
    duration_sec: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    attrs: Dict[str, Any] = Field(default_factory=dict)  # any extra metadata


class SoundSelection(BaseModel):
    scenario: str
    selected_file: str
    rel_path: str
    format: str
    duration_sec: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    confidence: float
    rationale: str
    alternatives_considered: List[SoundCandidate] = Field(default_factory=list)


# ---------- Dependencies ----------


@dataclass
class World:
    sounds_dir: str
    log: List[Dict[str, Any]]  # in-memory event log


# ---------- Helpers (internal, not tools) ----------

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".aiff", ".aif"}


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _sidecar_json_for(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def _slug_to_tags(slug: str) -> List[str]:
    # infer tags from filename (e.g., "rain_loop_chill" -> ["rain","loop","chill"])
    parts = re.split(r"[_\-\.\s]+", slug.lower())
    return sorted({p for p in parts if p and not p.isdigit()})


def _read_sidecar(sidecar_path: str) -> Dict[str, Any]:
    if not os.path.exists(sidecar_path):
        return {}
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def _wav_duration_seconds(file_path: str) -> Optional[float]:
    # Stdlib duration for WAV only; other formats return None (no external deps)
    if os.path.splitext(file_path)[1].lower() != ".wav":
        return None
    try:
        import wave

        with contextlib.closing(wave.open(file_path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return round(frames / float(rate), 3)
    except Exception:
        return None


# ---------- Agent ----------

sound_agent = Agent(
    "openai:gpt-4o",
    deps_type=World,
    output_type=SoundSelection,
    system_prompt=(
        "You are a careful sound selector. Choose ONE audio file that best fits the user's scenario, "
        "using ONLY the provided tools. Do not invent metadata.\n\n"
        "Required workflow (repeat as needed):\n"
        "1) Call catalog_sounds() to list all available files with tags, duration, format, and metadata.\n"
        "2) If needed, call find_sounds_by_tag(tag) to narrow candidates.\n"
        "3) Prefer files whose tags and attributes most closely match the scenario (e.g., mood, genre, sfx type, 'loop', 'ambient', 'impact').\n"
        "4) If the scenario implies looping/background, prefer items tagged 'loop'/'loopable' or with longer durations; "
        "   for UI cues/alerts, prefer short 'click','beep','whoosh','impact' items.\n"
        "5) If no exact match exists, select the closest fit and explain the trade-off.\n\n"
        "Return SoundSelection including rationale, confidence (0-1), and alternatives_considered."
    ),
)


# ---------- Tools ----------


@sound_agent.tool
@log_tool
def catalog_sounds(ctx: RunContext[World]) -> List[Dict[str, Any]]:
    """
    Scan the sounds directory and return a catalog of files with inferred + sidecar metadata.
    Each item contains: file, rel_path, format, duration_sec (WAV only), tags, attrs.
    """
    base = ctx.deps.sounds_dir
    results: List[Dict[str, Any]] = []
    if not os.path.isdir(base):
        return results

    for root, _, files in os.walk(base):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in AUDIO_EXTS:
                continue
            full_path = os.path.join(root, fn)
            rel_path = os.path.relpath(full_path, base)
            sidecar = _read_sidecar(_sidecar_json_for(full_path))
            inferred_tags = _slug_to_tags(_stem(fn))

            tags = sorted({*(sidecar.get("tags") or []), *inferred_tags})
            attrs = {k: v for k, v in sidecar.items() if k != "tags"}

            item = {
                "file": os.path.abspath(full_path),
                "rel_path": rel_path.replace(os.sep, "/"),
                "format": ext.lstrip("."),
                "duration_sec": _wav_duration_seconds(full_path),
                "tags": tags,
                "attrs": attrs,
            }
            results.append(item)
    return results


@sound_agent.tool
@log_tool
def find_sounds_by_tag(ctx: RunContext[World], tag: str) -> List[Dict[str, Any]]:
    """
    Return catalog entries whose tags contain the given tag (case-insensitive).
    """
    tag_norm = (tag or "").strip().lower()
    return [
        it
        for it in catalog_sounds(ctx)
        if tag_norm in (t.lower() for t in it.get("tags", []))
    ]


# ---------- Example runner (you provide the scenario at runtime) ----------

if __name__ == "__main__":
    # Point this to your real sounds directory
    world = World(
        sounds_dir="sounds",
        log=[],
    )

    scenario = input(
        "Describe your scenario (e.g., 'calm looping rain ambience for a meditation app'): "
    ).strip()
    world.log.append({"ts": _ts(), "event": "user_request", "scenario": scenario})

    result = sound_agent.run_sync(scenario, deps=world)

    world.log.append(
        {"ts": _ts(), "event": "agent_output", "output": result.output.model_dump()}
    )

    print("\n---- Selection ----")
    print(result.output.model_dump_json(indent=2))

    print("\n---- Log ----")
    print(json.dumps(world.log, indent=2, default=str))
