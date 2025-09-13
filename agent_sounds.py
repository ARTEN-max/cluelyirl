from __future__ import annotations

import os
import json
import math
import functools
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
    # alternatives_considered: List[SoundCandidate] = Field(default_factory=list)


# ---------- Dependencies ----------


@dataclass
class World:
    sounds_dir: str
    log: List[Dict[str, Any]]  # in-memory event log


# ---------- Helpers (internal) ----------


def _mk_candidate(
    ctx: RunContext[World], filename: str, tags: List[str], attrs: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a SoundCandidate-like dict without touching the filesystem."""
    rel_path = filename
    return {
        "file": os.path.abspath(os.path.join(ctx.deps.sounds_dir, rel_path)),
        "rel_path": rel_path,
        "format": os.path.splitext(filename)[1].lstrip(".").lower(),
        "duration_sec": None,  # unknown (don't invent)
        "tags": sorted(set(t.strip().lower() for t in tags if t)),
        "attrs": dict(attrs or {}),
    }


# ---------- Agent ----------

sound_agent = Agent(
    "openai:gpt-4o",
    deps_type=World,
    output_type=SoundSelection,
    system_prompt=(
        "You are a careful sound selector. ONLY if appropriate choose ONE audio file that best fits the user's scenario, "
        "using ONLY the curated sound tools provided (do NOT scan the filesystem and do NOT invent metadata).\n\n"
        "Workflow:\n"
        "1) Analyze the conversation transcript provided where where the last line is what was most recently said. Only if something notable happens, look for a sound. Otherwise, don't select any file and just give a short rationale of what happened.\n"
        "2) Call list_curated_sounds() to see all available options with tags and scenario notes.\n"
        "3) If helpful, call find_curated_by_tag(tag) to narrow candidates.\n"
        "4) Prefer files whose tags/attributes most closely match the scenario (mood, sfx type, 'impact', 'stinger', 'ambient').\n"
        "5) If the scenario implies looping/background, prefer items tagged 'ambient','music','background'; "
        "   for UI cues/alerts/punchlines, prefer short 'impact','stinger','boom' items.\n"
        "6) If no exact match exists, select the closest fit and explain the trade-off.\n\n"
        "Return SoundSelection with rationale, confidence (0-1), and alternatives_considered."
    ),
)


# ---------- Curated sound tools ----------
# Add new sounds by following the same pattern: define a tool function that returns _mk_candidate(...)


@sound_agent.tool
@log_tool
def sound_cinematic_boom(ctx: RunContext[World]) -> Dict[str, Any]:
    """
    Big cinematic impact suitable for dramatic reveals, trailer hits, boss entrances,
    scene transitions, or 'fail/sudden stop' stingers.
    """
    return _mk_candidate(
        ctx,
        "cinematic_boom.m4a",
        tags=[
            "impact",
            "boom",
            "cinematic",
            "stinger",
            "trailer",
            "dramatic",
            "transition",
        ],
        attrs={
            "scenarios": "Dramatic reveal, scene transition, boss entrance, trailer hit, game over sting.",
            "intensity": "high",
            "recommended_use": "short punctuations rather than background.",
        },
    )


@sound_agent.tool
@log_tool
def sound_vine_boom(ctx: RunContext[World]) -> Dict[str, Any]:
    """
    Iconic meme 'boom' for comedic punchlines, reaction moments, and meme edits.
    Works great for short-form video, streams, and humorous overlays.
    """
    return _mk_candidate(
        ctx,
        "vine_boom.m4a",
        tags=["meme", "boom", "stinger", "comedy", "bass", "impact", "reaction"],
        attrs={
            "scenarios": "Comedic punchline, reaction cut, meme edit, stream highlight.",
            "intensity": "medium-high",
            "recommended_use": "single-hit emphasis; avoid overuse.",
        },
    )


@sound_agent.tool
@log_tool
def sound_death_note_l_theme(ctx: RunContext[World]) -> Dict[str, Any]:
    """
    Tense, investigative background music (L's theme vibe): analytical, mysterious, and focused.
    Good for deduction sequences, coding montages, stealth or crime board reveals.
    """
    return _mk_candidate(
        ctx,
        "death_note_l_theme.m4a",
        tags=[
            "music",
            "tense",
            "mysterious",
            "investigative",
            "piano",
            "strings",
            "background",
        ],
        attrs={
            "scenarios": "Detective/analysis sequence, hacker/coding montage, stealth planning, noir ambience.",
            "intensity": "low-medium",
            "recommended_use": "background underscore; may need manual looping.",
        },
    )


# ---------- Aggregation / filtering tools over the curated set ----------


@sound_agent.tool
@log_tool
def list_curated_sounds(ctx: RunContext[World]) -> List[Dict[str, Any]]:
    """Return all curated sounds with their tags and scenario notes."""
    # Call the individual tools to keep a single source of truth per sound:
    return [
        sound_cinematic_boom(ctx),
        sound_vine_boom(ctx),
        sound_death_note_l_theme(ctx),
    ]


@sound_agent.tool
@log_tool
def find_curated_by_tag(ctx: RunContext[World], tag: str) -> List[Dict[str, Any]]:
    """Filter curated sounds by tag (case-insensitive exact tag match)."""
    tag_norm = (tag or "").strip().lower()
    return [it for it in list_curated_sounds(ctx) if tag_norm in it.get("tags", [])]


# ---------- Example runner (you provide the scenario at runtime) ----------

if __name__ == "__main__":
    # Point this to your real sounds directory; no scanning is performed.
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
