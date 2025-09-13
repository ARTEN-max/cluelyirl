from __future__ import annotations

import os
import json
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

AGENT_SYSTEM_PROMPT = """You are a careful sound selector. Your PRIME DIRECTIVE is to AVOID playing any sound unless the moment is clearly warranted, safe, and context-appropriate.

You will be provided with a transcript which is a conversation between two people where each segment is prepended with a timestamp of when they were talking. The last line with the latest timestamp is what was most recently said.

ONLY if appropriate choose ONE audio file that best fits the user's scenario based on what was most recently said, using ONLY the curated sound tools provided and the provided description of the sound (do NOT scan the filesystem and do NOT invent metadata). The people in conversation may mention the sound itself, but you shouldn't play the sound unless they specifically demand to play the sound.

all of the valid sound files to choose from include: (death_note_l_theme.m4a, short_ding.m4a, short_vibration.m4a, heated_long_vibration.m4a, short_cinematic_boom.m4a, short_vibration_double.m4a, vine_boom.m4a)

Workflow:
1) Analyze the conversation transcript provided. If nothing notable happens for less than 3 consecutive turns, don't select any file—just return a short rationale.
2) If a sound is warranted, call whichever curated sound tools seem relevant for key moments (e.g., sound_cinematic_boom, sound_vine_boom, sound_death_note_l_theme) or when the ovarall conversation is reaching a point that warrants a conversational cue (for example if the conversation has been dry for 3 or more turns use the sound_short_vibration. If the conversation gets heated in an argument, play sound_heated_long_vibration) to inspect their tags/notes.
"""

sound_agent = Agent(
    "openai:gpt-4.1-nano-2025-04-14",
    deps_type=World,
    output_type=SoundSelection,
    system_prompt=AGENT_SYSTEM_PROMPT,
)


# ---------- Curated sound tools ----------


@sound_agent.tool
@log_tool
def sound_cinematic_boom(ctx: RunContext[World]) -> Dict[str, Any]:
    """
    For dramatic reveals similar to the vine-boom but more subtle for less intense scenarios.
    """
    return _mk_candidate(
        ctx,
        "short_cinematic_boom.m4a",
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
            "scenarios": "Dramatic reveal",
            "intensity": "high",
            "recommended_use": "short punctuations rather than background.",
        },
    )


@sound_agent.tool
@log_tool
def sound_vine_boom(ctx: RunContext[World]) -> Dict[str, Any]:
    """
    Iconic meme 'boom' for comedic punchlines, reaction moments, and meme edits. This fits best in scenarios where someone says something that is "out-of-pocket" or unexpected and causes the other person to not know how to react. It also works when one person makes a punchy remark about the other person.
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
def sound_short_vibration(ctx: RunContext[World]) -> Dict[str, Any]:
    """
    This vibration should sound when the conversation gets dry for multiple turns to nudge them to ask more questions and re-engage into the conversation.
    """
    return _mk_candidate(
        ctx,
        "short_vibration.m4a",
        tags=[],
        attrs={
            "scenarios": "This vibration should sound when the conversation gets dry to nudge them to ask more questions and re-engage into the conversation.",
            "intensity": "low-medium",
            "recommended_use": "subtle nudge",
        },
    )


@sound_agent.tool
@log_tool
def sound_heated_long_vibration(ctx: RunContext[World]) -> Dict[str, Any]:
    """
    This sound should play when the people are having a heated conversation or an argument about something. This is a nudge to remind them to pause and take a step back to reflect. They might be touching on a sensitive topic.
    """
    return _mk_candidate(
        ctx,
        "heated_long_vibration.m4a",
        tags=[],
        attrs={
            "scenarios": "This sound should play when the people are having a heated conversation or an argument about something. This is a nudge to remind them to pause and take a step back to reflect.",
            "intensity": "medium-high",
            "recommended_use": "nudge to prevent conversation from getting heated",
        },
    )


# @sound_agent.tool
# @log_tool
# def sound_short_vibration_double(ctx: RunContext[World]) -> Dict[str, Any]:
#     """
#     This vibration should sound when the conversation gets dry to nudge them to ask more questions and re-engage into the conversation.
#     """
#     return _mk_candidate(
#         ctx,
#         "short_vibration_double.m4a",
#         tags=[],
#         attrs={
#             "scenarios": "This vibration should sound when the conversation gets dry to nudge them to ask more questions and re-engage into the conversation.",
#             "intensity": "medium-high",
#             "recommended_use": "subtle nudge",
#         },
#     )


@sound_agent.tool
@log_tool
def sound_death_note_l_theme(ctx: RunContext[World]) -> Dict[str, Any]:
    """
    Tense, investigative background music (L's theme vibe): analytical, mysterious, and focused.
    Good for deduction sequences, coding montages, stealth or crime board reveals. This is good for moments where one of the people in conversation announce that they are going to lock in or if they are plotting/planning something.
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
