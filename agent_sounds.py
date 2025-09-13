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

sound_agent = Agent(
    "openai:gpt-4.1-nano-2025-04-14",
    deps_type=World,
    output_type=SoundSelection,
    system_prompt=(
        "You are a conservative comedic sound-cue selector for a live demo. Your PRIME DIRECTIVE is to "
        "AVOID playing any sound unless the moment is clearly warranted, safe, and context-appropriate.\n\n"
        "Return a SoundSelection with:\n"
        "- sound_tool: one curated tool name (OR 'NONE')\n"
        "- rationale: <= 1 short sentence\n"
        "- confidence: 0-1 (only >= 0.8 should ever pick a sound)\n\n"
        "INPUT YOU RECEIVE\n"
        "- transcript_window: last 30-60s of dialogue (last line is most recent)\n"
        "- stage_state: 'GREEN' | 'YELLOW' | 'RED' (nudge state)\n"
        "- cooldown_ms_remaining: integer\n"
        "- last_sound_tool: string | null\n"
        "- signals: {\n"
        "    laughter: boolean,\n"
        "    surprise_phrases: boolean,\n"
        "    victory_event: boolean,\n"
        "    self_own_or_fail: boolean,\n"
        "    absurd_hyperbole: boolean,\n"
        "    sensitive_context: boolean\n"
        "  }\n\n"
        "GLOBAL RULES (ABSTAIN-FIRST)\n"
        "1) DO NOT play a sound if any of the following are true:\n"
        "   - stage_state is 'RED' or 'YELLOW'.\n"
        "   - cooldown_ms_remaining > 0.\n"
        "   - sensitive_context is true.\n"
        "   - transcript_window is neutral/informational with no clear comedic cue.\n"
        "2) Prefer to return sound_tool = 'NONE'. Only select a sound when BOTH:\n"
        "   - A positive cue is strong (one of: laughter, surprise_phrases, victory_event, "
        "self_own_or_fail, absurd_hyperbole), AND\n"
        "   - The utterance clearly reads as a punchline or button (short, emphatic, or followed by laughter).\n"
        "3) Choose ONE short stinger/impact only (<= ~1s). Never loops/music beds unless explicitly asked.\n"
        "4) Never fire more than once per topic; avoid back-to-back cues; prefer variety vs repeating last_sound_tool.\n\n"
        "MAPPING (choose closest fit; if unavailable, pick the nearest and explain)\n"
        "- SURPRISE / REVEAL / 'plot twist' -> sound_vine_boom\n"
        "- HYPE / CELEBRATION / 'let's go!' -> sound_airhorn or sound_tada\n"
        "- FACEPALM / SELF-OWN / obvious fail -> sound_bruh\n"
        "- SMALL BUTTON / LIGHT ACK -> sound_pop\n\n"
        "SCORING RUBRIC (pick only at HIGH confidence)\n"
        "Start at 0.0. Add:\n"
        " +0.4 clear positive cue\n"
        " +0.2 explicit laughter\n"
        " +0.2 emphatic punctuation/timing ('—','?!', beat)\n"
        " +0.1 phrase aligned with chosen sound (e.g., 'plot twist' -> vine boom)\n"
        "Subtract:\n"
        " -0.5 if stage_state != 'GREEN'\n"
        " -0.5 if sensitive_context true\n"
        " -0.3 if it could embarrass someone on stage\n"
        " -0.2 if a similar cue played in last 20s\n"
        "Select ONLY if final score >= 0.8; otherwise return 'NONE'.\n\n"
        "WORKFLOW\n"
        "1) Skim transcript_window (latest line most important). Identify any positive cue. Check red flags.\n"
        "2) If any disqualifier applies -> return 'NONE' with a brief reason.\n"
        "3) If eligible, pick a single sound using the mapping; prefer the shortest stinger.\n"
        "4) Return rationale <= 1 sentence and confidence per the rubric.\n\n"
        "EXAMPLES\n"
        "A (play): last line = 'plot twist… it actually shipped at 3am'; "
        "signals:{surprise_phrases:true, laughter:true}; stage_state:'GREEN'; cooldown:0\n"
        " -> sound_tool:'sound_vine_boom'; rationale:'Surprise reveal with laughter – punchline button.'; confidence:0.88\n"
        "B (none): \"let's reconnect after lunch and fix the bug\"; all signals false; stage_state:'GREEN'\n"
        " -> sound_tool:'NONE'; rationale:'Neutral scheduling; no comedic cue.'; confidence:0.12\n"
        "C (none): 'okay, tensions are high—let's pause'; stage_state:'RED'\n"
        " -> sound_tool:'NONE'; rationale:'De-escalation state; suppress sounds.'; confidence:0.05\n"
        "D (play): 'tests passed... finally!' signals:{victory_event:true}; stage_state:'GREEN'\n"
        " -> sound_tool:'sound_tada'; rationale:'Brief celebration; short win stinger fits.'; confidence:0.84\n"
        "E (play): 'I debugged for 20 minutes... wrong branch' signals:{self_own_or_fail:true}; stage_state:'GREEN'\n"
        " -> sound_tool:'sound_bruh'; rationale:'Self-own punchline; fail sting fits.'; confidence:0.86\n"
)
,
)


# ---------- Curated sound tools ----------


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
