from __future__ import annotations

import json
import functools
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Any

from pydantic import BaseModel
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


# ---------- Output model ----------


class Stop(BaseModel):
    name: str
    time: str
    price_user_currency: float


class Plan(BaseModel):
    city: str
    date: str
    forecast: str
    currency: str
    walk_minutes: int
    total_cost: float
    stops: List[Stop]


# ---------- Dependencies (fake data we’ll query via tools) ----------


@dataclass
class World:
    fx: Dict[tuple[str, str], float]  # (from_ccy, to_ccy) -> rate
    weather: Dict[str, Dict[str, str]]  # city -> {date -> forecast}
    coords: Dict[str, Tuple[float, float]]  # place -> (lat, lon)
    attractions: Dict[
        str, List[Dict[str, Any]]
    ]  # city -> list of {name, tags, price_local, currency, time}
    log: List[Dict[str, Any]]  # in-memory event log


# ---------- Agent ----------

trip_agent = Agent(
    "openai:gpt-4o",
    deps_type=World,
    output_type=Plan,
    system_prompt=(
        "You are a careful micro-itinerary planner. Build a 2-stop *walking* afternoon "
        "plan using ONLY the tools. Do not invent facts.\n\n"
        "Workflow you MUST follow (and may repeat as needed):\n"
        "1) Call get_weather(city, date) to fetch the forecast.\n"
        "2) Call list_attractions(city, interest) to get candidate stops; pick two.\n"
        "3) Call walking_minutes(stop1, stop2) and ensure total walking ≤ 45 minutes; "
        "   if it's over, call list_attractions again to try a different pairing and re-check.\n"
        "4) Sum local prices for the two stops and call convert_currency(amount, from_ccy, user_ccy).\n\n"
        "When done, return a Plan with: city, date, forecast, currency (user currency), "
        "walk_minutes, total_cost, and two Stop entries including price_user_currency."
    ),
)


# ---------- Tools (each is wrapped with @log_tool) ----------


@trip_agent.tool
@log_tool
def get_weather(ctx: RunContext[World], city: str, date: str) -> str:
    """Return the forecast string for a given city/date."""
    return ctx.deps.weather.get(city, {}).get(date, "unknown")


@trip_agent.tool
@log_tool
def list_attractions(
    ctx: RunContext[World], city: str, interest: str
) -> List[Dict[str, Any]]:
    """Return candidate attractions (dicts) matching an interest tag for a city."""
    items = ctx.deps.attractions.get(city, [])
    interest_lower = interest.lower()
    return [
        it
        for it in items
        if any(interest_lower in t.lower() for t in it.get("tags", []))
    ]


@trip_agent.tool
@log_tool
def walking_minutes(ctx: RunContext[World], start: str, end: str) -> int:
    """
    Rough walking time between two named places using simple grid distance.
    (This is a stub for demo purposes.)
    """
    a = ctx.deps.coords.get(start)
    b = ctx.deps.coords.get(end)
    if not a or not b:
        return 999  # unknown place -> treat as too far
    # simple Manhattan distance * 10 minutes per degree-of-lat/lon (toy model)
    minutes = int((abs(a[0] - b[0]) + abs(a[1] - b[1])) * 10 * 60)
    return max(1, minutes)


@trip_agent.tool
@log_tool
def convert_currency(
    ctx: RunContext[World], amount: float, from_ccy: str, to_ccy: str
) -> float:
    """Convert amount from one currency to another using a provided rate table."""
    rate = ctx.deps.fx.get((from_ccy.upper(), to_ccy.upper()))
    if rate is None:
        rate = 1.0  # fallback for demo stability
    return round(amount * rate, 2)


# ---------- Demo data ----------

world = World(
    fx={
        ("EUR", "USD"): 1.08,
        ("USD", "USD"): 1.00,
    },
    weather={
        "Paris": {
            "2025-09-20": "Sunny, 24°C",
        }
    },
    coords={
        "Louvre Museum": (48.8606, 2.3376),
        "Musée d'Orsay": (48.8599, 2.3266),
        "Centre Pompidou": (48.8607, 2.3522),
        "Luxembourg Gardens": (48.8462, 2.3371),
    },
    attractions={
        "Paris": [
            {
                "name": "Louvre Museum",
                "tags": ["art", "museum"],
                "price_local": 17.0,
                "currency": "EUR",
                "time": "14:00",
            },
            {
                "name": "Musée d'Orsay",
                "tags": ["art", "museum"],
                "price_local": 16.0,
                "currency": "EUR",
                "time": "16:00",
            },
            {
                "name": "Centre Pompidou",
                "tags": ["art", "modern"],
                "price_local": 15.0,
                "currency": "EUR",
                "time": "15:30",
            },
            {
                "name": "Luxembourg Gardens",
                "tags": ["park", "walk"],
                "price_local": 0.0,
                "currency": "EUR",
                "time": "17:00",
            },
        ]
    },
    log=[],
)

# ---------- Run with request/response logging ----------

user_prompt = (
    "I'm into art. Plan me a 2-stop afternoon in Paris for 2025-09-20. "
    "Keep total walking time under 45 minutes and show prices in USD."
)

world.log.append({"ts": _ts(), "event": "user_request", "prompt": user_prompt})

result = trip_agent.run_sync(user_prompt, deps=world)

world.log.append(
    {"ts": _ts(), "event": "agent_output", "output": result.output.model_dump()}
)

# ---------- Inspect outputs ----------

print("---- Plan ----")
print(result.output.model_dump_json(indent=2))

print("\n---- Log ----")
print(json.dumps(world.log, indent=2, default=str))
