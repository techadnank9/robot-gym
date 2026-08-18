"""
Gemini Robotics-ER 1.6 as the deciding layer for the library fetch task.

ER 1.6 is a reasoning model, not a motor-control model: it reads camera frames
and a grounded world state, and returns *which skill to run next*. It never
sees joint angles and never emits them. The blind locomotion policy keeps the
robot upright; this decides where the robot should be going.

The API key is read from ``secrets/gemini_key`` or ``GEMINI_API_KEY``. Keep the
key out of ``antioch.yaml``: manifest values are literals that travel with the
project definition.

Set ``--adapter scripted`` to run the same task with a deterministic policy;
that path needs no key and is the control case when the model does something
surprising.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_NAME = "gemini-robotics-er-1.6-preview"
KEY_FILE = Path("secrets/gemini_key")

# One call every few seconds: the reasoning layer runs far slower than the
# 50 Hz gait, and vision calls are metered.
MIN_INTERVAL_S = 4.0

SKILLS = ("navigate_book", "grasp", "navigate_goal", "release", "recover", "wait")

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "skill": {"type": "string", "enum": list(SKILLS)},
        "rationale": {"type": "string"},
        "expected_outcome": {"type": "string"},
    },
    "required": ["skill", "rationale", "expected_outcome"],
}

SYSTEM_INSTRUCTION = """You are the deciding layer for a Unitree G1 humanoid robot in a real library that was 3D-scanned from an iPhone LiDAR capture.

The robot's task: walk to the book, pick it up, carry it to the drop-off table, and release it there.

You choose ONE skill per decision. A separate blind locomotion policy handles balance and walking; you never control joints.

Skills:
- navigate_book: walk toward the book. Use when the book is not yet within reach.
- grasp: close the hand on the book. Only when the book is within reach_radius.
- navigate_goal: walk toward the drop-off table. Only while carrying the book.
- release: let the book go. Only when carrying and within the goal radius.
- recover: stand back up. Use when fallen is true.
- wait: hold still for one cycle. Use when nothing else is safe.

Rules:
- If fallen is true, choose recover.
- Never grasp unless within reach_radius of the book.
- Never release unless carrying and inside the goal radius.
- Prefer progress; wait is a last resort.

Answer with JSON only."""


@dataclass(frozen=True)
class Decision:
    """One skill choice, with the reasoning that produced it."""

    skill: str
    rationale: str
    expected_outcome: str
    model_name: str
    latency_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "rationale": self.rationale,
            "expected_outcome": self.expected_outcome,
            "model": self.model_name,
            "latency_s": round(self.latency_s, 3),
        }


def read_api_key() -> str | None:
    """Return the Gemini key from the synced file or the environment."""

    if KEY_FILE.is_file():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


class ScriptedAdapter:
    """
    Deterministic skill selection from the grounded state.

    This is the control case: the same task, the same skill interface, no
    model and no key. When the robot misbehaves under ER 1.6, running this
    says whether the fault is in the reasoning or in everything below it.
    """

    model_name = "scripted"

    def decide(self, state: dict[str, Any], frames: list[bytes]) -> Decision:
        started = time.monotonic()
        skill = self._select(state)
        return Decision(
            skill=skill,
            rationale="deterministic rule over the grounded state",
            expected_outcome=f"advance {skill}",
            model_name=self.model_name,
            latency_s=time.monotonic() - started,
        )

    @staticmethod
    def _select(state: dict[str, Any]) -> str:
        if state.get("fallen"):
            return "recover"
        if state.get("carrying"):
            if state.get("goal_distance_m", math.inf) <= state.get("goal_radius_m", 0.9):
                return "release"
            return "navigate_goal"
        if state.get("book_distance_m", math.inf) <= state.get("reach_radius_m", 0.75):
            return "grasp"
        return "navigate_book"


class GeminiERAdapter:
    """Ask Gemini Robotics-ER 1.6 which skill to run next."""

    def __init__(self, api_key: str, model_name: str = MODEL_NAME, min_interval_s: float = MIN_INTERVAL_S) -> None:
        from google import genai

        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)
        self._min_interval_s = min_interval_s
        self._last_call = 0.0
        self._fallback = ScriptedAdapter()

    def decide(self, state: dict[str, Any], frames: list[bytes]) -> Decision:
        from google.genai import types

        # Hold the cadence; the gait does not wait for the reasoning layer
        wait = self._min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        started = time.monotonic()
        contents: list[Any] = [
            types.Part.from_bytes(data=frame, mime_type="image/jpeg") for frame in frames if frame
        ]
        contents.append(
            "Grounded state:\n"
            + json.dumps(state, indent=2)
            + "\n\nThe images are the robot's head camera. Choose the single best next skill."
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_json_schema=DECISION_SCHEMA,
                ),
            )
            parsed = response.parsed if getattr(response, "parsed", None) is not None else json.loads(response.text)
            skill = str(parsed["skill"])
            if skill not in SKILLS:
                raise ValueError(f"model returned unknown skill {skill!r}")
            decision = Decision(
                skill=skill,
                rationale=str(parsed.get("rationale", ""))[:240],
                expected_outcome=str(parsed.get("expected_outcome", ""))[:240],
                model_name=self.model_name,
                latency_s=time.monotonic() - started,
            )
        except Exception as error:  # noqa: BLE001 - a live match should not die on one bad call
            # Falling back keeps the robot moving and makes the failure visible
            # in the log rather than ending the run
            decision = self._fallback.decide(state, frames)
            print(f"[gemini] call failed ({type(error).__name__}: {error}); used scripted fallback", flush=True)

        self._last_call = time.monotonic()
        return decision


def build_adapter(name: str) -> ScriptedAdapter | GeminiERAdapter:
    """
    Build the requested adapter, refusing to silently downgrade.

    :param name: ``"gemini"`` or ``"scripted"``.
    """

    if name == "scripted":
        return ScriptedAdapter()
    if name != "gemini":
        raise ValueError(f"unknown adapter {name!r}")

    key = read_api_key()
    if not key:
        raise RuntimeError(
            "Gemini Robotics-ER needs an API key. Put it in secrets/gemini_key "
            "(one line, no quotes) or set GEMINI_API_KEY, then re-run. "
            "Use --adapter scripted to run without a model."
        )
    return GeminiERAdapter(key)
