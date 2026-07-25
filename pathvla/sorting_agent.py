from __future__ import annotations

import json
import math
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pathvla.errors import ConfigurationError, PlanningError, VLAEndpointError


DEFAULT_GEMINI_ROBOTICS_MODEL = "gemini-robotics-er-1.6-preview"
DEFAULT_MIN_REQUEST_INTERVAL_S = 13.0
DEFAULT_RATE_LIMIT_RETRIES = 3

# Keep the request schema deliberately small. Passing the Pydantic class through
# GenerateContentConfig.response_schema makes google-genai convert
# `additionalProperties` into an unsupported `additional_properties` field for
# the Robotics-ER endpoint. response_json_schema preserves the JSON Schema wire
# format, and the Pydantic model below still performs strict response validation.
GEMINI_DECISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["navigate", "pick", "place", "finish"],
            "description": "Exactly one next robot action.",
        },
        "target": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Exact scene object ID, or null for finish.",
        },
        "destination": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Exact destination bin ID for place; otherwise null.",
        },
        "rationale": {
            "type": "string",
            "description": "Brief observable reason, without hidden chain-of-thought.",
        },
        "expected_outcome": {
            "type": "string",
            "description": "Observable postcondition expected after execution.",
        },
    },
    "required": ["action", "target", "destination", "rationale", "expected_outcome"],
}


class SortingAction(str, Enum):
    NAVIGATE = "navigate"
    PICK = "pick"
    PLACE = "place"
    FINISH = "finish"


class SortableObjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["item", "bin", "obstacle", "surface"]
    color: str
    pose: list[float] = Field(min_length=3, max_length=3)
    status: Literal["available", "held", "sorted", "static"]
    assigned_bin: str | None = None


class SortingWorldStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_pose: list[float] = Field(min_length=3, max_length=3)
    held_object: str | None = None
    objects: list[SortableObjectModel]
    completed_actions: list[str] = Field(default_factory=list)
    rejected_actions: list[str] = Field(default_factory=list)

    @property
    def task_complete(self) -> bool:
        items = [obj for obj in self.objects if obj.kind == "item"]
        return bool(items) and all(obj.status == "sorted" for obj in items)


class GeminiActionDecision(BaseModel):
    """Exactly one grounded action selected by Gemini Robotics-ER."""

    model_config = ConfigDict(extra="forbid")

    action: SortingAction
    target: str | None = Field(
        default=None,
        description="Scene object ID. Required for navigate, pick, and place; null for finish.",
    )
    destination: str | None = Field(
        default=None,
        description="Destination bin ID. Required only for place.",
    )
    rationale: str = Field(
        min_length=1,
        max_length=240,
        description="Brief, observable reason for this next action; do not include hidden chain-of-thought.",
    )
    expected_outcome: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_shape(self) -> "GeminiActionDecision":
        if self.action == SortingAction.FINISH:
            if self.target is not None or self.destination is not None:
                raise ValueError("finish must not include target or destination")
            return self
        if self.target is None:
            raise ValueError(f"{self.action.value} requires target")
        if self.action == SortingAction.PLACE and self.destination is None:
            raise ValueError("place requires destination")
        if self.action != SortingAction.PLACE and self.destination is not None:
            raise ValueError(f"{self.action.value} must not include destination")
        return self


def validate_grounded_action(
    decision: GeminiActionDecision,
    state: SortingWorldStateModel,
    interaction_radius_m: float = 1.1,
) -> None:
    """Reject hallucinated or unsafe actions before they reach the simulator."""

    objects = {obj.name: obj for obj in state.objects}
    if decision.action == SortingAction.FINISH:
        if not state.task_complete:
            raise PlanningError("Gemini requested finish before all items were sorted.")
        return

    target = objects.get(decision.target or "")
    if target is None:
        raise PlanningError(f"Gemini selected unknown target '{decision.target}'.")

    if decision.action == SortingAction.NAVIGATE:
        if target.kind not in {"item", "bin"}:
            raise PlanningError(f"Cannot navigate to {target.kind} target '{target.name}' in this demo.")
        return

    if decision.action == SortingAction.PICK:
        if target.kind != "item" or target.status != "available":
            raise PlanningError(f"Target '{target.name}' is not an available sortable item.")
        if state.held_object is not None:
            raise PlanningError(f"The robot is already holding '{state.held_object}'.")
        if math.dist(state.robot_pose[:2], target.pose[:2]) > interaction_radius_m:
            raise PlanningError(f"Robot must navigate closer to '{target.name}' before picking it.")
        return

    if state.held_object != target.name or target.status != "held":
        raise PlanningError(f"Robot is not holding place target '{target.name}'.")
    destination = objects.get(decision.destination or "")
    if destination is None or destination.kind != "bin":
        raise PlanningError(f"Destination '{decision.destination}' is not a known bin.")
    if target.color != destination.color:
        raise PlanningError(
            f"Color mismatch: '{target.name}' is {target.color} but '{destination.name}' is {destination.color}."
        )
    if math.dist(state.robot_pose[:2], destination.pose[:2]) > interaction_radius_m:
        raise PlanningError(f"Robot must navigate closer to '{destination.name}' before placing.")


SYSTEM_INSTRUCTION = """You are the embodied reasoning controller for a Unitree G1 sorting demo.
Choose exactly one next action from the supplied schema. Base the decision on the current camera image,
the task, the typed scene registry, and action history. Use only exact object IDs from the registry.

Control contract:
- navigate(target): move near an available item or a destination bin.
- pick(target): only when the robot is near that available item and its hand is empty.
- place(target, destination): target must be the held item; destination must be the visually and
  semantically correct bin; the robot must already be near the destination.
- finish: only after every sortable item is visibly and geometrically inside its correct bin.

Sort by the user's rule, not by object name alone. Never claim an action succeeded before the returned
world state confirms it. If a previous action was rejected, correct the precondition on the next turn.
"""


class GeminiRoboticsERAgent:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        thinking_budget: int = 1024,
        min_request_interval_s: float | None = None,
        rate_limit_retries: int | None = None,
        wait_callback: Callable[[float], None] | None = None,
        status_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ConfigurationError("GEMINI_API_KEY is required; no planner fallback is configured.")
        self.model = model or os.getenv("GEMINI_ROBOTICS_MODEL") or DEFAULT_GEMINI_ROBOTICS_MODEL
        self.thinking_budget = thinking_budget
        self.min_request_interval_s = (
            min_request_interval_s
            if min_request_interval_s is not None
            else float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL_S", DEFAULT_MIN_REQUEST_INTERVAL_S))
        )
        self.rate_limit_retries = (
            rate_limit_retries
            if rate_limit_retries is not None
            else int(os.getenv("GEMINI_RATE_LIMIT_RETRIES", DEFAULT_RATE_LIMIT_RETRIES))
        )
        if self.min_request_interval_s < 12.0:
            raise ConfigurationError(
                "GEMINI_MIN_REQUEST_INTERVAL_S must be at least 12 seconds for the 5 RPM Robotics-ER quota."
            )
        if self.rate_limit_retries < 0:
            raise ConfigurationError("GEMINI_RATE_LIMIT_RETRIES must not be negative.")
        self.wait_callback = wait_callback
        self.status_callback = status_callback
        self._last_request_started_at: float | None = None
        try:
            from google import genai
        except ImportError as exc:
            raise ConfigurationError("Install google-genai to use Gemini Robotics-ER.") from exc
        self._client = genai.Client(api_key=resolved_key)

    def decide(
        self,
        instruction: str,
        state: SortingWorldStateModel,
        camera_frames: list[Path],
    ) -> GeminiActionDecision:
        if not camera_frames:
            raise ConfigurationError("Gemini Robotics-ER requires at least one current camera frame.")
        try:
            from google.genai import types
        except ImportError as exc:
            raise ConfigurationError("Install google-genai to use Gemini Robotics-ER.") from exc

        contents = []
        for frame_path in camera_frames:
            if not frame_path.is_file():
                raise ConfigurationError(f"Camera frame does not exist: {frame_path}")
            contents.append(
                types.Part.from_bytes(data=frame_path.read_bytes(), mime_type=_mime_type(frame_path))
            )
        prompt_state = state.model_dump(mode="json")
        contents.append(
            "Task:\n"
            f"{instruction}\n\n"
            "Current typed world state:\n"
            f"{json.dumps(prompt_state, indent=2)}\n\n"
            "Select the single safest next action."
        )
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            response_mime_type="application/json",
            response_json_schema=GEMINI_DECISION_JSON_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
        )
        response = None
        for attempt in range(self.rate_limit_retries + 1):
            self._wait_for_request_slot()
            self._last_request_started_at = time.monotonic()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    if attempt < self.rate_limit_retries:
                        retry_delay = max(
                            _retry_delay_seconds(message) + 1.0,
                            self.min_request_interval_s,
                        )
                        print(
                            f"Gemini Robotics-ER quota reached; retrying in {retry_delay:.1f}s "
                            f"({attempt + 1}/{self.rate_limit_retries})."
                        )
                        if self.status_callback is not None:
                            self.status_callback("quota_retry", retry_delay)
                        self._cooperative_wait(retry_delay)
                        self._last_request_started_at = None
                        continue
                if "403" in message:
                    message += " Robotics-ER rejects unrestricted keys; restrict the key in Google AI Studio."
                raise VLAEndpointError(f"Gemini Robotics-ER request failed: {message}") from exc
        if response is None:
            raise VLAEndpointError("Gemini Robotics-ER did not return a response.")

        try:
            if getattr(response, "parsed", None) is not None:
                return GeminiActionDecision.model_validate(response.parsed)
            return GeminiActionDecision.model_validate_json(response.text)
        except Exception as exc:  # noqa: BLE001
            raw = getattr(response, "text", "")
            raise VLAEndpointError(f"Gemini returned an invalid action: {raw[:500]}") from exc

    def _wait_for_request_slot(self) -> None:
        if self._last_request_started_at is None:
            return
        elapsed = time.monotonic() - self._last_request_started_at
        remaining = self.min_request_interval_s - elapsed
        if remaining > 0:
            print(f"Gemini rate limit: waiting {remaining:.1f}s before the next action decision.")
            if self.status_callback is not None:
                self.status_callback("request_interval", remaining)
            self._cooperative_wait(remaining)

    def _cooperative_wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            chunk = min(0.1, remaining)
            if self.wait_callback is None:
                time.sleep(chunk)
            else:
                self.wait_callback(chunk)


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    raise ConfigurationError(f"Unsupported camera image type: {path.suffix}")


def _retry_delay_seconds(message: str) -> float:
    patterns = (
        r"retryDelay['\"\s:]+([0-9]+(?:\.[0-9]+)?)s",
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return DEFAULT_MIN_REQUEST_INTERVAL_S
