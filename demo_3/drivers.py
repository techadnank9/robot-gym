from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from demo_3.schemas import Skill


GEMINI_POLICY_SCHEMA = {
    "type": "object",
    "properties": {
        "skill": {
            "type": "string",
            "enum": [skill.value for skill in Skill],
        },
        "rationale": {
            "type": "string",
            "description": "A brief observable reason for the selected skill.",
        },
        "expected_outcome": {
            "type": "string",
            "description": "The next observable state expected from the skill.",
        },
    },
    "required": ["skill", "rationale", "expected_outcome"],
}

SYSTEM_INSTRUCTION = """You control one Unitree G1 in a two-player delivery race.
Select exactly one next skill. Use only the cameras and compact grounded status.
Navigate to your payload, grasp it, carry it through the center checkpoint,
navigate to your goal, then release it. If fallen, recover. Do not claim success
before status confirms it. Return only the requested JSON schema."""


class _SharedRequestLimiter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_started_s: float | None = None

    def acquire(self, minimum_interval_s: float) -> None:
        with self.lock:
            if self.last_started_s is not None:
                remaining = minimum_interval_s - (time.monotonic() - self.last_started_s)
                if remaining > 0:
                    time.sleep(remaining)
            self.last_started_s = time.monotonic()


_LIMITERS_LOCK = threading.Lock()
_KEY_LIMITERS: dict[str, _SharedRequestLimiter] = {}


def _limiter_for_key(api_key: str) -> _SharedRequestLimiter:
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    with _LIMITERS_LOCK:
        return _KEY_LIMITERS.setdefault(fingerprint, _SharedRequestLimiter())


@dataclass(frozen=True)
class PolicyDecision:
    skill: Skill
    rationale: str
    expected_outcome: str
    inference_id: str
    model_name: str
    latency_s: float


class PolicyAdapter(Protocol):
    model_name: str

    def decide(
        self,
        player_id: str,
        status: dict[str, Any],
        camera_jpegs: tuple[bytes, bytes],
    ) -> PolicyDecision: ...


class ScriptedPolicyAdapter:
    """Deterministic policy used for validation without consuming API quota."""

    model_name = "Deterministic validation policy"

    def __init__(self) -> None:
        self._counter = 0

    def decide(
        self,
        player_id: str,
        status: dict[str, Any],
        camera_jpegs: tuple[bytes, bytes],
    ) -> PolicyDecision:
        del camera_jpegs
        self._counter += 1
        if status["fallen"]:
            skill = Skill.RECOVER
        elif status["delivered"]:
            skill = Skill.WAIT
        elif status["nearGoal"] and status["checkpointCrossed"]:
            skill = Skill.RELEASE
        elif status["carrying"]:
            skill = Skill.NAVIGATE_GOAL
        elif status["nearObject"]:
            skill = Skill.GRASP
        else:
            skill = Skill.NAVIGATE_OBJECT
        return PolicyDecision(
            skill=skill,
            rationale=f"{player_id} selected the next grounded race phase.",
            expected_outcome=f"Advance {skill.value}.",
            inference_id=f"scripted-{player_id}-{self._counter}",
            model_name=self.model_name,
            latency_s=0.0,
        )


class GeminiERPolicyAdapter:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        min_interval_s: float = 12.0,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini Robotics-ER")
        self.model_name = (
            model_name
            or os.getenv("GEMINI_ROBOTICS_MODEL")
            or "gemini-robotics-er-1.6-preview"
        )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai for Gemini Robotics-ER") from exc
        self._client = genai.Client(api_key=resolved_key)
        self._min_interval_s = max(12.0, float(min_interval_s))
        self._limiter = _limiter_for_key(resolved_key)
        self._lock = threading.Lock()
        self._counter = 0

    def decide(
        self,
        player_id: str,
        status: dict[str, Any],
        camera_jpegs: tuple[bytes, bytes],
    ) -> PolicyDecision:
        from google.genai import types

        self._limiter.acquire(self._min_interval_s)
        with self._lock:
            self._counter += 1
        started = time.monotonic()
        contents: list[Any] = [
            types.Part.from_bytes(data=image, mime_type="image/jpeg")
            for image in camera_jpegs
            if image
        ]
        contents.append(
            "Player: "
            + player_id
            + "\nGrounded race status:\n"
            + json.dumps(status, indent=2)
            + "\nSelect the single safest next skill."
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                response_mime_type="application/json",
                response_json_schema=GEMINI_POLICY_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
        )
        value = response.parsed if getattr(response, "parsed", None) is not None else json.loads(response.text)
        return PolicyDecision(
            skill=Skill(value["skill"]),
            rationale=str(value["rationale"])[:240],
            expected_outcome=str(value["expected_outcome"])[:240],
            inference_id=f"gemini-{player_id}-{self._counter}",
            model_name=self.model_name,
            latency_s=time.monotonic() - started,
        )


class CustomHTTPPolicyAdapter:
    """Provider-neutral JSON adapter for host-approved alternative models."""

    def __init__(
        self,
        endpoint: str,
        *,
        model_name: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        parsed = urlparse(endpoint)
        local_http = parsed.scheme == "http" and parsed.hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
        if not ((parsed.scheme == "https" and parsed.netloc) or local_http):
            raise ValueError("Custom policy endpoint must use HTTPS or localhost")
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._counter = 0

    def decide(
        self,
        player_id: str,
        status: dict[str, Any],
        camera_jpegs: tuple[bytes, bytes],
    ) -> PolicyDecision:
        self._counter += 1
        body = json.dumps(
            {
                "protocolVersion": "3.0",
                "playerId": player_id,
                "model": self.model_name,
                "status": status,
                "cameras": [
                    {
                        "mimeType": "image/jpeg",
                        "data": base64.b64encode(image).decode("ascii"),
                    }
                    for image in camera_jpegs
                ],
                "allowedSkills": [skill.value for skill in Skill],
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                value = json.loads(response.read(1_000_001))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Custom policy request failed: {exc}") from exc
        return PolicyDecision(
            skill=Skill(value["skill"]),
            rationale=str(value.get("rationale", "Custom policy decision."))[:240],
            expected_outcome=str(value.get("expectedOutcome", "Advance the delivery."))[:240],
            inference_id=str(value.get("inferenceId") or f"custom-{player_id}-{self._counter}"),
            model_name=self.model_name,
            latency_s=time.monotonic() - started,
        )


def build_policy_adapter(
    adapter_name: str,
    *,
    player_id: str,
    model_name: str | None = None,
    endpoint: str | None = None,
) -> PolicyAdapter:
    if adapter_name == "gemini-er":
        player_key = os.getenv(f"DEMO3_{player_id.upper()}_GEMINI_API_KEY")
        player_model = os.getenv(f"DEMO3_{player_id.upper()}_GEMINI_MODEL")
        return GeminiERPolicyAdapter(
            api_key=player_key,
            model_name=model_name or player_model,
        )
    if adapter_name == "scripted":
        return ScriptedPolicyAdapter()
    if adapter_name == "http":
        if not endpoint:
            raise RuntimeError(f"{player_id} custom HTTP policy requires an endpoint")
        key = os.getenv(f"DEMO3_{player_id.upper()}_POLICY_KEY")
        return CustomHTTPPolicyAdapter(
            endpoint,
            model_name=model_name or "Custom policy model",
            api_key=key,
        )
    raise ValueError(f"Unknown policy adapter: {adapter_name}")
