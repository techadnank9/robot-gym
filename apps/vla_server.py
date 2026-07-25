from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from isaac_ext.pathvla_unitree import __version__
from pathvla.plan_validator import validate_plan_dict
from pathvla.schemas import HealthResponseModel, PlanResponseModel, SceneSnapshotModel, VLARequestModel


class InferResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoals: list[dict[str, Any]] = Field(min_length=1)


PLAN_JSON_SCHEMA: dict[str, Any] = {
    "name": "pathvla_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subgoals": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["navigate", "inspect", "pickup", "drop", "return_home"],
                        },
                        "target": {"type": "string"},
                        "constraints": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "avoid": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "safe_distance_m": {"type": "number"},
                            },
                            "required": ["avoid", "safe_distance_m"],
                        },
                    },
                    "required": ["type", "target", "constraints"],
                },
            }
        },
        "required": ["subgoals"],
    },
}


SYSTEM_PROMPT = """You are a robotics task planner.
Return only a valid JSON plan following the provided schema.
Use only object names that appear in the scene.
Do not invent targets, objects, or actions.
Prefer short subgoal sequences grounded in the instruction and scene.
If the instruction implies returning home, use target `home_marker`.
For avoidance, include named objects in constraints.avoid.
Use safe_distance_m between 0.4 and 1.0 depending on clutter.
Do not emit reasoning, explanations, markdown, or code fences.
"""

app = FastAPI(title="PathVLA VLA Server", version=__version__)


def _require_openai_client() -> OpenAI:
    base_url = os.getenv("VLA_LLM_BASE_URL")
    api_key = os.getenv("VLA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if base_url:
        # OpenAI-compatible local/server backends such as Gemma served via vLLM can run without auth.
        return OpenAI(base_url=base_url, api_key=api_key or "local-no-auth")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Set VLA_LLM_BASE_URL for a local OpenAI-compatible server or OPENAI_API_KEY for OpenAI cloud.",
        )
    return OpenAI(api_key=api_key)


def _resolve_model_name(payload: VLARequestModel) -> str:
    model_name = (
        payload.model_name
        or os.getenv("VLA_LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("VLA_MODEL_NAME")
    )
    if not model_name:
        raise HTTPException(status_code=500, detail="Set VLA_LLM_MODEL or include model_name in the request.")
    return model_name


def _build_user_prompt(payload: VLARequestModel) -> str:
    scene_summary = {
        "scene_name": payload.scene.scene_name,
        "bounds": payload.scene.bounds,
        "robot": payload.scene.robot.model_dump(mode="json"),
        "objects": [obj.model_dump(mode="json") for obj in payload.scene.objects],
    }
    return (
        "Instruction:\n"
        f"{payload.instruction}\n\n"
        "Scene JSON:\n"
        f"{json.dumps(scene_summary, indent=2)}\n"
    )


def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", "")
    if output_text:
        return output_text
    try:
        return json.dumps(response.output[0].content[0].json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not extract structured model output: {exc}") from exc


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"<\|think\|>.*?(?:<turn\|>|$)", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        raise HTTPException(status_code=502, detail=f"Model did not return JSON. Raw output: {cleaned[:500]}")

    depth = 0
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Model returned invalid JSON candidate: {candidate[:500]} ({exc})",
                    ) from exc
                if isinstance(parsed, dict):
                    return parsed
                break

    raise HTTPException(status_code=502, detail=f"Could not recover JSON object from model output: {cleaned[:500]}")


def _normalize_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "subgoals" in payload:
        return payload

    if "type" in payload and "target" in payload:
        constraints = payload.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}
        avoid = payload.get("avoid")
        safe_distance_m = payload.get("safe_distance_m")
        if isinstance(avoid, list) and "avoid" not in constraints:
            constraints["avoid"] = avoid
        if isinstance(safe_distance_m, (int, float)) and "safe_distance_m" not in constraints:
            constraints["safe_distance_m"] = float(safe_distance_m)
        constraints.setdefault("avoid", [])
        constraints.setdefault("safe_distance_m", 0.6)
        return {
            "subgoals": [
                {
                    "type": payload["type"],
                    "target": payload["target"],
                    "constraints": constraints,
                }
            ]
        }

    if "action" in payload and "args" in payload and isinstance(payload["args"], dict):
        action = payload.get("type") or payload.get("action")
        args = payload["args"]
        target = args.get("target") or payload.get("target")
        if action and target:
            normalized_action = {
                "go_to": "navigate",
                "goto": "navigate",
                "navigate": "navigate",
                "inspect": "inspect",
                "pickup": "pickup",
                "pick_up": "pickup",
                "drop": "drop",
                "return_home": "return_home",
                "go_home": "return_home",
            }.get(str(action).lower(), str(action).lower())

            constraints = args.get("constraints")
            avoid = args.get("avoid") or payload.get("avoid")
            safe_distance_m = args.get("safe_distance_m") or payload.get("safe_distance_m")
            if not isinstance(constraints, dict):
                constraints = {}
            if isinstance(avoid, list) and "avoid" not in constraints:
                constraints["avoid"] = avoid
            if isinstance(safe_distance_m, (int, float)) and "safe_distance_m" not in constraints:
                constraints["safe_distance_m"] = float(safe_distance_m)
            constraints.setdefault("avoid", [])
            constraints.setdefault("safe_distance_m", 0.6)

            return {
                "subgoals": [
                    {
                        "type": normalized_action,
                        "target": target,
                        "constraints": constraints,
                    }
                ]
            }

    if "plan" not in payload or not isinstance(payload["plan"], list):
        return payload

    normalized_subgoals: list[dict[str, Any]] = []
    for item in payload["plan"]:
        if not isinstance(item, dict):
            continue

        action = item.get("type") or item.get("action")
        target = item.get("target")
        if not action or not target:
            continue

        normalized_action = {
            "go_to": "navigate",
            "goto": "navigate",
            "navigate": "navigate",
            "inspect": "inspect",
            "pickup": "pickup",
            "pick_up": "pickup",
            "drop": "drop",
            "return_home": "return_home",
            "go_home": "return_home",
        }.get(str(action).lower(), str(action).lower())

        constraints = item.get("constraints")
        avoid = item.get("avoid")
        safe_distance_m = item.get("safe_distance_m")
        if not isinstance(constraints, dict):
            constraints = {}
        if isinstance(avoid, list) and "avoid" not in constraints:
            constraints["avoid"] = avoid
        if isinstance(safe_distance_m, (int, float)) and "safe_distance_m" not in constraints:
            constraints["safe_distance_m"] = float(safe_distance_m)
        constraints.setdefault("avoid", [])
        constraints.setdefault("safe_distance_m", 0.6)

        normalized_subgoals.append(
            {
                "type": normalized_action,
                "target": target,
                "constraints": constraints,
            }
        )

    if normalized_subgoals:
        return {"subgoals": normalized_subgoals}
    return payload


def _infer_with_responses(client: OpenAI, model_name: str, payload: VLARequestModel) -> str:
    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(payload)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": PLAN_JSON_SCHEMA["name"],
                "strict": True,
                "schema": PLAN_JSON_SCHEMA["schema"],
            }
        },
    )
    return _extract_response_text(response)


def _infer_with_chat_completions(client: OpenAI, model_name: str, payload: VLARequestModel) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(payload)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": PLAN_JSON_SCHEMA,
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise HTTPException(status_code=500, detail="Chat completions backend returned empty content.")
    return content


@app.get("/health", response_model=HealthResponseModel)
def health():
    return HealthResponseModel(ok=True, service="pathvla-vla-server", version=__version__)


@app.post("/infer", response_model=PlanResponseModel)
def infer(request: VLARequestModel):
    payload = VLARequestModel(
        instruction=request.instruction,
        scene=SceneSnapshotModel.model_validate(request.scene.model_dump(mode="json")),
        model_name=request.model_name,
    )
    client = _require_openai_client()
    model_name = _resolve_model_name(payload)
    try:
        raw_text = _infer_with_responses(client, model_name, payload)
    except Exception as exc:  # noqa: BLE001
        try:
            raw_text = _infer_with_chat_completions(client, model_name, payload)
        except Exception as chat_exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI-compatible request failed. responses error: {exc}; chat.completions error: {chat_exc}",
            ) from chat_exc

    parsed = validate_plan_dict(_normalize_plan_payload(_extract_json_object(raw_text)))
    return parsed
