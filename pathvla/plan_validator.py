from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pathvla.errors import VLAEndpointError
from pathvla.schemas import PlanResponseModel


def validate_plan_dict(payload: dict[str, Any], bad_response_path: Path | None = None) -> PlanResponseModel:
    try:
        return PlanResponseModel.model_validate(payload)
    except ValidationError as exc:
        if bad_response_path is not None:
            bad_response_path.parent.mkdir(parents=True, exist_ok=True)
            bad_response_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise VLAEndpointError(f"Invalid VLA response schema: {exc}") from exc


def validate_plan_json(payload: str, bad_response_path: Path | None = None) -> PlanResponseModel:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        if bad_response_path is not None:
            bad_response_path.parent.mkdir(parents=True, exist_ok=True)
            bad_response_path.write_text(payload, encoding="utf-8")
        raise VLAEndpointError(f"VLA response was not valid JSON: {exc}") from exc
    return validate_plan_dict(data, bad_response_path=bad_response_path)
