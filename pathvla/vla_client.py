from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from pathvla.errors import VLAEndpointError
from pathvla.plan_validator import validate_plan_dict
from pathvla.schemas import PlanResponseModel, SceneSnapshotModel, VLAEndpointConfigModel, VLARequestModel


def load_vla_endpoint_config(timeout_s: float = 30.0) -> VLAEndpointConfigModel:
    endpoint = os.getenv("VLA_ENDPOINT")
    if not endpoint:
        raise VLAEndpointError("VLA_ENDPOINT is required. No mock planner is enabled.")
    return VLAEndpointConfigModel(
        endpoint=endpoint,
        api_key=os.getenv("VLA_API_KEY"),
        model_name=os.getenv("VLA_MODEL_NAME"),
        timeout_s=timeout_s,
    )


class HTTPVLAClient:
    def __init__(self, config: VLAEndpointConfigModel):
        self.config = config

    def infer(
        self,
        instruction: str,
        scene: SceneSnapshotModel,
        output_dir: Path,
        include_camera_images: bool = False,
    ) -> PlanResponseModel:
        scene_payload = scene.model_dump(mode="json")
        if not include_camera_images:
            scene_payload["camera_images"] = None
        payload = VLARequestModel(
            instruction=instruction,
            scene=SceneSnapshotModel.model_validate(scene_payload),
            model_name=self.config.model_name,
        ).model_dump(mode="json", exclude_none=True)

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            response = requests.post(
                str(self.config.endpoint),
                json=payload,
                headers=headers,
                timeout=self.config.timeout_s,
            )
        except requests.RequestException as exc:
            raise VLAEndpointError(f"Failed to reach VLA endpoint {self.config.endpoint}: {exc}") from exc

        if response.status_code >= 400:
            raise VLAEndpointError(
                f"VLA endpoint returned HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise VLAEndpointError(f"VLA endpoint response was not JSON: {response.text[:500]}") from exc

        bad_path = output_dir / "bad_vla_response.json"
        return validate_plan_dict(data, bad_response_path=bad_path)


def rule_based_debug_plan(instruction: str) -> PlanResponseModel:
    lowered = instruction.lower()
    subgoals: list[dict[str, Any]] = []
    avoid = []
    if "avoid the chair" in lowered or "avoid chair" in lowered:
        avoid.append("chair")
    if "avoid the table" in lowered or "avoid table" in lowered:
        avoid.append("table")
    if "red bin" in lowered:
        subgoals.append(
            {
                "type": "navigate",
                "target": "red_bin",
                "constraints": {"avoid": avoid, "safe_distance_m": 0.6},
            }
        )
    if "inspect the table" in lowered or "inspect table" in lowered:
        subgoals.append(
            {
                "type": "inspect",
                "target": "table",
                "constraints": {"avoid": avoid, "safe_distance_m": 0.6},
            }
        )
    if "bottle" in lowered and "pick" in lowered:
        subgoals.append(
            {
                "type": "pickup",
                "target": "bottle",
                "constraints": {"avoid": avoid, "safe_distance_m": 0.4},
            }
        )
    if "bring it to the red bin" in lowered or "drop" in lowered:
        subgoals.append(
            {
                "type": "drop",
                "target": "red_bin",
                "constraints": {"avoid": avoid, "safe_distance_m": 0.6},
            }
        )
    if "return home" in lowered or "go home" in lowered or "home" in lowered:
        subgoals.append(
            {
                "type": "return_home",
                "target": "home_marker",
                "constraints": {"avoid": avoid, "safe_distance_m": 0.6},
            }
        )
    if not subgoals:
        subgoals.append(
            {
                "type": "navigate",
                "target": "red_bin",
                "constraints": {"avoid": avoid, "safe_distance_m": 0.6},
            }
        )
    return validate_plan_dict({"subgoals": subgoals})
