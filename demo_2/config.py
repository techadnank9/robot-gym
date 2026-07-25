from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from demo_2.errors import Demo2ConfigurationError


class MotionLimits(BaseModel):
    """Conservative pilot limits with non-overridable hard ceilings."""

    model_config = ConfigDict(extra="forbid")

    max_forward_mps: float = Field(default=0.15, gt=0.0, le=0.25)
    max_lateral_mps: float = Field(default=0.08, gt=0.0, le=0.15)
    max_yaw_rate_rps: float = Field(default=0.20, gt=0.0, le=0.35)
    max_command_duration_s: float = Field(default=0.50, gt=0.0, le=1.0)
    stop_duration_s: float = Field(default=0.20, gt=0.0, le=0.5)
    settle_time_s: float = Field(default=0.25, ge=0.0, le=2.0)


class Demo2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_model: Literal["g1_29dof"] = "g1_29dof"
    sdk_timeout_s: float = Field(default=5.0, gt=0.0, le=15.0)
    allowed_fsm_ids: list[int] = Field(default_factory=lambda: [500], min_length=1)
    limits: MotionLimits = Field(default_factory=MotionLimits)
    allowed_arm_actions: dict[str, int] = Field(
        default_factory=lambda: {"release-arm": 99, "right-hand-up": 23}
    )

    @field_validator("allowed_fsm_ids")
    @classmethod
    def unique_fsm_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_fsm_ids must not contain duplicates")
        return value

    @field_validator("allowed_arm_actions")
    @classmethod
    def validate_arm_actions(cls, value: dict[str, int]) -> dict[str, int]:
        for name, action_id in value.items():
            if not name or name.strip() != name or " " in name:
                raise ValueError("arm action names must be non-empty kebab-case tokens")
            if action_id <= 0:
                raise ValueError(f"arm action '{name}' must have a positive SDK action ID")
        return value


def default_config_path() -> Path:
    return Path(__file__).with_name("config.yaml")


def load_config(path: Path | str | None = None) -> Demo2Config:
    resolved = Path(path) if path is not None else default_config_path()
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Demo2ConfigurationError(f"Could not read demo_2 config: {resolved}") from exc
    if not isinstance(payload, dict):
        raise Demo2ConfigurationError(f"demo_2 config must be a YAML mapping: {resolved}")
    try:
        return Demo2Config.model_validate(payload)
    except ValueError as exc:
        raise Demo2ConfigurationError(f"Invalid demo_2 config: {exc}") from exc
