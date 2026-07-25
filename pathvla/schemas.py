from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SubgoalType(str, Enum):
    NAVIGATE = "navigate"
    INSPECT = "inspect"
    PICKUP = "pickup"
    DROP = "drop"
    RETURN_HOME = "return_home"


class ConstraintModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    avoid: list[str] = Field(default_factory=list)
    safe_distance_m: float = Field(default=0.6, gt=0.0)


class SubgoalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: SubgoalType
    target: str
    constraints: ConstraintModel = Field(default_factory=ConstraintModel)


class PlanResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoals: list[SubgoalModel] = Field(min_length=1)


class SceneObjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    pose: list[float] = Field(min_length=3, max_length=3)
    type: str
    avoidance_radius: float = Field(default=0.5, gt=0.0)


class RobotStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    pose: list[float] = Field(min_length=3, max_length=3)


class SceneSnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_name: str
    objects: list[SceneObjectModel]
    robot: RobotStateModel
    bounds: dict[str, list[float]]
    camera_images: list[str] | None = None


class VLARequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    scene: SceneSnapshotModel
    model_name: str | None = None


class PlannerDiagnosticModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_resolution_m: float
    expanded_nodes: int
    blocked_cells: int
    target_name: str


class WaypointPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    subgoal_type: SubgoalType
    waypoints: list[list[float]] = Field(min_length=1)
    path_length_m: float = Field(ge=0.0)
    avoided_objects: list[str] = Field(default_factory=list)
    diagnostics: PlannerDiagnosticModel


class RunMode(str, Enum):
    WEBRTC = "webrtc"
    REMOTE_DESKTOP = "remote_desktop"
    NONE = "none"


class RunRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    scene: Literal["room", "warehouse"] = "room"
    live: RunMode = RunMode.WEBRTC
    record_video: bool = True
    allow_proxy: bool = False
    allow_kinematic_control: bool = False
    require_vla: bool = True
    allow_rule_planner: bool = False
    require_video: bool = False
    output_dir: str | None = None


class RunResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["created", "running", "completed", "failed"]
    instruction: str
    scene: str
    live: RunMode
    output_dir: str
    message: str | None = None
    plan: PlanResponseModel | None = None
    waypoint_plans: list[WaypointPlanModel] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    video_path: str | None = None
    final_frame_path: str | None = None
    logs_path: str | None = None


class HealthResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    service: str
    version: str


class ChecksResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_ok: bool
    isaac_ok: bool
    livestream_ok: bool
    details: dict[str, Any]


class VLAEndpointConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: HttpUrl
    api_key: str | None = None
    model_name: str | None = None
    timeout_s: float = Field(default=30.0, gt=0.0)
