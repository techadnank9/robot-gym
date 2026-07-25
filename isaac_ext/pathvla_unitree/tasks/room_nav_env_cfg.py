from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class SceneBoundsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: list[float] = Field(min_length=2, max_length=2)
    y: list[float] = Field(min_length=2, max_length=2)
    z_floor: float = 0.0


class RoomGeometryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_height: float = 2.5
    wall_thickness: float = 0.1


class SceneObjectConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    shape: str
    color: list[float] = Field(min_length=3, max_length=3)
    size: list[float] = Field(min_length=3, max_length=3)
    pose: list[float] = Field(min_length=3, max_length=3)
    avoidance_radius: float = Field(gt=0.0)
    semantic_color: str | None = None


class CameraConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    pose: list[float] = Field(min_length=3, max_length=3)
    look_at: list[float] = Field(min_length=3, max_length=3)


class SceneInnerConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    environment_usd_path_env: str | None = None
    environment_prim_path: str = "/World/Scene/Environment"
    bounds: SceneBoundsModel
    room: RoomGeometryModel
    robot_spawn: list[float] = Field(min_length=3, max_length=3)
    objects: list[SceneObjectConfigModel]
    cameras: list[CameraConfigModel]


class SceneConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: SceneInnerConfigModel


class RobotLocomotionConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller_module: str | None = None
    controller_class: str | None = None
    controller_kwargs: dict[str, Any] = Field(default_factory=dict)


class RobotInnerConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    usd_path_env: str
    default_prim_path: str
    proxy_prim_path: str
    spawn_translation: list[float] = Field(min_length=3, max_length=3)
    spawn_orientation_xyzw: list[float] = Field(min_length=4, max_length=4)
    locomotion: RobotLocomotionConfigModel


class RobotConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot: RobotInnerConfigModel


class LivestreamInnerConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_mode: str
    signaling_port: int
    http_port: int
    udp_port_range: str
    required_extensions: list[str]
    browser_path_hint: str


class LivestreamConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    livestream: LivestreamInnerConfigModel


class VLAInnerConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_timeout_s: float = 30.0
    include_camera_images: bool = False
    endpoint_suffix: str = "/infer"


class VLAConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vla: VLAInnerConfigModel


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_scene_config(scene_name: str) -> SceneConfigModel:
    mapping = {
        "room": project_root() / "config" / "scene_room.yaml",
        "warehouse": project_root() / "config" / "scene_warehouse.yaml",
        "sorting_lab": project_root() / "config" / "scene_sorting_lab.yaml",
    }
    if scene_name not in mapping:
        raise ValueError(f"Unsupported scene '{scene_name}'")
    return SceneConfigModel.model_validate(_load_yaml(mapping[scene_name]))


def load_robot_config() -> RobotConfigModel:
    return RobotConfigModel.model_validate(_load_yaml(project_root() / "config" / "robot.yaml"))


def load_livestream_config() -> LivestreamConfigModel:
    return LivestreamConfigModel.model_validate(_load_yaml(project_root() / "config" / "livestream.yaml"))


def load_vla_config() -> VLAConfigModel:
    return VLAConfigModel.model_validate(_load_yaml(project_root() / "config" / "vla.yaml"))
