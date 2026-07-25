from __future__ import annotations

from dataclasses import dataclass, field

from pathvla.schemas import RobotStateModel, SceneObjectModel, SceneSnapshotModel


@dataclass
class SemanticObjectState:
    name: str
    type: str
    prim_path: str
    pose: list[float]
    avoidance_radius: float
    color: str | None = None


@dataclass
class SemanticSceneState:
    scene_name: str
    bounds: dict[str, list[float]]
    object_states: list[SemanticObjectState] = field(default_factory=list)
    camera_prim_paths: list[str] = field(default_factory=list)

    def to_snapshot(self, robot_name: str, robot_pose: list[float], camera_images: list[str] | None = None) -> SceneSnapshotModel:
        return SceneSnapshotModel(
            scene_name=self.scene_name,
            objects=[
                SceneObjectModel(
                    name=obj.name,
                    pose=obj.pose,
                    type=obj.type,
                    avoidance_radius=obj.avoidance_radius,
                )
                for obj in self.object_states
            ],
            robot=RobotStateModel(name=robot_name, pose=robot_pose),
            bounds=self.bounds,
            camera_images=camera_images,
        )
