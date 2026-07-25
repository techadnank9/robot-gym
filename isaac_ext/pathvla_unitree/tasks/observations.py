from __future__ import annotations

from pxr import Usd, UsdGeom

from isaac_ext.pathvla_unitree.tasks.semantic_scene import SemanticSceneState


def get_prim_translation(stage, prim_path: str) -> list[float]:
    prim = stage.GetPrimAtPath(prim_path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = transform.ExtractTranslation()
    return [float(translation[0]), float(translation[1]), float(translation[2])]


def refresh_semantic_scene_poses(stage, semantic_scene: SemanticSceneState) -> None:
    for obj in semantic_scene.object_states:
        obj.pose = get_prim_translation(stage, obj.prim_path)
