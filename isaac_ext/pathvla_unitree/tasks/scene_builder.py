from __future__ import annotations

import math
import os
from pathlib import Path

from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

from isaac_ext.pathvla_unitree.tasks.object_spawner import spawn_semantic_object
from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import SceneConfigModel
from isaac_ext.pathvla_unitree.tasks.semantic_scene import SemanticObjectState, SemanticSceneState


def _set_transform_with_lookat(camera: UsdGeom.Camera, eye: list[float], target: list[float]) -> None:
    xformable = UsdGeom.Xformable(camera)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0))
    ops = xformable.GetOrderedXformOps()
    if ops:
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                op.Set(matrix.GetInverse())
                return
    xformable.AddTransformOp().Set(matrix.GetInverse())


def _spawn_ground_plane(stage, bounds) -> None:
    x0, x1 = bounds.x
    y0, y1 = bounds.y
    center = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, bounds.z_floor - 0.01]
    size = [x1 - x0, y1 - y0, 0.02]
    plane = UsdGeom.Cube.Define(stage, "/World/Scene/GroundPlane")
    plane.CreateSizeAttr(1.0)
    xformable = UsdGeom.Xformable(plane)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*center))
    xformable.AddScaleOp().Set(Gf.Vec3f(*size))
    plane.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.42, 0.44)])
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())


def _spawn_walls(stage, scene_cfg: SceneConfigModel) -> None:
    bounds = scene_cfg.scene.bounds
    wall_height = scene_cfg.scene.room.wall_height
    thickness = scene_cfg.scene.room.wall_thickness
    x_min, x_max = bounds.x
    y_min, y_max = bounds.y
    walls = [
        ("NorthWall", [(x_min + x_max) / 2.0, y_max, wall_height / 2.0], [x_max - x_min, thickness, wall_height]),
        ("SouthWall", [(x_min + x_max) / 2.0, y_min, wall_height / 2.0], [x_max - x_min, thickness, wall_height]),
        ("EastWall", [x_max, (y_min + y_max) / 2.0, wall_height / 2.0], [thickness, y_max - y_min, wall_height]),
        ("WestWall", [x_min, (y_min + y_max) / 2.0, wall_height / 2.0], [thickness, y_max - y_min, wall_height]),
    ]
    for wall_name, pose, size in walls:
        wall = UsdGeom.Cube.Define(stage, f"/World/Scene/Walls/{wall_name}")
        wall.CreateSizeAttr(1.0)
        xformable = UsdGeom.Xformable(wall)
        xformable.AddTranslateOp().Set(Gf.Vec3d(*pose))
        xformable.AddScaleOp().Set(Gf.Vec3f(*size))
        wall.CreateDisplayColorAttr([Gf.Vec3f(0.76, 0.77, 0.8)])
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())


def _maybe_load_environment_asset(stage, scene_cfg: SceneConfigModel, logger) -> bool:
    env_var = scene_cfg.scene.environment_usd_path_env
    if not env_var:
        return False
    usd_path = os.getenv(env_var)
    if not usd_path:
        return False
    if not Path(usd_path).exists():
        logger.warning("Environment USD path from %s does not exist: %s. Falling back to primitive scene.", env_var, usd_path)
        return False

    prim_path = scene_cfg.scene.environment_prim_path
    env_prim = stage.DefinePrim(prim_path, "Xform")
    env_prim.GetReferences().AddReference(usd_path)
    logger.info("Loaded environment asset from %s into %s", usd_path, prim_path)
    return True


def _spawn_lighting(stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(350.0)
    dome.CreateExposureAttr(0.2)
    dome.CreateColorAttr(Gf.Vec3f(0.97, 0.98, 1.0))

    key_light = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key_light.CreateIntensityAttr(1600.0)
    key_light.CreateAngleAttr(0.6)
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    key_xform = UsdGeom.Xformable(key_light)
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))

    fill_light = UsdLux.SphereLight.Define(stage, "/World/Lights/FillLight")
    fill_light.CreateIntensityAttr(6000.0)
    fill_light.CreateRadiusAttr(0.5)
    fill_light.CreateColorAttr(Gf.Vec3f(0.9, 0.93, 1.0))
    fill_xform = UsdGeom.Xformable(fill_light)
    fill_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -1.5, 3.2))


def build_scene(stage, scene_cfg: SceneConfigModel, logger) -> SemanticSceneState:
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Scene", "Xform")
    stage.DefinePrim("/World/Scene/Objects", "Xform")
    stage.DefinePrim("/World/Scene/Walls", "Xform")
    stage.DefinePrim("/World/Cameras", "Xform")
    stage.DefinePrim("/World/Lights", "Xform")
    loaded_environment_asset = _maybe_load_environment_asset(stage, scene_cfg, logger)
    if not loaded_environment_asset:
        _spawn_ground_plane(stage, scene_cfg.scene.bounds)
        _spawn_walls(stage, scene_cfg)
    _spawn_lighting(stage)

    object_states: list[SemanticObjectState] = []
    for object_cfg in scene_cfg.scene.objects:
        prim_path = spawn_semantic_object(stage, "/World/Scene/Objects", object_cfg)
        object_states.append(
            SemanticObjectState(
                name=object_cfg.name,
                type=object_cfg.type,
                prim_path=prim_path,
                pose=list(object_cfg.pose),
                avoidance_radius=object_cfg.avoidance_radius,
                color=object_cfg.semantic_color,
            )
        )
        logger.info("Spawned semantic object %s at %s", object_cfg.name, object_cfg.pose)

    camera_paths: list[str] = []
    for camera_cfg in scene_cfg.scene.cameras:
        camera = UsdGeom.Camera.Define(stage, f"/World/Cameras/{camera_cfg.name}")
        camera.CreateFocalLengthAttr(28.0 if camera_cfg.name == "main_camera" else 18.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        camera.CreateExposureAttr(0.0)
        _set_transform_with_lookat(camera, camera_cfg.pose, camera_cfg.look_at)
        camera_paths.append(f"/World/Cameras/{camera_cfg.name}")
        logger.info("Created camera %s", camera_cfg.name)

    return SemanticSceneState(
        scene_name=scene_cfg.scene.name,
        bounds={
            "x": list(scene_cfg.scene.bounds.x),
            "y": list(scene_cfg.scene.bounds.y),
            "z_floor": [scene_cfg.scene.bounds.z_floor, scene_cfg.scene.bounds.z_floor],
        },
        object_states=object_states,
        camera_prim_paths=camera_paths,
    )
