from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pxr import Gf, UsdGeom

from pathvla.errors import RobotAssetError
from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import RobotConfigModel


@dataclass
class RobotHandle:
    name: str
    prim_path: str
    is_proxy: bool
    control_mode: str
    controller: Any | None


def _set_translation(prim, translation: list[float]) -> None:
    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translation))


def _instantiate_controller(robot_cfg: RobotConfigModel):
    locomotion = robot_cfg.robot.locomotion
    if not locomotion.controller_module or not locomotion.controller_class:
        return None
    module = importlib.import_module(locomotion.controller_module)
    cls = getattr(module, locomotion.controller_class)
    return cls(**locomotion.controller_kwargs)


def load_robot(
    stage,
    robot_cfg: RobotConfigModel,
    allow_proxy: bool,
    allow_kinematic_control: bool,
    logger,
    spawn_translation: list[float] | None = None,
) -> RobotHandle:
    asset_env = robot_cfg.robot.usd_path_env
    usd_path = os.getenv(asset_env)
    controller = _instantiate_controller(robot_cfg)
    prim_path = robot_cfg.robot.default_prim_path
    resolved_spawn_translation = list(spawn_translation or robot_cfg.robot.spawn_translation)

    if usd_path and Path(usd_path).exists():
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(usd_path)
        _set_translation(prim, resolved_spawn_translation)
        logger.info("Loaded Unitree G1 asset from %s", usd_path)
        if controller is not None:
            return RobotHandle(
                name=robot_cfg.robot.name,
                prim_path=prim_path,
                is_proxy=False,
                control_mode="policy",
                controller=controller,
            )
        if allow_kinematic_control:
            message = "Using kinematic control, not physically realistic humanoid locomotion."
            logger.warning(message)
            print(message)
            return RobotHandle(
                name=robot_cfg.robot.name,
                prim_path=prim_path,
                is_proxy=False,
                control_mode="kinematic",
                controller=None,
            )
        raise RobotAssetError("No locomotion policy/controller configured.")

    if not allow_proxy:
        raise RobotAssetError("Unitree G1 USD asset not found. Set UNITREE_G1_USD_PATH or rerun with --allow-proxy.")

    message = "Using G1 proxy, not real Unitree G1 asset."
    logger.warning(message)
    print(message)
    prim_path = robot_cfg.robot.proxy_prim_path
    proxy = UsdGeom.Capsule.Define(stage, prim_path)
    proxy.CreateRadiusAttr(0.2)
    proxy.CreateHeightAttr(1.4)
    proxy.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.2)])
    _set_translation(proxy.GetPrim(), [resolved_spawn_translation[0], resolved_spawn_translation[1], resolved_spawn_translation[2] + 0.9])

    if controller is not None:
        return RobotHandle(
            name=f"{robot_cfg.robot.name}_proxy",
            prim_path=prim_path,
            is_proxy=True,
            control_mode="policy",
            controller=controller,
        )
    if allow_kinematic_control:
        kinematic_message = "Using kinematic control, not physically realistic humanoid locomotion."
        logger.warning(kinematic_message)
        print(kinematic_message)
        return RobotHandle(
            name=f"{robot_cfg.robot.name}_proxy",
            prim_path=prim_path,
            is_proxy=True,
            control_mode="kinematic",
            controller=None,
        )
    raise RobotAssetError("No locomotion policy/controller configured.")


def load_g1_sorting_robot(stage, robot_cfg: RobotConfigModel, logger, spawn_translation: list[float]) -> RobotHandle:
    """Load the real G1 USD for the bounded simulator skill controller.

    The sorting demo owns root navigation and manipulation skills directly, so it
    intentionally does not instantiate the separate locomotion-policy interface.
    """

    usd_path = os.getenv(robot_cfg.robot.usd_path_env)
    if not usd_path:
        bundled = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "unitree_model"
            / "G1"
            / "29dof"
            / "usd"
            / "g1_29dof_rev_1_0"
            / "g1_29dof_rev_1_0.usd"
        )
        usd_path = str(bundled)
    if not Path(usd_path).is_file():
        raise RobotAssetError(
            "Unitree G1 USD not found. Run scripts/download_g1_usd.sh or set UNITREE_G1_USD_PATH."
        )

    prim_path = robot_cfg.robot.default_prim_path
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(Path(usd_path).resolve()))
    _set_translation(prim, list(spawn_translation))
    logger.info("Loaded official Unitree G1 USD for sorting from %s", usd_path)
    return RobotHandle(
        name=robot_cfg.robot.name,
        prim_path=prim_path,
        is_proxy=False,
        control_mode="sorting_skills",
        controller=None,
    )
