from __future__ import annotations

from pxr import Gf, UsdGeom, UsdPhysics

from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import SceneObjectConfigModel


def _apply_transform(xformable: UsdGeom.Xformable, translation: list[float], scale: list[float]) -> None:
    ops = xformable.GetOrderedXformOps()
    if not ops:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xformable.AddScaleOp().Set(Gf.Vec3f(*scale))
        return
    translate_op = None
    scale_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if scale_op is None:
        scale_op = xformable.AddScaleOp()
    translate_op.Set(Gf.Vec3d(*translation))
    scale_op.Set(Gf.Vec3f(*scale))


def spawn_semantic_object(stage, root_path: str, object_cfg: SceneObjectConfigModel) -> str:
    prim_path = f"{root_path}/{object_cfg.name}"
    if object_cfg.shape == "cube":
        prim = UsdGeom.Cube.Define(stage, prim_path)
        prim.CreateSizeAttr(1.0)
        _apply_transform(UsdGeom.Xformable(prim), object_cfg.pose, object_cfg.size)
    elif object_cfg.shape == "cylinder":
        prim = UsdGeom.Cylinder.Define(stage, prim_path)
        prim.CreateRadiusAttr(float(max(object_cfg.size[0], object_cfg.size[1]) / 2.0))
        prim.CreateHeightAttr(float(object_cfg.size[2]))
        _apply_transform(UsdGeom.Xformable(prim), object_cfg.pose, [1.0, 1.0, 1.0])
    elif object_cfg.shape == "bucket":
        root = stage.DefinePrim(prim_path, "Xform")
        _apply_transform(UsdGeom.Xformable(root), object_cfg.pose, [1.0, 1.0, 1.0])
        width, depth, height = object_cfg.size
        wall = min(width, depth) * 0.09
        pieces = [
            ("Bottom", [0.0, 0.0, wall / 2.0], [width, depth, wall]),
            ("North", [0.0, (depth - wall) / 2.0, height / 2.0], [width, wall, height]),
            ("South", [0.0, -(depth - wall) / 2.0, height / 2.0], [width, wall, height]),
            ("East", [(width - wall) / 2.0, 0.0, height / 2.0], [wall, depth, height]),
            ("West", [-(width - wall) / 2.0, 0.0, height / 2.0], [wall, depth, height]),
        ]
        for piece_name, translation, scale in pieces:
            piece = UsdGeom.Cube.Define(stage, f"{prim_path}/{piece_name}")
            piece.CreateSizeAttr(1.0)
            _apply_transform(UsdGeom.Xformable(piece), translation, scale)
            piece.CreateDisplayColorAttr([Gf.Vec3f(*object_cfg.color)])
            UsdPhysics.CollisionAPI.Apply(piece.GetPrim())
        prim = UsdGeom.Xform(root)
    else:
        raise ValueError(f"Unsupported object shape {object_cfg.shape}")
    if object_cfg.shape != "bucket":
        prim.CreateDisplayColorAttr([Gf.Vec3f(*object_cfg.color)])
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    prim.GetPrim().SetCustomDataByKey("semantic_name", object_cfg.name)
    prim.GetPrim().SetCustomDataByKey("semantic_type", object_cfg.type)
    prim.GetPrim().SetCustomDataByKey("avoidance_radius", object_cfg.avoidance_radius)
    return prim_path
