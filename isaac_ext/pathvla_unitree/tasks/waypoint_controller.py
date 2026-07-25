from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from pxr import Gf, UsdGeom

from pathvla.errors import PlanningError
from pathvla.schemas import SceneSnapshotModel, WaypointPlanModel


def _set_robot_translation(stage, prim_path: str, translation: list[float]) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translation))


@dataclass
class WaypointExecutionResult:
    completed_subgoals: int
    trace: list[list[float]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


class WaypointController:
    def __init__(self, stage, robot_handle, step_fn, logger, waypoint_tolerance_m: float = 0.2):
        self.stage = stage
        self.robot_handle = robot_handle
        self.step_fn = step_fn
        self.logger = logger
        self.waypoint_tolerance_m = waypoint_tolerance_m

    def execute(
        self,
        waypoint_plans: list[WaypointPlanModel],
        scene_snapshot: SceneSnapshotModel,
        trace_path: Path | None = None,
    ) -> WaypointExecutionResult:
        result = WaypointExecutionResult(completed_subgoals=0)
        current_pose = list(scene_snapshot.robot.pose)
        result.trace.append(list(current_pose))

        for waypoint_plan in waypoint_plans:
            self.logger.info("Executing subgoal %s -> %s", waypoint_plan.subgoal_type.value, waypoint_plan.target)
            try:
                for waypoint in waypoint_plan.waypoints:
                    current_pose = self._move_to_waypoint(current_pose, waypoint)
                    result.trace.append(list(current_pose))
                    self._check_clearance(current_pose, scene_snapshot, active_target=waypoint_plan.target)
                result.completed_subgoals += 1
            except PlanningError as exc:
                message = f"Execution failed for {waypoint_plan.target}: {exc}"
                self.logger.error(message)
                result.failures.append(message)
                break

        if trace_path is not None:
            trace_path.write_text(json.dumps(result.trace, indent=2), encoding="utf-8")
        return result

    def _move_to_waypoint(self, current_pose: list[float], waypoint: list[float]) -> list[float]:
        x, y, z = current_pose
        target = waypoint[:3]
        while math.dist((x, y), (target[0], target[1])) > self.waypoint_tolerance_m:
            if self.robot_handle.control_mode == "policy":
                if self.robot_handle.controller is None:
                    raise PlanningError("Policy control requested but no controller instance is available.")
                if not hasattr(self.robot_handle.controller, "compute_next_pose"):
                    raise PlanningError("Configured controller must expose compute_next_pose(current_pose, target_pose).")
                next_pose = self.robot_handle.controller.compute_next_pose([x, y, z], target)
                x, y, z = next_pose[:3]
            elif self.robot_handle.control_mode == "kinematic":
                dx = target[0] - x
                dy = target[1] - y
                norm = max(math.dist((x, y), (target[0], target[1])), 1e-6)
                step_size = min(0.08, norm)
                x += step_size * dx / norm
                y += step_size * dy / norm
                z = current_pose[2]
            else:
                raise PlanningError(f"Unsupported control mode {self.robot_handle.control_mode}")
            _set_robot_translation(self.stage, self.robot_handle.prim_path, [x, y, z])
            self.step_fn()
        final_pose = [target[0], target[1], current_pose[2]]
        _set_robot_translation(self.stage, self.robot_handle.prim_path, final_pose)
        self.step_fn()
        return list(final_pose)

    def _check_clearance(self, pose: list[float], scene_snapshot: SceneSnapshotModel, active_target: str | None = None) -> None:
        for obj in scene_snapshot.objects:
            if active_target is not None and obj.name == active_target:
                continue
            if obj.type not in {"obstacle", "inspectable", "container"}:
                continue
            distance = math.dist(pose[:2], obj.pose[:2]) - obj.avoidance_radius
            if distance < 0.0:
                raise PlanningError(f"Collision/proximity violation with {obj.name}: distance={distance:.3f}")
