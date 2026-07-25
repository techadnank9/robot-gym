from __future__ import annotations

import math
from dataclasses import dataclass, field

from pxr import Gf, UsdGeom

from pathvla.errors import PlanningError
from pathvla.schemas import ConstraintModel, SubgoalModel
from pathvla.sorting_agent import (
    GeminiActionDecision,
    SortableObjectModel,
    SortingAction,
    SortingWorldStateModel,
    validate_grounded_action,
)
from pathvla.waypoint_planner import AStarWaypointPlanner

from isaac_ext.pathvla_unitree.tasks.observations import get_prim_translation, refresh_semantic_scene_poses


def _set_translation(stage, prim_path: str, translation: list[float]) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise PlanningError(f"USD prim is missing: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    translate_op = next(
        (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translation))


@dataclass
class SortingSkillTrace:
    action: str
    target: str | None
    destination: str | None
    success: bool
    detail: str
    robot_pose: list[float]


@dataclass
class SortingSkillController:
    stage: object
    robot_handle: object
    semantic_scene: object
    step_fn: object
    logger: object
    item_status: dict[str, str] = field(default_factory=dict)
    assigned_bins: dict[str, str] = field(default_factory=dict)
    held_object: str | None = None
    completed_actions: list[str] = field(default_factory=list)
    rejected_actions: list[str] = field(default_factory=list)
    trace: list[SortingSkillTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        for obj in self.semantic_scene.object_states:
            if obj.type == "sort_item":
                self.item_status[obj.name] = "available"
        if not self.item_status:
            raise PlanningError("Sorting scene has no objects with type 'sort_item'.")

    def world_state(self) -> SortingWorldStateModel:
        refresh_semantic_scene_poses(self.stage, self.semantic_scene)
        robot_pose = get_prim_translation(self.stage, self.robot_handle.prim_path)
        objects = []
        for obj in self.semantic_scene.object_states:
            if obj.type == "sort_item":
                kind = "item"
                status = self.item_status[obj.name]
            elif obj.type == "container":
                kind = "bin"
                status = "static"
            elif obj.type == "obstacle":
                kind = "obstacle"
                status = "static"
            else:
                kind = "surface"
                status = "static"
            objects.append(
                SortableObjectModel(
                    name=obj.name,
                    kind=kind,
                    color=obj.color or "unknown",
                    pose=list(obj.pose),
                    status=status,
                    assigned_bin=self.assigned_bins.get(obj.name),
                )
            )
        return SortingWorldStateModel(
            robot_pose=robot_pose,
            held_object=self.held_object,
            objects=objects,
            completed_actions=list(self.completed_actions),
            rejected_actions=list(self.rejected_actions),
        )

    def reject(self, decision: GeminiActionDecision, reason: str) -> None:
        message = f"{decision.action.value} rejected: {reason}"
        self.rejected_actions.append(message)
        self.logger.warning(message)
        self.trace.append(
            SortingSkillTrace(
                action=decision.action.value,
                target=decision.target,
                destination=decision.destination,
                success=False,
                detail=reason,
                robot_pose=self.world_state().robot_pose,
            )
        )

    def execute(self, decision: GeminiActionDecision) -> str:
        state = self.world_state()
        validate_grounded_action(decision, state)
        if decision.action == SortingAction.NAVIGATE:
            detail = self._navigate(decision.target or "")
        elif decision.action == SortingAction.PICK:
            detail = self._pick(decision.target or "")
        elif decision.action == SortingAction.PLACE:
            detail = self._place(decision.target or "", decision.destination or "")
        else:
            detail = "All items are geometrically sorted; episode complete."
        summary = f"{decision.action.value}: {detail}"
        self.completed_actions.append(summary)
        final_pose = self.world_state().robot_pose
        self.trace.append(
            SortingSkillTrace(
                action=decision.action.value,
                target=decision.target,
                destination=decision.destination,
                success=True,
                detail=detail,
                robot_pose=final_pose,
            )
        )
        self.logger.info(summary)
        return detail

    def _navigate(self, target_name: str) -> str:
        target = self._semantic_object(target_name)
        robot_pose = get_prim_translation(self.stage, self.robot_handle.prim_path)
        snapshot = self.semantic_scene.to_snapshot(self.robot_handle.name, robot_pose)
        subgoal_type = "pickup" if target.type == "sort_item" else "drop"
        avoid = [obj.name for obj in self.semantic_scene.object_states if obj.type == "obstacle"]
        plan = AStarWaypointPlanner(grid_resolution_m=0.2).plan(
            snapshot,
            SubgoalModel(
                type=subgoal_type,
                target=target_name,
                constraints=ConstraintModel(avoid=avoid, safe_distance_m=0.55),
            ),
        )
        current = robot_pose
        for waypoint in plan.waypoints[1:]:
            current = self._move_robot_segment(current, waypoint)
        return f"reached {target_name} in {len(plan.waypoints)} waypoints ({plan.path_length_m:.2f} m)"

    def _pick(self, target_name: str) -> str:
        target = self._semantic_object(target_name)
        robot_pose = get_prim_translation(self.stage, self.robot_handle.prim_path)
        grasp_pose = [robot_pose[0] + 0.28, robot_pose[1] - 0.24, robot_pose[2] + 0.48]
        self._move_object_arc(target.prim_path, target.pose, grasp_pose, lift=0.32)
        self.held_object = target_name
        self.item_status[target_name] = "held"
        refresh_semantic_scene_poses(self.stage, self.semantic_scene)
        return f"grasped {target_name} with the right-hand task-space skill"

    def _place(self, target_name: str, destination_name: str) -> str:
        target = self._semantic_object(target_name)
        destination = self._semantic_object(destination_name)
        current = get_prim_translation(self.stage, target.prim_path)
        place_pose = [destination.pose[0], destination.pose[1], destination.pose[2] + 0.2]
        self._move_object_arc(target.prim_path, current, place_pose, lift=0.25)
        self.held_object = None
        self.item_status[target_name] = "sorted"
        self.assigned_bins[target_name] = destination_name
        refresh_semantic_scene_poses(self.stage, self.semantic_scene)
        return f"released {target_name} inside {destination_name}"

    def _move_robot_segment(self, start: list[float], target: list[float]) -> list[float]:
        planar_distance = math.dist(start[:2], target[:2])
        steps = max(1, int(math.ceil(planar_distance / 0.08)))
        for index in range(1, steps + 1):
            alpha = index / steps
            pose = [
                start[0] + (target[0] - start[0]) * alpha,
                start[1] + (target[1] - start[1]) * alpha,
                start[2],
            ]
            _set_translation(self.stage, self.robot_handle.prim_path, pose)
            if self.held_object is not None:
                held = self._semantic_object(self.held_object)
                _set_translation(
                    self.stage,
                    held.prim_path,
                    [pose[0] + 0.28, pose[1] - 0.24, pose[2] + 0.48],
                )
            self.step_fn()
        return [target[0], target[1], start[2]]

    def _move_object_arc(
        self,
        prim_path: str,
        start: list[float],
        target: list[float],
        lift: float,
        steps: int = 30,
    ) -> None:
        for index in range(1, steps + 1):
            alpha = index / steps
            arc = math.sin(math.pi * alpha) * lift
            pose = [
                start[0] + (target[0] - start[0]) * alpha,
                start[1] + (target[1] - start[1]) * alpha,
                start[2] + (target[2] - start[2]) * alpha + arc,
            ]
            _set_translation(self.stage, prim_path, pose)
            self.step_fn()

    def _semantic_object(self, name: str):
        for obj in self.semantic_scene.object_states:
            if obj.name == name:
                return obj
        raise PlanningError(f"Unknown semantic object '{name}'.")
