from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from pathvla.errors import ConfigurationError, PlanningError
from pathvla.schemas import ConstraintModel, SceneSnapshotModel, SubgoalModel
from pathvla.sorting_agent import (
    GeminiActionDecision,
    SortableObjectModel,
    SortingAction,
    SortingWorldStateModel,
    validate_grounded_action,
)
from pathvla.waypoint_planner import AStarWaypointPlanner


RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

RIGHT_HAND_OPEN = {
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.0,
    "right_hand_thumb_2_joint": 0.0,
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_0_joint": 0.0,
    "right_hand_middle_1_joint": 0.0,
}

RIGHT_HAND_CLOSED = {
    "right_hand_thumb_0_joint": -0.55,
    "right_hand_thumb_1_joint": -0.45,
    "right_hand_thumb_2_joint": -1.15,
    "right_hand_index_0_joint": 1.05,
    "right_hand_index_1_joint": 1.25,
    "right_hand_middle_0_joint": 1.05,
    "right_hand_middle_1_joint": 1.25,
}

RIGHT_ARM_CARRY = {
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.35,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.45,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


@dataclass
class MacSkillTrace:
    action: str
    target: str | None
    destination: str | None
    success: bool
    detail: str
    robot_pose: list[float]


def build_sorting_lab_xml(g1_xml_path: Path, scene_cfg: Any) -> str:
    """Add the sorting lab to Menagerie's G1-with-hands MJCF model."""

    if not g1_xml_path.is_file():
        raise ConfigurationError(
            f"G1 MJCF not found: {g1_xml_path}. Run scripts/download_g1_mjcf.sh."
        )
    root = ET.parse(g1_xml_path).getroot()
    root.set("model", "gemini_g1_sorting_lab")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str((g1_xml_path.parent / "assets").resolve()))

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.7 0.7 0.7",
        ambient="0.25 0.25 0.25",
        specular="0.5 0.5 0.5",
    )
    ET.SubElement(visual, "global", azimuth="135", elevation="-24")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ConfigurationError("G1 MJCF has no worldbody.")
    right_wrist = root.find(".//body[@name='right_wrist_yaw_link']")
    if right_wrist is None:
        raise ConfigurationError("G1 MJCF has no right wrist body for grasp control.")
    ET.SubElement(
        right_wrist,
        "site",
        name="right_grasp_site",
        pos="0.14 -0.0046 0",
        size="0.012",
        rgba="0.1 1 0.2 0.65",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="lab_floor",
        type="plane",
        size="5 4 0.05",
        rgba="0.16 0.18 0.21 1",
        friction="0.9 0.02 0.002",
    )
    ET.SubElement(worldbody, "light", pos="0 -1 4", dir="0 0 -1", directional="true")
    ET.SubElement(worldbody, "light", pos="3 -3 2.5", diffuse="0.55 0.60 0.70")

    x_min, x_max = scene_cfg.scene.bounds.x
    y_min, y_max = scene_cfg.scene.bounds.y
    wall_h = scene_cfg.scene.room.wall_height
    wall_t = scene_cfg.scene.room.wall_thickness
    walls = [
        ("north_wall", [(x_min + x_max) / 2, y_max, wall_h / 2], [(x_max - x_min) / 2, wall_t / 2, wall_h / 2]),
        ("south_wall", [(x_min + x_max) / 2, y_min, wall_h / 2], [(x_max - x_min) / 2, wall_t / 2, wall_h / 2]),
        ("east_wall", [x_max, (y_min + y_max) / 2, wall_h / 2], [wall_t / 2, (y_max - y_min) / 2, wall_h / 2]),
        ("west_wall", [x_min, (y_min + y_max) / 2, wall_h / 2], [wall_t / 2, (y_max - y_min) / 2, wall_h / 2]),
    ]
    for name, pos, size in walls:
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="box",
            pos=_numbers(pos),
            size=_numbers(size),
            rgba="0.44 0.47 0.52 1",
        )

    for obj in scene_cfg.scene.objects:
        if obj.shape == "bucket":
            _add_bucket(worldbody, obj)
        elif obj.type == "sort_item":
            _add_sort_item(worldbody, obj)
        else:
            geom_type = "box" if obj.shape == "cube" else "cylinder"
            size = _mujoco_size(obj.shape, obj.size)
            body = ET.SubElement(worldbody, "body", name=obj.name, pos=_numbers(obj.pose))
            ET.SubElement(
                body,
                "geom",
                name=f"{obj.name}_geom",
                type=geom_type,
                size=_numbers(size),
                rgba=_numbers([*obj.color, 1.0]),
                friction="0.8 0.02 0.002",
            )

    target = ET.SubElement(worldbody, "body", name="camera_target", pos="0 0 0.75", mocap="true")
    ET.SubElement(target, "geom", type="sphere", size="0.01", rgba="0 0 0 0", contype="0", conaffinity="0")
    for camera in scene_cfg.scene.cameras:
        ET.SubElement(
            worldbody,
            "camera",
            name=camera.name,
            pos=_numbers(camera.pose),
            mode="targetbody",
            target="camera_target",
            fovy="58" if camera.name == "main_camera" else "55",
        )
    return ET.tostring(root, encoding="unicode")


def _add_sort_item(worldbody: ET.Element, obj: Any) -> None:
    body = ET.SubElement(worldbody, "body", name=obj.name, pos=_numbers(obj.pose), mocap="true")
    ET.SubElement(
        body,
        "geom",
        name=f"{obj.name}_geom",
        type="box" if obj.shape == "cube" else "cylinder",
        size=_numbers(_mujoco_size(obj.shape, obj.size)),
        rgba=_numbers([*obj.color, 1.0]),
        mass="0.12",
        friction="0.8 0.02 0.002",
    )


def _add_bucket(worldbody: ET.Element, obj: Any) -> None:
    body = ET.SubElement(worldbody, "body", name=obj.name, pos=_numbers(obj.pose))
    width, depth, height = obj.size
    wall = min(width, depth) * 0.09
    pieces = [
        ("bottom", [0, 0, wall / 2], [width / 2, depth / 2, wall / 2]),
        ("north", [0, (depth - wall) / 2, height / 2], [width / 2, wall / 2, height / 2]),
        ("south", [0, -(depth - wall) / 2, height / 2], [width / 2, wall / 2, height / 2]),
        ("east", [(width - wall) / 2, 0, height / 2], [wall / 2, depth / 2, height / 2]),
        ("west", [-(width - wall) / 2, 0, height / 2], [wall / 2, depth / 2, height / 2]),
    ]
    for piece, pos, size in pieces:
        ET.SubElement(
            body,
            "geom",
            name=f"{obj.name}_{piece}",
            type="box",
            pos=_numbers(pos),
            size=_numbers(size),
            rgba=_numbers([*obj.color, 1.0]),
            friction="0.9 0.02 0.002",
        )


def _mujoco_size(shape: str, size: list[float]) -> list[float]:
    if shape == "cube":
        return [value / 2 for value in size]
    if shape == "cylinder":
        return [max(size[0], size[1]) / 2, size[2] / 2]
    raise ValueError(f"Unsupported MuJoCo shape: {shape}")


def _numbers(values) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


class MacMuJoCoSortingEnv:
    def __init__(self, g1_xml_path: Path, scene_cfg: Any, output_dir: Path, logger, headless: bool):
        try:
            import mujoco
        except ImportError as exc:
            raise ConfigurationError("Install requirements-mac.txt to run MuJoCo.") from exc
        self.mujoco = mujoco
        self.scene_cfg = scene_cfg
        self.output_dir = output_dir
        self.logger = logger
        xml = build_sorting_lab_xml(g1_xml_path, scene_cfg)
        (output_dir / "compiled_sorting_lab.xml").write_text(xml, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        stand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if stand_id < 0:
            raise ConfigurationError("The Menagerie G1 model has no stand keyframe.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, stand_id)
        self.data.ctrl[:] = self.model.key_ctrl[stand_id]
        base_qpos = self.data.joint("floating_base_joint").qpos
        base_qpos[0:3] = scene_cfg.scene.robot_spawn
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.viewer = None
        self._ui_status = "Initializing"
        self._ui_decision: GeminiActionDecision | None = None
        self._ui_progress: list[str] = []
        if not headless:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            self.viewer.cam.distance = 8.0
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -24
            self.set_agent_ui("Ready — waiting for task")
            self.viewer.sync()
        self.sync()

    def robot_pose(self) -> list[float]:
        qpos = self.data.joint("floating_base_joint").qpos
        return [float(qpos[0]), float(qpos[1]), float(qpos[2])]

    def object_pose(self, name: str) -> list[float]:
        pos = self.data.body(name).xpos
        return [float(pos[0]), float(pos[1]), float(pos[2])]

    def set_robot_pose(self, pose: list[float]) -> None:
        qpos = self.data.joint("floating_base_joint").qpos
        qpos[:3] = pose[:3]
        self.data.joint("floating_base_joint").qvel[:] = 0
        self.sync()

    def set_item_pose(self, name: str, pose: list[float]) -> None:
        self._set_item_mocap(name, pose)
        self.sync()

    def _set_item_mocap(self, name: str, pose: list[float]) -> None:
        body_id = self.model.body(name).id
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise PlanningError(f"Item '{name}' is not a mocap body.")
        self.data.mocap_pos[mocap_id] = pose

    def grasp_site_pose(self) -> list[float]:
        pos = self.data.site("right_grasp_site").xpos
        return [float(pos[0]), float(pos[1]), float(pos[2])]

    def set_joint_targets(
        self,
        targets: dict[str, float],
        steps: int = 24,
        follow_item: str | None = None,
    ) -> None:
        starts = {name: float(self.data.joint(name).qpos[0]) for name in targets}
        for index in range(1, steps + 1):
            alpha = 0.5 - 0.5 * math.cos(math.pi * index / steps)
            for name, target in targets.items():
                value = starts[name] + (target - starts[name]) * alpha
                joint = self.model.joint(name)
                value = float(np.clip(value, joint.range[0], joint.range[1]))
                self.data.joint(name).qpos[0] = value
                actuator_id = self.model.actuator(name).id
                self.data.ctrl[actuator_id] = value
            self.mujoco.mj_forward(self.model, self.data)
            if follow_item is not None:
                self._set_item_mocap(follow_item, self.grasp_site_pose())
            self.sync(0.012 if self.viewer is not None else 0.0)

    def move_right_hand_to(
        self,
        target: list[float],
        follow_item: str | None = None,
        max_iterations: int = 90,
        tolerance_m: float = 0.04,
    ) -> None:
        site_id = self.model.site("right_grasp_site").id
        dof_ids = np.array([self.model.joint(name).dofadr[0] for name in RIGHT_ARM_JOINTS])
        qpos_ids = [self.model.joint(name).qposadr[0] for name in RIGHT_ARM_JOINTS]
        target_array = np.asarray(target, dtype=float)
        for _ in range(max_iterations):
            self.mujoco.mj_forward(self.model, self.data)
            error = target_array - self.data.site_xpos[site_id]
            if float(np.linalg.norm(error)) <= tolerance_m:
                if follow_item is not None:
                    self._set_item_mocap(follow_item, self.grasp_site_pose())
                    self.sync()
                return
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            jacobian = jacp[:, dof_ids]
            damping = 0.035
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(3),
                error,
            )
            max_delta = float(np.max(np.abs(delta)))
            if max_delta > 0.09:
                delta *= 0.09 / max_delta
            for name, qpos_id, joint_delta in zip(RIGHT_ARM_JOINTS, qpos_ids, delta, strict=True):
                joint = self.model.joint(name)
                value = float(np.clip(self.data.qpos[qpos_id] + joint_delta, joint.range[0], joint.range[1]))
                self.data.qpos[qpos_id] = value
                self.data.ctrl[self.model.actuator(name).id] = value
            self.mujoco.mj_forward(self.model, self.data)
            if follow_item is not None:
                self._set_item_mocap(follow_item, self.grasp_site_pose())
            self.sync(0.01 if self.viewer is not None else 0.0)
        final_error = math.dist(self.grasp_site_pose(), target)
        raise PlanningError(f"Right-hand IK could not reach target; final error={final_error:.3f} m")

    def set_agent_ui(
        self,
        status: str,
        decision: GeminiActionDecision | None = None,
        progress: list[str] | None = None,
    ) -> None:
        self._ui_status = status
        if decision is not None:
            self._ui_decision = decision
        if progress is not None:
            self._ui_progress = list(progress)
        if self.viewer is None or not self.viewer.is_running():
            return
        current = self._ui_decision
        left = "GEMINI ROBOTICS-ER 1.6\n" f"Status: {self._ui_status}"
        right = ""
        if current is not None:
            left += f"\nAction: {current.action.value}"
            if current.target:
                left += f"\nTarget: {current.target}"
            if current.destination:
                left += f" -> {current.destination}"
            right = (
                "RATIONALE\n"
                f"{_wrap_overlay(current.rationale)}\n\n"
                "EXPECTED OUTCOME\n"
                f"{_wrap_overlay(current.expected_outcome)}"
            )
        progress_text = "PROGRESS\n" + (
            "\n".join(self._ui_progress[-5:]) if self._ui_progress else "No completed actions yet"
        )
        self.viewer.set_texts(
            [
                (
                    self.mujoco.mjtFontScale.mjFONTSCALE_150,
                    self.mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    left,
                    right,
                ),
                (
                    self.mujoco.mjtFontScale.mjFONTSCALE_100,
                    self.mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    progress_text,
                    "",
                ),
            ]
        )

    def sync(self, delay_s: float = 0.0) -> None:
        self.mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
        if delay_s > 0:
            time.sleep(delay_s)

    def capture(self, action_index: int) -> list[Path]:
        frame_paths = []
        for camera_index, camera in enumerate(self.scene_cfg.scene.cameras):
            self.renderer.update_scene(self.data, camera=camera.name)
            pixels = self.renderer.render()
            frame_path = self.output_dir / "agent_frames" / f"step_{action_index:03d}_camera_{camera_index}.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(frame_path, pixels)
            frame_paths.append(frame_path)
        return frame_paths

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        self.renderer.close()


@dataclass
class MacSortingController:
    env: MacMuJoCoSortingEnv
    logger: Any
    item_status: dict[str, str] = field(default_factory=dict)
    assigned_bins: dict[str, str] = field(default_factory=dict)
    held_object: str | None = None
    completed_actions: list[str] = field(default_factory=list)
    rejected_actions: list[str] = field(default_factory=list)
    trace: list[MacSkillTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        for obj in self.env.scene_cfg.scene.objects:
            if obj.type == "sort_item":
                self.item_status[obj.name] = "available"

    def world_state(self) -> SortingWorldStateModel:
        objects = []
        for obj in self.env.scene_cfg.scene.objects:
            kind = "item" if obj.type == "sort_item" else "bin" if obj.type == "container" else "obstacle" if obj.type == "obstacle" else "surface"
            objects.append(
                SortableObjectModel(
                    name=obj.name,
                    kind=kind,
                    color=obj.semantic_color or "unknown",
                    pose=self.env.object_pose(obj.name),
                    status=self.item_status[obj.name] if kind == "item" else "static",
                    assigned_bin=self.assigned_bins.get(obj.name),
                )
            )
        return SortingWorldStateModel(
            robot_pose=self.env.robot_pose(),
            held_object=self.held_object,
            objects=objects,
            completed_actions=list(self.completed_actions),
            rejected_actions=list(self.rejected_actions),
        )

    def reject(self, decision: GeminiActionDecision, reason: str) -> None:
        self.rejected_actions.append(f"{decision.action.value} rejected: {reason}")
        self.trace.append(MacSkillTrace(decision.action.value, decision.target, decision.destination, False, reason, self.env.robot_pose()))

    def execute(self, decision: GeminiActionDecision) -> str:
        validate_grounded_action(decision, self.world_state())
        self.env.set_agent_ui(
            f"Executing {decision.action.value}",
            decision=decision,
            progress=self.completed_actions,
        )
        if decision.action == SortingAction.NAVIGATE:
            detail = self._navigate(decision.target or "")
        elif decision.action == SortingAction.PICK:
            detail = self._pick(decision.target or "")
        elif decision.action == SortingAction.PLACE:
            detail = self._place(decision.target or "", decision.destination or "")
        else:
            detail = "all items are geometrically sorted"
        self.completed_actions.append(f"{decision.action.value}: {detail}")
        self.env.set_agent_ui(
            "Action complete",
            decision=decision,
            progress=self.completed_actions,
        )
        self.trace.append(MacSkillTrace(decision.action.value, decision.target, decision.destination, True, detail, self.env.robot_pose()))
        self.logger.info("%s: %s", decision.action.value, detail)
        return detail

    def _navigate(self, target_name: str) -> str:
        target_cfg = self._object_cfg(target_name)
        state = self.world_state()
        snapshot = SceneSnapshotModel(
            scene_name="sorting_lab",
            objects=[
                {
                    "name": obj.name,
                    "pose": obj.pose,
                    "type": self._object_cfg(obj.name).type,
                    "avoidance_radius": self._object_cfg(obj.name).avoidance_radius,
                }
                for obj in state.objects
            ],
            robot={"name": "unitree_g1", "pose": state.robot_pose},
            bounds={
                "x": list(self.env.scene_cfg.scene.bounds.x),
                "y": list(self.env.scene_cfg.scene.bounds.y),
                "z_floor": [0.0, 0.0],
            },
        )
        plan = AStarWaypointPlanner(0.2).plan(
            snapshot,
            SubgoalModel(
                type="pickup" if target_cfg.type == "sort_item" else "drop",
                target=target_name,
                constraints=ConstraintModel(avoid=["safety_pillar"], safe_distance_m=0.55),
            ),
        )
        current = state.robot_pose
        for waypoint in plan.waypoints[1:]:
            current = self._move_robot(current, waypoint)
        return f"reached {target_name} via {len(plan.waypoints)} waypoints"

    def _pick(self, target_name: str) -> str:
        item = self.env.object_pose(target_name)
        current = self.env.robot_pose()
        self._move_robot(current, [item[0] - 0.4, item[1] + 0.24, current[2]])
        self.env.set_joint_targets(RIGHT_HAND_OPEN, steps=18)
        self.env.move_right_hand_to([item[0], item[1], item[2] + 0.18])
        self.env.move_right_hand_to([item[0], item[1], item[2] + 0.025])
        self.env.set_joint_targets(RIGHT_HAND_CLOSED, steps=24)
        self.held_object = target_name
        self.item_status[target_name] = "held"
        self.env.set_item_pose(target_name, self.env.grasp_site_pose())
        grasp = self.env.grasp_site_pose()
        self.env.move_right_hand_to(
            [grasp[0], grasp[1], grasp[2] + 0.25],
            follow_item=target_name,
        )
        self.env.set_joint_targets(RIGHT_ARM_CARRY, steps=28, follow_item=target_name)
        return f"grasped {target_name} with articulated right arm and fingers"

    def _place(self, target_name: str, destination_name: str) -> str:
        destination = self.env.object_pose(destination_name)
        current = self.env.robot_pose()
        self._move_robot(
            current,
            [destination[0] - 0.3, destination[1] + 0.24, current[2]],
        )
        self.env.move_right_hand_to(
            [destination[0], destination[1], destination[2] + 0.85],
            follow_item=target_name,
        )
        self.env.move_right_hand_to(
            [destination[0], destination[1], destination[2] + 0.70],
            follow_item=target_name,
        )
        self.env.set_joint_targets(RIGHT_HAND_OPEN, steps=24, follow_item=target_name)
        self.env.set_item_pose(
            target_name,
            [destination[0], destination[1], destination[2] + 0.2],
        )
        self.held_object = None
        self.item_status[target_name] = "sorted"
        self.assigned_bins[target_name] = destination_name
        retract = self.env.grasp_site_pose()
        self.env.move_right_hand_to([retract[0] - 0.18, retract[1], retract[2] + 0.2])
        self.env.set_joint_targets(RIGHT_ARM_CARRY, steps=24)
        return f"opened right hand and released {target_name} inside {destination_name}"

    def _move_robot(self, start: list[float], target: list[float]) -> list[float]:
        distance = math.dist(start[:2], target[:2])
        steps = max(1, math.ceil(distance / 0.07))
        for index in range(1, steps + 1):
            alpha = index / steps
            pose = [start[0] + (target[0] - start[0]) * alpha, start[1] + (target[1] - start[1]) * alpha, start[2]]
            self.env.set_robot_pose(pose)
            if self.held_object:
                self.env.set_item_pose(self.held_object, self.env.grasp_site_pose())
            self.env.sync(0.008 if self.env.viewer is not None else 0.0)
        return [target[0], target[1], start[2]]

    def _object_cfg(self, name: str):
        for obj in self.env.scene_cfg.scene.objects:
            if obj.name == name:
                return obj
        raise PlanningError(f"Unknown object '{name}'.")


def _wrap_overlay(text: str, width: int = 46) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and len(" ".join([*current, word])) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)
