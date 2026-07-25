from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal

from demo_2.config import Demo2Config
from demo_2.controller import RealG1Controller, VelocityCommand
from demo_2.errors import HardwareSafetyError, UnsupportedCapabilityError
from demo_2.policy_sil import LEG_JOINT_NAMES, PolicySilTransport, PolicyStateSnapshot
from pathvla.mujoco_lab import MacSortingController
from pathvla.schemas import ConstraintModel, SceneSnapshotModel, SubgoalModel
from pathvla.waypoint_planner import AStarWaypointPlanner


@dataclass(frozen=True)
class BridgeTraceEntry:
    action: str
    mode: str
    detail: str
    command: dict[str, float] | None = None


@dataclass
class SdkSortingBridge:
    """Maps digital-twin movement segments to finite SDK2 velocity calls."""

    controller: RealG1Controller
    config: Demo2Config
    mode: Literal["shadow", "live"]
    twin_aligned: bool = False
    trace: list[BridgeTraceEntry] = field(default_factory=list)

    def initialize(self) -> None:
        self.controller.initialize()
        probe = self.controller.probe()
        self.trace.append(
            BridgeTraceEntry(
                action="probe",
                mode=self.mode,
                detail=f"command backend ready with FSM {probe.probe.fsm_id if probe.probe else None}",
            )
        )

    def move_segment(self, start: list[float], target: list[float]) -> None:
        if self.mode == "live" and not self.twin_aligned:
            raise HardwareSafetyError(
                "Live SDK navigation requires --twin-aligned after measuring the physical lab frame "
                "against the MuJoCo sorting-lab frame."
            )
        dx = float(target[0] - start[0])
        dy = float(target[1] - start[1])
        if math.hypot(dx, dy) < 1e-6:
            return
        limits = self.config.limits
        total_s = max(
            abs(dx) / limits.max_forward_mps,
            abs(dy) / limits.max_lateral_mps,
            1e-3,
        )
        chunks = max(1, math.ceil(total_s / limits.max_command_duration_s))
        duration_s = total_s / chunks
        vx = dx / total_s
        vy = dy / total_s
        for index in range(chunks):
            command = VelocityCommand(vx=vx, vy=vy, yaw_rate=0.0, duration_s=duration_s)
            self.controller.move(command)
            self.trace.append(
                BridgeTraceEntry(
                    action="navigate_segment",
                    mode=self.mode,
                    detail=f"bounded SDK velocity chunk {index + 1}/{chunks}",
                    command={
                        "vx": vx,
                        "vy": vy,
                        "yaw_rate": 0.0,
                        "duration_s": duration_s,
                    },
                )
            )

    def move_policy_segment(
        self,
        target: list[float],
        *,
        pose_provider: Callable[[], list[float]],
        yaw_provider: Callable[[], float],
        position_tolerance_m: float = 0.09,
    ) -> list[float]:
        """Closed-loop waypoint tracking using forward and yaw policy commands."""
        limits = self.config.limits
        enter_walk_heading_rad = 0.25
        leave_walk_heading_rad = 0.90
        walking = False
        locked_turn_heading: float | None = None
        for index in range(240):
            current = pose_provider()
            dx = float(target[0] - current[0])
            dy = float(target[1] - current[1])
            distance = math.hypot(dx, dy)
            if distance <= position_tolerance_m:
                return current
            desired_yaw = math.atan2(dy, dx)
            current_yaw = yaw_provider()
            yaw_error = _wrap_angle(desired_yaw - current_yaw)
            if not walking and locked_turn_heading is None:
                locked_turn_heading = desired_yaw
            turn_error = _wrap_angle(
                (locked_turn_heading if locked_turn_heading is not None else desired_yaw)
                - current_yaw
            )
            if walking and abs(yaw_error) > leave_walk_heading_rad:
                walking = False
                locked_turn_heading = desired_yaw
                turn_error = yaw_error
            elif not walking and abs(turn_error) <= enter_walk_heading_rad:
                walking = True
                locked_turn_heading = None
            if not walking:
                command = VelocityCommand(
                    vx=0.0,
                    vy=0.0,
                    yaw_rate=math.copysign(limits.max_yaw_rate_rps, turn_error),
                    duration_s=limits.max_command_duration_s,
                )
                detail = "policy turn-to-waypoint"
            else:
                body_forward_error = (
                    math.cos(current_yaw) * dx + math.sin(current_yaw) * dy
                )
                body_lateral_error = (
                    -math.sin(current_yaw) * dx + math.cos(current_yaw) * dy
                )
                vx = min(
                    limits.max_forward_mps,
                    max(0.04, body_forward_error / 0.6),
                )
                vy = max(
                    -limits.max_lateral_mps,
                    min(limits.max_lateral_mps, body_lateral_error / 0.6),
                )
                yaw_rate = max(
                    -limits.max_yaw_rate_rps,
                    min(limits.max_yaw_rate_rps, yaw_error * 0.8),
                )
                duration_s = min(
                    limits.max_command_duration_s,
                    max(0.15, distance / max(vx, 1e-6)),
                )
                command = VelocityCommand(
                    vx=vx,
                    vy=vy,
                    yaw_rate=yaw_rate,
                    duration_s=duration_s,
                )
                detail = "policy forward waypoint tracking"
            self.controller.move(command)
            self.trace.append(
                BridgeTraceEntry(
                    action="navigate_segment",
                    mode=self.mode,
                    detail=f"{detail} chunk {index + 1}",
                    command=asdict(command),
                )
            )
        current = pose_provider()
        remaining = math.dist(current[:2], target[:2])
        raise HardwareSafetyError(
            f"Policy locomotion did not converge to waypoint; {remaining:.3f} m remains."
        )

    def move_policy_route(
        self,
        waypoints: list[list[float]],
        *,
        pose_provider: Callable[[], list[float]],
        yaw_provider: Callable[[], float],
    ) -> list[float]:
        """Track a smoothed A* route without forcing a turn every 20 cm."""
        if not waypoints:
            return pose_provider()
        targets = list(waypoints[1::2])
        if not targets or targets[-1] != waypoints[-1]:
            targets.append(waypoints[-1])
        current = pose_provider()
        for target in targets:
            current = self.move_policy_segment(
                target,
                pose_provider=pose_provider,
                yaw_provider=yaw_provider,
                position_tolerance_m=0.22,
            )
        return current

    def manipulation(self, action: str, target: str, destination: str | None = None) -> None:
        if self.mode == "live":
            raise UnsupportedCapabilityError(
                f"Real SDK {action} is blocked for '{target}': configure a calibrated driver for the "
                "exact G1 hand/gripper, force feedback, and camera-to-base transform first."
            )
        destination_text = f" -> {destination}" if destination else ""
        self.trace.append(
            BridgeTraceEntry(
                action=action,
                mode=self.mode,
                detail=(
                    f"shadow only: MuJoCo executes {target}{destination_text}; "
                    "no unsupported SDK manipulation command is emitted"
                ),
            )
        )

    def stop(self) -> None:
        self.controller.stop()
        self.trace.append(BridgeTraceEntry("stop", self.mode, "zero-velocity SDK stop requested"))

    def close(self) -> None:
        self.controller.close()

    def trace_payload(self) -> list[dict[str, object]]:
        return [asdict(entry) for entry in self.trace]


@dataclass
class SdkMirroredSortingController(MacSortingController):
    """Demo 1 simulator skills with an SDK command bridge at execution points."""

    sdk_bridge: SdkSortingBridge | None = None
    _policy_visual_origin: tuple[float, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _policy_sim_origin: tuple[float, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        transport = self._policy_transport()
        if transport is not None:
            self._policy_visual_origin = tuple(self.env.robot_pose())
            self._policy_sim_origin = transport.state_snapshot().position
            transport.set_state_callback(self._sync_policy_state)

    def _move_robot(self, start: list[float], target: list[float]) -> list[float]:
        if self.sdk_bridge is not None and self._policy_transport() is not None:
            return self.sdk_bridge.move_policy_segment(
                target,
                pose_provider=self.env.robot_pose,
                yaw_provider=self._visible_yaw,
                position_tolerance_m=0.15,
            )
        if self.sdk_bridge is not None:
            self.sdk_bridge.move_segment(start, target)
        return super()._move_robot(start, target)

    def _navigate(self, target_name: str) -> str:
        if self.sdk_bridge is None or self._policy_transport() is None:
            return super()._navigate(target_name)
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
                constraints=ConstraintModel(
                    avoid=["safety_pillar"],
                    safe_distance_m=0.55,
                ),
            ),
        )
        self.sdk_bridge.move_policy_route(
            plan.waypoints[1:],
            pose_provider=self.env.robot_pose,
            yaw_provider=self._visible_yaw,
        )
        return f"reached {target_name} via {len(plan.waypoints)} waypoints"

    def _pick(self, target_name: str) -> str:
        if self.sdk_bridge is not None:
            self.sdk_bridge.manipulation("pick", target_name)
        return super()._pick(target_name)

    def _place(self, target_name: str, destination_name: str) -> str:
        if self.sdk_bridge is not None:
            self.sdk_bridge.manipulation("place", target_name, destination_name)
        return super()._place(target_name, destination_name)

    def _policy_transport(self) -> PolicySilTransport | None:
        if self.sdk_bridge is None:
            return None
        transport = self.sdk_bridge.controller.transport
        return transport if isinstance(transport, PolicySilTransport) else None

    def _sync_policy_state(self, state: PolicyStateSnapshot) -> None:
        assert self._policy_visual_origin is not None
        assert self._policy_sim_origin is not None
        visual_x, visual_y, visual_z = self._policy_visual_origin
        sim_x, sim_y, sim_z = self._policy_sim_origin
        base_qpos = self.env.data.joint("floating_base_joint").qpos
        base_qpos[:3] = (
            visual_x + state.position[0] - sim_x,
            visual_y + state.position[1] - sim_y,
            visual_z + state.position[2] - sim_z,
        )
        base_qpos[3:7] = state.quaternion_wxyz
        self.env.data.joint("floating_base_joint").qvel[:] = 0.0
        for name, value in zip(LEG_JOINT_NAMES, state.leg_joint_positions, strict=True):
            self.env.data.joint(name).qpos[0] = value
        self.env.mujoco.mj_forward(self.env.model, self.env.data)
        if self.held_object is not None:
            self.env._set_item_mocap(self.held_object, self.env.grasp_site_pose())
        self.env.sync()

    def _visible_yaw(self) -> float:
        quaternion = self.env.data.joint("floating_base_joint").qpos[3:7]
        qw, qx, qy, qz = (float(value) for value in quaternion)
        return math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi
