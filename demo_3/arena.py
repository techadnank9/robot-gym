from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from demo_3.low_level import DualUnitreeLocomotion
from demo_3.model import (
    BODY_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
    build_dual_g1_xml,
    default_g1_path,
    load_scene,
)
from demo_3.schemas import (
    MatchEvent,
    MatchPhase,
    PlayerConfig,
    PlayerStatus,
    Skill,
    TeleopFrame,
)


RIGHT_ARM_JOINT_NAMES = BODY_JOINT_NAMES[-7:]
RIGHT_HAND_OPEN = np.asarray([0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05])
RIGHT_HAND_CLOSED = np.asarray([-0.45, -0.40, -1.20, 1.10, 1.25, 1.10, 1.25])
RIGHT_ARM_CARRY = {
    "right_shoulder_pitch_joint": 0.20,
    "right_shoulder_roll_joint": -0.34,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.42,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


@dataclass
class PlayerRuntime:
    config: PlayerConfig
    status: PlayerStatus
    frame: TeleopFrame
    last_valid_frame_s: float
    arm_targets: dict[str, float] = field(default_factory=lambda: dict(RIGHT_ARM_CARRY))
    hand_close: float = 0.0
    had_payload_contact: bool = False
    transported: bool = False
    stable_since_s: float | None = None


class DualG1RaceArena:
    """Authoritative shared-world MuJoCo race for Demo 3."""

    def __init__(
        self,
        players: tuple[PlayerConfig, PlayerConfig],
        *,
        scene_path: Path | str | None = None,
        g1_path: Path | str | None = None,
        viewer: bool = False,
        realtime: bool = False,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Demo 3 requires MuJoCo") from exc
        self.mujoco = mujoco
        self.scene = load_scene(scene_path)
        self.xml = build_dual_g1_xml(self.scene, g1_path or default_g1_path())
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.viewer_enabled = viewer
        self.realtime = realtime
        self.viewer = None
        self.match_id = f"demo3-{uuid.uuid4().hex[:10]}"
        self.phase = MatchPhase.LOBBY
        self.winner: str | None = None
        self.started_wall_s: float | None = None
        self.events: list[MatchEvent] = []
        self._control_counter = 0
        self._renderers: dict[tuple[int, int], Any] = {}
        now = time.monotonic()
        self.players: dict[str, PlayerRuntime] = {}
        for config in players:
            status = PlayerStatus(
                player_id=config.player_id,
                display_name=config.display_name,
                mode=config.mode,
                model_name=config.public_model_name,
                api_calls_remaining=5 if config.mode.value == "policy" else None,
            )
            self.players[config.player_id] = PlayerRuntime(
                config=config,
                status=status,
                frame=TeleopFrame.neutral(config.player_id),
                last_valid_frame_s=now,
            )
        if set(self.players) != {"p1", "p2"}:
            raise ValueError("DualG1RaceArena requires exactly p1 and p2")
        self._initialize_pose()
        for player_id, runtime in self.players.items():
            runtime.arm_targets = {
                name: float(
                    self.data.qpos[
                        self.model.joint(f"{player_id}_{name}").qposadr[0]
                    ]
                )
                for name in RIGHT_ARM_JOINT_NAMES
            }
        self.locomotion = DualUnitreeLocomotion(self.model, self.data, self.mujoco)
        self._initialize_actuators()
        mujoco.mj_forward(self.model, self.data)
        if viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.distance = 7.2
            self.viewer.cam.azimuth = 125
            self.viewer.cam.elevation = -24
            self.viewer.sync()

    @property
    def simulation_time_s(self) -> float:
        return float(self.data.time)

    def start(self) -> None:
        if self.phase != MatchPhase.LOBBY:
            raise RuntimeError("match can only start from the lobby")
        self.phase = MatchPhase.RUNNING
        self.started_wall_s = time.monotonic()
        self._event("match_started", payload={"seed": self.scene["seed"]})

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        self._renderers.clear()

    def submit_frame(self, frame: TeleopFrame) -> None:
        runtime = self.players[frame.player_id]
        if frame.sequence <= runtime.frame.sequence and frame.sequence != 0:
            return
        runtime.frame = frame
        runtime.last_valid_frame_s = time.monotonic()
        runtime.status.connected = frame.connected
        runtime.status.current_skill = frame.skill
        runtime.hand_close = frame.hand_close

    def set_skill(
        self,
        player_id: str,
        skill: Skill,
        *,
        rationale: str = "",
        api_calls_remaining: int | None = None,
    ) -> None:
        runtime = self.players[player_id]
        runtime.status.current_skill = skill
        runtime.frame = runtime.frame.model_copy(
            update={
                "sequence": runtime.frame.sequence + 1,
                "timestamp_s": time.monotonic(),
                "connected": True,
                "deadman": True,
                "skill": skill,
            }
        )
        runtime.last_valid_frame_s = time.monotonic()
        runtime.status.rationale = rationale
        if api_calls_remaining is not None:
            runtime.status.api_calls_remaining = api_calls_remaining
        self._event(
            "player_skill",
            player_id=player_id,
            payload={"skill": skill.value, "rationale": rationale},
        )

    def step(self, count: int = 1) -> None:
        if self.phase not in {MatchPhase.RUNNING, MatchPhase.COUNTDOWN}:
            return
        for _ in range(count):
            started = time.perf_counter()
            self._apply_commands()
            self._update_arm_targets()
            self._apply_contact_grip()
            self.locomotion.apply_torques()
            self.mujoco.mj_step(self.model, self.data)
            self._control_counter += 1
            if self._control_counter % self.locomotion.decimation == 0:
                self.locomotion.update(self.simulation_time_s)
                self._sample_match_state()
                if self.viewer is not None:
                    if not self.viewer.is_running():
                        self.phase = MatchPhase.ABORTED
                        return
                    self.viewer.sync()
            if self.realtime:
                remaining = float(self.model.opt.timestep) - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
            if self.phase == MatchPhase.FINISHED:
                return

    def render(self, camera: str, width: int = 640, height: int = 360) -> np.ndarray:
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = self.mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = renderer
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def state_payload(self) -> dict[str, Any]:
        elapsed = (
            max(0.0, time.monotonic() - self.started_wall_s)
            if self.started_wall_s is not None
            else 0.0
        )
        return {
            "protocolVersion": "3.0",
            "matchId": self.match_id,
            "phase": self.phase.value,
            "simulationTime": round(self.simulation_time_s, 4),
            "elapsedTime": round(elapsed, 3),
            "winner": self.winner,
            "result": f"{self.winner.upper()} WON" if self.winner else None,
            "players": {
                player_id: runtime.status.model_dump(mode="json")
                for player_id, runtime in self.players.items()
            },
            "poses": {
                player_id: {
                    "robot": self._base_position(player_id).tolist(),
                    "payload": self._payload_position(player_id).tolist(),
                }
                for player_id in ("p1", "p2")
            },
        }

    def policy_status(self, player_id: str) -> dict[str, Any]:
        runtime = self.players[player_id]
        base = self._base_position(player_id)
        payload = self._payload_position(player_id)
        goal = np.asarray(self.scene["players"][player_id]["goal"], dtype=float)
        return {
            "fallen": runtime.status.fallen,
            "carrying": runtime.status.carrying,
            "checkpointCrossed": runtime.status.checkpoint_crossed,
            "delivered": runtime.status.delivered,
            "nearObject": bool(np.linalg.norm(base[:2] - payload[:2]) < 0.43),
            "nearGoal": bool(np.linalg.norm(base[:2] - goal[:2]) < 0.52),
            "currentSkill": runtime.status.current_skill.value,
            "opponentProgress": self.players["p2" if player_id == "p1" else "p1"].status.progress,
        }

    def write_evidence(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scene.xml").write_text(self.xml, encoding="utf-8")
        (directory / "match_state.json").write_text(
            json.dumps(self.state_payload(), indent=2), encoding="utf-8"
        )
        (directory / "events.json").write_text(
            json.dumps([event.model_dump(mode="json") for event in self.events], indent=2),
            encoding="utf-8",
        )

    def _initialize_pose(self) -> None:
        source_model = self.mujoco.MjModel.from_xml_path(str(default_g1_path()))
        source_data = self.mujoco.MjData(source_model)
        self.mujoco.mj_resetDataKeyframe(source_model, source_data, 0)
        for player_id in ("p1", "p2"):
            for joint_name in (*BODY_JOINT_NAMES, *RIGHT_HAND_JOINT_NAMES):
                source_joint = source_model.joint(joint_name)
                target_joint = self.model.joint(f"{player_id}_{joint_name}")
                self.data.qpos[target_joint.qposadr[0]] = source_data.qpos[source_joint.qposadr[0]]

    def _initialize_actuators(self) -> None:
        for player_id in ("p1", "p2"):
            for name in BODY_JOINT_NAMES[12:]:
                joint = self.model.joint(f"{player_id}_{name}")
                actuator = self.model.actuator(f"{player_id}_{name}")
                self.data.ctrl[actuator.id] = self.data.qpos[joint.qposadr[0]]
            self._set_hand(player_id, 0.0)

    def _apply_commands(self) -> None:
        for player_id, runtime in self.players.items():
            frame = runtime.frame
            human_stale = runtime.config.mode.value == "human" and (
                not frame.connected
                or time.monotonic() - runtime.last_valid_frame_s
                > float(self.scene["scoring"]["stale_target_seconds"])
            )
            if human_stale:
                runtime.status.connected = False
                self.locomotion.set_command(player_id, 0, 0, 0)
                if (
                    time.monotonic() - runtime.last_valid_frame_s
                    > float(self.scene["scoring"]["disconnect_dq_seconds"])
                ):
                    runtime.status.disqualified = True
                continue
            runtime.status.connected = True
            if runtime.config.mode.value == "human":
                if frame.deadman:
                    self.locomotion.set_command(
                        player_id,
                        0.55 * frame.move_y,
                        0.28 * frame.move_x,
                        0.9 * frame.yaw,
                    )
                else:
                    self.locomotion.set_command(player_id, 0, 0, 0)
                continue
            skill = runtime.status.current_skill
            if skill in {Skill.NAVIGATE_OBJECT, Skill.NAVIGATE_GOAL}:
                if skill == Skill.NAVIGATE_OBJECT:
                    target = self.scene["players"][player_id]["object"]
                else:
                    base_position = self._base_position(player_id)
                    base_x = float(base_position[0])
                    base_y = float(base_position[1])
                    holding_lane = base_x < 0.72 if player_id == "p1" else base_x > -0.72
                    clearing_plinth = (
                        player_id == "p1" and base_x < -1.68 and base_y < -0.34
                    ) or (
                        player_id == "p2" and base_x > 1.68 and base_y > 0.34
                    )
                    if clearing_plinth:
                        target = [-1.92, -0.20] if player_id == "p1" else [1.92, 0.20]
                    elif holding_lane:
                        target = [0.82, -0.38] if player_id == "p1" else [-0.82, 0.38]
                    else:
                        target = self.scene["players"][player_id]["goal"]
                self._navigate_to(player_id, np.asarray(target[:2], dtype=float))
            else:
                self.locomotion.set_command(player_id, 0, 0, 0)

    def _navigate_to(self, player_id: str, target_xy: np.ndarray) -> None:
        base = self._base_position(player_id)
        delta = target_xy - base[:2]
        distance = float(np.linalg.norm(delta))
        yaw = self._base_yaw(player_id)
        desired = math.atan2(float(delta[1]), float(delta[0]))
        error = _wrap_angle(desired - yaw)
        forward = min(0.38, max(0.0, distance - 0.08)) * max(0.0, math.cos(error))
        lateral = float(np.clip(math.sin(error) * min(distance, 0.7), -0.22, 0.22))
        self.locomotion.set_command(
            player_id,
            forward,
            lateral,
            float(np.clip(1.8 * error, -0.9, 0.9)),
        )

    def _update_arm_targets(self) -> None:
        for player_id, runtime in self.players.items():
            skill = runtime.status.current_skill
            if skill == Skill.GRASP:
                payload = self._payload_position(player_id)
                self._ik_step(player_id, payload + np.asarray([0.0, 0.0, 0.015]))
                distance = float(
                    np.linalg.norm(self.data.site(f"{player_id}_right_grasp_site").xpos - payload)
                )
                runtime.hand_close = 1.0 if distance < 0.46 else 0.0
            elif skill == Skill.NAVIGATE_GOAL:
                for name, target in RIGHT_ARM_CARRY.items():
                    current = runtime.arm_targets.get(name, target)
                    runtime.arm_targets[name] = current + float(
                        np.clip(target - current, -0.01, 0.01)
                    )
            elif skill == Skill.RELEASE:
                goal = np.asarray(self.scene["players"][player_id]["goal"], dtype=float)
                payload = self._payload_position(player_id)
                if np.linalg.norm(payload[:2] - goal[:2]) > 0.25:
                    runtime.hand_close = 1.0
                    self._ik_step(
                        player_id,
                        np.asarray([goal[0], goal[1], max(0.34, goal[2] + 0.28)]),
                    )
                else:
                    runtime.hand_close = 0.0
            elif runtime.config.mode.value == "human":
                runtime.hand_close = runtime.frame.hand_close
            self._apply_arm(player_id, runtime.arm_targets)
            self._set_hand(player_id, runtime.hand_close)

    def _ik_step(self, player_id: str, target: np.ndarray) -> None:
        runtime = self.players[player_id]
        site_id = self.model.site(f"{player_id}_right_grasp_site").id
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        dof_ids = np.asarray(
            [self.model.joint(f"{player_id}_{name}").dofadr[0] for name in RIGHT_ARM_JOINT_NAMES]
        )
        current = self.data.site(site_id).xpos
        error = np.asarray(target, dtype=float) - current
        jacobian = jacp[:, dof_ids]
        damping = 0.04
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(3), error
        )
        delta = np.clip(delta, -0.035, 0.035)
        for name, change in zip(RIGHT_ARM_JOINT_NAMES, delta, strict=True):
            joint = self.model.joint(f"{player_id}_{name}")
            current_target = float(self.data.qpos[joint.qposadr[0]])
            runtime.arm_targets[name] = float(
                np.clip(current_target + change, joint.range[0], joint.range[1])
            )

    def _apply_arm(self, player_id: str, targets: dict[str, float]) -> None:
        for name, value in targets.items():
            joint = self.model.joint(f"{player_id}_{name}")
            actuator = self.model.actuator(f"{player_id}_{name}")
            self.data.ctrl[actuator.id] = float(np.clip(value, joint.range[0], joint.range[1]))

    def _set_hand(self, player_id: str, close: float) -> None:
        targets = RIGHT_HAND_OPEN + np.clip(close, 0.0, 1.0) * (
            RIGHT_HAND_CLOSED - RIGHT_HAND_OPEN
        )
        for name, value in zip(RIGHT_HAND_JOINT_NAMES, targets, strict=True):
            joint = self.model.joint(f"{player_id}_{name}")
            actuator = self.model.actuator(f"{player_id}_{name}")
            self.data.ctrl[actuator.id] = float(np.clip(value, joint.range[0], joint.range[1]))

    def _apply_contact_grip(self) -> None:
        """Model the hand's missing tactile grip controller with bounded forces.

        The assist is armed only after real payload/robot contact while the
        fingers are closed. It applies force through MuJoCo rather than changing
        object pose or enabling a weld, so collisions and payload dynamics remain
        authoritative.
        """

        for player_id, runtime in self.players.items():
            body_id = self.model.body(f"{player_id}_payload").id
            self.data.xfrc_applied[body_id] = 0.0
            contact = self._player_payload_contact(player_id)
            if contact and runtime.hand_close > 0.65:
                runtime.had_payload_contact = True
            if not runtime.had_payload_contact or runtime.hand_close <= 0.65:
                continue
            payload = self._payload_position(player_id)
            grasp = self.data.site(f"{player_id}_right_grasp_site").xpos
            joint = self.model.joint(f"{player_id}_payload_joint")
            velocity = self.data.qvel[joint.dofadr[0] : joint.dofadr[0] + 3]
            force = 38.0 * (grasp - payload) - 2.8 * velocity
            norm = float(np.linalg.norm(force))
            if norm > 22.0:
                force *= 22.0 / norm
            self.data.xfrc_applied[body_id, :3] = force

    def _sample_match_state(self) -> None:
        for player_id, runtime in self.players.items():
            payload = self._payload_position(player_id)
            grasp = self.data.site(f"{player_id}_right_grasp_site").xpos
            contact = self._player_payload_contact(player_id)
            if contact:
                runtime.had_payload_contact = True
            runtime.status.carrying = bool(
                runtime.had_payload_contact
                and runtime.hand_close > 0.65
                and np.linalg.norm(payload - grasp) < 0.36
                and payload[2] > 0.48
            )
            checkpoint = float(self.scene["arena"]["checkpoint_x"])
            if runtime.status.carrying:
                start_x = float(self.scene["players"][player_id]["spawn"][0])
                crossed = payload[0] >= checkpoint if start_x < checkpoint else payload[0] <= checkpoint
                runtime.status.checkpoint_crossed |= bool(crossed)
                runtime.transported |= runtime.status.checkpoint_crossed
            runtime.status.fallen = self._is_fallen(player_id)
            runtime.status.progress = self._progress(player_id)
            if self._delivery_stable(player_id, runtime):
                runtime.status.delivered = True
                runtime.status.progress = 1.0
                if self.winner is None:
                    self.winner = player_id
                    self.phase = MatchPhase.FINISHED
                    self._event("match_finished", player_id=player_id, payload={"winner": player_id})
        disqualified = [
            player_id
            for player_id, runtime in self.players.items()
            if runtime.status.disqualified
        ]
        if self.phase != MatchPhase.RUNNING or not disqualified:
            return
        if len(disqualified) == 2:
            self.phase = MatchPhase.ABORTED
            self._event("match_aborted", payload={"reason": "both_players_disqualified"})
            return
        loser = disqualified[0]
        self.winner = "p2" if loser == "p1" else "p1"
        self.phase = MatchPhase.FINISHED
        self._event(
            "match_finished",
            player_id=self.winner,
            payload={"winner": self.winner, "reason": "opponent_disqualified"},
        )

    def _delivery_stable(self, player_id: str, runtime: PlayerRuntime) -> bool:
        if not runtime.transported or runtime.hand_close > 0.25:
            runtime.stable_since_s = None
            return False
        payload = self._payload_position(player_id)
        goal = np.asarray(self.scene["players"][player_id]["goal"], dtype=float)
        half = np.asarray(self.scene["scoring"]["goal_half_size"], dtype=float)
        payload_half = np.asarray([0.18, 0.04])
        inside = bool(np.all(np.abs(payload[:2] - goal[:2]) <= half[:2] - payload_half))
        joint = self.model.joint(f"{player_id}_payload_joint")
        velocity = self.data.qvel[joint.dofadr[0] : joint.dofadr[0] + 6]
        stable = (
            inside
            and np.linalg.norm(velocity[:3])
            <= float(self.scene["scoring"]["max_linear_speed_mps"])
            and np.linalg.norm(velocity[3:])
            <= float(self.scene["scoring"]["max_angular_speed_rps"])
        )
        if not stable:
            runtime.stable_since_s = None
            return False
        if runtime.stable_since_s is None:
            runtime.stable_since_s = self.simulation_time_s
        return (
            self.simulation_time_s - runtime.stable_since_s
            >= float(self.scene["scoring"]["stable_seconds"])
        )

    def _player_payload_contact(self, player_id: str) -> bool:
        payload_geom = self.model.geom(f"{player_id}_payload_geom").id
        prefix = f"{player_id}_"
        for contact in self.data.contact:
            if payload_geom not in {contact.geom1, contact.geom2}:
                continue
            other = contact.geom2 if contact.geom1 == payload_geom else contact.geom1
            name = self.model.geom(other).name or ""
            body_name = self.model.body(self.model.geom_bodyid[other]).name or ""
            robot_contact = name.startswith(prefix) or body_name.startswith(prefix)
            arena_contact = name.startswith(
                (
                    f"{player_id}_pickup_",
                    f"{player_id}_goal",
                    f"{player_id}_bucket_",
                    f"{player_id}_payload",
                )
            ) or body_name.startswith(f"{player_id}_payload")
            if robot_contact and not arena_contact:
                return True
        return False

    def _is_fallen(self, player_id: str) -> bool:
        joint = self.model.joint(f"{player_id}_floating_base_joint")
        qpos = self.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 7]
        height = float(qpos[2])
        up_z = 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)
        return bool(height < 0.48 or up_z < math.cos(math.radians(60)))

    def _progress(self, player_id: str) -> float:
        runtime = self.players[player_id]
        if runtime.status.delivered:
            return 1.0
        if runtime.status.checkpoint_crossed:
            return 0.78
        if runtime.status.carrying:
            return 0.48
        if runtime.had_payload_contact:
            return 0.30
        start = np.asarray(self.scene["players"][player_id]["spawn"][:2], dtype=float)
        obj = np.asarray(self.scene["players"][player_id]["object"][:2], dtype=float)
        current = self._base_position(player_id)[:2]
        total = max(0.1, float(np.linalg.norm(obj - start)))
        return float(np.clip(0.18 * np.linalg.norm(current - start) / total, 0.0, 0.18))

    def _base_position(self, player_id: str) -> np.ndarray:
        joint = self.model.joint(f"{player_id}_floating_base_joint")
        return self.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 3].copy()

    def _payload_position(self, player_id: str) -> np.ndarray:
        joint = self.model.joint(f"{player_id}_payload_joint")
        return self.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 3].copy()

    def _base_yaw(self, player_id: str) -> float:
        joint = self.model.joint(f"{player_id}_floating_base_joint")
        q = self.data.qpos[joint.qposadr[0] + 3 : joint.qposadr[0] + 7]
        w, x, y, z = [float(value) for value in q]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _event(
        self,
        event_type: str,
        *,
        player_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            MatchEvent(
                event_type=event_type,
                match_id=self.match_id,
                timestamp_s=time.time(),
                simulation_time_s=self.simulation_time_s,
                player_id=player_id,
                payload=payload or {},
            )
        )


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi
