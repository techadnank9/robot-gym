from __future__ import annotations

import json
import math
from pathlib import Path
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from demo_3.arena import (
    DualG1RaceArena,
    PlayerRuntime,
    RIGHT_ARM_CARRY,
    RIGHT_ARM_JOINT_NAMES,
)
from demo_3.low_level import DualUnitreeLocomotion
from demo_3.model import (
    build_dual_g1_xml,
    default_g1_path,
    load_scene,
)
from demo_3.schemas import (
    MatchPhase,
    PlayerConfig,
    PlayerStatus,
    Skill,
    TeleopFrame,
)
from demo_5.command_channel import ConstrainedLocomotion
from demo_5.evidence import compare_hardware_reference
from demo_5.model import add_template_73_background
from demo_5.perception import DelayedCameraPerception


EASY_GRASP_RADIUS_HUMAN_M = 1.25
EASY_GRASP_RADIUS_POLICY_M = 1.45
POLICY_NEAR_OBJECT_RADIUS_M = 0.60
EASY_RELEASE_RADIUS_M = 0.90
EASY_RELEASE_DROP_HEIGHT_M = 0.38


class SimToRealG1RaceArena(DualG1RaceArena):
    """Demo 3 race rules with a deliberately hardware-shaped control boundary."""

    def __init__(
        self,
        players: tuple[PlayerConfig, PlayerConfig],
        *,
        viewer: bool = False,
        realtime: bool = False,
        domain_seed: int = 5,
        hardware_log: Path | None = None,
        viewer_key_callback: Callable[[int], None] | None = None,
        grasp_mode: Literal["easy", "mechanical"] = "easy",
    ) -> None:
        if grasp_mode not in {"easy", "mechanical"}:
            raise ValueError("grasp_mode must be 'easy' or 'mechanical'")
        self.domain_seed = domain_seed
        self.hardware_log = hardware_log
        self.grasp_mode = grasp_mode
        self.rng = np.random.default_rng(domain_seed)
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Demo 5 requires MuJoCo") from exc
        self.mujoco = mujoco
        self.scene = load_scene()
        self.xml = add_template_73_background(
            build_dual_g1_xml(self.scene, default_g1_path())
        )
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.viewer_enabled = viewer
        self.realtime = realtime
        self.viewer = None
        self.match_id = f"demo5-{uuid.uuid4().hex[:10]}"
        self.phase = MatchPhase.LOBBY
        self.winner: str | None = None
        self.started_wall_s: float | None = None
        self.events = []
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
            raise ValueError("SimToRealG1RaceArena requires exactly p1 and p2")
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
                key_callback=viewer_key_callback,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.reset_viewer_camera()
            self.viewer.sync()
        self.domain_parameters = self._randomize_domain()
        self._payload_reset_pose = {
            player_id: self._payload_joint_qpos(player_id).copy()
            for player_id in ("p1", "p2")
        }
        base_locomotion = self.locomotion
        self.locomotion = ConstrainedLocomotion(
            base_locomotion,
            self.model,
            self.data,
            self.rng,
            motor_strength={
                "p1": float(self.domain_parameters["p1MotorStrength"]),
                "p2": float(self.domain_parameters["p2MotorStrength"]),
            },
            dropout_probability=float(self.domain_parameters["packetDropProbability"]),
            joint_position_noise_rad=float(self.domain_parameters["jointPositionNoiseRad"]),
            joint_velocity_noise_rps=float(self.domain_parameters["jointVelocityNoiseRps"]),
        )
        self.perception = DelayedCameraPerception(
            self.model,
            self.data,
            self.mujoco,
            self.scene,
            self.rng,
            position_noise_m=float(self.domain_parameters["cameraPositionNoiseM"]),
            miss_probability=float(self.domain_parameters["cameraMissProbability"]),
        )
        self._odom = {
            player_id: self._base_position(player_id)
            for player_id in ("p1", "p2")
        }
        self._contact_streak = {"p1": 0, "p2": 0}
        self._contact_lost = {"p1": 0, "p2": 0}
        self._grasp_confirmed = {"p1": False, "p2": False}
        self._grasp_started_s: dict[str, float | None] = {"p1": None, "p2": None}
        self._grasp_phase = {"p1": "approach", "p2": "approach"}
        self._grasp_reference_z: dict[str, float | None] = {"p1": None, "p2": None}
        self._grasp_attempts = {"p1": 0, "p2": 0}
        self._easy_attached = {"p1": False, "p2": False}
        self._payload_collision = {
            player_id: (
                int(
                    self.model.geom_contype[
                        self.model.geom(f"{player_id}_payload_geom").id
                    ]
                ),
                int(
                    self.model.geom_conaffinity[
                        self.model.geom(f"{player_id}_payload_geom").id
                    ]
                ),
            )
            for player_id in ("p1", "p2")
        }
        self._reset_count = {"p1": 0, "p2": 0}
        self._reset_attempt_baseline = {"p1": 0, "p2": 0}
        self._reset_lock_until_s = {"p1": 0.0, "p2": 0.0}
        self.trajectory: list[dict[str, object]] = []

    def close(self) -> None:
        if hasattr(self, "perception"):
            self.perception.close()
        super().close()

    def step(self, count: int = 1) -> None:
        if self.phase not in {MatchPhase.RUNNING, MatchPhase.COUNTDOWN}:
            return
        for _ in range(count):
            started = time.perf_counter()
            self.perception.update(self.simulation_time_s)
            self._apply_commands()
            self._update_arm_targets()
            self._apply_contact_grip()
            self.locomotion.apply_torques()
            self.mujoco.mj_step(self.model, self.data)
            self._control_counter += 1
            if self._control_counter % self.locomotion.decimation == 0:
                self._update_odometry()
                self.locomotion.update(self.simulation_time_s)
                self._sample_match_state()
                self._record_trajectory()
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

    def state_payload(self) -> dict[str, Any]:
        payload = super().state_payload()
        payload["profileVersion"] = "5.0"
        payload.pop("poses", None)
        payload["simToReal"] = {
            "privilegedControl": self.grasp_mode == "easy",
            "graspMode": self.grasp_mode,
            "easyGraspCaptureRadiusM": {
                "human": EASY_GRASP_RADIUS_HUMAN_M,
                "policy": EASY_GRASP_RADIUS_POLICY_M,
            },
            "easyReleaseAssist": {
                "captureRadiusM": EASY_RELEASE_RADIUS_M,
                "dropHeightM": EASY_RELEASE_DROP_HEIGHT_M,
                "appliesTo": ["human", "policy"],
            },
            "graspAttachment": (
                "kinematic_pose_lock" if self.grasp_mode == "easy" else "none"
            ),
            "graspAssistForceN": 0.0,
            "domainSeed": self.domain_seed,
            "perception": self.perception.report() if hasattr(self, "perception") else {},
            "sdkChannels": self.locomotion.report() if hasattr(self.locomotion, "report") else {},
            "graspAttempts": dict(self._grasp_attempts),
            "refereeRecovery": {
                player_id: {
                    "resetAvailable": self._reset_available(player_id),
                    "resetCount": self._reset_count[player_id],
                    "resetsRemaining": max(0, 2 - self._reset_count[player_id]),
                    "failedGraspsSinceReset": (
                        self._grasp_attempts[player_id]
                        - self._reset_attempt_baseline[player_id]
                    ),
                    "penaltyRemainingS": round(
                        max(0.0, self._reset_lock_until_s[player_id] - self.simulation_time_s),
                        3,
                    ),
                }
                for player_id in ("p1", "p2")
            },
        }
        return payload

    def policy_status(self, player_id: str) -> dict[str, Any]:
        runtime = self.players[player_id]
        observation = self.perception.snapshot(player_id)
        base = self._odom[player_id]
        payload = _position(observation.payload.position)
        goal = _position(observation.goal.position)
        near_goal = (
            self._easy_release_ready(player_id)
            if self.grasp_mode == "easy"
            else bool(
                goal is not None
                and np.linalg.norm(base[:2] - goal[:2]) < 0.58
            )
        )
        return {
            "fallen": runtime.status.fallen,
            "carrying": self._grasp_confirmed[player_id],
            "checkpointCrossed": runtime.status.checkpoint_crossed,
            "delivered": runtime.status.delivered,
            "nearObject": bool(
                payload is not None
                and observation.payload.confidence >= 0.2
                and np.linalg.norm(base[:2] - payload[:2])
                < POLICY_NEAR_OBJECT_RADIUS_M
            ),
            "nearGoal": near_goal,
            "objectVisible": observation.payload.position is not None,
            "objectConfidence": observation.payload.confidence,
            "goalSource": observation.goal.source,
            "graspAttempts": self._grasp_attempts[player_id],
            "payloadDown": bool(
                self._payload_position(player_id)[2]
                < max(0.55, float(self._payload_reset_pose[player_id][2]) - 0.12)
            ),
            "resetAvailable": self._reset_available(player_id),
            "resetCount": self._reset_count[player_id],
            "recoveryInstruction": (
                "Select recover to request a penalized referee payload reset."
                if self._reset_available(player_id)
                else "Referee payload reset is unavailable while the object remains reachable."
            ),
            "currentSkill": runtime.status.current_skill.value,
            "opponentProgress": self.players[
                "p2" if player_id == "p1" else "p1"
            ].status.progress,
        }

    def write_evidence(self, directory: Path) -> None:
        super().write_evidence(directory)
        perception_errors = self._perception_errors()
        report = {
            "profileVersion": "5.0",
            "matchId": self.match_id,
            "claims": {
                "controllerUsesGroundTruthObjectPose": self.grasp_mode == "easy",
                "externalGraspAssistForceN": 0.0,
                "kinematicGraspAttachment": self.grasp_mode == "easy",
                "graspMode": self.grasp_mode,
                "officialUnitreeLocomotionPolicy": True,
                "sdkRateAndLatencyModeled": True,
                "cameraAndJointNoiseModeled": True,
                "domainRandomizationEnabled": True,
                "missRecoveryEnabled": True,
            },
            "domainParameters": self.domain_parameters,
            "perception": {
                **self.perception.report(),
                "evaluationOnlyErrors": perception_errors,
            },
            "sdkChannels": self.locomotion.report(),
            "graspAttempts": self._grasp_attempts,
            "refereeRecovery": {
                "maximumResetsPerPlayer": 2,
                "penaltySeconds": 3.0,
                "resetCount": self._reset_count,
            },
            "manipulationEvaluation": self._manipulation_evaluation(),
            "hardwareReplayComparison": compare_hardware_reference(
                self.trajectory,
                self.hardware_log,
            ),
        }
        (directory / "sim_to_real_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        (directory / "trajectory.json").write_text(
            json.dumps(self.trajectory, indent=2),
            encoding="utf-8",
        )
        for player_id, channel in self.locomotion.channels.items():
            (directory / f"{player_id}_sdk_command_trace.json").write_text(
                json.dumps(channel.trace, indent=2),
                encoding="utf-8",
            )

    def _apply_commands(self) -> None:
        for player_id, runtime in self.players.items():
            frame = runtime.frame
            if self.simulation_time_s < self._reset_lock_until_s[player_id]:
                runtime.status.connected = True
                self.locomotion.set_command(player_id, 0.0, 0.0, 0.0)
                continue
            human_stale = runtime.config.mode.value == "human" and (
                not frame.connected
                or time.monotonic() - runtime.last_valid_frame_s
                > float(self.scene["scoring"]["stale_target_seconds"])
            )
            if human_stale:
                runtime.status.connected = False
                self.locomotion.set_command(player_id, 0.0, 0.0, 0.0)
                if (
                    time.monotonic() - runtime.last_valid_frame_s
                    > float(self.scene["scoring"]["disconnect_dq_seconds"])
                ):
                    runtime.status.disqualified = True
                continue
            runtime.status.connected = True
            if runtime.status.current_skill == Skill.RECOVER and not runtime.status.fallen:
                if self._reset_available(player_id):
                    self._perform_payload_reset(player_id)
                else:
                    runtime.status.rationale = (
                        "Referee reset denied: payload remains physically reachable."
                    )
                    runtime.status.current_skill = Skill.WAIT
                self.locomotion.set_command(player_id, 0.0, 0.0, 0.0)
                continue
            if runtime.config.mode.value == "human":
                if frame.deadman:
                    self.locomotion.set_command(
                        player_id,
                        0.52 * frame.move_y,
                        0.26 * frame.move_x,
                        0.90 * frame.yaw,
                    )
                else:
                    self.locomotion.set_command(player_id, 0.0, 0.0, 0.0)
                continue
            skill = runtime.status.current_skill
            if skill == Skill.NAVIGATE_OBJECT:
                estimate = self.perception.snapshot(player_id).payload
                target = _position(estimate.position)
                if target is None or estimate.confidence < 0.2:
                    direction = 1.0 if player_id == "p1" else -1.0
                    self.locomotion.set_command(player_id, 0.0, 0.0, 0.28 * direction)
                else:
                    distance = float(np.linalg.norm(self._odom[player_id][:2] - target[:2]))
                    if (
                        self._grasp_attempts[player_id] > 0
                        and distance < POLICY_NEAR_OBJECT_RADIUS_M
                    ):
                        runtime.status.current_skill = Skill.GRASP
                        runtime.status.rationale = "Recovery controller reacquired the payload."
                        runtime.frame = runtime.frame.model_copy(
                            update={
                                "sequence": runtime.frame.sequence + 1,
                                "skill": Skill.GRASP,
                            }
                        )
                    else:
                        self._navigate_estimate(player_id, target[:2])
            elif skill == Skill.NAVIGATE_GOAL:
                target = self._goal_waypoint(player_id)
                self._navigate_estimate(player_id, target)
            else:
                self.locomotion.set_command(player_id, 0.0, 0.0, 0.0)

    def _navigate_estimate(self, player_id: str, target_xy: np.ndarray) -> None:
        base = self._odom[player_id]
        delta = target_xy - base[:2]
        distance = float(np.linalg.norm(delta))
        yaw = self._base_yaw(player_id) + float(self.rng.normal(0.0, 0.008))
        desired = math.atan2(float(delta[1]), float(delta[0]))
        error = _wrap_angle(desired - yaw)
        forward = min(0.40, max(0.0, distance - 0.10)) * max(0.0, math.cos(error))
        lateral = float(np.clip(math.sin(error) * min(distance, 0.6), -0.20, 0.20))
        self.locomotion.set_command(
            player_id,
            forward,
            lateral,
            float(np.clip(1.6 * error, -0.82, 0.82)),
        )

    def _goal_waypoint(self, player_id: str) -> np.ndarray:
        base = self._odom[player_id]
        base_x, base_y = float(base[0]), float(base[1])
        holding_lane = base_x < 0.72 if player_id == "p1" else base_x > -0.72
        clearing_plinth = (
            player_id == "p1" and base_x < -1.68 and base_y < -0.34
        ) or (
            player_id == "p2" and base_x > 1.68 and base_y > 0.34
        )
        if clearing_plinth:
            return np.asarray(
                [-1.92, -0.20] if player_id == "p1" else [1.92, 0.20],
                dtype=float,
            )
        if holding_lane:
            return np.asarray(
                [0.82, -0.38] if player_id == "p1" else [-0.82, 0.38],
                dtype=float,
            )
        estimate = self.perception.snapshot(player_id).goal
        goal = _position(estimate.position)
        if goal is not None:
            return goal[:2]
        return np.asarray(self.scene["players"][player_id]["goal"][:2], dtype=float)

    def _update_arm_targets(self) -> None:
        for player_id, runtime in self.players.items():
            skill = runtime.status.current_skill
            if skill != Skill.GRASP:
                self._grasp_started_s[player_id] = None
                self._grasp_phase[player_id] = "approach"
                self._grasp_reference_z[player_id] = None
            if skill == Skill.GRASP:
                if self.grasp_mode == "easy":
                    self._update_easy_grasp(player_id)
                    self._apply_arm(player_id, runtime.arm_targets)
                    self._set_hand(player_id, runtime.hand_close)
                    continue
                if self._grasp_started_s[player_id] is None:
                    self._grasp_started_s[player_id] = self.simulation_time_s
                    self._grasp_phase[player_id] = "approach"
                estimate = self.perception.snapshot(player_id).payload
                target = _position(estimate.position)
                if target is not None and estimate.confidence >= 0.2:
                    if self._grasp_reference_z[player_id] is None:
                        self._grasp_reference_z[player_id] = float(target[2])
                    reset_height = float(self._payload_reset_pose[player_id][2])
                    payload_height = float(self._payload_position(player_id)[2])
                    if payload_height < max(0.55, reset_height - 0.12):
                        self._retry_grasp(
                            player_id,
                            "Referee detected a fallen payload; reset is now available.",
                        )
                        self._apply_arm(player_id, runtime.arm_targets)
                        self._set_hand(player_id, runtime.hand_close)
                        continue
                    pregrasp, engage, lift = _grasp_targets(
                        self._odom[player_id],
                        target,
                    )
                    grasp = self.data.site(f"{player_id}_right_grasp_site").xpos
                    phase = self._grasp_phase[player_id]
                    if phase == "approach":
                        self._ik_step(player_id, pregrasp)
                        runtime.hand_close = 0.0
                        if np.linalg.norm(grasp - pregrasp) < 0.10:
                            self._grasp_phase[player_id] = "engage"
                    elif phase == "engage":
                        self._ik_step(player_id, engage)
                        runtime.hand_close = 1.0 if np.linalg.norm(grasp - engage) < 0.12 else 0.25
                        if self._grasp_confirmed[player_id]:
                            self._grasp_phase[player_id] = "lift"
                    else:
                        self._ik_step(player_id, lift)
                        runtime.hand_close = 1.0
                else:
                    runtime.hand_close = 0.0
                started = self._grasp_started_s[player_id]
                if (
                    started is not None
                    and self.simulation_time_s - started > 2.5
                    and not self._grasp_confirmed[player_id]
                ):
                    self._retry_grasp(player_id, "Grasp not confirmed by contact; reacquiring.")
            elif skill == Skill.NAVIGATE_GOAL:
                if not self._grasp_confirmed[player_id]:
                    self._retry_grasp(player_id, "Payload contact lost; retrying grasp.")
                else:
                    for name, target in RIGHT_ARM_CARRY.items():
                        current = runtime.arm_targets.get(name, target)
                        runtime.arm_targets[name] = current + float(
                            np.clip(target - current, -0.008, 0.008)
                        )
                    runtime.hand_close = 1.0
            elif skill == Skill.RELEASE:
                if self.grasp_mode == "easy":
                    near = self._easy_release_ready(player_id)
                else:
                    observation = self.perception.snapshot(player_id)
                    goal = _position(observation.goal.position)
                    near = bool(
                        goal is not None
                        and np.linalg.norm(self._odom[player_id][:2] - goal[:2]) < 0.58
                    )
                runtime.hand_close = 0.0 if near else 1.0
                if near:
                    self._grasp_confirmed[player_id] = False
                elif self.grasp_mode == "easy":
                    distance = self._distance_to_goal(player_id)
                    runtime.status.rationale = (
                        f"Move within {EASY_RELEASE_RADIUS_M:.2f} m of the bucket "
                        f"before releasing ({distance:.2f} m)."
                    )
            elif runtime.config.mode.value == "human":
                runtime.hand_close = runtime.frame.hand_close
            self._apply_arm(player_id, runtime.arm_targets)
            self._set_hand(player_id, runtime.hand_close)

    def _apply_contact_grip(self) -> None:
        if self.grasp_mode == "easy":
            self._apply_easy_grip()
            return
        for player_id, runtime in self.players.items():
            body_id = self.model.body(f"{player_id}_payload").id
            self.data.xfrc_applied[body_id] = 0.0
            contact = self._player_payload_contact(player_id)
            if contact and runtime.hand_close > 0.65:
                runtime.had_payload_contact = True
                self._contact_streak[player_id] += 1
                self._contact_lost[player_id] = 0
                if self._contact_streak[player_id] >= 3:
                    self._grasp_confirmed[player_id] = True
            else:
                self._contact_streak[player_id] = 0
                self._contact_lost[player_id] += 1
                if self._contact_lost[player_id] > 100 or runtime.hand_close < 0.25:
                    self._grasp_confirmed[player_id] = False

    def _update_easy_grasp(self, player_id: str) -> None:
        runtime = self.players[player_id]
        payload = self._payload_position(player_id)
        base = self._base_position(player_id)
        distance = float(np.linalg.norm(payload[:2] - base[:2]))
        capture_radius = (
            EASY_GRASP_RADIUS_POLICY_M
            if runtime.config.mode.value == "policy"
            else EASY_GRASP_RADIUS_HUMAN_M
        )
        if not self._easy_attached[player_id] and distance <= capture_radius:
            self._set_easy_attached(player_id, True)
            runtime.had_payload_contact = True
            runtime.status.rationale = (
                "Easy grasp attached the payload to the right hand."
            )
            self._grasp_phase[player_id] = "assisted"
            self._event(
                "assisted_grasp_attached",
                player_id=player_id,
                payload={
                    "captureRadiusM": capture_radius,
                    "distanceM": round(distance, 4),
                },
            )
        if self._easy_attached[player_id]:
            self._grasp_confirmed[player_id] = True
            runtime.hand_close = 1.0
            for name, target in RIGHT_ARM_CARRY.items():
                current = runtime.arm_targets.get(name, target)
                runtime.arm_targets[name] = current + float(
                    np.clip(target - current, -0.018, 0.018)
                )
        else:
            runtime.hand_close = 0.0
            runtime.status.rationale = (
                f"Move within {capture_radius:.2f} m of the payload to use easy grasp "
                f"({distance:.2f} m)."
            )

    def _apply_easy_grip(self) -> None:
        for player_id, runtime in self.players.items():
            body_id = self.model.body(f"{player_id}_payload").id
            self.data.xfrc_applied[body_id] = 0.0
            if not self._easy_attached[player_id]:
                continue
            if (
                runtime.status.current_skill == Skill.RELEASE
                and runtime.hand_close < 0.25
                and self._easy_release_ready(player_id)
            ):
                self._position_payload_for_easy_release(player_id)
                self._set_easy_attached(player_id, False)
                self._grasp_confirmed[player_id] = False
                runtime.had_payload_contact = False
                runtime.status.rationale = (
                    "Easy release aligned the payload above the bucket; gravity drop active."
                )
                self._event(
                    "assisted_grasp_released",
                    player_id=player_id,
                    payload={
                        "releaseRadiusM": EASY_RELEASE_RADIUS_M,
                        "dropHeightM": EASY_RELEASE_DROP_HEIGHT_M,
                        "gravityDrop": True,
                    },
                )
                continue
            joint = self.model.joint(f"{player_id}_payload_joint")
            qpos_start = joint.qposadr[0]
            dof_start = joint.dofadr[0]
            grasp = self.data.site(f"{player_id}_right_grasp_site").xpos
            self.data.qpos[qpos_start : qpos_start + 3] = (
                grasp + np.asarray([0.0, 0.0, -0.055])
            )
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
            runtime.had_payload_contact = True
            self._grasp_confirmed[player_id] = True

    def _distance_to_goal(self, player_id: str) -> float:
        goal = np.asarray(self.scene["players"][player_id]["goal"], dtype=float)
        return float(np.linalg.norm(self._base_position(player_id)[:2] - goal[:2]))

    def _easy_release_ready(self, player_id: str) -> bool:
        return self._distance_to_goal(player_id) <= EASY_RELEASE_RADIUS_M

    def _position_payload_for_easy_release(self, player_id: str) -> None:
        goal = np.asarray(self.scene["players"][player_id]["goal"], dtype=float)
        joint = self.model.joint(f"{player_id}_payload_joint")
        qpos_start = joint.qposadr[0]
        dof_start = joint.dofadr[0]
        self.data.qpos[qpos_start : qpos_start + 3] = [
            goal[0],
            goal[1],
            EASY_RELEASE_DROP_HEIGHT_M,
        ]
        self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[dof_start : dof_start + 6] = 0.0

    def _set_easy_attached(self, player_id: str, attached: bool) -> None:
        self._easy_attached[player_id] = attached
        geom_id = self.model.geom(f"{player_id}_payload_geom").id
        if attached:
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
        else:
            (
                self.model.geom_contype[geom_id],
                self.model.geom_conaffinity[geom_id],
            ) = self._payload_collision[player_id]

    def _sample_match_state(self) -> None:
        disqualified = {
            player_id: runtime.status.disqualified
            for player_id, runtime in self.players.items()
        }
        for runtime in self.players.values():
            runtime.status.disqualified = False
        super()._sample_match_state()
        for player_id, value in disqualified.items():
            self.players[player_id].status.disqualified = value
        if self.phase == MatchPhase.RUNNING and any(disqualified.values()):
            self.phase = MatchPhase.ABORTED
            self.winner = None
            self._event(
                "match_aborted",
                payload={
                    "reason": "player_disconnected",
                    "players": [
                        player_id
                        for player_id, value in disqualified.items()
                        if value
                    ],
                },
            )

    def _retry_grasp(self, player_id: str, rationale: str) -> None:
        runtime = self.players[player_id]
        self._set_easy_attached(player_id, False)
        self._grasp_attempts[player_id] += 1
        self._grasp_started_s[player_id] = None
        self._grasp_phase[player_id] = "approach"
        self._grasp_reference_z[player_id] = None
        self._grasp_confirmed[player_id] = False
        runtime.hand_close = 0.0
        runtime.status.current_skill = Skill.NAVIGATE_OBJECT
        runtime.status.rationale = rationale
        runtime.frame = runtime.frame.model_copy(
            update={
                "sequence": runtime.frame.sequence + 1,
                "skill": Skill.NAVIGATE_OBJECT,
            }
        )

    def _reset_available(self, player_id: str) -> bool:
        runtime = self.players[player_id]
        payload = self._payload_position(player_id)
        repeated_failures = (
            self._grasp_attempts[player_id]
            - self._reset_attempt_baseline[player_id]
            >= 3
        )
        below_reachable_height = payload[2] < max(
            0.55,
            float(self._payload_reset_pose[player_id][2]) - 0.12,
        )
        return bool(
            self._reset_count[player_id] < 2
            and not runtime.transported
            and not runtime.status.delivered
            and not runtime.status.carrying
            and (below_reachable_height or repeated_failures)
        )

    def request_payload_reset(self, player_id: str) -> bool:
        if player_id not in self.players:
            return False
        if self._reset_available(player_id):
            self._perform_payload_reset(player_id)
            return True
        runtime = self.players[player_id]
        failed = (
            self._grasp_attempts[player_id]
            - self._reset_attempt_baseline[player_id]
        )
        runtime.status.rationale = (
            "Referee reset denied: payload is reachable and "
            f"only {failed}/3 failed grasps are recorded."
        )
        return False

    def _perform_payload_reset(self, player_id: str) -> None:
        runtime = self.players[player_id]
        self._set_easy_attached(player_id, False)
        joint = self.model.joint(f"{player_id}_payload_joint")
        qpos_start = joint.qposadr[0]
        dof_start = joint.dofadr[0]
        self.data.qpos[qpos_start : qpos_start + 7] = self._payload_reset_pose[player_id]
        self.data.qvel[dof_start : dof_start + 6] = 0.0
        self.data.xfrc_applied[self.model.body(f"{player_id}_payload").id] = 0.0
        runtime.had_payload_contact = False
        runtime.transported = False
        runtime.stable_since_s = None
        runtime.status.carrying = False
        runtime.status.checkpoint_crossed = False
        runtime.status.delivered = False
        runtime.status.current_skill = Skill.WAIT
        runtime.status.rationale = (
            "Referee restored an unreachable payload; three-second penalty active."
        )
        runtime.hand_close = 0.0
        self._grasp_confirmed[player_id] = False
        self._grasp_phase[player_id] = "approach"
        self._grasp_reference_z[player_id] = None
        self._reset_count[player_id] += 1
        self._reset_attempt_baseline[player_id] = self._grasp_attempts[player_id]
        self._reset_lock_until_s[player_id] = self.simulation_time_s + 3.0
        self.mujoco.mj_forward(self.model, self.data)
        self._event(
            "payload_referee_reset",
            player_id=player_id,
            payload={
                "resetCount": self._reset_count[player_id],
                "penaltySeconds": 3.0,
                "reason": "payload_below_reachable_height",
            },
        )

    def _update_odometry(self) -> None:
        for player_id in ("p1", "p2"):
            truth = self._base_position(player_id)
            noise = self.rng.normal(0.0, 0.012, 3)
            noise[2] *= 0.25
            self._odom[player_id] = truth + noise

    def _record_trajectory(self) -> None:
        for player_id in ("p1", "p2"):
            observation = self.perception.snapshot(player_id)
            self.trajectory.append(
                {
                    "simulationTime": round(self.simulation_time_s, 4),
                    "playerId": player_id,
                    "robot": self._base_position(player_id).tolist(),
                    "odometry": self._odom[player_id].tolist(),
                    "payloadGroundTruth": self._payload_position(player_id).tolist(),
                    "payloadEstimate": observation.payload.position,
                    "payloadConfidence": observation.payload.confidence,
                    "skill": self.players[player_id].status.current_skill.value,
                    "graspPhase": self._grasp_phase[player_id],
                }
            )

    def _randomize_domain(self) -> dict[str, float]:
        floor_friction = float(self.rng.uniform(0.72, 1.08))
        self.model.geom("arena_floor").friction[:] = [floor_friction, 0.02, 0.002]
        payload_mass_scale = float(self.rng.uniform(0.78, 1.28))
        for player_id in ("p1", "p2"):
            body = self.model.body(f"{player_id}_payload")
            self.model.body_mass[body.id] *= payload_mass_scale
            self.model.body_inertia[body.id] *= payload_mass_scale
            geom = self.model.geom(f"{player_id}_payload_geom")
            geom.friction[:] = [
                float(self.rng.uniform(0.8, 1.45)),
                0.015,
                0.001,
            ]
            joint = self.model.joint(f"{player_id}_payload_joint")
            self.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 2] += self.rng.uniform(
                -0.07,
                0.07,
                2,
            )
        light_scale = float(self.rng.uniform(0.78, 1.18))
        self.model.light_diffuse[:] *= light_scale
        self.mujoco.mj_forward(self.model, self.data)
        return {
            "floorFriction": floor_friction,
            "payloadMassScale": payload_mass_scale,
            "lightScale": light_scale,
            "p1MotorStrength": float(self.rng.uniform(0.91, 1.0)),
            "p2MotorStrength": float(self.rng.uniform(0.91, 1.0)),
            "packetDropProbability": float(self.rng.uniform(0.01, 0.04)),
            "jointPositionNoiseRad": float(self.rng.uniform(0.0015, 0.004)),
            "jointVelocityNoiseRps": float(self.rng.uniform(0.008, 0.025)),
            "cameraPositionNoiseM": float(self.rng.uniform(0.018, 0.038)),
            "cameraMissProbability": float(self.rng.uniform(0.02, 0.08)),
        }

    def _payload_joint_qpos(self, player_id: str) -> np.ndarray:
        joint = self.model.joint(f"{player_id}_payload_joint")
        return self.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 7]

    def _perception_errors(self) -> dict[str, object]:
        players: dict[str, object] = {}
        for player_id in ("p1", "p2"):
            estimate = _position(self.perception.snapshot(player_id).payload.position)
            if estimate is None:
                players[player_id] = {"status": "no_current_estimate"}
                continue
            error = estimate - self._payload_position(player_id)
            players[player_id] = {
                "status": "evaluated",
                "positionErrorM": float(np.linalg.norm(error)),
                "errorVectorM": error.tolist(),
            }
        return players

    def _manipulation_evaluation(self) -> dict[str, object]:
        players: dict[str, object] = {}
        for player_id in ("p1", "p2"):
            heights = [
                float(sample["payloadGroundTruth"][2])
                for sample in self.trajectory
                if sample["playerId"] == player_id
                and isinstance(sample.get("payloadGroundTruth"), list)
            ]
            players[player_id] = {
                "minimumPayloadHeightM": min(heights) if heights else None,
                "graspAttempts": self._grasp_attempts[player_id],
                "contactConfirmed": self._grasp_confirmed[player_id],
            }
        return players


def _position(value: list[float] | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.shape == (3,) and np.all(np.isfinite(array)) else None


def _grasp_targets(
    base_position: np.ndarray,
    payload_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    approach = payload_position[:2] - base_position[:2]
    length = float(np.linalg.norm(approach))
    if length < 1e-6:
        approach = np.asarray([1.0, 0.0], dtype=float)
    else:
        approach = approach / length
    pregrasp = payload_position.copy()
    pregrasp[:2] -= 0.16 * approach
    pregrasp[2] += 0.11
    engage = payload_position.copy()
    engage[:2] -= 0.035 * approach
    engage[2] += 0.025
    lift = engage.copy()
    lift[2] += 0.16
    return pregrasp, engage, lift


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi
