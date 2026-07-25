from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from demo_2.errors import Demo2ConfigurationError, HardwareSafetyError, Sdk2Error
from demo_2.transport import ProbeResult


@dataclass(frozen=True)
class SilFaultConfig:
    command_latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    watchdog_timeout_s: float = 0.10
    seed: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.command_latency_ms <= 250.0:
            raise ValueError("command_latency_ms must be between 0 and 250")
        if not 0.0 <= self.packet_loss_rate <= 1.0:
            raise ValueError("packet_loss_rate must be between 0 and 1")
        if not 0.02 <= self.watchdog_timeout_s <= 0.5:
            raise ValueError("watchdog_timeout_s must be between 0.02 and 0.5")


@dataclass(frozen=True)
class PolicyStateSnapshot:
    simulation_time_s: float
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    yaw_rad: float
    leg_joint_positions: tuple[float, ...]


@dataclass
class SilEvidence:
    official_revision: str
    policy_sha256: str
    model_path: str
    real_deploy_config_sha256: str
    real_deploy_lowcmd_topic: str
    real_deploy_lowstate_topic: str
    deployment_parity: dict[str, bool]
    simulation_time_s: float = 0.0
    command_frames_sent: int = 0
    command_frames_delivered: int = 0
    command_frames_dropped: int = 0
    watchdog_activations: int = 0
    min_base_height_m: float = float("inf")
    max_tilt_deg: float = 0.0
    max_abs_torque_nm: float = 0.0
    torque_saturation_samples: int = 0
    torque_samples: int = 0
    joint_limit_violations: int = 0
    non_finite_samples: int = 0
    tracking_squared_error_sum: float = 0.0
    tracking_samples: int = 0
    fallen: bool = False
    fall_reason: str | None = None
    start_pose: list[float] | None = None
    final_pose: list[float] | None = None

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["tracking_rmse_mps"] = (
            math.sqrt(self.tracking_squared_error_sum / self.tracking_samples)
            if self.tracking_samples
            else None
        )
        value["torque_saturation_ratio"] = (
            self.torque_saturation_samples / self.torque_samples
            if self.torque_samples
            else None
        )
        if self.start_pose is not None and self.final_pose is not None:
            value["planar_displacement_m"] = math.dist(
                self.start_pose[:2],
                self.final_pose[:2],
            )
        else:
            value["planar_displacement_m"] = None
        value["passed"] = (
            not self.fallen
            and self.non_finite_samples == 0
            and self.joint_limit_violations == 0
            and self.min_base_height_m >= 0.55
            and self.max_tilt_deg <= 35.0
        )
        return value


class PolicySilTransport:
    """Dynamic MuJoCo transport driven by Unitree's official G1 RL policy.

    It mirrors the 50 Hz policy observations and 500 Hz simulated PD torque
    loop used by Unitree's published MuJoCo example. Provenance checks confirm
    that policy inputs, joint targets, gains, and scales match Unitree's pinned
    real SDK2 ``rt/lowcmd`` deployment example.
    """

    is_hardware = False
    backend_name = "policy-sil"

    def __init__(
        self,
        official_root: Path | str | None = None,
        *,
        faults: SilFaultConfig | None = None,
        viewer: bool = False,
        realtime: bool = False,
    ) -> None:
        self.official_root = Path(official_root) if official_root else default_official_root()
        self.faults = faults or SilFaultConfig()
        self.enable_viewer = viewer
        self.realtime = realtime
        self.mujoco = None
        self.torch = None
        self.model = None
        self.data = None
        self.policy = None
        self.viewer = None
        self._initialized = False
        self._rng = np.random.default_rng(self.faults.seed)
        self._desired_command = np.zeros(3, dtype=np.float32)
        self._active_command = np.zeros(3, dtype=np.float32)
        self._command_queue: deque[tuple[float, np.ndarray]] = deque()
        self._last_delivered_at = 0.0
        self._watchdog_active = False
        self._counter = 0
        self._action = np.zeros(12, dtype=np.float32)
        self._target_dof_pos = np.zeros(12, dtype=np.float32)
        self._obs = np.zeros(47, dtype=np.float32)
        self._config: dict[str, Any] = {}
        self._lock: dict[str, Any] = {}
        self._leg_joint_ids: list[int] = []
        self._leg_qpos_ids: list[int] = []
        self._leg_dof_ids: list[int] = []
        self._state_callback: Callable[[PolicyStateSnapshot], None] | None = None
        self.evidence: SilEvidence | None = None

    def initialize(self) -> None:
        if self._initialized:
            return
        self._lock = _load_lock()
        _validate_official_checkout(self.official_root, self._lock)
        config_path = self.official_root / self._lock["mujoco_config_relative_path"]
        self._config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        real_config_path = self.official_root / self._lock["real_config_relative_path"]
        real_config = yaml.safe_load(real_config_path.read_text(encoding="utf-8"))
        parity = _deployment_parity(self._config, real_config)
        mismatches = [name for name, matches in parity.items() if not matches]
        if mismatches:
            raise Demo2ConfigurationError(
                "Pinned Unitree MuJoCo and real-deploy configurations diverge for: "
                + ", ".join(mismatches)
            )
        try:
            import mujoco
            import torch
        except ImportError as exc:
            raise Demo2ConfigurationError(
                "Policy SIL requires MuJoCo and PyTorch. Install demo_2/requirements-sil.txt."
            ) from exc
        self.mujoco = mujoco
        self.torch = torch
        model_path = self.official_root / self._lock["mujoco_model_relative_path"]
        policy_path = self.official_root / self._lock["policy_relative_path"]
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(self._config["simulation_dt"])
        self.policy = torch.jit.load(str(policy_path))
        if hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()
        self._target_dof_pos = self.default_angles.copy()
        self._leg_joint_ids = [self.model.joint(name).id for name in LEG_JOINT_NAMES]
        self._leg_qpos_ids = [self.model.joint(name).qposadr[0] for name in LEG_JOINT_NAMES]
        self._leg_dof_ids = [self.model.joint(name).dofadr[0] for name in LEG_JOINT_NAMES]
        mujoco.mj_forward(self.model, self.data)
        start_pose = [float(value) for value in self.data.qpos[:3]]
        start_tilt_deg = math.degrees(tilt_angle(self.data.qpos[3:7]))
        self.evidence = SilEvidence(
            official_revision=self._lock["revision"],
            policy_sha256=self._lock["policy_sha256"],
            model_path=str(model_path),
            real_deploy_config_sha256=self._lock["real_config_sha256"],
            real_deploy_lowcmd_topic=str(real_config["lowcmd_topic"]),
            real_deploy_lowstate_topic=str(real_config["lowstate_topic"]),
            deployment_parity=parity,
            min_base_height_m=start_pose[2],
            max_tilt_deg=start_tilt_deg,
            start_pose=start_pose,
            final_pose=start_pose,
        )
        if self.enable_viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.trackbodyid = self.model.body("pelvis").id
            self.viewer.cam.distance = 3.0
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -18
            self.viewer.sync()
        self._initialized = True

    @property
    def default_angles(self) -> np.ndarray:
        return np.asarray(self._config["default_angles"], dtype=np.float32)

    @property
    def kps(self) -> np.ndarray:
        return np.asarray(self._config["kps"], dtype=np.float32)

    @property
    def kds(self) -> np.ndarray:
        return np.asarray(self._config["kds"], dtype=np.float32)

    def probe(self) -> ProbeResult:
        self._require_initialized()
        return ProbeResult(
            backend=self.backend_name,
            network_interface="simulated-command-channel",
            connected=True,
            fsm_id=500,
            detail=(
                "official G1 policy in MuJoCo; joint targets and controller constants "
                "verified against Unitree's real SDK2 lowcmd deployment"
            ),
        )

    def set_velocity(self, vx: float, vy: float, yaw_rate: float, duration_s: float) -> None:
        self._require_initialized()
        command = np.asarray([vx, vy, yaw_rate], dtype=np.float32)
        if not np.all(np.isfinite(command)):
            raise HardwareSafetyError("Policy SIL rejected a non-finite velocity command.")
        self._desired_command = command

    def wait(self, duration_s: float) -> None:
        self._require_initialized()
        self._run_for(duration_s)

    def stop(self, duration_s: float) -> None:
        self._require_initialized()
        self._desired_command = np.zeros(3, dtype=np.float32)
        self._run_for(duration_s)

    def execute_arm_action(self, action_id: int) -> None:
        raise Sdk2Error(
            "The official G1 locomotion policy controls 12 leg joints only; arm actions are out of scope."
        )

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        self.policy = None
        self.data = None
        self.model = None
        self._state_callback = None
        self._initialized = False

    def set_state_callback(
        self,
        callback: Callable[[PolicyStateSnapshot], None] | None,
    ) -> None:
        self._state_callback = callback
        if callback is not None and self._initialized:
            callback(self.state_snapshot())

    def state_snapshot(self) -> PolicyStateSnapshot:
        self._require_initialized()
        assert self.data is not None
        position = tuple(float(value) for value in self.data.qpos[:3])
        quaternion = tuple(float(value) for value in self.data.qpos[3:7])
        leg_positions = tuple(float(value) for value in self.data.qpos[self._leg_qpos_ids])
        return PolicyStateSnapshot(
            simulation_time_s=self.simulation_time,
            position=position,
            quaternion_wxyz=quaternion,
            yaw_rad=yaw_from_quaternion(self.data.qpos[3:7]),
            leg_joint_positions=leg_positions,
        )

    def evidence_payload(self) -> dict[str, Any]:
        self._require_initialized()
        assert self.evidence is not None
        return {
            **self.evidence.payload(),
            "faults": asdict(self.faults),
            "command": [float(value) for value in self._active_command],
        }

    def _run_for(self, duration_s: float) -> None:
        assert self.model is not None and self.data is not None and self.mujoco is not None
        assert self.evidence is not None
        steps = max(1, math.ceil(duration_s / self.model.opt.timestep))
        decimation = int(self._config["control_decimation"])
        for _ in range(steps):
            if self.viewer is not None and not self.viewer.is_running():
                break
            step_started = time.perf_counter()
            self._apply_pd_torque()
            self.mujoco.mj_step(self.model, self.data)
            self._counter += 1
            if self._counter % decimation == 0:
                self._publish_command_frame()
                self._deliver_command_frames()
                self._apply_watchdog()
                self._run_policy()
            self._sample_evidence()
            if self._state_callback is not None and self._counter % decimation == 0:
                self._state_callback(self.state_snapshot())
            if self.evidence.fallen:
                raise HardwareSafetyError(f"Policy SIL fall guard triggered: {self.evidence.fall_reason}")
            if self.viewer is not None and self._counter % decimation == 0:
                self.viewer.sync()
            if self.realtime:
                remaining = self.model.opt.timestep - (time.perf_counter() - step_started)
                if remaining > 0.0:
                    time.sleep(remaining)

    def _publish_command_frame(self) -> None:
        assert self.evidence is not None
        self.evidence.command_frames_sent += 1
        if self._rng.random() < self.faults.packet_loss_rate:
            self.evidence.command_frames_dropped += 1
            return
        arrival = self.simulation_time + self.faults.command_latency_ms / 1000.0
        self._command_queue.append((arrival, self._desired_command.copy()))

    def _deliver_command_frames(self) -> None:
        assert self.evidence is not None
        while self._command_queue and self._command_queue[0][0] <= self.simulation_time + 1e-12:
            _, command = self._command_queue.popleft()
            self._active_command = command
            self._last_delivered_at = self.simulation_time
            self._watchdog_active = False
            self.evidence.command_frames_delivered += 1

    def _apply_watchdog(self) -> None:
        assert self.evidence is not None
        if self.simulation_time - self._last_delivered_at <= self.faults.watchdog_timeout_s:
            return
        if not self._watchdog_active:
            self.evidence.watchdog_activations += 1
            self._watchdog_active = True
        self._active_command[:] = 0.0

    def _apply_pd_torque(self) -> None:
        assert self.data is not None and self.evidence is not None
        q = self.data.qpos[self._leg_qpos_ids]
        dq = self.data.qvel[self._leg_dof_ids]
        torque = (self._target_dof_pos - q) * self.kps - dq * self.kds
        self.evidence.max_abs_torque_nm = max(
            self.evidence.max_abs_torque_nm,
            float(np.max(np.abs(torque))),
        )
        saturated = np.abs(torque) > TORQUE_LIMITS
        self.evidence.torque_saturation_samples += int(np.count_nonzero(saturated))
        self.evidence.torque_samples += int(torque.size)
        self.data.ctrl[:] = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

    def _run_policy(self) -> None:
        assert self.data is not None and self.policy is not None and self.torch is not None
        q = self.data.qpos[self._leg_qpos_ids]
        dq = self.data.qvel[self._leg_dof_ids]
        quat = self.data.qpos[3:7]
        gravity = gravity_orientation(quat)
        period = 0.8
        phase = self.simulation_time % period / period
        self._obs[:3] = self.data.qvel[3:6] * float(self._config["ang_vel_scale"])
        self._obs[3:6] = gravity
        self._obs[6:9] = self._active_command * np.asarray(
            self._config["cmd_scale"],
            dtype=np.float32,
        )
        self._obs[9:21] = (q - self.default_angles) * float(self._config["dof_pos_scale"])
        self._obs[21:33] = dq * float(self._config["dof_vel_scale"])
        self._obs[33:45] = self._action
        self._obs[45:47] = [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]
        with self.torch.no_grad():
            output = self.policy(self.torch.from_numpy(self._obs).unsqueeze(0))
        self._action = output.detach().cpu().numpy().squeeze().astype(np.float32)
        self._target_dof_pos = (
            self._action * float(self._config["action_scale"]) + self.default_angles
        )

    def _sample_evidence(self) -> None:
        assert self.model is not None and self.data is not None and self.evidence is not None
        state = np.concatenate((self.data.qpos, self.data.qvel, self.data.ctrl))
        if not np.all(np.isfinite(state)):
            self.evidence.non_finite_samples += 1
            self.evidence.fallen = True
            self.evidence.fall_reason = "non-finite physics state"
            return
        height = float(self.data.qpos[2])
        tilt_deg = math.degrees(tilt_angle(self.data.qpos[3:7]))
        self.evidence.min_base_height_m = min(self.evidence.min_base_height_m, height)
        self.evidence.max_tilt_deg = max(self.evidence.max_tilt_deg, tilt_deg)
        q = self.data.qpos[self._leg_qpos_ids]
        for joint_id, value in zip(self._leg_joint_ids, q, strict=True):
            low, high = self.model.jnt_range[joint_id]
            if value < low - 1e-4 or value > high + 1e-4:
                self.evidence.joint_limit_violations += 1
        velocity_error = self.data.qvel[:2] - self._active_command[:2]
        self.evidence.tracking_squared_error_sum += float(np.dot(velocity_error, velocity_error))
        self.evidence.tracking_samples += 2
        self.evidence.simulation_time_s = self.simulation_time
        self.evidence.final_pose = [float(value) for value in self.data.qpos[:3]]
        if height < 0.55:
            self.evidence.fallen = True
            self.evidence.fall_reason = f"base height {height:.3f} m below 0.55 m"
        elif tilt_deg > 35.0:
            self.evidence.fallen = True
            self.evidence.fall_reason = f"tilt {tilt_deg:.1f} deg above 35 deg"

    @property
    def simulation_time(self) -> float:
        assert self.model is not None
        return self._counter * self.model.opt.timestep

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise Sdk2Error("Policy SIL transport has not been initialized.")


def gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.asarray(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def tilt_angle(quaternion: np.ndarray) -> float:
    gravity = gravity_orientation(quaternion)
    return math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    qw, qx, qy, qz = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def default_official_root() -> Path:
    return Path(__file__).with_name("vendor") / "unitree_rl_gym"


def _load_lock() -> dict[str, Any]:
    path = Path(__file__).with_name("official_rl_lock.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_official_checkout(root: Path, lock: dict[str, Any]) -> None:
    if not root.is_dir():
        raise Demo2ConfigurationError(
            f"Official Unitree RL checkout is missing: {root}. "
            "Run scripts/setup_demo_2_sil.sh."
        )
    for path_key, hash_key in (
        ("policy_relative_path", "policy_sha256"),
        ("mujoco_config_relative_path", "mujoco_config_sha256"),
        ("real_config_relative_path", "real_config_sha256"),
    ):
        path = root / lock[path_key]
        if not path.is_file():
            raise Demo2ConfigurationError(f"Official Unitree file is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != lock[hash_key]:
            raise Demo2ConfigurationError(
                f"Official Unitree file hash mismatch for {path}; expected {lock[hash_key]}, got {digest}."
            )
    model_path = root / lock["mujoco_model_relative_path"]
    if not model_path.is_file():
        raise Demo2ConfigurationError(f"Official Unitree MuJoCo model is missing: {model_path}")


def _deployment_parity(
    mujoco_config: dict[str, Any],
    real_config: dict[str, Any],
) -> dict[str, bool]:
    parity: dict[str, bool] = {}
    for key in (
        "policy_path",
        "kps",
        "kds",
        "default_angles",
        "ang_vel_scale",
        "dof_pos_scale",
        "dof_vel_scale",
        "action_scale",
        "cmd_scale",
        "num_actions",
        "num_obs",
    ):
        parity[key] = mujoco_config.get(key) == real_config.get(key)
    parity["control_period"] = math.isclose(
        float(mujoco_config["simulation_dt"]) * int(mujoco_config["control_decimation"]),
        float(real_config["control_dt"]),
        abs_tol=1e-12,
    )
    parity["leg_motor_mapping"] = real_config.get("leg_joint2motor_idx") == list(range(12))
    parity["lowcmd_topic"] = real_config.get("lowcmd_topic") == "rt/lowcmd"
    parity["lowstate_topic"] = real_config.get("lowstate_topic") == "rt/lowstate"
    return parity


LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

TORQUE_LIMITS = np.asarray(
    [88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * 2,
    dtype=np.float32,
)
