from __future__ import annotations

import math
import time
from pathlib import Path

from demo_2.errors import Demo2ConfigurationError, Sdk2Error
from demo_2.transport import ProbeResult


class MujocoTransport:
    """Visual G1 command twin for demo_2.

    The base and gait joints are advanced kinematically so this backend previews
    the bounded command envelope. It is not a learned or dynamically validated
    locomotion controller.
    """

    is_hardware = False
    backend_name = "mujoco"

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        headless: bool = False,
        linger_s: float = 3.0,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else (
            root / "assets" / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"
        )
        self.headless = headless
        self.linger_s = max(0.0, linger_s)
        self.mujoco = None
        self.model = None
        self.data = None
        self.viewer = None
        self._initialized = False
        self._velocity = (0.0, 0.0, 0.0)
        self._yaw = 0.0
        self._gait_phase = 0.0
        self._arm_action_active = False
        self._stand_qpos: dict[str, float] = {}

    def initialize(self) -> None:
        if self._initialized:
            return
        if not self.model_path.is_file():
            raise Demo2ConfigurationError(
                f"G1 MuJoCo scene is missing: {self.model_path}. Run 'make mac-setup' first."
            )
        try:
            import mujoco
        except ImportError as exc:
            raise Demo2ConfigurationError(
                "MuJoCo is not installed. Run 'make mac-setup' before using --backend mujoco."
            ) from exc
        try:
            self.mujoco = mujoco
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.data = mujoco.MjData(self.model)
            stand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            if stand_id < 0:
                raise Demo2ConfigurationError("The G1 MuJoCo model has no 'stand' keyframe.")
            mujoco.mj_resetDataKeyframe(self.model, self.data, stand_id)
            self.data.ctrl[:] = self.model.key_ctrl[stand_id]
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._stand_qpos = {
                name: float(self.data.joint(name).qpos[0])
                for name in _GAIT_JOINTS | _RIGHT_ARM_JOINTS
            }
            if not self.headless:
                import mujoco.viewer

                self.viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    show_left_ui=False,
                    show_right_ui=False,
                )
                self.viewer.cam.trackbodyid = self.model.body("pelvis").id
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
                self.viewer.cam.distance = 3.6
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -18
                self.viewer.sync()
        except Demo2ConfigurationError:
            raise
        except Exception as exc:
            raise Demo2ConfigurationError(f"Could not initialize demo_2 MuJoCo G1: {exc}") from exc
        self._initialized = True

    def probe(self) -> ProbeResult:
        self._require_initialized()
        return ProbeResult(
            backend="mujoco",
            network_interface=None,
            connected=True,
            fsm_id=500,
            detail="MuJoCo G1 command twin ready (kinematic preview)",
        )

    def set_velocity(self, vx: float, vy: float, yaw_rate: float, duration_s: float) -> None:
        self._require_initialized()
        self._arm_action_active = False
        self._velocity = (vx, vy, yaw_rate)

    def wait(self, duration_s: float) -> None:
        self._require_initialized()
        steps = max(1, math.ceil(duration_s * 60.0))
        step_s = duration_s / steps
        for _ in range(steps):
            if not self._viewer_running():
                break
            self._advance(step_s)
            self._sync(step_s)

    def stop(self, duration_s: float) -> None:
        self._require_initialized()
        self._velocity = (0.0, 0.0, 0.0)
        self._restore_gait_pose()
        self._forward_and_sync()

    def execute_arm_action(self, action_id: int) -> None:
        self._require_initialized()
        if action_id == 99:
            targets = {name: self._stand_qpos[name] for name in _RIGHT_ARM_JOINTS}
            self._arm_action_active = False
        elif action_id == 23:
            targets = {
                "right_shoulder_pitch_joint": -1.45,
                "right_shoulder_roll_joint": -0.20,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.35,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
            }
            self._arm_action_active = True
        else:
            raise Sdk2Error(f"MuJoCo preview does not implement Unitree arm action {action_id}.")
        self._animate_joint_targets(targets, duration_s=1.0)

    def close(self) -> None:
        if not self._initialized:
            return
        if self.viewer is not None and self.linger_s > 0.0:
            self.wait(self.linger_s)
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        self.data = None
        self.model = None
        self.mujoco = None
        self._initialized = False

    def base_pose(self) -> tuple[float, float, float]:
        self._require_initialized()
        assert self.data is not None
        qpos = self.data.joint("floating_base_joint").qpos
        return float(qpos[0]), float(qpos[1]), self._yaw

    def _advance(self, step_s: float) -> None:
        assert self.data is not None
        vx, vy, yaw_rate = self._velocity
        qpos = self.data.joint("floating_base_joint").qpos
        world_vx = math.cos(self._yaw) * vx - math.sin(self._yaw) * vy
        world_vy = math.sin(self._yaw) * vx + math.cos(self._yaw) * vy
        qpos[0] += world_vx * step_s
        qpos[1] += world_vy * step_s
        self._yaw += yaw_rate * step_s
        qpos[3:7] = [
            math.cos(self._yaw / 2.0),
            0.0,
            0.0,
            math.sin(self._yaw / 2.0),
        ]
        self.data.joint("floating_base_joint").qvel[:] = 0.0
        planar_speed = math.hypot(vx, vy)
        effort = min(1.0, planar_speed / 0.15 + abs(yaw_rate) / 0.20)
        self._gait_phase += step_s * (4.0 + 4.0 * effort)
        self._apply_gait(effort)
        assert self.mujoco is not None and self.model is not None
        self.mujoco.mj_forward(self.model, self.data)

    def _apply_gait(self, effort: float) -> None:
        phase = math.sin(self._gait_phase)
        left_lift = max(0.0, -phase)
        right_lift = max(0.0, phase)
        offsets = {
            "left_hip_pitch_joint": 0.16 * phase * effort,
            "right_hip_pitch_joint": -0.16 * phase * effort,
            "left_knee_joint": 0.12 * left_lift * effort,
            "right_knee_joint": 0.12 * right_lift * effort,
            "left_ankle_pitch_joint": -0.06 * left_lift * effort,
            "right_ankle_pitch_joint": -0.06 * right_lift * effort,
            "left_shoulder_pitch_joint": -0.10 * phase * effort,
            "right_shoulder_pitch_joint": 0.10 * phase * effort,
        }
        for name, offset in offsets.items():
            if self._arm_action_active and name in _RIGHT_ARM_JOINTS:
                continue
            self._set_joint(name, self._stand_qpos[name] + offset)

    def _restore_gait_pose(self) -> None:
        for name in _GAIT_JOINTS:
            if self._arm_action_active and name in _RIGHT_ARM_JOINTS:
                continue
            self._set_joint(name, self._stand_qpos[name])

    def _animate_joint_targets(self, targets: dict[str, float], duration_s: float) -> None:
        assert self.data is not None
        starts = {name: float(self.data.joint(name).qpos[0]) for name in targets}
        steps = max(1, math.ceil(duration_s * 60.0))
        for index in range(1, steps + 1):
            if not self._viewer_running():
                break
            alpha = 0.5 - 0.5 * math.cos(math.pi * index / steps)
            for name, target in targets.items():
                self._set_joint(name, starts[name] + (target - starts[name]) * alpha)
            self._forward_and_sync(duration_s / steps)

    def _set_joint(self, name: str, value: float) -> None:
        assert self.data is not None and self.model is not None
        joint = self.model.joint(name)
        clipped = float(min(max(value, joint.range[0]), joint.range[1]))
        self.data.joint(name).qpos[0] = clipped
        self.data.ctrl[self.model.actuator(name).id] = clipped

    def _forward_and_sync(self, sleep_s: float = 0.0) -> None:
        assert self.mujoco is not None and self.model is not None and self.data is not None
        self.mujoco.mj_forward(self.model, self.data)
        self._sync(sleep_s)

    def _sync(self, sleep_s: float) -> None:
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    def _viewer_running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise Sdk2Error("MuJoCo transport has not been initialized.")


_GAIT_JOINTS = {
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
}

_RIGHT_ARM_JOINTS = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}
