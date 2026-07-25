from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from demo_2.config import Demo2Config
from demo_2.errors import HardwareSafetyError, UnsupportedCapabilityError
from demo_2.transport import G1Transport, ProbeResult


REAL_MOTION_ACK = "MOVE_REAL_UNITREE_G1"


@dataclass(frozen=True)
class MotionAuthorization:
    enable_real_motion: bool = False
    acknowledgement: str = ""
    operator_present: bool = False
    remote_estop_ready: bool = False
    area_clear: bool = False


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    yaw_rate: float
    duration_s: float


@dataclass(frozen=True)
class ExecutionReport:
    operation: str
    backend: str
    status: str
    probe: ProbeResult | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.probe is not None:
            payload["probe"] = asdict(self.probe)
        return payload


class RealG1Controller:
    """Fail-closed control facade for finite high-level SDK2 actions."""

    def __init__(
        self,
        transport: G1Transport,
        config: Demo2Config,
        authorization: MotionAuthorization | None = None,
    ) -> None:
        self.transport = transport
        self.config = config
        self.authorization = authorization or MotionAuthorization()
        self._initialized = False

    def initialize(self) -> None:
        self.transport.initialize()
        self._initialized = True

    def probe(self) -> ExecutionReport:
        self._require_initialized()
        result = self.transport.probe()
        return ExecutionReport("probe", result.backend, "ok", result, result.detail)

    def move(self, command: VelocityCommand) -> ExecutionReport:
        self._require_initialized()
        self._validate_authorization()
        self._validate_velocity(command)
        probe = self._require_ready_fsm()
        operation_error: BaseException | None = None
        try:
            self.transport.set_velocity(
                command.vx,
                command.vy,
                command.yaw_rate,
                command.duration_s,
            )
            self.transport.wait(command.duration_s)
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            self._stop_after_motion(operation_error)
        return ExecutionReport(
            operation="move",
            backend=probe.backend,
            status="completed",
            probe=probe,
            detail=(
                f"finite velocity command completed: vx={command.vx:.3f}, "
                f"vy={command.vy:.3f}, yaw_rate={command.yaw_rate:.3f}, "
                f"duration={command.duration_s:.3f}"
            ),
        )

    def execute_arm_action(self, name: str) -> ExecutionReport:
        self._require_initialized()
        self._validate_authorization()
        action_id = self.config.allowed_arm_actions.get(name)
        if action_id is None:
            allowed = ", ".join(sorted(self.config.allowed_arm_actions)) or "<none>"
            raise HardwareSafetyError(f"Arm action '{name}' is not allowlisted. Allowed: {allowed}.")
        probe = self._require_ready_fsm()
        self.transport.stop(self.config.limits.stop_duration_s)
        self.transport.wait(self.config.limits.settle_time_s)
        try:
            self.transport.execute_arm_action(action_id)
        except BaseException as exc:
            self._stop_after_motion(exc)
            raise
        self.transport.stop(self.config.limits.stop_duration_s)
        return ExecutionReport(
            operation="arm-action",
            backend=probe.backend,
            status="completed",
            probe=probe,
            detail=f"allowlisted Unitree arm action '{name}' ({action_id}) requested while stationary",
        )

    def stop(self) -> ExecutionReport:
        self._require_initialized()
        self.transport.stop(self.config.limits.stop_duration_s)
        return ExecutionReport(
            operation="stop",
            backend=self.transport.backend_name,
            status="completed",
            probe=None,
            detail="zero-velocity stop command sent",
        )

    def run_sorting(self) -> None:
        raise UnsupportedCapabilityError(
            "Autonomous real-world sorting is intentionally blocked: demo_2 has no calibrated "
            "hand/gripper driver, camera-to-base calibration, object pose estimator, force feedback, "
            "or whole-body manipulation controller."
        )

    def close(self) -> None:
        if self._initialized:
            self.transport.close()
        self._initialized = False

    def _require_ready_fsm(self) -> ProbeResult:
        probe = self.transport.probe()
        if probe.fsm_id not in self.config.allowed_fsm_ids:
            raise HardwareSafetyError(
                f"G1 FSM state {probe.fsm_id!r} is not motion-authorized; "
                f"allowed states are {self.config.allowed_fsm_ids}."
            )
        return probe

    def _validate_authorization(self) -> None:
        if not self.transport.is_hardware:
            return
        missing = []
        if not self.authorization.enable_real_motion:
            missing.append("--enable-real-motion")
        if self.authorization.acknowledgement != REAL_MOTION_ACK:
            missing.append(f"--acknowledge {REAL_MOTION_ACK}")
        if not self.authorization.operator_present:
            missing.append("--operator-present")
        if not self.authorization.remote_estop_ready:
            missing.append("--remote-estop-ready")
        if not self.authorization.area_clear:
            missing.append("--area-clear")
        if missing:
            raise HardwareSafetyError(
                "Real motion is locked. Satisfy every physical safety gate: " + ", ".join(missing)
            )

    def _validate_velocity(self, command: VelocityCommand) -> None:
        values = (command.vx, command.vy, command.yaw_rate, command.duration_s)
        if not all(math.isfinite(value) for value in values):
            raise HardwareSafetyError("Velocity commands must contain only finite numbers.")
        limits = self.config.limits
        if abs(command.vx) > limits.max_forward_mps:
            raise HardwareSafetyError(
                f"|vx|={abs(command.vx):.3f} exceeds {limits.max_forward_mps:.3f} m/s."
            )
        if abs(command.vy) > limits.max_lateral_mps:
            raise HardwareSafetyError(
                f"|vy|={abs(command.vy):.3f} exceeds {limits.max_lateral_mps:.3f} m/s."
            )
        if abs(command.yaw_rate) > limits.max_yaw_rate_rps:
            raise HardwareSafetyError(
                f"|yaw_rate|={abs(command.yaw_rate):.3f} exceeds "
                f"{limits.max_yaw_rate_rps:.3f} rad/s."
            )
        if command.duration_s <= 0.0 or command.duration_s > limits.max_command_duration_s:
            raise HardwareSafetyError(
                f"duration={command.duration_s:.3f} must be > 0 and <= "
                f"{limits.max_command_duration_s:.3f} seconds."
            )
        if command.vx == 0.0 and command.vy == 0.0 and command.yaw_rate == 0.0:
            raise HardwareSafetyError("Use the stop command instead of a zero velocity move.")

    def _stop_after_motion(self, operation_error: BaseException | None) -> None:
        try:
            self.transport.stop(self.config.limits.stop_duration_s)
        except BaseException as stop_error:
            if operation_error is None:
                raise
            if hasattr(operation_error, "add_note"):
                operation_error.add_note(f"Additionally, the SDK stop command failed: {stop_error}")

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise HardwareSafetyError("Controller has not been initialized.")
