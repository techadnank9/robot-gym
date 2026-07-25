from __future__ import annotations

from dataclasses import dataclass, field
import pytest

from demo_2.config import Demo2Config, MotionLimits, load_config
from demo_2.controller import (
    REAL_MOTION_ACK,
    MotionAuthorization,
    RealG1Controller,
    VelocityCommand,
)
from demo_2.errors import HardwareSafetyError, Sdk2Error, UnsupportedCapabilityError
from demo_2.full_demo import parse_args as parse_full_demo_args
from demo_2.full_demo import run as run_full_demo
from demo_2.mujoco_transport import MujocoTransport
from demo_2.policy_sil import PolicySilTransport, SilFaultConfig, default_official_root
from demo_2.sorting_bridge import SdkSortingBridge
from demo_2.transport import DryRunTransport, Sdk2Transport, SdkBindings
from pathvla.sorting_agent import GeminiActionDecision


def authorized() -> MotionAuthorization:
    return MotionAuthorization(
        enable_real_motion=True,
        acknowledgement=REAL_MOTION_ACK,
        operator_present=True,
        remote_estop_ready=True,
        area_clear=True,
    )


def test_default_config_is_conservative_and_loadable() -> None:
    config = load_config()

    assert config.robot_model == "g1_29dof"
    assert config.allowed_fsm_ids == [500]
    assert config.limits.max_forward_mps == 0.15
    assert config.limits.max_command_duration_s == 0.5


def test_config_hard_ceiling_cannot_be_raised() -> None:
    with pytest.raises(ValueError):
        MotionLimits(max_forward_mps=0.26)
    with pytest.raises(ValueError):
        MotionLimits(max_command_duration_s=1.01)


def test_dry_run_move_probes_executes_and_stops() -> None:
    transport = DryRunTransport()
    controller = RealG1Controller(transport, Demo2Config())
    controller.initialize()

    report = controller.move(VelocityCommand(0.05, 0.0, 0.0, 0.25))

    assert report.status == "completed"
    assert [entry["command"] for entry in transport.commands] == [
        "initialize",
        "probe",
        "set_velocity",
        "stop",
    ]


def test_limit_violation_is_rejected_before_transport_motion() -> None:
    transport = DryRunTransport()
    controller = RealG1Controller(transport, Demo2Config())
    controller.initialize()

    with pytest.raises(HardwareSafetyError, match="exceeds"):
        controller.move(VelocityCommand(0.16, 0.0, 0.0, 0.25))

    assert not any(entry["command"] == "set_velocity" for entry in transport.commands)


def test_unexpected_fsm_is_rejected_before_transport_motion() -> None:
    transport = DryRunTransport(fsm_id=1)
    controller = RealG1Controller(transport, Demo2Config())
    controller.initialize()

    with pytest.raises(HardwareSafetyError, match="not motion-authorized"):
        controller.move(VelocityCommand(0.05, 0.0, 0.0, 0.25))

    assert not any(entry["command"] == "set_velocity" for entry in transport.commands)


@dataclass
class HardwareLikeTransport(DryRunTransport):
    is_hardware: bool = True


def test_hardware_motion_requires_every_authorization_gate() -> None:
    transport = HardwareLikeTransport()
    controller = RealG1Controller(transport, Demo2Config())
    controller.initialize()

    with pytest.raises(HardwareSafetyError, match="Real motion is locked"):
        controller.move(VelocityCommand(0.05, 0.0, 0.0, 0.25))

    assert not any(entry["command"] == "probe" for entry in transport.commands)


def test_move_attempts_stop_when_velocity_rpc_fails() -> None:
    @dataclass
    class FailingTransport(HardwareLikeTransport):
        def set_velocity(self, vx: float, vy: float, yaw_rate: float, duration_s: float) -> None:
            self.commands.append({"command": "set_velocity"})
            raise Sdk2Error("injected failure")

    transport = FailingTransport()
    controller = RealG1Controller(
        transport,
        Demo2Config(),
        authorization=authorized(),
    )
    controller.initialize()

    with pytest.raises(Sdk2Error, match="injected"):
        controller.move(VelocityCommand(0.05, 0.0, 0.0, 0.25))

    assert transport.commands[-1]["command"] == "stop"


def test_arm_action_is_allowlisted_and_base_is_stopped() -> None:
    transport = DryRunTransport()
    controller = RealG1Controller(transport, Demo2Config())
    controller.initialize()

    controller.execute_arm_action("right-hand-up")

    assert [entry["command"] for entry in transport.commands][-4:] == [
        "probe",
        "stop",
        "arm_action",
        "stop",
    ]
    with pytest.raises(HardwareSafetyError, match="not allowlisted"):
        controller.execute_arm_action("clap")


def test_sorting_is_explicitly_blocked() -> None:
    controller = RealG1Controller(DryRunTransport(), Demo2Config())

    with pytest.raises(UnsupportedCapabilityError, match="Autonomous real-world sorting"):
        controller.run_sorting()


def test_mujoco_command_twin_moves_g1_base_headlessly() -> None:
    pytest.importorskip("mujoco")
    transport = MujocoTransport(headless=True, linger_s=0.0)
    controller = RealG1Controller(transport, Demo2Config())
    controller.initialize()
    start_x, start_y, _ = transport.base_pose()

    report = controller.move(VelocityCommand(0.05, 0.0, 0.0, 0.25))
    end_x, end_y, _ = transport.base_pose()
    controller.close()

    assert report.backend == "mujoco"
    assert end_x == pytest.approx(start_x + 0.0125)
    assert end_y == pytest.approx(start_y)


def test_policy_sil_uses_official_policy_dynamics_and_real_deploy_parity() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    official_root = default_official_root()
    if not official_root.is_dir():
        pytest.skip("Run scripts/setup_demo_2_sil.sh to install the pinned official assets.")
    transport = PolicySilTransport(official_root)
    controller = RealG1Controller(transport, Demo2Config())
    controller.initialize()
    policy_states = []
    transport.set_state_callback(policy_states.append)

    controller.move(VelocityCommand(0.15, 0.0, 0.0, 0.5))
    evidence = transport.evidence_payload()
    controller.close()

    assert evidence["passed"] is True
    assert evidence["planar_displacement_m"] > 0.05
    assert evidence["min_base_height_m"] > 0.70
    assert evidence["max_tilt_deg"] < 10.0
    assert evidence["torque_saturation_samples"] == 0
    assert evidence["joint_limit_violations"] == 0
    assert all(evidence["deployment_parity"].values())
    assert evidence["real_deploy_lowcmd_topic"] == "rt/lowcmd"
    assert len(policy_states) > 1
    assert len(policy_states[-1].leg_joint_positions) == 12
    assert policy_states[-1].position[0] > policy_states[0].position[0]


def test_policy_sil_watchdog_zeros_command_after_total_packet_loss() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    official_root = default_official_root()
    if not official_root.is_dir():
        pytest.skip("Run scripts/setup_demo_2_sil.sh to install the pinned official assets.")
    transport = PolicySilTransport(
        official_root,
        faults=SilFaultConfig(packet_loss_rate=1.0, watchdog_timeout_s=0.10),
    )
    controller = RealG1Controller(transport, Demo2Config())
    controller.initialize()

    controller.move(VelocityCommand(0.15, 0.0, 0.0, 0.5))
    evidence = transport.evidence_payload()
    controller.close()

    assert evidence["passed"] is True
    assert evidence["command_frames_delivered"] == 0
    assert evidence["command_frames_dropped"] > 0
    assert evidence["watchdog_activations"] == 1
    assert evidence["command"] == [0.0, 0.0, 0.0]


def test_sdk_shadow_bridge_splits_navigation_into_bounded_commands() -> None:
    transport = DryRunTransport()
    config = Demo2Config()
    bridge = SdkSortingBridge(
        RealG1Controller(transport, config),
        config,
        mode="shadow",
    )
    bridge.initialize()

    bridge.move_segment([0.0, 0.0, 0.79], [0.30, 0.10, 0.79])

    moves = [entry for entry in transport.commands if entry["command"] == "set_velocity"]
    assert len(moves) > 1
    assert all(abs(float(entry["vx"])) <= config.limits.max_forward_mps for entry in moves)
    assert all(abs(float(entry["vy"])) <= config.limits.max_lateral_mps for entry in moves)
    assert all(
        float(entry["duration_s"]) <= config.limits.max_command_duration_s
        for entry in moves
    )


def test_live_bridge_blocks_unconfigured_real_manipulation() -> None:
    transport = HardwareLikeTransport()
    config = Demo2Config()
    bridge = SdkSortingBridge(
        RealG1Controller(transport, config, authorization=authorized()),
        config,
        mode="live",
        twin_aligned=True,
    )
    bridge.initialize()

    with pytest.raises(UnsupportedCapabilityError, match="calibrated driver"):
        bridge.manipulation("pick", "red_cube")


def test_full_demo_uses_sorting_lab_vla_loop_and_sdk_shadow(tmp_path) -> None:
    pytest.importorskip("mujoco")
    sequence = [
        ("navigate", "red_cube", None),
        ("pick", "red_cube", None),
        ("navigate", "red_bin", None),
        ("place", "red_cube", "red_bin"),
        ("navigate", "blue_cube", None),
        ("pick", "blue_cube", None),
        ("navigate", "blue_bin", None),
        ("place", "blue_cube", "blue_bin"),
        ("navigate", "red_can", None),
        ("pick", "red_can", None),
        ("navigate", "red_bin", None),
        ("place", "red_can", "red_bin"),
        ("navigate", "blue_can", None),
        ("pick", "blue_can", None),
        ("navigate", "blue_bin", None),
        ("place", "blue_can", "blue_bin"),
        ("finish", None, None),
    ]

    class ScriptedAgent:
        model = "scripted-test-agent"

        def __init__(self) -> None:
            self.index = 0

        def decide(self, instruction, state, camera_frames):
            action, target, destination = sequence[self.index]
            self.index += 1
            return GeminiActionDecision(
                action=action,
                target=target,
                destination=destination,
                rationale="Exercise the same validated action loop.",
                expected_outcome="The twin and SDK shadow traces advance together.",
            )

    args = parse_full_demo_args(
        [
            "--execution-backend",
            "sdk-shadow",
            "--headless",
            "--max-actions",
            "17",
            "--output-dir",
            str(tmp_path / "demo_2_full"),
        ]
    )
    result = run_full_demo(args, agent_factory=lambda _args, _wait: ScriptedAgent())

    assert result["status"] == "completed"
    assert result["actions"] == 17
    items = [obj for obj in result["final_state"]["objects"] if obj["kind"] == "item"]
    assert items and all(obj["status"] == "sorted" for obj in items)
    bridge_trace = result["sdk_bridge_trace"]
    assert any(entry["action"] == "navigate_segment" for entry in bridge_trace)
    assert any(entry["action"] == "pick" and entry["mode"] == "shadow" for entry in bridge_trace)


@dataclass
class FakeLocoClient:
    calls: list[tuple] = field(default_factory=list)

    def SetTimeout(self, timeout: float) -> None:
        self.calls.append(("timeout", timeout))

    def Init(self) -> None:
        self.calls.append(("init",))

    def GetFsmId(self):
        self.calls.append(("fsm",))
        return 0, 500

    def SetVelocity(self, vx: float, vy: float, yaw: float, duration: float):
        self.calls.append(("velocity", vx, vy, yaw, duration))
        return 0


@dataclass
class FakeArmClient:
    calls: list[tuple] = field(default_factory=list)

    def SetTimeout(self, timeout: float) -> None:
        self.calls.append(("timeout", timeout))

    def Init(self) -> None:
        self.calls.append(("init",))

    def ExecuteAction(self, action_id: int):
        self.calls.append(("action", action_id))
        return 0


def test_sdk2_adapter_uses_official_high_level_clients() -> None:
    initialized = []
    loco = FakeLocoClient()
    arm = FakeArmClient()
    bindings = SdkBindings(
        channel_factory_initialize=lambda domain, interface: initialized.append((domain, interface)),
        loco_client_class=lambda: loco,
        arm_client_class=lambda: arm,
    )
    transport = Sdk2Transport(
        "enp2s0",
        5.0,
        bindings=bindings,
        interface_names=lambda: {"enp2s0"},
        system_name=lambda: "Linux",
    )

    transport.initialize()
    assert transport.probe().fsm_id == 500
    transport.set_velocity(0.05, 0.0, 0.0, 0.25)
    transport.stop(0.2)
    transport.execute_arm_action(23)

    assert initialized == [(0, "enp2s0")]
    assert ("velocity", 0.05, 0.0, 0.0, 0.25) in loco.calls
    assert ("velocity", 0.0, 0.0, 0.0, 0.2) in loco.calls
    assert ("action", 23) in arm.calls


def test_sdk2_adapter_rejects_non_linux_before_loading_sdk() -> None:
    transport = Sdk2Transport(
        "enp2s0",
        5.0,
        interface_names=lambda: {"enp2s0"},
        system_name=lambda: "Darwin",
    )

    with pytest.raises(Exception, match="Linux"):
        transport.initialize()
