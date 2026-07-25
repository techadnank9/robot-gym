from __future__ import annotations

import platform
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from demo_2.errors import Demo2ConfigurationError, Sdk2Error


@dataclass(frozen=True)
class ProbeResult:
    backend: str
    network_interface: str | None
    connected: bool
    fsm_id: int | None
    detail: str


class G1Transport(Protocol):
    is_hardware: bool
    backend_name: str

    def initialize(self) -> None: ...

    def probe(self) -> ProbeResult: ...

    def set_velocity(self, vx: float, vy: float, yaw_rate: float, duration_s: float) -> None: ...

    def wait(self, duration_s: float) -> None: ...

    def stop(self, duration_s: float) -> None: ...

    def execute_arm_action(self, action_id: int) -> None: ...

    def close(self) -> None: ...


@dataclass
class DryRunTransport:
    """Records the exact commands that would be sent, without loading SDK2."""

    fsm_id: int = 500
    is_hardware: bool = False
    backend_name: str = "dry-run"
    commands: list[dict[str, object]] = field(default_factory=list)
    initialized: bool = False

    def initialize(self) -> None:
        self.initialized = True
        self.commands.append({"command": "initialize"})

    def probe(self) -> ProbeResult:
        self._require_initialized()
        self.commands.append({"command": "probe"})
        return ProbeResult("dry-run", None, True, self.fsm_id, "simulated SDK2 connection")

    def set_velocity(self, vx: float, vy: float, yaw_rate: float, duration_s: float) -> None:
        self._require_initialized()
        self.commands.append(
            {
                "command": "set_velocity",
                "vx": vx,
                "vy": vy,
                "yaw_rate": yaw_rate,
                "duration_s": duration_s,
            }
        )

    def stop(self, duration_s: float) -> None:
        self._require_initialized()
        self.commands.append({"command": "stop", "duration_s": duration_s})

    def wait(self, duration_s: float) -> None:
        self._require_initialized()

    def execute_arm_action(self, action_id: int) -> None:
        self._require_initialized()
        self.commands.append({"command": "arm_action", "action_id": action_id})

    def close(self) -> None:
        if self.initialized:
            self.commands.append({"command": "close"})
        self.initialized = False

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise Sdk2Error("Transport has not been initialized.")


@dataclass(frozen=True)
class SdkBindings:
    channel_factory_initialize: Callable[[int, str], None]
    loco_client_class: type
    arm_client_class: type


def load_sdk_bindings() -> SdkBindings:
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    except ImportError as exc:
        raise Demo2ConfigurationError(
            "unitree_sdk2py is not installed. Follow demo_2/README.md on the Linux robot-control host."
        ) from exc
    return SdkBindings(ChannelFactoryInitialize, LocoClient, G1ArmActionClient)


class Sdk2Transport:
    """Small adapter around Unitree's official high-level G1 SDK2 clients."""

    is_hardware = True
    backend_name = "sdk2"

    def __init__(
        self,
        network_interface: str,
        timeout_s: float,
        *,
        bindings: SdkBindings | None = None,
        interface_names: Callable[[], set[str]] | None = None,
        system_name: Callable[[], str] | None = None,
    ) -> None:
        self.network_interface = network_interface
        self.timeout_s = timeout_s
        self._bindings = bindings
        self._interface_names = interface_names or _system_interface_names
        self._system_name = system_name or platform.system
        self._loco_client = None
        self._arm_client = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        if self._system_name() != "Linux":
            raise Demo2ConfigurationError("Real Unitree SDK2 control is supported only from a Linux host.")
        if not self.network_interface:
            raise Demo2ConfigurationError("--network-interface is required for the sdk2 backend.")
        if self.network_interface not in self._interface_names():
            raise Demo2ConfigurationError(
                f"Network interface '{self.network_interface}' does not exist on this host."
            )
        bindings = self._bindings or load_sdk_bindings()
        try:
            bindings.channel_factory_initialize(0, self.network_interface)
            self._loco_client = bindings.loco_client_class()
            self._loco_client.SetTimeout(self.timeout_s)
            self._loco_client.Init()
        except Exception as exc:
            raise Sdk2Error(f"Failed to initialize Unitree SDK2 on {self.network_interface}: {exc}") from exc
        self._bindings = bindings
        self._initialized = True

    def probe(self) -> ProbeResult:
        loco = self._require_loco()
        try:
            code, fsm_id = loco.GetFsmId()
        except Exception as exc:
            raise Sdk2Error(f"Failed to read G1 FSM state: {exc}") from exc
        self._require_success(code, "GetFsmId")
        if not isinstance(fsm_id, int):
            raise Sdk2Error(f"GetFsmId returned an invalid state: {fsm_id!r}")
        return ProbeResult(
            backend="sdk2",
            network_interface=self.network_interface,
            connected=True,
            fsm_id=fsm_id,
            detail="Unitree SDK2 LocoClient responded",
        )

    def set_velocity(self, vx: float, vy: float, yaw_rate: float, duration_s: float) -> None:
        loco = self._require_loco()
        try:
            code = loco.SetVelocity(vx, vy, yaw_rate, duration_s)
        except Exception as exc:
            raise Sdk2Error(f"SetVelocity failed: {exc}") from exc
        self._require_success(code, "SetVelocity")

    def stop(self, duration_s: float) -> None:
        loco = self._require_loco()
        try:
            code = loco.SetVelocity(0.0, 0.0, 0.0, duration_s)
        except Exception as exc:
            raise Sdk2Error(f"Emergency stop command failed: {exc}") from exc
        self._require_success(code, "SetVelocity(stop)")

    def wait(self, duration_s: float) -> None:
        self._require_loco()
        time.sleep(duration_s)

    def execute_arm_action(self, action_id: int) -> None:
        self._require_loco()
        if self._arm_client is None:
            assert self._bindings is not None
            try:
                self._arm_client = self._bindings.arm_client_class()
                self._arm_client.SetTimeout(self.timeout_s)
                self._arm_client.Init()
            except Exception as exc:
                raise Sdk2Error(f"Failed to initialize G1 arm-action service: {exc}") from exc
        try:
            code = self._arm_client.ExecuteAction(action_id)
        except Exception as exc:
            raise Sdk2Error(f"Arm action {action_id} failed: {exc}") from exc
        self._require_success(code, f"ExecuteAction({action_id})")

    def close(self) -> None:
        self._arm_client = None
        self._loco_client = None
        self._initialized = False

    def _require_loco(self):
        if not self._initialized or self._loco_client is None:
            raise Sdk2Error("Transport has not been initialized.")
        return self._loco_client

    @staticmethod
    def _require_success(code: object, operation: str) -> None:
        if code != 0:
            raise Sdk2Error(f"Unitree SDK2 {operation} returned error code {code!r}.")


def _system_interface_names() -> set[str]:
    return {name for _, name in socket.if_nameindex()}
