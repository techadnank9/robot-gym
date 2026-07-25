from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from demo_2.transport import DryRunTransport


@dataclass
class ChannelMetrics:
    requested: int = 0
    transmitted: int = 0
    dropped: int = 0
    delivered: int = 0
    watchdog_stops: int = 0
    clipped: int = 0


class SDKCompatibleCommandChannel:
    """A deterministic approximation of the SDK2 velocity command boundary."""

    def __init__(
        self,
        player_id: str,
        rng: np.random.Generator,
        *,
        rate_hz: float = 50.0,
        latency_range_s: tuple[float, float] = (0.04, 0.08),
        dropout_probability: float = 0.02,
        watchdog_s: float = 0.12,
        transport: Any | None = None,
    ) -> None:
        self.player_id = player_id
        self.rng = rng
        self.period_s = 1.0 / rate_hz
        self.latency_range_s = latency_range_s
        self.dropout_probability = dropout_probability
        self.watchdog_s = watchdog_s
        self.transport = transport or DryRunTransport()
        self.transport.initialize()
        self.limits = np.asarray([0.45, 0.22, 0.80], dtype=np.float32)
        self.slew_per_packet = np.asarray([0.08, 0.06, 0.16], dtype=np.float32)
        self.desired = np.zeros(3, dtype=np.float32)
        self.active = np.zeros(3, dtype=np.float32)
        self._last_sample = np.zeros(3, dtype=np.float32)
        self._next_sample_s: float | None = None
        self._last_delivery_s = 0.0
        self._request_fresh = False
        self._watchdog_active = False
        self._packets: deque[tuple[float, np.ndarray]] = deque()
        self.metrics = ChannelMetrics()
        self.trace: list[dict[str, object]] = []

    def request(self, vx: float, vy: float, yaw_rate: float) -> None:
        requested = np.asarray([vx, vy, yaw_rate], dtype=np.float32)
        if not np.all(np.isfinite(requested)):
            raise ValueError("SDK command must contain finite values")
        self.desired = requested
        self._request_fresh = True
        self.metrics.requested += 1

    def tick(self, simulation_time_s: float) -> np.ndarray:
        if self._next_sample_s is None:
            self._next_sample_s = simulation_time_s
        while simulation_time_s + 1e-9 >= self._next_sample_s:
            if self._request_fresh:
                self._sample_packet(self._next_sample_s)
                self._request_fresh = False
            self._next_sample_s += self.period_s
        while self._packets and self._packets[0][0] <= simulation_time_s + 1e-9:
            _, self.active = self._packets.popleft()
            self._last_delivery_s = simulation_time_s
            self.metrics.delivered += 1
        if simulation_time_s - self._last_delivery_s > self.watchdog_s:
            if np.any(self.active):
                self.metrics.watchdog_stops += 1
            if not self._watchdog_active:
                self.transport.stop(self.period_s)
                self._watchdog_active = True
            self.active = np.zeros(3, dtype=np.float32)
        else:
            self._watchdog_active = False
        return self.active.copy()

    def report(self) -> dict[str, object]:
        return {
            **asdict(self.metrics),
            "rateHz": round(1.0 / self.period_s, 2),
            "latencyRangeMs": [
                round(1000.0 * self.latency_range_s[0], 1),
                round(1000.0 * self.latency_range_s[1], 1),
            ],
            "dropoutProbability": self.dropout_probability,
            "watchdogMs": round(1000.0 * self.watchdog_s, 1),
            "transportBackend": self.transport.backend_name,
            "activeCommand": self.active.tolist(),
        }

    def _sample_packet(self, sample_time_s: float) -> None:
        clipped = np.clip(self.desired, -self.limits, self.limits)
        if not np.allclose(clipped, self.desired):
            self.metrics.clipped += 1
        delta = np.clip(clipped - self._last_sample, -self.slew_per_packet, self.slew_per_packet)
        command = np.round((self._last_sample + delta) / 0.005) * 0.005
        self._last_sample = command.astype(np.float32)
        if self.rng.random() < self.dropout_probability:
            self.metrics.dropped += 1
            self.trace.append(
                {
                    "sampleTime": round(sample_time_s, 4),
                    "status": "dropped",
                    "command": command.tolist(),
                }
            )
            return
        latency = float(self.rng.uniform(*self.latency_range_s))
        delivery_s = sample_time_s + latency
        self._packets.append((delivery_s, command.copy()))
        self.transport.set_velocity(
            float(command[0]),
            float(command[1]),
            float(command[2]),
            self.period_s,
        )
        self.metrics.transmitted += 1
        self.trace.append(
            {
                "sampleTime": round(sample_time_s, 4),
                "deliveryTime": round(delivery_s, 4),
                "status": "queued",
                "command": command.tolist(),
            }
        )


class ConstrainedLocomotion:
    """Wrap the official Unitree policy with SDK-rate and actuator uncertainty."""

    def __init__(
        self,
        base: Any,
        model: Any,
        data: Any,
        rng: np.random.Generator,
        *,
        motor_strength: dict[str, float],
        dropout_probability: float,
        joint_position_noise_rad: float,
        joint_velocity_noise_rps: float,
    ) -> None:
        self.base = base
        self.model = model
        self.data = data
        self.rng = rng
        self.motor_strength = motor_strength
        self.joint_position_noise_rad = joint_position_noise_rad
        self.joint_velocity_noise_rps = joint_velocity_noise_rps
        self.channels = {
            player_id: SDKCompatibleCommandChannel(
                player_id,
                rng,
                dropout_probability=dropout_probability,
            )
            for player_id in ("p1", "p2")
        }

    @property
    def decimation(self) -> int:
        return self.base.decimation

    def set_command(self, player_id: str, vx: float, vy: float, yaw_rate: float) -> None:
        self.channels[player_id].request(vx, vy, yaw_rate)

    def update(self, simulation_time_s: float) -> None:
        for player_id, channel in self.channels.items():
            self.base.set_command(player_id, *channel.tick(simulation_time_s))
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        try:
            for player_id in ("p1", "p2"):
                q_ids = self.base._leg_qpos[player_id]
                dq_ids = self.base._leg_dof[player_id]
                self.data.qpos[q_ids] += self.rng.normal(
                    0.0, self.joint_position_noise_rad, len(q_ids)
                )
                self.data.qvel[dq_ids] += self.rng.normal(
                    0.0, self.joint_velocity_noise_rps, len(dq_ids)
                )
            self.base.update(simulation_time_s)
        finally:
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel

    def apply_torques(self) -> None:
        self.base.apply_torques()
        for player_id, actuator_ids in self.base._leg_actuator.items():
            strength = self.motor_strength[player_id]
            noise = self.rng.normal(1.0, 0.008, len(actuator_ids))
            self.data.ctrl[actuator_ids] *= strength * noise

    def report(self) -> dict[str, object]:
        return {
            player_id: {
                **channel.report(),
                "motorStrengthScale": self.motor_strength[player_id],
                "jointPositionNoiseRad": self.joint_position_noise_rad,
                "jointVelocityNoiseRps": self.joint_velocity_noise_rps,
            }
            for player_id, channel in self.channels.items()
        }
