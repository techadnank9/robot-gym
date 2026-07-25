from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass
class ObjectEstimate:
    position: list[float] | None
    confidence: float
    source: str
    captured_at_s: float
    delivered_at_s: float
    pixels: int = 0


@dataclass
class PerceptionSnapshot:
    payload: ObjectEstimate
    goal: ObjectEstimate


class DelayedCameraPerception:
    """Segmented RGB-D camera estimator with noise, misses and delivery delay."""

    def __init__(
        self,
        model: Any,
        data: Any,
        mujoco_module: Any,
        scene: dict[str, Any],
        rng: np.random.Generator,
        *,
        rate_hz: float = 5.0,
        latency_range_s: tuple[float, float] = (0.08, 0.14),
        position_noise_m: float = 0.025,
        miss_probability: float = 0.04,
        width: int = 160,
        height: int = 120,
    ) -> None:
        self.model = model
        self.data = data
        self.mujoco = mujoco_module
        self.scene = scene
        self.rng = rng
        self.period_s = 1.0 / rate_hz
        self.latency_range_s = latency_range_s
        self.position_noise_m = position_noise_m
        self.miss_probability = miss_probability
        self.width = width
        self.height = height
        self._renderer: Any | None = None
        self._next_capture_s = {"p1": 0.0, "p2": 0.0}
        self._pending: dict[str, deque[tuple[float, PerceptionSnapshot]]] = {
            "p1": deque(),
            "p2": deque(),
        }
        missing = ObjectEstimate(None, 0.0, "uninitialized", 0.0, 0.0)
        self.latest = {
            "p1": PerceptionSnapshot(missing, missing),
            "p2": PerceptionSnapshot(missing, missing),
        }
        self.capture_count = 0
        self.miss_count = 0
        self.render_failures = 0

    def update(self, simulation_time_s: float) -> None:
        for player_id in ("p1", "p2"):
            if simulation_time_s + 1e-9 >= self._next_capture_s[player_id]:
                self._capture(player_id, simulation_time_s)
                self._next_capture_s[player_id] += self.period_s
            pending = self._pending[player_id]
            while pending and pending[0][0] <= simulation_time_s + 1e-9:
                _, self.latest[player_id] = pending.popleft()

    def snapshot(self, player_id: str) -> PerceptionSnapshot:
        return self.latest[player_id]

    def report(self) -> dict[str, object]:
        return {
            "cameraRateHz": round(1.0 / self.period_s, 2),
            "latencyRangeMs": [
                round(1000.0 * self.latency_range_s[0], 1),
                round(1000.0 * self.latency_range_s[1], 1),
            ],
            "positionNoiseM": self.position_noise_m,
            "configuredMissProbability": self.miss_probability,
            "captures": self.capture_count,
            "misses": self.miss_count,
            "renderFailures": self.render_failures,
            "players": {
                player_id: {
                    "payload": asdict(snapshot.payload),
                    "goal": asdict(snapshot.goal),
                }
                for player_id, snapshot in self.latest.items()
            },
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None

    def _capture(self, player_id: str, captured_at_s: float) -> None:
        self.capture_count += 1
        latency = float(self.rng.uniform(*self.latency_range_s))
        delivered_at_s = captured_at_s + latency
        try:
            payload = self._estimate_geom(
                player_id,
                f"{player_id}_payload_geom",
                captured_at_s,
                delivered_at_s,
            )
            goal = self._estimate_geom(
                player_id,
                f"{player_id}_bucket_bottom",
                captured_at_s,
                delivered_at_s,
            )
        except Exception:  # noqa: BLE001
            self.render_failures += 1
            payload = ObjectEstimate(None, 0.0, "camera_error", captured_at_s, delivered_at_s)
            goal = ObjectEstimate(None, 0.0, "camera_error", captured_at_s, delivered_at_s)
        if goal.position is None:
            mapped = np.asarray(self.scene["players"][player_id]["goal"], dtype=float)
            mapped += self.rng.normal(0.0, self.position_noise_m * 1.5, 3)
            goal = ObjectEstimate(
                mapped.tolist(),
                0.55,
                "noisy_map",
                captured_at_s,
                delivered_at_s,
            )
        self._pending[player_id].append(
            (delivered_at_s, PerceptionSnapshot(payload=payload, goal=goal))
        )

    def _estimate_geom(
        self,
        player_id: str,
        geom_name: str,
        captured_at_s: float,
        delivered_at_s: float,
    ) -> ObjectEstimate:
        if self.rng.random() < self.miss_probability:
            self.miss_count += 1
            return ObjectEstimate(None, 0.0, "camera_miss", captured_at_s, delivered_at_s)
        if self._renderer is None:
            self._renderer = self.mujoco.Renderer(
                self.model,
                height=self.height,
                width=self.width,
            )
        camera_name = f"{player_id}_ego_camera"
        self._renderer.update_scene(self.data, camera=camera_name)
        self._renderer.enable_segmentation_rendering()
        segmentation = self._renderer.render().copy()
        self._renderer.disable_segmentation_rendering()
        geom_id = self.model.geom(geom_name).id
        geom_type = int(self.mujoco.mjtObj.mjOBJ_GEOM)
        mask = (segmentation[:, :, 0] == geom_id) & (segmentation[:, :, 1] == geom_type)
        pixels = int(np.count_nonzero(mask))
        if pixels < 4:
            self.miss_count += 1
            return ObjectEstimate(None, 0.0, "not_visible", captured_at_s, delivered_at_s)
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        rows, columns = np.nonzero(mask)
        distance = float(np.median(depth[mask]))
        u = float(np.mean(columns))
        v = float(np.mean(rows))
        camera_id = self.model.camera(camera_name).id
        fovy = math.radians(float(self.model.cam_fovy[camera_id]))
        focal = 0.5 * self.height / math.tan(0.5 * fovy)
        camera_point = np.asarray(
            [
                (u - (self.width - 1) / 2) * distance / focal,
                -((v - (self.height - 1) / 2) * distance / focal),
                -distance,
            ],
            dtype=float,
        )
        rotation = self.data.cam_xmat[camera_id].reshape(3, 3)
        camera_position = self.data.cam_xpos[camera_id].copy()
        camera_position += self.rng.normal(0.0, self.position_noise_m * 0.35, 3)
        world = camera_position + rotation @ camera_point
        world += self.rng.normal(0.0, self.position_noise_m, 3)
        confidence = float(np.clip(pixels / 90.0, 0.2, 0.95))
        return ObjectEstimate(
            world.tolist(),
            confidence,
            "segmented_rgbd",
            captured_at_s,
            delivered_at_s,
            pixels,
        )
