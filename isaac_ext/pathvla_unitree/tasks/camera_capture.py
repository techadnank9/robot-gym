from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pathvla.errors import RecordingError


def _resolve_camera_class():
    try:
        from isaacsim.sensors.camera import Camera

        return Camera
    except ImportError:
        try:
            from omni.isaac.sensor import Camera

            return Camera
        except ImportError as exc:
            raise RecordingError("Isaac camera sensor API is unavailable.") from exc


class IsaacMultiCameraCapture:
    def __init__(self, camera_prim_paths: list[str], step_fn, output_dir: Path, logger):
        if not camera_prim_paths:
            raise RecordingError("The sorting demo requires at least one scene camera.")
        camera_cls = _resolve_camera_class()
        self.step_fn = step_fn
        self.output_dir = output_dir / "agent_frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.cameras = []
        for index, prim_path in enumerate(camera_prim_paths):
            try:
                camera = camera_cls(
                    prim_path=prim_path,
                    name=f"gemini_camera_{index}",
                    resolution=(640, 480),
                )
            except TypeError:
                camera = camera_cls(prim_path=prim_path, resolution=(640, 480))
            camera.initialize()
            self.cameras.append(camera)
        for _ in range(4):
            self.step_fn()

    def capture(self, action_index: int) -> list[Path]:
        frames: list[Path] = []
        self.step_fn()
        for camera_index, camera in enumerate(self.cameras):
            rgba = camera.get_rgba()
            if rgba is None or np.asarray(rgba).size == 0:
                raise RecordingError(f"Camera {camera_index} returned no pixels.")
            array = np.asarray(rgba)
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
            frame_path = self.output_dir / f"step_{action_index:03d}_camera_{camera_index}.png"
            imageio.imwrite(frame_path, array)
            frames.append(frame_path)
        self.logger.info("Captured %d current camera frames for Gemini", len(frames))
        return frames
