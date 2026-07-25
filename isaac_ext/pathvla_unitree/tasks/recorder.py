from __future__ import annotations

import json
import time
from pathlib import Path

import imageio.v2 as imageio

from pathvla.errors import RecordingError


class IsaacRecorder:
    def __init__(self, output_dir: Path, logger, require_video: bool, preferred_camera_path: str | None = None):
        self.output_dir = output_dir
        self.logger = logger
        self.require_video = require_video
        self.preferred_camera_path = preferred_camera_path
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.frame_paths: list[Path] = []

    def capture_frame(self, frame_index: int) -> None:
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        except ImportError as exc:
            if self.require_video:
                raise RecordingError("Viewport capture utilities are unavailable in this Isaac build.") from exc
            self.logger.warning("Viewport capture utilities unavailable; video recording disabled.")
            return

        viewport = get_active_viewport()
        if viewport is None:
            if self.require_video:
                raise RecordingError("No active viewport available for capture.")
            self.logger.warning("No active viewport available; video recording disabled.")
            return

        if self.preferred_camera_path:
            try:
                viewport.camera_path = self.preferred_camera_path
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to switch viewport to preferred camera %s: %s", self.preferred_camera_path, exc)

        frame_path = self.frames_dir / f"frame_{frame_index:05d}.png"
        capture_viewport_to_file(viewport, str(frame_path))
        self.frame_paths.append(frame_path)

    def finalize_video(self) -> tuple[str | None, str | None]:
        if not self.frame_paths:
            if self.require_video:
                raise RecordingError("No captured frames available to build rollout.mp4.")
            return None, None
        video_path = self.output_dir / "rollout.mp4"
        final_frame_path = self.output_dir / "final_frame.png"
        existing_frames = self._resolve_existing_frames()
        if not existing_frames:
            if self.require_video:
                raise RecordingError("No captured frame files were written before video finalization.")
            self.logger.warning("No captured frame files were present; skipping video finalization.")
            return None, None
        try:
            with imageio.get_writer(video_path, fps=10, format="FFMPEG") as writer:
                for frame_path in existing_frames:
                    writer.append_data(imageio.imread(frame_path))
            imageio.imwrite(final_frame_path, imageio.imread(existing_frames[-1]))
        except Exception as exc:  # noqa: BLE001
            if self.require_video:
                raise RecordingError(f"Failed to finalize video: {exc}") from exc
            self.logger.warning("Failed to finalize video: %s", exc)
            return None, None
        return str(video_path), str(final_frame_path)

    def write_trace(self, trace: list[list[float]]) -> str:
        trace_path = self.output_dir / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        return str(trace_path)

    def _resolve_existing_frames(self) -> list[Path]:
        deadline = time.time() + 5.0
        pending = list(self.frame_paths)
        while pending and time.time() < deadline:
            pending = [frame_path for frame_path in pending if not frame_path.exists()]
            if pending:
                time.sleep(0.1)
        existing_frames = [frame_path for frame_path in self.frame_paths if frame_path.exists()]
        missing_frames = [str(frame_path) for frame_path in self.frame_paths if not frame_path.exists()]
        if missing_frames:
            self.logger.warning("Skipping missing captured frames during video finalization: %s", missing_frames)
        return existing_frames
