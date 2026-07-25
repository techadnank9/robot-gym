from __future__ import annotations

import os
from typing import Any

from demo_3.live_server import LiveMatchServer


class VLGEWorldAdapter:
    """VLGE-facing boundary for embedded now and native world sync later."""

    def __init__(
        self,
        *,
        http_port: int = 8083,
        websocket_port: int = 8763,
    ) -> None:
        self.local = LiveMatchServer(http_port=http_port, websocket_port=websocket_port)
        self.remote: Any | None = None

    @property
    def url(self) -> str:
        return self.local.url

    def start(self) -> None:
        self.local.start()
        if os.getenv("REPLIT_ENABLED", "0") == "1":
            from pathvla.replit_worker import ReplitWorker

            self.remote = ReplitWorker.from_env()
            self.remote.start()

    def publish(self, state: dict[str, Any], frames: dict[str, bytes] | None = None) -> None:
        self.local.publish(state, frames)
        if self.remote is None:
            return
        match_id = str(state.get("matchId") or "")
        self.remote.send_telemetry_event("demo3_match_state", match_id, state)
        for camera, jpeg in (frames or {}).items():
            self.remote.send_frame(
                jpeg,
                run_id=match_id,
                camera_id=f"demo3_{camera}",
            )

    def get_command(self) -> dict[str, Any] | None:
        return self.local.get_command()

    def close(self) -> None:
        if self.remote is not None:
            self.remote.stop()
        self.remote = None
        self.local.close()
