from __future__ import annotations

import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
import queue
import threading
from typing import Any

from demo_3.live_server import LiveMatchServer, _StaticHandler


_DEMO3_WEB = Path(__file__).resolve().parents[1] / "demo_3" / "web"
_DEMO5_WEB = Path(__file__).with_name("web")


class _Demo5StaticHandler(_StaticHandler):
    web_root = _DEMO5_WEB

    def do_GET(self) -> None:  # noqa: N802
        relative = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        self.web_root = _DEMO3_WEB if relative in {"app.js", "style.css"} else _DEMO5_WEB
        super().do_GET()


class Demo5LiveMatchServer(LiveMatchServer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._priority_commands: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)

    def start(self) -> None:
        self._http = ThreadingHTTPServer((self.host, self.http_port), _Demo5StaticHandler)
        self._http_thread = threading.Thread(
            target=self._http.serve_forever,
            daemon=True,
            name="demo5-http",
        )
        self._http_thread.start()
        self._ws_thread = threading.Thread(
            target=self._run_websocket,
            daemon=True,
            name="demo5-ws",
        )
        self._ws_thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("Demo 5 WebSocket server did not start")

    def get_command(self) -> dict[str, Any] | None:
        try:
            return self._priority_commands.get_nowait()
        except queue.Empty:
            return super().get_command()

    async def _handle_client(self, websocket: Any) -> None:
        self._clients.add(websocket)
        try:
            if self._latest_message is not None:
                await websocket.send(self._latest_message)
            async for raw in websocket:
                try:
                    value = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(value, dict):
                    continue
                if value.get("type") == "reset_payload":
                    try:
                        self._priority_commands.put_nowait(value)
                    except queue.Full:
                        try:
                            self._priority_commands.get_nowait()
                            self._priority_commands.put_nowait(value)
                        except (queue.Empty, queue.Full):
                            pass
                    continue
                if value.get("type") not in {"start", "stop", "teleop"}:
                    continue
                try:
                    self._commands.put_nowait(value)
                except queue.Full:
                    try:
                        self._commands.get_nowait()
                        self._commands.put_nowait(value)
                    except (queue.Empty, queue.Full):
                        pass
        finally:
            self._clients.discard(websocket)


class Demo5VLGEWorldAdapter:
    """Demo 5 bridge using the established VLGE-compatible match surface."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        http_port: int = 8085,
        websocket_port: int = 8765,
    ) -> None:
        self.local = Demo5LiveMatchServer(
            host=host,
            http_port=http_port,
            websocket_port=websocket_port,
        )
        self.websocket_port = websocket_port
        self.remote: Any | None = None

    @property
    def url(self) -> str:
        return f"{self.local.url}/?wsPort={self.websocket_port}"

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
        self.remote.send_telemetry_event("demo5_sim_to_real_state", match_id, state)
        for camera, jpeg in (frames or {}).items():
            self.remote.send_frame(
                jpeg,
                run_id=match_id,
                camera_id=f"demo5_{camera}",
            )

    def get_command(self) -> dict[str, Any] | None:
        return self.local.get_command()

    def close(self) -> None:
        if self.remote is not None:
            self.remote.stop()
        self.remote = None
        self.local.close()
