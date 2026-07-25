from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import queue
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class _StaticHandler(BaseHTTPRequestHandler):
    web_root: Path

    def do_GET(self) -> None:  # noqa: N802
        relative = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        requested = (self.web_root / relative).resolve()
        if self.web_root.resolve() not in requested.parents and requested != self.web_root.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(requested.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("Demo 3 HTTP: " + format, *args)


class LiveMatchServer:
    """Small local HTTP/WebSocket server suitable for a VLGE iframe."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        http_port: int = 8083,
        websocket_port: int = 8763,
    ) -> None:
        self.host = host
        self.http_port = http_port
        self.websocket_port = websocket_port
        self.web_root = Path(__file__).with_name("web")
        self._http: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[Any] = set()
        self._latest_message: str | None = None
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=16)
        self._started = threading.Event()
        self._shutdown = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    def start(self) -> None:
        handler = type("Demo3StaticHandler", (_StaticHandler,), {"web_root": self.web_root})
        self._http = ThreadingHTTPServer((self.host, self.http_port), handler)
        self._http_thread = threading.Thread(
            target=self._http.serve_forever,
            daemon=True,
            name="demo3-http",
        )
        self._http_thread.start()
        self._ws_thread = threading.Thread(target=self._run_websocket, daemon=True, name="demo3-ws")
        self._ws_thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("Demo 3 WebSocket server did not start")

    def close(self) -> None:
        self._shutdown.set()
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=3)
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=3)

    def publish(self, state: dict[str, Any], frames: dict[str, bytes] | None = None) -> None:
        message = {
            "type": "match_state",
            "state": state,
            "frames": {
                camera: base64.b64encode(data).decode("ascii")
                for camera, data in (frames or {}).items()
            },
        }
        self._latest_message = json.dumps(message, separators=(",", ":"))
        if self._ws_loop is not None and self._ws_loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(self._latest_message), self._ws_loop)

    def get_command(self) -> dict[str, Any] | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def _run_websocket(self) -> None:
        asyncio.run(self._websocket_main())

    async def _websocket_main(self) -> None:
        import websockets

        self._ws_loop = asyncio.get_running_loop()
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.websocket_port,
            max_size=32_000,
        ):
            self._started.set()
            while not self._shutdown.is_set():
                await asyncio.sleep(0.1)

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
                if not isinstance(value, dict) or value.get("type") not in {
                    "start",
                    "stop",
                    "teleop",
                }:
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

    async def _broadcast(self, message: str) -> None:
        clients = list(self._clients)
        if not clients:
            return
        results = await asyncio.gather(
            *(client.send(message) for client in clients),
            return_exceptions=True,
        )
        for client, result in zip(clients, results, strict=True):
            if isinstance(result, Exception):
                self._clients.discard(client)
