"""Outbound Replit control-plane client for the native macOS robot worker."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from pathvla.errors import ConfigurationError


logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.0"
DEFAULT_QUEUE_SIZE = 200
MAX_PENDING_TASKS = 32
MAX_INSTRUCTION_LENGTH = 2_000
MAX_FRAME_BYTES = 1_500_000


def make_event(
    event_type: str,
    run_id: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the versioned event envelope expected by the Replit server."""

    if not event_type or len(event_type) > 80:
        raise ValueError("event_type must contain 1-80 characters")
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "eventId": str(uuid.uuid4()),
        "runId": run_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "eventType": event_type,
        "payload": payload,
    }


class _BoundedMessageQueue:
    """Thread-safe queue that evicts camera frames before control events."""

    def __init__(self, maxsize: int = DEFAULT_QUEUE_SIZE):
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._items: deque[dict[str, Any]] = deque()
        self._condition = threading.Condition()

    def put(self, message: dict[str, Any], *, is_frame: bool) -> bool:
        with self._condition:
            if len(self._items) >= self.maxsize:
                frame_index = next(
                    (index for index, item in enumerate(self._items) if item.get("type") == "camera_frame"),
                    None,
                )
                if frame_index is None:
                    return False
                del self._items[frame_index]
            self._items.append(message)
            self._condition.notify()
            return True

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)
            return self._items.popleft()

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)


def normalize_task_command(message: dict[str, Any]) -> dict[str, Any]:
    """Validate the safe subset of a dashboard task command."""

    if message.get("type") != "task_command":
        raise ValueError("Expected a task_command message")
    instruction = message.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task_command requires a non-empty instruction")
    instruction = instruction.strip()
    if len(instruction) > MAX_INSTRUCTION_LENGTH:
        raise ValueError(f"instruction exceeds {MAX_INSTRUCTION_LENGTH} characters")
    task_id = message.get("taskId")
    if task_id is not None and (not isinstance(task_id, str) or len(task_id) > 128):
        raise ValueError("taskId must be a string of at most 128 characters")
    run_id = message.get("runId")
    if run_id is not None and (not isinstance(run_id, str) or len(run_id) > 128):
        raise ValueError("runId must be a string of at most 128 characters")
    options = message.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    return {
        "type": "task_command",
        "command": str(message.get("command") or "run_task"),
        "taskId": task_id,
        "runId": run_id,
        "instruction": instruction,
        "options": options,
    }


@dataclass
class ReplitWorker:
    """Thread-safe, non-blocking WebSocket client used by the Mac simulator."""

    control_url: str = field(default_factory=lambda: os.environ["REPLIT_CONTROL_URL"])
    worker_token: str = field(default_factory=lambda: os.environ["REPLIT_WORKER_TOKEN"], repr=False)
    queue_size: int = DEFAULT_QUEUE_SIZE
    heartbeat_interval_s: float = 25.0

    _out_queue: _BoundedMessageQueue = field(init=False, repr=False)
    _pending_tasks: queue.Queue[dict[str, Any]] = field(
        default_factory=lambda: queue.Queue(maxsize=MAX_PENDING_TASKS),
        init=False,
        repr=False,
    )
    _shutdown_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _run_stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _connected: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _last_frame_hashes: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _active_run_id: str | None = field(default=None, init=False, repr=False)
    _active_run_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_configuration()
        self._out_queue = _BoundedMessageQueue(self.queue_size)

    @classmethod
    def from_env(cls) -> "ReplitWorker":
        missing = [name for name in ("REPLIT_CONTROL_URL", "REPLIT_WORKER_TOKEN") if not os.getenv(name)]
        if missing:
            raise ConfigurationError(f"Replit integration requires: {', '.join(missing)}")
        return cls()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        """Start the connection thread without blocking MuJoCo."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="replit-worker")
        self._thread.start()
        self.send_telemetry_event("worker_started", None, {"runtime": "native_macos_mujoco"})
        logger.info("Replit worker background thread started")

    def stop(self) -> None:
        """Shut down the connection thread; this is distinct from stopping a run."""

        self.send_telemetry_event("worker_shutdown", None, {})
        deadline = time.monotonic() + 0.75
        while self.connected and len(self._out_queue) and time.monotonic() < deadline:
            time.sleep(0.01)
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def begin_run(self, run_id: str) -> None:
        with self._active_run_lock:
            self._active_run_id = run_id
        self._run_stop_event.clear()

    def end_run(self, run_id: str) -> None:
        with self._active_run_lock:
            if self._active_run_id == run_id:
                self._active_run_id = None
        self._run_stop_event.clear()

    def request_run_stop(self) -> None:
        self._run_stop_event.set()

    def is_stop_requested(self) -> bool:
        return self._run_stop_event.is_set()

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_event.is_set()

    def wait_until_connected(self, timeout: float = 30.0) -> bool:
        return self._connected.wait(timeout=timeout)

    def send_telemetry_event(
        self,
        event_type: str,
        run_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        message = {"type": "telemetry_event", "event": make_event(event_type, run_id, payload)}
        self._enqueue(message, is_frame=False)

    def send_frame(
        self,
        jpeg_bytes: bytes,
        run_id: str | None = None,
        camera_id: str = "main_camera",
    ) -> None:
        if not jpeg_bytes:
            return
        if len(jpeg_bytes) > MAX_FRAME_BYTES:
            logger.warning("Dropping oversized Replit frame (%d bytes)", len(jpeg_bytes))
            return
        digest = hashlib.sha256(jpeg_bytes).hexdigest()
        if self._last_frame_hashes.get(camera_id) == digest:
            return
        self._last_frame_hashes[camera_id] = digest
        message = {
            "type": "camera_frame",
            "runId": run_id,
            "cameraId": camera_id,
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "mimeType": "image/jpeg",
            "frame": base64.b64encode(jpeg_bytes).decode("ascii"),
        }
        self._enqueue(message, is_frame=True)

    def get_pending_task(self) -> dict[str, Any] | None:
        try:
            return self._pending_tasks.get_nowait()
        except queue.Empty:
            return None

    def send_run_result(
        self,
        run_id: str,
        actions_completed: int,
        actions_rejected: int,
        report: dict[str, Any] | None = None,
    ) -> None:
        message = {
            "type": "run_result",
            "runId": run_id,
            "actionsCompleted": actions_completed,
            "actionsRejected": actions_rejected,
            "report": report or {},
        }
        self._enqueue(message, is_frame=False)

    def _validate_configuration(self) -> None:
        parsed = urlparse(self.control_url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ConfigurationError("REPLIT_CONTROL_URL must be a valid ws:// or wss:// URL")
        if parsed.scheme == "ws" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ConfigurationError("Use wss:// for non-local Replit connections")
        if not self.worker_token.strip():
            raise ConfigurationError("REPLIT_WORKER_TOKEN must not be empty")
        if self.heartbeat_interval_s <= 0:
            raise ConfigurationError("heartbeat_interval_s must be positive")

    def _enqueue(self, message: dict[str, Any], *, is_frame: bool) -> None:
        if not self._out_queue.put(message, is_frame=is_frame):
            logger.warning("Outgoing Replit queue full; dropped %s", message.get("type"))

    def _run_loop(self) -> None:
        asyncio.run(self._async_run_loop())

    async def _async_run_loop(self) -> None:
        backoff = 1.0
        while not self._shutdown_event.is_set():
            try:
                await self._connect_and_run()
                backoff = 1.0
            except Exception as exc:  # noqa: BLE001
                logger.warning("Replit connection error: %s; reconnecting in %.1fs", exc, backoff)
            finally:
                self._connected.clear()
            if not self._shutdown_event.is_set():
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _connect_and_run(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise ConfigurationError("Install requirements-mac.txt for Replit WebSocket support") from exc

        headers = {"Authorization": f"Bearer {self.worker_token}"}
        async with websockets.connect(
            self.control_url,
            additional_headers=headers,
            max_size=2_000_000,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            logger.info("Connected to Replit control server at %s", self.control_url)
            self._connected.set()
            await websocket.send(
                json.dumps(
                    {
                        "type": "telemetry_event",
                        "event": make_event("worker_connected", None, {"runtime": "native_macos_mujoco"}),
                    }
                )
            )
            tasks = [
                asyncio.create_task(self._heartbeat_loop(websocket)),
                asyncio.create_task(self._recv_loop(websocket)),
                asyncio.create_task(self._send_loop(websocket)),
                asyncio.create_task(self._shutdown_loop()),
            ]
            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                for task in done:
                    exception = task.exception()
                    if exception is not None:
                        raise exception
                if not self._shutdown_event.is_set():
                    raise ConnectionError("Replit WebSocket closed")
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _heartbeat_loop(self, websocket: Any) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.heartbeat_interval_s)
            await websocket.send(json.dumps({"type": "heartbeat", "protocolVersion": PROTOCOL_VERSION}))

    async def _shutdown_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)

    async def _recv_loop(self, websocket: Any) -> None:
        async for raw in websocket:
            try:
                message = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Ignored malformed JSON from Replit")
                continue
            if not isinstance(message, dict):
                continue
            message_type = message.get("type")
            if message_type == "task_command":
                try:
                    task = normalize_task_command(message)
                    self._pending_tasks.put_nowait(task)
                    logger.info("Received Replit task %s", task.get("taskId") or "<unassigned>")
                except (ValueError, queue.Full) as exc:
                    logger.warning("Rejected Replit task command: %s", exc)
            elif message_type == "stop_requested":
                requested_run = message.get("runId")
                with self._active_run_lock:
                    active_run = self._active_run_id
                if requested_run is None or requested_run == active_run:
                    logger.info("Dashboard requested cooperative stop for run %s", active_run)
                    self._run_stop_event.set()

    async def _send_loop(self, websocket: Any) -> None:
        while not self._shutdown_event.is_set():
            try:
                message = await asyncio.to_thread(self._out_queue.get, 1.0)
            except queue.Empty:
                continue
            await websocket.send(json.dumps(message, separators=(",", ":")))
