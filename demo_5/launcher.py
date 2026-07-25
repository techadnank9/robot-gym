from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from typing import Any, Callable

from demo_5.vlge_adapter import _Demo5StaticHandler


MATCH_MODES = ("ai-vs-ai", "human-vs-ai", "human-vs-human")
_ADAPTERS = {"gemini-er", "scripted", "http"}


class MatchLauncher:
    """Persistent, shell-free supervisor for browser-selected Demo 5 matches."""

    def __init__(
        self,
        *,
        python_bin: str,
        project_root: Path,
        match_host: str,
        match_http_port: int,
        websocket_port: int,
        grasp_mode: str,
        render_profile: str,
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.python_bin = python_bin
        self.project_root = project_root
        self.match_host = match_host
        self.match_http_port = match_http_port
        self.websocket_port = websocket_port
        self.grasp_mode = grasp_mode
        self.render_profile = render_profile
        self._popen = popen
        self._lock = threading.Lock()
        self._process: Any | None = None
        self._generation = 0
        self._status = "idle"
        self._mode: str | None = None
        self._adapters: dict[str, str] = {}
        self._error: str | None = None

    def state(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is not None:
                self._status = "finished" if process.returncode == 0 else "error"
                if process.returncode != 0 and self._error is None:
                    self._error = f"Match process exited with code {process.returncode}."
            return {
                "status": self._status,
                "mode": self._mode,
                "generation": self._generation,
                "pid": getattr(process, "pid", None) if process is not None else None,
                "adapters": dict(self._adapters),
                "error": self._error,
                "websocketPort": self.websocket_port,
                "modes": list(MATCH_MODES),
            }

    def start_match(self, mode: str) -> dict[str, Any]:
        if mode not in MATCH_MODES:
            raise ValueError(f"Unsupported match mode: {mode}")
        with self._lock:
            self._stop_locked()
            command, adapters = self.command_for_mode(mode)
            self._generation += 1
            generation = self._generation
            self._mode = mode
            self._adapters = adapters
            self._error = None
            self._status = "starting"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(self.project_root)
            try:
                process = self._popen(
                    command,
                    cwd=self.project_root,
                    env=environment,
                )
            except Exception as exc:
                self._status = "error"
                self._error = str(exc)
                raise
            self._process = process
            self._status = "running"
            threading.Thread(
                target=self._monitor,
                args=(process, generation),
                daemon=True,
                name=f"demo5-match-{generation}",
            ).start()
            return self.state_unlocked()

    def command_for_mode(self, mode: str) -> tuple[list[str], dict[str, str]]:
        if mode not in MATCH_MODES:
            raise ValueError(f"Unsupported match mode: {mode}")
        command = [
            self.python_bin,
            "-m",
            "demo_5",
            "--headless",
            "--realtime",
            "--host",
            self.match_host,
            "--http-port",
            str(self.match_http_port),
            "--websocket-port",
            str(self.websocket_port),
            "--grasp-mode",
            self.grasp_mode,
            "--render-profile",
            self.render_profile,
            "--keyboard-ready-timeout",
            os.getenv("DEMO5_TWO_PLAYER_READY_TIMEOUT", "300"),
        ]
        adapters: dict[str, str] = {}
        if mode == "ai-vs-ai":
            p1_adapter = self._adapter_for("p1")
            p2_adapter = self._adapter_for("p2")
            adapters = {"p1": p1_adapter, "p2": p2_adapter}
            command.extend(
                [
                    "--p1",
                    "policy",
                    "--p1-adapter",
                    p1_adapter,
                    "--p2",
                    "policy",
                    "--p2-adapter",
                    p2_adapter,
                ]
            )
            self._append_endpoint(command, "p1", p1_adapter)
            self._append_endpoint(command, "p2", p2_adapter)
        elif mode == "human-vs-ai":
            p2_adapter = self._adapter_for("p2")
            adapters = {"p2": p2_adapter}
            command.extend(
                [
                    "--p1",
                    "human",
                    "--p1-input",
                    "keyboard",
                    "--p2",
                    "policy",
                    "--p2-adapter",
                    p2_adapter,
                ]
            )
            self._append_endpoint(command, "p2", p2_adapter)
        else:
            command.extend(
                [
                    "--p1",
                    "human",
                    "--p1-input",
                    "keyboard",
                    "--p2",
                    "human",
                    "--p2-input",
                    "keyboard",
                ]
            )
        return command, adapters

    def close(self) -> None:
        with self._lock:
            self._stop_locked()
            self._status = "stopped"

    def state_unlocked(self) -> dict[str, Any]:
        process = self._process
        return {
            "status": self._status,
            "mode": self._mode,
            "generation": self._generation,
            "pid": getattr(process, "pid", None) if process is not None else None,
            "adapters": dict(self._adapters),
            "error": self._error,
            "websocketPort": self.websocket_port,
            "modes": list(MATCH_MODES),
        }

    def _adapter_for(self, player_id: str) -> str:
        explicit = os.getenv(f"DEMO5_{player_id.upper()}_ADAPTER")
        if player_id == "p2":
            explicit = explicit or os.getenv("DEMO5_OPPONENT_ADAPTER")
        if explicit:
            if explicit not in _ADAPTERS:
                raise ValueError(
                    f"DEMO5_{player_id.upper()}_ADAPTER must be one of "
                    f"{', '.join(sorted(_ADAPTERS))}"
                )
            return explicit
        has_gemini_key = bool(
            os.getenv(f"DEMO3_{player_id.upper()}_GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        return "gemini-er" if has_gemini_key else "scripted"

    @staticmethod
    def _append_endpoint(command: list[str], player_id: str, adapter: str) -> None:
        if adapter != "http":
            return
        endpoint = os.getenv(f"DEMO5_{player_id.upper()}_ENDPOINT")
        if not endpoint:
            raise ValueError(
                f"DEMO5_{player_id.upper()}_ENDPOINT is required for the HTTP adapter"
            )
        command.extend([f"--{player_id}-endpoint", endpoint])

    def _stop_locked(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            self._process = None
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        self._process = None

    def _monitor(self, process: Any, generation: int) -> None:
        return_code = process.wait()
        with self._lock:
            if generation != self._generation or process is not self._process:
                return
            self._status = "finished" if return_code == 0 else "error"
            if return_code != 0:
                self._error = f"Match process exited with code {return_code}."


class LauncherHandler(_Demo5StaticHandler):
    launcher: MatchLauncher

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/launcher":
            self._send_json(self.launcher.state())
            return
        if path == "/health":
            self._send_json({"status": "ok", "launcher": self.launcher.state()})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/matches":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("Request body must be between 1 and 4096 bytes.")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("Request body must be a JSON object.")
            mode = str(value.get("mode") or "")
            state = self.launcher.start_match(mode)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"status": "error", "error": str(exc)}, status=400)
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json({"status": "error", "error": str(exc)}, status=500)
            return
        self._send_json(state, status=202)

    def _send_json(self, value: dict[str, Any], *, status: int = 200) -> None:
        content = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Demo 5 match launcher")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8085)
    parser.add_argument("--match-host", default="0.0.0.0")
    parser.add_argument("--match-http-port", type=int, default=8086)
    parser.add_argument("--websocket-port", type=int, default=8765)
    parser.add_argument("--grasp-mode", choices=("easy", "mechanical"), default="easy")
    parser.add_argument(
        "--render-profile",
        choices=("performance", "quality"),
        default="performance",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    launcher = MatchLauncher(
        python_bin=sys.executable,
        project_root=root,
        match_host=args.match_host,
        match_http_port=args.match_http_port,
        websocket_port=args.websocket_port,
        grasp_mode=args.grasp_mode,
        render_profile=args.render_profile,
    )
    handler = type(
        "Demo5LauncherHandler",
        (LauncherHandler,),
        {"launcher": launcher},
    )
    server = ThreadingHTTPServer((args.host, args.http_port), handler)
    if pod_id := os.getenv("RUNPOD_POD_ID"):
        print(f"Demo 5 match launcher: https://{pod_id}-{args.http_port}.proxy.runpod.net")
    else:
        print(f"Demo 5 match launcher: http://127.0.0.1:{args.http_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        launcher.close()


if __name__ == "__main__":
    main()
