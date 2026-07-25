"""Long-running Mac worker that executes validated tasks received from Replit."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

from pathvla.mujoco_sorting_demo import parse_args, run
from pathvla.replit_worker import ReplitWorker


logger = logging.getLogger(__name__)


def args_from_task(task: dict[str, Any]):
    """Map the dashboard's small option allowlist onto the existing Mac CLI."""

    args = parse_args([])
    args.instruction = task["instruction"]
    options = task.get("options", {})

    def bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
        value = options.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
        return value

    def bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
        value = options.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be a number from {minimum} to {maximum}")
        return float(value)

    def boolean(name: str, default: bool) -> bool:
        value = options.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
        return value

    allowed = {
        "maxActions",
        "maxRejections",
        "thinkingBudget",
        "recordVideo",
        "headless",
        "lingerSeconds",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"Unsupported remote options: {', '.join(unknown)}")

    args.max_actions = bounded_int("maxActions", args.max_actions, 1, 50)
    args.max_rejections = bounded_int("maxRejections", args.max_rejections, 0, 10)
    args.thinking_budget = bounded_int("thinkingBudget", args.thinking_budget, 0, 8192)
    args.record_video = boolean("recordVideo", True)
    args.headless = boolean("headless", False)
    args.linger_seconds = bounded_float("lingerSeconds", 1.0, 0.0, 60.0)
    return args


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    worker = ReplitWorker.from_env()
    worker.start()
    print(f"Connecting to Replit at {worker.control_url} ...", flush=True)
    if not worker.wait_until_connected(timeout=30):
        print("Initial connection timed out; the worker will keep reconnecting in the background.", flush=True)
    else:
        print("Connected. Waiting for dashboard tasks. Press Ctrl+C to stop.", flush=True)

    try:
        while not worker.is_shutdown_requested():
            task = worker.get_pending_task()
            if task is None:
                time.sleep(0.25)
                continue
            task_id = task.get("taskId")
            run_id = task.get("runId") or str(uuid4())
            worker.send_telemetry_event(
                "task_received",
                run_id,
                {"task_id": task_id, "instruction": task["instruction"]},
            )
            try:
                args = args_from_task(task)
                result = run(args, worker=worker, remote_run_id=run_id, task_id=task_id)
                print(json.dumps({"run_id": run_id, "status": result["status"]}, indent=2), flush=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Remote task %s failed: %s", task_id or run_id, exc)
    except KeyboardInterrupt:
        print("Stopping Replit Mac worker.", flush=True)
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
