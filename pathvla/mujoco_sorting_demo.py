from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import imageio.v2 as imageio
from PIL import Image

from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import load_scene_config
from pathvla.errors import PathVLAError, PlanningError, RunStopped
from pathvla.logging_utils import configure_run_logger
from pathvla.mujoco_lab import MacMuJoCoSortingEnv, MacSortingController
from pathvla.sorting_agent import GeminiRoboticsERAgent

if TYPE_CHECKING:
    from pathvla.replit_worker import ReplitWorker


DEFAULT_TASK = "Sort every red item into the red bucket and every blue item into the blue bucket."


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_g1_mjcf() -> Path:
    return project_root() / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Native macOS MuJoCo + Gemini Robotics-ER G1 sorting demo.")
    parser.add_argument("--instruction", default=DEFAULT_TASK)
    parser.add_argument("--model", default=None)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--max-actions", type=int, default=24)
    parser.add_argument("--max-rejections", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--linger-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--g1-mjcf", default=str(default_g1_mjcf()))
    return parser.parse_args(argv)


def _new_run_dir(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = (Path(os.getenv("PATHVLA_OUTPUT_ROOT", "outputs")) / f"mac_sort_{stamp}_{uuid4().hex[:8]}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _video(frame_paths: list[Path], destination: Path) -> str | None:
    selected = [path for path in frame_paths if path.name.endswith("camera_0.png")]
    if not selected:
        return None
    with imageio.get_writer(destination, fps=2, format="FFMPEG") as writer:
        for path in selected:
            writer.append_data(imageio.imread(path))
    return str(destination)


def _telemetry(
    worker: ReplitWorker | None,
    event_type: str,
    run_id: str,
    payload: dict[str, Any],
) -> None:
    if worker is not None:
        worker.send_telemetry_event(event_type, run_id, payload)


def _jpeg_bytes(path: Path, quality: int = 72) -> bytes:
    with Image.open(path) as image:
        output = BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()


def _stream_frames(
    worker: ReplitWorker | None,
    run_id: str,
    paths: list[Path],
    camera_names: list[str],
    action_index: int,
) -> None:
    if worker is None:
        return
    for index, path in enumerate(paths):
        camera_id = camera_names[index] if index < len(camera_names) else f"camera_{index}"
        worker.send_frame(_jpeg_bytes(path), run_id=run_id, camera_id=camera_id)
    _telemetry(
        worker,
        "frame_available",
        run_id,
        {"action_index": action_index, "cameras": camera_names[: len(paths)]},
    )


def _check_remote_stop(worker: ReplitWorker | None) -> None:
    if worker is not None and worker.is_stop_requested():
        raise RunStopped("Dashboard stop acknowledged at a safe action boundary.")


def _replit_enabled() -> bool:
    return os.getenv("REPLIT_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def run(
    args,
    worker: ReplitWorker | None = None,
    remote_run_id: str | None = None,
    task_id: str | None = None,
) -> dict:
    if args.max_actions < 1:
        raise ValueError("--max-actions must be positive")
    run_id = remote_run_id or str(uuid4())
    if worker is not None:
        worker.begin_run(run_id)
    run_dir = _new_run_dir(args.output_dir)
    logger = configure_run_logger(f"mac_sort.{run_dir.name}", run_dir / "logs.txt")
    result = {
        "status": "running",
        "runtime": "native_macos_mujoco",
        "run_id": run_id,
        "task_id": task_id,
        "instruction": args.instruction,
        "output_dir": str(run_dir),
        "started_at": datetime.now(UTC).isoformat(),
    }
    _write_json(run_dir / "result.json", result)
    env = None
    controller = None
    records = []
    frames: list[Path] = []
    rejections = 0
    camera_names: list[str] = []
    _telemetry(
        worker,
        "run_started",
        run_id,
        {
            "task_id": task_id,
            "instruction": args.instruction,
            "runtime": "native_macos_mujoco",
            "gemini_model": args.model or os.getenv("GEMINI_ROBOTICS_MODEL", "gemini-robotics-er-1.6-preview"),
        },
    )
    try:
        _check_remote_stop(worker)
        scene_cfg = load_scene_config("sorting_lab")
        camera_names = [camera.name for camera in scene_cfg.scene.cameras]
        env = MacMuJoCoSortingEnv(Path(args.g1_mjcf).expanduser().resolve(), scene_cfg, run_dir, logger, args.headless)
        controller = MacSortingController(env, logger)
        if args.validate_only:
            captured = env.capture(0)
            frames.extend(captured)
            _stream_frames(worker, run_id, captured, camera_names, 0)
            result.update(
                {
                    "status": "completed",
                    "mode": "validate_only",
                    "model_summary": {
                        "nq": env.model.nq,
                        "nv": env.model.nv,
                        "nu": env.model.nu,
                        "bodies": env.model.nbody,
                    },
                    "initial_state": controller.world_state().model_dump(mode="json"),
                    "frames": [str(path) for path in frames],
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            )
            _write_json(run_dir / "result.json", result)
            _telemetry(worker, "task_completed", run_id, {"mode": "validate_only"})
            if worker is not None:
                worker.send_run_result(run_id, 0, 0, result)
            return result

        def wait_and_check(delay_s: float) -> None:
            env.sync(delay_s)
            _check_remote_stop(worker)

        def gemini_status(kind: str, delay_s: float) -> None:
            _telemetry(
                worker,
                "gemini_rate_limit_wait",
                run_id,
                {"kind": kind, "delay_seconds": round(delay_s, 3), "limit_rpm": 5},
            )

        agent = GeminiRoboticsERAgent(
            model=args.model,
            thinking_budget=args.thinking_budget,
            wait_callback=wait_and_check,
            status_callback=gemini_status,
        )
        result["model"] = agent.model
        completed = False
        for index in range(args.max_actions):
            _check_remote_stop(worker)
            current_frames = env.capture(index)
            frames.extend(current_frames)
            _stream_frames(worker, run_id, current_frames, camera_names, index)
            before = controller.world_state()
            env.set_agent_ui(
                "Observing scene — waiting for Gemini (5 RPM limit)",
                progress=controller.completed_actions,
            )
            env.sync()
            _telemetry(
                worker,
                "observing_scene",
                run_id,
                {"action_index": index, "held_object": before.held_object},
            )
            _telemetry(
                worker,
                "waiting_for_gemini",
                run_id,
                {"action_index": index, "model": agent.model, "limit_rpm": 5},
            )
            decision = agent.decide(args.instruction, before, current_frames)
            _check_remote_stop(worker)
            env.set_agent_ui(
                "Decision received",
                decision=decision,
                progress=controller.completed_actions,
            )
            env.sync()
            _telemetry(
                worker,
                "decision_received",
                run_id,
                {
                    "action_index": index,
                    "action": decision.action.value,
                    "target": decision.target,
                    "destination": decision.destination,
                    "rationale": decision.rationale,
                    "expected_outcome": decision.expected_outcome,
                },
            )
            print(
                f"Gemini: {decision.action.value}"
                f" target={decision.target or '-'}"
                f" destination={decision.destination or '-'}\n"
                f"Rationale: {decision.rationale}\n"
                f"Expected: {decision.expected_outcome}",
                flush=True,
            )
            record = {
                "index": index,
                "frames": [str(path) for path in current_frames],
                "world_state_before": before.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }
            try:
                _telemetry(
                    worker,
                    "action_started",
                    run_id,
                    {
                        "action_index": index,
                        "action": decision.action.value,
                        "target": decision.target,
                        "destination": decision.destination,
                    },
                )
                detail = controller.execute(decision)
                record["execution"] = {"success": True, "detail": detail}
                completed = decision.action.value == "finish"
                _telemetry(
                    worker,
                    "action_completed",
                    run_id,
                    {
                        "action_index": index,
                        "action": decision.action.value,
                        "detail": detail,
                        "completed_actions": len(controller.completed_actions),
                    },
                )
            except PlanningError as exc:
                rejections += 1
                controller.reject(decision, str(exc))
                env.set_agent_ui(
                    f"Action rejected — {exc}",
                    decision=decision,
                    progress=controller.completed_actions,
                )
                env.sync()
                record["execution"] = {"success": False, "detail": str(exc)}
                _telemetry(
                    worker,
                    "action_rejected",
                    run_id,
                    {
                        "action_index": index,
                        "action": decision.action.value,
                        "reason": str(exc),
                        "rejected_actions": rejections,
                    },
                )
                if rejections > args.max_rejections:
                    raise PlanningError(f"Gemini exceeded the rejection budget ({args.max_rejections}).") from exc
            record["world_state_after"] = controller.world_state().model_dump(mode="json")
            records.append(record)
            _write_json(run_dir / "gemini_action_plan.json", records)
            _check_remote_stop(worker)
            if completed:
                break
        if not completed:
            raise PlanningError(f"Gemini did not finish within {args.max_actions} actions.")
        final_state = controller.world_state()
        if not final_state.task_complete:
            raise PlanningError("Final geometric sorting verification failed.")
        env.set_agent_ui(
            "Task complete — geometric verification passed",
            progress=controller.completed_actions,
        )
        env.sync()
        final_frames = env.capture(len(records))
        frames.extend(final_frames)
        _stream_frames(worker, run_id, final_frames, camera_names, len(records))
        video_path = _video(frames, run_dir / "rollout.mp4") if args.record_video else None
        _write_json(run_dir / "execution_trace.json", [asdict(entry) for entry in controller.trace])
        result.update(
            {
                "status": "completed",
                "actions": len(records),
                "rejected_actions": rejections,
                "final_state": final_state.model_dump(mode="json"),
                "final_frames": [str(path) for path in final_frames],
                "video_path": video_path,
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(run_dir / "result.json", result)
        _telemetry(
            worker,
            "task_completed",
            run_id,
            {
                "actions_completed": len(controller.completed_actions),
                "actions_rejected": rejections,
                "geometric_verification": True,
            },
        )
        if worker is not None:
            worker.send_run_result(run_id, len(controller.completed_actions), rejections, result)
        if not args.headless and args.linger_seconds > 0:
            time.sleep(args.linger_seconds)
        return result
    except RunStopped as exc:
        if controller is not None:
            _write_json(run_dir / "execution_trace.json", [asdict(entry) for entry in controller.trace])
            final_state = controller.world_state().model_dump(mode="json")
            completed_actions = len(controller.completed_actions)
        else:
            final_state = None
            completed_actions = 0
        result.update(
            {
                "status": "stopped",
                "reason": str(exc),
                "actions": completed_actions,
                "rejected_actions": rejections,
                "final_state": final_state,
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(run_dir / "result.json", result)
        _telemetry(
            worker,
            "run_stopped",
            run_id,
            {"reason": str(exc), "actions_completed": completed_actions},
        )
        if worker is not None:
            worker.send_run_result(run_id, completed_actions, rejections, result)
        return result
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "failed", "error": str(exc), "finished_at": datetime.now(UTC).isoformat()})
        _write_json(run_dir / "result.json", result)
        _telemetry(worker, "task_failed", run_id, {"error": str(exc)})
        if worker is not None:
            completed_actions = len(controller.completed_actions) if controller is not None else 0
            worker.send_run_result(run_id, completed_actions, rejections, result)
        raise
    finally:
        if env is not None:
            env.close()
        if worker is not None:
            worker.end_run(run_id)


def main(argv: list[str] | None = None) -> int:
    worker = None
    try:
        if _replit_enabled():
            from pathvla.replit_worker import ReplitWorker

            worker = ReplitWorker.from_env()
            worker.start()
        result = run(parse_args(argv), worker=worker)
    except (PathVLAError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Mac demo failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if worker is not None:
            worker.stop()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
