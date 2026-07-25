from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import imageio.v2 as imageio

from pathvla.errors import PathVLAError, PlanningError
from pathvla.logging_utils import configure_run_logger
from pathvla.sorting_agent import GeminiRoboticsERAgent

from isaac_ext.pathvla_unitree.tasks.room_nav_env import create_stage, launch_simulation_app
from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import (
    load_livestream_config,
    load_robot_config,
    load_scene_config,
)


DEFAULT_TASK = "Sort every red item into the red bucket and every blue item into the blue bucket."


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run closed-loop Gemini Robotics-ER sorting with the official Unitree G1 USD."
    )
    parser.add_argument("--instruction", default=DEFAULT_TASK)
    parser.add_argument("--live", choices=["webrtc", "none"], default="webrtc")
    parser.add_argument("--model", default=None)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--max-actions", type=int, default=24)
    parser.add_argument("--max-rejections", type=int, default=3)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def _new_run_dir(explicit: str | None) -> Path:
    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
    else:
        root = Path(os.getenv("PATHVLA_OUTPUT_ROOT", "outputs"))
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (root / f"gemini_sort_{stamp}_{uuid4().hex[:8]}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_video(frame_paths: list[Path], output_path: Path) -> str | None:
    main_camera_frames = [path for path in frame_paths if path.name.endswith("camera_0.png")]
    if not main_camera_frames:
        return None
    with imageio.get_writer(output_path, fps=2, format="FFMPEG") as writer:
        for frame_path in main_camera_frames:
            writer.append_data(imageio.imread(frame_path))
    return str(output_path)


def run(args) -> dict:
    if args.max_actions < 1:
        raise ValueError("--max-actions must be positive")
    if args.max_rejections < 0:
        raise ValueError("--max-rejections must not be negative")

    run_dir = _new_run_dir(args.output_dir)
    logger = configure_run_logger(f"gemini_sort.{run_dir.name}", run_dir / "logs.txt")
    result = {
        "status": "running",
        "instruction": args.instruction,
        "model": args.model,
        "output_dir": str(run_dir),
        "started_at": datetime.now(UTC).isoformat(),
    }
    _write_json(run_dir / "result.json", result)
    app = None
    action_records = []
    all_frames: list[Path] = []
    try:
        agent = GeminiRoboticsERAgent(model=args.model, thinking_budget=args.thinking_budget)
        result["model"] = agent.model
        app = launch_simulation_app(headless=args.live == "none")

        from isaac_ext.pathvla_unitree.tasks.camera_capture import IsaacMultiCameraCapture
        from isaac_ext.pathvla_unitree.tasks.livestream import configure_livestream
        from isaac_ext.pathvla_unitree.tasks.robot_loader import load_g1_sorting_robot
        from isaac_ext.pathvla_unitree.tasks.scene_builder import build_scene
        from isaac_ext.pathvla_unitree.tasks.sorting_skills import SortingSkillController

        stage = create_stage()
        scene_cfg = load_scene_config("sorting_lab")
        semantic_scene = build_scene(stage, scene_cfg, logger)
        robot = load_g1_sorting_robot(
            stage,
            load_robot_config(),
            logger,
            spawn_translation=scene_cfg.scene.robot_spawn,
        )
        for _ in range(12):
            app.update()
        live_info = configure_livestream(args.live, load_livestream_config(), logger)
        controller = SortingSkillController(stage, robot, semantic_scene, app.update, logger)
        cameras = IsaacMultiCameraCapture(
            semantic_scene.camera_prim_paths,
            app.update,
            run_dir,
            logger,
        )

        rejection_count = 0
        completed = False
        for action_index in range(args.max_actions):
            frames = cameras.capture(action_index)
            all_frames.extend(frames)
            before = controller.world_state()
            decision = agent.decide(args.instruction, before, frames)
            record = {
                "index": action_index,
                "observation_frames": [str(path) for path in frames],
                "world_state_before": before.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }
            try:
                detail = controller.execute(decision)
                record["execution"] = {"success": True, "detail": detail}
                if decision.action.value == "finish":
                    completed = True
            except PlanningError as exc:
                rejection_count += 1
                controller.reject(decision, str(exc))
                record["execution"] = {"success": False, "detail": str(exc)}
                if rejection_count > args.max_rejections:
                    raise PlanningError(
                        f"Gemini exceeded the action rejection budget ({args.max_rejections})."
                    ) from exc
            record["world_state_after"] = controller.world_state().model_dump(mode="json")
            action_records.append(record)
            _write_json(run_dir / "gemini_action_plan.json", action_records)
            if completed:
                break

        if not completed:
            raise PlanningError(f"Gemini did not finish within {args.max_actions} actions.")

        final_state = controller.world_state()
        if not final_state.task_complete:
            raise PlanningError("Gemini finished but geometric task verification failed.")
        video_path = _build_video(all_frames, run_dir / "rollout.mp4") if args.record_video else None
        trace = [asdict(item) for item in controller.trace]
        _write_json(run_dir / "execution_trace.json", trace)
        result.update(
            {
                "status": "completed",
                "finished_at": datetime.now(UTC).isoformat(),
                "actions": len(action_records),
                "rejected_actions": rejection_count,
                "final_state": final_state.model_dump(mode="json"),
                "video_path": video_path,
                "live_instructions": live_info["instructions"],
            }
        )
        _write_json(run_dir / "result.json", result)
        return result
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": str(exc),
                "actions": len(action_records),
            }
        )
        _write_json(run_dir / "result.json", result)
        raise
    finally:
        if app is not None:
            app.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (PathVLAError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Sorting demo failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
