from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pathvla.errors import ConfigurationError, IsaacRuntimeError, PathVLAError, PlanningError, VLAEndpointError
from pathvla.logging_utils import configure_run_logger
from pathvla.metrics import summarize_run_metrics
from pathvla.plan_validator import validate_plan_dict
from pathvla.run_registry import persist_result, prepare_run_directory, write_json
from pathvla.schemas import RunMode, RunRequestModel, RunResultModel
from pathvla.vla_client import HTTPVLAClient, load_vla_endpoint_config, rule_based_debug_plan
from pathvla.waypoint_planner import AStarWaypointPlanner

from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import (
    load_livestream_config,
    load_robot_config,
    load_scene_config,
    load_vla_config,
)


def resolve_simulation_app_class():
    try:
        from isaacsim import SimulationApp

        return SimulationApp
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp

            return SimulationApp
        except ImportError as exc:  # noqa: PERF203
            raise IsaacRuntimeError("Isaac Sim / Isaac Lab Python runtime not found.") from exc


def launch_simulation_app(headless: bool):
    simulation_app_cls = resolve_simulation_app_class()
    return simulation_app_cls({"headless": headless, "width": 1280, "height": 720, "renderer": "RayTracedLighting"})


def create_stage():
    try:
        import omni.usd
    except ImportError as exc:
        raise IsaacRuntimeError("omni.usd is not available inside this Isaac runtime.") from exc
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    if stage is None:
        raise IsaacRuntimeError("Failed to create a USD stage inside Isaac.")
    return stage


def build_plan(args, scene_snapshot, run_dir: Path, logger):
    if args.require_vla:
        vla_config = load_vla_config()
        client = HTTPVLAClient(load_vla_endpoint_config(timeout_s=vla_config.vla.request_timeout_s))
        logger.info("Calling VLA endpoint for instruction: %s", args.instruction)
        try:
            plan = client.infer(
                instruction=args.instruction,
                scene=scene_snapshot,
                output_dir=run_dir,
                include_camera_images=vla_config.vla.include_camera_images,
            )
        except VLAEndpointError as exc:
            if not args.allow_rule_planner:
                raise
            message = (
                "DEBUG ONLY: VLA request failed and rule planner fallback is enabled. "
                "This is not real VLA mode."
            )
            logger.warning("%s VLA error: %s", message, exc)
            print(message)
            logger.info("Calling debug rule planner fallback for instruction: %s", args.instruction)
            plan = rule_based_debug_plan(args.instruction)
            logger.info("Received debug rule planner fallback response with %d subgoals", len(plan.subgoals))
            return plan
        logger.info("Received VLA response with %d subgoals", len(plan.subgoals))
        return plan
    if args.allow_rule_planner:
        message = "DEBUG ONLY: rule planner enabled. This is not VLA mode."
        logger.warning(message)
        print(message)
        logger.info("Calling debug rule planner for instruction: %s", args.instruction)
        plan = rule_based_debug_plan(args.instruction)
        logger.info("Received debug rule planner response with %d subgoals", len(plan.subgoals))
        return plan
    raise ConfigurationError("VLA planning is required unless --allow-rule-planner is explicitly passed.")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run a live Isaac scene with VLA-driven semantic navigation.")
    parser.add_argument("--scene", choices=["room", "warehouse"], default="room")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--live", choices=[mode.value for mode in RunMode], default=RunMode.WEBRTC.value)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--require-vla", dest="require_vla", action="store_true")
    parser.add_argument("--no-require-vla", dest="require_vla", action="store_false")
    parser.add_argument("--allow-proxy", action="store_true")
    parser.add_argument("--allow-kinematic-control", action="store_true")
    parser.add_argument("--allow-rule-planner", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.set_defaults(require_vla=True)
    return parser.parse_args(argv)


def run(args) -> RunResultModel:
    request = RunRequestModel(
        instruction=args.instruction,
        scene=args.scene,
        live=RunMode(args.live),
        record_video=args.record_video,
        allow_proxy=args.allow_proxy,
        allow_kinematic_control=args.allow_kinematic_control,
        require_vla=args.require_vla,
        allow_rule_planner=args.allow_rule_planner,
        require_video=args.require_video,
        output_dir=args.output_dir,
    )
    run_id, run_dir = prepare_run_directory(request)
    log_path = run_dir / "logs.txt"
    logger = configure_run_logger(f"pathvla.{run_id}", log_path)
    result = RunResultModel(
        run_id=run_id,
        status="running",
        instruction=request.instruction,
        scene=request.scene,
        live=request.live,
        output_dir=str(run_dir),
        logs_path=str(log_path),
    )
    persist_result(result)

    simulation_app = None
    try:
        scene_cfg = load_scene_config(args.scene)
        robot_cfg = load_robot_config()
        livestream_cfg = load_livestream_config()
        headless = request.live == RunMode.NONE
        simulation_app = launch_simulation_app(headless=headless)

        from isaac_ext.pathvla_unitree.tasks.livestream import configure_livestream
        from isaac_ext.pathvla_unitree.tasks.observations import get_prim_translation, refresh_semantic_scene_poses
        from isaac_ext.pathvla_unitree.tasks.recorder import IsaacRecorder
        from isaac_ext.pathvla_unitree.tasks.robot_loader import load_robot
        from isaac_ext.pathvla_unitree.tasks.scene_builder import build_scene
        from isaac_ext.pathvla_unitree.tasks.waypoint_controller import WaypointController

        stage = create_stage()
        logger.info("USD stage created")
        semantic_scene = build_scene(stage, scene_cfg, logger)
        logger.info("Semantic scene built with %d objects and %d cameras", len(semantic_scene.object_states), len(semantic_scene.camera_prim_paths))
        robot_handle = load_robot(
            stage=stage,
            robot_cfg=robot_cfg,
            allow_proxy=request.allow_proxy,
            allow_kinematic_control=request.allow_kinematic_control,
            logger=logger,
            spawn_translation=scene_cfg.scene.robot_spawn,
        )
        logger.info(
            "Robot handle ready: name=%s prim_path=%s proxy=%s control_mode=%s",
            robot_handle.name,
            robot_handle.prim_path,
            robot_handle.is_proxy,
            robot_handle.control_mode,
        )
        for _ in range(5):
            simulation_app.update()
        logger.info("Initial post-load simulation updates completed")

        livestream_info = configure_livestream(request.live, livestream_cfg, logger)
        logger.info("Livestream configured: mode=%s", livestream_info["mode"])
        robot_pose = get_prim_translation(stage, robot_handle.prim_path)
        logger.info("Robot pose sampled: %s", robot_pose)
        refresh_semantic_scene_poses(stage, semantic_scene)
        logger.info("Semantic scene poses refreshed")
        scene_snapshot = semantic_scene.to_snapshot(robot_name=robot_handle.name, robot_pose=robot_pose)
        logger.info("Scene snapshot built")
        plan = build_plan(args, scene_snapshot, run_dir, logger)
        logger.info("Plan received with %d subgoals", len(plan.subgoals))
        write_json(run_dir / "plan.json", plan.model_dump(mode="json"))
        logger.info("Plan persisted to %s", run_dir / "plan.json")

        planner = AStarWaypointPlanner()
        waypoint_plans = []
        for subgoal in plan.subgoals:
            logger.info("Planning waypoint path for subgoal type=%s target=%s", subgoal.type, subgoal.target)
            refresh_semantic_scene_poses(stage, semantic_scene)
            current_pose = get_prim_translation(stage, robot_handle.prim_path)
            subgoal_scene = semantic_scene.to_snapshot(robot_name=robot_handle.name, robot_pose=current_pose)
            waypoint_plan = planner.plan(subgoal_scene, subgoal)
            waypoint_plans.append(waypoint_plan)
            logger.info(
                "Waypoint path ready for target=%s with %d waypoints length=%.3f m",
                waypoint_plan.target,
                len(waypoint_plan.waypoints),
                waypoint_plan.path_length_m,
            )
        write_json(
            run_dir / "waypoints.json",
            {"waypoint_plans": [wp.model_dump(mode="json") for wp in waypoint_plans]},
        )
        logger.info("Waypoint plans persisted to %s", run_dir / "waypoints.json")

        preferred_camera_path = semantic_scene.camera_prim_paths[0] if semantic_scene.camera_prim_paths else None
        recorder = IsaacRecorder(
            run_dir,
            logger,
            require_video=request.require_video,
            preferred_camera_path=preferred_camera_path,
        )
        logger.info("Recorder initialized with preferred camera %s", preferred_camera_path)
        controller = WaypointController(
            stage=stage,
            robot_handle=robot_handle,
            step_fn=simulation_app.update,
            logger=logger,
        )
        logger.info("Waypoint controller initialized")
        for frame_index in range(3):
            if request.record_video:
                recorder.capture_frame(frame_index)
            simulation_app.update()
        logger.info("Pre-execution warmup frames complete")
        execution = controller.execute(
            waypoint_plans=waypoint_plans,
            scene_snapshot=scene_snapshot,
            trace_path=run_dir / "trace.json",
        )
        logger.info(
            "Execution completed: completed_subgoals=%d failures=%d trace_samples=%d",
            execution.completed_subgoals,
            len(execution.failures),
            len(execution.trace),
        )
        if request.record_video:
            for frame_index in range(3, 3 + len(execution.trace)):
                recorder.capture_frame(frame_index)
            logger.info("Post-execution frame capture complete for %d trace frames", len(execution.trace))
        video_path, final_frame_path = recorder.finalize_video() if request.record_video else (None, None)
        logger.info("Video finalize complete: video_path=%s final_frame_path=%s", video_path, final_frame_path)
        metrics = summarize_run_metrics(
            trace=execution.trace,
            scene=scene_snapshot,
            completed_subgoals=execution.completed_subgoals,
            total_subgoals=len(waypoint_plans),
            planner_path_length_m=sum(plan.path_length_m for plan in waypoint_plans),
        )
        logger.info("Metrics computed: %s", metrics)

        status = "completed" if execution.completed_subgoals == len(waypoint_plans) else "failed"
        result = RunResultModel(
            run_id=run_id,
            status=status,
            instruction=request.instruction,
            scene=request.scene,
            live=request.live,
            output_dir=str(run_dir),
            message=livestream_info["instructions"] if status == "completed" else "\n".join(execution.failures),
            plan=plan,
            waypoint_plans=waypoint_plans,
            metrics=metrics,
            video_path=video_path,
            final_frame_path=final_frame_path,
            logs_path=str(log_path),
        )
        persist_result(result)
        return result
    except PathVLAError as exc:
        result.status = "failed"
        result.message = str(exc)
        persist_result(result)
        raise
    except Exception as exc:  # noqa: BLE001
        result.status = "failed"
        result.message = f"Unhandled error: {exc}"
        persist_result(result)
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
