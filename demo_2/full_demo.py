from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

import imageio.v2 as imageio

from demo_2.config import load_config
from demo_2.controller import REAL_MOTION_ACK, MotionAuthorization, RealG1Controller
from demo_2.errors import Demo2Error
from demo_2.policy_sil import PolicySilTransport, SilFaultConfig, default_official_root
from demo_2.sorting_bridge import SdkMirroredSortingController, SdkSortingBridge
from demo_2.transport import DryRunTransport, Sdk2Transport
from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import load_scene_config
from pathvla.errors import PathVLAError, PlanningError
from pathvla.logging_utils import configure_run_logger
from pathvla.mujoco_lab import MacMuJoCoSortingEnv, MacSortingController
from pathvla.mujoco_sorting_demo import DEFAULT_TASK, default_g1_mjcf
from pathvla.sorting_agent import GeminiRoboticsERAgent


AgentFactory = Callable[[argparse.Namespace, Callable[[float], None]], object]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Demo 2: Demo 1's MuJoCo sorting lab and Gemini loop with an optional "
            "Unitree SDK2 execution bridge."
        )
    )
    parser.add_argument(
        "--execution-backend",
        choices=("mujoco", "policy-sil", "sdk-shadow", "sdk-live"),
        default="policy-sil",
    )
    parser.add_argument("--instruction", default=DEFAULT_TASK)
    parser.add_argument("--model", default=None)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--max-actions", type=int, default=24)
    parser.add_argument("--max-rejections", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--linger-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir")
    parser.add_argument("--g1-mjcf", default=str(default_g1_mjcf()))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--network-interface")
    parser.add_argument("--unitree-rl-gym-root", type=Path, default=default_official_root())
    parser.add_argument("--sil-latency-ms", type=float, default=0.0)
    parser.add_argument("--sil-packet-loss", type=float, default=0.0)
    parser.add_argument("--sil-watchdog-seconds", type=float, default=0.10)
    parser.add_argument("--sil-seed", type=int, default=1)
    parser.add_argument("--sil-realtime", action="store_true")
    parser.add_argument("--enable-real-motion", action="store_true")
    parser.add_argument("--acknowledge", default="", metavar=REAL_MOTION_ACK)
    parser.add_argument("--operator-present", action="store_true")
    parser.add_argument("--remote-estop-ready", action="store_true")
    parser.add_argument("--area-clear", action="store_true")
    parser.add_argument(
        "--twin-aligned",
        action="store_true",
        help="Confirm the physical lab x/y frame was measured against the MuJoCo frame.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, agent_factory: AgentFactory | None = None) -> dict[str, object]:
    if args.max_actions < 1:
        raise ValueError("--max-actions must be positive")
    if args.max_rejections < 0:
        raise ValueError("--max-rejections must not be negative")
    run_dir = _new_run_dir(args.output_dir)
    logger = configure_run_logger(f"demo_2.full.{run_dir.name}", run_dir / "logs.txt")
    result: dict[str, object] = {
        "status": "running",
        "runtime": "demo_2_mujoco_vla_sdk_bridge",
        "execution_backend": args.execution_backend,
        "instruction": args.instruction,
        "output_dir": str(run_dir),
        "started_at": datetime.now(UTC).isoformat(),
    }
    _write_json(run_dir / "result.json", result)
    env = None
    controller = None
    bridge = None
    records: list[dict[str, object]] = []
    frames: list[Path] = []
    rejections = 0
    try:
        scene_cfg = load_scene_config("sorting_lab")
        env = MacMuJoCoSortingEnv(
            Path(args.g1_mjcf).expanduser().resolve(),
            scene_cfg,
            run_dir,
            logger,
            args.headless,
        )
        if args.execution_backend == "mujoco":
            controller = MacSortingController(env, logger)
        else:
            bridge = _build_bridge(args)
            bridge.initialize()
            controller = SdkMirroredSortingController(env, logger, sdk_bridge=bridge)
            _write_bridge_artifacts(run_dir, bridge)

        if args.validate_only:
            captured = env.capture(0)
            frames.extend(captured)
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
                    "sdk_bridge_trace": bridge.trace_payload() if bridge else [],
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            )
            _write_json(run_dir / "result.json", result)
            return result

        def wait_with_viewer(delay_s: float) -> None:
            env.sync(delay_s)

        if agent_factory is None:
            agent = GeminiRoboticsERAgent(
                model=args.model,
                thinking_budget=args.thinking_budget,
                wait_callback=wait_with_viewer,
            )
        else:
            agent = agent_factory(args, wait_with_viewer)
        result["model"] = getattr(agent, "model", args.model or "injected-agent")
        completed = False
        for index in range(args.max_actions):
            current_frames = env.capture(index)
            frames.extend(current_frames)
            before = controller.world_state()
            env.set_agent_ui(
                "Observing twin — waiting for Gemini",
                progress=controller.completed_actions,
            )
            env.sync()
            decision = agent.decide(args.instruction, before, current_frames)
            env.set_agent_ui(
                "Decision received",
                decision=decision,
                progress=controller.completed_actions,
            )
            env.sync()
            print(
                f"Gemini: {decision.action.value}"
                f" target={decision.target or '-'}"
                f" destination={decision.destination or '-'}\n"
                f"Rationale: {decision.rationale}\n"
                f"Expected: {decision.expected_outcome}",
                flush=True,
            )
            record: dict[str, object] = {
                "index": index,
                "frames": [str(path) for path in current_frames],
                "world_state_before": before.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }
            try:
                detail = controller.execute(decision)
                record["execution"] = {"success": True, "detail": detail}
                completed = decision.action.value == "finish"
            except PlanningError as exc:
                rejections += 1
                controller.reject(decision, str(exc))
                record["execution"] = {"success": False, "detail": str(exc)}
                if rejections > args.max_rejections:
                    raise PlanningError(
                        f"Gemini exceeded the rejection budget ({args.max_rejections})."
                    ) from exc
            record["world_state_after"] = controller.world_state().model_dump(mode="json")
            records.append(record)
            _write_json(run_dir / "gemini_action_plan.json", records)
            if bridge is not None:
                _write_bridge_artifacts(run_dir, bridge)
            if completed:
                break
        if not completed:
            raise PlanningError(f"Gemini did not finish within {args.max_actions} actions.")
        final_state = controller.world_state()
        if not final_state.task_complete:
            raise PlanningError("Final geometric sorting verification failed.")
        env.set_agent_ui(
            "Task complete — twin verification passed",
            progress=controller.completed_actions,
        )
        env.sync()
        final_frames = env.capture(len(records))
        frames.extend(final_frames)
        video_path = _video(frames, run_dir / "rollout.mp4") if args.record_video else None
        _write_json(
            run_dir / "execution_trace.json",
            [asdict(entry) for entry in controller.trace],
        )
        if bridge is not None:
            _write_bridge_artifacts(run_dir, bridge)
        result.update(
            {
                "status": "completed",
                "actions": len(records),
                "rejected_actions": rejections,
                "final_state": final_state.model_dump(mode="json"),
                "final_frames": [str(path) for path in final_frames],
                "video_path": video_path,
                "sdk_bridge_trace": bridge.trace_payload() if bridge else [],
                "hardware_manipulation": (
                    "not_configured" if args.execution_backend != "mujoco" else "not_requested"
                ),
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(run_dir / "result.json", result)
        if not args.headless and args.linger_seconds > 0:
            time.sleep(args.linger_seconds)
        return result
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": str(exc),
                "actions": len(records),
                "rejected_actions": rejections,
                "sdk_bridge_trace": bridge.trace_payload() if bridge else [],
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(run_dir / "result.json", result)
        if bridge is not None:
            _write_bridge_artifacts(run_dir, bridge)
        raise
    finally:
        if bridge is not None:
            try:
                bridge.stop()
            except Exception as exc:
                logger.error("Final SDK stop failed: %s", exc)
            _write_bridge_artifacts(run_dir, bridge)
            bridge.close()
        if env is not None:
            env.close()


def _build_bridge(args: argparse.Namespace) -> SdkSortingBridge:
    config = load_config(args.config)
    authorization = MotionAuthorization(
        enable_real_motion=args.enable_real_motion,
        acknowledgement=args.acknowledge,
        operator_present=args.operator_present,
        remote_estop_ready=args.remote_estop_ready,
        area_clear=args.area_clear,
    )
    if args.execution_backend == "sdk-shadow":
        transport = DryRunTransport()
        mode = "shadow"
    elif args.execution_backend == "policy-sil":
        transport = PolicySilTransport(
            args.unitree_rl_gym_root,
            faults=SilFaultConfig(
                command_latency_ms=args.sil_latency_ms,
                packet_loss_rate=args.sil_packet_loss,
                watchdog_timeout_s=args.sil_watchdog_seconds,
                seed=args.sil_seed,
            ),
            realtime=args.sil_realtime,
        )
        mode = "shadow"
    else:
        transport = Sdk2Transport(args.network_interface or "", config.sdk_timeout_s)
        mode = "live"
    sdk_controller = RealG1Controller(transport, config, authorization)
    return SdkSortingBridge(
        controller=sdk_controller,
        config=config,
        mode=mode,
        twin_aligned=args.twin_aligned,
    )


def _new_run_dir(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = Path(os.getenv("PATHVLA_OUTPUT_ROOT", "outputs"))
        path = (root / f"demo_2_sort_{stamp}_{uuid4().hex[:8]}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_bridge_artifacts(run_dir: Path, bridge: SdkSortingBridge) -> None:
    _write_json(run_dir / "sdk_bridge_trace.json", bridge.trace_payload())
    transport = bridge.controller.transport
    if isinstance(transport, DryRunTransport):
        _write_json(run_dir / "sdk_commands.json", transport.commands)
    if isinstance(transport, PolicySilTransport):
        _write_json(run_dir / "sil_evidence.json", transport.evidence_payload())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _video(frame_paths: list[Path], destination: Path) -> str | None:
    selected = [path for path in frame_paths if path.name.endswith("camera_0.png")]
    if not selected:
        return None
    with imageio.get_writer(destination, fps=2, format="FFMPEG") as writer:
        for path in selected:
            writer.append_data(imageio.imread(path))
    return str(destination)


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (Demo2Error, PathVLAError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Demo 2 failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
