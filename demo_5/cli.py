from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from demo_3.cli import (
    PolicySlot,
    _configure_human_input,
    _jpeg,
    _policy_images,
    _poll_policy_results,
    _render_frames,
)
from demo_3.drivers import build_policy_adapter
from demo_3.gamepad import MacGamepad
from demo_3.schemas import (
    MatchPhase,
    PlayerConfig,
    PlayerMode,
    Skill,
    TeleopFrame,
    match_mode,
)
from demo_5.arena import SimToRealG1RaceArena
from demo_5.mujoco_keyboard import MujocoKeyboard
from demo_5.vlge_adapter import Demo5VLGEWorldAdapter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Demo 5: sim-to-real constrained VLGE G1 1v1"
    )
    parser.add_argument("--p1", choices=("policy", "human"), default="policy")
    parser.add_argument("--p2", choices=("policy", "human"), default="policy")
    parser.add_argument("--p1-name", default="Vector")
    parser.add_argument("--p2-name", default="Nova")
    parser.add_argument("--p1-adapter", choices=("gemini-er", "scripted", "http"), default="gemini-er")
    parser.add_argument("--p2-adapter", choices=("gemini-er", "scripted", "http"), default="gemini-er")
    parser.add_argument("--p1-model")
    parser.add_argument("--p2-model")
    parser.add_argument("--p1-endpoint")
    parser.add_argument("--p2-endpoint")
    parser.add_argument("--p1-gamepad", type=int, default=0)
    parser.add_argument("--p2-gamepad", type=int, default=1)
    parser.add_argument(
        "--p1-input",
        choices=("auto", "gamepad", "keyboard", "mujoco-keyboard", "idle"),
        default="auto",
    )
    parser.add_argument(
        "--p2-input",
        choices=("auto", "gamepad", "keyboard", "mujoco-keyboard", "idle"),
        default="auto",
    )
    parser.add_argument("--domain-seed", type=int, default=5)
    parser.add_argument(
        "--grasp-mode",
        choices=("easy", "mechanical"),
        default="easy",
        help=(
            "Easy uses a disclosed snap-to-hand attachment; mechanical requires "
            "finger/payload contact."
        ),
    )
    parser.add_argument(
        "--hardware-log",
        type=Path,
        help="Optional G1 SDK telemetry JSON/JSONL for trajectory divergence evidence.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace physics to wall time even without the native viewer.",
    )
    parser.add_argument("--no-live-ui", action="store_true")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP/WebSocket bind host; use 0.0.0.0 on RunPod.",
    )
    parser.add_argument("--http-port", type=int, default=8085)
    parser.add_argument("--websocket-port", type=int, default=8765)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument(
        "--keyboard-ready-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait in the lobby for browser keyboard telemetry.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--result-linger-seconds", type=float, default=8.0)
    parser.add_argument(
        "--render-profile",
        choices=("performance", "quality"),
        default="performance",
        help=(
            "Live spectator stream profile. Performance staggers lower-resolution "
            "camera frames; quality restores the original three-camera 5 Hz stream."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    demo_label: str = "Demo 5",
    profile_version: str = "5.0",
    locomotion_scale: float = 1.0,
    match_prefix: str = "demo5",
    output_prefix: str = "demo_5_sim_to_real",
) -> dict[str, Any]:
    if sys.platform != "darwin" and not args.headless:
        raise RuntimeError(f"Visible {demo_label} currently requires macOS and mjpython")
    if args.hardware_log is not None and not args.hardware_log.is_file():
        raise FileNotFoundError(f"Hardware telemetry log not found: {args.hardware_log}")
    player_configs = (
        PlayerConfig(
            player_id="p1",
            display_name=args.p1_name,
            mode=PlayerMode(args.p1),
            model_adapter=args.p1_adapter,
            model_name=args.p1_model,
        ),
        PlayerConfig(
            player_id="p2",
            display_name=args.p2_name,
            mode=PlayerMode(args.p2),
            model_adapter=args.p2_adapter,
            model_name=args.p2_model,
        ),
    )
    input_names = {"p1": args.p1_input, "p2": args.p2_input}
    native_player_ids = [
        config.player_id
        for config in player_configs
        if (
            config.mode == PlayerMode.HUMAN
            and input_names[config.player_id] == "mujoco-keyboard"
        )
    ]
    if len(native_player_ids) > 1:
        raise RuntimeError(
            "The MuJoCo window has one keyboard; assign mujoco-keyboard to only one player"
        )
    if native_player_ids and args.headless:
        raise RuntimeError("mujoco-keyboard requires the visible MuJoCo viewer")
    native_keyboard = (
        MujocoKeyboard(native_player_ids[0]) if native_player_ids else None
    )
    output_dir = args.output_dir or _default_output_dir(output_prefix)
    arena = SimToRealG1RaceArena(
        player_configs,
        viewer=not args.headless,
        realtime=not args.headless or args.realtime,
        domain_seed=args.domain_seed,
        hardware_log=args.hardware_log,
        viewer_key_callback=(
            native_keyboard.on_key if native_keyboard is not None else None
        ),
        grasp_mode=args.grasp_mode,
        locomotion_scale=locomotion_scale,
        profile_version=profile_version,
        match_prefix=match_prefix,
    )
    live: Demo5VLGEWorldAdapter | None = None
    gamepads: dict[str, MacGamepad] = {}
    keyboard_players: set[str] = set()
    idle_players: set[str] = set()
    slots: dict[str, PolicySlot] = {}
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="demo5-policy")
    try:
        if args.validate_only:
            arena.start()
            arena.step(1000)
            report = {
                "status": "ok",
                "profileVersion": profile_version,
                "model": {
                    "nq": arena.model.nq,
                    "nv": arena.model.nv,
                    "nu": arena.model.nu,
                },
                "state": arena.state_payload(),
            }
            arena.write_evidence(output_dir)
            (output_dir / "validation.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
            return report

        if not args.no_live_ui:
            live = Demo5VLGEWorldAdapter(
                host=args.host,
                http_port=args.http_port,
                websocket_port=args.websocket_port,
            )
            live.start()
            print(f"VLGE-embeddable {demo_label} view: {live.url}")
            if pod_id := os.getenv("RUNPOD_POD_ID"):
                print(
                    "RunPod public view: "
                    f"https://{pod_id}-{args.http_port}.proxy.runpod.net"
                )

        for config in player_configs:
            if config.mode == PlayerMode.HUMAN:
                index = args.p1_gamepad if config.player_id == "p1" else args.p2_gamepad
                input_name = args.p1_input if config.player_id == "p1" else args.p2_input
                if input_name == "idle":
                    idle_players.add(config.player_id)
                    arena.players[config.player_id].status.model_name = "Idle station"
                    arena.players[config.player_id].status.rationale = (
                        "Neutral hold: no movement or manipulation commands."
                    )
                    _refresh_idle_players(arena, idle_players, force=True)
                elif input_name == "mujoco-keyboard":
                    if native_keyboard is None:
                        raise RuntimeError("MuJoCo keyboard initialization failed")
                    arena.players[config.player_id].status.model_name = "MuJoCo keyboard"
                    arena.players[config.player_id].status.rationale = (
                        "Native viewer controls ready; select a direction and use Space to stop."
                    )
                    arena.submit_frame(native_keyboard.poll())
                else:
                    _configure_human_input(
                        arena,
                        config.player_id,
                        input_name=input_name,
                        gamepad_index=index,
                        live=live,
                        gamepads=gamepads,
                        keyboard_players=keyboard_players,
                        allow_multiple_browser_players=True,
                    )
            else:
                adapter_name = args.p1_adapter if config.player_id == "p1" else args.p2_adapter
                endpoint = args.p1_endpoint if config.player_id == "p1" else args.p2_endpoint
                adapter = build_policy_adapter(
                    adapter_name,
                    player_id=config.player_id,
                    model_name=config.model_name,
                    endpoint=endpoint,
                )
                slots[config.player_id] = PolicySlot(adapter=adapter)
                arena.players[config.player_id].status.model_name = adapter.model_name

        if len(keyboard_players) > 1 and live is not None:
            _print_remote_player_urls(
                live.url,
                keyboard_players,
                http_port=args.http_port,
            )

        print(f"Match mode: {match_mode(player_configs).value}")
        print(
            f"{demo_label} grasp: {args.grasp_mode}"
            + (
                " (assisted snap-to-hand; not sim-to-real grasp evidence)"
                if args.grasp_mode == "easy"
                else " (contact-only)"
            )
        )
        print(
            f"{demo_label} constraints: camera estimates, 50 Hz SDK channel, "
            "latency/dropout/noise/randomization"
        )
        print(
            f"{demo_label} environment: Template_73_Export V-BLDR room "
            "(15 material-aware, visual-only mesh layers)"
        )
        print(
            f"{demo_label} locomotion: {locomotion_scale:.1f}x directional profile "
            f"({arena.velocity_limits[0]:.2f} m/s forward, "
            f"{arena.velocity_limits[1]:.2f} m/s lateral, "
            f"{arena.velocity_limits[2]:.2f} rad/s yaw)"
        )
        print(f"Live render profile: {args.render_profile}")
        if native_keyboard is not None:
            print(
                f"{native_keyboard.player_id.upper()} MuJoCo keyboard: focus the "
                "MuJoCo window; use arrow keys to walk, Q/E to turn, Space to stop"
            )
            print(
                "Manipulation: G grasp, C carry, R release, U recover, "
                "X reset payload; Home resets camera zoom"
            )
        if keyboard_players:
            _wait_for_keyboard_ready(
                arena,
                live,
                keyboard_players,
                timeout_s=args.keyboard_ready_timeout,
            )
        arena.start()
        last_publish_s = -1.0
        publish_index = 0
        started_s = time.monotonic()
        while arena.phase == MatchPhase.RUNNING:
            _drain_demo5_live_commands(arena, live, keyboard_players)
            _refresh_idle_players(arena, idle_players)
            if native_keyboard is not None:
                arena.submit_frame(native_keyboard.poll())
                if notice := native_keyboard.consume_notice():
                    print(
                        f"{native_keyboard.player_id.upper()} MuJoCo input: {notice}",
                        flush=True,
                    )
                    if notice in {"forward", "backward", "left", "right"}:
                        _restore_broadcast_camera(arena)
                if native_keyboard.consume_reset_request():
                    arena.request_payload_reset(native_keyboard.player_id)
                if native_keyboard.consume_camera_reset_request():
                    arena.reset_viewer_camera()
            for player_id, gamepad in gamepads.items():
                arena.submit_frame(gamepad.poll())
            _poll_policy_results(arena, slots)
            _apply_policy_completion_guardrails(arena, slots)
            _schedule_demo5_policy_decisions(arena, slots, executor)
            arena.step()
            if arena.simulation_time_s - last_publish_s >= 0.2:
                if live is not None:
                    live.publish(
                        arena.state_payload(),
                        _render_live_frames(arena, args.render_profile, publish_index),
                    )
                    publish_index += 1
                last_publish_s = arena.simulation_time_s
            if args.max_seconds > 0 and time.monotonic() - started_s >= args.max_seconds:
                arena.phase = MatchPhase.ABORTED
                break
        report = {
            "status": arena.phase.value,
            "winner": arena.winner,
            "matchMode": match_mode(player_configs).value,
            "profileVersion": profile_version,
            "state": arena.state_payload(),
        }
        if live is not None:
            # A one-time full-resolution result frame does not affect match pace.
            live.publish(arena.state_payload(), _render_frames(arena))
            if arena.phase == MatchPhase.FINISHED:
                time.sleep(max(0.0, args.result_linger_seconds))
        arena.write_evidence(output_dir)
        (output_dir / "result.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        return report
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        for gamepad in gamepads.values():
            gamepad.close()
        if live is not None:
            live.close()
        arena.close()


def _wait_for_keyboard_ready(
    arena: SimToRealG1RaceArena,
    live: Demo5VLGEWorldAdapter | None,
    keyboard_players: set[str],
    *,
    timeout_s: float,
) -> None:
    if live is None:
        raise RuntimeError("Browser keyboard readiness requires the live UI")
    if timeout_s <= 0:
        raise ValueError("--keyboard-ready-timeout must be positive")
    names = ", ".join(sorted(player_id.upper() for player_id in keyboard_players))
    print(f"Waiting for {names} browser keyboard at {live.url}")
    started_s = time.monotonic()
    last_publish_s = -1.0
    while True:
        now = time.monotonic()
        if now - last_publish_s >= 1.0:
            live.publish(arena.state_payload())
            last_publish_s = now
        _drain_demo5_live_commands(arena, live, keyboard_players)
        if all(arena.players[player_id].frame.sequence > 0 for player_id in keyboard_players):
            print(f"{names} browser keyboard ready; starting match")
            return
        if now - started_s >= timeout_s:
            raise RuntimeError(
                f"Timed out waiting for {names} keyboard. Open {live.url} "
                "and keep that browser tab focused."
            )
        time.sleep(0.02)


def _drain_demo5_live_commands(
    arena: SimToRealG1RaceArena,
    live: Demo5VLGEWorldAdapter | None,
    keyboard_players: set[str],
) -> None:
    if live is None:
        return
    while command := live.get_command():
        command_type = command.get("type")
        if command_type == "reset_payload":
            player_id = command.get("player_id")
            if player_id in keyboard_players:
                arena.request_payload_reset(str(player_id))
            continue
        if command_type != "teleop":
            continue
        frame_data = command.get("frame")
        if not isinstance(frame_data, dict):
            continue
        try:
            frame = TeleopFrame.model_validate(frame_data)
        except ValueError:
            continue
        if frame.player_id in keyboard_players:
            arena.submit_frame(frame)


def _refresh_idle_players(
    arena: SimToRealG1RaceArena,
    player_ids: set[str],
    *,
    force: bool = False,
) -> None:
    now = time.monotonic()
    for player_id in player_ids:
        runtime = arena.players[player_id]
        if not force and now - runtime.last_valid_frame_s < 0.04:
            continue
        arena.submit_frame(
            runtime.frame.model_copy(
                update={
                    "sequence": runtime.frame.sequence + 1,
                    "timestamp_s": now,
                    "connected": True,
                    "deadman": False,
                    "move_x": 0.0,
                    "move_y": 0.0,
                    "yaw": 0.0,
                    "skill": Skill.WAIT,
                    "hand_close": 0.0,
                }
            )
        )


def _restore_broadcast_camera(arena: SimToRealG1RaceArena) -> None:
    """Undo MuJoCo's built-in arrow-key camera movement during native teleop."""

    arena.reset_viewer_camera()


def _player_url(base_url: str, player_id: str) -> str:
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}player={player_id}"


def _print_remote_player_urls(
    local_url: str,
    player_ids: set[str],
    *,
    http_port: int,
) -> None:
    pod_id = os.getenv("RUNPOD_POD_ID")
    if pod_id:
        base_url = f"https://{pod_id}-{http_port}.proxy.runpod.net"
        label = "RunPod"
    else:
        base_url = local_url
        label = "Local"
    print("Two-player browser seats (open one URL on each player's device):")
    for player_id in sorted(player_ids):
        print(f"  {player_id.upper()} {label}: {_player_url(base_url, player_id)}")


def _apply_policy_completion_guardrails(
    arena: SimToRealG1RaceArena,
    slots: dict[str, PolicySlot],
) -> None:
    """Complete an already-grounded delivery even if the model call is delayed."""

    for player_id, slot in slots.items():
        status = arena.policy_status(player_id)
        if not (
            Skill(status["currentSkill"]) == Skill.NAVIGATE_GOAL
            and status["carrying"]
            and status["checkpointCrossed"]
            and status["nearGoal"]
        ):
            continue
        if slot.pending is not None:
            slot.pending.cancel()
            slot.pending = None
        arena.set_skill(
            player_id,
            Skill.RELEASE,
            rationale=(
                "Grounded completion guardrail: checkpoint crossed and payload "
                "arrived within the assisted bucket-release radius."
            ),
            api_calls_remaining=slot.calls_remaining,
        )


def _schedule_demo5_policy_decisions(
    arena: SimToRealG1RaceArena,
    slots: dict[str, PolicySlot],
    executor: ThreadPoolExecutor,
) -> None:
    for player_id, slot in slots.items():
        if slot.pending is not None or slot.calls_remaining <= 0:
            continue
        status = arena.policy_status(player_id)
        if (
            status["resetAvailable"]
            and slot.adapter.model_name == "Deterministic validation policy"
        ):
            arena.set_skill(
                player_id,
                Skill.RECOVER,
                rationale="Validation policy requested an available referee reset.",
                api_calls_remaining=slot.calls_remaining,
            )
            continue
        current = Skill(status["currentSkill"])
        needs_decision = (
            status["resetAvailable"]
            or current == Skill.WAIT
            or status["fallen"]
            or (current == Skill.NAVIGATE_OBJECT and status["nearObject"])
            or (current == Skill.GRASP and status["carrying"])
            or (current == Skill.NAVIGATE_GOAL and not status["carrying"])
            or (current == Skill.NAVIGATE_GOAL and status["nearGoal"])
            or (current == Skill.RELEASE and status["delivered"])
        )
        if not needs_decision or arena.simulation_time_s - slot.last_decision_s < 0.5:
            continue
        camera_jpegs = _policy_images(arena, player_id, slot.adapter)
        slot.pending = executor.submit(
            slot.adapter.decide,
            player_id,
            status,
            camera_jpegs,
        )
        slot.calls_remaining -= 1
        slot.last_decision_s = arena.simulation_time_s
        arena.players[player_id].status.api_calls_remaining = slot.calls_remaining


def _render_live_frames(
    arena: SimToRealG1RaceArena,
    profile: str,
    publish_index: int,
) -> dict[str, bytes]:
    if profile == "quality":
        return _render_frames(arena)
    if profile != "performance":
        raise ValueError(f"Unknown render profile: {profile}")

    # Keep state updates at 5 Hz, but render one camera per update instead of
    # blocking physics on all three. The browser retains the previous JPEG for
    # omitted cameras. Broadcast updates at 2.5 Hz; each ego view at 1.25 Hz.
    schedule = ("broadcast", "p1", "broadcast", "p2")
    camera = schedule[publish_index % len(schedule)]
    if camera == "broadcast":
        return {
            "broadcast": _jpeg(
                arena.render("broadcast_camera", width=720, height=405)
            )
        }
    return {
        camera: _jpeg(
            arena.render(f"{camera}_ego_camera", width=320, height=180)
        )
    }


def _default_output_dir(prefix: str = "demo_5_sim_to_real") -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / f"{prefix}_{stamp}"


def main() -> None:
    try:
        report = run(parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001
        print(f"Demo 5 failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(report, indent=2))
