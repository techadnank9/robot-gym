from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from demo_3.arena import DualG1RaceArena
from demo_3.drivers import PolicyAdapter, PolicyDecision, build_policy_adapter
from demo_3.gamepad import MacGamepad
from demo_3.schemas import (
    MatchPhase,
    PlayerConfig,
    PlayerMode,
    Skill,
    TeleopFrame,
    match_mode,
)
from demo_3.vlge_adapter import VLGEWorldAdapter


@dataclass
class PolicySlot:
    adapter: PolicyAdapter
    calls_remaining: int = 5
    pending: Future[PolicyDecision] | None = None
    last_decision_s: float = -1000.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Demo 3: VLGE G1 1v1 on a Mac")
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
        choices=("auto", "gamepad", "keyboard"),
        default="auto",
        help="Human input source; auto falls back to browser keyboard.",
    )
    parser.add_argument(
        "--p2-input",
        choices=("auto", "gamepad", "keyboard"),
        default="auto",
        help="Human input source; auto falls back to browser keyboard.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-live-ui", action="store_true")
    parser.add_argument("--http-port", type=int, default=8083)
    parser.add_argument("--websocket-port", type=int, default=8763)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Wall-clock abort limit; 0 keeps the agreed no-time-limit rule.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--result-linger-seconds",
        type=float,
        default=8.0,
        help="Keep the VLGE winner screen online after a completed match.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "darwin" and not args.headless:
        raise RuntimeError("Visible Demo 3 currently requires macOS and mjpython")
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
    output_dir = args.output_dir or _default_output_dir()
    arena = DualG1RaceArena(
        player_configs,
        viewer=not args.headless,
        realtime=not args.headless,
    )
    live: VLGEWorldAdapter | None = None
    gamepads: dict[str, MacGamepad] = {}
    keyboard_players: set[str] = set()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="demo3-policy")
    slots: dict[str, PolicySlot] = {}
    try:
        if args.validate_only:
            arena.start()
            arena.step(1000)
            report = {
                "status": "ok",
                "model": {
                    "nq": arena.model.nq,
                    "nv": arena.model.nv,
                    "nu": arena.model.nu,
                    "bodies": arena.model.nbody,
                },
                "state": arena.state_payload(),
            }
            arena.write_evidence(output_dir)
            (output_dir / "validation.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            return report

        if not args.no_live_ui:
            live = VLGEWorldAdapter(
                http_port=args.http_port,
                websocket_port=args.websocket_port,
            )
            live.start()
            print(f"VLGE-embeddable match view: {live.url}")

        for config in player_configs:
            if config.mode == PlayerMode.HUMAN:
                index = args.p1_gamepad if config.player_id == "p1" else args.p2_gamepad
                input_name = args.p1_input if config.player_id == "p1" else args.p2_input
                _configure_human_input(
                    arena,
                    config.player_id,
                    input_name=input_name,
                    gamepad_index=index,
                    live=live,
                    gamepads=gamepads,
                    keyboard_players=keyboard_players,
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

        print(f"Match mode: {match_mode(player_configs).value}")
        arena.start()
        last_publish_s = -1.0
        started_s = time.monotonic()
        while arena.phase == MatchPhase.RUNNING:
            _drain_live_commands(arena, live, keyboard_players)
            for player_id, gamepad in gamepads.items():
                arena.submit_frame(gamepad.poll())
            _poll_policy_results(arena, slots)
            _schedule_policy_decisions(arena, slots, executor)
            arena.step()
            if arena.simulation_time_s - last_publish_s >= 0.2:
                frames = _render_frames(arena) if live is not None else {}
                if live is not None:
                    live.publish(arena.state_payload(), frames)
                last_publish_s = arena.simulation_time_s
            if args.max_seconds > 0 and time.monotonic() - started_s >= args.max_seconds:
                arena.phase = MatchPhase.ABORTED
                break
        report = {
            "status": arena.phase.value,
            "winner": arena.winner,
            "matchMode": match_mode(player_configs).value,
            "state": arena.state_payload(),
        }
        if live is not None:
            live.publish(arena.state_payload(), _render_frames(arena))
            if arena.phase == MatchPhase.FINISHED:
                time.sleep(max(0.0, args.result_linger_seconds))
        arena.write_evidence(output_dir)
        (output_dir / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        for gamepad in gamepads.values():
            gamepad.close()
        if live is not None:
            live.close()
        arena.close()


def _configure_human_input(
    arena: DualG1RaceArena,
    player_id: str,
    *,
    input_name: str,
    gamepad_index: int,
    live: VLGEWorldAdapter | None,
    gamepads: dict[str, MacGamepad],
    keyboard_players: set[str],
    allow_multiple_browser_players: bool = False,
) -> None:
    if input_name == "keyboard":
        _configure_keyboard(
            arena,
            player_id,
            live,
            keyboard_players,
            allow_multiple=allow_multiple_browser_players,
        )
        return
    try:
        gamepad = MacGamepad(player_id, index=gamepad_index)
    except RuntimeError as exc:
        if input_name != "auto":
            raise
        if live is None:
            raise RuntimeError(
                f"{exc}. Browser keyboard fallback requires the live UI."
            ) from exc
        _configure_keyboard(
            arena,
            player_id,
            live,
            keyboard_players,
            allow_multiple=allow_multiple_browser_players,
        )
        print(
            f"No usable gamepad for {player_id.upper()}; browser keyboard enabled at {live.url}"
        )
        return
    gamepads[player_id] = gamepad
    arena.players[player_id].status.model_name = gamepad.name


def _configure_keyboard(
    arena: DualG1RaceArena,
    player_id: str,
    live: VLGEWorldAdapter | None,
    keyboard_players: set[str],
    *,
    allow_multiple: bool = False,
) -> None:
    if live is None:
        raise RuntimeError("Browser keyboard input requires the live UI")
    if keyboard_players and not allow_multiple:
        assigned = next(iter(keyboard_players)).upper()
        raise RuntimeError(
            "One local browser keyboard can control only one player; "
            f"{assigned} already has it. Connect a gamepad for the other human."
        )
    keyboard_players.add(player_id)
    arena.players[player_id].status.model_name = "Browser keyboard"


def _drain_live_commands(
    arena: DualG1RaceArena,
    live: VLGEWorldAdapter | None,
    keyboard_players: set[str],
) -> None:
    if live is None:
        return
    while command := live.get_command():
        if command.get("type") != "teleop":
            continue
        frame_data = command.get("frame")
        if not isinstance(frame_data, dict):
            continue
        try:
            frame = TeleopFrame.model_validate(frame_data)
        except ValueError:
            continue
        if frame.player_id not in keyboard_players:
            continue
        arena.submit_frame(frame)


def _schedule_policy_decisions(
    arena: DualG1RaceArena,
    slots: dict[str, PolicySlot],
    executor: ThreadPoolExecutor,
) -> None:
    for player_id, slot in slots.items():
        if slot.pending is not None or slot.calls_remaining <= 0:
            continue
        status = arena.policy_status(player_id)
        current = Skill(status["currentSkill"])
        needs_decision = (
            current == Skill.WAIT
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
        slot.pending = executor.submit(slot.adapter.decide, player_id, status, camera_jpegs)
        slot.calls_remaining -= 1
        slot.last_decision_s = arena.simulation_time_s
        arena.players[player_id].status.api_calls_remaining = slot.calls_remaining


def _poll_policy_results(arena: DualG1RaceArena, slots: dict[str, PolicySlot]) -> None:
    for player_id, slot in slots.items():
        future = slot.pending
        if future is None or not future.done():
            continue
        slot.pending = None
        try:
            decision = future.result()
        except Exception as exc:  # noqa: BLE001
            arena.players[player_id].status.rationale = f"Policy error: {exc}"
            continue
        arena.set_skill(
            player_id,
            decision.skill,
            rationale=decision.rationale,
            api_calls_remaining=slot.calls_remaining,
        )


def _policy_images(
    arena: DualG1RaceArena,
    player_id: str,
    adapter: PolicyAdapter,
) -> tuple[bytes, bytes]:
    if adapter.model_name == "Deterministic validation policy":
        return b"", b""
    return (
        _jpeg(arena.render(f"{player_id}_ego_camera", 640, 360)),
        _jpeg(arena.render("overhead_camera", 640, 360)),
    )


def _render_frames(arena: DualG1RaceArena) -> dict[str, bytes]:
    return {
        "broadcast": _jpeg(arena.render("broadcast_camera", 960, 540)),
        "p1": _jpeg(arena.render("p1_ego_camera", 480, 270)),
        "p2": _jpeg(arena.render("p2_ego_camera", 480, 270)),
    }


def _jpeg(array: Any) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=78, optimize=True)
    return buffer.getvalue()


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / f"demo_3_1v1_{stamp}"


def main() -> None:
    args = parse_args()
    try:
        report = run(args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001
        print(f"Demo 3 failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
