from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demo_3.schemas import PlayerConfig
from demo_5.command_channel import SDKCompatibleCommandChannel
from demo_5.evidence import compare_hardware_reference


def players():
    return (
        PlayerConfig(player_id="p1", display_name="Vector"),
        PlayerConfig(player_id="p2", display_name="Nova"),
    )


def test_runpod_cli_enables_external_realtime_headless_mode():
    from demo_5.cli import parse_args

    args = parse_args(
        [
            "--headless",
            "--realtime",
            "--host",
            "0.0.0.0",
            "--http-port",
            "8085",
            "--websocket-port",
            "8765",
        ]
    )
    assert args.headless and args.realtime
    assert args.host == "0.0.0.0"
    assert args.http_port == 8085
    assert args.websocket_port == 8765


def test_browser_match_client_includes_remote_gamepad_controls():
    root = Path(__file__).resolve().parents[1]
    app = (root / "demo_3" / "web" / "app.js").read_text(encoding="utf-8")

    assert "navigator.getGamepads" in app
    assert 'selectedSkill = "grasp"' in app
    assert 'selectedSkill = "navigate_goal"' in app
    assert 'selectedSkill = "release"' in app
    assert 'type: "reset_payload"' in app


def test_sdk_channel_clips_slews_delays_and_watchdogs():
    channel = SDKCompatibleCommandChannel(
        "p1",
        np.random.default_rng(2),
        latency_range_s=(0.04, 0.04),
        dropout_probability=0.0,
    )
    channel.request(2.0, -2.0, 3.0)
    assert np.allclose(channel.tick(0.0), 0.0)
    delivered = channel.tick(0.04)
    assert np.allclose(delivered, [0.08, -0.06, 0.16])
    assert channel.metrics.clipped == 1
    assert channel.metrics.delivered == 1
    assert any(
        command["command"] == "set_velocity"
        for command in channel.transport.commands
    )

    stopped = channel.tick(0.18)
    assert np.allclose(stopped, 0.0)
    assert channel.metrics.watchdog_stops == 1
    assert channel.transport.commands[-1]["command"] == "stop"


def test_demo5_uses_slightly_faster_sdk_velocity_limits():
    channel = SDKCompatibleCommandChannel(
        "p1",
        np.random.default_rng(2),
        dropout_probability=0.0,
    )
    assert np.allclose(channel.limits, [0.52, 0.26, 0.90])


def test_hardware_reference_comparison(tmp_path):
    reference = tmp_path / "hardware.json"
    reference.write_text(
        json.dumps(
            [
                {"playerId": "p1", "robot": [0.1, 0.0, 0.8]},
                {"playerId": "p1", "robot": [0.2, 0.0, 0.8]},
            ]
        ),
        encoding="utf-8",
    )
    result = compare_hardware_reference(
        [
            {"playerId": "p1", "robot": [0.0, 0.0, 0.8]},
            {"playerId": "p1", "robot": [0.1, 0.0, 0.8]},
        ],
        reference,
    )
    assert result["status"] == "compared"
    assert result["players"]["p1"]["planarRmseM"] == pytest.approx(0.1)


def test_demo5_removes_privileged_public_poses_and_grasp_force():
    pytest.importorskip("mujoco")
    arena = pytest.importorskip("demo_5.arena").SimToRealG1RaceArena(
        players(),
        grasp_mode="mechanical",
    )
    try:
        state = arena.state_payload()
        assert state["profileVersion"] == "5.0"
        assert "poses" not in state
        assert state["simToReal"]["privilegedControl"] is False
        assert state["simToReal"]["graspAssistForceN"] == 0.0
        arena._apply_contact_grip()
        for player_id in ("p1", "p2"):
            body_id = arena.model.body(f"{player_id}_payload").id
            assert np.allclose(arena.data.xfrc_applied[body_id], 0.0)
    finally:
        arena.close()


def test_easy_grasp_discloses_and_locks_payload_to_hand():
    pytest.importorskip("mujoco")
    from demo_3.schemas import Skill
    from demo_5.arena import SimToRealG1RaceArena

    arena = SimToRealG1RaceArena(players(), grasp_mode="easy")
    try:
        state = arena.state_payload()
        assert state["simToReal"]["privilegedControl"] is True
        assert state["simToReal"]["graspMode"] == "easy"
        arena.players["p1"].status.current_skill = Skill.GRASP
        arena._update_arm_targets()
        arena._apply_contact_grip()

        assert arena._easy_attached["p1"]
        assert arena._grasp_confirmed["p1"]
        assert arena.model.geom("p1_payload_geom").contype == 0
        payload = arena._payload_position("p1")
        grasp = arena.data.site("p1_right_grasp_site").xpos
        assert np.linalg.norm(payload - grasp) < 0.08

        arena.players["p1"].status.current_skill = Skill.RELEASE
        arena.players["p1"].hand_close = 0.0
        arena._apply_contact_grip()
        assert not arena._easy_attached["p1"]
        assert arena.model.geom("p1_payload_geom").contype != 0
    finally:
        arena.close()


def test_template_73_background_is_material_aware_and_visual_only():
    mujoco = pytest.importorskip("mujoco")
    arena = pytest.importorskip("demo_5.arena").SimToRealG1RaceArena(players())
    try:
        background_ids = [
            index
            for index in range(arena.model.ngeom)
            if (
                mujoco.mj_id2name(arena.model, mujoco.mjtObj.mjOBJ_GEOM, index)
                or ""
            ).startswith("demo5_background_geom_")
        ]
        assert len(background_ids) == 15
        assert np.all(arena.model.geom_contype[background_ids] == 0)
        assert np.all(arena.model.geom_conaffinity[background_ids] == 0)
        assert np.all(arena.model.geom_group[background_ids] == 2)
        assert arena.model.ntex >= 10
        assert any(
            np.allclose(
                arena.model.mat_rgba[arena.model.geom_matid[index], :3],
                [0.685535, 1.0, 0.996828],
            )
            for index in background_ids
        )
        floor_id = arena.model.geom("arena_floor").id
        assert arena.model.geom_contype[floor_id] != 0
        assert arena.model.geom_rgba[floor_id, 3] == 0
    finally:
        arena.close()


def test_domain_randomization_is_seed_reproducible():
    pytest.importorskip("mujoco")
    from demo_5.arena import SimToRealG1RaceArena

    first = SimToRealG1RaceArena(players(), domain_seed=17)
    second = SimToRealG1RaceArena(players(), domain_seed=17)
    try:
        assert first.domain_parameters == second.domain_parameters
        assert 0.72 <= first.domain_parameters["floorFriction"] <= 1.08
        assert 0.01 <= first.domain_parameters["packetDropProbability"] <= 0.04
    finally:
        first.close()
        second.close()


def test_keyboard_lobby_waits_for_first_valid_frame():
    from demo_5.cli import _wait_for_keyboard_ready

    frame = {
        "protocol_version": "3.0",
        "player_id": "p1",
        "sequence": 1,
        "timestamp_s": 1.0,
        "connected": True,
        "deadman": False,
        "move_x": 0.0,
        "move_y": 0.0,
        "yaw": 0.0,
        "skill": "wait",
        "hand_close": 0.0,
    }
    commands = iter(({"type": "teleop", "frame": frame}, None))
    submitted = []
    player = SimpleNamespace(frame=SimpleNamespace(sequence=0))

    def submit(value):
        submitted.append(value)
        player.frame = value

    arena = SimpleNamespace(
        players={"p1": player},
        state_payload=lambda: {"phase": "lobby"},
        submit_frame=submit,
    )
    live = SimpleNamespace(
        url="http://127.0.0.1:8085/?wsPort=8765",
        publish=lambda state: None,
        get_command=lambda: next(commands),
    )

    _wait_for_keyboard_ready(arena, live, {"p1"}, timeout_s=0.5)
    assert submitted[0].player_id == "p1"


def test_performance_render_profile_staggers_one_camera_per_publish():
    from demo_5.cli import _render_live_frames

    calls = []

    def render(camera, width, height):
        calls.append((camera, width, height))
        return np.zeros((height, width, 3), dtype=np.uint8)

    arena = SimpleNamespace(render=render)
    frames = [
        _render_live_frames(arena, "performance", index)
        for index in range(4)
    ]

    assert [set(frame) for frame in frames] == [
        {"broadcast"},
        {"p1"},
        {"broadcast"},
        {"p2"},
    ]
    assert calls == [
        ("broadcast_camera", 720, 405),
        ("p1_ego_camera", 320, 180),
        ("broadcast_camera", 720, 405),
        ("p2_ego_camera", 320, 180),
    ]


def test_idle_input_refreshes_a_connected_neutral_frame():
    from demo_5.cli import _refresh_idle_players
    from demo_3.schemas import TeleopFrame

    runtime = SimpleNamespace(
        frame=TeleopFrame.neutral("p2"),
        last_valid_frame_s=0.0,
    )
    submitted = []

    def submit(frame):
        submitted.append(frame)
        runtime.frame = frame
        runtime.last_valid_frame_s = frame.timestamp_s

    arena = SimpleNamespace(players={"p2": runtime}, submit_frame=submit)
    _refresh_idle_players(arena, {"p2"}, force=True)

    assert len(submitted) == 1
    assert submitted[0].player_id == "p2"
    assert submitted[0].connected
    assert not submitted[0].deadman
    assert submitted[0].skill.value == "wait"
    assert submitted[0].move_x == submitted[0].move_y == submitted[0].yaw == 0


def test_mujoco_keyboard_drives_motion_stops_and_expires():
    from demo_5.mujoco_keyboard import MujocoKeyboard

    now = [10.0]
    keyboard = MujocoKeyboard("p1", motion_timeout_s=0.9, clock=lambda: now[0])

    keyboard.on_key(keyboard._GLFW_UP)
    frame = keyboard.poll()
    assert frame.connected and frame.deadman
    assert frame.move_y == 1.0
    assert keyboard.consume_notice() == "forward"
    assert keyboard.consume_notice() is None

    keyboard.on_key(keyboard._GLFW_LEFT)
    frame = keyboard.poll()
    assert frame.move_y == 1.0
    assert frame.move_x == -1.0

    keyboard.on_key(ord(" "))
    frame = keyboard.poll()
    assert not frame.deadman
    assert frame.move_x == frame.move_y == frame.yaw == 0.0

    keyboard.on_key(keyboard._GLFW_RIGHT)
    now[0] += 1.0
    frame = keyboard.poll()
    assert not frame.deadman
    assert frame.move_x == 0.0


def test_mujoco_keyboard_default_motion_latches_until_space():
    from demo_5.mujoco_keyboard import MujocoKeyboard

    now = [10.0]
    keyboard = MujocoKeyboard("p1", clock=lambda: now[0])
    keyboard.on_key(keyboard._GLFW_UP)
    now[0] += 5.0
    assert keyboard.poll().move_y == 1.0
    keyboard.on_key(ord(" "))
    assert not keyboard.poll().deadman


def test_mujoco_keyboard_maps_manipulation_and_reset():
    from demo_5.mujoco_keyboard import MujocoKeyboard

    keyboard = MujocoKeyboard("p2")
    keyboard.on_key(ord("g"))
    frame = keyboard.poll()
    assert frame.skill.value == "grasp"
    assert frame.hand_close == 1.0

    keyboard.on_key(ord("R"))
    frame = keyboard.poll()
    assert frame.skill.value == "release"
    assert frame.hand_close == 0.0

    keyboard.on_key(ord("X"))
    assert keyboard.consume_reset_request()
    assert not keyboard.consume_reset_request()

    keyboard.on_key(keyboard._GLFW_HOME)
    assert keyboard.consume_camera_reset_request()
    assert not keyboard.consume_camera_reset_request()


def test_disconnect_aborts_instead_of_awarding_incomplete_opponent():
    pytest.importorskip("mujoco")
    from demo_5.arena import SimToRealG1RaceArena

    arena = SimToRealG1RaceArena(players())
    try:
        arena.start()
        arena.players["p1"].status.disqualified = True
        arena._sample_match_state()
        assert arena.phase.value == "aborted"
        assert arena.winner is None
        assert not arena.players["p2"].status.delivered
    finally:
        arena.close()


def test_staged_grasp_approaches_from_above_and_robot_side():
    from demo_5.arena import _grasp_targets

    base = np.asarray([-2.0, -0.7, 0.8])
    payload = np.asarray([-1.5, -1.0, 0.78])
    pregrasp, engage, lift = _grasp_targets(base, payload)
    direction = payload[:2] - base[:2]

    assert pregrasp[2] > engage[2] > payload[2]
    assert lift[2] > pregrasp[2]
    assert np.dot(pregrasp[:2] - payload[:2], direction) < 0
    assert np.linalg.norm(pregrasp[:2] - payload[:2]) > np.linalg.norm(
        engage[:2] - payload[:2]
    )


def test_referee_reset_requires_unreachable_payload_and_applies_penalty():
    pytest.importorskip("mujoco")
    from demo_5.arena import SimToRealG1RaceArena
    from demo_3.schemas import Skill

    arena = SimToRealG1RaceArena(players())
    try:
        arena.start()
        assert not arena.policy_status("p1")["resetAvailable"]

        joint = arena.model.joint("p1_payload_joint")
        arena.data.qpos[joint.qposadr[0] + 2] = 0.08
        arena.mujoco.mj_forward(arena.model, arena.data)
        assert arena.policy_status("p1")["resetAvailable"]

        arena.players["p1"].status.current_skill = Skill.RECOVER
        arena._apply_commands()

        assert arena._payload_position("p1")[2] > 0.7
        assert arena._reset_count["p1"] == 1
        assert arena._reset_lock_until_s["p1"] == pytest.approx(
            arena.simulation_time_s + 3.0
        )
        assert not arena.policy_status("p1")["resetAvailable"]
        assert arena.events[-1].event_type == "payload_referee_reset"
    finally:
        arena.close()


def test_three_failed_grasps_enable_reset_while_payload_is_upright():
    pytest.importorskip("mujoco")
    from demo_5.arena import SimToRealG1RaceArena

    arena = SimToRealG1RaceArena(players())
    try:
        assert arena._payload_position("p1")[2] > 0.7
        arena._grasp_attempts["p1"] = 3
        assert arena.policy_status("p1")["resetAvailable"]
        assert arena.request_payload_reset("p1")
        assert arena._reset_count["p1"] == 1
        assert not arena.policy_status("p1")["resetAvailable"]
    finally:
        arena.close()


def test_reset_command_has_priority_over_teleop_backlog():
    from demo_5.vlge_adapter import Demo5LiveMatchServer

    server = Demo5LiveMatchServer(http_port=18086, websocket_port=18766)
    server._commands.put_nowait({"type": "teleop"})
    server._priority_commands.put_nowait(
        {"type": "reset_payload", "player_id": "p1"}
    )
    assert server.get_command()["type"] == "reset_payload"
    assert server.get_command()["type"] == "teleop"
