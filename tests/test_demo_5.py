from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from demo_3.schemas import PlayerConfig, PlayerMode
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
    assert 'qs.get("player")' in app
    assert "browserPlayers.length > 1" in app
    assert "Date.now() * 1000" in app


def test_runpod_launcher_has_two_browser_human_mode():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts" / "run_g1_demo_5_runpod.sh").read_text(
        encoding="utf-8"
    )

    assert "human-vs-human|hvh)" in launcher
    assert "--p1 human --p1-input keyboard" in launcher
    assert "--p2 human --p2-input keyboard" in launcher


def test_landing_page_exposes_all_match_modes_and_rematch():
    root = Path(__file__).resolve().parents[1]
    html = (root / "demo_5" / "web" / "index.html").read_text(encoding="utf-8")
    lobby = (root / "demo_5" / "web" / "lobby.js").read_text(encoding="utf-8")
    poster = root / "demo_5" / "web" / "arena-poster.jpg"

    assert 'data-mode="ai-vs-ai"' in html
    assert 'data-mode="human-vs-ai"' in html
    assert 'data-mode="human-vs-human"' in html
    assert 'id="rematch-button"' in html
    assert 'fetch(path' in lobby
    assert '"/api/matches"' in lobby
    assert "startMatch(selectedMode, { rematch: true })" in lobby
    assert poster.stat().st_size > 50_000


def test_persistent_launcher_builds_all_three_safe_match_commands(
    monkeypatch,
    tmp_path,
):
    from demo_5.launcher import MatchLauncher

    for name in (
        "GEMINI_API_KEY",
        "DEMO3_P1_GEMINI_API_KEY",
        "DEMO3_P2_GEMINI_API_KEY",
        "DEMO5_P1_ADAPTER",
        "DEMO5_P2_ADAPTER",
        "DEMO5_OPPONENT_ADAPTER",
    ):
        monkeypatch.delenv(name, raising=False)
    launcher = MatchLauncher(
        python_bin="/usr/bin/python3",
        project_root=tmp_path,
        match_host="0.0.0.0",
        match_http_port=8086,
        websocket_port=8765,
        grasp_mode="easy",
        render_profile="performance",
    )

    ai_command, adapters = launcher.command_for_mode("ai-vs-ai")
    mixed_command, mixed_adapters = launcher.command_for_mode("human-vs-ai")
    human_command, human_adapters = launcher.command_for_mode("human-vs-human")

    assert adapters == {"p1": "scripted", "p2": "scripted"}
    assert mixed_adapters == {"p2": "scripted"}
    assert human_adapters == {}
    assert ai_command[:3] == ["/usr/bin/python3", "-m", "demo_5"]
    assert ai_command.count("policy") == 2
    assert "scripted" in ai_command
    assert "human" in mixed_command and "policy" in mixed_command
    assert human_command.count("human") == 2
    assert human_command.count("keyboard") == 2
    assert "--websocket-port" in human_command
    assert "8765" in human_command


def test_persistent_launcher_uses_gemini_only_when_a_key_exists(
    monkeypatch,
    tmp_path,
):
    from demo_5.launcher import MatchLauncher

    monkeypatch.setenv("GEMINI_API_KEY", "configured-secret")
    monkeypatch.delenv("DEMO5_P1_ADAPTER", raising=False)
    monkeypatch.delenv("DEMO5_P2_ADAPTER", raising=False)
    monkeypatch.delenv("DEMO5_OPPONENT_ADAPTER", raising=False)
    launcher = MatchLauncher(
        python_bin="/usr/bin/python3",
        project_root=tmp_path,
        match_host="0.0.0.0",
        match_http_port=8086,
        websocket_port=8765,
        grasp_mode="easy",
        render_profile="performance",
    )

    _, adapters = launcher.command_for_mode("ai-vs-ai")

    assert adapters == {"p1": "gemini-er", "p2": "gemini-er"}


def test_persistent_launcher_replaces_finished_or_active_match(tmp_path):
    from demo_5.launcher import MatchLauncher

    class FakeProcess:
        next_pid = 100

        def __init__(self):
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.returncode = None
            self.done = threading.Event()
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15
            self.done.set()

        def kill(self):
            self.returncode = -9
            self.done.set()

        def wait(self, timeout=None):
            if not self.done.wait(timeout):
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

    processes = []

    def popen(command, **kwargs):
        process = FakeProcess()
        process.command = command
        process.kwargs = kwargs
        processes.append(process)
        return process

    launcher = MatchLauncher(
        python_bin="/usr/bin/python3",
        project_root=tmp_path,
        match_host="0.0.0.0",
        match_http_port=8086,
        websocket_port=8765,
        grasp_mode="easy",
        render_profile="performance",
        popen=popen,
    )
    first = launcher.start_match("human-vs-ai")
    second = launcher.start_match("human-vs-human")

    assert first["generation"] == 1
    assert second["generation"] == 2
    assert processes[0].terminated
    assert processes[1].poll() is None
    assert second["mode"] == "human-vs-human"
    assert processes[1].kwargs["cwd"] == tmp_path
    assert isinstance(processes[1].command, list)

    launcher.close()
    assert processes[1].terminated


def test_demo5_allows_two_distinct_browser_control_seats():
    from demo_3.cli import _configure_human_input

    arena = SimpleNamespace(
        players={
            "p1": SimpleNamespace(status=SimpleNamespace(model_name="")),
            "p2": SimpleNamespace(status=SimpleNamespace(model_name="")),
        }
    )
    live = SimpleNamespace(url="http://127.0.0.1:8085/?wsPort=8765")
    keyboard_players: set[str] = set()

    for player_id in ("p1", "p2"):
        _configure_human_input(
            arena,
            player_id,
            input_name="keyboard",
            gamepad_index=0,
            live=live,
            gamepads={},
            keyboard_players=keyboard_players,
            allow_multiple_browser_players=True,
        )

    assert keyboard_players == {"p1", "p2"}
    assert arena.players["p1"].status.model_name == "Browser keyboard"
    assert arena.players["p2"].status.model_name == "Browser keyboard"


def test_runpod_two_player_urls_assign_separate_seats(monkeypatch, capsys):
    from demo_5.cli import _print_remote_player_urls

    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    _print_remote_player_urls(
        "http://0.0.0.0:8085/?wsPort=8765",
        {"p2", "p1"},
        http_port=8085,
    )
    output = capsys.readouterr().out

    assert "https://pod-123-8085.proxy.runpod.net?player=p1" in output
    assert "https://pod-123-8085.proxy.runpod.net?player=p2" in output


def test_policy_arrival_guardrail_releases_without_an_api_call():
    from demo_3.schemas import Skill
    from demo_5.cli import _apply_policy_completion_guardrails

    pending = SimpleNamespace(cancel=lambda: True)
    slot = SimpleNamespace(pending=pending, calls_remaining=0)
    selected = []
    arena = SimpleNamespace(
        policy_status=lambda player_id: {
            "currentSkill": "navigate_goal",
            "carrying": True,
            "checkpointCrossed": True,
            "nearGoal": True,
        },
        set_skill=lambda player_id, skill, **kwargs: selected.append(
            (player_id, skill, kwargs)
        ),
    )

    _apply_policy_completion_guardrails(arena, {"p2": slot})

    assert slot.pending is None
    assert selected[0][0] == "p2"
    assert selected[0][1] == Skill.RELEASE
    assert selected[0][2]["api_calls_remaining"] == 0
    assert "Grounded completion guardrail" in selected[0][2]["rationale"]


def test_policy_arrival_guardrail_requires_checkpoint_and_payload():
    from demo_5.cli import _apply_policy_completion_guardrails

    selected = []
    statuses = [
        {
            "currentSkill": "navigate_goal",
            "carrying": True,
            "checkpointCrossed": False,
            "nearGoal": True,
        },
        {
            "currentSkill": "navigate_goal",
            "carrying": False,
            "checkpointCrossed": True,
            "nearGoal": True,
        },
    ]
    arena = SimpleNamespace(
        policy_status=lambda player_id: statuses.pop(0),
        set_skill=lambda *args, **kwargs: selected.append((args, kwargs)),
    )
    slots = {
        player_id: SimpleNamespace(pending=None, calls_remaining=2)
        for player_id in ("p1", "p2")
    }

    _apply_policy_completion_guardrails(arena, slots)

    assert selected == []


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
        assert state["matchMode"] == "ai-vs-ai"
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
        assert state["simToReal"]["easyGraspCaptureRadiusM"] == {
            "human": 1.25,
            "policy": 1.45,
        }
        assert state["simToReal"]["easyReleaseAssist"] == {
            "captureRadiusM": 0.90,
            "dropHeightM": 0.38,
            "appliesTo": ["human", "policy"],
        }
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
        assert arena._easy_attached["p1"]

        goal = np.asarray(arena.scene["players"]["p1"]["goal"], dtype=float)
        base_joint = arena.model.joint("p1_floating_base_joint")
        arena.data.qpos[base_joint.qposadr[0] : base_joint.qposadr[0] + 2] = (
            goal[:2] + np.asarray([-0.75, 0.0])
        )
        arena.mujoco.mj_forward(arena.model, arena.data)
        assert arena.policy_status("p1")["nearGoal"]
        arena.players["p1"].hand_close = 0.0
        arena._apply_contact_grip()

        assert not arena._easy_attached["p1"]
        assert arena.model.geom("p1_payload_geom").contype != 0
        released = arena._payload_position("p1")
        assert np.allclose(released[:2], goal[:2])
        assert released[2] == pytest.approx(0.38)
        assert arena.events[-1].payload["gravityDrop"] is True

        for _ in range(350):
            arena.mujoco.mj_step(arena.model, arena.data)
        settled = arena._payload_position("p1")
        assert settled[2] < released[2]
        assert np.all(
            np.abs(settled[:2] - goal[:2])
            < np.asarray(arena.scene["scoring"]["goal_half_size"][:2])
        )
    finally:
        arena.close()


def test_easy_grasp_gives_policy_player_a_small_capture_assist():
    pytest.importorskip("mujoco")
    from demo_5.arena import (
        EASY_GRASP_RADIUS_HUMAN_M,
        EASY_GRASP_RADIUS_POLICY_M,
        POLICY_NEAR_OBJECT_RADIUS_M,
        SimToRealG1RaceArena,
    )

    mixed_players = (
        PlayerConfig(
            player_id="p1",
            display_name="Vector",
            mode=PlayerMode.HUMAN,
        ),
        PlayerConfig(
            player_id="p2",
            display_name="Nova",
            mode=PlayerMode.POLICY,
        ),
    )
    arena = SimToRealG1RaceArena(mixed_players, grasp_mode="easy")
    try:
        assert EASY_GRASP_RADIUS_HUMAN_M == 1.25
        assert EASY_GRASP_RADIUS_POLICY_M == 1.45
        assert POLICY_NEAR_OBJECT_RADIUS_M == 0.60
        assert (
            EASY_GRASP_RADIUS_POLICY_M - EASY_GRASP_RADIUS_HUMAN_M
            == pytest.approx(0.20)
        )
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


def test_two_player_browser_lobby_waits_for_both_seats():
    from demo_3.schemas import TeleopFrame
    from demo_5.cli import _wait_for_keyboard_ready

    players_by_id = {
        player_id: SimpleNamespace(frame=TeleopFrame.neutral(player_id))
        for player_id in ("p1", "p2")
    }
    frames = [
        TeleopFrame.neutral("p1", sequence=1).model_dump(mode="json"),
        TeleopFrame.neutral("p2", sequence=1).model_dump(mode="json"),
    ]
    commands = iter(
        [
            {"type": "teleop", "frame": frame}
            for frame in frames
        ]
        + [None]
    )
    submitted = []

    def submit(frame):
        submitted.append(frame)
        players_by_id[frame.player_id].frame = frame

    arena = SimpleNamespace(
        players=players_by_id,
        state_payload=lambda: {"phase": "lobby"},
        submit_frame=submit,
    )
    live = SimpleNamespace(
        url="http://127.0.0.1:8085/?wsPort=8765",
        publish=lambda state: None,
        get_command=lambda: next(commands),
    )

    _wait_for_keyboard_ready(arena, live, {"p1", "p2"}, timeout_s=0.5)

    assert {frame.player_id for frame in submitted} == {"p1", "p2"}
    assert all(player.frame.sequence == 1 for player in players_by_id.values())


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
