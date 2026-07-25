from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from demo_3.arena import DualG1RaceArena
from demo_3.drivers import (
    CustomHTTPPolicyAdapter,
    ScriptedPolicyAdapter,
    _limiter_for_key,
)
from demo_3.model import build_dual_g1_xml, load_scene
from demo_3.schemas import PlayerConfig, PlayerMode, Skill, TeleopFrame, match_mode


def players():
    return (
        PlayerConfig(player_id="p1", display_name="Vector"),
        PlayerConfig(player_id="p2", display_name="Nova"),
    )


def test_player_modes_and_teleop_bounds():
    mixed = (
        PlayerConfig(player_id="p1", display_name="Human", mode=PlayerMode.HUMAN),
        PlayerConfig(player_id="p2", display_name="Model"),
    )
    assert match_mode(mixed).value == "human-vs-ai"
    with pytest.raises(ValueError):
        TeleopFrame(
            player_id="p1",
            sequence=1,
            timestamp_s=time.monotonic(),
            move_x=1.5,
            move_y=0,
            yaw=0,
        )


def test_dual_model_is_namespaced_and_dynamic():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(build_dual_g1_xml(load_scene()))
    assert model.nq == 114
    assert model.nu == 86
    assert model.joint("p1_floating_base_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("p2_floating_base_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("p1_payload_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("p2_payload_joint").type == mujoco.mjtJoint.mjJNT_FREE
    for player_id in ("p1", "p2"):
        for piece in ("bottom", "north", "south", "east", "west"):
            geom = model.geom(f"{player_id}_bucket_{piece}")
            assert geom.contype != 0
            assert geom.conaffinity != 0


def test_two_policies_hold_finite_upright_shared_world():
    pytest.importorskip("torch")
    arena = DualG1RaceArena(players())
    try:
        arena.start()
        arena.step(1000)
        state = arena.state_payload()
        assert state["simulationTime"] == pytest.approx(2.0)
        assert not state["players"]["p1"]["fallen"]
        assert not state["players"]["p2"]["fallen"]
        assert state["players"]["p1"]["connected"]
        assert state["players"]["p2"]["connected"]
    finally:
        arena.close()


def test_scripted_adapter_selects_grounded_sequence():
    adapter = ScriptedPolicyAdapter()
    base = {
        "fallen": False,
        "delivered": False,
        "nearGoal": False,
        "checkpointCrossed": False,
        "carrying": False,
        "nearObject": False,
    }
    assert adapter.decide("p1", base, (b"", b"")).skill == Skill.NAVIGATE_OBJECT
    assert (
        adapter.decide("p1", {**base, "nearObject": True}, (b"", b"")).skill
        == Skill.GRASP
    )
    assert (
        adapter.decide("p1", {**base, "carrying": True}, (b"", b"")).skill
        == Skill.NAVIGATE_GOAL
    )


def test_custom_policy_requires_https_or_exact_localhost():
    CustomHTTPPolicyAdapter(
        "http://localhost:9000/policy",
        model_name="Local model",
    )
    with pytest.raises(ValueError, match="HTTPS"):
        CustomHTTPPolicyAdapter(
            "http://localhost.evil.example/policy",
            model_name="Untrusted model",
        )


def test_gemini_limiter_is_shared_only_by_matching_key():
    assert _limiter_for_key("shared-key") is _limiter_for_key("shared-key")
    assert _limiter_for_key("shared-key") is not _limiter_for_key("other-key")


def test_browser_keyboard_teleop_is_validated_and_player_scoped():
    from demo_3.cli import _drain_live_commands

    valid = TeleopFrame(
        player_id="p1",
        sequence=4,
        timestamp_s=1.0,
        deadman=True,
        move_x=0.25,
        move_y=1.0,
        yaw=-0.5,
    ).model_dump(mode="json")
    commands = iter(
        (
            {"type": "teleop", "frame": {**valid, "move_y": 2.0}},
            {"type": "teleop", "frame": {**valid, "player_id": "p2"}},
            {"type": "teleop", "frame": valid},
            None,
        )
    )
    live = SimpleNamespace(get_command=lambda: next(commands))
    received = []
    arena = SimpleNamespace(submit_frame=received.append)

    _drain_live_commands(arena, live, {"p1"})

    assert len(received) == 1
    assert received[0].player_id == "p1"
    assert received[0].move_y == 1.0


def test_auto_human_input_falls_back_to_browser_keyboard(monkeypatch):
    from demo_3.cli import _configure_human_input

    def no_gamepad(*args, **kwargs):
        raise RuntimeError("No gamepad found")

    monkeypatch.setattr("demo_3.cli.MacGamepad", no_gamepad)
    arena = SimpleNamespace(
        players={
            "p1": SimpleNamespace(
                status=SimpleNamespace(model_name="Mac gamepad"),
            )
        }
    )
    keyboard_players = set()

    _configure_human_input(
        arena,
        "p1",
        input_name="auto",
        gamepad_index=0,
        live=SimpleNamespace(url="http://127.0.0.1:8083"),
        gamepads={},
        keyboard_players=keyboard_players,
    )

    assert keyboard_players == {"p1"}
    assert arena.players["p1"].status.model_name == "Browser keyboard"


@pytest.mark.integration
def test_scripted_match_reaches_authoritative_winner(tmp_path):
    from demo_3.cli import parse_args, run

    args = parse_args(
        [
            "--headless",
            "--no-live-ui",
            "--p1-adapter",
            "scripted",
            "--p2-adapter",
            "scripted",
            "--max-seconds",
            "25",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report = run(args)
    assert report["status"] == "finished"
    assert report["winner"] in {"p1", "p2"}
    assert report["state"]["result"] in {"P1 WON", "P2 WON"}
    assert (tmp_path / "result.json").is_file()
