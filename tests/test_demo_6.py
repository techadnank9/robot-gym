from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demo_3.schemas import PlayerConfig, PlayerMode, TeleopFrame
from demo_5.command_channel import SDKCompatibleCommandChannel
from demo_5.arena import SimToRealG1RaceArena
from demo_6.cli import parse_args
from demo_6.profile import DIRECTIONAL_SPEED_SCALE, PROFILE_VERSION, VELOCITY_LIMITS


def test_demo6_uses_three_times_planar_demo5_velocity():
    assert PROFILE_VERSION == "6.0"
    assert DIRECTIONAL_SPEED_SCALE == 3.0
    assert np.allclose(VELOCITY_LIMITS, [1.56, 0.78, 0.90])


def test_demo6_channel_accepts_the_turbo_limits_and_slew():
    channel = SDKCompatibleCommandChannel(
        "p2",
        np.random.default_rng(6),
        latency_range_s=(0.0, 0.0),
        dropout_probability=0.0,
        velocity_limits=VELOCITY_LIMITS,
        slew_per_packet=(0.24, 0.18, 0.16),
    )
    channel.request(3.0, -3.0, 3.0)

    assert np.allclose(channel.tick(0.0), [0.24, -0.18, 0.16])
    assert channel.metrics.clipped == 1
    assert np.allclose(channel.limits, [1.56, 0.78, 0.90])


def test_demo6_uses_separate_local_ports_by_default():
    args = parse_args(
        [
            "--p1",
            "human",
            "--p1-input",
            "idle",
            "--p2",
            "human",
            "--p2-input",
            "gamepad",
        ]
    )

    assert args.http_port == 8086
    assert args.websocket_port == 8766


def test_browser_client_recognizes_demo6_ports_and_arrow_controls():
    root = Path(__file__).resolve().parents[1]
    app = (root / "demo_3" / "web" / "app.js").read_text(encoding="utf-8")

    assert 'location.port === "8086"' in app
    assert 'demo6Host ? "8766"' in app
    assert "const demo5Host = demo6Host" in app


def test_demo6_human_direction_reaches_three_times_demo5_command():
    pytest.importorskip("mujoco")
    players = (
        PlayerConfig(
            player_id="p1",
            display_name="Idle",
            mode=PlayerMode.HUMAN,
        ),
        PlayerConfig(
            player_id="p2",
            display_name="Gamepad",
            mode=PlayerMode.HUMAN,
        ),
    )
    arena = SimToRealG1RaceArena(
        players,
        locomotion_scale=DIRECTIONAL_SPEED_SCALE,
        profile_version=PROFILE_VERSION,
        match_prefix="demo6",
    )
    try:
        frame = TeleopFrame.neutral("p2", sequence=1).model_copy(
            update={
                "connected": True,
                "deadman": True,
                "move_x": -1.0,
                "move_y": 1.0,
                "yaw": 1.0,
            }
        )
        arena.submit_frame(frame)
        arena._apply_commands()

        assert np.allclose(
            arena.locomotion.channels["p2"].desired,
            [1.56, -0.78, 0.90],
        )
        assert arena.state_payload()["simToReal"]["locomotionProfile"] == {
            "directionalSpeedScale": 3.0,
            "velocityLimits": {
                "forwardMps": 1.56,
                "lateralMps": 0.78,
                "yawRateRps": 0.9,
            },
            "simulationOnly": True,
        }
    finally:
        arena.close()
