from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from demo_2.config import load_config
from demo_2.controller import (
    REAL_MOTION_ACK,
    MotionAuthorization,
    RealG1Controller,
    VelocityCommand,
)
from demo_2.errors import Demo2Error
from demo_2.mujoco_transport import MujocoTransport
from demo_2.transport import DryRunTransport, Sdk2Transport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m demo_2",
        description="MuJoCo command twin and safety-gated Unitree G1 SDK2 pilot. Defaults to dry-run.",
    )
    parser.add_argument("--config", type=Path, help="Path to a reviewed demo_2 YAML config.")
    parser.add_argument("--backend", choices=("dry-run", "mujoco", "sdk2"), default="dry-run")
    parser.add_argument("--network-interface", help="Linux interface connected to the G1.")
    parser.add_argument("--headless", action="store_true", help="Do not open the MuJoCo viewer.")
    parser.add_argument(
        "--linger",
        type=float,
        default=3.0,
        help="Seconds to keep the MuJoCo viewer open after a command.",
    )
    parser.add_argument("--enable-real-motion", action="store_true")
    parser.add_argument("--acknowledge", default="", metavar=REAL_MOTION_ACK)
    parser.add_argument("--operator-present", action="store_true")
    parser.add_argument("--remote-estop-ready", action="store_true")
    parser.add_argument("--area-clear", action="store_true")

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe", help="Read the current G1 FSM state; never commands motion.")
    commands.add_parser("stop", help="Send an immediate zero-velocity command.")

    move = commands.add_parser("move", help="Send one bounded velocity command, then stop.")
    move.add_argument("--vx", type=float, required=True, help="Forward velocity in m/s.")
    move.add_argument("--vy", type=float, default=0.0, help="Lateral velocity in m/s.")
    move.add_argument("--yaw-rate", type=float, default=0.0, help="Yaw rate in rad/s.")
    move.add_argument("--duration", type=float, required=True, help="Duration in seconds.")

    arm = commands.add_parser("arm-action", help="Run one reviewed Unitree built-in arm action.")
    arm.add_argument("--name", required=True, help="Allowlisted name from demo_2/config.yaml.")

    commands.add_parser(
        "sorting",
        help="Report why autonomous hardware sorting remains blocked.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controller: RealG1Controller | None = None
    try:
        config = load_config(args.config)
        if args.backend == "dry-run":
            transport = DryRunTransport()
        elif args.backend == "mujoco":
            transport = MujocoTransport(headless=args.headless, linger_s=args.linger)
        else:
            transport = Sdk2Transport(args.network_interface or "", config.sdk_timeout_s)
        authorization = MotionAuthorization(
            enable_real_motion=args.enable_real_motion,
            acknowledgement=args.acknowledge,
            operator_present=args.operator_present,
            remote_estop_ready=args.remote_estop_ready,
            area_clear=args.area_clear,
        )
        controller = RealG1Controller(transport, config, authorization)
        if args.command == "sorting":
            controller.run_sorting()
        controller.initialize()
        if args.command == "probe":
            report = controller.probe()
        elif args.command == "stop":
            report = controller.stop()
        elif args.command == "move":
            report = controller.move(
                VelocityCommand(
                    vx=args.vx,
                    vy=args.vy,
                    yaw_rate=args.yaw_rate,
                    duration_s=args.duration,
                )
            )
        elif args.command == "arm-action":
            report = controller.execute_arm_action(args.name)
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        if controller is not None:
            try:
                controller.stop()
            except Demo2Error as exc:
                print(f"WARNING: stop after interrupt failed: {exc}", file=sys.stderr)
        print("Interrupted; stop requested.", file=sys.stderr)
        return 130
    except Demo2Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if controller is not None:
            controller.close()
