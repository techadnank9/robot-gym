from __future__ import annotations

import json
import sys

from demo_5.cli import parse_args as parse_demo5_args
from demo_5.cli import run as run_demo5
from demo_6.profile import DIRECTIONAL_SPEED_SCALE, PROFILE_VERSION


def parse_args(argv: list[str] | None = None):
    values = list(sys.argv[1:] if argv is None else argv)
    if "--http-port" not in values:
        values.extend(["--http-port", "8086"])
    if "--websocket-port" not in values:
        values.extend(["--websocket-port", "8766"])
    return parse_demo5_args(values)


def run(args):
    return run_demo5(
        args,
        demo_label="Demo 6",
        profile_version=PROFILE_VERSION,
        locomotion_scale=DIRECTIONAL_SPEED_SCALE,
        match_prefix="demo6",
        output_prefix="demo_6_turbo",
    )


def main() -> None:
    try:
        report = run(parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001
        print(f"Demo 6 failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(report, indent=2))
