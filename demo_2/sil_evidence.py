from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from demo_2.config import Demo2Config, load_config
from demo_2.controller import RealG1Controller, VelocityCommand
from demo_2.errors import Demo2Error, HardwareSafetyError
from demo_2.policy_sil import PolicySilTransport, SilFaultConfig, default_official_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable Unitree G1 policy-in-MuJoCo safety evidence for Demo 2."
    )
    parser.add_argument("--official-root", type=Path, default=default_official_root())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    run_dir = _new_run_dir(args.output_dir)
    report: dict[str, Any] = {
        "status": "running",
        "kind": "unitree_g1_policy_sil_evidence",
        "started_at": datetime.now(UTC).isoformat(),
        "output_dir": str(run_dir),
        "scenarios": [],
        "guardrail_checks": _guardrail_checks(config, args.official_root),
    }
    _write_json(run_dir / "summary.json", report)

    scenarios = (
        (
            "nominal_forward",
            SilFaultConfig(seed=7),
            [VelocityCommand(0.15, 0.0, 0.0, 0.5)] * 8,
            {"minimum_displacement_m": 0.25},
        ),
        (
            "lateral_and_yaw",
            SilFaultConfig(seed=11),
            [VelocityCommand(0.0, 0.08, 0.0, 0.5)] * 4
            + [VelocityCommand(0.0, 0.0, 0.20, 0.5)] * 4,
            {"minimum_displacement_m": 0.08},
        ),
        (
            "latency_and_loss",
            SilFaultConfig(command_latency_ms=40.0, packet_loss_rate=0.15, seed=19),
            [VelocityCommand(0.15, 0.0, 0.0, 0.5)] * 8,
            {"minimum_displacement_m": 0.20, "must_drop_packets": True},
        ),
        (
            "watchdog_total_loss",
            SilFaultConfig(packet_loss_rate=1.0, watchdog_timeout_s=0.10, seed=23),
            [VelocityCommand(0.15, 0.0, 0.0, 0.5)] * 2,
            {"must_activate_watchdog": True},
        ),
    )

    for index, (name, faults, commands, criteria) in enumerate(scenarios):
        evidence = _run_scenario(
            name=name,
            official_root=args.official_root,
            config=config,
            faults=faults,
            commands=commands,
            criteria=criteria,
            viewer=args.viewer and index == 0,
            realtime=args.realtime,
        )
        report["scenarios"].append(evidence)
        _write_json(run_dir / f"{name}.json", evidence)
        _write_json(run_dir / "summary.json", report)

    report["passed"] = all(
        check["passed"] for check in report["guardrail_checks"]
    ) and all(scenario["scenario_passed"] for scenario in report["scenarios"])
    report["status"] = "completed" if report["passed"] else "failed"
    report["finished_at"] = datetime.now(UTC).isoformat()
    _write_json(run_dir / "summary.json", report)
    return report


def _run_scenario(
    *,
    name: str,
    official_root: Path,
    config: Demo2Config,
    faults: SilFaultConfig,
    commands: list[VelocityCommand],
    criteria: dict[str, Any],
    viewer: bool,
    realtime: bool,
) -> dict[str, Any]:
    transport = PolicySilTransport(
        official_root,
        faults=faults,
        viewer=viewer,
        realtime=realtime,
    )
    controller = RealG1Controller(transport, config)
    controller.initialize()
    try:
        probe = controller.probe()
        for command in commands:
            controller.move(command)
        evidence = transport.evidence_payload()
    finally:
        controller.close()

    criteria_results = {"dynamic_safety": bool(evidence["passed"])}
    if "minimum_displacement_m" in criteria:
        criteria_results["minimum_displacement"] = (
            float(evidence["planar_displacement_m"]) >= criteria["minimum_displacement_m"]
        )
    if criteria.get("must_drop_packets"):
        criteria_results["packet_loss_exercised"] = int(evidence["command_frames_dropped"]) > 0
    if criteria.get("must_activate_watchdog"):
        criteria_results["watchdog_exercised"] = int(evidence["watchdog_activations"]) > 0
    return {
        "name": name,
        "probe": probe.to_dict(),
        "commands": [asdict(command) for command in commands],
        "criteria": criteria,
        "criteria_results": criteria_results,
        "scenario_passed": all(criteria_results.values()),
        "evidence": evidence,
    }


def _guardrail_checks(config: Demo2Config, official_root: Path) -> list[dict[str, Any]]:
    transport = PolicySilTransport(official_root)
    controller = RealG1Controller(transport, config)
    checks = (
        ("forward_limit", VelocityCommand(config.limits.max_forward_mps + 0.01, 0.0, 0.0, 0.1)),
        ("lateral_limit", VelocityCommand(0.0, config.limits.max_lateral_mps + 0.01, 0.0, 0.1)),
        ("yaw_limit", VelocityCommand(0.0, 0.0, config.limits.max_yaw_rate_rps + 0.01, 0.1)),
        (
            "duration_limit",
            VelocityCommand(0.01, 0.0, 0.0, config.limits.max_command_duration_s + 0.01),
        ),
        ("non_finite", VelocityCommand(math.nan, 0.0, 0.0, 0.1)),
        ("zero_move", VelocityCommand(0.0, 0.0, 0.0, 0.1)),
    )
    controller.initialize()
    results: list[dict[str, Any]] = []
    try:
        for name, command in checks:
            try:
                controller.move(command)
            except HardwareSafetyError as exc:
                results.append({"name": name, "passed": True, "detail": str(exc)})
            else:
                results.append(
                    {"name": name, "passed": False, "detail": "unsafe command was accepted"}
                )
    finally:
        controller.close()
    return results


def _new_run_dir(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = Path(os.getenv("PATHVLA_OUTPUT_ROOT", "outputs"))
        path = (root / f"demo_2_sil_{stamp}_{uuid4().hex[:8]}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except (Demo2Error, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Demo 2 policy SIL failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
