from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def compare_hardware_reference(
    simulated_samples: list[dict[str, object]],
    reference_path: Path | None,
) -> dict[str, object]:
    if reference_path is None:
        return {
            "status": "not_provided",
            "detail": "Pass --hardware-log to compare a G1 SDK telemetry replay.",
        }
    reference = _read_samples(reference_path)
    simulated = _positions_by_player(simulated_samples)
    hardware = _positions_by_player(reference)
    players: dict[str, object] = {}
    for player_id in ("p1", "p2"):
        sim = simulated.get(player_id, [])
        real = hardware.get(player_id, [])
        count = min(len(sim), len(real))
        if count == 0:
            players[player_id] = {"samples": 0, "status": "insufficient_data"}
            continue
        error = np.asarray(sim[:count]) - np.asarray(real[:count])
        planar = np.linalg.norm(error[:, :2], axis=1)
        players[player_id] = {
            "samples": count,
            "planarRmseM": float(np.sqrt(np.mean(planar**2))),
            "planarP95M": float(np.percentile(planar, 95)),
            "status": "compared",
        }
    return {
        "status": "compared",
        "reference": str(reference_path),
        "players": players,
    }


def _read_samples(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get("samples", [])
    if not isinstance(value, list):
        raise ValueError("hardware log must be a JSON list, JSONL, or {'samples': [...]} object")
    return [sample for sample in value if isinstance(sample, dict)]


def _positions_by_player(
    samples: list[dict[str, object]],
) -> dict[str, list[list[float]]]:
    output: dict[str, list[list[float]]] = {"p1": [], "p2": []}
    for sample in samples:
        player_id = sample.get("playerId") or sample.get("player_id")
        position = sample.get("robot") or sample.get("position")
        if player_id not in output or not isinstance(position, list) or len(position) < 3:
            continue
        output[str(player_id)].append([float(value) for value in position[:3]])
    return output
