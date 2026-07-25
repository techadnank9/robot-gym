from __future__ import annotations

import math
from typing import Any

from pathvla.schemas import SceneSnapshotModel


def compute_path_length(trace: list[list[float]]) -> float:
    total = 0.0
    for idx in range(1, len(trace)):
        total += math.dist(trace[idx - 1][:3], trace[idx][:3])
    return total


def compute_min_obstacle_distance(trace: list[list[float]], scene: SceneSnapshotModel) -> float:
    min_distance = float("inf")
    for pose in trace:
        for obj in scene.objects:
            if obj.type in {"obstacle", "inspectable", "container"}:
                distance = math.dist(pose[:2], obj.pose[:2]) - obj.avoidance_radius
                min_distance = min(min_distance, distance)
    return min_distance if min_distance != float("inf") else 0.0


def summarize_run_metrics(
    trace: list[list[float]],
    scene: SceneSnapshotModel,
    completed_subgoals: int,
    total_subgoals: int,
    planner_path_length_m: float,
) -> dict[str, Any]:
    return {
        "trace_samples": len(trace),
        "trace_path_length_m": compute_path_length(trace),
        "planner_path_length_m": planner_path_length_m,
        "min_obstacle_distance_m": compute_min_obstacle_distance(trace, scene),
        "completed_subgoals": completed_subgoals,
        "total_subgoals": total_subgoals,
        "success_rate": (completed_subgoals / total_subgoals) if total_subgoals else 0.0,
    }
