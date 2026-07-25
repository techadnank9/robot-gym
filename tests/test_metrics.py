from pathvla.metrics import summarize_run_metrics
from pathvla.schemas import SceneSnapshotModel


def test_metrics_summary():
    scene = SceneSnapshotModel(
        scene_name="room",
        objects=[
            {"name": "chair", "pose": [1.0, 1.0, 0.5], "type": "obstacle", "avoidance_radius": 0.5},
        ],
        robot={"name": "unitree_g1", "pose": [0.0, 0.0, 0.0]},
        bounds={"x": [-4.0, 4.0], "y": [-4.0, 4.0], "z_floor": [0.0, 0.0]},
    )
    metrics = summarize_run_metrics(
        trace=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        scene=scene,
        completed_subgoals=1,
        total_subgoals=2,
        planner_path_length_m=1.2,
    )
    assert metrics["trace_samples"] == 3
    assert metrics["success_rate"] == 0.5
