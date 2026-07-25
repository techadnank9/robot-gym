from pathvla.schemas import SceneSnapshotModel
from pathvla.vla_client import rule_based_debug_plan
from pathvla.waypoint_planner import AStarWaypointPlanner


def test_waypoint_planner_returns_path():
    scene = SceneSnapshotModel(
        scene_name="room",
        objects=[
            {"name": "red_bin", "pose": [2.0, 0.0, 0.3], "type": "container", "avoidance_radius": 0.4},
            {"name": "chair", "pose": [1.0, 0.0, 0.4], "type": "obstacle", "avoidance_radius": 0.5},
            {"name": "table", "pose": [-1.0, 1.0, 0.75], "type": "inspectable", "avoidance_radius": 0.8},
        ],
        robot={"name": "unitree_g1", "pose": [0.0, -1.5, 0.0]},
        bounds={"x": [-4.0, 4.0], "y": [-4.0, 4.0], "z_floor": [0.0, 0.0]},
    )
    subgoal = rule_based_debug_plan("Go to the red bin, avoid the chair.").subgoals[0]
    planner = AStarWaypointPlanner(grid_resolution_m=0.25)
    waypoint_plan = planner.plan(scene, subgoal)
    assert waypoint_plan.path_length_m > 0.0
    assert waypoint_plan.waypoints[-1][0] >= 1.5
