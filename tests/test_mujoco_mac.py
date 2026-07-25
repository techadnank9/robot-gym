from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import load_scene_config
from pathvla.mujoco_lab import MacMuJoCoSortingEnv, MacSortingController
from pathvla.sorting_agent import GeminiActionDecision


def decision(action, target=None, destination=None):
    return GeminiActionDecision(
        action=action,
        target=target,
        destination=destination,
        rationale="Execute the next validated test action.",
        expected_outcome="The simulator state satisfies the action postcondition.",
    )


def test_native_mujoco_g1_can_complete_sorting(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    g1_path = project_root / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"
    if not g1_path.is_file():
        pytest.skip("Run scripts/download_g1_mjcf.sh first")
    logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()
    env = MacMuJoCoSortingEnv(g1_path, load_scene_config("sorting_lab"), tmp_path, logger, headless=True)
    try:
        controller = MacSortingController(env, logger)
        destinations = {
            "red_cube": "red_bin",
            "blue_cube": "blue_bin",
            "red_can": "red_bin",
            "blue_can": "blue_bin",
        }
        for item, destination in destinations.items():
            controller.execute(decision("navigate", target=item))
            controller.execute(decision("pick", target=item))
            controller.execute(decision("navigate", target=destination))
            controller.execute(decision("place", target=item, destination=destination))
        controller.execute(decision("finish"))
        assert controller.world_state().task_complete
        assert len(controller.trace) == 17
    finally:
        env.close()
