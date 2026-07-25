import pytest


pytestmark = pytest.mark.integration


def _has_isaac():
    try:
        from isaac_ext.pathvla_unitree.tasks.room_nav_env import resolve_simulation_app_class

        resolve_simulation_app_class()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_isaac(), reason="Isaac runtime not available")
def test_isaac_launch():
    from isaac_ext.pathvla_unitree.tasks.room_nav_env import launch_simulation_app

    app = launch_simulation_app(headless=True)
    app.update()
    app.close()


@pytest.mark.skipif(not _has_isaac(), reason="Isaac runtime not available")
def test_room_scene_creation():
    from isaac_ext.pathvla_unitree.tasks.room_nav_env import create_stage, launch_simulation_app
    from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import load_scene_config
    from isaac_ext.pathvla_unitree.tasks.scene_builder import build_scene

    app = launch_simulation_app(headless=True)
    try:
        stage = create_stage()
        semantic_scene = build_scene(stage, load_scene_config("room"), logger=type("L", (), {"info": lambda *args, **kwargs: None})())
        assert semantic_scene.scene_name == "room"
        assert any(obj.name == "red_bin" for obj in semantic_scene.object_states)
    finally:
        app.close()


@pytest.mark.skipif(not _has_isaac(), reason="Isaac runtime not available")
def test_short_path_run_dev_mode():
    from isaac_ext.pathvla_unitree.tasks.room_nav_env import parse_args, run

    args = parse_args(
        [
            "--scene",
            "room",
            "--instruction",
            "Go to the red bin.",
            "--live",
            "none",
            "--record-video",
            "--allow-proxy",
            "--allow-kinematic-control",
            "--allow-rule-planner",
        ]
    )
    result = run(args)
    assert result.status in {"completed", "failed"}
