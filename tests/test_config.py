from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import (
    load_livestream_config,
    load_robot_config,
    load_scene_config,
    load_vla_config,
)


def test_scene_configs_load():
    assert load_scene_config("room").scene.name == "room"
    assert load_scene_config("warehouse").scene.name == "warehouse"
    sorting_lab = load_scene_config("sorting_lab").scene
    assert sorting_lab.name == "sorting_lab"
    assert sum(obj.type == "sort_item" for obj in sorting_lab.objects) == 4
    assert sum(obj.shape == "bucket" for obj in sorting_lab.objects) == 2


def test_other_configs_load():
    assert load_robot_config().robot.name == "unitree_g1"
    assert load_livestream_config().livestream.signaling_port == 8211
    assert load_vla_config().vla.request_timeout_s == 30
