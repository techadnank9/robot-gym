from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

LEG_JOINT_NAMES = BODY_JOINT_NAMES[:12]

RIGHT_HAND_JOINT_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

TORQUE_LIMITS = (88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50)

_REFERENCE_ATTRIBUTES = {
    "body",
    "body1",
    "body2",
    "joint",
    "joint1",
    "joint2",
    "site",
    "site1",
    "site2",
    "target",
    "objname",
    "tendon",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_g1_path() -> Path:
    return project_root() / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"


def load_scene(path: Path | str | None = None) -> dict[str, Any]:
    scene_path = Path(path) if path else Path(__file__).with_name("scene.yaml")
    data = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
    if data.get("protocol_version") != "3.0":
        raise ValueError("Demo 3 scene protocol_version must be 3.0")
    if set(data.get("players", {})) != {"p1", "p2"}:
        raise ValueError("Demo 3 scene must define exactly p1 and p2")
    return data


def build_dual_g1_xml(
    scene: dict[str, Any],
    g1_path: Path | str | None = None,
) -> str:
    """Create one shared MuJoCo model containing two namespaced G1s.

    Mesh assets are shared, while every body, joint, actuator, sensor, geom and
    site belonging to a player is prefixed. Leg position actuators are replaced
    by bounded torque motors so the official Unitree locomotion policy can drive
    both robots inside the same contact simulation.
    """

    source_path = Path(g1_path) if g1_path else default_g1_path()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"G1-with-hands MJCF not found at {source_path}. Run scripts/download_g1_mjcf.sh."
        )
    source = ET.parse(source_path).getroot()
    pelvis = source.find("./worldbody/body[@name='pelvis']")
    source_actuators = source.find("actuator")
    if pelvis is None or source_actuators is None:
        raise ValueError("G1-with-hands model is missing pelvis or actuators")

    root = ET.Element("mujoco", model="vlge_g1_1v1_demo_3")
    compiler = copy.deepcopy(source.find("compiler"))
    if compiler is None:
        compiler = ET.Element("compiler")
    compiler.set("meshdir", str((source_path.parent / "assets").resolve()))
    root.append(compiler)
    ET.SubElement(
        root,
        "option",
        timestep=str(scene["physics"]["timestep_s"]),
        integrator="implicitfast",
        gravity="0 0 -9.81",
        cone="elliptic",
    )
    for tag in ("default", "asset"):
        element = source.find(tag)
        if element is not None:
            root.append(copy.deepcopy(element))
    _add_visual(root)

    worldbody = ET.SubElement(root, "worldbody")
    _add_arena(worldbody, scene)
    actuator = ET.SubElement(root, "actuator")
    sensor = ET.SubElement(root, "sensor")

    for player_id in ("p1", "p2"):
        player = scene["players"][player_id]
        robot = copy.deepcopy(pelvis)
        _add_robot_sites(robot)
        _prefix_tree(robot, f"{player_id}_")
        robot.set("pos", _numbers(player["spawn"]))
        yaw = math.radians(float(player["yaw_deg"]))
        robot.set("quat", _numbers([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
        torso = robot.find(f".//body[@name='{player_id}_torso_link']")
        if torso is not None:
            ET.SubElement(
                torso,
                "geom",
                name=f"{player_id}_identity_panel",
                type="box",
                pos="0.02 0 0.19",
                size="0.055 0.105 0.075",
                rgba=_numbers(player["color"]),
                contype="0",
                conaffinity="0",
                group="1",
            )
        worldbody.append(robot)
        for source_item in source_actuators:
            joint_name = source_item.get("joint")
            if not joint_name:
                continue
            if joint_name in LEG_JOINT_NAMES:
                index = LEG_JOINT_NAMES.index(joint_name)
                item = ET.Element(
                    "motor",
                    name=f"{player_id}_{joint_name}",
                    joint=f"{player_id}_{joint_name}",
                    gear="1",
                    ctrllimited="true",
                    ctrlrange=f"{-TORQUE_LIMITS[index]} {TORQUE_LIMITS[index]}",
                )
            else:
                item = copy.deepcopy(source_item)
                _prefix_tree(item, f"{player_id}_")
            actuator.append(item)

        source_sensors = source.find("sensor")
        if source_sensors is not None:
            for source_item in source_sensors:
                item = copy.deepcopy(source_item)
                _prefix_tree(item, f"{player_id}_")
                sensor.append(item)

    return ET.tostring(root, encoding="unicode")


def _add_visual(root: ET.Element) -> None:
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.72 0.75 0.82",
        ambient="0.20 0.22 0.27",
        specular="0.55 0.58 0.65",
    )
    ET.SubElement(
        visual,
        "global",
        azimuth="128",
        elevation="-23",
        offwidth="1024",
        offheight="576",
    )
    ET.SubElement(visual, "rgba", haze="0.08 0.10 0.14 1")


def _add_arena(worldbody: ET.Element, scene: dict[str, Any]) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        name="arena_floor",
        type="plane",
        size="5 4 0.05",
        rgba="0.055 0.064 0.08 1",
        friction=_numbers(scene["physics"]["floor_friction"]),
    )
    ET.SubElement(worldbody, "light", pos="0 0 5", dir="0 0 -1", directional="true")
    ET.SubElement(worldbody, "light", pos="-3 -3 2.4", diffuse="0.55 0.22 0.18")
    ET.SubElement(worldbody, "light", pos="3 3 2.4", diffuse="0.18 0.34 0.65")

    x0, x1, y0, y1 = [float(value) for value in scene["arena"]["bounds"]]
    wall_h = 0.28
    for name, pos, size in (
        ("north_wall", [(x0 + x1) / 2, y1, wall_h], [(x1 - x0) / 2, 0.06, wall_h]),
        ("south_wall", [(x0 + x1) / 2, y0, wall_h], [(x1 - x0) / 2, 0.06, wall_h]),
        ("east_wall", [x1, (y0 + y1) / 2, wall_h], [0.06, (y1 - y0) / 2, wall_h]),
        ("west_wall", [x0, (y0 + y1) / 2, wall_h], [0.06, (y1 - y0) / 2, wall_h]),
    ):
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="box",
            pos=_numbers(pos),
            size=_numbers(size),
            rgba="0.20 0.22 0.27 1",
            friction="0.8 0.02 0.002",
        )

    ET.SubElement(
        worldbody,
        "geom",
        name="checkpoint_line",
        type="box",
        pos="0 0 0.006",
        size="0.025 1.85 0.006",
        rgba="0.88 0.90 0.94 0.42",
        contype="0",
        conaffinity="0",
    )
    goal_size = scene["scoring"]["goal_half_size"]
    for player_id in ("p1", "p2"):
        player = scene["players"][player_id]
        color = player["color"]
        goal = player["goal"]
        half_x, half_y, _ = [float(value) for value in goal_size]
        wall = 0.035
        wall_half_height = 0.15
        floor_half_height = 0.025
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{player_id}_goal_marker",
            type="box",
            pos=_numbers([goal[0], goal[1], floor_half_height + 0.002]),
            size=_numbers([half_x - wall, half_y - wall, 0.003]),
            rgba=_numbers([color[0], color[1], color[2], 0.28]),
            contype="0",
            conaffinity="0",
        )
        bucket_pieces = (
            (
                "bottom",
                [goal[0], goal[1], floor_half_height],
                [half_x, half_y, floor_half_height],
            ),
            (
                "north",
                [goal[0], goal[1] + half_y - wall / 2, wall_half_height],
                [half_x, wall / 2, wall_half_height],
            ),
            (
                "south",
                [goal[0], goal[1] - half_y + wall / 2, wall_half_height],
                [half_x, wall / 2, wall_half_height],
            ),
            (
                "east",
                [goal[0] + half_x - wall / 2, goal[1], wall_half_height],
                [wall / 2, half_y - wall, wall_half_height],
            ),
            (
                "west",
                [goal[0] - half_x + wall / 2, goal[1], wall_half_height],
                [wall / 2, half_y - wall, wall_half_height],
            ),
        )
        for piece, position, size in bucket_pieces:
            ET.SubElement(
                worldbody,
                "geom",
                name=f"{player_id}_bucket_{piece}",
                type="box",
                pos=_numbers(position),
                size=_numbers(size),
                rgba=_numbers([color[0], color[1], color[2], 0.82]),
                friction="0.95 0.02 0.002",
                solref="0.008 1",
            )
        object_position = player["object"]
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{player_id}_pickup_plinth",
            type="box",
            pos=_numbers([object_position[0], object_position[1], 0.37]),
            size="0.22 0.22 0.37",
            rgba="0.16 0.17 0.20 1",
            friction="0.9 0.02 0.002",
        )
        payload = ET.SubElement(
            worldbody,
            "body",
            name=f"{player_id}_payload",
            pos=_numbers(object_position),
        )
        ET.SubElement(payload, "freejoint", name=f"{player_id}_payload_joint")
        ET.SubElement(
            payload,
            "geom",
            name=f"{player_id}_payload_geom",
            type="box",
            size="0.18 0.04 0.04",
            mass="0.10",
            rgba=_numbers(color),
            friction="1.2 0.015 0.001",
            solref="0.006 1",
        )

    ET.SubElement(
        worldbody,
        "camera",
        name="broadcast_camera",
        pos="0 -5.8 2.65",
        xyaxes="1 0 0 0 0.42 0.91",
        fovy="52",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="overhead_camera",
        pos="0 0 7.4",
        quat="0 1 0 0",
        fovy="48",
    )


def _add_robot_sites(robot: ET.Element) -> None:
    wrist = robot.find(".//body[@name='right_wrist_yaw_link']")
    torso = robot.find(".//body[@name='torso_link']")
    if wrist is None or torso is None:
        raise ValueError("G1 model is missing right wrist or torso")
    ET.SubElement(
        wrist,
        "site",
        name="right_grasp_site",
        pos="0.145 -0.0046 0",
        size="0.012",
        rgba="0.2 1 0.3 0.8",
    )
    ET.SubElement(
        torso,
        "camera",
        name="ego_camera",
        pos="0.10 0 0.34",
        quat="0.7071068 0 -0.7071068 0",
        fovy="66",
    )


def _prefix_tree(element: ET.Element, prefix: str) -> None:
    for node in element.iter():
        if node.get("name"):
            node.set("name", prefix + str(node.get("name")))
        for attribute in _REFERENCE_ATTRIBUTES:
            value = node.get(attribute)
            if value:
                node.set(attribute, prefix + value)


def _numbers(values: Any) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
