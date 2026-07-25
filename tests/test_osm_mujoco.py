import json
import math
from pathlib import Path

import pytest

from pathvla.osm_mujoco import (
    ElevationGrid,
    GeoBounds,
    LocalProjection,
    build_mujoco_scene,
    parse_osm,
    write_heightfield_png,
)


ROOT = Path(__file__).resolve().parents[1]


def test_local_projection_round_trip_and_axis_convention():
    projection = LocalProjection(37.795, -122.395)
    x, y = projection.to_local(37.796, -122.394)
    assert x > 80.0
    assert y > 110.0
    latitude, longitude = projection.to_geo(x, y)
    assert latitude == pytest.approx(37.796)
    assert longitude == pytest.approx(-122.394)


def test_osm_parser_identifies_building_road_and_sidewalk():
    data = parse_osm(ROOT / "tests/fixtures/osm_tiny.osm")
    assert data.node_count == 16
    assert len(data.ways) == 7
    assert {way.tags.get("highway") for way in data.ways} == {None, "residential", "footway"}
    relation_building = next(way for way in data.ways if way.osm_id == "relation_500_0")
    assert relation_building.coordinates[0] == relation_building.coordinates[-1]


def test_elevation_grid_bilinear_sampling_and_png(tmp_path):
    bounds = GeoBounds(37.0, -123.0, 38.0, -122.0)
    grid = ElevationGrid(bounds, 2, 2, ((0.0, 10.0), (20.0, 30.0)))
    assert grid.at_geo(37.5, -122.5) == pytest.approx(15.0)
    target = tmp_path / "height.png"
    write_heightfield_png(grid, target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_fixture_builds_georeferenced_mujoco_scene(tmp_path):
    g1 = ROOT / "assets/mujoco_menagerie/unitree_g1/g1_with_hands.xml"
    if not g1.is_file():
        pytest.skip("Run scripts/download_g1_mjcf.sh first")
    config = tmp_path / "scene.yaml"
    config.write_text(
        """
scene:
  name: fixture_sf
  bounds: {south: 37.7949, west: -122.3951, north: 37.7951, east: -122.3947}
  robot_spawn: {latitude: 37.7950, longitude: -122.3949, heading_degrees: 0, base_height_m: 0.82}
  elevation: {provider: flat, rows: 3, columns: 3}
  rendering: {default_building_height_m: 12, level_height_m: 3, road_thickness_m: 0.04, camera_height_m: 40}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = build_mujoco_scene(
        config,
        tmp_path / "output",
        g1,
        osm_path=ROOT / "tests/fixtures/osm_tiny.osm",
    )
    xml = result.xml_path.read_text(encoding="utf-8")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert 'type="hfield"' in xml
    assert 'name="building_100"' in xml
    assert 'name="building_relation_500_0"' in xml
    assert 'name="building_color_100"' in xml
    assert 'name="surface_park_600"' in xml
    assert 'name="road_200_0"' in xml
    assert 'name="sidewalk_300_0"' in xml
    assert result.feature_counts == {
        "roads": 1,
        "sidewalks": 1,
        "buildings": 2,
        "other_paths": 0,
        "colored_surfaces": 1,
    }
    assert math.dist(result.spawn_xyz[:2], (0.0, 0.0)) < 0.01
    assert metadata["axis_convention"]["x"] == "east (meters)"

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(result.xml_path))
    assert model.nbody > 1


def test_golden_gate_landmark_adds_suspension_structure(tmp_path):
    g1 = ROOT / "assets/mujoco_menagerie/unitree_g1/g1_with_hands.xml"
    if not g1.is_file():
        pytest.skip("Run scripts/download_g1_mjcf.sh first")
    config = tmp_path / "bridge.yaml"
    config.write_text(
        """
scene:
  name: fixture_bridge
  bounds: {south: 37.7949, west: -122.3951, north: 37.7951, east: -122.3947}
  robot_spawn: {latitude: 37.7950, longitude: -122.3949, heading_degrees: 0, base_height_m: 0.82, surface_elevation_m: 10}
  elevation: {provider: flat, rows: 3, columns: 3}
  rendering: {default_building_height_m: 12, level_height_m: 3, road_thickness_m: 0.04, camera_height_m: 40}
  landmark:
    type: golden_gate_bridge
    south_anchor: {latitude: 37.79492, longitude: -122.3950}
    south_tower: {latitude: 37.79496, longitude: -122.39496}
    north_tower: {latitude: 37.79504, longitude: -122.39484}
    north_anchor: {latitude: 37.79508, longitude: -122.39480}
    deck_elevation_m: 10
    deck_width_m: 8
    tower_top_elevation_m: 30
    main_cable_mid_elevation_m: 13
    anchor_cable_elevation_m: 15
    water_elevation_m: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = build_mujoco_scene(
        config,
        tmp_path / "bridge_output",
        g1,
        osm_path=ROOT / "tests/fixtures/osm_tiny.osm",
    )
    xml = result.xml_path.read_text(encoding="utf-8")
    assert 'name="golden_gate_deck"' in xml
    assert 'name="golden_gate_south_tower_leg_east"' in xml
    assert 'name="golden_gate_main_cable_0_0"' in xml
    assert 'name="landmark_overview"' in xml
    assert result.feature_counts["bridge_structures"] == 1
    assert (result.xml_path.parent / "pathvla_asphalt.png").is_file()

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(result.xml_path))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "landmark_overview") >= 0
