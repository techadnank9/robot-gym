from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from pathvla.errors import ConfigurationError


EARTH_RADIUS_M = 6_378_137.0
OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map"
USGS_EPQS_URL = "https://epqs.nationalmap.gov/v1/json"


@dataclass(frozen=True)
class GeoBounds:
    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.south < self.north <= 90.0):
            raise ValueError("Expected -90 <= south < north <= 90")
        if not (-180.0 <= self.west < self.east <= 180.0):
            raise ValueError("Expected -180 <= west < east <= 180")

    @property
    def center(self) -> tuple[float, float]:
        return (self.south + self.north) / 2.0, (self.west + self.east) / 2.0

    @property
    def osm_bbox(self) -> str:
        return f"{self.west:.7f},{self.south:.7f},{self.east:.7f},{self.north:.7f}"


@dataclass(frozen=True)
class LocalProjection:
    latitude: float
    longitude: float

    def to_local(self, latitude: float, longitude: float) -> tuple[float, float]:
        x = EARTH_RADIUS_M * math.radians(longitude - self.longitude) * math.cos(math.radians(self.latitude))
        y = EARTH_RADIUS_M * math.radians(latitude - self.latitude)
        return x, y

    def to_geo(self, x: float, y: float) -> tuple[float, float]:
        latitude = self.latitude + math.degrees(y / EARTH_RADIUS_M)
        longitude = self.longitude + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(self.latitude))))
        return latitude, longitude


@dataclass(frozen=True)
class OSMWay:
    osm_id: str
    coordinates: tuple[tuple[float, float], ...]
    tags: dict[str, str]


@dataclass(frozen=True)
class OSMData:
    ways: tuple[OSMWay, ...]
    node_count: int
    source_bounds: GeoBounds | None = None


@dataclass(frozen=True)
class ElevationGrid:
    bounds: GeoBounds
    rows: int
    columns: int
    values_m: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.rows < 2 or self.columns < 2:
            raise ValueError("Elevation grid must be at least 2 by 2")
        if len(self.values_m) != self.rows or any(len(row) != self.columns for row in self.values_m):
            raise ValueError("Elevation values do not match grid dimensions")

    @property
    def minimum(self) -> float:
        return min(min(row) for row in self.values_m)

    @property
    def maximum(self) -> float:
        return max(max(row) for row in self.values_m)

    def at_geo(self, latitude: float, longitude: float) -> float:
        row = (latitude - self.bounds.south) / (self.bounds.north - self.bounds.south) * (self.rows - 1)
        column = (longitude - self.bounds.west) / (self.bounds.east - self.bounds.west) * (self.columns - 1)
        row = min(max(row, 0.0), self.rows - 1.0)
        column = min(max(column, 0.0), self.columns - 1.0)
        r0, c0 = int(math.floor(row)), int(math.floor(column))
        r1, c1 = min(r0 + 1, self.rows - 1), min(c0 + 1, self.columns - 1)
        tr, tc = row - r0, column - c0
        south = self.values_m[r0][c0] * (1.0 - tc) + self.values_m[r0][c1] * tc
        north = self.values_m[r1][c0] * (1.0 - tc) + self.values_m[r1][c1] * tc
        return south * (1.0 - tr) + north * tr


@dataclass(frozen=True)
class SceneConfig:
    name: str
    bounds: GeoBounds
    spawn_latitude: float
    spawn_longitude: float
    spawn_heading_degrees: float
    robot_base_height_m: float
    spawn_elevation_offset_m: float
    spawn_surface_elevation_m: float | None
    elevation_provider: str
    elevation_rows: int
    elevation_columns: int
    default_building_height_m: float
    level_height_m: float
    road_thickness_m: float
    camera_height_m: float
    elevated_level_height_m: float
    surface_overrides_m: dict[str, float]
    landmark: dict[str, Any] | None


@dataclass(frozen=True)
class BuildResult:
    xml_path: Path
    metadata_path: Path
    osm_path: Path
    heightfield_path: Path
    spawn_xyz: tuple[float, float, float]
    spawn_quaternion: tuple[float, float, float, float]
    feature_counts: dict[str, int]


def load_scene_config(path: Path) -> SceneConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["scene"]
        bbox = raw["bounds"]
        spawn = raw["robot_spawn"]
        elevation = raw.get("elevation", {})
        rendering = raw.get("rendering", {})
        config = SceneConfig(
            name=str(raw["name"]),
            bounds=GeoBounds(
                south=float(bbox["south"]),
                west=float(bbox["west"]),
                north=float(bbox["north"]),
                east=float(bbox["east"]),
            ),
            spawn_latitude=float(spawn["latitude"]),
            spawn_longitude=float(spawn["longitude"]),
            spawn_heading_degrees=float(spawn.get("heading_degrees", 0.0)),
            robot_base_height_m=float(spawn.get("base_height_m", 0.82)),
            spawn_elevation_offset_m=float(spawn.get("elevation_offset_m", 0.0)),
            spawn_surface_elevation_m=(
                float(spawn["surface_elevation_m"])
                if spawn.get("surface_elevation_m") is not None
                else None
            ),
            elevation_provider=str(elevation.get("provider", "usgs")),
            elevation_rows=int(elevation.get("rows", 7)),
            elevation_columns=int(elevation.get("columns", 7)),
            default_building_height_m=float(rendering.get("default_building_height_m", 12.0)),
            level_height_m=float(rendering.get("level_height_m", 3.1)),
            road_thickness_m=float(rendering.get("road_thickness_m", 0.04)),
            camera_height_m=float(rendering.get("camera_height_m", 75.0)),
            elevated_level_height_m=float(rendering.get("elevated_level_height_m", 4.0)),
            surface_overrides_m={
                str(osm_id): float(height)
                for osm_id, height in rendering.get("surface_overrides_m", {}).items()
            },
            landmark=dict(raw["landmark"]) if raw.get("landmark") else None,
        )
        if not (config.bounds.south <= config.spawn_latitude <= config.bounds.north):
            raise ValueError("robot spawn latitude is outside scene bounds")
        if not (config.bounds.west <= config.spawn_longitude <= config.bounds.east):
            raise ValueError("robot spawn longitude is outside scene bounds")
        if config.elevation_rows < 2 or config.elevation_columns < 2:
            raise ValueError("elevation rows and columns must be at least 2")
        return config
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Invalid OSM scene config {path}: {exc}") from exc


def download_osm(bounds: GeoBounds, destination: Path, *, timeout_s: float = 60.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{OSM_API_URL}?{urllib.parse.urlencode({'bbox': bounds.osm_bbox})}"
    request = urllib.request.Request(url, headers={"User-Agent": "PathVLA-MuJoCo/1.0 (OSM scene builder)"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read()
    except Exception as exc:
        raise ConfigurationError(f"Unable to download OpenStreetMap bbox {bounds.osm_bbox}: {exc}") from exc
    if not payload.lstrip().startswith(b"<?xml") and not payload.lstrip().startswith(b"<osm"):
        raise ConfigurationError("OpenStreetMap returned a non-XML response")
    destination.write_bytes(payload)


def parse_osm(path: Path) -> OSMData:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ConfigurationError(f"Cannot parse OSM file {path}: {exc}") from exc
    source_bounds = None
    bounds_node = root.find("bounds")
    if bounds_node is not None:
        try:
            source_bounds = GeoBounds(
                south=float(bounds_node.get("minlat", "")),
                west=float(bounds_node.get("minlon", "")),
                north=float(bounds_node.get("maxlat", "")),
                east=float(bounds_node.get("maxlon", "")),
            )
        except ValueError:
            source_bounds = None
    nodes: dict[str, tuple[float, float]] = {}
    for node in root.findall("node"):
        node_id = node.get("id")
        if node_id is not None and node.get("lat") is not None and node.get("lon") is not None:
            nodes[node_id] = (float(node.get("lat", "0")), float(node.get("lon", "0")))
    ways: list[OSMWay] = []
    ways_by_id: dict[str, OSMWay] = {}
    for way in root.findall("way"):
        refs = [nd.get("ref") for nd in way.findall("nd")]
        coordinates = tuple(nodes[ref] for ref in refs if ref in nodes)
        if len(coordinates) < 2:
            continue
        tags = {tag.get("k", ""): tag.get("v", "") for tag in way.findall("tag") if tag.get("k")}
        parsed = OSMWay(osm_id=way.get("id", "unknown"), coordinates=coordinates, tags=tags)
        ways.append(parsed)
        ways_by_id[parsed.osm_id] = parsed
    for relation in root.findall("relation"):
        tags = {tag.get("k", ""): tag.get("v", "") for tag in relation.findall("tag") if tag.get("k")}
        if tags.get("type") != "multipolygon" or (
            tags.get("building", "no") == "no" and _surface_kind(tags) is None
        ):
            continue
        fragments = []
        for member in relation.findall("member"):
            if member.get("type") != "way" or member.get("role", "outer") != "outer":
                continue
            member_way = ways_by_id.get(member.get("ref", ""))
            if member_way is not None:
                fragments.append(member_way.coordinates)
        for index, ring in enumerate(_stitch_rings(fragments)):
            relation_id = relation.get("id", "unknown")
            relation_tags = {**tags, "pathvla:source": f"relation/{relation_id}"}
            ways.append(OSMWay(f"relation_{relation_id}_{index}", ring, relation_tags))
    return OSMData(ways=tuple(ways), node_count=len(nodes), source_bounds=source_bounds)


def _stitch_rings(
    fragments: Sequence[Sequence[tuple[float, float]]],
) -> list[tuple[tuple[float, float], ...]]:
    remaining = [list(fragment) for fragment in fragments if len(fragment) >= 2]
    rings: list[tuple[tuple[float, float], ...]] = []
    while remaining:
        ring = remaining.pop(0)
        while ring[0] != ring[-1]:
            match_index = None
            reverse = False
            for index, fragment in enumerate(remaining):
                if fragment[0] == ring[-1]:
                    match_index = index
                    break
                if fragment[-1] == ring[-1]:
                    match_index = index
                    reverse = True
                    break
            if match_index is None:
                break
            fragment = remaining.pop(match_index)
            if reverse:
                fragment.reverse()
            ring.extend(fragment[1:])
        if len(ring) >= 4 and ring[0] == ring[-1]:
            rings.append(tuple(ring))
    return rings


def fetch_usgs_elevation(latitude: float, longitude: float, *, timeout_s: float = 30.0) -> float:
    query = urllib.parse.urlencode(
        {"x": f"{longitude:.8f}", "y": f"{latitude:.8f}", "wkid": 4326, "units": "Meters", "includeDate": "false"}
    )
    request = urllib.request.Request(
        f"{USGS_EPQS_URL}?{query}",
        headers={"User-Agent": "PathVLA-MuJoCo/1.0 (terrain sampler)"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            value: Any = payload.get("value")
            if value is None:
                value = payload["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
            elevation = float(value)
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    else:
        raise ConfigurationError(
            f"USGS elevation query failed at {latitude:.7f}, {longitude:.7f}: {last_error}"
        ) from last_error
    if not math.isfinite(elevation) or elevation <= -1_000_000:
        raise ConfigurationError(f"USGS returned invalid elevation {elevation!r}")
    return elevation


def build_elevation_grid(config: SceneConfig, cache_path: Path, *, flat: bool = False) -> ElevationGrid:
    if flat or config.elevation_provider == "flat":
        values = tuple(tuple(0.0 for _ in range(config.elevation_columns)) for _ in range(config.elevation_rows))
    elif config.elevation_provider == "usgs":
        cached: list[list[float | None]] | None = None
        if cache_path.is_file():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_bounds = raw.get("bounds")
            bounds_match = cached_bounds is None or cached_bounds == config.bounds.__dict__
            if (
                int(raw.get("rows", 0)) == config.elevation_rows
                and int(raw.get("columns", 0)) == config.elevation_columns
                and bounds_match
            ):
                cached = raw.get("values_m")
                if cached is not None and cached_bounds is None:
                    _write_elevation_cache(cache_path, config, cached)
        rows: list[list[float | None]] = cached or [
            [None for _ in range(config.elevation_columns)] for _ in range(config.elevation_rows)
        ]
        for row in range(config.elevation_rows):
            latitude = config.bounds.south + row / (config.elevation_rows - 1) * (config.bounds.north - config.bounds.south)
            for column in range(config.elevation_columns):
                if rows[row][column] is not None:
                    continue
                longitude = config.bounds.west + column / (config.elevation_columns - 1) * (config.bounds.east - config.bounds.west)
                rows[row][column] = fetch_usgs_elevation(latitude, longitude)
                _write_elevation_cache(cache_path, config, rows)
                time.sleep(0.05)
        values = tuple(
            tuple(float(value) for value in row if value is not None)
            for row in rows
        )
    else:
        raise ConfigurationError(f"Unsupported elevation provider: {config.elevation_provider}")
    return ElevationGrid(config.bounds, config.elevation_rows, config.elevation_columns, values)


def _write_elevation_cache(
    cache_path: Path,
    config: SceneConfig,
    values: Sequence[Sequence[float | None]],
) -> None:
    cache_path.write_text(
        json.dumps(
            {
                "provider": "USGS EPQS",
                "bounds": config.bounds.__dict__,
                "rows": config.elevation_rows,
                "columns": config.elevation_columns,
                "values_m": values,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_heightfield_png(grid: ElevationGrid, path: Path, resolution: int = 65) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    low, high = grid.minimum, grid.maximum
    span = high - low
    width = max(grid.columns, resolution)
    height = max(grid.rows, resolution)
    rows: list[bytes] = []
    for row in range(height):
        latitude = grid.bounds.north - row / (height - 1) * (grid.bounds.north - grid.bounds.south)
        values = [
            grid.at_geo(
                latitude,
                grid.bounds.west + column / (width - 1) * (grid.bounds.east - grid.bounds.west),
            )
            for column in range(width)
        ]
        pixels = bytes(0 if span <= 1e-9 else round((value - low) / span * 255.0) for value in values)
        rows.append(b"\x00" + pixels)
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def build_mujoco_scene(
    config_path: Path,
    output_dir: Path,
    g1_xml_path: Path,
    *,
    osm_path: Path | None = None,
    refresh_osm: bool = False,
    flat_elevation: bool = False,
) -> BuildResult:
    config = load_scene_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    osm_path = osm_path or output_dir / "map.osm"
    if refresh_osm or not osm_path.is_file():
        download_osm(config.bounds, osm_path)
    osm = parse_osm(osm_path)
    if osm.source_bounds is not None and not _bounds_contain(osm.source_bounds, config.bounds):
        raise ConfigurationError(
            f"Cached OSM file {osm_path} does not cover the configured bounds; use --refresh-osm"
        )
    elevation_cache = output_dir / "elevation.json"
    elevation = build_elevation_grid(config, elevation_cache, flat=flat_elevation)
    heightfield_path = output_dir / "elevation.png"
    write_heightfield_png(elevation, heightfield_path)
    xml, spawn_xyz, spawn_quaternion, counts = build_world_xml(
        g1_xml_path,
        config,
        osm,
        elevation,
        heightfield_path,
    )
    xml_path = output_dir / "scene.xml"
    xml_path.write_text(xml, encoding="utf-8")
    projection = LocalProjection(*config.bounds.center)
    metadata_path = output_dir / "scene_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "scene": config.name,
                "source": {
                    "openstreetmap": "https://www.openstreetmap.org/copyright",
                    "elevation": "flat override" if flat_elevation else config.elevation_provider,
                },
                "bounds": config.bounds.__dict__,
                "origin": {"latitude": projection.latitude, "longitude": projection.longitude},
                "axis_convention": {"x": "east (meters)", "y": "north (meters)", "z": "up (meters)"},
                "robot_spawn": {
                    "latitude": config.spawn_latitude,
                    "longitude": config.spawn_longitude,
                    "heading_degrees_clockwise_from_north": config.spawn_heading_degrees,
                    "xyz": spawn_xyz,
                    "quaternion_wxyz": spawn_quaternion,
                },
                "elevation_m": {"minimum": elevation.minimum, "maximum": elevation.maximum},
                "osm": {"nodes": osm.node_count, "ways": len(osm.ways), "rendered_features": counts},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return BuildResult(xml_path, metadata_path, osm_path, heightfield_path, spawn_xyz, spawn_quaternion, counts)


def _bounds_contain(container: GeoBounds, requested: GeoBounds, tolerance: float = 1e-7) -> bool:
    return (
        container.south <= requested.south + tolerance
        and container.west <= requested.west + tolerance
        and container.north >= requested.north - tolerance
        and container.east >= requested.east - tolerance
    )


def build_world_xml(
    g1_xml_path: Path,
    config: SceneConfig,
    osm: OSMData,
    elevation: ElevationGrid,
    heightfield_path: Path,
) -> tuple[str, tuple[float, float, float], tuple[float, float, float, float], dict[str, int]]:
    if not g1_xml_path.is_file():
        raise ConfigurationError(f"G1 MJCF not found: {g1_xml_path}. Run 'make download-g1-mjcf'.")
    root = ET.parse(g1_xml_path).getroot()
    root.set("model", config.name)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str((g1_xml_path.parent / "assets").resolve()))
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    _add_materials(asset, heightfield_path.parent)

    projection = LocalProjection(*config.bounds.center)
    southwest = projection.to_local(config.bounds.south, config.bounds.west)
    northeast = projection.to_local(config.bounds.north, config.bounds.east)
    half_x = (northeast[0] - southwest[0]) / 2.0
    half_y = (northeast[1] - southwest[1]) / 2.0
    clip_bounds = (-half_x, -half_y, half_x, half_y)
    elevation_span = max(elevation.maximum - elevation.minimum, 0.05)
    ET.SubElement(
        asset,
        "hfield",
        name="sf_elevation",
        file=str(heightfield_path.resolve()),
        size=_numbers((half_x, half_y, elevation_span, 2.0)),
    )

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.5 0.5 0.5", ambient="0.16 0.16 0.16", specular="0.22 0.22 0.22")
    ET.SubElement(visual, "rgba", haze="0.72 0.79 0.84 1")
    # These are multipliers of model extent, not distances in meters. The
    # landmark scenes span kilometers, so a small multiplier is required for
    # close robot cameras.
    ET.SubElement(visual, "map", znear="0.0001", zfar="5")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720", azimuth="140", elevation="-25")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ConfigurationError("G1 MJCF has no worldbody")
    ET.SubElement(worldbody, "geom", name="terrain", type="hfield", hfield="sf_elevation", pos=f"0 0 {elevation.minimum:.6g}", material="terrain", friction="1.0 0.02 0.002")
    ET.SubElement(
        worldbody,
        "light",
        pos=f"{-half_x * 0.4:.6g} {-half_y * 0.3:.6g} {config.camera_height_m:.6g}",
        dir="0.2 0.2 -1",
        diffuse="0.52 0.55 0.60",
        specular="0.18 0.18 0.18",
        directional="true",
    )

    landmark_type = (config.landmark or {}).get("type")
    if landmark_type == "golden_gate_bridge":
        _add_golden_gate_bridge(asset, worldbody, projection, config, half_x, half_y)

    counts = {"roads": 0, "sidewalks": 0, "buildings": 0, "other_paths": 0, "colored_surfaces": 0}
    if landmark_type == "golden_gate_bridge":
        counts["bridge_structures"] = 1
    scene_meshes: list[
        tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], str]
    ] = []
    for way in osm.ways:
        if _is_building(way):
            mesh = _building_mesh(way, projection, elevation, config, clip_bounds)
            if mesh is not None:
                name = f"building_{_safe_id(way.osm_id)}"
                vertices, faces = mesh
                material = _building_material(asset, way)
                scene_meshes.append((name, vertices, faces, material))
                counts["buildings"] += 1
            continue
        surface_kind = _surface_kind(way.tags)
        if surface_kind is not None and _is_closed_way(way):
            mesh = _surface_mesh(way, projection, elevation, config, clip_bounds)
            if mesh is not None:
                name = f"surface_{surface_kind}_{_safe_id(way.osm_id)}"
                vertices, faces = mesh
                scene_meshes.append((name, vertices, faces, surface_kind))
                counts["colored_surfaces"] += 1
        highway = way.tags.get("highway")
        if not highway:
            continue
        if landmark_type == "golden_gate_bridge" and _is_golden_gate_way(way.tags):
            continue
        kind = _highway_kind(highway)
        width = _way_width(way.tags, highway)
        material = _path_material(kind, way.tags)
        if kind == "road":
            counts["roads"] += 1
        elif kind == "sidewalk":
            counts["sidewalks"] += 1
        else:
            counts["other_paths"] += 1
        local = [projection.to_local(lat, lon) for lat, lon in way.coordinates]
        for index, (a, b) in enumerate(zip(local, local[1:])):
            clipped = _clip_segment(a, b, clip_bounds)
            if clipped is None:
                continue
            a, b = clipped
            alat, alon = projection.to_geo(*a)
            blat, blon = projection.to_geo(*b)
            za, zb = elevation.at_geo(alat, alon), elevation.at_geo(blat, blon)
            vertical_offset = _way_vertical_offset(way, config)
            _add_way_segment(
                worldbody,
                f"{kind}_{_safe_id(way.osm_id)}_{index}",
                a,
                b,
                (za + zb) / 2.0 + vertical_offset + config.road_thickness_m / 2.0,
                width,
                config.road_thickness_m,
                material,
            )

    for name, vertices, faces, material in scene_meshes:
        ET.SubElement(asset, "mesh", name=name, vertex=_vertices(vertices), face=_faces(faces))
        ET.SubElement(worldbody, "geom", name=name, type="mesh", mesh=name, material=material, friction="0.9 0.02 0.002")

    spawn_x, spawn_y = projection.to_local(config.spawn_latitude, config.spawn_longitude)
    if config.spawn_surface_elevation_m is not None:
        spawn_surface_z = config.spawn_surface_elevation_m
    else:
        spawn_surface_z = (
            elevation.at_geo(config.spawn_latitude, config.spawn_longitude)
            + config.spawn_elevation_offset_m
        )
    spawn_z = spawn_surface_z + config.robot_base_height_m

    camera_target = ET.SubElement(worldbody, "body", name="map_center", pos=f"0 0 {elevation.minimum:.6g}", mocap="true")
    ET.SubElement(camera_target, "geom", type="sphere", size="0.01", rgba="0 0 0 0", contype="0", conaffinity="0")
    ET.SubElement(
        worldbody,
        "camera",
        name="map_overview",
        pos=_numbers((-half_x * 0.75, -half_y * 0.8, config.camera_height_m)),
        mode="targetbody",
        target="map_center",
        fovy="52",
    )
    spawn_target = ET.SubElement(
        worldbody,
        "body",
        name="spawn_focus",
        pos=_numbers((spawn_x, spawn_y, spawn_surface_z + 0.8)),
        mocap="true",
    )
    ET.SubElement(spawn_target, "geom", type="sphere", size="0.01", rgba="0 0 0 0", contype="0", conaffinity="0")
    ET.SubElement(
        worldbody,
        "camera",
        name="robot_overview",
        pos=_numbers((spawn_x - 20.0, spawn_y - 25.0, spawn_surface_z + 18.0)),
        mode="targetbody",
        target="spawn_focus",
        fovy="24",
    )
    yaw = math.radians(90.0 - config.spawn_heading_degrees)
    quaternion = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    return ET.tostring(root, encoding="unicode"), (spawn_x, spawn_y, spawn_z), quaternion, counts


def _add_materials(asset: ET.Element, texture_dir: Path) -> None:
    texture_specs = {
        "pathvla_asphalt": ((116, 120, 124), 34, 11),
        "pathvla_concrete": ((185, 180, 168), 22, 17),
        "pathvla_steel": ((178, 172, 165), 18, 23),
        "pathvla_water": ((118, 142, 156), 20, 29),
    }
    texture_paths: dict[str, Path] = {}
    for name, (base, variation, seed) in texture_specs.items():
        texture_path = texture_dir / f"{name}.png"
        _write_surface_texture(texture_path, base, variation, seed, water=name == "pathvla_water")
        texture_paths[name] = texture_path
    existing_textures = {node.get("name") for node in asset.findall("texture")}
    if "pathvla_sky" not in existing_textures:
        ET.SubElement(
            asset,
            "texture",
            name="pathvla_sky",
            type="skybox",
            builtin="gradient",
            rgb1="0.18 0.42 0.72",
            rgb2="0.88 0.93 1",
            width="512",
            height="3072",
        )
    for name, path in texture_paths.items():
        if name not in existing_textures:
            ET.SubElement(asset, "texture", name=name, type="2d", file=str(path.resolve()))
    existing = {node.get("name") for node in asset.findall("material")}
    for name, rgba, specular, texture in (
        ("terrain", "0.42 0.52 0.33 1", "0.05", None),
        ("road", "0.15 0.17 0.20 1", "0.12", "pathvla_asphalt"),
        ("sidewalk", "0.72 0.68 0.59 1", "0.08", "pathvla_concrete"),
        ("path", "0.63 0.42 0.24 1", "0.05", None),
        ("paving", "0.78 0.70 0.57 1", "0.08", "pathvla_concrete"),
        ("building", "0.65 0.54 0.43 1", "0.18", "pathvla_concrete"),
        ("building_glass", "0.35 0.66 0.82 1", "0.7", None),
        ("building_brick", "0.62 0.23 0.15 1", "0.12", None),
        ("building_concrete", "0.61 0.62 0.60 1", "0.1", "pathvla_concrete"),
        ("park", "0.18 0.52 0.20 1", "0.03", None),
        ("grass", "0.34 0.68 0.25 1", "0.02", None),
        ("garden", "0.12 0.58 0.30 1", "0.03", None),
        ("shrub", "0.08 0.38 0.16 1", "0.02", None),
        ("forest", "0.05 0.30 0.12 1", "0.02", None),
        ("water", "0.08 0.42 0.78 1", "0.65", "pathvla_water"),
        ("playground", "0.92 0.48 0.10 1", "0.05", None),
        ("plaza", "0.82 0.72 0.55 1", "0.08", "pathvla_concrete"),
        ("bridge_vermilion", "0.72 0.12 0.035 1", "0.42", "pathvla_steel"),
        ("bridge_dark_steel", "0.38 0.055 0.025 1", "0.32", "pathvla_steel"),
        ("bridge_concrete", "0.58 0.56 0.50 1", "0.12", "pathvla_concrete"),
        ("lane_white", "0.92 0.92 0.84 1", "0.08", None),
        ("lane_yellow", "0.94 0.64 0.05 1", "0.08", None),
        ("bay_water", "0.055 0.24 0.38 0.94", "0.9", "pathvla_water"),
        ("vehicle_red", "0.66 0.035 0.025 1", "0.65", "pathvla_steel"),
        ("vehicle_blue", "0.04 0.18 0.52 1", "0.65", "pathvla_steel"),
        ("vehicle_silver", "0.55 0.58 0.62 1", "0.72", "pathvla_steel"),
        ("vehicle_white", "0.86 0.86 0.82 1", "0.58", "pathvla_steel"),
    ):
        if name not in existing:
            material_node = ET.SubElement(
                asset,
                "material",
                name=name,
                rgba=rgba,
                specular=specular,
                shininess="0.25",
                emission="0.025",
            )
            if texture is not None:
                material_node.set("texture", texture)
                material_node.set("texrepeat", "12 12")
                material_node.set("texuniform", "true")
    bay_water = next((node for node in asset.findall("material") if node.get("name") == "bay_water"), None)
    if bay_water is not None:
        bay_water.set("shininess", "0.9")
        bay_water.set("reflectance", "0.28")


def _write_surface_texture(
    path: Path,
    base: tuple[int, int, int],
    variation: int,
    seed: int,
    *,
    water: bool = False,
) -> None:
    width = height = 128
    rng = random.Random(seed)
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        wave = math.sin(y / 5.5) * 8.0 if water else 0.0
        for x in range(width):
            grain = rng.uniform(-variation, variation) + wave
            if not water and rng.random() < 0.018:
                grain += rng.choice((-35.0, 30.0))
            for channel in base:
                row.append(round(min(255.0, max(0.0, channel + grain))))
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _is_golden_gate_way(tags: dict[str, str]) -> bool:
    name = tags.get("name", "")
    return (
        "Golden Gate Bridge" in name
        or tags.get("bridge:name") == "Golden Gate"
        or tags.get("man_made") == "bridge" and name == "Golden Gate Bridge"
    )


def _add_golden_gate_bridge(
    asset: ET.Element,
    worldbody: ET.Element,
    projection: LocalProjection,
    config: SceneConfig,
    half_x: float,
    half_y: float,
) -> None:
    landmark = config.landmark or {}

    def geo_point(key: str) -> tuple[float, float]:
        raw = landmark[key]
        return projection.to_local(float(raw["latitude"]), float(raw["longitude"]))

    south_anchor = geo_point("south_anchor")
    south_tower = geo_point("south_tower")
    north_tower = geo_point("north_tower")
    north_anchor = geo_point("north_anchor")
    deck_z = float(landmark.get("deck_elevation_m", 67.0))
    deck_width = float(landmark.get("deck_width_m", 27.0))
    tower_top_z = float(landmark.get("tower_top_elevation_m", 227.0))
    cable_mid_z = float(landmark.get("main_cable_mid_elevation_m", 82.0))
    water_z = float(landmark.get("water_elevation_m", 0.0))

    dx = north_anchor[0] - south_anchor[0]
    dy = north_anchor[1] - south_anchor[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    vx, vy = -uy, ux
    yaw = math.atan2(uy, ux)

    ET.SubElement(
        worldbody,
        "geom",
        name="golden_gate_bay",
        type="box",
        pos=_numbers((0.0, 0.0, water_z - 0.18)),
        size=_numbers((half_x, half_y, 0.3)),
        material="bay_water",
        contype="0",
        conaffinity="0",
    )

    _bridge_box(
        worldbody,
        "golden_gate_deck",
        _midpoint(south_anchor, north_anchor),
        deck_z - 1.4,
        (length / 2.0, deck_width / 2.0, 1.4),
        yaw,
        "bridge_dark_steel",
    )
    _bridge_strip(worldbody, "golden_gate_asphalt", south_anchor, north_anchor, deck_z + 0.045, 19.2, 0.09, "road")
    for side in (-1.0, 1.0):
        offset = side * 11.65
        start = (south_anchor[0] + vx * offset, south_anchor[1] + vy * offset)
        end = (north_anchor[0] + vx * offset, north_anchor[1] + vy * offset)
        _bridge_strip(worldbody, f"golden_gate_sidewalk_{'east' if side > 0 else 'west'}", start, end, deck_z + 0.10, 3.1, 0.16, "bridge_concrete")
        _add_capsule(
            worldbody,
            f"golden_gate_guardrail_{'east' if side > 0 else 'west'}",
            (start[0], start[1], deck_z + 1.25),
            (end[0], end[1], deck_z + 1.25),
            0.10,
            "bridge_vermilion",
        )

    for index, offset in enumerate((-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0)):
        start = (south_anchor[0] + vx * offset, south_anchor[1] + vy * offset)
        end = (north_anchor[0] + vx * offset, north_anchor[1] + vy * offset)
        material = "lane_yellow" if index in {0, 3, 6} else "lane_white"
        _bridge_strip(worldbody, f"golden_gate_lane_{index}", start, end, deck_z + 0.155, 0.13, 0.025, material)

    for tower_name, tower in (("south", south_tower), ("north", north_tower)):
        for side in (-1.0, 1.0):
            center = (tower[0] + vx * side * 15.0, tower[1] + vy * side * 15.0)
            _bridge_box(
                worldbody,
                f"golden_gate_{tower_name}_tower_leg_{'east' if side > 0 else 'west'}",
                center,
                (tower_top_z - 10.0) / 2.0,
                (5.0, 3.2, (tower_top_z + 10.0) / 2.0),
                yaw,
                "bridge_vermilion",
            )
        for level_index, z in enumerate((deck_z + 13.0, 139.0, 203.0)):
            a = (tower[0] - vx * 18.2, tower[1] - vy * 18.2)
            b = (tower[0] + vx * 18.2, tower[1] + vy * 18.2)
            _bridge_strip(
                worldbody,
                f"golden_gate_{tower_name}_crossbeam_{level_index}",
                a,
                b,
                z,
                6.0,
                4.0,
                "bridge_vermilion",
            )

    cable_radius = float(landmark.get("main_cable_radius_m", 0.46))
    cable_sides = (-deck_width / 2.0 - 1.7, deck_width / 2.0 + 1.7)
    south_anchor_cable_z = float(landmark.get("anchor_cable_elevation_m", 92.0))
    north_anchor_cable_z = south_anchor_cable_z
    cable_sections = (
        (south_anchor, south_tower, lambda t: south_anchor_cable_z + (tower_top_z - south_anchor_cable_z) * t * t),
        (
            south_tower,
            north_tower,
            lambda t: cable_mid_z + 4.0 * (tower_top_z - cable_mid_z) * (t - 0.5) ** 2,
        ),
        (north_tower, north_anchor, lambda t: north_anchor_cable_z + (tower_top_z - north_anchor_cable_z) * (1.0 - t) ** 2),
    )
    for side_index, offset in enumerate(cable_sides):
        cable_points: list[tuple[float, float, float]] = []
        for section_index, (start, end, z_at) in enumerate(cable_sections):
            for sample in range(25):
                if section_index and sample == 0:
                    continue
                t = sample / 24.0
                x = start[0] + (end[0] - start[0]) * t + vx * offset
                y = start[1] + (end[1] - start[1]) * t + vy * offset
                cable_points.append((x, y, z_at(t)))
        for index, (start, end) in enumerate(zip(cable_points, cable_points[1:])):
            _add_capsule(
                worldbody,
                f"golden_gate_main_cable_{side_index}_{index}",
                start,
                end,
                cable_radius,
                "bridge_vermilion",
            )

        for span_index, (start, end, z_at) in enumerate(cable_sections):
            count = max(4, round(math.dist(start, end) / 32.0))
            for hanger_index in range(1, count):
                t = hanger_index / count
                cable_z = z_at(t)
                if cable_z <= deck_z + 3.0:
                    continue
                x = start[0] + (end[0] - start[0]) * t + vx * offset
                y = start[1] + (end[1] - start[1]) * t + vy * offset
                _add_capsule(
                    worldbody,
                    f"golden_gate_hanger_{side_index}_{span_index}_{hanger_index}",
                    (x, y, deck_z + 0.8),
                    (x, y, cable_z),
                    0.095,
                    "bridge_vermilion",
                )

    truss_count = max(12, round(length / 38.0))
    for side_index, offset in enumerate((-deck_width / 2.0, deck_width / 2.0)):
        previous_top = None
        previous_bottom = None
        for index in range(truss_count + 1):
            t = index / truss_count
            x = south_anchor[0] + dx * t + vx * offset
            y = south_anchor[1] + dy * t + vy * offset
            top = (x, y, deck_z - 0.2)
            bottom = (x, y, deck_z - 7.2)
            _add_capsule(worldbody, f"golden_gate_truss_v_{side_index}_{index}", top, bottom, 0.13, "bridge_dark_steel")
            if previous_top is not None and previous_bottom is not None:
                _add_capsule(worldbody, f"golden_gate_truss_t_{side_index}_{index}", previous_top, top, 0.15, "bridge_dark_steel")
                _add_capsule(worldbody, f"golden_gate_truss_b_{side_index}_{index}", previous_bottom, bottom, 0.15, "bridge_dark_steel")
                diagonal_start = previous_bottom if index % 2 else previous_top
                diagonal_end = top if index % 2 else bottom
                _add_capsule(worldbody, f"golden_gate_truss_d_{side_index}_{index}", diagonal_start, diagonal_end, 0.12, "bridge_dark_steel")
            previous_top, previous_bottom = top, bottom

    vehicle_materials = ("vehicle_red", "vehicle_blue", "vehicle_silver", "vehicle_white")
    lane_offsets = (-7.5, -4.5, -1.5, 1.5, 4.5, 7.5)
    for index in range(18):
        t = 0.06 + index / 17.0 * 0.88
        lane_offset = lane_offsets[(index * 5) % len(lane_offsets)]
        center = (
            south_anchor[0] + dx * t + vx * lane_offset,
            south_anchor[1] + dy * t + vy * lane_offset,
        )
        material = vehicle_materials[index % len(vehicle_materials)]
        _bridge_box(worldbody, f"bridge_vehicle_{index}_body", center, deck_z + 0.72, (2.25, 0.92, 0.62), yaw, material)
        _bridge_box(worldbody, f"bridge_vehicle_{index}_cabin", center, deck_z + 1.55, (1.12, 0.78, 0.35), yaw, "building_glass")

    lamp_count = max(12, round(length / 90.0))
    for side_index, offset in enumerate((-12.4, 12.4)):
        for index in range(lamp_count + 1):
            t = index / lamp_count
            x = south_anchor[0] + dx * t + vx * offset
            y = south_anchor[1] + dy * t + vy * offset
            _add_capsule(
                worldbody,
                f"golden_gate_lamp_post_{side_index}_{index}",
                (x, y, deck_z + 0.2),
                (x, y, deck_z + 7.7),
                0.075,
                "bridge_vermilion",
            )
            ET.SubElement(
                worldbody,
                "geom",
                name=f"golden_gate_lamp_{side_index}_{index}",
                type="sphere",
                pos=_numbers((x, y, deck_z + 7.9)),
                size="0.16",
                material="lane_white",
                contype="0",
                conaffinity="0",
            )

    bridge_center = _midpoint(south_tower, north_tower)
    target = ET.SubElement(
        worldbody,
        "body",
        name="golden_gate_camera_target",
        pos=_numbers((*bridge_center, deck_z + 18.0)),
        mocap="true",
    )
    ET.SubElement(target, "geom", type="sphere", size="0.01", rgba="0 0 0 0", contype="0", conaffinity="0")
    camera_offset = -850.0
    ET.SubElement(
        worldbody,
        "camera",
        name="landmark_overview",
        pos=_numbers(
            (
                bridge_center[0] + vx * camera_offset - ux * 120.0,
                bridge_center[1] + vy * camera_offset - uy * 120.0,
                deck_z + 250.0,
            )
        ),
        mode="targetbody",
        target="golden_gate_camera_target",
        fovy="75",
    )
    robot_xy = projection.to_local(config.spawn_latitude, config.spawn_longitude)
    deck_target_xy = (robot_xy[0] + ux * 85.0, robot_xy[1] + uy * 85.0)
    deck_target = ET.SubElement(
        worldbody,
        "body",
        name="golden_gate_deck_camera_target",
        pos=_numbers((*deck_target_xy, deck_z + 1.4)),
        mocap="true",
    )
    ET.SubElement(deck_target, "geom", type="sphere", size="0.01", rgba="0 0 0 0", contype="0", conaffinity="0")
    ET.SubElement(
        worldbody,
        "camera",
        name="bridge_deck_view",
        pos=_numbers(
            (
                robot_xy[0] - ux * 16.0 + vx * 4.0,
                robot_xy[1] - uy * 16.0 + vy * 4.0,
                deck_z + 6.2,
            )
        ),
        mode="targetbody",
        target="golden_gate_deck_camera_target",
        fovy="62",
    )
    robot_target = ET.SubElement(
        worldbody,
        "body",
        name="golden_gate_robot_camera_target",
        pos=_numbers((*robot_xy, deck_z + 1.0)),
        mocap="true",
    )
    ET.SubElement(robot_target, "geom", type="sphere", size="0.01", rgba="0 0 0 0", contype="0", conaffinity="0")
    robot_camera_position = (
        robot_xy[0] - ux * 7.0 + vx * 4.5,
        robot_xy[1] - uy * 7.0 + vy * 4.5,
        deck_z + 4.4,
    )
    robot_camera_target = (robot_xy[0], robot_xy[1], deck_z + 1.0)
    ET.SubElement(
        worldbody,
        "camera",
        name="bridge_robot_view",
        pos=_numbers(robot_camera_position),
        xyaxes=_camera_xyaxes(robot_camera_position, robot_camera_target),
        fovy="52",
    )


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _camera_xyaxes(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> str:
    forward = tuple(target[index] - position[index] for index in range(3))
    forward_length = math.sqrt(sum(value * value for value in forward))
    forward = tuple(value / forward_length for value in forward)
    z_axis = tuple(-value for value in forward)
    x_axis = (-z_axis[1], z_axis[0], 0.0)
    x_length = math.hypot(x_axis[0], x_axis[1])
    if x_length < 1e-9:
        x_axis = (1.0, 0.0, 0.0)
    else:
        x_axis = tuple(value / x_length for value in x_axis)
    y_axis = (
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    )
    return _numbers((*x_axis, *y_axis))


def _bridge_box(
    worldbody: ET.Element,
    name: str,
    center: tuple[float, float],
    z: float,
    size: tuple[float, float, float],
    yaw: float,
    material: str,
) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        name=name,
        type="box",
        pos=_numbers((*center, z)),
        size=_numbers(size),
        quat=_numbers((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))),
        material=material,
        friction="0.9 0.02 0.002",
    )


def _bridge_strip(
    worldbody: ET.Element,
    name: str,
    start: tuple[float, float],
    end: tuple[float, float],
    z: float,
    width: float,
    thickness: float,
    material: str,
) -> None:
    _add_way_segment(worldbody, name, start, end, z, width, thickness, material)


def _add_capsule(
    worldbody: ET.Element,
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    material: str,
) -> None:
    if math.dist(start, end) < 0.02:
        return
    ET.SubElement(
        worldbody,
        "geom",
        name=name,
        type="capsule",
        fromto=_numbers((*start, *end)),
        size=f"{radius:.6g}",
        material=material,
        contype="0",
        conaffinity="0",
    )


def _building_material(asset: ET.Element, way: OSMWay) -> str:
    color = _parse_color(way.tags.get("building:colour") or way.tags.get("building:color"))
    if color is not None:
        name = f"building_color_{_safe_id(way.osm_id)}"
        ET.SubElement(
            asset,
            "material",
            name=name,
            rgba=_numbers((*color, 1.0)),
            specular="0.3",
            shininess="0.35",
            emission="0.02",
        )
        return name
    material = way.tags.get("building:material", "").lower()
    if material in {"glass", "mirror"}:
        return "building_glass"
    if material in {"brick", "masonry"}:
        return "building_brick"
    if material in {"concrete", "cement_block"}:
        return "building_concrete"
    return "building"


def _parse_color(value: str | None) -> tuple[float, float, float] | None:
    if not value:
        return None
    colors = {
        "white": "#ffffff",
        "gray": "#808080",
        "grey": "#808080",
        "black": "#202020",
        "red": "#c0392b",
        "brown": "#795548",
        "beige": "#d7c7a5",
        "blue": "#3979b9",
        "green": "#4f8a4b",
    }
    cleaned = colors.get(value.strip().lower(), value.strip())
    if cleaned.startswith("#") and len(cleaned) == 7:
        try:
            rgb = tuple(int(cleaned[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
        except ValueError:
            return None
        return tuple(min(0.92, channel * 0.84 + 0.08) for channel in rgb)
    return None


def _path_material(kind: str, tags: dict[str, str]) -> str:
    surface = tags.get("surface", "").lower()
    if surface in {"paving_stones", "sett", "cobblestone", "concrete", "concrete:plates"}:
        return "paving"
    if surface in {"dirt", "earth", "ground", "fine_gravel", "gravel", "woodchips"}:
        return "path"
    if kind == "road":
        return "road"
    return "sidewalk"


def _surface_kind(tags: dict[str, str]) -> str | None:
    leisure = tags.get("leisure", "")
    landuse = tags.get("landuse", "")
    natural = tags.get("natural", "")
    if tags.get("water") or tags.get("waterway") == "riverbank" or natural == "water":
        return "water"
    if leisure == "playground":
        return "playground"
    if leisure in {"garden"}:
        return "garden"
    if leisure in {"park", "common", "recreation_ground"}:
        return "park"
    if landuse in {"grass", "meadow"} or natural in {"grassland", "heath"}:
        return "grass"
    if natural in {"shrubbery", "scrub"}:
        return "shrub"
    if landuse == "forest" or natural == "wood":
        return "forest"
    if tags.get("amenity") in {"fountain"}:
        return "water"
    if tags.get("area") == "yes" and tags.get("highway") == "pedestrian":
        return "plaza"
    return None


def _is_closed_way(way: OSMWay) -> bool:
    return len(way.coordinates) >= 4 and way.coordinates[0] == way.coordinates[-1]


def _is_building(way: OSMWay) -> bool:
    return way.tags.get("building", "no") != "no" and _is_closed_way(way)


def _surface_mesh(
    way: OSMWay,
    projection: LocalProjection,
    elevation: ElevationGrid,
    config: SceneConfig,
    clip_bounds: tuple[float, float, float, float],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None:
    points = [projection.to_local(lat, lon) for lat, lon in way.coordinates[:-1]]
    points = _clip_polygon(points, clip_bounds)
    if len(points) < 3:
        return None
    if _signed_area(points) < 0:
        points.reverse()
    top_faces = _triangulate(points)
    if not top_faces:
        return None
    vertical_offset = _way_vertical_offset(way, config)
    thickness = 0.08
    top_z = [
        elevation.at_geo(*projection.to_geo(x, y)) + vertical_offset + 0.025
        for x, y in points
    ]
    count = len(points)
    vertices = [
        (x, y, z - thickness) for (x, y), z in zip(points, top_z)
    ] + [
        (x, y, z) for (x, y), z in zip(points, top_z)
    ]
    faces: list[tuple[int, int, int]] = []
    faces.extend((c, b, a) for a, b, c in top_faces)
    faces.extend((a + count, b + count, c + count) for a, b, c in top_faces)
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count))
        faces.append((index, next_index + count, index + count))
    return vertices, faces


def _way_vertical_offset(way: OSMWay, config: SceneConfig) -> float:
    override = config.surface_overrides_m.get(way.osm_id)
    if override is not None:
        return override
    levels = _parse_levels(way.tags.get("level"))
    if levels:
        return max(levels) * config.elevated_level_height_m
    try:
        layer = float(way.tags.get("layer", "0"))
    except ValueError:
        layer = 0.0
    return layer * 3.0 if layer > 0 else 0.0


def _parse_levels(value: str | None) -> list[float]:
    if not value:
        return []
    result = []
    for part in value.replace(",", ";").split(";"):
        try:
            result.append(float(part.strip()))
        except ValueError:
            continue
    return result


def _building_mesh(
    way: OSMWay,
    projection: LocalProjection,
    elevation: ElevationGrid,
    config: SceneConfig,
    clip_bounds: tuple[float, float, float, float],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None:
    points = [projection.to_local(lat, lon) for lat, lon in way.coordinates[:-1]]
    points = _clip_polygon(points, clip_bounds)
    if len(points) < 3:
        return None
    if _signed_area(points) < 0:
        points.reverse()
    try:
        height = _parse_distance(way.tags.get("height"))
    except ValueError:
        height = None
    if height is None or height <= 0.1:
        try:
            levels = float(way.tags.get("building:levels", ""))
        except ValueError:
            levels = 0.0
        height = levels * config.level_height_m if levels > 0 else config.default_building_height_m
    base = min(elevation.at_geo(*projection.to_geo(x, y)) for x, y in points)
    count = len(points)
    vertices = [(x, y, base) for x, y in points] + [(x, y, base + height) for x, y in points]
    top_faces = _triangulate(points)
    if not top_faces:
        return None
    faces: list[tuple[int, int, int]] = []
    faces.extend((c, b, a) for a, b, c in top_faces)
    faces.extend((a + count, b + count, c + count) for a, b, c in top_faces)
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count))
        faces.append((index, next_index + count, index + count))
    return vertices, faces


def _clip_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a line segment to a local XY rectangle using Liang-Barsky."""

    min_x, min_y, max_x, max_y = bounds
    dx, dy = end[0] - start[0], end[1] - start[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, start[0] - min_x), (dx, max_x - start[0]), (-dy, start[1] - min_y), (dy, max_y - start[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            t0 = max(t0, ratio)
        else:
            t1 = min(t1, ratio)
        if t0 > t1:
            return None
    return (
        (start[0] + t0 * dx, start[1] + t0 * dy),
        (start[0] + t1 * dx, start[1] + t1 * dy),
    )


def _clip_polygon(
    points: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Clip a polygon to a local XY rectangle using Sutherland-Hodgman."""

    min_x, min_y, max_x, max_y = bounds
    result = list(points)

    def clip_edge(
        polygon: list[tuple[float, float]],
        inside,
        intersection,
    ) -> list[tuple[float, float]]:
        if not polygon:
            return []
        output: list[tuple[float, float]] = []
        previous = polygon[-1]
        previous_inside = inside(previous)
        for current in polygon:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous, previous_inside = current, current_inside
        return output

    def vertical(a: tuple[float, float], b: tuple[float, float], x: float) -> tuple[float, float]:
        t = 0.0 if abs(b[0] - a[0]) < 1e-12 else (x - a[0]) / (b[0] - a[0])
        return x, a[1] + t * (b[1] - a[1])

    def horizontal(a: tuple[float, float], b: tuple[float, float], y: float) -> tuple[float, float]:
        t = 0.0 if abs(b[1] - a[1]) < 1e-12 else (y - a[1]) / (b[1] - a[1])
        return a[0] + t * (b[0] - a[0]), y

    result = clip_edge(result, lambda p: p[0] >= min_x, lambda a, b: vertical(a, b, min_x))
    result = clip_edge(result, lambda p: p[0] <= max_x, lambda a, b: vertical(a, b, max_x))
    result = clip_edge(result, lambda p: p[1] >= min_y, lambda a, b: horizontal(a, b, min_y))
    return clip_edge(result, lambda p: p[1] <= max_y, lambda a, b: horizontal(a, b, max_y))


def _triangulate(points: Sequence[tuple[float, float]]) -> list[tuple[int, int, int]]:
    remaining = list(range(len(points)))
    faces: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3 and guard < len(points) * len(points):
        ear_found = False
        for offset, current in enumerate(remaining):
            previous = remaining[offset - 1]
            following = remaining[(offset + 1) % len(remaining)]
            a, b, c = points[previous], points[current], points[following]
            if _cross(a, b, c) <= 1e-9:
                continue
            if any(
                _point_in_triangle(points[index], a, b, c)
                for index in remaining
                if index not in {previous, current, following}
            ):
                continue
            faces.append((previous, current, following))
            del remaining[offset]
            ear_found = True
            break
        if not ear_found:
            return []
        guard += 1
    if len(remaining) == 3:
        faces.append(tuple(remaining))
    return faces


def _signed_area(points: Sequence[tuple[float, float]]) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1])) / 2.0


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_in_triangle(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    c1, c2, c3 = _cross(a, b, point), _cross(b, c, point), _cross(c, a, point)
    return c1 >= -1e-9 and c2 >= -1e-9 and c3 >= -1e-9


def _add_way_segment(
    worldbody: ET.Element,
    name: str,
    start: tuple[float, float],
    end: tuple[float, float],
    z: float,
    width: float,
    thickness: float,
    material: str,
) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 0.05:
        return
    yaw = math.atan2(dy, dx)
    ET.SubElement(
        worldbody,
        "geom",
        name=name,
        type="box",
        pos=_numbers(((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, z)),
        size=_numbers((length / 2.0, width / 2.0, thickness / 2.0)),
        quat=_numbers((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))),
        material=material,
        friction="1.0 0.02 0.002",
    )


def _highway_kind(highway: str) -> str:
    if highway in {"footway", "pedestrian", "sidewalk", "platform"}:
        return "sidewalk"
    if highway in {"path", "steps", "cycleway", "bridleway", "track"}:
        return "path"
    return "road"


def _way_width(tags: dict[str, str], highway: str) -> float:
    try:
        explicit = _parse_distance(tags.get("width"))
    except ValueError:
        explicit = None
    if explicit is not None and explicit > 0:
        return min(explicit, 30.0)
    if highway in {"footway", "sidewalk", "path"}:
        return 1.8
    if highway in {"pedestrian", "platform"}:
        return 4.0
    if highway == "steps":
        return 2.0
    try:
        lanes = max(float(tags.get("lanes", "0")), 0.0)
    except ValueError:
        lanes = 0.0
    if lanes:
        return min(max(lanes * 3.1, 3.0), 24.0)
    return {
        "motorway": 12.0,
        "trunk": 10.0,
        "primary": 9.0,
        "secondary": 8.0,
        "tertiary": 7.0,
        "residential": 6.0,
        "service": 4.0,
    }.get(highway, 4.0)


def _parse_distance(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    cleaned = value.strip().lower()
    if cleaned.endswith("ft"):
        return float(cleaned[:-2].strip()) * 0.3048
    if cleaned.endswith("m"):
        cleaned = cleaned[:-1].strip()
    return float(cleaned)


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character == "_" else "_" for character in value)


def _numbers(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _vertices(values: Iterable[tuple[float, float, float]]) -> str:
    return " ".join(_numbers(value) for value in values)


def _faces(values: Iterable[tuple[int, int, int]]) -> str:
    return " ".join(" ".join(str(index) for index in value) for value in values)


def run_scene(result: BuildResult, *, headless: bool, seconds: float) -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise ConfigurationError("Install requirements-mac.txt to run the MuJoCo scene") from exc
    model = mujoco.MjModel.from_xml_path(str(result.xml_path))
    data = mujoco.MjData(model)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_id < 0:
        raise ConfigurationError("The G1 model has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    data.ctrl[:] = model.key_ctrl[stand_id]
    base = data.joint("floating_base_joint").qpos
    base[:3] = result.spawn_xyz
    base[3:7] = result.spawn_quaternion
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if headless:
        steps = max(1, round(seconds / model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(model, data)
        return
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as viewer:
        robot_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "bridge_robot_view")
        if robot_camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = robot_camera_id
        else:
            viewer.cam.lookat[:] = result.spawn_xyz
            viewer.cam.lookat[2] -= 0.7
            viewer.cam.distance = 18.0
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = -35.0
        started = time.monotonic()
        while viewer.is_running() and (seconds <= 0 or time.monotonic() - started < seconds):
            mujoco.mj_step(model, data)
            viewer.sync()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and run a georeferenced OpenStreetMap scene in MuJoCo")
    parser.add_argument("--config", type=Path, default=Path("config/osm_sf_golden_gate_bridge.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mujoco_sf_golden_gate_bridge"))
    parser.add_argument("--osm", type=Path, help="Use an existing .osm file instead of output-dir/map.osm")
    parser.add_argument(
        "--g1-xml",
        type=Path,
        default=Path("assets/mujoco_menagerie/unitree_g1/g1_with_hands.xml"),
    )
    parser.add_argument("--refresh-osm", action="store_true", help="Download the configured bbox again")
    parser.add_argument("--flat-elevation", action="store_true", help="Explicitly replace USGS terrain with a flat heightfield")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0, help="Viewer duration; zero runs until closed")
    args = parser.parse_args(argv)
    result = build_mujoco_scene(
        args.config,
        args.output_dir,
        args.g1_xml,
        osm_path=args.osm,
        refresh_osm=args.refresh_osm,
        flat_elevation=args.flat_elevation,
    )
    print(f"MuJoCo scene: {result.xml_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Rendered features: {result.feature_counts}")
    if not args.build_only:
        run_scene(result, headless=args.headless, seconds=args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
