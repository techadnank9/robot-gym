"""Load a converted room scan into MuJoCo with the Unitree G1 standing in it.

The scan is a photogrammetry mesh, so it is attached as visual geometry only. MuJoCo
collides mesh geoms as their convex hull, which for a room mesh is a solid block that
would trap the robot; collision is therefore provided by a floor plane and four wall
boxes fitted to the scan bounds.

    .venv-mac/bin/mjpython -m pathvla.library_mujoco
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from pathvla.errors import ConfigurationError

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCAN_DIR = REPOSITORY_ROOT / "assets" / "library_scan"
DEFAULT_G1_XML = REPOSITORY_ROOT / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "outputs" / "library_scene.xml")
    parser.add_argument("--spawn", type=float, nargs=3, default=None, help="Robot spawn x y z")
    parser.add_argument("--no-walls", action="store_true", help="Skip the fitted collision walls")
    parser.add_argument("--screenshot", type=Path, help="Render a still to this path instead of opening the viewer")
    parser.add_argument("--video", type=Path, help="Render an orbiting video of the simulation instead of opening the viewer")
    parser.add_argument("--seconds", type=float, default=8.0, help="Video duration")
    parser.add_argument("--build-only", action="store_true", help="Write the MJCF and exit")
    return parser.parse_args()


def _numbers(values) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def load_manifest(scan_dir: Path) -> dict:
    manifest_path = scan_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ConfigurationError(
            f"Scan manifest not found: {manifest_path}. "
            "Run scripts/convert_library_scan.py under Blender first."
        )
    return json.loads(manifest_path.read_text())


def build_world_xml(g1_xml_path: Path, scan_dir: Path, manifest: dict, spawn, add_walls: bool) -> str:
    if not g1_xml_path.is_file():
        raise ConfigurationError(f"G1 MJCF not found: {g1_xml_path}. Run 'make download-g1-mjcf'.")

    root = ET.parse(g1_xml_path).getroot()
    root.set("model", "g1_library_scan")

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    # The G1 meshes resolve through meshdir; scan assets are referenced absolutely.
    compiler.set("meshdir", str((g1_xml_path.parent / "assets").resolve()))

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.28 0.31 0.36",
        rgb2="0.05 0.06 0.08",
        width="512",
        height="3072",
    )
    ET.SubElement(asset, "material", name="library_floor", rgba="0.35 0.33 0.31 1", reflectance="0.05")
    ET.SubElement(asset, "material", name="library_wall", rgba="0.4 0.4 0.42 1")

    for part in manifest["parts"]:
        mesh_name = part["name"]
        ET.SubElement(asset, "mesh", name=mesh_name, file=str((scan_dir / part["obj"]).resolve()))
        if part.get("texture"):
            ET.SubElement(
                asset,
                "texture",
                type="2d",
                name=f"{mesh_name}_tex",
                file=str((scan_dir / "textures" / part["texture"]).resolve()),
            )
            ET.SubElement(
                asset,
                "material",
                name=f"{mesh_name}_mat",
                texture=f"{mesh_name}_tex",
                specular="0.1",
                shininess="0.1",
            )
        else:
            ET.SubElement(asset, "material", name=f"{mesh_name}_mat", rgba=_numbers(part["rgba"]))

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.45 0.45 0.45", ambient="0.35 0.35 0.35", specular="0.1 0.1 0.1")
    ET.SubElement(visual, "map", znear="0.01", zfar="60")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080", azimuth="140", elevation="-15")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ConfigurationError("G1 MJCF has no worldbody")

    minimum = manifest["bounds_min"]
    maximum = manifest["bounds_max"]
    half_x = (maximum[0] - minimum[0]) / 2.0
    half_y = (maximum[1] - minimum[1]) / 2.0
    height = maximum[2]

    # Visual-only scan geometry: group 2 matches the G1's own visual class.
    for part in manifest["parts"]:
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{part['name']}_visual",
            type="mesh",
            mesh=part["name"],
            material=f"{part['name']}_mat",
            contype="0",
            conaffinity="0",
            density="0",
            group="2",
        )

    # Collision proxies. The plane carries the robot; the boxes keep it in the room.
    ET.SubElement(
        worldbody,
        "geom",
        name="library_floor",
        type="plane",
        size=f"{half_x + 1:.6g} {half_y + 1:.6g} 0.05",
        material="library_floor",
        friction="1.0 0.02 0.002",
        group="3",
    )
    if add_walls:
        thickness = 0.1
        walls = {
            "wall_x_min": ((minimum[0] - thickness, 0.0, height / 2.0), (thickness, half_y, height / 2.0)),
            "wall_x_max": ((maximum[0] + thickness, 0.0, height / 2.0), (thickness, half_y, height / 2.0)),
            "wall_y_min": ((0.0, minimum[1] - thickness, height / 2.0), (half_x, thickness, height / 2.0)),
            "wall_y_max": ((0.0, maximum[1] + thickness, height / 2.0), (half_x, thickness, height / 2.0)),
        }
        for name, (position, size) in walls.items():
            ET.SubElement(
                worldbody,
                "geom",
                name=name,
                type="box",
                pos=_numbers(position),
                size=_numbers(size),
                material="library_wall",
                group="3",
                rgba="0.4 0.4 0.42 0",
            )

    ET.SubElement(
        worldbody,
        "light",
        pos=f"0 0 {max(height - 0.3, 1.0):.6g}",
        dir="0 0 -1",
        diffuse="0.6 0.6 0.58",
        specular="0.1 0.1 0.1",
        directional="true",
    )

    if spawn is not None:
        keyframe = root.find("keyframe")
        if keyframe is not None:
            key = keyframe.find("key")
            if key is not None and key.get("qpos"):
                values = key.get("qpos").split()
                values[0:3] = [f"{value:.6g}" for value in spawn]
                key.set("qpos", " ".join(values))

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir.expanduser().resolve()
    manifest = load_manifest(scan_dir)

    xml = build_world_xml(args.g1_xml, scan_dir, manifest, args.spawn, not args.no_walls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml)
    print(f"wrote {args.output}")
    print(f"room {_numbers(manifest['dimensions'])} m, {len(manifest['parts'])} scan parts")

    if args.build_only:
        return

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.output))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print(f"compiled: {model.ngeom} geoms, {model.nmesh} meshes, {model.ntex} textures, {model.nq} dof")

    if args.screenshot:
        import imageio

        renderer = mujoco.Renderer(model, height=1080, width=1920)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance = max(manifest["dimensions"][0], manifest["dimensions"][1]) * 0.8
        camera.azimuth = 140
        camera.elevation = -15
        camera.lookat[:] = [0.0, 0.0, 1.0]
        renderer.update_scene(data, camera)
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(args.screenshot, renderer.render())
        print(f"wrote {args.screenshot}")
        return

    if args.video:
        import imageio

        fps = 30
        renderer = mujoco.Renderer(model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance = max(manifest["dimensions"][0], manifest["dimensions"][1]) * 0.55
        camera.elevation = -12
        camera.lookat[:] = [0.0, 0.0, 1.0]

        args.video.parent.mkdir(parents=True, exist_ok=True)
        frames = int(args.seconds * fps)
        steps_per_frame = max(int(round(1.0 / (fps * model.opt.timestep))), 1)
        with imageio.get_writer(args.video, fps=fps, macro_block_size=None) as writer:
            for frame in range(frames):
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                camera.azimuth = 110 + 60.0 * frame / max(frames - 1, 1)
                renderer.update_scene(data, camera)
                writer.append_data(renderer.render())
        print(f"wrote {args.video} ({frames} frames, base z={data.qpos[2]:.3f})")
        return

    import time

    import mujoco.viewer

    # The blocking viewer.launch() path throws inside the Simulate constructor on this
    # macOS/Python build; launch_passive drives the window from the caller and works.
    print("opening viewer - close the window to exit")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
