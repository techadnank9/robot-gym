from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--material-source-fbx",
        type=Path,
        help="Optional ASCII FBX used to recover exact diffuse color factors.",
    )
    parser.add_argument(
        "--texture-root",
        type=Path,
        help="Directory containing exported V-BLDR textures; searched recursively.",
    )
    parser.add_argument(
        "--room-width-m",
        type=float,
        default=17.0,
        help="Width of the imported interior after conversion (default: 17m).",
    )
    return parser.parse_args(sys.argv[separator + 1 :])


def world_bounds(objects: list[bpy.types.Object]) -> tuple[list[float], list[float]]:
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        for corner in obj.bound_box
    ]
    return (
        [min(point[index] for point in points) for index in range(3)],
        [max(point[index] for point in points) for index in range(3)],
    )


def fbx_material_colors(path: Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    source = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r'Material:\s+\d+,\s+"Material::(?P<name>[^"]+)"[\s\S]*?'
        r'P:\s+"DiffuseColor"[^\r\n]*?,'
        r"(?P<red>[-+0-9.eE]+),(?P<green>[-+0-9.eE]+),(?P<blue>[-+0-9.eE]+)"
        r"(?=[\r\n])"
    )
    return {
        match.group("name"): [
            float(match.group("red")),
            float(match.group("green")),
            float(match.group("blue")),
            1.0,
        ]
        for match in pattern.finditer(source)
    }


def material_rgba(
    material: bpy.types.Material | None,
    overrides: dict[str, list[float]],
) -> list[float]:
    if material is None:
        return [0.42, 0.44, 0.48, 1.0]
    if material.name in overrides:
        return [
            round(max(0.0, min(1.0, float(value))), 6)
            for value in overrides[material.name]
        ]
    color = list(material.diffuse_color)
    if material.use_nodes and material.node_tree:
        principled = next(
            (
                node
                for node in material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if principled is not None:
            color = list(principled.inputs["Base Color"].default_value)
            color[3] = float(principled.inputs["Alpha"].default_value)
    return [round(max(0.0, min(1.0, float(value))), 6) for value in color]


def base_color_image(material: bpy.types.Material | None) -> bpy.types.Image | None:
    if material is None or not material.use_nodes or not material.node_tree:
        return None
    principled = next(
        (
            node
            for node in material.node_tree.nodes
            if node.type == "BSDF_PRINCIPLED"
        ),
        None,
    )
    if principled is not None:
        links = principled.inputs["Base Color"].links
        if links and links[0].from_node.type == "TEX_IMAGE":
            return links[0].from_node.image
    return next(
        (
            node.image
            for node in material.node_tree.nodes
            if node.type == "TEX_IMAGE"
            and node.image is not None
            and "normal" not in node.image.name.lower()
        ),
        None,
    )


def export_image(
    image: bpy.types.Image | None,
    output_dir: Path,
    exported: dict[str, str],
) -> str | None:
    if image is None:
        return None
    cache_key = image.filepath or image.name
    if cache_key in exported:
        return exported[cache_key]
    texture_dir = output_dir / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
    if not image.has_data and (source_path is None or not source_path.is_file()):
        return None
    relative = Path("textures") / f"{safe_name(image.name)}.png"
    destination = output_dir / relative
    if (
        source_path is not None
        and source_path.is_file()
        and source_path.suffix.lower() == ".png"
    ):
        shutil.copy2(source_path, destination)
    else:
        image.save_render(str(destination.resolve()))
    exported[cache_key] = relative.as_posix()
    return relative.as_posix()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return cleaned[:64] or "material"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relink_missing_images(texture_root: Path | None) -> dict[str, str]:
    if texture_root is None:
        return {}
    if not texture_root.is_dir():
        raise FileNotFoundError(f"Texture root not found: {texture_root}")
    candidates: dict[str, list[Path]] = {}
    for path in texture_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            candidates.setdefault(path.name.lower(), []).append(path)
    relinked: dict[str, str] = {}
    for image in bpy.data.images:
        if image.has_data or not image.filepath:
            continue
        basename = Path(image.filepath.replace("\\", "/")).name.lower()
        matches = candidates.get(basename, [])
        if len(matches) == 1:
            image.filepath = str(matches[0].resolve())
            image.reload()
            if image.has_data:
                relinked[image.name] = str(matches[0].resolve())
    return relinked


def main() -> None:
    args = parse_args()
    color_overrides = fbx_material_colors(args.material_source_fbx)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if args.input.suffix.lower() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    else:
        bpy.ops.import_scene.fbx(
            filepath=str(args.input.resolve()),
            use_anim=False,
            ignore_leaf_bones=True,
        )
    relinked_images = relink_missing_images(args.texture_root)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("FBX contains no mesh objects")
    minimum, maximum = world_bounds(meshes)
    materials = sorted(
        {
            slot.material.name
            for obj in meshes
            for slot in obj.material_slots
            if slot.material is not None
        }
    )
    images = sorted(
        {
            str(Path(image.filepath).expanduser())
            for image in bpy.data.images
            if image.filepath
        }
    )
    report = {
        "input": str(args.input),
        "objects": len(meshes),
        "vertices": sum(len(obj.data.vertices) for obj in meshes),
        "triangles": sum(
            sum(max(0, len(polygon.vertices) - 2) for polygon in obj.data.polygons)
            for obj in meshes
        ),
        "bounds": {
            "minimum": minimum,
            "maximum": maximum,
            "size": [maximum[index] - minimum[index] for index in range(3)],
        },
        "materials": materials,
        "images": images,
        "relinkedImages": relinked_images,
        "meshObjects": [
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "boundsSize": [
                    max((obj.matrix_world @ Vector(corner))[index] for corner in obj.bound_box)
                    - min((obj.matrix_world @ Vector(corner))[index] for corner in obj.bound_box)
                    for index in range(3)
                ],
            }
            for obj in meshes
        ],
    }
    print("DEMO5_FBX_REPORT=" + json.dumps(report, separators=(",", ":")))
    if args.inspect_only:
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --inspect-only is used")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The export contains a very large New York panorama around a room interior.
    # Fit and center the room, not the panorama, around the 7.4 x 4.2 m race arena.
    room_meshes = [obj for obj in meshes if "pano" not in obj.name.lower()]
    if not room_meshes:
        room_meshes = meshes
    room_minimum, room_maximum = world_bounds(room_meshes)
    room_width = room_maximum[0] - room_minimum[0]
    scale = args.room_width_m / room_width
    floor_objects = [
        obj for obj in room_meshes if obj.name.split(".", 1)[0] == "Object002"
    ]
    if floor_objects:
        floor_minimum, floor_maximum = world_bounds(floor_objects)
        floor_z = floor_maximum[2]
        room_center_x = (floor_minimum[0] + floor_maximum[0]) * 0.5
        room_center_y = (floor_minimum[1] + floor_maximum[1]) * 0.5
    else:
        floor_z = room_minimum[2]
        room_center_x = (room_minimum[0] + room_maximum[0]) * 0.5
        room_center_y = (room_minimum[1] + room_maximum[1]) * 0.5
    transform = (
        Matrix.Translation(
            Vector(
                (
                    -room_center_x * scale,
                    -room_center_y * scale,
                    -floor_z * scale - 0.025,
                )
            )
        )
        @ Matrix.Scale(scale, 4)
    )
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world

    # Join and separate by material. MuJoCo applies one material per geom, so
    # material-specific OBJ parts preserve the FBX's base colors/transparency.
    for obj in bpy.context.scene.objects:
        obj.select_set(obj in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.convert(target="MESH")
    bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = "demo5_background"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")

    parts: list[dict[str, object]] = []
    exported_images: dict[str, str] = {}
    separated = sorted(
        [obj for obj in bpy.context.selected_objects if obj.type == "MESH"],
        key=lambda obj: obj.name,
    )
    for index, obj in enumerate(separated):
        used_material_indices = sorted(
            {polygon.material_index for polygon in obj.data.polygons}
        )
        material = (
            obj.material_slots[used_material_indices[0]].material
            if used_material_indices
            and used_material_indices[0] < len(obj.material_slots)
            else None
        )
        material_name = material.name if material else f"material_{index:02d}"
        texture = export_image(
            base_color_image(material),
            args.output_dir,
            exported_images,
        )
        filename = f"{index:02d}_{safe_name(material_name)}.obj"
        for candidate in bpy.context.scene.objects:
            candidate.select_set(candidate == obj)
        bpy.context.view_layer.objects.active = obj
        if material_name == "M_Interior_Wodden_Floor":
            # This V-BLDR floor is exported with its visible side wound
            # downward. Flip it so MuJoCo's one-sided mesh renderer shows the
            # textured surface to the broadcast and robot cameras.
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.flip_normals()
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.wm.obj_export(
            filepath=str((args.output_dir / filename).resolve()),
            export_selected_objects=True,
            export_materials=False,
            export_triangulated_mesh=True,
            forward_axis="Y",
            up_axis="Z",
        )
        parts.append(
            {
                "mesh": filename,
                "material": material_name,
                "rgba": material_rgba(material, color_overrides),
                "triangles": len(obj.data.polygons),
                "texture": texture,
                # The panorama texture is externally linked and unavailable.
                # Plaster_Walls_3 is the camera-facing enclosure plane; keeping
                # it would hide the arena from Demo 5's broadcast camera.
                "enabled": (
                    "pano" not in material_name.lower()
                    and material_name != "M_Interior_White_Plaster_Walls_3"
                ),
            }
        )

    converted_minimum, converted_maximum = world_bounds(separated)
    metadata = {
        **report,
        "sourceSha256": file_sha256(args.input),
        "normalization": {
            "scale": scale,
            "roomCenterSource": [room_center_x, room_center_y],
            "floorZSource": floor_z,
            "convertedBounds": {
                "minimum": converted_minimum,
                "maximum": converted_maximum,
            },
        },
        "parts": parts,
        "missingExternalTextures": [
            image for image in images if not Path(image).is_file()
        ],
    }
    (args.output_dir / "template_73_background.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
