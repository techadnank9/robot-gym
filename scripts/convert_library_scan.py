"""Convert a Polycam-style GLB room scan into MuJoCo-ready OBJ meshes and PNG textures.

Run under Blender:

    blender --background --factory-startup --python scripts/convert_library_scan.py -- \
        --input ~/Downloads/scan.glb --output-dir assets/library_scan

The GLB is split by material so each output OBJ carries exactly one texture, which is
what MuJoCo's one-material-per-geom model expects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
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


def import_scan(path: Path) -> list[bpy.types.Object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def texture_image(material: bpy.types.Material) -> bpy.types.Image | None:
    if material is None or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node.image
    return None


def split_by_material(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    if all(len(obj.material_slots) <= 1 for obj in objects):
        return objects
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def floor_height(objects: list[bpy.types.Object], resolution: float = 0.05) -> float:
    """Estimate the floor plane as the lowest large cluster of upward-facing surface area.

    A raw scan's minimum z is noise (stray points below the floor), so the floor is found
    by area-weighting horizontal triangles into z bins and taking the lowest dominant bin.
    """
    bins: dict[int, float] = {}
    for obj in objects:
        obj.data.calc_loop_triangles()
        matrix = obj.matrix_world
        for triangle in obj.data.loop_triangles:
            if (matrix.to_3x3() @ triangle.normal).normalized().z < 0.9:
                continue
            corners = [matrix @ obj.data.vertices[index].co for index in triangle.vertices]
            area = (corners[1] - corners[0]).cross(corners[2] - corners[0]).length / 2.0
            centre = sum(corners, Vector((0.0, 0.0, 0.0))) / 3.0
            bins[int(centre.z / resolution)] = bins.get(int(centre.z / resolution), 0.0) + area

    if not bins:
        return min(obj.matrix_world @ Vector(obj.bound_box[0]) for obj in objects).z

    threshold = max(bins.values()) * 0.25
    candidates = [index for index, area in bins.items() if area >= threshold]
    return min(candidates) * resolution


def recentre(objects: list[bpy.types.Object], offset: Vector) -> None:
    for obj in objects:
        obj.matrix_world.translation += offset


def export_part(obj: bpy.types.Object, destination: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=str(destination),
        export_selected_objects=True,
        export_materials=False,
        export_uv=True,
        export_normals=True,
        export_triangulated_mesh=True,
        forward_axis="Y",
        up_axis="Z",
    )


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    meshes = import_scan(source)
    if not meshes:
        raise SystemExit(f"No mesh data found in {source}")

    minimum, maximum = world_bounds(meshes)
    dimensions = [maximum[index] - minimum[index] for index in range(3)]
    for obj in meshes:
        obj.data.calc_loop_triangles()
    triangles = sum(len(obj.data.loop_triangles) for obj in meshes)

    print("=" * 60)
    print(f"source        : {source}")
    print(f"mesh objects  : {len(meshes)}")
    print(f"vertices      : {sum(len(obj.data.vertices) for obj in meshes)}")
    print(f"triangles     : {triangles}")
    print(f"bounds min    : {[round(value, 3) for value in minimum]}")
    print(f"bounds max    : {[round(value, 3) for value in maximum]}")
    print(f"dimensions    : {[round(value, 3) for value in dimensions]} (x y z, metres)")
    materials = {
        slot.material.name
        for obj in meshes
        for slot in obj.material_slots
        if slot.material is not None
    }
    print(f"materials     : {len(materials)} -> {sorted(materials)}")
    for name in sorted(materials):
        image = texture_image(bpy.data.materials[name])
        detail = f"{image.size[0]}x{image.size[1]}" if image else "no image texture"
        print(f"  - {name}: {detail}")
    print("=" * 60)

    floor_z = floor_height(meshes)
    print(f"floor plane   : z = {floor_z:.3f} (raw scan coordinates)")

    if args.inspect_only:
        return

    # MuJoCo convention: floor at z=0, room centred on the origin so the robot spawns inside.
    offset = Vector(
        (
            -(minimum[0] + maximum[0]) / 2.0,
            -(minimum[1] + maximum[1]) / 2.0,
            -floor_z,
        )
    )
    recentre(meshes, offset)
    minimum, maximum = world_bounds(meshes)
    print(f"recentred     : min={[round(v, 3) for v in minimum]} max={[round(v, 3) for v in maximum]}")

    output_dir = args.output_dir.expanduser().resolve()
    texture_dir = output_dir / "textures"
    output_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    parts: list[dict[str, object]] = []
    for index, obj in enumerate(split_by_material(meshes)):
        obj.data.calc_loop_triangles()
        if not obj.data.loop_triangles:
            continue
        material = obj.material_slots[0].material if obj.material_slots else None
        part_name = f"library_part_{index:02d}"
        obj_path = output_dir / f"{part_name}.obj"
        export_part(obj, obj_path)

        texture_name = None
        image = texture_image(material)
        if image is not None:
            texture_name = f"{part_name}.png"
            image.file_format = "PNG"
            image.save(filepath=str(texture_dir / texture_name))

        base_color = [0.7, 0.7, 0.7, 1.0]
        if material is not None and material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    base_color = list(node.inputs["Base Color"].default_value)
                    break

        parts.append(
            {
                "name": part_name,
                "obj": obj_path.name,
                "texture": texture_name,
                "material": material.name if material else None,
                "vertices": len(obj.data.vertices),
                "triangles": len(obj.data.loop_triangles),
                "rgba": [round(value, 4) for value in base_color],
            }
        )
        print(f"exported {part_name}: {len(obj.data.vertices)} verts, texture={texture_name}")

    manifest = {
        "source": str(source),
        "bounds_min": [round(value, 6) for value in minimum],
        "bounds_max": [round(value, 6) for value in maximum],
        "dimensions": [round(value, 6) for value in dimensions],
        "floor_z_raw": round(floor_z, 6),
        "recentre_offset": [round(value, 6) for value in offset],
        "parts": parts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
