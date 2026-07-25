from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET


def background_asset_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "background"


def add_template_73_background(
    xml: str,
    *,
    asset_dir: Path | str | None = None,
) -> str:
    """Add the converted V-BLDR room as visual-only MuJoCo geometry."""

    directory = Path(asset_dir) if asset_dir else background_asset_dir()
    manifest_path = directory / "template_73_background.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Demo 5 background manifest not found at {manifest_path}. "
            "Run scripts/convert_demo5_background.py through Blender first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = ET.fromstring(xml)
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Demo 5 base MJCF is missing asset or worldbody")
    arena_floor = worldbody.find("./geom[@name='arena_floor']")
    if arena_floor is not None:
        # Keep the authoritative race-floor collider while revealing the
        # V-BLDR wooden floor rendered 2.5 cm beneath it.
        arena_floor.set("rgba", "1 1 1 0")

    background = ET.SubElement(worldbody, "body", name="demo5_vbldr_background")
    for index, part in enumerate(manifest["parts"]):
        if not part.get("enabled", True):
            continue
        mesh_path = (directory / str(part["mesh"])).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Demo 5 background mesh not found: {mesh_path}")
        mesh_name = f"demo5_background_mesh_{index:02d}"
        material_name = f"demo5_background_material_{index:02d}"
        rgba = " ".join(str(float(value)) for value in part["rgba"])
        ET.SubElement(
            asset,
            "mesh",
            name=mesh_name,
            file=str(mesh_path),
        )
        material_attributes = {
            "name": material_name,
            "rgba": rgba,
            "specular": "0.18",
            "shininess": "0.12",
        }
        texture_file = part.get("texture")
        if texture_file:
            texture_path = (directory / str(texture_file)).resolve()
            if not texture_path.is_file():
                raise FileNotFoundError(
                    f"Demo 5 background texture not found: {texture_path}"
                )
            texture_name = f"demo5_background_texture_{index:02d}"
            ET.SubElement(
                asset,
                "texture",
                name=texture_name,
                type="2d",
                file=str(texture_path),
            )
            material_attributes["texture"] = texture_name
        ET.SubElement(asset, "material", **material_attributes)
        ET.SubElement(
            background,
            "geom",
            name=f"demo5_background_geom_{index:02d}",
            type="mesh",
            mesh=mesh_name,
            material=material_name,
            contype="0",
            conaffinity="0",
            group="2",
        )
    return ET.tostring(root, encoding="unicode")
