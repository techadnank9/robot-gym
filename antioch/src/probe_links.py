import antioch
def main():
    antioch.boot()
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    import omni.usd
    from pxr import Usd, UsdGeom
    root = get_assets_root_path()
    add_reference_to_stage(usd_path=f"{root}/Isaac/Robots/Unitree/G1/g1.usd", prim_path="/World/g1")
    stage = omni.usd.get_context().get_stage()
    names = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/g1")):
        n = prim.GetName().lower()
        if any(k in n for k in ("wrist", "hand", "palm", "elbow", "shoulder")):
            names.append(prim.GetPath().pathString)
    print("LINKS_BEGIN")
    for n in names:
        print("  ", n)
    print("LINKS_END")
main()
