"""
Scenarios in the scanned library room.

    antioch scenario collect
    antioch scenario run --scenario library_navigation
    antioch scenario run --scenario library_navigation --set forward_speed=0.8

The room is the photogrammetry scan published as the `library-real` asset:
textured shelves, columns, seating and signage, floor at z=0, centred on the
origin, roughly 21.4 x 16.3 m with a 5.0 m ceiling.
"""

from __future__ import annotations

import antioch

logger = antioch.Logger("library")

ROOM_ASSET = "library-real"
ROOM_PRIM = "/World/library"


def _light_the_room() -> None:
    from isaacsim.core.utils.prims import create_prim

    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 500.0})


@antioch.scenario(tags=["smoke"], capture=False)
def library_room_loads(run: antioch.ScenarioRun) -> None:
    """
    Load the scanned library and confirm its geometry arrives on the stage.
    """

    import numpy as np
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Usd, UsdGeom

    world = antioch.world()
    antioch.load_asset(ROOM_ASSET, prim_path=ROOM_PRIM)
    _light_the_room()
    world.reset()
    set_camera_view(eye=[9.0, -9.0, 5.0], target=[0.0, 0.0, 1.0], camera_prim_path="/OmniverseKit_Persp")

    stage = antioch.stage()
    room = stage.GetPrimAtPath(ROOM_PRIM)
    run.check("the room prim exists", bool(room and room.IsValid()))

    meshes = [p for p in Usd.PrimRange(room) if p.IsA(UsdGeom.Mesh)]
    points = 0
    faces = 0
    for prim in meshes:
        geom = UsdGeom.Mesh(prim)
        points += len(geom.GetPointsAttr().Get() or [])
        faces += len(geom.GetFaceVertexCountsAttr().Get() or [])
    run.add_result("mesh_count", len(meshes))
    run.add_result("point_count", points)
    run.add_result("face_count", faces)
    # The scan arrives as one joined mesh, so geometry density is the real
    # signal that the room loaded — a mesh count would pass on an empty prim
    run.check("the room brought its geometry", points > 50_000, detail=f"{points} points across {len(meshes)} mesh(es)")

    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = bounds.ComputeWorldBound(room).ComputeAlignedRange()
    size = np.array(box.GetSize())
    run.add_result("room_size_m", [round(float(v), 2) for v in size])
    run.check(
        "the room is room-sized",
        10.0 < float(size[0]) < 30.0 and 10.0 < float(size[1]) < 30.0,
        detail=f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m",
    )

    shaders = [p for p in Usd.PrimRange(room) if p.GetTypeName() == "Shader"]
    run.add_result("shader_count", len(shaders))
    run.check("the room kept its materials", len(shaders) > 0, detail=f"{len(shaders)} shader prims")

    # The first capture of a run returns a pre-draw frame, so spend it here
    antioch.capture_viewport()
    for _ in range(30):
        world.step(render=True)

    frame = antioch.capture_viewport()
    published = False
    if frame is not None:
        rgb = np.asarray(frame)[:, :, :3]
        if 10.0 <= float(rgb.mean()) <= 220.0:
            logger.image("camera/rgb", rgb)
            published = True
            # A textured scan is not a flat grey plate: colour spread is what
            # separates the real mesh from an untextured stand-in
            run.add_result("frame_colour_spread", round(float(rgb.std()), 2))
    run.check("the viewport published a usable frame", published)


@antioch.scenario(tags=["smoke"], capture=False)
def library_navigation(
    run: antioch.ScenarioRun,
    forward_speed: float = antioch.param(0.5, ge=0.1, le=2.0, description="Body forward speed in m/s"),
    steps: int = antioch.param(300, ge=30, description="Physics steps to simulate"),
    start_x: float = antioch.param(-3.0, ge=-6.0, le=6.0, description="Start position along x"),
    start_y: float = antioch.param(0.0, ge=-6.0, le=6.0, description="Start position along y"),
) -> None:
    """
    Drive a body across the library's open floor and verify it stays inside
    the room, keeps its footing, and makes forward progress.

    The body is a kinematic stand-in for the G1: this measures the room's
    collision surfaces and free space, which is what a locomotion policy
    needs before it is worth loading.
    """

    import numpy as np
    import rerun as rr
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.viewports import set_camera_view

    world = antioch.world()
    antioch.load_asset(ROOM_ASSET, prim_path=ROOM_PRIM)
    _light_the_room()

    # A G1-sized box: 0.3 x 0.3 footprint, centre at hip height
    body = world.scene.add(
        DynamicCuboid(
            prim_path="/World/body",
            name="body",
            position=np.array([start_x, start_y, 0.65]),
            scale=np.array([0.3, 0.3, 1.3]),
            color=np.array([0.9, 0.35, 0.1]),
        )
    )
    world.reset()
    set_camera_view(eye=[start_x - 4.0, start_y - 4.0, 3.0], target=[start_x + 2.0, start_y, 0.8], camera_prim_path="/OmniverseKit_Persp")

    physics_dt = float(world.get_physics_dt())
    start = np.array([start_x, start_y])
    min_height = float("inf")
    left_room = False
    travelled = 0.0
    previous = start.copy()

    for step in range(steps):
        # Hold a constant forward velocity; gravity and contacts do the rest
        velocity = body.get_linear_velocity()
        body.set_linear_velocity(np.array([forward_speed, 0.0, float(velocity[2])]))
        world.step(render=step % 4 == 0)

        position = body.get_world_pose()[0]
        planar = np.array([float(position[0]), float(position[1])])
        travelled += float(np.linalg.norm(planar - previous))
        previous = planar
        height = float(position[2])
        min_height = min(min_height, height)
        left_room = left_room or abs(planar[0]) > 7.5 or abs(planar[1]) > 7.5

        logger.scalar("body/height", height)
        logger.scalar("body/x", float(planar[0]))
        logger.value(
            "scene/body",
            rr.Boxes3D(centers=[[float(position[0]), float(position[1]), height]], sizes=[[0.3, 0.3, 1.3]], colors=[[230, 90, 25]]),
        )

    final = body.get_world_pose()[0]
    displacement = float(np.linalg.norm(np.array([float(final[0]), float(final[1])]) - start))

    run.add_result("displacement_m", round(displacement, 3))
    run.add_result("path_length_m", round(travelled, 3))
    run.add_result("min_height_m", round(min_height, 3))
    run.add_result("final_position", [round(float(v), 3) for v in final])

    expected = forward_speed * steps * physics_dt
    run.check("the body moved across the floor", displacement > 0.25, detail=f"{displacement:.2f} m from the start")
    run.check("the body stayed on its feet", min_height > 0.25, detail=f"lowest centre height {min_height:.2f} m")
    run.check("the body stayed inside the room", not left_room, detail="never passed the 7.5 m room bound")
    run.check(
        "the run was not a free slide",
        displacement <= expected + 1.0,
        detail=f"{displacement:.2f} m against {expected:.2f} m of commanded travel",
    )
