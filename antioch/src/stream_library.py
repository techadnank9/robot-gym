"""
Hold the scanned library open on a machine and keep its livestream alive.

    antioch run src/stream_library.py                     stream until stopped
    antioch run src/stream_library.py -- --seconds 900     stop on your own clock
    antioch run src/stream_library.py -- --orbit           slowly circle the room

The room is static geometry, so this pumps ``world.render()`` rather than
stepping physics: the stream stays live and the GPU is not spent advancing a
scene that never moves. Physics is available the moment something dynamic is
added — see ``library_navigation`` in ``src/library.py``.

Stop it with Ctrl-C, or from Mission Control in the console.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import TYPE_CHECKING, cast

import antioch

if TYPE_CHECKING:
    from isaacsim.core.api import World

ROOM_ASSET = "library-real"
ROOM_PRIM = "/World/library"

# Framing that sits inside the room rather than staring at it from outside
EYE_HEIGHT = 1.5
ORBIT_RADIUS = 7.5
ORBIT_PERIOD_S = 90.0
HEARTBEAT_S = 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream the scanned library")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="How long to hold the stream open; 0 means until the process is stopped",
    )
    parser.add_argument("--orbit", action="store_true", help="Slowly circle the camera around the room")
    parser.add_argument("--version", default=None, help="Asset revision to load; latest by default")
    arguments = parser.parse_args()

    antioch.boot()

    from isaacsim.core.api import World as WorldSingleton
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.viewports import set_camera_view

    world = cast("World", WorldSingleton())

    antioch.load_asset(ROOM_ASSET, prim_path=ROOM_PRIM, version=arguments.version)
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 500.0})

    # reset() binds physics handles and settles the stage before the first frame
    world.reset()
    set_camera_view(eye=[6.0, -6.0, 2.4], target=[-2.0, 1.0, 1.2], camera_prim_path="/OmniverseKit_Persp")

    limit = arguments.seconds if arguments.seconds > 0 else None
    started = time.monotonic()
    deadline = started + limit if limit else None
    next_heartbeat = started + HEARTBEAT_S
    frames = 0

    horizon = "until stopped" if deadline is None else f"for {limit:g}s"
    print(f"streaming {ROOM_ASSET} {horizon}", flush=True)

    try:
        while deadline is None or time.monotonic() < deadline:
            if arguments.orbit:
                angle = 2.0 * math.pi * ((time.monotonic() - started) % ORBIT_PERIOD_S) / ORBIT_PERIOD_S
                set_camera_view(
                    eye=[ORBIT_RADIUS * math.cos(angle), ORBIT_RADIUS * math.sin(angle), EYE_HEIGHT],
                    target=[0.0, 0.0, 1.2],
                    camera_prim_path="/OmniverseKit_Persp",
                )

            # The scene is static, so a render is all the stream needs; stepping
            # physics here would spend the GPU without changing a single pixel
            world.render()
            frames += 1

            now = time.monotonic()
            if now >= next_heartbeat:
                elapsed = now - started
                print(f"streaming {elapsed:7.0f}s  {frames} frames  {frames / elapsed:5.1f} fps", flush=True)
                next_heartbeat = now + HEARTBEAT_S
    except KeyboardInterrupt:
        print("stream stopped by request", flush=True)

    elapsed = max(time.monotonic() - started, 1e-6)
    print(f"held the stream {elapsed:.0f}s over {frames} frames ({frames / elapsed:.1f} fps)", flush=True)


if __name__ == "__main__":
    main()
