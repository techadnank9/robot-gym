"""
Probe the engine image for a usable Unitree G1 and its locomotion policy.

    antioch run --no-stream src/probe_assets.py

Isaac ships an H1 sample policy; whether a G1 body and a G1 policy are on the
asset server decides how the fetch task gets built, so this asks before any
of that is designed around the answer.
"""

from __future__ import annotations

import antioch


def main() -> None:
    antioch.boot()

    import omni.client
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    print(f"ASSETS_ROOT={root}", flush=True)

    def listdir(path: str) -> list[str]:
        result, entries = omni.client.list(path)
        if result != omni.client.Result.OK:
            return []
        return sorted(e.relative_path for e in entries)

    for path in (
        f"{root}/Isaac/Robots/Unitree",
        f"{root}/Isaac/Samples/Policies",
        f"{root}/Isaac/Samples/Mujoco_Menagerie",
    ):
        print(f"\n--- {path}", flush=True)
        for name in listdir(path):
            print(f"    {name}", flush=True)

    for candidate in (
        f"{root}/Isaac/Robots/Unitree/G1",
        f"{root}/Isaac/Samples/Policies/g1",
        f"{root}/Isaac/Samples/Mujoco_Menagerie/unitree_g1",
    ):
        entries = listdir(candidate)
        print(f"\n--- {candidate}  ({len(entries)} entries)", flush=True)
        for name in entries:
            print(f"    {name}", flush=True)


if __name__ == "__main__":
    main()
