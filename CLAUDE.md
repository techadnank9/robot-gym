# Working in this repository

Notes for Claude Code and other agents. Conventions and hard-won details that
are not obvious from reading the tree.

## Layout

- `demo_3/` — the base competitive arena. `DualG1RaceArena` owns `MjModel` /
  `MjData` and the step loop; `model.py` generates the dual-G1 scene by loading
  the G1 MJCF twice and prefixing every element per player.
- `demo_5/` — `SimToRealG1RaceArena` extends demo_3 with domain randomization,
  the degraded command channel, camera-grounded perception, and evidence output.
- `demo_6/` — a 3x planar-speed gameplay profile. Simulation-only, and labeled
  as such; do not present it as a real-G1 operating envelope.
- `demo_2/` — SDK2 transport and the pinned official Unitree locomotion policy.
- `pathvla/` — world generators and the VLA client. Not part of the arena stack.
- `isaac_ext/` — Isaac Lab / Isaac Sim extension, separate from the MuJoCo path.

## The layering boundary

Models never command joint torques. The chain is:

```
decision -> guarded skill -> 50 Hz velocity command -> Unitree locomotion policy -> MuJoCo @ 500 Hz
```

Keep it that way. Policies choose grounded intent (`navigate_object`, `grasp`,
`navigate_goal`, `release`, `recover`, `wait`) and the arena validates each
decision against physical state before executing. Cross-model comparison is
only meaningful because everything below that boundary is fixed.

`demo_5/command_channel.py` and `demo_5/perception.py` deliberately degrade
commands and observations (latency, dropout, slew limits, watchdog, position
noise, misses). These are features, not bugs; do not "fix" them into ground
truth.

## MuJoCo conventions

**Always use `mujoco.viewer.launch_passive`, never `mujoco.viewer.launch`.**
On macOS with Python 3.14 and MuJoCo 3.11, the blocking `launch()` path raises
`RuntimeError: Caught an unknown exception!` from inside the `Simulate`
constructor. This is not a sandbox, GUI-session, or scene problem: it
reproduces from a real Terminal in an Aqua session with a one-geom model, and
`mujoco.viewer._MJPYTHON` is correctly installed on `MainThread` under
`mjpython`. The passive path, where the caller owns the stepping loop, works.

**Mesh geoms collide as their convex hull.** For room-scale environment
meshes this produces a solid block that traps the robot. Attach scene meshes as
visual-only (`contype=0 conaffinity=0`, group 2) and provide collision
separately with planes and boxes.

**Building scenes:** parse the G1 MJCF with `ElementTree`, set `meshdir` to an
absolute path, and add assets and geoms to that tree. See
`pathvla/osm_mujoco.py` and `pathvla/library_mujoco.py`. Reference your own
mesh and texture files with absolute paths, since `meshdir` is model-wide.

**Interactive viewers need a GUI-attached terminal.** They cannot be launched
from an agent shell. Use `--screenshot` or `--video` to verify rendering work
instead of asking the user to look.

## Room scans

`scripts/convert_library_scan.py` (Blender) converts a photogrammetry GLB into
MuJoCo-ready parts; `pathvla/library_mujoco.py` builds and launches the scene.
See the "Real room scans" section of the README.

`assets/library_scan/` is gitignored. Converted scans run to roughly 100 MB and
usually capture a private space and the people in it, so they stay local and
are regenerated with `make mac-library-convert SCAN_GLB=<file>.glb`. Do not
commit scan output to this repository, which is public.

## Verification

- `python3 -m pytest tests -m "not integration"` for the suite.
- For scene changes, compile and settle the model before claiming it works:
  reset to the keyframe, step a couple of seconds, and check the base height is
  stable, that no NaN appears in `qpos`, and which geoms the contacts resolve
  against. A scene that compiles can still trap or drop the robot.
