# Demo 5: reduced sim-to-real gap

Demo 5 is an isolated successor to Demo 3. It keeps the shared VLGE/MuJoCo
1v1 task and the hardware-shaped locomotion path. Its default `easy` grasp is
intentionally game-like; use `--grasp-mode mechanical` for manipulation
sim-to-real evaluation.

## What changes

- High-level navigation and manipulation consume delayed segmented RGB-D
  estimates. MuJoCo object coordinates remain referee-only.
- Easy grasp (the default) snaps a payload within 1.25 m to the right hand,
  disables its collisions while carried, and restores collisions on release.
  State and evidence label this as privileged kinematic assistance.
- `--grasp-mode mechanical` removes that attachment and accepts a grasp only
  after finger closure and robot/payload contact.
- Manipulation uses a staged side approach: open-hand pre-grasp above the
  payload, horizontal engagement, contact-confirmed closure, then lift. A
  downward payload estimate opens the hand and triggers reacquisition.
- Commands cross the same `G1Transport.set_velocity` contract as SDK2, wrapped
  by a 50 Hz channel with clipping, slew limits, 40–80 ms latency, packet loss
  and a 120 ms stop watchdog.
- Joint observations, motor strength, friction, payload mass, lighting, object
  placement, camera position and detections are randomized from a recorded seed.
- Missed or lost grasps return to perception and reacquisition instead of being
  silently stabilized.
- Every run writes camera error, command trace, randomized parameters and
  optional hardware-replay divergence evidence.
- Demo 5 loads the material-separated `Template_73_Export` V-BLDR room around
  the arena. Its meshes are visual-only (`contype=0`, `conaffinity=0`), so the
  race floor, obstacles, buckets, scoring, and contact physics remain authoritative.

This is stronger simulation evidence, not a claim that autonomous manipulation
is safe or ready for an untethered physical G1.

## V-BLDR room asset

The checked-in conversion preserves the FBX geometry, UVs, material groups,
diffuse color factors, and the camera-safe interior layout. The source FBX does
not embed its images, but its 14 referenced PNG dependencies are included under
`demo_5/assets/background/vlge_dependencies`. Albedo images are copied and
wired into MuJoCo automatically; the source normal maps are retained for an
external renderer. Demo 5's authoritative race-floor collider is transparent,
allowing the visual-only V-BLDR wooden floor beneath it to remain visible.

MuJoCo's built-in renderer displays RGB/albedo textures but not V-BLDR's full
normal-map/PBR shader. Geometry, UV placement, material colors, transparency,
and albedo detail remain available in the native match view.

To regenerate the material-separated meshes after obtaining the dependencies:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python scripts/convert_demo5_background.py -- \
  --input demo_5/assets/background/template_73_source.glb \
  --material-source-fbx /path/to/Template_73_Export.fbx \
  --texture-root /path/to/extracted/V-BLDR-export \
  --output-dir demo_5/assets/background
```

`--texture-root` searches the extracted export recursively by image filename,
so its internal directory structure does not need to match the original
Windows paths stored in the FBX. Loaded albedo images are copied into
`demo_5/assets/background/textures` and wired into MuJoCo automatically.

## Run

### RunPod / Linux browser match

On a RunPod PyTorch Pod, expose HTTP ports 8085 and 8765, then run:

```bash
bash scripts/setup_runpod.sh
scripts/run_g1_demo_5_runpod.sh play
```

Open `https://POD_ID-8085.proxy.runpod.net`. The RunPod launcher uses EGL
headless rendering, wall-clock-paced physics, external interface binding, and
automatic RunPod WebSocket proxy discovery. The first game uses a local
scripted opponent and requires no API key.

For a joystick connected to your laptop or desktop, press any controller button
after opening the page. The browser automatically switches from keyboard to the
Gamepad API: left stick moves, right stick turns, A/Cross grasps, X/Square
carries, B/Circle releases, Y/Triangle recovers, and Start/Options requests a
payload reset. The controller does not need USB passthrough to the Pod.

Run `scripts/run_g1_demo_5_runpod.sh practice` for one human and an idle
opponent, or export `DEMO3_P2_GEMINI_API_KEY` and run
`scripts/run_g1_demo_5_runpod.sh gemini` for Gemini Robotics-ER.

Validate the constrained stack:

```bash
scripts/run_g1_demo_5.sh validate
```

Run deterministic model players:

```bash
scripts/run_g1_demo_5.sh scripted --domain-seed 5
```

Run a keyboard player against Gemini Robotics-ER:

```bash
scripts/run_g1_demo_5.sh match \
  --p1 human \
  --p1-input mujoco-keyboard \
  --p2 policy
```

Run a local joystick player against an idle G1:

```bash
scripts/run_g1_demo_5.sh match \
  --p1 human --p1-input gamepad --p1-gamepad 0 \
  --p2 human --p2-input idle
```

The left stick or D-pad walks, the right stick turns, and returning the controls
to center stops. No shoulder-button deadman is required. Face buttons remain
A/Cross grasp, X/Square carry, B/Circle release, and Y/Triangle recover.
View/Back resets the MuJoCo camera zoom and angle. The movement ceiling is
slightly faster than the original Demo 5 profile: 0.52 m/s forward, 0.26 m/s
lateral, and 0.90 rad/s yaw.

Practice with one human while the second robot remains connected in a neutral
standing hold:

```bash
scripts/run_g1_demo_5.sh match \
  --p1 human --p1-input mujoco-keyboard \
  --p2 human --p2-input idle
```

The idle station emits zero velocity, open-hand, `wait` frames and makes no
model/API calls. It remains a physical robot in the arena, so collisions and
falls are still simulated.

With `mujoco-keyboard`, focus the native MuJoCo window and use the arrow keys
to select a walking direction or Q/E to turn. Native callbacks do not report
key-up events, so motion remains active until Space is pressed. G performs the
easy grasp, C carries, R releases, U recovers, and X requests a referee payload
reset. Home resets the camera zoom and viewing angle. Some letters are also
MuJoCo visualization shortcuts; any resulting display-color change does not
alter the robot physics.

Easy grasp gives the AI a small accessibility assist: policy players receive a
1.45 m capture radius and an earlier 0.60 m `nearObject` decision signal, while
human players retain the 1.25 m capture radius. Both use the same disclosed
snap-to-hand attachment once grasp succeeds.

The older `--p1-input keyboard` mode remains available for browser control.
For that mode, open `http://127.0.0.1:8085/?wsPort=8765` and keep the browser
focused. A connected browser gamepad is automatically preferred over keyboard.
The simulator waits in its lobby until browser telemetry arrives.

The default `--render-profile performance` keeps scoreboard telemetry at 5 Hz
while staggering lower-resolution broadcast and ego frames so camera rendering
does not stall physics. To restore the original three-camera 5 Hz stream:

```bash
scripts/run_g1_demo_5.sh match --render-profile quality
```

A referee reset is granted when the payload is below the arm's feasible pickup
height or after three recorded failed grasp attempts. It remains unavailable
after checkpoint transport. Each player gets at most two resets and is
motion-locked for three simulation seconds afterward. AI policies receive
`resetAvailable` in grounded status and may select the existing `recover` skill
to request the same reset.

Use separate restricted Gemini keys for a true model-vs-model match:

```bash
export DEMO3_P1_GEMINI_API_KEY="<player-one key>"
export DEMO3_P2_GEMINI_API_KEY="<player-two key>"
scripts/run_g1_demo_5.sh match
```

## Hardware replay comparison

Pass a JSON list, JSONL file, or `{"samples": [...]}` object containing
`playerId` and `robot: [x, y, z]` samples:

```bash
scripts/run_g1_demo_5.sh validate --hardware-log g1_sdk_telemetry.jsonl
```

Evidence is written below `outputs/demo_5_sim_to_real_*`, including
`sim_to_real_report.json`, `trajectory.json`, and each player's SDK command
trace. Ground truth appears only in evidence and authoritative scoring.
