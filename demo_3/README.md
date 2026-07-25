# Demo 3: VLGE G1 1v1 on macOS

Demo 3 is isolated from Demos 1 and 2. It runs two G1 robots in one shared
MuJoCo contact simulation and exposes a live broadcast UI intended to be
embedded as the main screen inside a VLGE V-BLDR world.

## What is real in the simulation

- one shared MuJoCo model, not two disconnected simulations;
- two independent copies of Unitree's pinned 47-observation/12-action
  locomotion policy running at 50 Hz;
- 500 Hz G1 dynamics, bounded leg torques, physical robot collisions, free
  payload bodies, articulated arms, and seven controlled right-hand joints;
- actual payload/robot contact is required before the bounded grip-force
  controller engages; payload poses are never teleported or mocap-followed;
- checkpoint, goal containment, release, velocity, and one-second stability
  checks are authoritative.

The contact-conditioned grip force represents the tactile grip controller that
is not included in the public G1 model. It acts through MuJoCo forces and is
recorded as simulation assistance. It is not evidence that an uncalibrated real
hand can reproduce the grasp.

## One-time setup

```bash
make mac-setup
scripts/setup_demo_2_sil.sh
```

Validate the combined model without opening a window:

```bash
scripts/run_g1_demo_3.sh validate
```

Run a complete deterministic proof match:

```bash
scripts/run_g1_demo_3.sh scripted
```

The MuJoCo viewer and the VLGE-embeddable broadcast are both enabled. Open
`http://127.0.0.1:8083` for the broadcast surface.

## Gemini Robotics-ER default

Both policy slots default to Gemini Robotics-ER:

```bash
export GEMINI_API_KEY="<restricted key>"
scripts/run_g1_demo_3.sh match
```

With one key, both players automatically share its five-RPM limiter. For a
true player-vs-player setup, give each slot its own restricted key:

```bash
export DEMO3_P1_GEMINI_API_KEY="<player-one restricted key>"
export DEMO3_P2_GEMINI_API_KEY="<player-two restricted key>"
scripts/run_g1_demo_3.sh match
```

The slot-specific keys take precedence over `GEMINI_API_KEY` and are never
included in match telemetry.

Choose a human for either slot:

```bash
scripts/run_g1_demo_3.sh match \
  --p1 human \
  --p2 policy \
  --p2-adapter gemini-er
```

Human input defaults to `auto`. Demo 3 tries the selected gamepad first and,
when none is available, keeps the match running with browser keyboard control
in the live view at `http://127.0.0.1:8083`. Click the view, then use W/A/S/D
to move, Q/E to turn, G to grasp, C to carry, R to release, and U to recover.
Movement keys are the keyboard deadman, so releasing them commands a stop.

To select the keyboard without probing SDL:

```bash
scripts/run_g1_demo_3.sh match \
  --p1 human \
  --p1-input keyboard \
  --p2 policy
```

For a gamepad, hold the left shoulder button as the deadman. The left stick
commands planar velocity, the right stick commands yaw, and the face buttons
select grasp, release, carry, and recovery skills. Triggers open and close the
hand.

Connect and wake the controller before launching. Gamepad SDL polling runs in a
separate helper process so it cannot conflict with MuJoCo's macOS Cocoa viewer.
If multiple controllers are connected, select one with `--p1-gamepad 0` or
`--p1-gamepad 1`. Use `--p1-input gamepad` to require a controller and fail
instead of falling back. One browser keyboard controls one player; a second
simultaneous human needs a gamepad.

Choose another approved model through a localhost or HTTPS JSON adapter:

```bash
export DEMO3_P1_POLICY_KEY="<provider key>"
scripts/run_g1_demo_3.sh match \
  --p1-adapter http \
  --p1-model "Player model" \
  --p1-endpoint "https://provider.example/v1/g1-policy"
```

The endpoint receives the versioned grounded status, allowed skills, and two
JPEG cameras. It returns `skill`, with optional `rationale`, `expectedOutcome`,
and `inferenceId`.

## VLGE

Before native VLGE runtime API access is available:

1. Build the visible arena in V-BLDR using the measurements in `scene.yaml`.
2. Add an embedded web/game surface pointing at a publicly deployed copy of
   `demo_3/web`.
3. Connect its WebSocket setting to the outbound match bridge.

For local testing, the built-in server uses HTTP port `8083` and WebSocket port
`8763`. `VLGEWorldAdapter`-compatible state is represented by the versioned
match payload; native entity synchronization can consume the same payload later.

To relay the same live frames and match state through the existing outbound
Replit bridge for a remotely hosted VLGE embed:

```bash
export REPLIT_ENABLED=1
export REPLIT_CONTROL_URL="wss://YOUR-APP.replit.app/ws/v1/worker"
export REPLIT_WORKER_TOKEN="<shared worker token>"
scripts/run_g1_demo_3.sh scripted
```

Every run saves the generated shared MJCF, match state, events, and result under
`outputs/demo_3_1v1_*`.
