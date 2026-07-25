# Robot Gym — VLGE Unitree G1 Race

Real Isaac Lab / Isaac Sim Unitree path simulation with VLA planning and live remote viewing.

## RunPod: playable in two commands

Demo 5 runs as a browser-playable, two-G1 MuJoCo race on Linux. The browser
shows the VLGE-styled match view and sends keyboard controls; MuJoCo and the
official Unitree locomotion policy run headlessly on the Pod.

### 1. Create the Pod

Use a RunPod GPU Pod with at least 64 GB system RAM. The current official
RunPod documentation uses this PyTorch image:

```text
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```

Recommended Pod settings:

- container disk: 40 GB or more;
- persistent volume: 20 GB or more, mounted at `/workspace`;
- expose HTTP ports: `8085,8765`;
- start with the web terminal or SSH.

Both ports must be exposed as **HTTP ports**. Port 8085 serves the match page;
8765 carries its WebSocket telemetry and controls. RunPod publishes them as
`https://POD_ID-PORT.proxy.runpod.net`.

### 2. Install and play

Run these commands in the Pod terminal:

```bash
cd /workspace
git clone https://github.com/KaushikSiva/robot-gym.git
cd robot-gym
bash scripts/setup_runpod.sh
scripts/run_g1_demo_5_runpod.sh play
```

The setup script creates an isolated environment, installs the Linux/EGL
runtime, downloads hash-checked pinned G1 assets and the official Unitree
locomotion policy, and renders a validation frame before reporting success.

The launcher prints the public URL. It has this form:

```text
https://YOUR_POD_ID-8085.proxy.runpod.net
```

Open it, click once inside the page, and use:

- arrow keys — walk; the command remains active until Space;
- Q/E — turn;
- Space — stop;
- G — easy grasp when within 1.25 m of the payload;
- C — carry pose;
- R — release near the bucket;
- U — recovery request;
- X — request a payload reset.

`play` needs no model key: Player 2 uses the local deterministic policy. For a
solo practice arena with the other G1 standing idle:

```bash
scripts/run_g1_demo_5_runpod.sh practice
```

For Gemini Robotics-ER as Player 2:

```bash
export DEMO3_P2_GEMINI_API_KEY="<restricted Gemini key>"
scripts/run_g1_demo_5_runpod.sh gemini
```

Validate again without opening a match:

```bash
scripts/run_g1_demo_5_runpod.sh validate
```

The RunPod HTTP proxy is public. Keep API keys only in environment variables,
never in the URL or repository. Stop the Pod when finished to stop billing.
RunPod's official port guide is
[here](https://docs.runpod.io/pods/configuration/expose-ports).

### Optional custom RunPod image

The repository also includes a ready-to-build image:

```bash
docker build -f docker/Dockerfile.runpod -t robot-gym:runpod .
```

Use `docker/Dockerfile.runpod` in a custom RunPod template, expose
`8085/http` and `8765/http`, and leave its default command unchanged. It starts
the playable human-versus-scripted match automatically.

### RunPod troubleshooting

- Page opens but says reconnecting: expose `8765` as an HTTP port too.
- No movement: click the browser page, tap an arrow, and press Space before
  changing directions.
- EGL initialization fails: use a RunPod NVIDIA/PyTorch Pod and rerun
  `bash scripts/setup_runpod.sh`.
- First start takes several minutes: PyTorch, MuJoCo assets, textures, and the
  pinned policy are installed once into the persistent `/workspace` volume.

## Demo 5: G1 race

Demo 5 keeps the VLGE 1v1 task and a hardware-shaped 50 Hz SDK command channel.
Locomotion uses the pinned official Unitree G1 policy. The default `easy` grasp
is intentionally game-like: it uses a disclosed snap-to-hand attachment so the
race is fun and reliable. Pass `--grasp-mode mechanical` when you specifically
want contact-only manipulation evidence.

The constrained stack still includes camera-derived RGB-D estimates, the SDK2
transport contract, command latency/dropout/watchdogs, sensor and actuator
uncertainty, seeded domain randomization, recovery, and hardware-log divergence
evidence. Assisted grasp runs are labeled as privileged in state and evidence.

```bash
scripts/run_g1_demo_5.sh validate
scripts/run_g1_demo_5.sh scripted
```

See `demo_5/README.md` for keyboard/model matches and hardware replay format.

## Georeferenced San Francisco MuJoCo Scene

The native MuJoCo path can generate a local San Francisco scene from live
OpenStreetMap and USGS elevation data. The default Golden Gate Bridge scene:

- downloads and caches a small OSM bounding box;
- converts WGS84 latitude/longitude to local east/north/up meters;
- turns roads and footways into MuJoCo geometry and extrudes building footprints;
- samples a cacheable USGS elevation grid and writes a MuJoCo heightfield; and
- places the Menagerie Unitree G1 at a geographic pose and heading.

The landmark renderer adds geometry that OSM does not contain: accurately
aligned towers, the suspended deck, main cables, vertical hangers, under-deck
trusses, six marked traffic lanes, two sidewalks, guardrails, and bay water. It
uses generated asphalt, concrete, painted-steel, and water textures plus a
cinematic bridge camera. This is a detailed MuJoCo digital twin, not scanned
photogrammetry; true film-level photorealism requires a licensed high-resolution
mesh and PBR texture set that is not present in OSM.

Set up the existing Mac runtime, build the scene once, and launch it:

```bash
make mac-setup
make mac-osm-build
make mac-osm
```

The first build needs network access. It writes `map.osm`, `elevation.json`,
`elevation.png`, `scene.xml`, and `scene_metadata.json` under
`outputs/mujoco_sf_golden_gate_bridge/`; later builds use those caches. To explicitly
test without elevation downloads, pass `--flat-elevation`. To fetch current OSM
data again, pass `--refresh-osm`:

```bash
.venv-mac/bin/python -m pathvla.osm_mujoco --build-only --refresh-osm
.venv-mac/bin/python -m pathvla.osm_mujoco --build-only --flat-elevation
```

Edit `config/osm_sf_golden_gate_bridge.yaml` to adjust the bridge crop and robot
latitude/longitude. Generated data retains OpenStreetMap attribution and a
source record in `scene_metadata.json`. OSM is a static map, not a live safety
system: it does not replace perception, localization, curb/stair modeling, or a
real locomotion controller. Multipolygon outer rings are supported; inner-ring
courtyards are currently filled as solid building geometry.

The Salesforce Park preset models the park on the level-3 roof of the Transit
Center rather than at street level. It also enables the semantic palette:
asphalt is dark gray, paving and sidewalks are warm stone, parks/grass/gardens
use distinct greens, water is blue, playgrounds are orange, and tagged building
colors/materials are retained (including the glass Salesforce Tower).

```bash
make mac-salesforce-park-build
make mac-salesforce-park
```

Its cached output is written to `outputs/mujoco_sf_salesforce_park/`. Geographic
and roof-height settings live in `config/osm_sf_salesforce_park.yaml`.

## Native Apple Silicon Mac Demo

The sorting demo now runs natively on macOS with MuJoCo. It does not use Docker,
CUDA, Isaac, or an NVIDIA GPU. The Mac path uses Google DeepMind MuJoCo
Menagerie's G1-with-hands MJCF because MuJoCo does not load USD directly.

The runtime has already been validated on Apple Silicon with MuJoCo 3.10.0:

- G1 model compiled with 50 configuration coordinates and 43 actuators
- indoor sorting lab compiled with 54 bodies
- oblique and overhead cameras rendered successfully
- the complete 17-action articulated-hand sorting execution passed locally

Set up once:

```bash
cd pathvla-unitree-isaac-live
make mac-setup
```

Run the full Gemini-controlled interactive demo:

```bash
export GEMINI_API_KEY="<restricted Gemini API key>"
make mac-demo
```

The interactive viewer is launched with MuJoCo's required `mjpython` macOS
launcher. Gemini receives both rendered camera images on every action turn.
There is no rule-planning fallback.

The Mac executor uses the G1's seven right-arm/wrist joints and seven right-hand
finger joints. For each pickup it moves into a manipulation stance, reaches to
the object with damped-least-squares IK, closes the thumb/index/middle fingers,
attaches the carried item to the wrist grasp site, lifts it, and releases it over
the selected bucket. This is an articulated simulation skill rather than a
learned low-level grasp policy.

The MuJoCo window also includes an agent panel with Gemini's selected action,
target, concise rationale, expected outcome, execution status, and recent
completed actions. These are inspectable decision summaries returned by the
model—not private hidden chain-of-thought. The same summaries are printed in the
terminal and saved in `gemini_action_plan.json`.

The agent spaces requests 13 seconds apart by default, keeping it below the
Robotics-ER free-tier limit of five requests per minute. HTTP 429 responses are
retried using Google's returned delay while the MuJoCo viewer remains responsive.
Override only when your paid quota permits it:

```bash
export GEMINI_MIN_REQUEST_INTERVAL_S=13
export GEMINI_RATE_LIMIT_RETRIES=3
```

To verify the simulator and cameras without making an API request:

```bash
make mac-smoke
```

Mac-specific files:

- `pathvla/mujoco_sorting_demo.py` — native agent loop
- `pathvla/mujoco_lab.py` — generated lab, renderer, and task skills
- `requirements-mac.txt` — isolated native dependencies
- `scripts/run_mac_demo.sh` — `mjpython` GUI launcher
- `assets/mujoco_g1.lock.json` — pinned Menagerie revision

## Replit Mac Worker Integration

The native Mac simulator can connect outward to a Replit control application by
authenticated WebSocket. Replit does not need inbound access to the Mac, and the
Gemini API key never leaves the Mac process.

Install the updated Mac dependencies once:

```bash
make mac-setup
```

Set the deployed Replit worker endpoint and the same worker token configured in
Replit Secrets:

```bash
export GEMINI_API_KEY="<restricted Gemini API key>"
export REPLIT_CONTROL_URL="wss://YOUR-APP.replit.app/ws/v1/worker"
export REPLIT_WORKER_TOKEN="<shared worker token>"
```

To mirror a normal local demo to the Replit dashboard:

```bash
export REPLIT_ENABLED=1
make mac-demo
```

To leave the Mac online and execute tasks submitted by the Replit dashboard:

```bash
make replit-worker
```

The worker accepts only `task_command` messages with a non-empty `instruction`
and this optional allowlist: `maxActions`, `maxRejections`, `thinkingBudget`,
`recordVideo`, `headless`, and `lingerSeconds`. It does not accept commands,
paths, model names, environment variables, or shell arguments from Replit.

The worker streams real compressed camera frames plus versioned events for run
startup, scene observation, Gemini waits/decisions, action execution/rejection,
completion, failure, and cooperative stopping. Decision payloads contain the
model's concise rationale and expected outcome, never hidden chain-of-thought.
Frames are deduplicated and evicted before control events when the bounded
outgoing queue is full. A dashboard stop is acknowledged between actions (or
during a Gemini quota wait), preserving a partial trace and final typed state.

The Replit service should send messages shaped like:

```json
{
  "type": "task_command",
  "taskId": "task-123",
  "runId": "run-123",
  "instruction": "Sort every red item into the red bucket.",
  "options": {"recordVideo": true}
}
```

To request a safe stop:

```json
{"type": "stop_requested", "runId": "run-123"}
```

Run the focused integration tests with `make replit-test`. The transport client
is in `pathvla/replit_worker.py`; the long-running Mac entry point is
`pathvla/replit_mac_worker.py`.

## Gemini Robotics-ER G1 Sorting Demo

The featured demo is a closed-loop embodied-reasoning task in an indoor lab:

```text
Sort every red item into the red bucket and every blue item into the blue bucket.
```

It uses:

- Google's current `gemini-robotics-er-1.6-preview` model for visual reasoning and online action selection
- the official Unitree G1 29-DoF USD from `unitreerobotics/unitree_model`
- two rendered Isaac camera views on every model turn
- a constrained action vocabulary: `navigate`, `pick`, `place`, and `finish`
- precondition, object-ID, proximity, color-match, and geometric completion checks
- an indoor room with a sorting table, four items, two open-top buckets, and an obstacle

There is no Gemini planner fallback and no proxy robot on this path. A missing
key, inaccessible model, missing USD, absent camera frame, or invalid action
causes an explicit failure. Rejected unsafe actions can be returned to Gemini
for correction, but are never silently replaced by rules.

### Architecture

```text
Isaac cameras + typed world state + task
                    |
                    v
       Gemini Robotics-ER 1.6
         one grounded action
                    |
                    v
          strict safety gate
                    |
                    v
 G1 navigation / pick / place simulator skills
                    |
                    v
       new rendered observation
```

Gemini Robotics-ER is the high-level embodied reasoning model, not a low-level
joint policy. The executors use bounded task-space simulation skills: A* root
navigation around obstacles plus grasp, carry, and release. The native Mac path
articulates the G1 arm, wrist, and fingers with task-space IK; the Isaac path
drives the real G1 USD on an Isaac stage. Neither is a learned locomotion/grasp
policy or a real-robot deployment.

### Run It

The Isaac version requires Ubuntu, an NVIDIA GPU, a compatible Isaac Sim/Isaac
Lab container, and a restricted Gemini API key. Use the native Mac workflow
above on macOS. Robotics-ER returns HTTP 403 for unrestricted keys, so configure
API restrictions in Google AI Studio first.

```bash
cd pathvla-unitree-isaac-live
cp .env.example .env

export ISAAC_BASE_IMAGE="<compatible Isaac Sim/Lab image>"
export HOST_WORKSPACE_ROOT="$PWD"
export BREV_PUBLIC_HOST="<GPU VM hostname>"
export GEMINI_API_KEY="<restricted Gemini API key>"

make download-g1
make build
make gemini-sort-demo
```

To change the instruction:

```bash
make gemini-sort-demo \
  SORT_INSTRUCTION="Put all blue objects in the blue bucket, then sort the red objects."
```

The action loop writes `gemini_action_plan.json`, `execution_trace.json`, every
camera observation, `result.json`, logs, and `rollout.mp4` under `outputs/`.

Relevant upstream documentation:

- [Gemini Robotics-ER 1.6](https://ai.google.dev/gemini-api/docs/robotics-overview)
- [Official Unitree robot models](https://github.com/unitreerobotics/unitree_model)

The asset checkout is ignored because it contains Git LFS binaries. Its verified
upstream commit and relative asset path are recorded in
`isaac_ext/pathvla_unitree/assets/unitree_g1.lock.json`.

## What This Is

This project runs a real Isaac Sim or Isaac Lab scene on a Linux NVIDIA GPU VM such as Brev, calls a real configured VLA endpoint to turn natural-language instructions into structured subgoals, plans waypoints in the live Isaac world, executes movement in simulation, and exposes API/dashboard surfaces for launch, live viewing, and replay.

Target example:

```text
Go to the red bin, avoid the chair, inspect the table, then return home.
```

## What This Is Not

- Not a toy local simulator
- Not a 2D fallback simulator
- Not a fake local Mac mode
- Not silent substitution when assets or endpoints are missing
- Not full humanoid locomotion training
- Not real robot control
- Not mocked VLA by default

## Strict Runtime Behavior

The default path is intentionally strict.

- Missing `VLA_ENDPOINT` fails
- Missing Unitree G1 USD asset fails
- Missing locomotion policy/controller fails
- Missing Isaac Sim / Isaac Lab fails
- Missing livestream support fails if `--live webrtc` is requested

Explicit fallback flags are required for development-only paths:

- `--allow-proxy`
- `--allow-kinematic-control`
- `--allow-rule-planner`

When used, these are printed and logged as non-primary development behavior.

## Repository Layout

```text
pathvla-unitree-isaac-live/
├── README.md
├── Makefile
├── .env.example
├── requirements-dev.txt
├── docker/
├── config/
├── pathvla/
├── isaac_ext/
├── apps/
├── scripts/
├── eval/
├── outputs/
└── tests/
```

## Prerequisites

- Ubuntu Linux VM with NVIDIA GPU
- `nvidia-smi` working
- Docker with NVIDIA container runtime
- Isaac Sim or Isaac Lab available via a compatible official base image or existing install
- Mac used only as remote client over SSH, browser, or Isaac livestream client

## Required Environment Variables

- `ISAAC_BASE_IMAGE`
- `VLA_ENDPOINT`
- `VLA_API_KEY` optional
- `VLA_MODEL_NAME` optional
- `OPENAI_API_KEY` for the bundled real VLA server
- `OPENAI_MODEL` optional, defaults in `.env.example`
- `UNITREE_G1_USD_PATH`
- `HOST_WORKSPACE_ROOT`
- `BREV_PUBLIC_HOST`
- `LIVESTREAM_PORTS`

Optional:

- `PATHVLA_OUTPUT_ROOT`
- `PATHVLA_API_HOST`
- `PATHVLA_API_PORT`
- `PATHVLA_DASHBOARD_PORT`

See [.env.example](./.env.example).

## Brev Setup

1. Create a Brev or equivalent Ubuntu GPU VM with NVIDIA drivers installed.
2. SSH from Mac into the VM.
3. Clone this repository.
4. Set environment variables.
5. Build the Isaac container.
6. Run the validation checks.

Commands:

```bash
bash scripts/setup_brev.sh
make check-gpu
make check-isaac
make check-livestream
make build
make live-demo
```

## Isaac Container Setup

This repo does not hardcode an Isaac image tag because NVIDIA image tags and access patterns change over time.

Set:

```bash
export ISAAC_BASE_IMAGE="<official-compatible-isaac-lab-or-isaac-sim-image>"
```

Examples depend on your NVIDIA entitlement and installation path. The Dockerfile uses:

```dockerfile
ARG ISAAC_BASE_IMAGE
FROM ${ISAAC_BASE_IMAGE}
```

You must supply a real Isaac-compatible base image.

## Live Viewing From Mac

### WebRTC / Isaac Livestream

Use:

```bash
make check-livestream
make live-demo
```

The system prints exact connection instructions including `BREV_PUBLIC_HOST` and the configured ports from [config/livestream.yaml](./config/livestream.yaml).

Open or forward the required ports on the VM. Then connect from the Mac using the Isaac WebRTC streaming client or the supported browser/client path documented by your Isaac build.

### Remote Desktop Alternative

Use `--live remote_desktop` and provision NICE DCV, VNC, or RDP separately. This path is documented, but it is not used as a silent fallback for broken livestream mode.

### Recorded Replay

Use:

```bash
make recorded-demo
```

This runs headless and records logs plus video when capture support is available.

## Main Strict Run On Brev

```bash
export VLA_ENDPOINT="https://your-vla-server/infer"
export HOST_WORKSPACE_ROOT="/home/shadeform/workspace"
export UNITREE_G1_USD_PATH="/host_workspace/unitree_assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
export BREV_PUBLIC_HOST="<your-brev-host>"
export ISAAC_BASE_IMAGE="<official-compatible-isaac-lab-or-isaac-sim-image>"

make setup-brev
make check-gpu
make check-isaac
make check-livestream
make build
make live-demo
```

Expected outcome:

- Starts Isaac Sim or Isaac Lab
- Builds the selected room or warehouse scene
- Loads Unitree G1 asset
- Calls the real VLA endpoint
- Validates returned JSON
- Plans waypoints using live semantic scene state
- Executes movement in the Isaac simulation
- Exposes livestream for Mac viewing
- Records outputs under `outputs/`

## Development-Only Fallback Run

This remains a real Isaac Sim or Isaac Lab run, but it is not the primary claim.

```bash
make live-demo ALLOW_PROXY=1 ALLOW_KINEMATIC=1 ALLOW_RULE_PLANNER=1
```

This prints that:

- a proxy robot is being used instead of a real Unitree G1 asset
- kinematic control is being used instead of realistic locomotion
- rule planner mode is enabled instead of VLA mode

## API And Dashboard

Start the API:

```bash
make api
```

Start the Streamlit dashboard:

```bash
make dashboard
```

The dashboard controls real Isaac runs. It does not emulate simulation in the browser.

## Real VLA Server

This repo now includes a real HTTP VLA server at [apps/vla_server.py](./apps/vla_server.py).
It supports either:

- OpenAI cloud models
- a same-host or in-network OpenAI-compatible server, such as a locally served Gemma 4 endpoint

Two supported deployment modes:

1. Docker Compose service on the same VM

```bash
docker compose -f docker/docker-compose.yaml up -d vla-server
export VLA_ENDPOINT="http://vla-server:5555/infer"
python scripts/test_vla_endpoint.py
```

2. Host process on the same VM

```bash
make vla
export VLA_ENDPOINT="http://host.docker.internal:5555/infer"
python scripts/test_vla_endpoint.py
```

Important:

- `127.0.0.1` does not work from the Isaac container unless the VLA server is running inside that same container.
- If the VLA server runs as a separate process on the same VM, use `host.docker.internal`.
- If the VLA server runs as the bundled compose service, use `http://vla-server:5555/infer`.

## Gemma 4 On The Same Host

If your Gemma 4 server runs on the same VM and exposes an OpenAI-compatible API, configure:

```bash
export VLA_LLM_BASE_URL="http://host.docker.internal:8000/v1"
export VLA_LLM_MODEL="gemma-4"
export VLA_ENDPOINT="http://vla-server:5555/infer"
```

Then start the planner service:

```bash
docker compose -f docker/docker-compose.yaml up -d vla-server
python scripts/test_vla_endpoint.py
```

In this setup:

- Isaac container -> `vla-server`
- bundled VLA server -> your same-host Gemma 4 server at `host.docker.internal`

About GR00T:

- GR00T can be part of a robotics stack, but it is not required for this repo's VLA planning endpoint.
- In this project, the VLA server is the natural-language-to-subgoal planner. Humanoid locomotion is a separate controller problem.

## Evaluation

Run the evaluation suite in recorded headless mode:

```bash
make eval
```

## Tests

Unit tests:

```bash
make test
```

Integration tests:

```bash
make integration-test
```

Integration tests require a configured Isaac environment and are marked with `pytest -m integration`.

## Troubleshooting

- `scripts/check_gpu.sh` fails: fix GPU driver or Docker NVIDIA runtime before anything else.
- `scripts/check_isaac.sh` fails: fix Isaac base image or host Isaac Python environment.
- `scripts/check_livestream.sh` fails: open the required ports and enable the supported Isaac streaming extensions.
- G1 asset missing: set `UNITREE_G1_USD_PATH` or rerun with `--allow-proxy`.
- No controller configured: provide a real controller config or rerun with `--allow-kinematic-control`.
- VLA invalid response: inspect `outputs/<run_id>/bad_vla_response.json`.
- VLA endpoint error or HTTP 500: strict mode fails loudly; only `--allow-rule-planner` enables a debug-only fallback planner.
