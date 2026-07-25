# Demo 2: Unitree G1 VLA, policy simulation, and SDK2 pilot

`demo_2` leaves Demo 1 unchanged. It reuses Demo 1's environment and VLA from a
separate runner, adds a dynamically simulated G1 locomotion policy, and keeps
the physical SDK2 path behind explicit safety gates.

## Strongest simulation evidence

Install the pinned official Unitree G1 policy and its exact MuJoCo/real-deploy
configuration:

```bash
scripts/setup_demo_2_sil.sh
.venv-mac/bin/pip install -r demo_2/requirements-sil.txt
```

Run the repeatable software-in-the-loop evidence suite:

```bash
scripts/run_demo_2_sil_evidence.sh
```

This is not base teleportation. The official TorchScript G1 policy receives the
same 47 observations at 50 Hz and produces the same 12 leg targets used by
Unitree's real SDK2 deployment example. MuJoCo integrates the official G1
dynamics at 500 Hz with the published gains and torque limits. The suite:

- drives forward, lateral, and yaw commands through `RealG1Controller`
- proves the production velocity, duration, finite-number, and zero-command
  guardrails reject unsafe inputs
- checks base height, tilt, finite state, joint ranges, torque saturation,
  velocity tracking, and displacement
- injects 40 ms command latency, 15% packet loss, and total packet loss
- verifies the command watchdog zeros motion during total loss
- records pinned source hashes and parity with Unitree's `rt/lowcmd` and
  `rt/lowstate` real-deployment configuration

Each scenario and the combined result are written to an
`outputs/demo_2_sil_*` directory as JSON evidence.

With `--execution-backend policy-sil`, the G1 visible in the sorting lab is
also policy-driven. Demo 2 mirrors the dynamic simulator's pelvis position,
height, quaternion, and all 12 leg-joint positions into the G1-with-hands
model. Navigation turns toward a smoothed A* route and then walks it using
bounded forward, lateral, and yaw policy commands. Demo 1's fixed-heading
position interpolation is not used by this backend.

This is materially stronger evidence than the visual command twin, but it is
not proof that an individual physical robot is safe. It does not model actuator
variation, floor friction, payload, calibration error, network scheduling, or
hardware faults. A restrained hardware validation remains required.

## Full Demo 1-equivalent runner

The full runner reuses Demo 1's exact:

- MuJoCo sorting-lab builder and G1-with-hands model
- main and overhead cameras
- Gemini Robotics-ER visual decision loop
- `navigate`, `pick`, `place`, and `finish` schema
- grounded-action safety validation
- A* waypoint planning, articulated MuJoCo IK, geometric completion checks,
  action plan, execution trace, camera frames, and optional video

Launch the complete environment and VLA loop with dynamic G1 policy simulation:

```bash
export GEMINI_API_KEY="<restricted Gemini Robotics-ER key>"
scripts/run_g1_demo_2.sh \
  --execution-backend policy-sil \
  --record-video
```

The four full-demo execution backends are:

| Backend | MuJoCo environment + VLA | Unitree SDK behavior |
|---|---:|---|
| `mujoco` | Full | No SDK commands |
| `policy-sil` | Full | Official G1 policy and dynamics, controller guardrails, recorded safety evidence |
| `sdk-shadow` | Full | Generates and records bounded SDK commands without hardware |
| `sdk-live` | Full | Sends bounded navigation through official SDK2; blocks unconfigured physical pick/place |

Validate the complete environment, cameras, G1 model, and SDK bridge without a
Gemini request:

```bash
scripts/run_g1_demo_2.sh \
  --execution-backend policy-sil \
  --headless \
  --validate-only
```

Full runs write `sdk_bridge_trace.json`. Policy SIL also writes
`sil_evidence.json`; SDK shadow writes `sdk_commands.json`. These sit alongside
the same action-plan, camera, trace, result, and video artifacts used by Demo 1.

`sdk-live` is intentionally not allowed to pretend that MuJoCo grasp success is
physical grasp success. It can execute navigation after every real-motion gate
and `--twin-aligned` are supplied, but it stops at the first `pick` or `place`
until a calibrated driver for the exact installed hand is added. The physical
sorting lab must also match the twin's measured object positions and coordinate
frame; MuJoCo camera images do not prove the real workspace is clear.

## What is implemented

- a pinned official G1 locomotion policy in the official dynamic MuJoCo model
- deployment-parity checks against Unitree's real low-level SDK2 configuration
- latency, packet-loss, watchdog, fall, joint, and torque evidence
- a separate MuJoCo G1-with-hands command twin for visual testing
- Unitree's official Python SDK2 `LocoClient`
- read-only FSM probing
- one finite, conservatively limited velocity command followed by an explicit
  zero-velocity stop
- reviewed, built-in Unitree arm actions while the base is stationary
- dry-run as the default backend
- fail-closed checks for Linux, network interface, FSM state, velocity,
  duration, operator presence, a clear test area, and a ready remote stop
- stop attempts after errors and keyboard interruption

This is a hardware compatibility pilot, not a claim that autonomous sorting is
ready. The `sorting` command is deliberately blocked because this repository
does not yet contain a calibrated G1 hand/gripper driver, camera-to-base
calibration, metric object-pose estimation, force/contact feedback, or a
whole-body manipulation controller.

## Supported host and robot assumptions

- dedicated Ubuntu Linux control computer
- Unitree G1 29-DoF with a firmware/AI-sport version compatible with the
  installed SDK2 checkout
- wired network interface connected and configured according to Unitree's
  developer quick start
- G1 in the vendor-supported high-level motion mode, not low-level debug mode
- trained operator holding the physical remote, with the area cleared and the
  emergency stop procedure already tested

Do not run hardware commands from macOS. The Mac can run dry-run validation and
the MuJoCo command twin.

## Run the MuJoCo version on Mac

Set up the existing isolated MuJoCo environment once:

```bash
make mac-setup
```

Then launch a visible bounded movement:

```bash
scripts/run_real_g1_demo_2.sh \
  --backend mujoco \
  --linger 5 \
  move --vx 0.15 --duration 0.50
```

Preview the allowlisted high-level arm pose:

```bash
scripts/run_real_g1_demo_2.sh \
  --backend mujoco \
  --linger 5 \
  arm-action --name right-hand-up
```

The launcher automatically uses `mjpython` on macOS. The MuJoCo base motion and
gait are kinematic visualizations of the same bounded command accepted by the
SDK2 backend. They are useful for inspecting direction, duration, limits, and
command sequencing, but they are not a dynamically validated locomotion policy.

Use `--headless --linger 0` for automated checks.

The `run_real_g1_demo_2.sh` commands in this section are low-level single-command
pilots. Use `run_g1_demo_2.sh` above for the complete sorting environment and
Gemini loop.

## Install on the Linux control computer

Unitree currently documents installing SDK2 Python from source after installing
CycloneDDS 0.10.x. Follow the official instructions rather than copying an
unreviewed SDK wheel:

1. [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
2. [Unitree developer quick start](https://support.unitree.com/home/en/developer/Quick_start)
3. [Unitree high-level motion services](https://support.unitree.com/home/en/developer/sports_services)

Then install this demo's ordinary dependencies:

```bash
python3 -m venv .venv-g1
source .venv-g1/bin/activate
python -m pip install -r demo_2/requirements.txt
```

Install the official SDK checkout into the same environment as Unitree
documents. `demo_2/sdk_lock.json` records the exact official revision against
which this adapter was reviewed:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git -C unitree_sdk2_python checkout 65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
python -m pip install -e ./unitree_sdk2_python
```

## Validate without a robot

Commands still default to dry-run unless `--backend mujoco` or `--backend sdk2`
is explicit:

```bash
python -m demo_2 probe
python -m demo_2 move --vx 0.05 --duration 0.25
python -m demo_2 arm-action --name right-hand-up
python -m demo_2 sorting
```

The last command must fail with an explanation; it proves the unsafe,
simulator-only behavior is not silently routed to hardware.

## Connect read-only

Replace `enp2s0` with the interface physically connected to the G1:

```bash
python -m demo_2 \
  --backend sdk2 \
  --network-interface enp2s0 \
  probe
```

`probe` sends no motion. It prints the current high-level FSM ID. The reviewed
default config accepts FSM `500`, matching the official SDK's `LocoClient.Start`
state. If the robot reports another state, stop and verify the installed robot
firmware and official documentation before reviewing `allowed_fsm_ids` in
`demo_2/config.yaml`.

For a live full-demo navigation trial on the Linux control host, all physical
safety gates are required:

```bash
scripts/run_g1_demo_2.sh \
  --execution-backend sdk-live \
  --network-interface enp2s0 \
  --headless \
  --enable-real-motion \
  --acknowledge MOVE_REAL_UNITREE_G1 \
  --operator-present \
  --remote-estop-ready \
  --area-clear \
  --twin-aligned
```

This command will stop with an explicit unsupported-capability error when
Gemini reaches physical manipulation. Do not bypass that boundary with a
simulator-only grasp.

## Send an explicit stop

The stop command is intentionally available without the motion authorization
flags:

```bash
python -m demo_2 \
  --backend sdk2 \
  --network-interface enp2s0 \
  stop
```

## First bounded motion

Put the robot in a manufacturer-approved test setup. Use a spotter and support
rig as appropriate for your lab. Keep the Unitree remote in hand.

```bash
python -m demo_2 \
  --backend sdk2 \
  --network-interface enp2s0 \
  --enable-real-motion \
  --acknowledge MOVE_REAL_UNITREE_G1 \
  --operator-present \
  --remote-estop-ready \
  --area-clear \
  move --vx 0.05 --duration 0.25
```

The command is below the default limit, lasts 250 ms, and is followed by a
separate zero-velocity command. The hard ceilings in code prevent the YAML from
raising speed above 0.25 m/s, lateral speed above 0.15 m/s, yaw rate above
0.35 rad/s, or command duration above one second.

Run a reviewed built-in arm action only while the base is stationary:

```bash
python -m demo_2 \
  --backend sdk2 \
  --network-interface enp2s0 \
  --enable-real-motion \
  --acknowledge MOVE_REAL_UNITREE_G1 \
  --operator-present \
  --remote-estop-ready \
  --area-clear \
  arm-action --name right-hand-up
```

Arm-action availability depends on the exact G1 edition and firmware. This path
uses Unitree's finite high-level action service; it does not publish raw joint
torques or use `rt/lowcmd`.

## Next work required for real sorting

Before enabling the original red/blue sorting task on hardware, implement and
validate these as separate reviewed components:

1. exact hand model and official hand driver (Dex3, gripper, or other)
2. camera intrinsics/extrinsics and robot localization
3. 6-DoF object and bin pose estimation with uncertainty bounds
4. collision-aware whole-body IK and balance-aware arm/waist coordination
5. force/contact/slip detection and grasp verification
6. independent workspace, joint, velocity, and torque safety monitors
7. hardware-in-the-loop tests, then tethered single-skill trials

Until those exist, `demo_2` will not accept Gemini `pick` or `place` decisions.
