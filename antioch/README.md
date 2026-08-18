# G1 in a scanned library

A Unitree G1 humanoid fetches a book from a real library, reconstructed from an
iPhone LiDAR scan, simulated in NVIDIA Isaac Sim on Antioch's cloud GPUs, with
Gemini Robotics-ER 1.6 choosing what the robot does next.

## What runs where

```
iPhone LiDAR -> Polycam -> GLB -> Blender (clean, recentre) -> USDZ
                                              |
                                     antioch assets push
                                              v
   ER 1.6      head camera + world state  ->  one skill      ~13 s
   skills      geometry                   ->  vx, vy, yaw    50 Hz
   motion.pt   blind balance reflex       ->  joint targets  50 Hz
   PhysX       contacts and torques                          500 Hz
```

The deciding layer never touches joints. The locomotion policy never sees the
room. Each layer only knows what it needs.

## Layout

| path | what it is |
|---|---|
| `src/fetch_book.py` | the task: walk to the shelf, take the book, carry it back |
| `src/g1_control.py` | G1 locomotion, arm postures, scene construction |
| `src/gemini_er.py` | Gemini Robotics-ER 1.6 adapter, and a scripted control case |
| `src/g1_live.py` | walking only, for watching the gait |
| `src/library.py` | scenarios that assert the room loads and is navigable |
| `src/probe_*.py` | measurements: assets, joint order, arm reach |
| `policies/g1_motion.pt` | pinned `unitree_rl_gym` policy (31,404 parameters), not committed — see below |

## Running it

```bash
# The policy weights are fetched hash-verified by the parent repo, not
# redistributed here
bash ../scripts/setup_demo_2_sil.sh
cp ../demo_2/vendor/unitree_rl_gym/deploy/pre_train/g1/motion.pt policies/g1_motion.pt

uv sync
source .venv/bin/activate
antioch auth login

antioch run --timeout 14400 src/fetch_book.py                      # ER 1.6 decides
antioch run --timeout 14400 src/fetch_book.py -- --adapter scripted # no model, no key
antioch scenario run --scenario library_room_loads                  # evidence run
```

The Gemini key is read from `secrets/gemini_key` (one line, gitignored) or
`GEMINI_API_KEY`. `--adapter scripted` needs neither and is the control case
when the model does something surprising.

## The policy port

`motion.pt` was trained for MuJoCo. Three things do not carry across:

- **Joint order.** MuJoCo runs left leg then right; Isaac interleaves them.
  Resolved by name, never by index.
- **Frames.** The policy reads angular velocity and gravity in the base frame;
  Isaac reports world frame.
- **Recurrence.** The policy is an LSTM and keeps hidden state across calls.
  Not clearing it on respawn feeds the memory of falling into the next attempt
  — measured at 1.94 of action difference for an identical observation.

## Things that cost real debugging time

- A referenced scan is visual geometry only. Without an explicit collider the
  robot falls to z = -19.6.
- `world.step(render=True)` folds rendering into the physics advance and
  silently changes the policy's effective rate. Step physics, render
  separately.
- The scan floor and a ground plane both at z = 0 make every footfall a
  contest between two surfaces. Walk on the plane; keep the scan for looks.
- `RigidPrim` on a link inside an articulation is treated as its own body and
  reports free fall. `XFormPrim` reads the rendered transform.
- A held object that still has a collider interpenetrates the hand and the
  solver throws the robot across the room, a fraction of a second after a
  successful grasp.

Together these took the gait from falling every three metres to completing the
task with no falls.

## What this is not

The grasp is a **kinematic attach**, the mode robot-gym calls "easy": inside
the grasp radius the book is locked to the palm and its collider is disabled.
The fingers do close and the book does ride in the hand, but physics is not
doing the holding. Runs label it `assisted attach, not a mechanical grasp`.
Contact-based grasping needs arm IK and force closure, and is not built here.

The shelf position is a constant. It was *derived* by asking ER 1.6 to point at
the bookshelf in a rendered view and ray-casting that pixel onto the mesh, but
the robot does not re-derive it while running.

Gemini Robotics-ER 1.6 free tier allows 20 requests per day and 5 per minute.
Past that the adapter falls back to the scripted policy and says so in the log.

## Credit

Locomotion policy and task structure follow
[robot-gym](https://github.com/techadnank9/robot-gym), which runs the same
pinned `unitree_rl_gym` G1 policy in MuJoCo.
