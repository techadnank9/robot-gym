# Demo 6 — 3× directional locomotion

Demo 6 reuses the complete Demo 5 VLGE/MuJoCo arena, perception, grasp,
release, recovery, model adapters and 50 Hz SDK-shaped command channel. It
changes only the locomotion profile:

- forward/backward command ceiling: **1.56 m/s** (3 × 0.52);
- left/right command ceiling: **0.78 m/s** (3 × 0.26);
- yaw command ceiling: **0.90 rad/s** (unchanged for controllability);
- planar command slew: 3× Demo 5, so a joystick reaches the higher ceiling
  without an extra-long acceleration ramp.

The profile applies to both human directional input and AI navigation so a
match remains fair. Demo 5 stays at its original limits.

This is an arcade-speed simulation profile. The pinned Unitree policy still
produces the joint actions, but commands above its original operating range
must not be treated as proof that a real G1 can safely achieve these speeds.

## Idle versus local gamepad

```bash
scripts/run_g1_demo_6.sh match \
  --p1 human --p1-input idle \
  --p2 human --p2-input gamepad --p2-gamepad 0
```

Demo 6 uses `http://127.0.0.1:8086/?wsPort=8766` by default.
