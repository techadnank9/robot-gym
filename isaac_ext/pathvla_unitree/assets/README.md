# Assets

The official Unitree G1 29-DoF USD is downloaded into the ignored
`unitree_model/` vendor checkout so its Git LFS payload is not duplicated in
this repository. The checked-out copy is pinned in `unitree_g1.lock.json`.

Download or verify it with:

```bash
make download-g1
```

The Gemini sorting demo resolves that local copy automatically. To use a
different copy, set:

```bash
export UNITREE_G1_USD_PATH=/absolute/path/to/g1_29dof_rev_1_0.usd
```

The Gemini sorting path never substitutes a proxy model.

The separate native Mac runtime uses the G1-with-hands MJCF from MuJoCo
Menagerie under the project-level `assets/mujoco_menagerie/` checkout. Run
`make download-g1-mjcf` to fetch its pinned revision.
