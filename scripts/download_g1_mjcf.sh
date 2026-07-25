#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/assets/mujoco_menagerie"
EXPECTED_COMMIT="71f066ad0be9cd271f7ed58c030243ef157af9f4"

if [[ ! -d "$DEST/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
fi
if [[ "$(git -C "$DEST" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  git -C "$DEST" fetch --depth 1 origin "$EXPECTED_COMMIT"
  git -C "$DEST" checkout --detach "$EXPECTED_COMMIT"
fi
git -C "$DEST" sparse-checkout set unitree_g1

MJCF="$DEST/unitree_g1/g1_with_hands.xml"
[[ -s "$MJCF" ]] || { echo "G1 MJCF download failed: $MJCF"; exit 1; }
[[ -s "$DEST/unitree_g1/assets/pelvis.STL" ]] || { echo "G1 mesh download is incomplete"; exit 1; }
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "Unexpected Menagerie commit"; exit 1; }
echo "MuJoCo Menagerie G1-with-hands ready: $MJCF"
