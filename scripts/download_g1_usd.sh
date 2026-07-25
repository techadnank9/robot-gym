#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/isaac_ext/pathvla_unitree/assets/unitree_model"
ASSET_SUBTREE="G1/29dof/usd/g1_29dof_rev_1_0"
EXPECTED_COMMIT="b6a8942b0803b6c137e58cef12beb4b03e4a2fa7"

command -v git >/dev/null || { echo "git is required"; exit 1; }
git lfs version >/dev/null || { echo "git-lfs is required"; exit 1; }

if [[ ! -d "$DEST/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/unitreerobotics/unitree_model.git "$DEST"
fi

if [[ "$(git -C "$DEST" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  git -C "$DEST" fetch --depth 1 origin "$EXPECTED_COMMIT"
  git -C "$DEST" checkout --detach "$EXPECTED_COMMIT"
fi
git -C "$DEST" sparse-checkout set "$ASSET_SUBTREE"
git -C "$DEST" lfs pull --include="$ASSET_SUBTREE/**"

USD="$DEST/$ASSET_SUBTREE/g1_29dof_rev_1_0.usd"
[[ -s "$USD" ]] || { echo "G1 USD download is incomplete: $USD"; exit 1; }
BASE_USD="$DEST/$ASSET_SUBTREE/configuration/g1_29dof_rev_1_0_base.usd"
[[ -s "$BASE_USD" ]] || { echo "G1 base USD dependency is incomplete: $BASE_USD"; exit 1; }
BASE_SIZE="$(wc -c < "$BASE_USD")"
[[ "$BASE_SIZE" -gt 1000000 ]] || { echo "G1 base USD is still a Git LFS pointer: $BASE_USD"; exit 1; }

ACTUAL_COMMIT="$(git -C "$DEST" rev-parse HEAD)"
echo "Unitree G1 USD ready: $USD"
echo "Upstream commit: $ACTUAL_COMMIT"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || { echo "Unexpected asset commit: $ACTUAL_COMMIT"; exit 1; }
