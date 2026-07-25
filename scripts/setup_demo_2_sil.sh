#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/demo_2/vendor/unitree_rl_gym"
REPOSITORY="https://github.com/unitreerobotics/unitree_rl_gym.git"
REVISION="276801e46c5d433564f24658bac64f254b7d2d4b"
POLICY_SHA256="cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"
CONFIG_SHA256="73044e7d355c61915695c16d6e09eb3efef46eec1e3d708fd3eb9157dfe3bbbb"
REAL_CONFIG_SHA256="fb6fd920c9180baf9a50c389c8119a9f25e0096626de7de982c127ecfbffad78"

mkdir -p "$(dirname "$DEST")"
if [[ -e "$DEST" && ! -d "$DEST/.git" ]]; then
  echo "Refusing to replace non-git path: $DEST" >&2
  exit 1
fi

if [[ ! -d "$DEST/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$REPOSITORY" "$DEST"
fi

origin="$(git -C "$DEST" remote get-url origin)"
if [[ "$origin" != "$REPOSITORY" ]]; then
  echo "Unexpected origin for $DEST: $origin" >&2
  exit 1
fi

git -C "$DEST" sparse-checkout init --cone
git -C "$DEST" sparse-checkout set \
  deploy/pre_train/g1 \
  deploy/deploy_mujoco/configs \
  deploy/deploy_real/configs \
  resources/robots/g1_description
git -C "$DEST" fetch --depth=1 origin "$REVISION"
git -C "$DEST" checkout --detach "$REVISION"

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

policy="$DEST/deploy/pre_train/g1/motion.pt"
config="$DEST/deploy/deploy_mujoco/configs/g1.yaml"
real_config="$DEST/deploy/deploy_real/configs/g1.yaml"
actual_policy="$(sha256_file "$policy")"
actual_config="$(sha256_file "$config")"
actual_real_config="$(sha256_file "$real_config")"

if [[ "$actual_policy" != "$POLICY_SHA256" ]]; then
  echo "Official G1 policy hash mismatch: $actual_policy" >&2
  exit 1
fi
if [[ "$actual_config" != "$CONFIG_SHA256" ]]; then
  echo "Official G1 MuJoCo config hash mismatch: $actual_config" >&2
  exit 1
fi
if [[ "$actual_real_config" != "$REAL_CONFIG_SHA256" ]]; then
  echo "Official G1 real-deploy config hash mismatch: $actual_real_config" >&2
  exit 1
fi

echo "Pinned official Unitree G1 policy and MuJoCo model at $DEST"
echo "Revision: $REVISION"
