#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == "Darwin" ]] || {
  echo "Demo 5's visible local runner currently requires macOS." >&2
  exit 1
}
[[ -x .venv-mac/bin/mjpython ]] || {
  echo "Run 'make mac-setup' first." >&2
  exit 1
}
[[ -f demo_2/vendor/unitree_rl_gym/deploy/pre_train/g1/motion.pt ]] || {
  echo "Run 'scripts/setup_demo_2_sil.sh' once to install the pinned Unitree policy." >&2
  exit 1
}

MODE="${1:-match}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "$MODE" in
  validate)
    exec .venv-mac/bin/python -m demo_5 --headless --validate-only "$@"
    ;;
  scripted)
    exec .venv-mac/bin/mjpython -m demo_5 \
      --p1-adapter scripted \
      --p2-adapter scripted \
      "$@"
    ;;
  match)
    exec .venv-mac/bin/mjpython -m demo_5 "$@"
    ;;
  *)
    echo "Usage: scripts/run_g1_demo_5.sh {validate|scripted|match} [options]" >&2
    exit 2
    ;;
esac
