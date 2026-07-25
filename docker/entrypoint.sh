#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pathvla-unitree-isaac-live
export PYTHONPATH=/workspace/pathvla-unitree-isaac-live:${PYTHONPATH:-}
exec "$@"
