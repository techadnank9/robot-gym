#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARCHIVE="pathvla_outputs_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" outputs
echo "Created $ARCHIVE"
