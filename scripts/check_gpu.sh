#!/usr/bin/env bash
set -euo pipefail

echo "[gpu] Checking host GPU visibility"
nvidia-smi

echo "[gpu] Checking Docker GPU visibility"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo "[gpu] GPU checks passed"
