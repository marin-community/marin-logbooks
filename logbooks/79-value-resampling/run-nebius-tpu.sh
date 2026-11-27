#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_DISABLE_TELEMETRY=1
uv run --no-sync fsutil cp "${1:?path to packed token windows}" nebius-inputs.npz
exec uv run --no-sync python -u -m experiments.post_training.goal_value.nebius_tpu nebius-inputs.npz
