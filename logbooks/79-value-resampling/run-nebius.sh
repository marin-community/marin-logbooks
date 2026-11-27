#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=LAZY
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
exec uv run --no-project --python 3.12 --with 'torch==2.9.1' --with 'transformers==4.57.6' \
    --with 'accelerate==1.10.1' --with 'pyarrow==23.0.1' \
    python -u -m experiments.post_training.goal_value.nebius_score \
    "${1:?path to frozen value head}" "${2:?path to excluded issue IDs}" --issues 192 "${@:3}"
