#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=LAZY
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
if [[ -z "${DAYTONA_API_KEY:-}" && -n "${RIGGING_SECRETS_DIR:-/run/rigging/secrets}" ]]; then
    DAYTONA_API_KEY="$(cat "${RIGGING_SECRETS_DIR:-/run/rigging/secrets}/DAYTONA_API_KEY")"
    export DAYTONA_API_KEY
fi
output="${IRIS_OUTPUT_DIR:-results}/paired-continuations.jsonl"
exec uv run --no-project --python 3.12 --with 'torch==2.9.1' --with 'transformers==4.57.6' \
    --with 'accelerate==1.10.1' --with 'daytona==0.200.2' --with 'swebench==4.1.0' \
    python -u -m experiments.post_training.goal_value.resampling \
    "${1:?path to prefix manifest}" "${2:?path to value head}" "$output" --actions 20 --tokens 6144 --replicates 1
