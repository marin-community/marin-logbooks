#!/usr/bin/env bash
set -euo pipefail

# Run from this research checkout. Every GPU invocation starts a fresh server.
campaign=perfmax-snowball-tuning-01a1eabe
operation=${1:-plan}
point=${2:-baseline}

case "$point" in
  baseline) point_args=() ;;
  c64) point_args=(--session-concurrency 64) ;;
  prefill4096-diagnostic)
    point_args=(--set serving.max_num_batched_tokens=4096 --diagnostic-seconds 600)
    ;;
  prefill4096)
    point_args=(--set serving.max_num_batched_tokens=4096)
    ;;
  *) echo "Unknown point: $point" >&2; exit 1 ;;
esac

marin-env uv run --frozen --python 3.12 --package marin-core --extra cpu \
  python -m experiments.evaluation.knotmeter.cli "$operation" \
  --profile snowball-h100 --campaign "$campaign" \
  --priority interactive \
  --logbook https://github.com/marin-community/marin-logbooks/issues/57 \
  "${point_args[@]}"
