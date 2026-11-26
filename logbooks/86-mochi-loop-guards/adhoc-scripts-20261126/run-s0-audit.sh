#!/bin/zsh
set -euo pipefail
set -a
source ~/.config/marin/credentials/cw-s3.env
set +a
uv run --no-sync iris --cluster cw-us-east-08a job run --no-wait --priority production \
  --cpu 16 --memory 64GB --disk 128GB --enable-extra-resources --timeout 14400 \
  --sync-package marin-core --job-name mochi-loop-guards-s0-audit-20261125a \
  -e MARIN_PREFIX s3://marin-us-east-08a/marin -- \
  uv run --package marin-core python -m experiments.post_training.mochi_loop_guards.audit \
  --run-output s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-f28d12bf4c-in-af5ab9b8/2026.11.25.9 \
  --output s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9
