#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Run once on each held task after the AgentX workers have stopped. The manifest
# is baseline-c32.json with source metadata naming this published diagnostic.
cd /app
exec env IRIS_OUTPUT_DIR=/iris/outputs/diagnostic-baseline PYTHONPATH=/app \
  uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python logbooks/19-hero-agentx-throughput/diagnostic_server.py \
  --manifest /tmp/diagnostic-baseline.json \
  --output /iris/outputs/diagnostic-baseline \
  --profile --serve-duration 5400
