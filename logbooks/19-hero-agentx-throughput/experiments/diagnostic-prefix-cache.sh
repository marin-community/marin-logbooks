#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd /app
exec env IRIS_OUTPUT_DIR=/iris/outputs/diagnostic-prefix-cache PYTHONPATH=/app \
  uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python logbooks/19-hero-agentx-throughput/diagnostic_server.py \
  --manifest /tmp/diagnostic-prefix-cache.json \
  --output /iris/outputs/diagnostic-prefix-cache \
  --prefix-cache --profile --profile-warmup-iterations 0 --serve-duration 5400
