#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd /app
exec env IRIS_OUTPUT_DIR=/iris/outputs/diagnostic-prefix-control PYTHONPATH=/app \
  uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python logbooks/19-hero-agentx-throughput/diagnostic_server.py \
  --manifest /tmp/diagnostic-prefix-control.json \
  --output /iris/outputs/diagnostic-prefix-control \
  --prefix-cache --serve-duration 3600
