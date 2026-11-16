#!/bin/bash
set -euo pipefail
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt
exec /opt/biomnibench-judge/bin/python /tests/grade_with_retry.py
