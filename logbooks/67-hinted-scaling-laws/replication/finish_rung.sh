#!/bin/bash
# Finish a slow rung across all of its replicas: wait (up to ~25 min) for each endpoint, mint URLs for the
# ready ones, and sample the remaining prompts round-robin over them, skipping prompts in <label>.part0.csv.
# Usage: finish_rung.sh <label> <endpoint>...
here=/Users/dml/.claude/jobs/d6afba72/tmp/replication
results=$here/results
label=$1; shift
mint() {
  local name=$1 url
  for _ in $(seq 1 50); do
    if uv run iris --cluster marin endpoints list "$name" 2>/dev/null | grep -q "$name"; then
      url=$(uv run iris --cluster marin endpoints mint "$name" --ttl-hours 12 2>/dev/null | grep -o "https://[^ ]*" | tail -1)
      if [ -n "$url" ] && curl -sf -m 20 "${url%/}/v1/models" > /dev/null; then echo "$url"; return; fi
    fi
    sleep 30
  done
}
urls=""
for name in "$@"; do
  url=$(mint "$name")
  if [ -n "$url" ]; then urls="${urls:+$urls,}$url"; else echo "$label: $name not ready, skipped" >> "$results/$label.log"; fi
done
[ -z "$urls" ] && { echo "$label: no replica ready" >> "$results/$label.log"; exit 1; }
uv run python "$here/sample_hinted_gsm8k.py" "delphi_$label" "$urls" "$results" --concurrency 48 \
  --exclude "$results/delphi_$label.part0.csv" --tag ".part1" >> "$results/$label.log" 2>&1
