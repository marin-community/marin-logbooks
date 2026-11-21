#!/bin/bash
# Serve every original-seed Delphi rung on Reno at production priority for the hinted-GSM8K replication.
# Run from the delphi-replication branch (marin-serve --priority). Output: name -> job id.
out=/Users/dml/.claude/jobs/d6afba72/tmp/delphi_serve_jobs.txt
: > "$out"
while read -r label repo gpu tp; do
  extra=()
  [ "$tp" != "1" ] && extra=(--tensor-parallel-size "$tp")
  bash /tmp/hsl-ref/scripts/with_cw_env.sh uv run marin-serve iris "$repo" --cluster marin --gpu "$gpu" \
    --target-cluster cw-rno2a --priority production --max-model-len 4096 --dtype bfloat16 \
    --name "delphi-$label-hinted" --endpoint-name "/serve/delphi-$label-hinted" --timeout-hours 8 --no-wait "${extra[@]}" \
    > "/Users/dml/.claude/jobs/d6afba72/tmp/serve_$label.log" 2>&1 &
  echo "$label $repo" >> "$out"
done <<'EOF'
3e18 marin-community/delphi-3e18-447Mparams-1.2Btokens H100x1 1
9e18 marin-community/delphi-9e18-550Mparams-2.9Btokens H100x1 1
2e19 marin-community/delphi-2e19-837Mparams-3.6Btokens H100x1 1
3e19 marin-community/delphi-3e19-998Mparams-5Btokens H100x1 1
5e19 rohithck/delphi-5e19-1.2Bparams-7.4Btokens H100x1 1
9e19 marin-community/delphi-9e19-1.4Bparams-10.6Btokens H100x1 1
2e20 marin-community/delphi-2e20-1.9Bparams-14.8Btokens H100x1 1
3e20 marin-community/delphi-3e20-2.5Bparams-18.6Btokens H100x1 1
6e20 rohithck/delphi-6e20-2.9Bparams-29.9Btokens H100x1 1
1e21 marin-community/delphi-1e21-3.4Bparams-46.3Btokens H100x1 1
1e22 marin-community/delphi-1e22-9.7Bparams-160Btokens H100x1 1
1e23 marin-community/delphi-1e23-25Bparams-628Btokens H100x2 2
EOF
wait
grep -h -i -E "job|endpoint|error" /Users/dml/.claude/jobs/d6afba72/tmp/serve_*.log | grep -v "^I2026" | head -40
