#!/bin/bash
cd $MARIN_CHECKOUT
set -a; source .env >/dev/null 2>&1; set +a; unset MARIN_PREFIX
R=s3://marin-us-east-02a/marin/evals; r=$1; D=$2
for t in samples evalchemy_results; do uv run fsutil cp -r $R/$r/results/data/$t $D/runs/$r/$t >/dev/null 2>&1 || echo "FAIL $r $t"; done
