#!/bin/bash
cd $MARIN_CHECKOUT; set -a; source .env >/dev/null 2>&1; set +a; unset MARIN_PREFIX
r=$1; D=$DATA_DIR/lce40/$r; R=s3://marin-us-east-02a/tmp/ttl=30d/lce-junk-filter/evals/$r/results/data
for t in samples steps; do uv run fsutil cp -r $R/$t $D/$t >/dev/null 2>&1 || echo "FAIL $r $t"; done
