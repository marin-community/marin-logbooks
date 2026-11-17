#!/bin/bash
cd $MARIN_CHECKOUT; set -a; source .env >/dev/null 2>&1; set +a; unset MARIN_PREFIX
r=$1; D=$DATA_DIR/b1005/$r; R=s3://marin-us-east-02a/marin/evals/$r/results/data
tabs=$(uv run fsutil ls $R/ 2>/dev/null | grep -v "^s3" | tr -d '/')
for t in $tabs; do
  case $t in
    samples|steps|evalchemy_results) ;;
    rollouts_v1) [[ $r == *-014646-* || $r == *-010843-* ]] || continue ;;
    blobs|_finestore_blob_parts) [[ $r == *pi-* || $r == *sotopia* ]] || continue ;;
    *) continue ;;
  esac
  uv run fsutil cp -r $R/$t $D/$t >/dev/null 2>&1 || echo "FAIL $r $t"
done
