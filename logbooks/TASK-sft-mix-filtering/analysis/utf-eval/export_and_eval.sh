#!/bin/bash
# Wait for a permanent checkpoint of the utf-normalize arm, export it, then launch the six-benchmark eval.
# Usage: export_and_eval.sh <step>
set -u
STEP=$1
RUN=s3://marin-us-east-02a/tmp/ttl=30d/sft-data-patch-11188/runs/utf-normalize
UTF=$RESEARCH_CHECKOUT
EVAL=$EVAL_LAUNCHER_CHECKOUT
set -a; source $MARIN_CHECKOUT/.env >/dev/null 2>&1; set +a; unset MARIN_PREFIX
cd $UTF
until uv run fsutil ls $RUN/sft/checkpoints/step-$STEP/metadata.json >/dev/null 2>&1; do sleep 600; done
echo "$(date -u +%FT%TZ) step $STEP committed"
JOB=sft-patch-11188-export-utf-normalize-step$STEP-r1
uv run iris --cluster=marin job run --no-wait --target-cluster=cw-rno2a --enable-extra-resources --priority production --gpu H100x1 --cpu 32 --memory 700GB --disk 256GB --timeout 21600 --job-name $JOB -e JAX_PLATFORMS cpu -- python -m experiments.sft_new_mix.patch_arm export --trained $RUN/sft --step $STEP --output $RUN/serve-step$STEP --source-revision 589608f599314bb95fb4d00e961efbfdc8e7c7f7 >/dev/null 2>&1
echo "$(date -u +%FT%TZ) submitted /dml/$JOB"
while true; do
  S=$(timeout 180 uv run iris --cluster=marin job list --prefix /dml/$JOB 2>/dev/null | grep -E "^/dml/$JOB +" | awk '{print $2}')
  case "$S" in succeeded) break;; failed|killed) echo "export $S"; exit 1;; esac
  sleep 300
done
uv run fsutil ls $RUN/serve-step$STEP/hf/config.json >/dev/null 2>&1 || { echo "export missing config.json"; exit 1; }
echo "$(date -u +%FT%TZ) export done"
cd $EVAL
E=experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs
uv run python experiments/evaluation/cli.py launch --model-config $ANALYSIS_DIR/utf-eval/utf-normalize-step$STEP.yaml $(for b in aime24 olympiadbench math500 gsm8k-0shot humanevalplus ifbench; do echo --evalchemy-config $E/$b.yaml; done) --accelerator H100x8 --federated_cluster cw-rno2a --priority production --run-id sft11188-utf-normalize-s$STEP --no-wait 2>&1 | grep -vE "^(I|W)[0-9]{8}|unparseable|Traceback|File |\^|return|pydantic|Input should|For further|metrics\.|model\.config" | tail -5
echo "$(date -u +%FT%TZ) eval launched sft11188-utf-normalize-s$STEP"
