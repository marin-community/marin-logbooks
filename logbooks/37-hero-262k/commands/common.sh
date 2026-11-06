# Sourced by the submit scripts. Refuses to run without an explicit GO and a built stats artifact.
ROOT=$WORKTREES/marin.hero-semifinal
EVALROOT=$WORKTREES/marin.hero-semifinal-eval
PARENT=s3://marin-us-east-02a/marin/grug/hero-ctx65k-step372k/2026.08.19.2/checkpoints/step-390251
SUFFIX=${SUFFIX:-r2}
EVAL_EVERY=${EVAL_EVERY:-100}
PORT_OFFSET=${PORT_OFFSET:-0}
PREFIX=s3://marin-us-east-08a/tmp/ttl=1d/hero-semifinal-20261105-$SUFFIX
STATS=s3://marin-us-east-02a/marin/grug/moe-hero/harrier-length-stats/2026.11.05
CAPACITY=1.5
SCRATCH=${SCRATCH:?results directory}
# Set WORKTREES to checkouts at the commits below and WANDB_API_KEY before sourcing.
ARMS=(keep166 global188 global200 log32768 log65536a20 log65536sq)
declare -A RECIPE=(
  [keep166]="--qk-mult 1.66"
  [global188]="--qk-mult 1.66 --qk-mult-long-scale 1.133"
  [global200]="--qk-mult 1.66 --qk-mult-long-scale 1.205"
  [log32768]="--qk-mult 1.66 --log-scaling-n-floor 32768 --log-scaling-alpha 0.1"
  [log65536a20]="--qk-mult 1.66 --log-scaling-n-floor 65536 --log-scaling-alpha 0.2"
  [log65536sq]="--qk-mult 1.66 --log-scaling-n-floor 65536 --log-scaling-alpha 0.1 --log-scaling-power 2"
  [global184]="--qk-mult 1.66 --qk-mult-long-scale 1.108"
)
declare -A PORT=([keep166]=32830 [global188]=32831 [global200]=32832 [log32768]=32833 [log65536a20]=32834 [log65536sq]=32835)
declare -A EVALPORT=([keep166]=32840 [global188]=32841 [global200]=32842 [log32768]=32843 [log65536a20]=32844 [log65536sq]=32845)
RESUMEPORT=32850
ZEROSTEPPORT=32851
# one save per arm, except global188, which rehearses mid-run save-then-train
declare -A CKPT_MIN=([keep166]=600 [global188]=20 [global200]=600 [log32768]=600 [log65536a20]=600 [log65536sq]=600)
# flags shared by every 262K training job; --eval-every 100 scores steps 390300 and 390351
TRAIN_COMMON="--seq-len 262144 --batch-size 16 --context-axis-size 4 --expert-axis-size 16 --schedule-steps 390251 --extension-steps 4335 --extension-final-lr-ratio 0.01 --optimizer-racks 11 --long-context-start-step 390251 --capacity-factor $CAPACITY --gc-interval 100 --save-checkpoints --seed 0 --version dev"
EVAL_COMMON="--context-axis-size 4 --mrcr --seq-len 4096 --seq-len 65536 --seq-len 262144 --version 2026.11.05"
iris_run() { # job-name port -- launcher args
  local job=$1 port=$2; shift 2
  uv run iris --cluster=marin job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive \
    --cpu 2 --memory 8GB --disk 32GB --job-name "$job" -e WANDB_API_KEY "$WANDB_API_KEY" -e IRIS_PORT_JAX "$port" -- "$@"
}
require_go() { [ "${1:-}" = "GO" ] || { echo "refusing: no go from the parent" >&2; exit 2; }; }
