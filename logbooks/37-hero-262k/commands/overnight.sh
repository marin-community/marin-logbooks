#!/usr/bin/env bash
# Six 11-rack arms at skew 4, one at a time, all from the interim commit 1eed8fff84.
#   overnight.sh DRY  <arm> <restore-path> <tag>             build the run locally, submit nothing
#   overnight.sh GO   <arm> <restore-path> <tag> <priority>  submit the arm's 100 training steps; priority has no default
#   overnight.sh BASE <arm> <restore-path> <tag>             no-training baseline eval of the restore checkpoint under the arm's recipe
#   overnight.sh EVAL <arm> <tag>                            offline eval of the arm's step-390351
# <tag> names the job, run and prefix, so a repeat with a new tag cannot collide. GO and DRY refuse a tag whose prefix has data.
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODE="${1:?DRY, GO, BASE or EVAL}"; ARM="${2:?arm}"
# --qk-mult-long-scale is explicit on every arm: the launch tree defaults to 1.133, so leaving it out would make the arm global 1.88.
case "$ARM" in
  keep166)     QK=(--qk-mult 1.66 --qk-mult-long-scale 1);     N=1 ;;
  log65536sq)  QK=(--qk-mult 1.66 --qk-mult-long-scale 1 --log-scaling-n-floor 65536 --log-scaling-alpha 0.1 --log-scaling-power 2); N=2 ;;
  global188)   QK=(--qk-mult 1.66 --qk-mult-long-scale 1.133); N=3 ;;
  log32768)    QK=(--qk-mult 1.66 --qk-mult-long-scale 1 --log-scaling-n-floor 32768 --log-scaling-alpha 0.1); N=4 ;;
  log65536a20) QK=(--qk-mult 1.66 --qk-mult-long-scale 1 --log-scaling-n-floor 65536 --log-scaling-alpha 0.2); N=5 ;;
  global200)   QK=(--qk-mult 1.66 --qk-mult-long-scale 1.205); N=6 ;;
  # The keep166 recipe from raw step-390251, for the paired comparison. Its job and prefix ignore <tag>; pass any tag.
  keep166-raw) QK=(--qk-mult 1.66 --qk-mult-long-scale 1);     N=7 ;;
  # First finalist's rehearsal, global layers only: 1.66 x 1.108 = 1.839. Held; not approved.
  global184)   QK=(--qk-mult 1.66 --qk-mult-long-scale 1.108); N=8 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
TREE=$WORKTREES/marin.hero-semifinal-overnight; COMMIT=1eed8fff84
LOADER_ENV=()
if [ "$MODE" = EVAL ]; then RESTORE=; TAG="${3:?tag}"; else RESTORE="${3:?restore path}"; TAG="${4:?tag}"; fi
[[ "$TAG" =~ ^[a-z0-9]+$ ]] || { echo "tag must be lowercase letters and digits"; exit 2; }
RUN=hero-lce11-$TAG-$ARM; PFX=s3://marin-us-east-08a/tmp/ttl=1d/hero-lce11-$TAG/$ARM; EVAL_RUN=hero-mrcr-lce11-$TAG-$ARM; RECIPE_ARM=$ARM
if [ "$ARM" = keep166-raw ]; then
  [ "$MODE" != BASE ] || { echo "raw's baseline exists: hero-mrcr-keep166-step390251"; exit 2; }
  [ "$MODE" = EVAL ] || [ "$RESTORE" = "$PARENT" ] || { echo "keep166-raw restores only $PARENT"; exit 3; }
  RUN=hero-lce11-keep166-raw; PFX=s3://marin-us-east-08a/tmp/ttl=1d/hero-lce11-keep166-raw; EVAL_RUN=hero-mrcr-lce11-keep166-raw; RECIPE_ARM=keep166
fi
# The semifinal's eval trees and flags, so raw's existing baselines stay comparable.
offline_eval() { # <run-id> <port> <checkpoint>
  local tree want
  if [ "$ARM" = log65536sq ]; then tree="$EVALROOT"sq; want=1b113f404c; else tree="$EVALROOT"; want=fcf4573757; fi
  cd "$tree"; [ "$(git rev-parse --short=10 HEAD)" = "$want" ] || { echo "eval tree moved: $tree"; exit 3; }
  iris_run "$1-coord" "$2" python -m experiments.grug.moe_hero_ep.launch_checkpoint_eval \
    --run-id "$1" --checkpoint "$3" $EVAL_COMMON ${RECIPE[$RECIPE_ARM]} --run
}
# Tag q64: the log65536sq arm at loader queue 64, from its own worktree at the SHA in q64.sha. It compares with the merged baseline.
if [ "$TAG" = q64 ]; then
  [ "$ARM" = log65536sq ] || { echo "tag q64 is only for log65536sq"; exit 2; }
  [ "$MODE" != BASE ] || { echo "q64 uses hero-mrcr-lce11-merged-log65536sq-base"; exit 2; }
  if [ "$MODE" != EVAL ]; then
    [ -s "$(dirname "$0")/q64.sha" ] || { echo "q64.sha not set: no SHA from hero-mem-growth yet"; exit 3; }
    TREE=$WORKTREES/marin.hero-semifinal-q64; COMMIT="$(cat "$(dirname "$0")/q64.sha")"
    LOADER_ENV=(-e MEMGROW_LOADER_BUFFER 64)
  fi
fi
# Arm global184 trains from its own worktree at hero-deploy's head, whose launcher has the schedule extension and queue 64 as constants.
EXTENSION_FLAGS=(--extension-steps 4335 --extension-final-lr-ratio 0.01 --optimizer-racks 11)
if [ "$ARM" = global184 ]; then
  [ "$TAG" = global184 ] || { echo "arm global184 takes tag global184"; exit 2; }
  EXTENSION_FLAGS=()
  if [ "$MODE" = DRY ] || [ "$MODE" = GO ]; then
    SHA_FILE="$(dirname "$0")/global184.sha"
    # DRY alone may use a preview head; GO needs the head hero-deploy sent.
    [ -s "$SHA_FILE" ] || { [ "$MODE" = DRY ] && SHA_FILE="$(dirname "$0")/global184.preview.sha"; }
    [ -s "$SHA_FILE" ] || { echo "global184.sha not set: no head from hero-deploy yet"; exit 3; }
    TREE=$WORKTREES/marin.hero-semifinal-global184; COMMIT="$(cut -c1-10 "$SHA_FILE")"
  fi
  RUN=hero-lce11-global184; PFX=s3://marin-us-east-08a/tmp/ttl=1d/hero-lce11-global184; EVAL_RUN=hero-mrcr-lce11-global184
fi
if [ "$MODE" = EVAL ]; then offline_eval $EVAL_RUN-step390351 $((32940+N)) "$PFX/checkpoints/step-390351"; exit 0; fi
if [ "$MODE" = BASE ]; then offline_eval $EVAL_RUN-base $((32920+N)) "$RESTORE"; exit 0; fi
# --num-steps and --long-context-start-step assume the restore is at 390251.
[ "$(rclone cat "marin-cw:${RESTORE#s3://}/metadata.json" | python3 -c 'import json,sys;print(json.load(sys.stdin)["step"])')" = 390251 ] || { echo "restore checkpoint is not at step 390251"; exit 3; }
[ -z "$(rclone lsf "marin-cw:${PFX#s3://}/" 2>/dev/null)" ] || { echo "prefix $PFX has data: pick a new tag"; exit 3; }
cd "$TREE"; [ -z "$(git status --porcelain)" ] || { echo dirty; exit 3; }
[ "$(git rev-parse --short=10 HEAD)" = "$COMMIT" ] || { echo "interim tree moved"; exit 3; }
FLAGS=(--run-id "$RUN" --seq-len 262144 --batch-size 176 --dp-racks 11 --context-axis-size 4 --expert-axis-size 16
  --num-steps 390351 --schedule-steps 390251 "${EXTENSION_FLAGS[@]}"
  --long-context-start-step 390251 --restore-from "$RESTORE" --capacity-factor 1.5 "${QK[@]}" --save-checkpoints
  --checkpoint-minutes 20 --checkpoint-path "$PFX/checkpoints" --eval-every 40 --gc-interval 100 --seed 0 --version dev)
if [ "$MODE" = DRY ]; then uv run python -m experiments.grug.moe_hero_ep.launch_diagnostics "${FLAGS[@]}"; exit 0; fi
[ "$MODE" = GO ] || { echo "unknown mode $MODE"; exit 2; }
PRIORITY="${5:?priority required}"
exec uv run iris --cluster=marin job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority "$PRIORITY" \
  --timeout 7200 --cpu 2 --memory 8GB --disk 32GB --job-name "$RUN-coord" -e WANDB_API_KEY "$WANDB_API_KEY" -e IRIS_PORT_JAX "$((32930+N))" \
  -e NCCL_DEBUG INFO -e NCCL_DEBUG_SUBSYS INIT "${LOADER_ENV[@]}" -- python -m experiments.grug.moe_hero_ep.launch_diagnostics "${FLAGS[@]}" --run
