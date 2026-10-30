---
title: "Quarter-power EMA-Gram preconditioning on Delphi"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/20"
related_issues: ["https://github.com/marin-community/marin/issues/10784"]
research_branch: "codex/20-ema-gram-polar"
marin_base_revision: "a16537859e11b68b7a1d4776421937fb8162ffc8"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: active
---

# Quarter-power EMA-Gram preconditioning on Delphi

## Question and decision criteria

Does alpha=1/4 improve held-out loss over alpha=0 and alpha=1/2 after
independent learning-rate and ridge tuning? Compare alpha in
{0, 1/8, 1/4, 1/3, 1/2}. The [protocol](../../experiments/ema_gram/README.md)
defines the EMA update, fixed controls, selection lock, paired seeds and
checkpoint probes. Training results are pending.

The initial comparison uses the 447M Delphi recipe. Thirty 250M-token pilot
cells tune rates and ridges. Expand edge winners before selecting settings.
Then run fresh 1.2B-token confirmations with three paired initialization/data
seeds. Norm-matched checkpoint probes and exact-SVD comparisons distinguish
step magnitude from directional and approximation effects. Choose and lock a
common held-out loss target from the pilot curves before confirmations.
Horizon and 998M-parameter follow-ups require reproducible confirmation evidence.

## Baseline

Implementation source: Marin commit
`0a71417cdcc76713669c3a014f07ba26358f3c79`, submitted in
[PR #10790](https://github.com/marin-community/marin/pull/10790).
Twenty-five focused tests pass. GitHub CI passed and automated review reported
no remaining actionable issues at this commit. These checks establish software
behavior, not training quality. The proposed four-CPU gradient-reduction test
was removed because it did not exercise the native trainer.

Earlier Ali experiments include [#6588](https://github.com/marin-community/marin/pull/6588)
and [#4919](https://github.com/marin-community/marin/issues/4919).
Their results do not establish the behavior of the new EMA-Gram update.

## Entries

### 2026-10-30 18:30 UTC — Execution plan

The initial pilot LR centers are 0.01, 0.005, 0.002, 0.001 and 0.0005 in
ascending alpha order. They are starting grids, not tuned recommendations.
The baseline center is near conventional Muon update scales; progressively
smaller centers for larger powers are exploratory because the trace-normalized
inverse powers can amplify weak Gram eigendirections. Selection will use
measured held-out loss and expand every edge winner. No center is assumed optimal.
`pilots.json` preserves all 30 cells; `first-pilot.json` selects the nearest
baseline rate to 0.01 for the first complete pilot. Pilot seeds are zero.

Use Iris CPU coordinators in us-central2; training children request TPU v4-8
beside the existing Delphi caches. Keep Iris's default preemption retry policy
and omit all queue, job and training-phase wall-clock limits. Do not release
any GPU allocation voluntarily. This plan requests TPUs, not GPUs.
Every training process uses fresh online W&B identity and incremental local
and artifact logs; only verified full-state restoration establishes resume.

Before training, execute the native online W&B logging check on an Iris CPU
worker. Local credential verification succeeded without printing credentials.
The Iris controller is reachable. No accelerator job has been submitted yet.
Publish this source and the plan before execution; record the resulting exact
commit and job identity in the next entry.

Commands (from the published research checkout):

```bash
uv run --no-sync iris --cluster=marin job run --no-wait --no-cancel-on-exit \
  --job-name ema-gram-20-logging-20261030 --priority batch --cpu 1 --memory 2GB \
  --region us-central2 --extra cpu \
  --sync-package marin-core --sync-package marin-levanter \
  -- python -m experiments.ema_gram.train \
  --plan logbooks/20-ema-gram-polar/first-pilot.json \
  --version 2026.10.30.20 --output /tmp/ema-gram-resolved.json --logging-check
```

After successful remote logging validation, submit the same launcher with
`--run` instead of `--logging-check` and a distinct coordinator name. Inspect
actual memory, step time, numerical stability and data/evaluation progress
before scheduling further pilot cells. Each pilot runs its requested 7,630
updates; this is not a wall-clock cutoff. Durable resolved configurations are
also available through the execution artifacts and online W&B run.

### 2026-10-30 18:35 UTC — Batch scheduling requirement

The first CPU logging-check submission was rejected before a job was launched:
the submitting budget identity permits at most the batch priority band. Use
`--priority batch` on coordinators; do not request interactive priority or
change account policy. No optimizer steps or W&B run occurred in this rejected
submission. The exact experimental source remains unchanged.

### 2026-10-30 18:35 UTC — Remote logging check submitted

Published execution source: `44c7eceff45d94ab2dce7c028f86bb16a34173be`.
Iris accepted CPU job `/kaiyuew@stanford.edu/ema-gram-20-logging-20261030`
with the command above and the permitted batch priority. Its workspace bundle
is 16.0 MB. This job performs no optimizer steps. Await successful completion
and the online W&B run before submitting the first TPU pilot.

### 2026-10-30 18:38 UTC — Include the tokenization import dependency

The logging check failed before W&B initialization: importing the native
training module imports tokenization, then `marin.datakit.normalize`, which
requires `dupekit`. The scoped environment included `marin-core` and
`marin-levanter` but omitted the existing `marin-dupekit` workspace package.
The failed task attempt was `/kaiyuew@stanford.edu/ema-gram-20-logging-20261030/0:0`;
Iris reported exit 1 after 41.7 seconds and zero preemptions. No optimizer
step ran. Add `--sync-package marin-dupekit` and use a fresh job identity for
the corrected logging check. Keep the model and optimizer configuration unchanged.

```bash
uv run --no-sync iris --cluster=marin job run --no-wait --no-cancel-on-exit \
  --job-name ema-gram-20-logging-r2-20261030 --priority batch \
  --cpu 1 --memory 2GB --region us-central2 --extra cpu \
  --sync-package marin-core --sync-package marin-levanter --sync-package marin-dupekit \
  -- python -m experiments.ema_gram.train \
  --plan logbooks/20-ema-gram-polar/first-pilot.json \
  --version 2026.10.30.20 --output /tmp/ema-gram-resolved.json --logging-check
```

### 2026-10-30 18:43 UTC — Fix the logging-check tracker lifecycle

The second remote check reached online W&B, creating
[run 64ef8844db12](https://wandb.ai/marin-community/marin/runs/ema-gram-logging-check-64ef8844db12),
but then failed before logging its success metric. `Tracker.__enter__` sets
the global tracker and returns None; the check incorrectly bound that return
value. Initialize the native tracker explicitly, log through that instance,
and call `finish()` in `finally` to flush and close it. This changes only the
non-training check, not optimizer behavior. The failed job was
`/kaiyuew@stanford.edu/ema-gram-20-logging-r2-20261030` at source
`fd963d4f2b8a410dea69d87bc4b11450179776d2`; no optimizer steps ran.

The fixed CLI passed a local online check:
[run ea1773cd64c5](https://wandb.ai/marin-community/marin/runs/ema-gram-logging-check-ea1773cd64c5).
Readback through the W&B API verified state `finished`,
`logging_check/success=1`, seed zero and the resolved 7,630-step configuration.
The 14 experiment tests and repository lint/type checks pass. Repeat the
corrected remote command with job name `ema-gram-20-logging-r3-20261030` before
training. W&B's automatic source capture exceeded its size guard; the published
Git source and Iris bundle provide provenance instead.

### 2026-10-30 18:45 UTC — Remote logging gate passed; first pilot launch

Iris job `/kaiyuew@stanford.edu/ema-gram-20-logging-r3-20261030` succeeded
(exit 0, 47.55 seconds). The [online W&B run](https://wandb.ai/marin-community/marin/runs/ema-gram-logging-check-243f407b06e2)
is `finished` with `logging_check/success=1`. API readback confirms seed 0,
7,630 planned steps, source `e4a476d18` and `git_dirty=false`; see
`remote-logging-check.json`. This is a non-training environment check.

Submit the first full pilot with alpha=0, LR=0.007071067811865476,
ridge=1e-4 and EMA horizon 100. It uses passive Grams, the 447M recipe,
batch 8 x context 4096 and all 7,630 requested updates. The resolved configuration
is `first-pilot-resolved.json`. Inspect numerical stability, measured step time,
actual memory and validation before launching the remaining grid. The artifact
root is `gs://marin-us-central2/ema-gram-pilot-3e18-a0-seed0-a46345a749ca/2026.10.30.20`.

```bash
uv run --no-sync iris --cluster=marin job run --no-wait --no-cancel-on-exit \
  --job-name ema-gram-20-first-pilot-20261030 --priority batch \
  --cpu 1 --memory 2GB --region us-central2 --extra cpu \
  --sync-package marin-core --sync-package marin-levanter --sync-package marin-dupekit \
  -e GIT_COMMIT "$(git rev-parse HEAD)" \
  -- python -m experiments.ema_gram.train \
  --plan logbooks/20-ema-gram-polar/first-pilot.json \
  --version 2026.10.30.20 --output /tmp/ema-gram-resolved.json --run
```

The CPU coordinator launches the TPU child through Fray. The child infers the
TPU dependency extra and inherits batch priority; preemption retries remain
at Iris defaults. Neither command nor training configuration adds a wall-clock
limit. Record the submitted source SHA and actual child/W&B identities after
submission; a queued coordinator or initialized run does not establish progress.

### 2026-10-30 18:46 UTC — First pilot coordinator accepted

Iris accepted `/kaiyuew@stanford.edu/ema-gram-20-first-pilot-20261030` at
18:45:55 UTC using published source
`84b33bd43738c0d0b1edcbbaee01e2f7a44ce70d` and the preceding command.
The 16.0 MB bundle comes from a clean checkout. Await actual child-task
progress and W&B initialization; this submission alone is not a training result.

### 2026-10-30 18:48 UTC — Waiting for TPU capacity

The coordinator reused all existing reference caches and created child
`/kaiyuew@stanford.edu/ema-gram-20-first-pilot-20261030/ema-gram-pilot-3e18-a0-seed0-a46345a749ca_d29904c3-dd3d12f5`.
At 18:48 UTC the child was pending with scheduler reason
`Insufficient TPUs (need 4, available 0)`. No training W&B run or optimizer
step is established. Retain the queued job; this is capacity wait, not an
application failure or authorization to alter cluster policy.

## Final decision

Pending pilot, confirmation and checkpoint-probe evidence. No exponent is
selected and no training advantage is claimed. The implementation PR remains
open; its code has not been merged.
