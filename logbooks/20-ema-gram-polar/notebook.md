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

### 2026-10-30 19:08 UTC — Fit the pilot into available host memory

A full scheduler diagnostic distinguished the capacity constraints:
179 matching workers had no free TPU chips, and one had free chips but only
158.5 GiB of host RAM available against the 200 GiB request. The autoscaler
reported two matching groups blocked by quota-pool tier monotonicity. Keep the
batch priority and quota policy unchanged. The source of the 200 GiB request
is Fray's generic 50%-of-host TPU reservation, not a measured requirement of
this model.

Reduce the common host-memory reservation to 128 GiB, retaining v4-8, 120 CPU
cores and the same TPU HBM. Actual host peak memory remains unmeasured; inspect
it when the worker runs. The data loader has bounded prefetching. This is a
reservation adjustment, not a change to batch size, precision, architecture,
optimizer, input data, schedule or evaluation. All 30 resolved pilot configs
were compared at the same version before and after the change: only
`resources.ram` changed from `200g` to `128g`; the remote submission resource
matches the config. All 14 experiment tests and lint/type checks pass.

At 19:07:40 UTC the original TPU child remained pending with no task attempts
or assigned worker. Replace that never-started request with a fresh coordinator
and version `2026.10.30.21`, preserving the old artifact identity and record.
Recheck that it is still pending immediately before replacement. If it has
started, retain that running job instead. No GPU allocation is involved.
The new resolved configuration is `first-pilot-128g-resolved.json`.

```bash
uv run --no-sync iris --cluster=marin job run --no-wait --no-cancel-on-exit \
  --job-name ema-gram-20-first-pilot-128g-20261030 --priority batch \
  --cpu 1 --memory 2GB --region us-central2 --extra cpu \
  --sync-package marin-core --sync-package marin-levanter --sync-package marin-dupekit \
  -e GIT_COMMIT "$(git rev-parse HEAD)" \
  -- python -m experiments.ema_gram.train \
  --plan logbooks/20-ema-gram-polar/first-pilot.json \
  --version 2026.10.30.21 --output /tmp/ema-gram-resolved.json --run
```

### 2026-10-30 19:10 UTC — Replacement accepted without releasing an accelerator

Immediately before replacement, the original TPU child was still pending;
its only task had an empty worker ID, zero attempts and no start timestamp.
The old CPU coordinator and unallocated TPU child then reached `killed`.
No accelerator allocation or training progress was released.

Iris accepted `/kaiyuew@stanford.edu/ema-gram-20-first-pilot-128g-20261030`
at 19:09:55 UTC from clean published source
`1430899ebd26830db3fbbf521d072cbca8dbc408`, with version `2026.10.30.21`.
The model, data and optimizer are unchanged; this is a fresh submission,
not checkpoint resume. Await assignment and measured host peak memory.

### 2026-10-30 19:27 UTC — Model-size host reservations

The 19:21 scheduler snapshot still showed zero attempts. One worker had free
matching TPUs but only 86.5 GiB host memory available, below the 128 GiB request.
Recomputed abstract full-state sizes for the exact 447M configuration and
128,256-token vocabulary: model 1,788,976,128 bytes, optimizer 5,561,858,072
bytes, total state 7,350,834,212 bytes. `state-shape-memory.json` records this
shape calculation. It allocates no arrays and runs no optimizer steps. This
is not a runtime peak measurement: compiler, checkpoint and data buffers are
additional. The native loader bounds its prefetch queue to 64 batches.

Use 64 GiB host RAM for all 447M variants and 128 GiB for all 998M variants.
Both retain v4-8, 120 CPU cores, unchanged TPU HBM and unchanged training
controls. A comparison of all 30 resolved pilots again found only
`resources.ram` changed; lint and type checks pass. Measure host peak after
assignment and investigate any OOM rather than assuming this estimate proves
runtime sufficiency.

If the 128 GiB child is still pending without a worker or attempts, replace it
with `ema-gram-20-first-pilot-64g-20261030`, using the preceding command with
version `2026.10.30.22`. Preserve any job that has started. The complete new
configuration is `first-pilot-64g-resolved.json`; no training setting is reduced.

### 2026-10-30 19:31 UTC — 64 GiB request accepted; waiting for TPU capacity

Immediately before replacement at 19:26:31 UTC, the 128 GiB child was pending,
with one task, no assigned worker, zero attempts and no start timestamp.
After canceling its CPU coordinator, both jobs reached `killed`. No accelerator
allocation or training progress was released.

Iris accepted `/kaiyuew@stanford.edu/ema-gram-20-first-pilot-64g-20261030`
at 19:26:43.818 UTC from published source
`0017f4d204a2260d22bb87116299d72b617d78fe` and version `2026.10.30.22`.
All reference data caches were reused. Child
`/kaiyuew@stanford.edu/ema-gram-20-first-pilot-64g-20261030/ema-gram-pilot-3e18-a0-seed0-a46345a749ca_70bb8dbd-2bf94274`
was created at 19:27:25.444 UTC. At 19:30 UTC it remained pending, with zero
failures/preemptions; the scheduler reported insufficient TPUs on 180 workers.
Keep this request queued. No training W&B run, optimizer step or measured
runtime peak is established. The existing online non-training logging gate
remains successful; the resource-only change does not alter tracking.

Artifacts will be written to
`gs://marin-us-central2/ema-gram-pilot-3e18-a0-seed0-a46345a749ca/2026.10.30.22`.
This is a fresh submission, not checkpoint resume. Contribution `daf40e45c`
contains the model-size reservations; its latest CI is still in progress.

### 2026-10-30 19:35 UTC — CI passes for the submitted resource configuration

The PR event monitor reported CI success at 19:34:40 UTC for contribution
`daf40e45cb79ca672de08de493cc9d453f973597`: 39 checks passed or were skipped,
with no failing or pending checks. The PR is open and requires review; no new
comments or reviews arrived alongside this event. CI validates code checks,
not accelerator execution, memory sufficiency or training quality. The existing
64 GiB TPU request and its log monitor remain in place.

## Final decision

Pending pilot, confirmation and checkpoint-probe evidence. No exponent is
selected and no training advantage is claimed. The implementation PR remains
open; its code has not been merged.
