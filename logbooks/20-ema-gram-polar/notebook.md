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

### 2026-10-30 20:12 UTC — First worker attempt ended during dependency setup

The first TPU task was assigned at 20:08:37.313 UTC. Logs show dependency
installation starting at 20:08:39 UTC, including JAX, libtpu and the repository
packages. The attempt ended at 20:09:41.499 UTC with `worker_failed`:
`worker reconcile failure threshold exceeded` on worker
`marin-tpu-v4-reserved-8-us-central2-b-20261029-1717-d731f7f6-worker-0`.
This is infrastructure/worker failure evidence, not a measured optimizer failure.

At 20:12 UTC Iris reports the logical job `running`, `failure_count=0` and
`preemption_count=1`. The task itself is pending without a worker, with reason
`Retrying (attempt 1, last: task_state_worker_failed)`. Its displayed
`syncing dependencies` status belongs to the previous attempt; it does not
prove a new process has started. Preserve the native retry policy and job ID.
No agent cancellation, manual resubmission or allocation release occurred.

The W&B API returned no runs in group `ema-gram-2026.10.30.22` at 20:11 UTC.
No optimizer step, training loss, model runtime memory or checkpoint restoration
is established. This is an infrastructure retry, not checkpoint resume.
`first-pilot-attempt0-status.json` preserves the bounded controller snapshot.

### 2026-10-30 20:31 UTC — Baseline trains; first evaluation and checkpoint pass

Native attempt 1 now trains on four TPU v4 chips. Online W&B began at
20:15:52 UTC, before the training metrics, with fresh ID
`ema-gram-pilot-3e18-a0-seed0-a46345a749ca-2026.10.30.22-412aad49e64f`.
[Live run](https://wandb.ai/marin-community/marin/runs/ema-gram-pilot-3e18-a0-seed0-a46345a749ca-2026.10.30.22-412aad49e64f).
The captured API snapshot reaches step 942 and 30,900,224 tokens. C4 English
validation loss is 5.4673399925 nats at logged step 762; the full-state
checkpoint at step 763 was saved at 20:25:41 UTC. XProf steps 100–109 were
uploaded at 20:19 UTC. These are partial baseline measurements, not a quality
comparison. `baseline-first-validation.json` contains bounded evidence and
artifact paths.

The `marin` Finelog `iris.task` query for 20:08–20:28 UTC returned 106 samples
for attempt 1 through 20:26:54 UTC. Its cumulative observed host peak is
24.159 GiB; accelerator-memory telemetry is null. The host peak includes the
first evaluation/checkpoint interval, but it is not the final run peak.
Steady training steps around 885–893 took roughly 0.35 seconds, excluding
validation and checkpoint overhead. Alpha zero does not exercise spectral
Gram powers, so it cannot establish nonzero-alpha cost.

Actual W&B API readback omitted empty `adapter` and `trainer.mesh.shared_mapping`
mappings. Contribution `af94166ce` accepts only these empty defaults while still
rejecting nonempty mismatches and missing required fields. The real run config
passes all remaining controls; 15 experiment tests and lint/type checks pass.
The running training code is unchanged.

Queue one alpha=1/4 pilot at the initial center LR=0.002, ridge=1e-4, seed=0,
horizon=100, version `2026.10.30.22`. Its plan is `quarter-pilot.json` and
resolved config is `quarter-pilot-resolved.json`. Preserve the baseline while
this independent full 7,630-step pilot measures spectral-power stability,
throughput and memory. Keep at most these two pilot jobs until both resource
profiles are understood; do not launch the full sweep yet. All model/training
controls are shared, with the intended exponent/LR changes only. Reuse the
successful native online logging gate; the entrypoint and W&B wiring are unchanged.
Publish this source/config before submission.

```bash
uv run --no-sync iris --cluster=marin job run --no-wait --no-cancel-on-exit \
  --job-name ema-gram-20-quarter-pilot-20261030 --priority batch \
  --cpu 1 --memory 2GB --region us-central2 --extra cpu \
  --sync-package marin-core --sync-package marin-levanter --sync-package marin-dupekit \
  -e GIT_COMMIT "$(git rev-parse HEAD)" \
  -- python -m experiments.ema_gram.train \
  --plan logbooks/20-ema-gram-polar/quarter-pilot.json \
  --version 2026.10.30.22 --output /tmp/ema-gram-quarter-resolved.json --run
```

### 2026-10-30 20:32 UTC — Quarter-power pilot submitted

Iris accepted `/kaiyuew@stanford.edu/ema-gram-20-quarter-pilot-20261030`
at 20:31:35 UTC from clean published source
`9fc93eb412368c03470fb8cfb0dda0e9957519dc`. The 16.0 MB bundle contains the
published quarter-power plan and version `2026.10.30.22`. This is a second
independent pilot; the baseline continues unchanged. Submission alone does
not establish TPU assignment, training progress or spectral-power cost.

### 2026-10-30 20:58 UTC — Quarter-power training starts with substantial step cost

The quarter-power task reached dependency setup at 20:41:38 UTC and online
W&B at 20:42:24 UTC. Native attempt 0 is training on TPU v4 with fresh run
[9a9c9264a39c](https://wandb.ai/marin-community/marin/runs/ema-gram-pilot-3e18-a0.25-seed0-a473c0b6d69b-2026.10.30.22-9a9c9264a39c).
The actual API configuration passes all protocol controls against the published
quarter-power plan/source `9fc93eb412368c03470fb8cfb0dda0e9957519dc`.

Captured step 0 took 517.295 seconds, including startup/compilation work.
Steps 1–6 took 38.147–41.617 seconds (median 40.659 seconds),
versus roughly 0.35 seconds for the baseline's observed steady steps. This
is early full-step overhead, not an isolated eigendecomposition measurement
or a matched convergence comparison. All captured training losses are finite;
there is no quarter-power validation result yet. The planned steps 100–109
trace will provide finer attribution. Do not change the requested full spectral
powers, matrix geometry, refresh cadence or 7,630-step schedule to hide the cost.
Keep only the two current pilots while investigating it.

A bounded `marin` / `iris.task` query for 20:41–20:58 UTC returned 136 samples
through 20:56:44 UTC. Attempt 0's cumulative observed host peak is 55.289 GiB;
accelerator-memory values are null. This precedes the first checkpoint interval
and does not establish final peak memory. `quarter-first-steps.json` preserves
the actual early step measurements and telemetry provenance.

Both runs emit the same reference-cache metadata warnings (cached max length
131072 versus configured None, and cached append_bos False versus configured
True). Their resolved data configs match the shared recipe; no caches were
rebuilt or substituted. These shared warnings do not explain a variant-only
change in input data.

The baseline continues: at 20:56 UTC its API snapshot reaches step 4302 and
141,000,704 tokens, with latest C4 validation loss 4.349377. These are partial
results, not a completed pilot. The report compatibility fix `af94166ce` passed
all 39 CI checks (passed/skipped, none failed/pending) at 20:35:34 UTC.

### 2026-10-30 21:18 UTC — Quarter-power pilot fails at the host-memory limit

The quarter-power child failed at 21:02:45 UTC with exit code 137. Iris reports
`OOM killed (container exceeded memory limit)`, and the task log records a
kernel OOM kill. W&B marks its fresh run as crashed at logged step 7, with
262,144 tokens and finite loss 12.168434. The bounded checkpoint-log query
returned no saved checkpoint. No checkpoint restoration is established.
`quarter-oom.json` preserves the controller, W&B and telemetry evidence.

The `marin` Finelog query over `iris.task` for 20:40–21:05 UTC returned 189
samples for attempt 0 through 21:02:37 UTC, with observed peak host memory
62.9296875 GiB against a 64 GiB reservation. Accelerator-memory telemetry is
null. Samples were around 44 GiB during steps 1–7; the sampled peak rose near
the kill. The last emitted artifact was `train_step.hlo.txt`; that proximity
does not establish the cause of the later memory spike. The planned trace at
steps 100–109 and the first validation were never reached.

Contribution `d19131429` raises future reservations to 128 GiB, uniform across
variants. A same-version comparison of the quarter-power resolved config
changes only `resources.ram` from `64g` to `128g`. Fifteen experiment tests and
lint/type checks pass. This provides headroom but has not been validated by
execution. The current baseline keeps its original reservation and continues:
at 21:13 UTC W&B recorded step 6146, 201,424,896 tokens and latest C4 English
validation loss 4.237224. No training advantage is claimed.

The prepared recovery uses the unchanged `quarter-pilot.json` plan with
version `2026.10.30.23` and a new coordinator name. The updated launcher sets
the TPU training child to 128 GiB; `--memory 2GB` below reserves memory only
for the CPU coordinator. `quarter-pilot-128g-resolved.json` is the dry-run
output of that plan and launcher. It preserves alpha=1/4, LR=0.002,
ridge=1e-4, seed=0, full spectral powers, 7,630 steps, online W&B and native
preemption retries. It would start a fresh process and W&B run. The existing
200 GiB and 128 GiB canceled baseline requests remain terminal.

Recovery is not submitted. The use-iris monitoring reference says to stop for
OOM/HBM failures; obtain the user's decision on this prepared recovery before
launch. Continue baseline completion checks and PR monitoring in the meantime.

```bash
uv run --no-sync iris --cluster=marin job run --no-wait --no-cancel-on-exit \
  --job-name ema-gram-20-quarter-pilot-128g-20261030 --priority batch \
  --cpu 1 --memory 2GB --region us-central2 --extra cpu \
  --sync-package marin-core --sync-package marin-levanter --sync-package marin-dupekit \
  -e GIT_COMMIT "$(git rev-parse HEAD)" \
  -- python -m experiments.ema_gram.train \
  --plan logbooks/20-ema-gram-polar/quarter-pilot.json \
  --version 2026.10.30.23 --output /tmp/ema-gram-quarter-128g-resolved.json --run
```

### 2026-10-30 21:43 UTC — First baseline pilot completes

The alpha=0, LR=0.007071067811865476, ridge=1e-4, seed=0 pilot completed
all 7,630 updates and 250,019,840 tokens. Iris reports successful child exit 0
at 21:29:39 UTC and successful coordinator completion at 21:30:02 UTC. W&B
is finished at zero-based step 7629. Final C4 English validation loss is
3.91522193 nats; the export contains ten evaluations and all 7,630 training
loss records, with no nonfinite values. The median recorded training-step
duration after step 0 is 0.347409 seconds, excluding evaluation overhead.
The last evaluation is recorded 4,416.48 seconds after the W&B runtime origin;
that clock excludes queueing and the earlier worker-failed attempt.

The final full-state checkpoint's `metadata.json` was read successfully from
`gs://marin-us-central2/ema-gram-pilot-3e18-a0-seed0-a46345a749ca/2026.10.30.22/checkpoints/step-7629`.
It records step 7629 and `is_temporary=false`. This verifies saved metadata;
no checkpoint restoration test has been performed. The attempt's durable
metrics contain 466 chunks, including the final log, summary, finish and
device-memory events in the last chunk.

The bounded `marin` / `iris.task` query for 20:14–21:32 UTC returned 648
samples through 21:29:33 UTC for native attempt 1. The maximum of its sampled cumulative
host-memory peak counter is 28.599609 GiB. Finelog accelerator-memory telemetry is
null, but the persisted JAX exit statistics report per-device allocator peaks
of 5.3600, 5.0375, 5.3598 and 5.0375 GiB. These allocator peaks do not measure
total device HBM occupancy and do not characterize nonzero-alpha spectral powers.

`baseline-completed.json` contains the result, curve, memory and checkpoint
evidence. `initial-attempts.json` and `initial-results.json` retain the completed
baseline and failed quarter-power attempt separately. The reporting CLI
validated both actual configurations and exported unsampled histories,
configuration and summary data. Gzip exports are preserved under the baseline
artifact root's `analysis/20261030-initial-attempt-report/`; their uploaded
bytes were read back and checked against SHA-256 hashes recorded in the
completion evidence.

The baseline is one completed pilot at one learning rate, not a tuned winner.
Quarter-power recovery remains unsubmitted pending the user's decision on the
prepared 128 GiB configuration. No new experiment was launched. PR #10790's
`d19131429` had all 39 CI checks passed or skipped at 21:22:26 UTC; the PR
remains under monitoring. The broader pilot sweep, locked confirmations and
checkpoint probes remain outstanding.

### 2026-10-30 22:04 UTC — Baseline XPlane attribution

The raw steps 100–109 XPlane contains ten `jit__train_step` module executions
on each of eight TPU trace planes. Their per-plane median durations range
from 0.322850 to 0.322954 seconds. The original 93,075,364-byte protobuf,
analysis source and output were copied into the same-region baseline artifact
root under `analysis/baseline-profile/` and verified by SHA-256 readback.
[The interval evidence](baseline-profile-intervals.json) records the complete
artifact URIs, hashes and method.

For each plane, clip annotated XLA Ops and Async XLA Ops to the ten training
module windows, then take the union of overlapping integer-picosecond intervals
separately for each category. Intervals tagged `update_gram_state` cover
16.59–16.60% of module time; `gram_direction` covers 13.57–13.58%; collective
annotations cover 27.85–27.98%. Categories may overlap and must not be added.
These annotation-based fractions do not measure exposed communication,
critical-path cost, or the complete optimizer cost: fusion and missing scopes
limit attribution. Alpha zero does not execute nonzero Gram spectral powers.

The generic `profile_summary.py` report inferred 126 steady-state steps from
the device Steps tracks, which also contain auxiliary modules. Its step-time
and jitter conclusions are therefore rejected for this run. The explicit
training-module windows above avoid that ambiguity. A host `train` annotation
also covers asynchronous dispatch rather than full device execution; its roughly
75 ms duration is not the training throughput. Keep the measured W&B median
0.347409 seconds as the recorded W&B training-step duration.

The failed quarter-power attempt never reached its steps 100–109 trace. A
bounded `marin` / `iris.profile` query for its exact task source during
20:50–21:03 UTC returned no CPU profile rows. Its log shows HLO lowering at
20:57:18–20:57:20, before the 21:02:45 OOM; the available record does not identify
the allocation that caused the later host-memory spike. No recovery or new
training was launched during this analysis.

### 2026-10-30 23:43 UTC — Authorized 128 GiB quarter-power recovery submitted

The user explicitly authorized the prepared recovery. Iris accepted
`/kaiyuew@stanford.edu/ema-gram-20-quarter-pilot-128g-20261030` at 23:42:58 UTC
from clean published source `44f8a1efdd8117260565f77965fcadd36cef717b`.
A fresh dry-run configuration exactly matched `quarter-pilot-128g-resolved.json`:
version `2026.10.30.23`, 128 GiB host RAM for the TPU child and 7,630 updates.
The coordinator uses 1 CPU and 2 GB; the training child keeps v4-8 and 120 CPUs.
The previously recorded submission command was used without a wall-clock limit
or preemption-retry override. The task-prefix query showed no existing recovery
before this single submission.

The recovery is configured to start a fresh process and W&B run; no checkpoint
restoration is claimed.
The prior OOM attempt stays in the record, and the completed baseline is not
restarted. Submission does not establish TPU assignment or training progress.
The earlier pending-confirmation state is resolved. Continue the resource and
stability gate before expanding the pilot sweep: measure nonzero-alpha step
cost, host/device memory and numerical stability, including the planned trace.

### 2026-10-31 07:22 UTC — Recovery trains after a native coordinator retry

The coordinator's first worker failed at 06:16:46 UTC. Iris retried the
coordinator as native attempt 1 at 06:16:54 UTC. The original child ending
`07f362a2-a2acc30b` was killed with reason `Parent task preempted`; it had no
worker, start time or execution attempts. The replacement child ending
`07f362a2-a3e52229` started at 06:17:34 UTC on native attempt 0. No manual
resubmission occurred and no checkpoint restoration is claimed.

Online W&B run
[d8755c4b4082](https://wandb.ai/marin-community/marin/runs/ema-gram-pilot-3e18-a0.25-seed0-a473c0b6d69b-2026.10.30.23-d8755c4b4082)
was created at 06:18:17 UTC before the first optimizer-step metrics. Its
metadata identifies four TPU v4 chips; actual model, optimizer, data and
trainer controls match the prepared protocol configuration. The snapshot reaches zero-based step 60, 1,998,848 tokens
and training loss 10.461051, with no nonfinite training losses observed.
The native trainer logged a temporary checkpoint at step 56 at 07:13:25 UTC.
There is no validation result yet. `quarter-128g-first-training.json` records
these observations, the run identity and the bounded history.

The `marin` / `iris.task` query over 06:17–07:17 UTC returned 502 samples
through 07:16:31 UTC for the replacement child's attempt 0. Its observed
host-memory peak is 91.729492 GiB, exceeding the failed attempt's 64 GiB
reservation while remaining below the new 128 GiB reservation. This is not
the final peak; accelerator-memory telemetry remains null.

Captured steps 10–60 have median recorded duration 40.190493 seconds, about
116 times the baseline's 0.347409-second median over steps 1–7629. These
windows are not matched in training progress. The first step took
492.818 seconds including startup/compilation. A shape-only evaluation of the
actual Qwen3 model and optimizer routing finds 77 matrix weights across 11
layers. Nonzero alpha requests two full spectral powers per matrix: 121
1024-by-1024 and 33 4096-by-4096 eigendecompositions per update, batched in the
implementation. Alpha zero skips these decompositions. This establishes the
extra computational work, but does not isolate its runtime fraction or rule
out sharding/communication overhead. The planned steps 100–109 trace is still
needed. Extrapolating the current median gives roughly 85 hours for 7,630
steps before extra evaluation/checkpoint overhead; this is an estimate, not
a deadline or a reason to truncate the requested schedule. Keep the sweep
unexpanded while collecting the scheduled trace, memory peaks and finite-loss
evidence needed to assess the nonzero-alpha resource gate.

The local log and PR observers exited during a network outage. After current
Iris state and GitHub feedback were refreshed, one observer for the replacement
child and one PR waiter were restored. PR #10790 remains open, review required,
with no new feedback found. This observer recovery did not alter training.

### 2026-10-31 08:04 UTC — Quarter-power trace lost its device timeline

The scheduled steps 100–109 profile ran from 07:42:48 to 07:49:44 UTC and
uploaded at 07:51:14 UTC. The saved XPlane contains 1,402,079 host events and
no device planes. The training container's TPU driver log reports at
07:50:49 UTC that 2.42 GiB of non-droppable XSpace data exceeded its 2.00 GiB
limit. The uploaded profile therefore cannot attribute TPU time to spectral
powers, collectives or other operations. Reject the generic host-only
summary's stall percentages and dispatch-based step estimates for that
purpose. The [quarter-power W&B run](https://wandb.ai/marin-community/marin/runs/ema-gram-pilot-3e18-a0.25-seed0-a473c0b6d69b-2026.10.30.23-d8755c4b4082) remains active: at 07:56 UTC it reached step 114,
3,768,320 tokens and training loss 8.187638; the last step took 41.049 seconds.

The run's saved `train_step.hlo.txt:v12579` W&B artifact provides a narrower
structural result. Its pre-optimization StableHLO contains eleven eigensolver calls on
`11x1024x1024` batches and three on `11x4096x4096` batches. Each eigensolver
wrapper lowers the layer batch to an 11-iteration `while` loop that selects
one matrix and calls its eigensolver. Thus a batched input does not establish
parallel execution of the eleven layers. This is not the final device
schedule and does not establish an eigensolver runtime fraction or how work
is distributed across chips.

`quarter-profile-coverage.json` links the original XPlane, saved StableHLO,
analysis source and coverage result in the run's durable object-storage
prefix. All four uploaded artifacts passed SHA-256 readback checks. The
bounded `marin` / `iris.task` query over 07:15–08:00 UTC returned 383 samples
through 07:59:54 UTC on the current child attempt. The attempt-wide host-memory high-water mark remains
91.729492 GiB; the maximum sampled current usage for the same task in this
window is 57.049805 GiB. The peak and current-usage measurements have different
meanings.

Keep the running pilot and its full schedule. The next cost investigation
needs a shorter device capture, initially one step, with explicit coverage
validation before interpreting timings. The existing training entrypoint
starts a scheduled capture but does not start a JAX remote profiler server;
no supported live recapture endpoint has been confirmed. Prepare a separately
identified optimizer diagnostic matching this pilot's 77 matrix weights and
sharding if a safe live capture is not available. Preserve the full left/right
Gram matrices, exponent 1/4 and refresh on every update; use online W&B for
diagnostic provenance and keep at most two accelerator jobs. Before expanding
the pilot sweep, obtain usable device cost evidence, confirm host memory stays
within the 128 GiB reservation, and inspect finite training and validation
metrics. No quarter-power validation result is available yet.

### 2026-10-31 08:17 UTC — Prepare an isolated optimizer cost diagnostic

The quarter-power pilot continues unchanged, reaching step 137 with loss
7.610778 and last-step duration 40.757 seconds at 08:11 UTC. A supported live
JAX profiler endpoint has not been found. Prepare one separate TPU v4-8
optimizer diagnostic to obtain a shorter device trace without restarting the
pilot or submitting a duplicate training run.

`optimizer_cost.py` calls the existing production Gram update, spectral-power
and polar-direction functions. It uses the pilot's seven matrix batches:
four `11x1024x1024`, two `11x4096x1024`, and one `11x1024x4096`. Optimizer state
is replicated, as in the saved StableHLO; gradient and direction arrays shard
rows for `o_proj`/`down_proj` and columns for the other five projections over
four chips. Synthetic FP32 gradients have seed
0 and expected combined Frobenius norm 0.1. Momentum and both Gram buffers start at zero. The input state is one prior
update using a different synthetic gradient from the same seeded stream.
Every timing reuses identical inputs.

Measure Gram refresh, full left/right quarter powers, alpha-zero direction,
and combined quarter-power refresh plus direction separately. Synchronize
outputs, record warm-up/compilation separately, then retain three execution
samples per case. Capture exactly one warmed combined invocation with Python
tracing disabled; verify that TPU device planes exist before interpreting it.
Spectral powers mean `(Gram / mean_eigenvalue + ridge I)^(-1/4)` on both
sides; the alpha-zero direction omits these powers and uses momentum directly
inside the polar approximation. Full matrices, per-update refresh, alpha 1/4,
ridge 1e-4 and five polar iterations match the live pilot. This diagnostic omits the language model, data,
AdamW and weight decay. Its synthetic spectrum and isolated kernels cannot
establish full-training time fractions or validation quality.

The small CPU end-to-end check passed finite-output checks. Directions from
the fused refresh-plus-direction function agreed with a separate refresh then
direction calculation at rtol 1e-4 and atol 1e-5. Its online W&B run
`ema-gram-20-cpu-smoke-fb05f631e002` finished; logging initialized before matrix
kernels, the report was read back from GCS, and the uploaded profile hashes
matched. `optimizer-cost-preflight.json` preserves the evidence and one
preceding local-output directory failure. CPU timings are not TPU performance
evidence. Lint passed for the new script.

`optimizer-cost-plan.json` requests one v4-8 task, 120 CPUs, 128 GiB host RAM
and 50 GiB disk in us-central2 at batch priority. It adds no wall-clock limit
or preemption-retry override. Publish this source and plan before the single
submission named `ema-gram-20-optimizer-cost-20261031`; this adds one diagnostic
beside the current pilot, with no other accelerator jobs for this experiment. Fresh process attempts use
separate online W&B identities and incremental local and GCS measurements.

### 2026-10-31 08:17 UTC — Submit the optimizer diagnostic once

Iris accepted `/kaiyuew@stanford.edu/ema-gram-20-optimizer-cost-20261031`
at 08:17:13 UTC from published source
`8790ebbdba20124e293de58b05d2824a78ca8ffc`. The exact-name prefix query was
empty before submission. The resource and measurement plan is unchanged from
`optimizer-cost-plan.json`. This is a separate non-training kernel diagnostic;
the quarter-power pilot continues on its existing child. Assignment and
diagnostic results remain unverified at submission.

### 2026-10-31 15:10 UTC — Quarter-power first validation

The EMA-Gram pilot with spectral exponent alpha=1/4 and a 128 GiB host-memory
reservation completed its first validation at step 762 (25,001,984 tokens).
C4 English validation loss is 5.67533445. All reported validation metrics are finite. The alpha-zero pilot's
C4 loss at the same step was 5.46733999, a difference of +0.20799446 for
quarter power. These are early single-seed pilot observations with different
learning rates (quarter power 0.002; alpha zero 0.00707107). Tuning and paired
confirmations remain incomplete, so this comparison does not select an exponent.

The online W&B run ending `d8755c4b4082` remains on child attempt 0 from
source `44f8a1efdd8117260565f77965fcadd36cef717b`, version `2026.10.30.23`.
Training continued past validation; the step-763 log reports 38.864 seconds
for its training step. Read back permanent checkpoint `step-763/metadata.json`:
step 763, `is_temporary=false`, timestamp `2026-10-31T15:11:40.099299`.
Checkpoint restoration has not been tested. `quarter-first-validation.json`
records the W&B URL, full checkpoint URI, metrics, and execution identity.

Across 510 Iris task host-memory samples from 14:11 to 15:11 UTC, the cumulative
attempt peak remains 91.7295 GiB; maximum sampled current memory is 48.3994 GiB.
These measurements do not establish device memory use. The separate optimizer
cost diagnostic remains queued for TPU capacity. Continue the pilot's full
7,630 updates and preserve that queued diagnostic; usable TPU execution-time evidence
is still required before expanding the sweep because each pilot update takes
about 40 seconds.

### 2026-10-31 17:00 UTC — Cancel the slow pilot; diagnose before training

Canceled the 128 GiB EMA-Gram alpha=1/4 pilot at the user's request. Iris confirmed the coordinator and its active TPU child
are `killed` at 17:00:54 UTC. The last W&B summary captured step 930,
30,507,008 tokens and training loss 5.80393124. The W&B API still reported
`running` immediately after cancellation; the authoritative Iris states are
terminal. This partial run did not finish its 7,630-update schedule.

The 51 measured update durations ending at step 905 had median 38.73798
seconds per update (range 38.56675–38.91911). Preserve the step-762 validation
and permanent step-763 checkpoint evidence in `quarter-first-validation.json`. The last temporary checkpoint
save log reports step 923 at 16:55:33 UTC; its completeness and restoration
have not been tested.
`quarter-user-canceled.json` records cancellation and process identities.

Keep `ema-gram-20-optimizer-cost-20261031` in the Iris batch queue. It remained
pending for TPU capacity at 17:00:54 UTC. Complete that diagnosis before
selecting and validating a performance change and launching further training.
The `follow-ema-gram-experiment` automation update was accepted with this order. Do not restart the canceled job or
expand the sweep while diagnosis is pending. Full, unblocked left/right Gram matrices and per-update
refresh remain part of the requested algorithm.

### 2026-10-31 17:12 UTC — Move diagnosis to v4-32

The user requires v4-32 and excludes v4-8. Canceled the v4-8 diagnostic at
17:07:18 UTC and verified its terminal state at 17:08:31 UTC. Its only completed
measurement was Gram refresh: median 0.086586 seconds over three samples with
finite outputs. The partial result is saved in `optimizer-cost-v4-8-partial.json`; spectral
power and full-update timings were not obtained.

One v4-32 diagnostic is prepared but not yet submitted, under
`optimizer-cost-v4-32-plan.json`. Iris maps
this request to four hosts and 16 TPU chips. The script initializes distributed
JAX, broadcasts a shared run identity, constructs global arrays from addressable
slices, and saves separate rank artifacts. Rank zero uses native online W&B.
Production kernels, full Gram geometry and per-update refresh match pilot
source `44f8a1efdd8117260565f77965fcadd36cef717b`.
The CPU smoke run `ema-gram-20-cpu-smoke-c362374bff13` passed all four kernel
cases, finite-output checks, fused-versus-separate comparison, and online W&B/GCS
writes. This checks the script path; it does not validate multihost TPU execution.

The new performance target is measured training MFU above 5%, followed by
restoration of quarter-power training from `quarter-pilot-128g-resolved.json`.
Checkpoint resume requires a verified restore; it is not yet claimed. That pilot's last MFU was
0.189564% (the logged value is already a percentage). On v4-32, the unchanged
global batch of eight examples (80,353,469,399,040 model FLOPs per update)
would require less than 0.36525 seconds per update to
exceed 5%, using the existing model-FLOP definition and 16 chips at 275 TFLOP/s.
Kernel timings alone cannot establish this target. Queue this diagnostic once
from published source; validate assignment, distributed execution, online W&B,
finite results and device profile coverage before choosing a performance change.

### 2026-10-31 17:12 UTC — Submit v4-32 diagnosis once

Iris accepted `ema-gram-20-optimizer-cost-v4-32-20261031` at 17:12:38 UTC
from published source `09c39d9ba`. The exact-name query was empty before
submission. Authoritative state is pending: four hosts must be coscheduled,
and the matching group currently has no free TPU capacity. The existing batch
request is retained; no v4-8 fallback is allowed. Submission/resource evidence
is in `optimizer-cost-v4-32-submitted.json`. Device execution is not yet verified.

### 2026-10-31 17:25 UTC — Prepare full-eigensolver candidates; reject first parallel preflight

The queued v4-32 job still uses published source `09c39d9ba`; none of the
following candidates changes that job or the training configuration. The
existing default spectral path is retained. Research source now exposes JAX's
full eigensolver selection in `relative_gram_power` and `gram_direction`.
The public `lax.linalg.eigh` API returns vectors before values, unlike the
previous `numpy.linalg.eigh` call; sixteen CPU power comparisons and the eleven
EMA-Gram unit tests passed after this change. Local validation explicitly sets
`PYTHONPATH=lib/levanter/src:lib/haliax/src` because the shared environment's
editable installs otherwise resolve to the contribution checkout.

`optimizer_parallel.py` assigns independent complete matrices to chips, padding
only the layer axis from eleven to sixteen. It retains full left/right Grams,
per-update refresh and the production polar iteration. A 16-device CPU check
covers square, tall, wide and unbatched matrices, alpha 0, 1/4 and 1/2, both ridge
values and three successive updates including zero initialization. Nine of
24 cases fail the unchanged elementwise tolerance (rtol=atol=1e-5), so this
candidate is not approved for training. The largest recorded direction-element
difference is 0.016813, in the rank-deficient half-power case. The check exits
with failure and preserves all cases in `optimizer-parallel-preflight.json`.

An isolating tall-matrix check found bitwise-equal momentum and both EMA
Grams, but different downstream directions. The left Gram power already differed
when computed with the changed matrix assignment; the right power was identical.
This locates a numerical discrepancy after the state update, not a TPU runtime
bottleneck. Do not relax the numerical threshold or infer hardware accuracy
from CPU behavior. Investigate full-solver accuracy on fixed inputs before
adopting any different solver or layout.

Offline TPU StableHLO export for batches of eleven 1024- and 4096-dimensional
full Gram powers succeeds for both the default solver and explicit Jacobi.
The default exports contain 17 and 27 `while` operations; Jacobi has none and
contains a full `Eigh` custom call. These counts describe pre-optimization
structure, not execution time. Hash-verified exports are stored under
`gs://marin-us-central2/ema-gram-20-optimizer-cost/offline-preflight/2026.10.31.17-24`.
The [JAX eigensolver API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.eigh.html)
and the installed JAX 0.11.1 lowering support selecting Jacobi on TPU. TPU
numerical accuracy, runtime, memory and actual training MFU remain unmeasured.
The v4-32 request remains in the batch queue; no additional accelerator job was
submitted. The goal remains measured training MFU above 5%, then restoration
of the quarter-power run specified in `quarter-pilot-128g-resolved.json`.
Checkpoint continuation requires a verified restore; that has not been tested.

### 2026-10-31 17:35 UTC — Isolate eigendecomposition parallelism and pass controlled CPU checks

The first whole-update candidate remains rejected. An isolating check on
identical tall-matrix EMA states found a 4.768e-7 difference in normalized
Grams after changing the whole computation's layout. On identical normalized
inputs, the standalone eigensolver outputs were bitwise equal on CPU. This
motivated a narrower execution change: keep normalization and reconstruction
replicated, and distribute only full eigendecompositions across independent
matrices. `parallel_eigh` adds explicit replicated boundaries and compilation
barriers around that operation, padding only the independent matrix batch.
It does not reduce either Gram or skip a refresh.

Without explicit boundaries, six of 24 small CPU cases failed the same
rtol=atol=1e-5 gate. Adding barriers alone did not fix them. With replicated
boundaries, all 24 small cases passed. The broader check uses embedding-sharded
gradients on sixteen CPU devices, with square/tall/wide matrices up to
128-by-128 and both ridge values. An initial broader comparison allowed the
reference's returned state layout to vary while forcing the candidate's state
to be replicated; five cases differed after the second nonzero update. After
matching both state layouts to the actual pilot and existing cost diagnostic
(replicated), all 24 broader cases passed at the unchanged tolerance. This is
a controlled CPU comparison, not TPU accuracy evidence. All intermediate
failures remain recorded in `optimizer-eigh-parallel-preflight.json`.

Eleven existing EMA-Gram tests also passed against the research checkout.
The updated diagnostic supports explicit eigensolver/layout choices and records
an exact-shape comparison against the original fused update on the same inputs,
including every returned direction and state leaf. Its one-device CPU smoke
run `ema-gram-20-cpu-smoke-6e5078cc2eac` completed all four cases, initialized
online W&B before kernels, passed the reference comparison, and uploaded its
report to GCS. This smoke validates the diagnostic path, not multihost execution
or device speed.

`optimizer-solver-followup-plan.json` prepares a single matrix-sharded QDWH
follow-up after the current baseline diagnostic supplies its cost and memory
evidence. Explicit full Jacobi is conditional on those results and still needs
TPU accuracy checks. Neither follow-up is submitted. The current v4-32 job
continues to use `09c39d9ba`; preserve its queue position. The contribution PR
is unchanged. Actual training MFU above 5% and verified checkpoint continuation
remain required before claiming the active goal complete.

### 2026-10-31 17:38 UTC — Validate a v4-32 mesh without changing global batch

The original training mesh would resolve to data=16, model=1 on v4-32 and
fails trainer validation: global batch 8 is not divisible by per-device
parallelism 2 times data-axis size 16. A data=4, model=4 mesh passes validation
with the original global batch 8 and microbatch 8. The model axis uses the
existing head/MLP mappings. This is a non-training configuration check; no
model or optimizer steps ran. `v4-32-training-mesh-preflight.json` preserves
the rejected configuration error and proposed resolved mappings.

The eigensolver-only candidate now supports assigning full matrices across
both data and model axes. A third 24-case CPU check with gradients sharded
on both matrix dimensions passes at rtol=atol=1e-5, with replicated state on
both reference and candidate. `optimizer-eigh-data-model-preflight.json`
records those checks. This prepares the original-batch hardware migration;
TPU correctness, measured MFU and checkpoint restoration remain unverified.
No further accelerator job or training run was submitted.

### 2026-10-31 17:59 UTC — Prepare v4-32 checkpoint continuation

The research branch now exposes the eigensolver-only matrix layout through
`EmaGramConfig`; the diagnostic and trainer use the same implementation.
The training CLI accepts v4-32 only, prepares data=4/model=4, and preserves
the 447M pilot's global batch 8 and 7,630 total updates. The prepared config schedules a
one-step profile at step 940, after the intended step-923 restore. These changes are unselected
research candidates; the queued diagnostic still uses `09c39d9ba`.

Two small-Qwen checks on sixteen virtual CPU devices restored all trainer
state from the original optimizer into the matrix-layout configuration.
Restored arrays were bitwise identical to the saved reference state. Two
subsequent updates matched the uninterrupted original optimizer in all state
and model leaves at rtol=atol=1e-5 for ridge 1e-4 and 1e-2. The existing
26 EMA/experiment checks and the new v4-32 recipe check passed in their CPU
scopes. The preflight record retains the initial version-validation failure
and the existing single-device test's incompatible-layout failure when run
with sixteen devices. No tolerances were relaxed.

The canceled pilot's temporary step-923 metadata exists under the regional
`tmp/ttl=14d/checkpoints-temp/` prefix, timestamped 16:55:33 UTC. It was not
in the permanent checkpoint directory. `training-integration-preflight.json`
records the exact URI and limitations; `quarter-v4-32-continuation-prepared.json`
is a dry-run config with full restoration required. The 447M checkpoint has
not been loaded into a new process. A successful small-model restore does
not establish that the original training has resumed.

The v4-32 diagnostic remained pending at 17:54 UTC with no worker assignment.
No additional accelerator job or training run was submitted. TPU correctness,
measured cost and memory, and actual training MFU above 5% remain open. Before
reporting a future training attempt, `experiments.ema_gram.report` must be
extended to validate the logged mesh, solver and restore settings against
that attempt's saved execution configuration. Current historical reports
retain their original source and resolved configurations.

### 2026-10-31 18:03 UTC — Validate reporting execution controls

The reporting manifest now requires an `execution` object copied from the
attempt's saved resolved configuration. The exporter builds the expected
recipe with those choices before comparing W&B controls. The v4-32 regression
accepts matching configurations and rejects differences between expected
execution settings and the logged mesh, eigensolver layout, solver
implementation, or restoration path. No comparison field was removed. Historical manifests remain paired with their original
pinned exporter and source. This closes the reporting preparation item above;
actual TPU diagnosis and training restoration remain pending.

### 2026-10-31 18:11 UTC — Prepare diagnosis on saved Gram state

The candidate diagnostic can now read the canceled pilot's saved FP32 Grams
and momentum through the native checkpoint loader. The step-923 manifest
lists 21 such arrays totaling 3,460,300,800 bytes. Only metadata and the
13,640-byte manifest were read locally; the original arrays will be read on
TPU in the same region after the queued baseline supplies its measurements.
The loader requires every selected array and its expected geometry.

A local generated checkpoint restored all 21 arrays bitwise on sixteen
virtual CPU devices. Fourteen direction comparisons (seven matrix groups,
two ridge values) passed at unchanged rtol=atol=1e-5. The final online CPU
smoke run `ema-gram-20-cpu-smoke-d177fadf2f26` completed five kernel cases,
passed both full-update and unchanged-saved-state comparisons, and uploaded
its report. That report was read back from GCS. A preceding smoke run
`562d68c5e1bc` verified the loader before paired reference timing was added.
`checkpoint-spectrum-preflight.json` preserves identities, hashes and scope.

The candidate job will measure the original and proposed implementations on
identical inputs. With `--checkpoint`, update gradients remain synthetic;
a separate direction case uses the saved Grams/momentum without refreshing
them. This avoids treating the earlier synthetic baseline and a later real
checkpoint as a controlled timing pair. It does not measure full-LM runtime
fractions or training MFU. Loading selected optimizer arrays is not full
trainer restoration, and no model or data progress is advanced.

The existing v4-32 diagnostic was still pending at 18:06:47 UTC; no additional
accelerator job was submitted. `optimizer-solver-followup-plan.json` retains
the sequential gate: obtain that job's cost/memory evidence, then test one
candidate on v4-32. Original-checkpoint TPU accuracy, execution cost, memory use and
actual training MFU remain unverified.

### 2026-11-01 02:39 UTC — Submit one v4-8 cost probe while v4-32 stays queued

The user allowed a v4-8 probe after the v4-32 diagnostic remained pending for
more than nine hours. This is a single-host device-side cost measurement. It
is not training, not MFU evidence and not v4-32 evidence. Training and the
performance gate remain v4-32 only.

The probe runs from published source `5e04d1467` on
`codex/20-ema-gram-polar-v4-8-probe`. That commit is the v4-32 diagnostic
source `09c39d9ba` with one change: the topology guard maps `v4-8` to four
devices and one process. The current research branch cannot be used for this
measurement. Its default `relative_gram_power` calls `lax.linalg.eigh`
directly, while `09c39d9ba` calls `jnp.linalg.eigh`, which symmetrizes its
input. The eigensolver difference is recorded here and has not been measured.

`optimizer-cost-v4-8-probe-plan.json` was committed as `174b83274` before
submission. It requests one v4-8 task, 120 CPUs, 128 GiB host memory and 50
GiB disk in us-central2 at batch priority, with no wall-clock limit and the
Iris default preemption policy. It writes to
`gs://marin-us-central2/ema-gram-20-optimizer-cost/2026.10.31.3`.

The first CLI attempt failed validation before queueing because `--tpu`
requires `--enable-extra-resources` on the entrypoint. No job was created.
The second attempt, after confirming no probe job with this name existed,
was accepted as `/kaiyuew@stanford.edu/ema-gram-20-optimizer-cost-v4-8-probe-20261031`
at 02:39:23 UTC. Its state is pending with 0 of 1 task scheduled and no
preemptions. The v4-32 job
`/kaiyuew@stanford.edu/ema-gram-20-optimizer-cost-v4-32-20261031` remains
pending since 17:12:38 UTC and was not resubmitted or changed.

Autoscaler state at 23:55 UTC on 2026-10-31 showed the us-central2 v4-8
reserved group with 188 ready slices, all in use, and 50 booting, in backoff
with degraded health. The v4-8 preemptible group had no slices and was in
backoff. The v4-32 reserved group had one ready slice in use and one booting
slice created at 22:01 UTC. These observations are not guarantees about
scheduling.

Expected outputs are the same four original-kernel cases, three samples each,
one profile invocation and an online W&B run named `ema-gram-20-tpu-cost-<attempt>`
once the job starts. Device execution is not verified until the job reports.

### 2026-11-01 07:14 UTC — Both optimizer diagnostics terminated before starting

Iris reports both `ema-gram-20-optimizer-cost-v4-8-probe-20261031` and
`ema-gram-20-optimizer-cost-v4-32-20261031` as `killed` with reason
"Terminated by user" under the submitting account. The change happened
between state polls at about 07:02 and 07:13 UTC. Neither job was ever
assigned workers, so no task ran and no W&B run, timing, profile or memory
measurement exists. The agent that submitted them did not cancel them, and
the origin of the cancellation is unconfirmed. Neither job has been
resubmitted; resubmission awaits the user's decision. The probe branch, its plan and both source revisions remain
published.

### 2026-11-01 07:45 UTC — Correct the eigensolver note; attribute the alpha slowdown

Correction to the 02:39 UTC entry: the current branch's default
`lax.linalg.eigh` call is not materially different from the pilot's
`jnp.linalg.eigh` call. In the installed JAX 0.11.1, `lax.linalg.eigh`
defaults to `symmetrize_input=True`, `lower=True` and
`sort_eigenvalues=True`, and `jnp.linalg.eigh` calls it with those values.
Both revisions also symmetrize the Gram explicitly before normalizing. The
default spectral path on the current branch is therefore the same
computation as the pilot's. The separate probe source equals the v4-32
diagnostic source plus the topology guard, so it remains valid, but it was
not required.

The alpha=1/4 slowdown can be attributed from existing runs without new TPU
time. The alpha=0 baseline (source `0017f4d20`) and the alpha=1/4 pilot
(source `44f8a1efd`) have identical optimizer library code; `train.py`
differs only in the host RAM request. Their resolved configurations differ
only in alpha, learning rate, host RAM and run identity. Both used v4-8,
global batch 8, sequence length 4096, the same mesh and 7,630 steps. At
`44f8a1efd`, `gram_direction` computes spectral powers and the two
left/right products only when alpha is nonzero; Gram refresh and the
Newton-Schulz polar factor run for both.

The baseline's median update took 0.3474 seconds over 7,629 steps. The
pilot's took 38.738 seconds over 51 updates ending at step 905. The
alpha-only branch therefore costs about 38.39 seconds per update, 99.1% of
the pilot step. The baseline device profile places Gram refresh at about
0.054 seconds and the polar direction at about 0.044 seconds per step.

The saved pilot StableHLO (sha256 `9f754267…`, matching the coverage record)
contains 11 batched 11x1024x1024 and 3 batched 11x4096x4096 eigensolver calls
per update: 121 and 33 matrices. Each batched call is an 11-iteration loop
around the TPU `Eigh` custom call. There are no host callbacks or host
memory placements. The pilot trace has no device plane because the TPU
driver dropped it: XSpace was 2.42 GiB against a 2.00 GiB limit. The other
alpha-only work, power reconstruction and two HIGHEST-precision products per
matrix, totals about 6.4e12 FLOP per update. With six bf16 passes this is
about 0.14 seconds at v4 peak, so the eigensolver accounts for nearly all of
the 38.4 seconds. This is an inference from full-step durations, HLO and
FLOP counts, not a device measurement.

This bounds the v4-32 gate. Replicated optimizer state makes every chip
compute all 154 eigendecompositions, and v4-32 uses the same v4 chips, so
the larger slice alone does not reduce eigensolver time. Distributing whole
matrices over 16 chips can at best divide it by 16: at least 2.40 seconds
per update, at most 0.76% MFU under the current FLOP definition before any
other work. Under an n-cubed cost model the 33 large matrices carry about
95% of the work, about 1.1 seconds each. At least one chip then holds three
of them, giving about 3.3 seconds and 0.55% MFU. If the rest of training
took 0.1 seconds on v4-32, the gate would leave about 0.27 seconds for the
eigensolver, about 9x below the ideal-distribution bound. Reaching more than
5% MFU therefore needs a per-matrix inverse-fourth-root computation roughly
an order of magnitude faster than the current TPU eigensolver, in addition
to ideal distribution. That changes the spectral solver, which the protocol
has kept as an exact eigendecomposition, and needs a user decision.

Prediction: the diagnostic's `spectral_powers` case should take about
35–39 seconds per invocation on either v4-8 or v4-32, because state is
replicated in both. A much smaller value would reopen this attribution, for
example if synthetic spectra converge faster than trained ones. The numbers
and limitations are in `alpha-cost-attribution.json`.

## 2026-11-02 20:27 UTC — Resume v4-32 cost diagnosis

Iris accepted a fresh non-training diagnostic submission at 20:23:13 UTC after the
user requested resuming research. Its four tasks were pending at 20:26:30
UTC; a canonical Iris state query still reported pending at 20:27:09 UTC.
At 20:29:47 UTC, Iris reported no matching workers and autoscaler demand
blocked by quota-pool tier monotonicity. No task had started.
The 20:30:13 UTC autoscaler snapshot records two failed preemptible v4-32
provisioning attempts: the cloud reported no capacity in us-central2-b.
These are provisioning failures; no diagnostic process has run.
The previous v4-32 diagnostic and v4-8 probe remain terminal, as recorded
on November 1.

The new job is `ema-gram-20-optimizer-cost-v4-32-20261102`. It runs frozen
source `09c39d9ba2f06fe196cbb8136467c9a152c9dc68` on four coscheduled
hosts, with 16 TPU v4 chips total and 120 CPUs, 128 GiB RAM and 50 GiB disk
per host. It retains batch priority and default preemption retries, with
no wall-clock limit. The [recovery plan](optimizer-cost-v4-32-recovery-plan.json)
records the exact command and output prefix. This process loads no checkpoint
and performs no training.

Before submission, the v4-32 autoscaler groups reported available with no
backoff or failures, but no ready, booting or requested v4-32 slice. Capacity assignment remains unverified.
The existing scheduled monitor now follows this job. One PR monitor follows
[the optimizer implementation PR](https://github.com/marin-community/marin/pull/10790).

A successful diagnostic should measure optimizer cost and memory, and collect
a device trace after compilation and warm-up. The November 1 estimates of
eigensolver cost and scaling depend on assumptions; they do not establish
a measured performance bound.
Candidate changes must preserve the optimizer definition and pass numerical,
memory and performance checks. Full checkpoint restoration and measured
training model FLOPs utilization (MFU) above 5% require subsequent validation
in the full trainer; both remain unverified.

## 2026-11-02 20:40 UTC — Available-hardware diagnostic

The user broadened hardware authorization to v4-8 or other available compute.
The 20:38 UTC Iris inventory showed all 192 healthy v4-8 workers busy and two
healthy single-host v5e-8 workers in us-west4 with no running tasks. Iris
accepted one non-training diagnostic, `ema-gram-20-optimizer-cost-v5e-8-20261102`,
at 20:40:06 UTC.

It uses published source `7ec075ff4ed128e14136e5dab00cfd52478847cf`, derived
from the original diagnostic with only topology and device-kind validation
changed. All kernel math and original matrix shapes are unchanged. Local
syntax, lint and formatting checks passed. The request has eight v5e chips,
96 CPUs, 128 GiB RAM and 50 GiB disk, batch priority, default preemption
retries and no wall-clock limit.

Synthetic inputs are generated on the execution host; reports and one device
trace use the separate GCS prefix ending `2026.11.02.2`. No checkpoint or
training dataset is read. The [plan](optimizer-cost-v5e-plan.json) records the
exact command. Assignment and measurements remain unverified. The queued
v4-32 request is preserved, and the scheduled monitor follows both jobs.

This probe can establish v5e kernel cost and numerical behavior; it cannot
establish v4 performance or actual training MFU. Measure the unchanged baseline
kernels on v5e first, then validate one candidate optimization that preserves
the optimizer definition at a time. Full checkpoint restoration and actual
training MFU remain later gates.

## 2026-11-02 20:42 UTC — v5e execution verified

Iris reports the v5e diagnostic running on attempt 0. Its GCS report identifies
eight native `TPU v5 lite` devices and one JAX process. The online
[W&B run](https://wandb.ai/marin-community/marin/runs/ema-gram-20-tpu-cost-fd4fde87827d)
reports successful logging initialization, eight devices and one host.

Gram refresh completed three finite samples, with median 0.0816146 seconds.
The [incremental evidence](optimizer-cost-v5e-startup.json) preserves the exact
samples and configuration. Spectral-power timing, the other kernel cases,
device trace and memory measurements remain pending. This is an isolated
synthetic kernel measurement; full training has not resumed.

## 2026-11-02 21:03 UTC — v5e baseline completed

Iris reports success with exit 0 on attempt 0. W&B finished with
`diagnostic/completed=1`. Three synchronized samples after warm-up give
medians of 20.2645 seconds for spectral powers, 20.8427 seconds for the full
quarter-power optimizer update, 0.08161 seconds for Gram refresh, and
0.03879 seconds for the alpha-zero direction. All measured kernel outputs
were finite. Spectral powers dominate these measured costs; separate kernel
timings do not form an additive profile decomposition or measure training MFU.

The [completion evidence](optimizer-cost-v5e-completed.json) includes the final
report. The trace contains more than 1.3 million events on each of eight TPU
planes. Every device reports the same JAX allocator peak, 11,722,465,280 bytes;
this does not measure all HBM use. Finelog observed a cumulative task host-memory
peak of 104.8994 GiB across 198 samples. The uploaded trace is 1.76 GB, larger
than expected. Future v5e artifacts will use us-west4 storage.

The [next candidate plan](optimizer-matrix-v5e-plan.json) assigns independent
full-matrix eigendecompositions across eight chips, retaining replicated
normalization and reconstruction. It compares every direction and updated
Gram/momentum leaf against the original implementation on identical synthetic
inputs, at rtol=atol=1e-5. An eight-device CPU smoke passed all four cases and
the paired comparison with online W&B and regional GCS readback. Dead refreshed
state is released before holding both outputs for comparison; the kernel math
is unchanged.

This synthetic screen precedes the required saved-checkpoint validation.
The 3.46 GB of saved Gram/momentum arrays remain in us-central2 for regional
validation. Full trainer restoration and actual training MFU remain unverified.

## 2026-11-02 21:05 UTC — Matrix candidate submitted

Iris accepted `ema-gram-20-matrix-cost-v5e-8-20261102` at 21:05:08 UTC from
published source `528a25982f524e91063d631f4e9f87220f6d5a16`. The
[plan](optimizer-matrix-v5e-plan.json) records the exact command and regional
output prefix. This is the only active candidate test; the completed baseline
is terminal and the v4-32 reference request remains queued. Startup and TPU
numerical/performance results are not yet verified.

## 2026-11-02 21:10 UTC — QDWH manual-axis compatibility failure

The matrix/QDWH candidate failed after Gram refresh, before spectral timing.
Iris reports failed attempt 0 with exit 1. W&B's finished state only means
tracking closed; no diagnostic-completed metric exists. JAX 0.11.1 rejected
an internal QDWH loop carry: an invariant boolean convergence flag became
varying across the manual data axis.

The [local reproducer](qdwh_vma_repro.py), using JAX's QDWH implementation on
eight CPU devices, produces the same error. Disabling varying-manual-axis
(VMA) checking for this eigensolver shard_map bypasses that static consistency check. The reproducer reports zero maximum
absolute difference for the QDWH polar-factor output against its unsharded
reference. Every manual axis is included in the eigenvector and eigenvalue
output specs, which concatenate all shards; no replicated-output promise is made. The eigensolver math and
paired numerical tolerance remain unchanged. See
[JAX's explanation of manual-axis types](https://docs.jax.dev/en/latest/notebooks/shard_map.html#scan-vma).

Eleven EMA-Gram tests passed; two matrix-assignment tests required more devices
than the ordinary pytest process exposed and were skipped. A separate online eight-device CPU
smoke passed four cases and all 28 output-leaf comparisons at rtol=atol=1e-5.
The [failure evidence](optimizer-matrix-v5e-failed.json) preserves this attempt.
A [fresh candidate](optimizer-matrix-v5e-vma-plan.json) is prepared for TPU
validation; CPU results do not establish TPU correctness or speed.

## 2026-11-02 21:11 UTC — Compatibility candidate submitted

Iris accepted `ema-gram-20-matrix-cost-v5e-8-vma-20261102` at 21:11:32 UTC
from published source `52bd24cf389658686a8b84cbf232ede36c678cb1`. Its
[plan](optimizer-matrix-v5e-vma-plan.json) retains the same numerical gates,
resources and default retry policy. This is a fresh process, with no
checkpoint resume. The failed predecessor remains terminal.

## 2026-11-02 21:33 UTC — Matrix spectral timing and full-update HBM failure

The matrix-distributed QDWH candidate produced finite inverse-quarter-power
Gram outputs with a three-sample median of 3.86221 seconds. The original
implementation measured a three-sample median of 20.26449 seconds in the
preceding eight-chip v5e job, using the same synthetic inputs and timing
protocol. This is a 5.25-fold difference between isolated spectral timings;
the candidate-versus-original numerical comparison has not run.

The candidate then failed during device executable loading for the full
optimizer update. TPU runtime
reported a 6.06G allocation request with 5.91G available. Iris confirms failed
attempt 0 with task exit 1. Full-update timing, numerical comparison and the
profiler trace after warm-up are missing. The [failure record](optimizer-matrix-v5e-vma-failed.json)
preserves the partial measurements and exact source.

Alongside the input state, the diagnostic retained a 3.46 GB copy of the
Gram/momentum state after refresh, used by the earlier timing cases. The full-update case does not read that bank. The
[revised diagnostic](optimizer-matrix-v5e-memory-plan.json) releases it and unused
spectral executables before full-update warm-up. It also moves reference
outputs to host memory outside timed intervals, so the paired comparison does
not retain two full device output banks. It records device allocator snapshots
before and after each case. Optimizer math, synthetic inputs and the 1e-5 absolute
and relative comparison tolerances match the failed diagnostic. An online
eight-device CPU smoke passed all four cases and 28 comparison leaves; its
GCS report was read back. Whether the full update
fits after this change remains a TPU validation question; diagnostic memory
use does not establish training memory requirements.

## 2026-11-02 21:37 UTC — Diagnostic memory-lifetime correction submitted

Iris accepted `ema-gram-20-matrix-cost-v5e-8-memory-20261102` at 21:37:13 UTC
from published source `cfbeddaf87bdbacf30f3a084232197a95019d4af`.
The [plan](optimizer-matrix-v5e-memory-plan.json) records its command and
regional output prefix. It retains eight v5e chips, batch priority, default
preemption retries and no wall-clock limit. This is the only active candidate;
the failed predecessor remains terminal and the v4-32 reference remains queued.
TPU correctness and full-update timing are still pending.

## 2026-11-02 21:48 UTC — Matrix/QDWH rejected by numerical comparison

The memory correction allowed the full optimizer update to run. In the same
v5e job, the candidate full-update median was 4.04522 seconds and the original full-update median was
20.84297 seconds. All timed outputs were finite, but all seven update-direction
tensors failed the fixed rtol=atol=1e-5 comparison. Their relative Frobenius
errors ranged from 0.00698 to 0.02902; the largest absolute error was 0.00944.
All 21 updated Gram/momentum tensors passed.

Iris confirms failed attempt 0 with task exit 1. The assertion stopped the
job before the warmed device-profile capture. W&B recorded a failed comparison;
its finished state does not mean diagnostic success. The
[failure evidence](optimizer-matrix-v5e-numerical-failure.json) contains the
same-input timings, all 28 comparisons and per-case device allocator readings.
Finelog observed a host-memory peak of 57.7998 GiB across 95 samples.

Reject this matrix/QDWH candidate for training at the unchanged tolerance.
The direction discrepancy is measured, but its contributions from the
solver and compiler/reduction ordering have not been isolated. The next
[planned candidate](optimizer-jacobi-v5e-plan.json) uses Jacobi to compute all eigenvalues and eigenvectors of each full matrix with the same matrix distribution and numerical gate. JAX supports
that solver on TPU/GPU but not CPU; an online CPU logging check passed and its GCS configuration was read back.
That check executed no kernels.
Validation on Gram/momentum arrays from the original training checkpoint,
full checkpoint restoration and actual training MFU have not been established.

## 2026-11-02 21:51 UTC — Jacobi candidate submitted

Iris accepted `ema-gram-20-jacobi-cost-v5e-8-20261102` at 21:51:56 UTC from
published source `a1938e336fa17a47661c9e89e18ee4b1c3fa88ef`. The
[plan](optimizer-jacobi-v5e-plan.json) records its exact command. It uses
eight v5e chips and the existing synthetic diagnostic, changing the candidate
solver from QDWH to Jacobi. The original implementation remains the same-job
reference. Fixed comparison tolerances, full matrix geometry, regional storage,
batch priority and default preemption retries are unchanged; no wall-clock
limit is set. This is the only active candidate, with TPU results pending.

## 2026-11-02 22:02 UTC — Correction to the validation status

The last sentence of the 21:48 entry accidentally omitted the negation in
its validation status. It is corrected above: original-checkpoint TPU
validation, full trainer restoration and actual training MFU remain unverified.
No training process was launched or resumed by these diagnostics.

## 2026-11-02 22:18 UTC — Jacobi is executing timed spectral samples

The Jacobi job remains running. Its incremental report still contains only
the Gram-refresh measurement. A read-only stack inspection found the main
thread at `measure` line 146, waiting for a timed spectral sample. That line
runs after the warm-up call and its finite-output assertion. Warm-up therefore
completed with finite outputs; the sample index and elapsed sample time were
not inspected. This observation does not establish a hang or a steady latency.
One timed sample may include stack-inspection overhead.

The latest task telemetry reports 8,375 MiB of task host memory and an 8,526 MiB
cumulative peak for attempt 0. Accelerator telemetry is absent. The
[progress evidence](optimizer-jacobi-v5e-progress.json) records the stack,
source lines and bounded telemetry query window. Keep the current job and
observer running; spectral timings, numerical agreement and final artifacts
remain pending. No concurrent candidate or training job was submitted during this check.

## 2026-11-02 22:32 UTC — Jacobi spectral timing is slower

On eight v5e chips, Jacobi completed three timed spectral-power samples,
each with finite outputs: 445.74329, 445.74568
and 445.74379 seconds, with median 445.74379 seconds. Warm-up, including
compilation and execution, took 459.59320 seconds. The original implementation
measured a spectral median of 20.26449 seconds in the earlier eight-chip v5e
job with matching synthetic inputs. Jacobi is about 22.0 times slower for this
isolated case. One sample may include the earlier stack-inspection overhead.

The [incremental report](optimizer-jacobi-v5e-spectral.json) preserves all
samples and device allocator snapshots. The job is now in the full-update
stage; same-job reference timing, numerical comparison and the warmed device
profile remain pending. Keep the current diagnostic running to completion.
No additional candidate or training job was submitted during this check.

## 2026-11-02 23:20 UTC — Reject Jacobi after the numerical comparison

Jacobi failed the fixed rtol=atol=1e-5 numerical gate at 23:07:05 UTC; Iris confirms
attempt 0 failed with task exit 1. All seven direction tensors fail, with
relative Frobenius errors of 0.00703–0.03754 and maximum absolute error
0.01024. All 21 updated Gram/momentum tensors pass. The same-job full-update
medians are 503.18352 seconds for Jacobi and 20.84245 seconds for the original
implementation: Jacobi is 24.14 times slower. All timed outputs are finite.

The [failure evidence](optimizer-jacobi-v5e-numerical-failure.json) preserves
all samples, leaf comparisons and device allocator snapshots. Finelog reports
a cumulative task host-memory peak of 20.17969 GiB across 680 samples from
21:51 to 23:10 UTC. W&B closed with the comparison marked false and no
`diagnostic/completed` field.
The assertion stopped execution before the warmed profile. Reject Jacobi for
training and retain this failed attempt without restarting it.

Both matrix-layout candidates now fail the direction comparison. Their error
patterns do not identify a cause. The [attribution plan](optimizer-divergence-attribution-plan.json)
compares frozen inputs through spectral powers, inner products, polar iteration
and reconstruction. It feeds identical powers and inner matrices through both
downstream layouts to test whether the discrepancy comes from the solver or
later operations. It is a design only; no new diagnostic
or training job has been submitted. The v4-32 reference remains queued.

## 2026-11-02 23:32 UTC — Implement frozen-input attribution

The [stage diagnostic](optimizer_attribution.py) first records the original
and matrix-layout fused full updates. It then compares the default eigensolver in replicated and matrix layouts
using identical refreshed Grams and momentum, records eigenpair residuals and reconstructed
powers, and compares downstream operations with identical powers and inner
matrices under replicated and direction-sharded output layouts. These explicit
layout controls do not describe the hidden layouts of the fused executables.

The online logging check succeeded on eight virtual CPU devices, and a
preliminary CPU smoke run is still collecting comparisons. Both used the
version before the following setup correction:
zero-state initialization and refresh must remain separate calls, as in the
original diagnostic. The implementation now preserves that boundary; validation
of this final version remains pending. No accelerator diagnostic was submitted.

Comparison failures are recorded while later diagnostic stages continue; this
never accepts a candidate for training. The completion marker means every stage
was collected, not that a cause was established. Durations
include compilation, and stage boundaries alter compilation; this diagnostic
makes no warmed-performance or training-MFU claim.

## 2026-11-02 23:49 UTC — Bound stalled report-upload requests

The preliminary CPU run completed 172 stages and passed all 101 comparisons.
The corrected run reached five finite stages and three passing comparisons,
then stalled while uploading its report. Its interrupt traceback confirms a
wait in GCS upload initiation; the local report contains the third comparison,
while the remote copy contains two. The [interrupted CPU attempt](optimizer-attribution-cpu-io-interruption.json)
retains the local report. It exited 130 after SIGINT; no accelerator was allocated.

Report and configuration writes now give each GCS request a 60-second timeout,
with the storage client's request retries retained. This bounds communication
requests, not the diagnostic's total runtime or the scheduler's retry policy.
The corrected harness still requires a complete CPU smoke run before submission.

## 2026-11-02 23:56 UTC — Attribution harness passes CPU validation

The final online CPU smoke run completed 173 stages and passed all 101
comparisons at rtol=atol=1e-5 on eight virtual devices. The process exited zero,
and the regional GCS report matches the local copy byte for byte. The
[preflight record](optimizer-attribution-preflight.json) pins source, script
hash and report checksum, and retains the earlier interrupted attempt.

This validates the harness only. The full-size v5e diagnostic will test the
same default eigensolver under both layouts and collect the stage comparisons;
TPU numerical behavior and the cause of the earlier discrepancy remain unknown.
No training or original-checkpoint read is part of this diagnostic.

## 2026-11-03 00:00 UTC — Attribution diagnostic running on v5e

Iris accepted `ema-gram-20-attribution-v5e-8-20261102` at 23:57:15 UTC on
November 2 from published source `39994b06be819fd277ffe36903f66de3eb55c4d4`.
The [plan](optimizer-divergence-attribution-plan.json) records the exact command
and resources. Startup verifies eight native TPU v5 lite devices, one JAX
process, the expected script hash and online W&B with its logging check set to
one. The [startup record](optimizer-attribution-started.json) links the run.

The job has completed the fused checks and entered the per-stage comparisons.
The final report and interpretation remain pending. This is one synthetic
non-training diagnostic; no original checkpoint was read. The v4-32 reference
request remains queued with its original priority.

## 2026-11-03 00:09 UTC — Polar output depends on layout in frozen-input controls

The [partial attribution report](optimizer-attribution-v5e-partial.json)
contains completed query, key, value and attention-output projection controls
on eight native v5e devices. For these four groups, the staged original and
matrix-layout eigensolvers produce exactly matching eigenvalues and reconstructed
powers. Both power branches produce identical outputs under the same replicated
downstream layout.

Changing downstream outputs from replicated to direction-sharded layout with
identical powers leaves the inner matrix unchanged but changes the polar output and final direction beyond
rtol=atol=1e-5 in all four groups. Feeding an identical inner matrix directly
into the polar function reproduces the layout discrepancy for query, key and
value, with relative Frobenius errors 0.00618–0.00640. For the attention-output
projection, this isolated polar output matches exactly.

These partial results establish layout sensitivity in the instrumented polar
computation for three groups. They do not establish the compiler mechanism or
fully explain the fused-update discrepancy. Feed-forward groups remain in
progress. No candidate is accepted for training.

## 2026-11-03 00:25 UTC — Attribution completes across all seven groups

The [completed diagnostic](optimizer-attribution-v5e-completed.json) exited zero
on attempt zero after 173 finite stages and 101 comparisons. W&B records
attribution completion and no candidate acceptance. Host peak was 70.09 GiB
across 211 Finelog samples; the largest recorded JAX allocator peak was
8,076,038,656 bytes per device.

For the left and right Grams of all seven groups, the original and matrix-layout
eigensolvers give exactly matching eigenvalues and reconstructed powers. Staged
and fused power outputs also match exactly in both implementations. Both power branches give
identical downstream results under a common replicated layout. Changing the
downstream output layout fails the polar and final-direction comparisons in
all seven groups. Inner matrices match exactly except for down-projection,
whose maximum absolute difference is 9.55e-12 and passes rtol=atol=1e-5.

In the isolated polar layout control, polar outputs differ for query, key and
value; they match exactly for attention-output and all three feed-forward groups. Thus the
completed controls expose downstream layout sensitivity without establishing
the compiler mechanism or fully explaining the fused-update discrepancy.

The next diagnostic will capture compiled operations and expose normalization
and each of the five polar iterates on identical inner inputs under replicated
and direction-sharded output layouts, preserving the original arithmetic.
Each traced final output will be checked against its uninstrumented counterpart
because additional outputs can change compilation. CPU validation and publication
must precede submission. No training optimization is accepted.

## 2026-11-03 00:39 UTC — Compiler and iterate harness passes CPU validation

The [compiler/iteration harness](optimizer-polar-compiler-preflight.json) passed
187 stages with finite outputs and all 122 comparisons at rtol=atol=1e-5 on
eight virtual CPU devices.
Its online W&B run finished and the process exited zero. The regional GCS report
matches the local bytes; all 116 compressed compiler artifacts passed checksum
and decompression checks.

This validates the harness only. The next synthetic TPU diagnostic uses the same
11-layer, 1024/4096-dimensional matrix geometry as the original optimizer-cost
diagnostic. It preserves the default eigensolver and existing five-step quintic
Newton–Schulz arithmetic, captures compiler IR, and compares traced final polar
outputs against uninstrumented calls. No checkpoint or training
is part of it. The initial local logging invocation failed before initialization
because an editable install resolved an older checkout; explicit import paths
corrected it, and the failed invocation is retained in the preflight record.

## 2026-11-03 00:42 UTC — Compiler attribution running on v5e

Iris accepted the compiler/iteration diagnostic at 00:39:29 UTC from published
source `0729daeaf8e1732e87725426c4eebf52dc7ff807`. The
[startup record](optimizer-polar-compiler-started.json) verifies eight native
TPU v5 lite devices, one JAX process, the expected script hash and an online
W&B run with `logging_check/success=1`. The job has entered the fused original update.
The [plan](optimizer-divergence-attribution-plan.json) records the exact command
and resources.

One log observer follows this process; the earlier attribution observer ended
naturally. The v4-32 reference request remains queued. No training or original checkpoint
read is part of this diagnostic.

## 2026-11-03 00:51 UTC — Polar iteration differences grow after normalization

The [partial compiler/iterate record](optimizer-polar-compiler-v5e-partial.json)
contains completed query, key, value and attention-output projection traces.
Input norms and normalized inputs match exactly across replicated and
direction-sharded outputs. For query, key and value, the first nonzero difference
appears in the output of iteration one. The unchanged rtol=atol=1e-5 gate first fails at
iteration three for query/value and iteration two for key. Each traced final
polar output exactly matches its uninstrumented counterpart for the same layout.
The attention-output control matches through all five iterations.

The isolated query-projection optimized HLO contains five explicit BF16
conversions into BF16 and zero all-reduces for replicated output, versus one conversion
and five all-reduces for sharded output. These are static compiler-text counts,
not runtime event counts or a causal attribution. The fused original and
matrix-layout updates also differ slightly in refreshed state, within tolerance,
so the full mismatch remains
unresolved. Feed-forward traces are still running. A read-only streaming
inspection of the larger fused compiler files ran inside the task; those
artifacts remain in their region, and no warmed-performance claim is made.

## 2026-11-03 00:54 UTC — Compiler attribution completes

The [compiler diagnostic](optimizer-polar-compiler-v5e-completed.json) succeeded
on attempt zero with task exit zero, 187 finite stages and 122 comparisons
collected. W&B records completion and no candidate acceptance. All fourteen
traced final polar outputs match their corresponding uninstrumented outputs
exactly. Query, key and value retain the growing differences between replicated
and direction-sharded outputs described above; attention-output, gate, up and down projections match through every
isolated polar iterate. The fused matrix-eigensolver candidate still fails
rtol=atol=1e-5 against the original fused update.

All 116 compiler objects are present with matching recorded sizes, totaling
20,058,285 compressed bytes. The final report and two small isolated compiler
objects were read back with verified SHA-256 values. The larger fused files
were inspected inside their original region; a full remote SHA-256 audit of
all compiler objects has not been performed. Host peak was 26.5 GiB across
117 Finelog samples, including the read-only compiler inspection. JAX allocator
peak was 8,076,038,656 bytes per device.

The [next controlled diagnostic](optimizer-divergence-attribution-plan.json)
will pass three refreshed-state banks through fixed direction-only executables:
a standalone refresh on the same initial state and gradients, and the states
returned by each fused implementation. Reusing each executable holds compilation
fixed while comparing input banks; it does not isolate every interaction between
inputs and compilation. Direction-only results must also be compared with the fused
references because removing refresh changes compilation. This is a design;
it has not been implemented or submitted. No optimization is accepted.

## 2026-11-03 01:10 UTC — Hold direction executables fixed across input states

The [EMA-Gram state comparison](optimizer_attribution.py) now accepts
`--state-crossfeed` to test sensitivity to refreshed input values while holding
each direction executable fixed. All three state banks use the same pre-update
state and gradients and represent the same post-refresh boundary. They come
from standalone refresh, fused original update and fused matrix-eigensolver
update. Their Gram/momentum fields and shapes match; their values can differ.
Host-array fingerprints check that evaluation leaves each bank unchanged.

Two evaluators compute directions directly from refreshed Grams and momentum:
the original eigensolver and matrix-distributed eigensolver. Both use identical
external input/output shardings; internal compiler layouts may differ. Each is
compiled once, processes all three banks plus a replay of the first, then releases
its executable before the other evaluator starts. Outputs are copied to host.
The [plan](optimizer-divergence-attribution-plan.json) records the thirteen stages,
twenty-six comparisons and eight compiler files expected from this diagnostic.

Comparisons vary input banks under one executable, vary evaluators on an identical
bank, and check each result against directions returned by both full fused updates
on the original pre-update inputs. Bank sensitivity would show that those input
changes matter under a fixed evaluator. Evaluator sensitivity would show that
the implementations differ on identical inputs. Neither alone explains every
interaction in the fused graphs. Mathematical operations, requested precision,
five polar iterations and rtol=atol=1e-5 remain unchanged. Removing refresh changes
compilation; this diagnostic makes no warmed-performance or training-acceptance
claim. CPU harness validation with online W&B tracking is next. No accelerator
diagnostic has been submitted.

## 2026-11-03 01:13 UTC — State comparison CPU preflight passes

The [online CPU preflight](optimizer-state-crossfeed-preflight.json) completed
all thirteen stages with finite outputs and passed all twenty-six comparisons
at rtol=atol=1e-5. Each evaluator reused one compiled executable for its three
state banks and first-bank replay; both replays were bitwise equal. The GCS
report was downloaded and matched the local report byte for byte. All eight
downloaded compiler files matched their recorded SHA-256 values and decompressed
successfully. The smoke used eight virtual CPU devices from
[published source](https://github.com/marin-community/marin-logbooks/commit/3210e49eb133ce3d116f18ce502dd03f93a0c607).
Both the full smoke and its separate logging check used online W&B. This does
not establish TPU numerical behavior.

The [plan](optimizer-divergence-attribution-plan.json) prepares one sequential
repetition of all thirteen stages and twenty-six comparisons on eight v5e chips
in us-west4, with 96 CPUs, 128 GiB host RAM, batch priority, default preemption
retries and no configured runtime limit. The diagnostic uses seed-zero synthetic
gradients and states with the original full matrix geometry; outputs remain
in the same region. No original checkpoint or training dataset is read.

## 2026-11-03 01:16 UTC — State comparison starts on v5e

Iris accepted `ema-gram-20-state-crossfeed-v5e-8-20261103` at 01:13:36 UTC from
[published source](https://github.com/marin-community/marin-logbooks/commit/489ba99fe00cbaa08e05c9ff46b6c7de466eff22). The
[startup record](optimizer-state-crossfeed-started.json) verifies attempt zero,
eight native TPU v5 lite devices, one JAX process, the expected script hash and
online W&B initialization. The matrix-eigensolver versus original full fused
comparison reproduces all seven direction failures at rtol=atol=1e-5, while
all twenty-one updated Gram/momentum leaves pass that same gate.
The fixed-executable state comparisons remain pending.

One log observer follows this job. The v4-32 reference remains queued; no other
candidate diagnostic or training run was submitted.

## 2026-11-03 01:26 UTC — Refreshed input differences affect fixed directions

The [partial state comparison](optimizer-state-crossfeed-v5e-partial.json)
has completed the original direction evaluator's four executions. Replaying the
common state produces bitwise-identical direction outputs. Holding this executable fixed while replacing
the common bank with the fused-original bank fails the 1e-5 direction comparison
for query, key, value and attention-output projections. Replacing it with the
fused-matrix bank fails for query, key, value, gate and up projections, with a
maximum relative Frobenius error of 0.0279447 across the seven projection groups,
using each common-bank direction norm as denominator. Both fused state banks
passed elementwise rtol=atol=1e-5 against the standalone common bank; that input
gate does not imply an output-error bound.

The original direction-only evaluator also differs from its own full fused
reference on all seven direction tensors even when given the state returned by
that reference. This is its returned post-refresh state; equality to every
internal value consumed in the fused graph has not been established. The
controls establish input sensitivity under one fixed executable and differing
outputs across the separate and fused execution paths. They do not fully explain
the original-versus-matrix fused discrepancy. At the latest report, the matrix
direction evaluator is compiling; no optimization is accepted.

Original direction compilation took 376.7 seconds. Finelog's cumulative task
host-memory peak through 01:23 UTC was 67.819 GiB, including original compilation
before the matrix evaluator began. A read-only stack inspection at 01:23:33 UTC found
the process waiting for device execution of the common bank. That inspection may
affect the single-call duration; no warmed-performance claim is made.

## 2026-11-03 01:38 UTC — State comparison completes without numerical acceptance

The [completed state comparison](optimizer-state-crossfeed-v5e-completed.json)
succeeded on attempt zero with task exit zero and online W&B completion.
All thirteen stages produced finite outputs; four of twenty-six comparisons
passed at rtol=atol=1e-5: the two returned-state comparisons against standalone
refresh and the two bitwise-identical direction replays.
Both fixed evaluators remain sensitive to input-bank changes. On identical
common or fused-original banks, the original and matrix evaluators fail the
direction tolerance for six groups; on the fused-matrix bank, all seven fail.
Every separate-evaluator result also fails against both full fused references.
Changing the input bank did not eliminate the discrepancy in these tests.
The matrix-eigensolver candidate remains rejected for training.

Finelog recorded 170 samples and a cumulative task host-memory peak of
78.0391 GiB. The maximum reported JAX allocator peak was 8,076,038,656 bytes
per device, which does not measure total HBM occupancy. All eight compiler
objects are present with matching sizes, totaling 38,905,232 compressed bytes.
The final report's SHA-256 is recorded; the compiler objects have not yet all
been read back for checksum verification. The existing v4-32 reference is
still queued, and the completed job's log observer ended naturally.

## 2026-11-03 01:45 UTC — Inspect compiler operations in their storage region

The next step is a [static compiler inventory](compiler-operation-inventory-plan.json)
of four existing optimized graphs: both full fused updates and both fixed
direction evaluators. A [streaming parser](compiler_operation_inventory.py)
selects tagged polar and Gram-refresh matrix operations, their direct operand
producer shapes/types, and collective operations carrying those same operation
metadata tags. Its local check on the two
previously downloaded query-polar files recovers fifteen matrix operations in
each. Replicated versus direction-sharded output files contain five versus one
explicit conversions to BF16, and zero versus five all-reduce instructions. These are static text counts, not runtime measurements or
proof of a numerical cause.

A CPU-only task will read and checksum the large files within us-west4 and
write a small inventory there. It executes no optimizer kernels or training;
no accelerator is requested. The source and manifest will be published before
execution. Numerical changes remain gated on a concrete hypothesis from the
complete evidence.

## 2026-11-03 01:54 UTC — Regional compiler inventory completes

The [CPU inventory](compiler-operation-inventory-completed.json) succeeded on
attempt zero with exit zero. Its four consumed optimized HLO files passed
full compressed-byte checksum and size verification in us-west4. The task
requested four CPUs and no accelerator; it ran on a worker whose name includes
TPU. Of the eight compiler files saved by the state comparison, the four StableHLO
files were not read by this analysis.

Each of the four graphs contains 105 polar-tagged matrix operations lowered as
convolutions, ten polar-tagged all-reduces, and 63 explicit BF16 conversions
across the entire graph, including operations without polar tags. Equal counts do not imply equal computation. Polar-tagged
all-gather counts are 133/134 for the original and matrix-distributed
eigensolver full updates, respectively, and 136/134 for the corresponding
evaluators that compute directions from externally supplied refreshed state. Some matrix output layouts and direct operand
shapes also differ. These are static compiler records; they establish neither
runtime frequency nor a cause of the direction mismatch between implementations.

Before testing another optimizer implementation on TPU, map the compiler metadata to source
matrix groups and polar iterations and inspect the matched operand-producing
computations. The current inventory stops at direct producers, many of which
are fusion calls. Any later layout intervention must retain the original
precision and pass all direction, Gram and momentum comparisons against the
untouched full-update reference at rtol=atol=1e-5.
No optimization is accepted, and no training has resumed.

## 2026-11-03 02:09 UTC — Map polar products by output dependencies

The three polar-operation stack-frame IDs each occur 35 times in a full graph,
so those IDs alone do not identify matrix groups or iterations. The
[follow-up parser](compiler-polar-lineage-plan.json) follows fusion parameters
and tuple elements from each returned direction. It requires fifteen distinct
polar matrix products in one dependency chain per group. After validating a total dependency order, it assigns those positions to
five iteration triplets using the three-product order in the unchanged
quintic source.
Unsupported polar control flow or shared group operations cause an error.

On each saved isolated query graph, it recovers fifteen operations and five
iterations. A synthetic composition with known graph identities and reversed
output order maps each output to its expected graph; a duplicated output is rejected. The full local
file-parser command also passed checksum, gzip and output checks. Immediate
operand-producing fusion bodies are retained for inspection. These checks
validate the static parser, not optimizer equivalence or runtime ordering.

The next CPU-only job will read the four existing optimized graphs in
us-west4 using the published parser revision. No accelerator or optimizer execution is
requested, and the large compiler files remain in their storage region.

## 2026-11-03 02:14 UTC — Tuple-parameter mapping failure

The [CPU mapping job](compiler-polar-lineage-failed.json) failed before accepting
a group mapping. The first full graph gave 28 upstream polar operations for
query projection, violating the requirement of exactly fifteen dependent
polar products per output. Its partial
inventory and verified input checksum were saved. No optimizer ran.

A local composition exposed a parser defect: when a fusion accepted a tuple
of two independent polar results, the original parameter representation lost
which tuple element was selected and merged both dependency sets. Tracking
parameter leaf paths recovers fifteen products per output in that reproducer.
The reversed-output test and both isolated query graphs still pass the
fifteen-product chain and unique-group checks. This
establishes a parser bug; the full graph must be checked again to determine
whether it explains the observed count of 28.

The [corrected CPU analysis](compiler-polar-lineage-tuples-plan.json) retains
the same chain and unique-group checks and now saves entry instructions on
mapping failure. It will use a fresh job identity and published parser source;
the failed job remains terminal. No numerical or precision setting changes.

## 2026-11-03 04:00 UTC — v4 reference timings survive a later host OOM

The [v4-32 reference](optimizer-cost-v4-32-oom-partial.json) left the queue and
ran four JAX processes on sixteen native TPU v4 chips. All four rank reports
preserve three finite warmed samples for each of the four diagnostic cases.
Rank-local spectral medians are
21.97214–21.97231 seconds; full-update medians are 23.14284–23.14329 seconds.
Rank-local Gram-refresh medians range from 0.117122 to 0.117175 seconds. Alpha-zero direction medians range from
0.06776 to 0.12654 seconds; the individual samples remain in the record.
These isolated synthetic measurements do not establish training MFU.

At 03:42:10 UTC, Iris task 3 was killed for exceeding its 128 GiB host-memory
limit; the other three tasks terminated because their coscheduled sibling
failed. Finelog recorded maximum observed cumulative host-memory peaks of 121.5, 119.7998, 121.6992 and
127.7998 GiB for Iris tasks 0–3, with 206–208 samples per task. JAX rank numbers
differ from Iris task indices; the report keeps their scopes separate.
All four timing reports had already been written. The source next enters
profiling and final artifact persistence, but the exact failing substage is
unobserved. No XPlane object or final device-memory record was found in the
run artifact directory.
W&B is crashed without a completion marker. The job remains terminal.

## 2026-11-03 04:00 UTC — Full-graph mapping remains unresolved

The [tuple-aware parser attempt](compiler-polar-lineage-tuples-failed.json)
also rejected the fused-original graph with 28 query ancestors. The full-graph
parse still failed after the local fix; its cause remains unresolved.
Its preserved entry instructions show multi-output asynchronous collective
fusions, and the same parser representation gives query/key ancestor sets
that overlap in 26 operations. This is
parser dependency information, not evidence of cross-matrix optimizer math.
Inspect those fusion bodies regionally before changing the mapping model or
running another analysis. No group/iteration mapping is accepted.

The previous PR-monitor handle was unavailable and process inventory showed
no survivor. Current comments and reviews contained no new actionable feedback;
the head remains d19131429. One replacement foreground monitor was started.
There is now no queued or running experiment; no accelerator job was resubmitted.

## Final decision

Pending pilot, confirmation and checkpoint-probe evidence. No exponent is
selected and no training advantage is claimed. The implementation PR remains
open; its code has not been merged.
