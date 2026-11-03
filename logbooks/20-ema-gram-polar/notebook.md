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

## 2026-11-03 04:14 UTC — Preserve independent combined-reduction outputs

The [CPU extraction in us-west4](compiler-ir-slices-completed.json) verified
the saved fused-original optimized HLO file and collected six fusion bodies.
The two asynchronous fusions selected because they return query/key directions
have first-output data dependencies only on their fusion-local parameters nine
and ten. Their other tuple buffers are separate data paths. This inspection
does not explain the optimizer direction discrepancies.

The entry graph also contains combined all-reduces. The parser incorrectly
assigned all operand dependencies to every result of those operations.
[OpenXLA semantics](https://openxla.org/xla/operation_semantics#allreduce)
specifies an independent reduction for each tuple element. A local composition
of two known isolated polar graphs reproduced the overcount: thirty polar matrix-product operations
per output before the correction, fifteen per output afterward. Tuple-parameter
and reversed-output fixtures still pass; duplicated outputs are still rejected.

The [implemented correction](compiler-polar-lineage-collectives-plan.json)
preserves independent all-reduce result dependencies. It still requires fifteen
polar products per matrix group, one ordered dependency chain per group, and no
product shared across groups. Full-graph validation remains pending.
The optimizer, its precision and numerical tolerance are unchanged.

## 2026-11-03 04:20 UTC — Full compiler mapping passes

The [corrected CPU mapping](compiler-polar-lineage-collectives-completed.json)
succeeded on all four optimized HLO files after verifying their checksums in
us-west4. Each graph maps 105 polar products to the query/key/value/output and
gate/up/down projection groups, fifteen products per group. Each of five
iterations contains the ordered products X Xᵀ, scaled A A and B X. The report
preserves 240, 238, 210 and 207 unique producer fusion bodies across selected
Gram/polar records for the original/matrix full updates and original/matrix
direction-only evaluators, respectively. Keeping combined-all-reduce tuple
results separate removes the observed mapping overcount. This CPU analysis
did not execute optimizer kernels.

A [matched-body comparison](compiler-matched-producers.json) covers all 105
product positions in each original/matrix pair, including nested producer
fusions. Instruction records preserve shapes, operand order, parameter indices,
scalar constants and attributes preceding metadata, while renaming local SSA
identifiers. The full-update pair differs at 58 positions and the direction-only
pair at 25. Removing only the literal `S(1)` layout tags makes the records textually equal
after SSA renaming at every position. This comparison excludes attributes after
metadata, backend configuration, upstream values and paths connecting producer
inputs. It establishes neither full precision equivalence nor a cause for the
direction discrepancies.

Next inspect matched upstream values and collective reduction paths, including
precision attributes, before selecting a layout intervention. Reuse the saved
regional compiler artifacts. No accelerator candidate is selected and there is
no queued or running experiment. Numerical equivalence to the untouched original full update, saved-state
checks, full trainer restoration and measured training MFU remain unverified.

## 2026-11-03 04:35 UTC — Compiler kernel settings differ at matched polar products

The [static CPU inspection](compiler-polar-paths-completed.json) succeeded and verified
all four optimized compiler files in us-west4. It maps the same polar reduction
groups in both full-update graphs and both direction-only graphs: query and key
X Xᵀ products share an all-reduce at each of five iterations, while value products
are reduced separately. All forty mapped entry all-reduces have the same recorded
algorithm configuration. This does not establish identical runtime reduction
order; nested collectives are outside these entry-level counts.

The matrix candidate distributes independent eigensolver matrices across devices;
the original evaluator is unchanged. The [matched configuration comparison](compiler-matched-paths.json) finds equal
convolution attributes before the metadata field (excluding metadata itself) and equal operation-level backend
configuration at all 105 matched polar operation sites in each original/matrix pair. The calling
fusion's backend configuration differs at 70 product sites for full updates and 28
for direction-only evaluators. Differences include kernel window and iteration
bounds, scoped-memory configuration records and compiler cost estimates. For the first query X Xᵀ
product, the full original uses kernel bounds `[11,32,1]` and iteration bounds
`[1,4,2,1,1]`; the matrix candidate uses `[1,128,1]` and `[11,1,1,1,1]`.
These are static compiler fields, not measured runtime counts or proof of a
numerical mechanism.

The two small [isolated query compiler files](compiler-isolated-query-paths.json)
were also read back with checksum verification. Their first-product kernel
settings differ under replicated versus direction-sharded output layouts, but
neither matches the corresponding full-update configuration. Earlier numerical
controls found identical normalized query inputs under the two output layouts
and a difference in their first-iteration outputs. This combination motivates inspecting individual products;
it does not identify which product first differs or prove tiling is the cause.

The next diagnostic is designed, not implemented or submitted: expose the first
iteration's products on identical frozen query/key/value inputs, preserving the
original expression association, coefficients, precision and sharding constraint.
Compare the traced final output with the shared uninstrumented polar function
at `rtol=atol=1e-5`, also reporting bitwise equality, before interpreting intermediate values. If tracing changes the final output,
report that limitation. Then use identical operands to test the first differing
product under controlled output layouts. Removing fusion is itself a perturbation.
CPU numerical-harness validation with online W&B and source published to the research
branch remain required before one
sequential TPU diagnostic. No experiment is currently running or queued; no
optimization is accepted for training.

## 2026-11-03 04:44 UTC — First-iteration product diagnostic

The [new diagnostic](optimizer-polar-products-plan.json) records the first polar
iteration's normalized input, three matrix products, intervening additions and
final five-iteration output for the full-size query/key/value groups. Replicated and direction-sharded output
layouts receive the same frozen inner tensor. The uninstrumented reference uses
the shared polar function; tracing retains its coefficients, expression
association, default precision and sharding constraint. The existing synthetic
state and refresh recipe is unchanged. No training data or checkpoint is read.

The initial eight-device CPU check completed 36 finite stages and passed all 30
comparisons at `rtol=atol=1e-5`. However, the direction-sharded traced final outputs
were not bitwise equal to their uninstrumented references. That version skipped
all product cross-feeds, so it did not validate the fixed-executable harness.
[This result is retained](optimizer-polar-products-cpu-instrumentation.json).

The revised harness rejects interpreting a bitwise-changed trace as the original
path and continues collecting
explicitly labeled instrumented-path controls. It selects the earliest first-iteration quantity that differs bitwise between
the output layouts, compiles one isolated evaluator per output layout,
and reuses each executable on two frozen operand banks plus a replay of the
first bank. Each bank contains operands saved from one layout's traced computation
of that selected quantity, starting from the common inner tensor. It compares banks under a fixed executable, layouts on an identical
bank, and isolated against traced outputs. When tracing changes the final output,
its intermediates cannot be assumed to reproduce the uninstrumented computation. Even a bitwise-matching final output
does not prove every internal operation is unchanged; isolating products changes compilation boundaries. These controls do not change the optimizer or accept a candidate
for training.

The [final CPU smoke](optimizer-polar-products-preflight.json) passed all 54 finite
stages and 60 comparisons at the unchanged tolerance. All six fixed-executable
replays were bitwise identical. All three direction-sharded traces still change
the final output bitwise, and their original-path interpretation flag remains
false. Online W&B finished successfully; the GCS report matched local bytes, and
all 36 compiler files (73,659 compressed bytes) passed checksum, decompression
and readback checks. A separate online logging check also passed. These small
CPU matrices do not establish TPU behavior. The next run is one non-training
v5e-8 diagnostic in us-west4 from published source, with default preemption
retries and no runtime limit.

## 2026-11-03 04:50 UTC — Product tracing changes the TPU computation

The [v5e diagnostic](optimizer-polar-products-v5e-completed.json) completed on eight
native TPU v5e chips with one JAX process, attempt zero and task exit zero. Iris
confirmed success at 04:48 UTC, and online W&B finished with the collection marker
and `candidate_accepted_for_training=false`. The run used published source
`0798f70171a97a351506ffed95f8a1e4fdc2cded`. Every element of all 54 recorded stage outputs was finite; 54 of 60
comparisons passed elementwise at `rtol=atol=1e-5`. The comparisons cover traced
versus uninstrumented outputs, output layouts, frozen operand banks and replays. Collection success is not numerical
acceptance.

For query/key/value, replicated traced final outputs exactly match their
uninstrumented references using the same output layout. Direction-sharded traced finals fail against their
uninstrumented references, with relative Frobenius errors 0.00539–0.00587.
Their intermediate tensors therefore cannot be assumed to reproduce the
uninstrumented path. Between the two instrumented layouts, normalization matches
exactly and the first nonzero difference among recorded intermediates is at A = X Xᵀ: maximum absolute error
2.3283e-10 in every group. All recorded first-iteration quantities pass tolerance;
the five-iteration final outputs fail, with relative errors 0.00630–0.00636.
The six failures are the three direction-sharded instrumentation checks and
the three traced final-output layout comparisons. This does not isolate the contribution of the first product to the final error.

In the separate A-product controls, the normalized-input banks saved from the two traced layouts are bitwise identical.
For each group, outputs are bitwise identical across the two fixed layout executables,
and each executable also replays its first bank bitwise. The isolated product does not reproduce the
instrumented layout difference. Compilation context remains relevant to the next
investigation, but neither a kernel-tiling cause nor the full optimizer mechanism
is established.

All 36 compiler artifacts, totaling 87,825 compressed bytes, passed checksum,
size and decompression checks. The final report checksum is recorded with the
completion evidence. Finelog observed a cumulative host peak of 13.3496 GiB over
31 samples through 04:48:33 UTC; the per-device JAX allocator peak was
7,116,572,672 bytes, not total HBM. No warmed performance or training MFU was
measured. Next inspect the saved trace and isolated-product compiler artifacts,
including normalization, casts and collective paths, before selecting another
numerical control. No experiment remains queued or running.

## 2026-11-03 05:00 UTC — The isolated product uses a different contraction split

[Output-dependency inspection](compiler-first-product-paths.json) of the saved
query/key/value graphs maps the first polar product A = XXᵀ to one matrix
product in each graph, where X is the normalized input. The traced executable
returns intermediate values for inspection; the isolated executable returns A alone. The mapping uses tuple output identity, operand ancestry and
source metadata. All consumed compressed files passed checksum and size checks.
In all three groups, traced sharded A multiplies operands with local shape
[11,1024,128], combines partial products with an all-reduce, then slices the
output. Isolated sharded A retains the full 1024-wide contraction and has no
collective on the selected output ancestry. Its right operand is declared BF16;
the traced operands are declared F32. These static declarations do not establish
runtime arithmetic precision or a numerical cause.

The previous isolated product therefore did not reproduce the traced product's
contraction split. A new diagnostic control requests column sharding internally
before the same X@Xᵀ expression, while preserving replicated external inputs and
direction-sharded outputs. Each fixed executable will evaluate the saved normalized inputs from both
traced layouts, which were bitwise identical, then replay the first to check repeatability. Compiler inspection must verify the resulting split;
a sharding constraint alone does not verify it. This is an isolated numerical
control, not an optimizer change. Tracing changed the direction-sharded final
output, so its operands cannot be assumed to be the original internal values.
CPU online validation and published source are required before one TPU run.
No experiment is currently queued or running.

The [CPU preflight](optimizer-polar-contracting-preflight.json) passed all 63 finite
stages and 84 comparisons at the unchanged tolerance on eight virtual devices.
Each of nine fixed executables replayed its first input bitwise. The report
matched GCS bytes, all 42 compiler artifacts passed checksum and decompression
checks, and online W&B finished with the collection marker. A separate online
logging check passed. Direction-sharded tracing still changes the final output
bitwise; its original-path interpretation flag remains false. These CPU results
validate the harness, not TPU behavior. The next run is one sequential v5e-8
control in us-west4 from the published preflight revision.

## 2026-11-03 05:13 UTC — Internal column sharding reproduces the traced first product

The [contraction control](optimizer-polar-contracting-v5e-completed.json) succeeded
on eight native TPU v5e chips, attempt zero and task exit zero, from published
source `bf251047c1bc846bf562b6d641e6108c0987fe2a`. Online W&B finished with the
collection marker and no training acceptance. All 63 stages were finite; 78 of
84 comparisons passed at `rtol=atol=1e-5`. The six failures remain the three
traced-versus-uninstrumented direction-sharded final checks and the three traced
final-output layout checks. All nine fixed executables replayed their first
inputs bitwise.

For query/key/value, the isolated A = XXᵀ executable with internal column
sharding matches the direction-sharded traced A with zero measured absolute
error. It differs from the replicated isolated A by at most 2.3283e-10. The
normalized X inputs from the two traces are bitwise identical and reused across
all fixed executables. [Compiler output ancestry](compiler-contracting-product-paths.json)
confirms that the new isolated executable contracts 128 columns locally and
combines partial products with an all-reduce before output slicing. Its operands
are declared BF16, whereas the traced product operands were declared F32. Those
declarations do not establish runtime arithmetic precision.

The source-level internal sharding constraint reproduces the traced first-product
difference in this isolated control. It does not separate collective summation,
casts and kernel selection, or establish how much that difference contributes
to the final error. Tracing still changes the direction-sharded five-iteration
output beyond tolerance, so this result does not explain the untouched full
optimizer discrepancy or accept a performance improvement.

All 42 compiler files, totaling 95,661 compressed bytes, passed checksum, size
and decompression checks. Finelog observed a 13.359375 GiB cumulative host peak
across 33 samples through 05:10:25 UTC. The JAX allocator peak was 7,116,572,672
bytes per device, not total HBM. No warmed performance or training MFU was measured.
The next proposed control compiles the remaining computation after the first A
product once. It holds that executable and normalized X fixed while substituting
replicated isolated A and internally column-sharded isolated A, then checks the
final polar output.
This tests conditional sensitivity to A; separating the computation still
changes compilation and requires its own reference checks. It is not implemented
or submitted. No experiment remains queued or running.

## 2026-11-03 05:19 UTC — Hold the polar continuation fixed while changing A

The new `--polar-continuation` diagnostic takes the normalized input X and
first product A = XXᵀ as external inputs. It computes the remainder of the first
quintic update and the four later updates, using the shared original coefficients,
expression association, source dtype and default precision. It returns only the
final polar output. For each query/key/value group and each output layout
(replicated or direction-sharded), one compiled executable evaluates A from the replicated isolated product, then A
from the internally column-sharded isolated product, then replays the first.
X remains identical across those calls. Exact input fingerprints and output
comparisons record repeatability and sensitivity to A under that executable.

Each result is compared with both output layouts of the shared uninstrumented
polar reference and the intermediate-returning trace. The tolerance remains
`rtol=atol=1e-5`, with bitwise equality reported separately. Removing the first
product from the compiled continuation changes compilation; reference failures
therefore prevent interpreting the continuation as the original fused path.
Even matching final values cannot verify internal identity. This control does
not change the optimizer or establish training performance.

The implementation reuses the fixed-executable harness from `--state-crossfeed`.
Before any TPU submission, validate both the new continuation mode and that
existing state-comparison mode on eight virtual CPU devices with online W&B.
The continuation mode should produce 81 finite stages, 150 comparisons and 54
compiler files; the state comparison retains 13 stages and 26 comparisons.
No accelerator job is queued or running.

The [CPU preflight](optimizer-polar-continuation-preflight.json) passed all 81
finite stages and 150 fixed-tolerance comparisons. All fifteen fixed executables
replayed their first inputs bitwise. The state-comparison regression passed all
13 stages and 26 comparisons, including both bitwise replays. Both online W&B
runs finished successfully, both reports matched GCS bytes, and all 54 new-mode
compiler files plus eight regression files passed checksum and decompression
checks. The separate online logging check also passed. These small CPU matrices
do not establish TPU behavior; the next step is one sequential v5e-8 control
from the published preflight revision.

## 2026-11-03 05:33 UTC — First-product changes affect a fixed continuation

The [continuation diagnostic](optimizer-polar-continuation-v5e-completed.json)
succeeded on eight native TPU v5e chips, attempt zero and task exit zero, from
published source `2da86114ff23d885b1912bf74a3c6d02c6e0d59a`. All 81 recorded
stages were finite; 93 of 150 comparisons passed at `rtol=atol=1e-5`. All fifteen
fixed executables replayed their first inputs bitwise. Online W&B finished with
the collection marker and no training acceptance.

The continuation computes the remainder of the first polar update and the four
later updates from supplied normalized X and A. Holding X and that compiled
continuation fixed, substituting A = XXᵀ
from the internally column-sharded isolated product for A from the replicated
isolated product changes the final output beyond tolerance in all three groups
and both output layouts. The relative Frobenius differences range from
0.005393 to 0.005850, using the output with replicated A as the norm denominator.
The A inputs differed by at most 2.3283e-10 per element.
Changing output layout while keeping the A bank fixed also fails all six
comparisons. These measurements establish conditional sensitivity to both A and
layout in the separate continuation. They do not give additive contributions to
the untouched full optimizer's error.

For query/key/value, the replicated continuation supplied with replicated A is
bitwise identical to the uninstrumented replicated polar result and the traced
replicated result. The direction-sharded continuation supplied with the
column-sharded A is bitwise identical to the traced sharded result. No
continuation result passes against the uninstrumented sharded reference. The
57 failures comprise six earlier tracing/final-layout checks, six A-bank checks,
six continuation-layout checks and 39 reference checks. Matching final outputs
does not verify identical internal values, and the untouched sharded computation
remains unreproduced.

All 54 compiler artifacts, totaling 127,623 compressed bytes, passed checksum,
size and decompression checks. Finelog observed a cumulative host peak of
13.359375 GiB over 48 samples through 05:30:44 UTC; the per-device JAX allocator
peak was 7,116,572,672 bytes, not total HBM. No warmed performance or training
MFU was measured. Next compare saved compiler paths for the uninstrumented,
traced and continuation computations, mapping products by metadata and output
dependencies. The files are already available locally; no new analysis
accelerator is needed. No optimizer intervention or new numerical experiment is
selected, and no experiment remains queued or running.

## 2026-11-03 05:45 UTC — First-iteration compiler paths remain different

The [saved-graph comparison](compiler-continuation-paths.json) finds differences
in the inspected product fields only in the first polar iteration. The parser
mapped all 18 query/key/value graphs by final-output dependencies: 15 ordered
products per uninstrumented or traced computation, and 14 per continuation
because its first A = XXᵀ is an external input. Each iteration computes
A = XXᵀ, C = (cA)A and D = (bA + C)X. Fusion-parameter and tuple-leaf traversal,
source metadata and contraction dimension labels validate the mapping without
using instruction text order. All 36 consumed StableHLO/optimized-HLO files
passed checksum, size and local-byte checks. Six uninstrumented mappings matched
the existing independent lineage parser; twelve earlier single-product ancestry
results remained unchanged.

For each group and output layout, uninstrumented-versus-traced configurations
of the enclosing compiled fusion differ at first-iteration A, C and D. These
configurations include kernel-window and iteration bounds; they are separate
from the product operation's own backend settings. The compared product
attributes, operation-level backend configurations, operand declarations and
calling-fusion configurations match at every product in iterations two through
five. This does not cover all upstream graph structure or internal values.
In the sharded paths, traced C declares a BF16 left operand and a 128-column
right operand, whereas uninstrumented C declares two F32 full-size operands.
The uninstrumented first-A fusion consumes input before normalization and the
norm denominator, and computes the division inside that fusion. The traced
first-A fusion consumes a separately computed normalized tensor. These
compiler declarations do not establish runtime arithmetic precision or
numerical cause.

For uninstrumented-versus-continuation sharded paths, operand declarations and
operation-level attributes/configurations match for all fourteen corresponding
products; only first C has a different calling-fusion configuration. The
continuation receives external normalized X and A, so this comparison does not
establish equal inputs to corresponding internal products. Its final output
still fails the uninstrumented sharded reference despite these matching fields.

Next design a control of the normalization/first-iteration boundary, retaining
source math and fixed tolerance. Distinguish input before normalization from
externally normalized input at the first-product computation, and compare final
outputs with the shared uninstrumented polar function. Changing that boundary
can change compilation; agreement with a final reference would not prove
internal identity. No precision or tiling override, numerical submission or
optimizer intervention is selected. No experiment is queued or running.

## 2026-11-03 05:57 UTC — Separate normalization from polar products

The [next diagnostic](optimizer-polar-normalization-plan.json) compares the
shared five-step polar function with the same quintic iterations applied to an
externally normalized input. The shared polar function with zero iterations
computes that input, preserving its norm, epsilon and sharding constraint. Each
square query/key/value group has shape [11, 1024, 1024] on TPU, using the seed-zero
synthetic Gram/momentum refresh recipe recorded in the diagnostic plan.

Two additional functions return only the first Gram product A = XXᵀ: one
normalizes its input inside the compiled function; the other receives the
separately normalized input. All three diagnostic functions, including the
five-step function, run under replicated and direction-sharded outputs with
replicated external inputs. Every fixed executable compiles once,
evaluates its input, then replays it. Input fingerprints, finite outputs,
per-stage memory, compiler artifacts, tolerance comparisons and bitwise checks
are recorded. The normalized five-step result is compared with both output
layouts of the shared uninstrumented polar reference.

Expected collection is 69 finite stages, 48 comparisons, eighteen fixed-executable
replays and 54 compiler files. A separate online logging check and an eight-device
CPU smoke must pass before one sequential v5e-8 diagnostic. Splitting
normalization or returning a first product changes compilation; final agreement
cannot verify internal identity. Precision, coefficients, tolerance and training
remain unchanged. These controls cannot establish an accepted optimizer
improvement. No accelerator has been submitted.

The [normalization-boundary CPU smoke](optimizer-polar-normalization-preflight.json)
completed all 69 finite stages and passed all 48 comparisons at
`rtol=atol=1e-5`. All eighteen fixed executables replayed bitwise. The online
W&B run finished, its GCS report matched local bytes, and all 54 compiler files
passed checksum, size and decompression checks. A separate online logging
check also passed. These checks use small matrices on eight virtual CPU devices
and do not establish TPU behavior. Next submit one sequential v5e-8 diagnostic
from the published source and preflight, with batch priority, default preemption
retries and no wall-clock limit.

## 2026-11-03 06:07 UTC — Normalization separation changes sharded polar results

The [normalization-boundary diagnostic](optimizer-polar-normalization-v5e-completed.json)
succeeded on eight native v5e chips, attempt zero and task exit zero, from source
`9e51263b02a40bdc477db5dd33c4852f10f511df`. All 69 stages were finite,
33 of 48 comparisons passed at `rtol=atol=1e-5`, and all eighteen fixed
executables replayed bitwise. Online W&B finished with the collection marker
and `candidate_accepted_for_training=false`.

For each query/key/value group, the isolated first Gram product A = XXᵀ matches
bitwise with normalization inside or outside its executable, under either
output layout. Those isolated products also match across replicated and
direction-sharded outputs. The externally normalized five-step polar result
matches the shared uninstrumented replicated reference bitwise under replicated
outputs. Under direction-sharded outputs it fails the corresponding
uninstrumented reference: relative Frobenius error is 0.005941 for query,
0.005004 for key and 0.005425 for value, with the uninstrumented sharded output
as denominator. The fifteen failed comparisons are three normalized-polar
layout checks, three uninstrumented-reference layout checks and nine
final-reference checks.

This measures sensitivity to separating normalization in a different compiled
computation. Equality of isolated first products does not establish equality of
the first products inside either five-step executable. The result does not
isolate runtime precision, collective summation or the untouched full optimizer
mechanism, and no optimization is accepted.

All 54 compiler files, totaling 103,743 compressed bytes, passed checksum, size
and decompression checks. Finelog observed a cumulative host peak of
13.359375 GiB over 39 samples through 06:06:00 UTC. The JAX allocator peak was
7,116,572,672 bytes per device; it is not total HBM. No warmed timing or training
MFU was measured. Next map the saved normalized-input five-step and
isolated-product compiler paths against their uninstrumented references using
output dependencies and source metadata. The files are local; no further
accelerator is selected or running.

## 2026-11-03 06:18 UTC — First-product contraction differs after normalization separation

The [normalization-control compiler comparison](compiler-normalization-boundary-paths.json)
maps 24 graphs: twelve five-step graphs with fifteen visible products each and
twelve isolated-product graphs. All 48 consumed IR objects passed checksum,
size and local-byte checks. The twelve full-graph product chains match the
existing independent lineage parser, and twelve earlier single-product ancestry
results are unchanged. Three replicated normalized-input graphs contain an
opaque `ConcatBitcast` custom call. The parser follows its declared operands
conservatively and makes no claim about hidden operations or runtime behavior.

For query/key/value under direction-sharded outputs, the original five-step
graph uses a 128-column local contraction for first A = XXᵀ followed by an
all-reduce. The externally normalized five-step graph uses a full 1,024-column
first contraction with no corresponding first-A all-reduce. Later iterations
retain local contractions and reductions. Among the compared product fields,
only first A changes operand declarations and enclosing-fusion configuration;
operation-level attributes and backend configurations match. Under replicated
outputs, operand declarations match throughout, while enclosing-fusion
configurations differ at first A and D = (bA + (cA)A)X. Static configurations do
not establish runtime precision or numerical cause.

Both isolated first-product variants use the full 1,024-column contraction.
With raw input and sharded output, the graph gathers a declared BF16 normalized
operand before multiplication; the externally normalized variant has no
collective in its output ancestry. Their outputs nevertheless match bitwise.
Thus those isolated controls did not hold the first-product contraction
partition equal to the original sharded five-step path.

After verifying both final-report checksums, the raw input before normalization
and normalized-input fingerprints match the earlier continuation experiment in
all three groups. All six new five-step output fingerprints match the
corresponding continuation supplied with replicated A. These endpoint matches
do not prove equal hidden first products.

Next design a paired first-product control that requests internal column
sharding for both raw-input normalization and externally normalized input.
Verify the compiled local contraction and reduction before interpreting
results. Use shared normalization and unchanged coefficients, dtype and
precision; feed the resulting A banks into fixed continuations holding
normalized X constant, with replay and original five-step reference checks.
This is a conditional diagnostic: constraints and boundaries can change casts
and kernel selection, and final agreement would not prove internal identity.
No optimizer intervention or accelerator submission is selected; the new
implementation needs its own online CPU validation and published source before
one sequential TPU diagnostic.

## 2026-11-03 06:29 UTC — Pair normalization boundaries with internal column sharding

The [paired control](optimizer-polar-partition-plan.json) now requests the same
internal column partition for two first Gram products A = XXᵀ. One normalizes
the raw input inside the compiled first-product function before applying the
constraint; the other receives a separately computed tensor using the same
shared normalization formula. Internal tensor equality is not assumed. Each path runs
under replicated and direction-sharded output layouts. Saved compiler ancestry
must verify the local contraction and reduction before interpreting the result.

For each query/key/value group, one fixed continuation per output layout will
evaluate all four resulting A banks with normalized X held fixed, then replay
the first bank. It computes the remainder of iteration one and four later
quintic iterations. Final outputs are compared to both shared, uninstrumented
five-step references at `rtol=atol=1e-5`, with bitwise equality also recorded.
The implementation reuses shared normalization, product, continuation and
fixed-executable helpers. Source-level precision settings, coefficients and full matrix shapes stay
unchanged. Constraints may change casts and kernel selection; final agreement
would not prove internal identity. Online CPU validation is pending, and no
accelerator has been submitted.

## 2026-11-03 06:36 UTC — Paired partition control passes CPU validation

The [CPU preflight](optimizer-polar-partition-preflight.json) from source
`dded42b0429b54f3287b1f918bef2a19bc76f986` completed 87 finite stages and
passed all 111 comparisons at `rtol=atol=1e-5`. All eighteen fixed executables
replayed bitwise. Online W&B run `31292c51856a` finished; the separate logging
check `c93a30308fb4` also finished with its success marker. The regional GCS
report matches local bytes, and all 54 compiler objects passed checksum, size
and decompression checks. This validates the small-matrix CPU harness; TPU
partitioning and numerical behavior remain unverified.

The next diagnostic uses one eight-chip native v5e host in us-west4 with
96 CPUs, 128 GiB host memory, batch priority, default preemption retries and no
wall-clock limit. Its [published plan](optimizer-polar-partition-plan.json)
requires checking the compiled first-product contraction and reduction before
interpreting numerical comparisons. It reads no original checkpoint or training
data and cannot establish optimizer acceptance or training MFU.

## 2026-11-03 06:44 UTC — Paired partition control reproduces selected sharded polar endpoints

The [paired partition diagnostic](optimizer-polar-partition-v5e-completed.json)
succeeded on eight native v5e chips, attempt zero and task exit zero, from
`c38032801a63afe4106a9230a2b088f6535c3eb9`. All 87 stages were finite,
42 of 111 comparisons passed at `rtol=atol=1e-5`, and all eighteen fixed
executables replayed bitwise. Online W&B run `b74aea8d111e` finished with
collection complete and `candidate_accepted_for_training=false`.

For query/key/value under each output layout, first A = XXᵀ is bitwise equal
whether shared normalization occurs inside the first-product executable or
its normalized input is supplied separately. All twelve saved first-product
graphs contain a local 128-column contraction followed by an all-reduce.
Matrix-product operands on raw-input paths are declared F32, and those on
externally normalized paths BF16;
these declarations do not establish runtime precision. Changing the product's
output layout still changes A by a maximum absolute elementwise difference
of 2.3283064e-10, although it passes the fixed tolerance.

For all three groups, the direction-sharded fixed continuation supplied with A
from either replicated-output first-product path is bitwise identical to the
shared uninstrumented sharded five-step reference. Supplying A from either
sharded-output product instead fails that reference, with relative Frobenius
errors 0.005872, 0.005390 and 0.005769 for query, key and value respectively.
The denominator is the original sharded reference norm. All replicated-output
continuations fail both original references. The 69 failures comprise three
original-reference layout checks, twelve A-bank substitutions under fixed
continuations, twelve continuation-output-layout checks and 42 reference checks.

This reproduces selected final endpoints while exposing sensitivity to the
first product's output layout despite matching contraction partitions. It does
not verify hidden internal values, isolate runtime precision or collective
order, or establish equivalence of the full optimizer. No warmed performance
or training MFU was measured, and no optimizer change is accepted.

All 54 compiler objects, totaling 108,825 compressed bytes, passed checksum,
size and decompression checks. The [first-product ancestry](compiler-partition-product-paths.json)
rechecked twelve optimized objects against local bytes. Each graph has one
visible product with 128-column operands and one all-reduce in its output ancestry. Finelog observed a host peak of 13.399414 GiB
across 58 samples through 06:43:17 UTC. The JAX allocator peak was
7,116,572,672 bytes per device; it does not measure total HBM. The sole log
observer ended naturally. No experiment is queued or running.

Next inspect saved first-product producer and reduction paths across output
layouts, including enclosing-kernel configurations and normalization producers.
Use the existing small artifacts and dependency mapping; no additional
accelerator is selected. Static differences alone cannot establish cause or
justify a precision or tiling override.

### 2026-11-03 — First-product producers match; reduction configurations differ

Within each normalization variant, the saved q/k/v first-Gram graphs place
the inspected static differences between output layouts after the local matrix
product. The product-producing instruction graphs match across replicated and
direction-sharded outputs after
renaming identifiers and removing source metadata. The comparison retains
shapes, constants, parameter indices, ordered repeated operands, nested fusion
bodies and backend configurations. It does not establish equal runtime values.
The [analysis](compiler-partition-reductions.json) covers eighteen graphs and
reverifies checksum, size and local bytes for all 36 consumed compiler objects.

All six replicated-output isolated products use a standalone all-reduce with
`RotatedPincerEmitter` and `UniDirection1DRingStrategy`. All six sharded-output
products use a custom fusion containing all-reduce and output slicing, configured
with `SingleInputAllReduceScatterFusion` and `StrategyRing`. Both paths contract
128 local columns. The original sharded five-step polar's first product also
uses the standalone all-reduce configuration. Its product-producing graph and
parent kernel configuration match both raw-input isolated products for each
q/k/v group. The externally normalized products have different producer graphs.

[Validation](compiler-partition-reductions-validation.json) checks six complete
polar product chains, twelve isolated one-product graphs and twelve comparisons
against the same-job original sharded reference. Ten tests of the graph-comparison procedure verify
invariance to identifier and metadata changes, and sensitivity to constants,
parameter indices, operand order and repetition, shapes and backend settings.
These static records support a focused reduction control; they do not establish
runtime reduction order, precision or the cause of the numerical discrepancy.
No optimizer change is accepted, and no new experiment is running or queued.

Next design a control that freezes each device's partial first-Gram product
and supplies the same bank of device partials to both reduction paths. Feed the
resulting Gram banks into fixed continuations with identical normalized inputs.
Materializing partial products changes compilation, so the control must verify
its compiler paths and compare final outputs with the untouched shared polar.
This control has not been implemented or submitted. It requires its own online
CPU validation and published source before one sequential TPU diagnostic.

### 2026-11-03 — Prepare a fixed-partial collective control

The new `--polar-collectives` diagnostic stores each device's unreduced local
first-Gram product in a leading axis sharded across devices, with one partial
per device. A fixed explicit all-reduce and a fixed reduce-scatter executable
then read the same bank. Each executable replays
its first input; the report preserves whole-bank and per-device fingerprints.
The two resulting Gram banks feed fixed continuations with identical
normalized inputs. Each continuation consumes the supplied first Gram product,
finishes iteration one, then performs four later iterations. Same-job references use the untouched shared polar and the
previous raw-input first-product function under both output layouts.

The diagnostic keeps the full synthetic state geometry, shared normalization,
coefficients, source precision and `rtol=atol=1e-5`. It changes no optimizer or
trainer code. Materializing partial products changes compilation, and explicit
collectives may choose different algorithms from the saved original graphs.
Compiler-path checks and reference comparisons are required before interpreting
the result. Even matching final outputs would not establish hidden internal
identity or full optimizer equivalence.

Online CPU run `848c0f491225` completed all 75 finite stages and all 69
comparisons at the fixed tolerance. All fifteen executable replays were bitwise
identical. Its GCS report matched local bytes, and all 60 compiler objects
passed checksum, size and decompression checks. The separate online logging
check `27a75440f198` also finished successfully. These CPU checks do not establish
TPU numerical or collective behavior. The [plan](optimizer-polar-collectives-plan.json)
requests one sequential eight-chip v5e diagnostic in us-west4, using published
source, regional outputs, batch priority, default preemption retries and no
wall-clock limit.

The diagnostic was accepted at 07:13:54 UTC from `b33a382549`.
[Startup checks](optimizer-polar-collectives-started.json) verified eight native
v5e devices, one JAX process, the expected source and script hashes, and online
W&B run `310521548952` with a successful logging check before kernels. One log
observer owns this attempt. Numerical and compiler-path results are pending.

### 2026-11-03 — Collective paths differ on identical frozen partials

The [frozen-partial diagnostic](optimizer-polar-collectives-v5e-completed.json)
succeeded on eight native v5e devices, attempt 0, task exit 0, verified at
07:19 UTC. It completed 75 finite stages; 33 of 69 comparisons passed the fixed
`rtol=atol=1e-5` gate. All fifteen fixed executables replayed bitwise. Online
W&B run `310521548952` finished with collection complete and
`candidate_accepted_for_training=false`.

For each q/k/v group, both reduction executables consumed the same fingerprinted
bank of device-local partial products. Their Gram outputs differed by at most
2.3283064e-10, which passes the elementwise tolerance. Each reduction output
matched the corresponding same-job raw-input first-Gram reference bitwise:
all-reduce matched replicated output, and reduce-scatter matched sharded output.
Holding normalized X and a continuation executable fixed, substituting these
Gram outputs changed the final result by 0.539–0.589% relative Frobenius error,
using the continuation supplied with all-reduce output as denominator.
All six substitutions failed the fixed tolerance.

For all three groups, the direction-sharded continuation supplied with
all-reduce output matched the untouched sharded five-step polar bitwise.
Supplying reduce-scatter output failed that reference. All replicated-output
continuations failed both untouched five-step polar references (replicated and
direction-sharded outputs). The 36 failed comparisons comprise
three original-reference layout checks, six Gram-bank substitutions, six
continuation-layout checks and 21 final-reference checks.

The [compiler-path inspection](compiler-frozen-collective-paths.json) verified
nine graphs and eighteen consumed objects. Each local-product graph contains one
convolution with 128-column operands and no collective reduction; each
reduction graph contains one collective and
no matrix product. All-reduce uses `RotatedPincerEmitter` with
`UniDirection1DRingStrategy`; reduce-scatter uses
`SingleInputAllReduceScatterFusion` with `StrategyRing`. This control demonstrates
numerical differences between these compiled collective paths on identical
partial inputs, and their effect under fixed continuations. It does not identify
the runtime summation order or precision, verify hidden original partial values,
or explain the complete fused optimizer discrepancy.

All 60 compiler objects, totaling 111,135 compressed bytes, passed checksum,
size and decompression checks. Finelog observed a host peak of 13.419922 GiB
across 46 samples through 07:18:58 UTC. The JAX allocator peak was 7,116,572,672
bytes per device; this is not total HBM. No warmed performance or training MFU
was measured. The log observer exited naturally; no experiment is running or
queued, and no optimizer is accepted.

Next map the first-Gram collective ancestry in the existing original and
matrix-layout full-update compiler records to the validated isolated paths.
Earlier equality of all-reduce configurations in HLO entry computations does
not by itself compare
enclosing collective fusions. Check group/iteration mapping and both operation
and enclosing-fusion settings before deciding whether this isolated mechanism
applies to the full-update discrepancy. Use existing records; no new numerical
submission or precision/tiling intervention is selected.

### 2026-11-03 — Full updates use the same first-Gram collective path

The [full-update mapping](compiler-full-update-collective-comparison.json)
finds no all-reduce-to-reduce-scatter substitution at the first q/k/v Gram
products. The [matrix-layout variant](optimizer-matrix-v5e-memory-plan.json)
distributes independent full-matrix eigendecompositions across devices. In both
updates, q and k feed a combined all-reduce through separate tuple operands; v feeds a separate all-reduce.
Each operand directly selects tuple element zero from its product-producing
fusion. The reductions are in the HLO entry computation, with no enclosing
collective fusion at those sites.

All six mapped paths use `RotatedPincerEmitter` and
`UniDirection1DRingStrategy`, matching the isolated all-reduce configuration.
The original and matrix-layout collective backend records differ only in their
barrier IDs. Their product-producing fusion settings differ in kernel window
bounds, iteration bounds, estimated cycles and scoped-memory records. These
static differences do not establish numerical cause or runtime precision.

The analysis verified the 7,108,888-byte captured regional report against its
published SHA256 and checked all six mappings against the earlier group/iteration
records. Original IR checksum checks belong to that earlier regional extraction;
the large HLO files were not downloaded again. The comparison preserves which
tuple operand belongs to q or k and checks the direct product-output projection.
It does not verify hidden input values or intermediate arithmetic inside each
product fusion.

The isolated collective experiment therefore does not explain the full-update
mismatch as an all-reduce-to-reduce-scatter replacement at these first-Gram
sites. Its conditional sensitivity result remains valid. No numerical experiment
ran in this analysis, no optimizer is accepted, and training MFU remains
unverified.

Next trace the first-product fusion inputs back to source gradient/state leaves
and normalization scalars in the existing records. Compare parameter-leaf roles
and intervening producer computations before designing a fixed-input product
control. Matching roles would not establish equal runtime values. Use local
captured records first; any missing large HLO inspection stays in-region. No
numerical submission, precision change or tiling override is selected.

## 2026-11-03 07:50 UTC — Preparing input-provenance extraction

The saved local records contain the q/k/v normalization divisions but omit
some enclosing fusion bodies needed to bind their inputs to source parameters.
A CPU-only extraction will read the two existing optimized HLO files in
us-west4 and trace those divisions and first polar Gram products back to entry
parameter leaves. The [plan](compiler-input-provenance-plan.json) records the
input checksums, exact command and resources; the
[parser](compiler_input_provenance.py) runs no optimizer kernels.

Local controls passed for nested fusion bindings, tuple projection, independent
combined-all-reduce operands, reversed output order, repeated operands and
missing-target rejection. Opaque calls are flagged and conservatively depend
on all declared operands; a control confirmed that this uncertainty remains
confined to the affected branch. These are static dependencies, not proof of
minimal semantic inputs or equal runtime values. The job requests four CPUs,
8 GiB memory and no accelerator, with default preemption retries and no runtime
limit. No numerical intervention or training run is selected.

## 2026-11-03 07:52 UTC — Full-update inputs map to the same source roles

The CPU-only provenance extraction succeeded on attempt 0 with task exit 0.
It verified both optimized HLO inputs by SHA256 and size in us-west4, then
mapped twelve selected operations and their twenty-four operand dependency
sets. The [completed record](compiler-input-provenance-completed.json) contains
the exact source, report checksum and source-leaf mappings.

For each of q, k and v, the original and matrix-eigensolver full updates map the
normalization numerator, normalization denominator and both first-Gram operands to that
group's gradient, momentum, left Gram and right Gram. Fusion parameter binding
and tuple projection preserve these source identities. Both operand positions
of each first-Gram product reference the same HLO value, and the report retains
both positions. No other group's entry parameter appears in these
conservative dependency sets.

Every path includes opaque calls or loops whose outputs are conservatively
assigned dependencies from all declared inputs. The matrix-layout k path lacks
the asynchronous-collective call tags found in the original k path. These tags
describe the parsed graph, not runtime execution or numerical differences.
The mapping does not evaluate eigensolver loops, establish minimal semantic
inputs, or show that the two implementations produce equal inner matrices or
normalization values. The earlier shared first-Gram reduction configuration
also remains insufficient to explain the full-update discrepancy.

Next design a diagnostic that exposes the q/k/v inputs to the polar operation
inside each full update, reusing shared optimizer math. Before interpreting
those captured values, compare all final direction and state leaves with each
untouched full-update reference at the fixed tolerance and record bitwise
agreement. Capturing values changes compilation; failed reference agreement
would forbid original-path interpretation, while matching final outputs would
still not prove internal identity. This is design work only. No numerical
submission is selected, no optimizer is accepted, and training MFU remains
unverified. The CPU task is terminal; no experiment or queue remains active.

## 2026-11-03 08:01 UTC — Preparing full-update polar-input capture

The diagnostic now exposes existing q/k/v normalization operands from the
traced shared `full_update` program. It appends the inner matrices and the
broadcast `norm + eps` denominators to the outputs while retaining every
original equation and all twenty-eight direction/state outputs. Mapping uses
source scopes and dependencies of final outputs, with one distinct polar
normalization required per matrix group. Abstract tracing found all seven
groups and preserved all 851 equations in the original small-shape program;
only the q/k/v groups receive extra outputs.

The [plan](optimizer-full-input-capture-plan.json) includes two controls for
each eigensolver layout: evaluate the unchanged traced program, then evaluate
that program with the six extra outputs. Both are compared against the direct,
untouched shared full-update function. Each controlled executable replays the
same fingerprinted input bank. The fixed rtol=atol=1e-5 gate covers all final direction
and state leaves; bitwise equality is also recorded. Adding outputs or using
the evaluator can change compilation, so failed agreement forbids interpreting
captured values as the original internal path. Final equality alone would not
prove internal identity.

Online CPU validation is pending. No accelerator job is submitted, no optimizer
change is accepted, and no checkpoint or training data is read.

## 2026-11-03 08:04 UTC — Full-update capture CPU validation passed

The new mode passed on eight virtual CPU devices: twelve finite stages,
eighteen comparisons at rtol=atol=1e-5, and four bitwise-identical fixed-executable
replays. Both unchanged-program evaluators and both capture variants match their
own untouched full-update references bitwise across all twenty-eight leaves.
The six captured q/k/v values also match bitwise across solver layouts in this CPU run.

The [preflight](optimizer-full-input-capture-preflight.json) records the fresh
online logging check and numerical run. Both W&B runs finished, the report read
back from GCS matches local bytes, and all twelve compiler files plus two
traced-program files passed size, SHA256 and decompression checks. The script
and capture-helper hashes identify the validated implementation.

This validates the small CPU harness. A single sequential v5e-8 diagnostic will
test the same controls at full matrix sizes, using the published source and
regional outputs. No training checkpoint is read. Expected numerical mismatches
will be retained; collection completion will not accept an optimizer or establish
training MFU. The original precision, refresh, shape and tolerance requirements
remain unchanged.

## 2026-11-03 08:05 UTC — Full-update capture running on v5e

One diagnostic was accepted at 08:04:05 UTC from published source
`fd3a36bba20f06a93425205a8592ff3e0a3f601d`. Iris reports attempt 0 running;
startup confirms eight native TPU v5 lite devices in one JAX process, the
validated script/helper hashes, and fresh online
[W&B tracking](https://wandb.ai/marin-community/marin/runs/ema-gram-20-attribution-e084152068d5)
with a successful logging check before kernels. The
[startup record](optimizer-full-input-capture-started.json) links the exact
regional output directory and execution identity.

The task requests 96 CPUs, 128 GiB host memory and one v5litepod-8 host in
us-west4, at batch priority with default preemption retries and no runtime
limit. One log observer is attached. Full-size numerical capture gates,
replays, memory and final artifacts remain pending. No original checkpoint
or training dataset is read; this is not a resumed training process.

## 2026-11-03 08:14 UTC — Original reference finite; evaluator compiling

The active full-update capture job completed its untouched original reference
with finite outputs. The 424.861-second call includes compilation and does not
measure warmed optimizer performance. Three stages are now finite, but no
numerical comparison has completed. The report has advanced to compilation of
the unchanged-program evaluator; a read-only, nonblocking stack inspection at
08:13 UTC found the main thread in JAX's `backend_compile_and_load` through
`fixed_bank_outputs`.

The [partial record](optimizer-full-input-capture-partial.json) preserves the
report checksum, compilation observation and telemetry query. Finelog recorded
75 samples through 08:12:22 UTC with a cumulative host-memory peak of
56.449 GiB, below the 128 GiB reservation. Its accelerator-memory field is null.
The original full-size program maps all seven polar normalizations and preserves
851 equations when appending the six q/k/v operands as outputs. Its small saved Jaxpr file
passed SHA256, size and decompression readback; the large compiler files remain
regional and have not been re-downloaded for this check.

The stack inspection may add compilation overhead. It establishes where the
process was waiting at that instant, not a completion estimate or a numerical
result. Keep this job and its sole log observer running. Reference agreement,
capture comparisons, fixed-executable replays and final artifacts remain
pending; no concurrent candidate or training run is selected.

## 2026-11-03 08:30 UTC — Capture changes original full-update outputs

The unchanged-program evaluator matches the untouched original full update
bitwise on all 28 output leaves. Appending the six q/k/v capture outputs changes
three direction tensors beyond the fixed `rtol=atol=1e-5` gate: v_proj, o_proj
and up_proj. Their relative Frobenius errors are 0.006707, 0.002827 and 0.024913,
respectively, using the untouched reference norm as denominator. The other 25
output leaves, including all 21 Gram/momentum leaves, match bitwise. Both fixed
executables replay bitwise on the identical input bank.

The [original-branch gate record](optimizer-full-input-capture-original-gate.json)
preserves all five comparisons, the failed capture gate and the report checksum.
Preserving the Jaxpr equations did not preserve the compiled computation's
outputs when extra outputs were added. The captured values therefore cannot be
interpreted as values from the untouched original path. Exact agreement of the
unchanged evaluator establishes endpoint agreement, not internal-value identity.

Iris still reports attempt 0 running, and the report has advanced to the untouched
matrix-layout reference. Seven stages are finite; three of five comparisons pass.
Finelog recorded 207 samples through 08:26:58 UTC with an observed cumulative
host peak of 90.590 GiB, below the 128 GiB reservation. Its accelerator-memory
field remains null. New large compiler files remain regional; their reported
checksums have not been independently read back in this check.

Keep the existing job and sole log observer running through the matrix branch
and final artifacts. No additional candidate is selected. These compilation and
single-call observations are not warmed performance or training MFU.

## 2026-11-03 08:46 UTC — Both full-update captures fail reference gates

The [capture diagnostic](optimizer-full-input-capture-v5e-completed.json)
completed successfully on attempt 0 with task exit 0. All 12 stages were finite,
12 of 18 comparisons passed at `rtol=atol=1e-5`, and all four fixed executables
replayed bitwise. Both unchanged-program evaluators match their own untouched
full-update references bitwise on all 28 leaves.

Adding six q/k/v outputs changes the original branch's v_proj, o_proj and
up_proj directions beyond tolerance, as recorded above. It also changes the
matrix branch's k_proj and down_proj directions beyond tolerance, with relative
Frobenius errors 0.006667 and 0.027983 against that branch's untouched reference.
All 21 updated state leaves pass in both captures, but the matrix capture changes
the down_proj left Gram by a maximum absolute elementwise error of
1.7764e-15. Both capture reference gates fail.

The six captured inner matrices and normalization denominators pass cross-solver
tolerance checks but are not bitwise equal. Those are values from altered,
instrumented computations; they cannot establish the values inside the untouched
full updates. The untouched matrix-versus-original comparison again fails all
seven directions while passing all 21 state leaves. Collection success does not
accept an optimization or explain the full numerical discrepancy.

Online W&B finished with `attribution_completed=1` and
`candidate_accepted_for_training=false`. All 14 artifact objects are present with
matching reported sizes, totaling 58,908,027 compressed bytes. The two small
Jaxpr files passed SHA256 and decompression readback; the twelve compiler files
remain regional and have not received a new checksum readback audit. Finelog
recorded 370 samples through 08:45:01 UTC and an observed host peak of
106.399 GiB. The JAX allocator peak was 8,218,514,432 bytes per device, which is
not total HBM. These calls do not measure warmed performance or training MFU.

The log observer ended naturally. No experiment or queue remains active. The
next step is static comparison of the six saved compiler graphs: untouched,
unchanged evaluator and capture, for each solver. Compare changed and unchanged
direction groups using output dependencies, preserving tuple-leaf associations;
keep large files in us-west4 and use CPU extraction if needed. Static differences
will not establish runtime values or cause. No new numerical submission,
precision override or tiling intervention is selected.

## 2026-11-03 09:01 UTC — Static capture-graph extraction validated locally

A [CPU-only extraction plan](compiler-capture-inventory-plan.json) will compare
six saved optimized-HLO files from the completed capture job. It retains the
existing 15-product chain check for each of seven direction groups, checks the
28 versus 34 output-leaf counts, and records normalization divisions, nested
producer bodies, collectives and enclosing-fusion configurations. Missing
normalization-to-first-product scope matches remain explicit.

Local parser fixtures built from a saved TPU query-polar graph passed both output
layouts, reversed direction-output ordering, repeated nonpolar operands and
wrong-output-count rejection. Each fixture maps 105 products and preserves all
21 compiled normalization divisions. The full file-processing CLI passed on
six fixture files. Non-JSON CPU backend attributes are now retained as opaque
text; JSON configurations keep their previous interpretation. The tiny CPU
optimizer graph lowers to only five tagged products per group and was rejected
by the unchanged full-size chain gate, so it is not used as a full-size parser
validation fixture. No numerical kernels were executed for this preflight.

Publish this source and plan before one us-west4 CPU extraction, with default
preemption retries and no wall-clock limit. Large compiler files stay regional;
the extraction will verify their hashes while reading them. Static records
cannot establish hidden original values, runtime precision or numerical cause.
No accelerator or numerical experiment is selected.

## 2026-11-03 09:07 UTC — Capture compiler records mapped; cause unresolved

The [regional CPU extraction](compiler-capture-inventory-completed.json)
finished on attempt 0 with task exit 0. It verified all six optimized-HLO hashes
in us-west4 and mapped 105 polar products in seven disjoint 15-product chains
per graph. Output counts are 28 for untouched/evaluator graphs and 34 for
capture graphs. Large HLO files stayed regional; the extracted record is
11,914,404 bytes. No optimizer kernels or training ran.

The [product comparison](compiler-capture-comparison.json) matches untouched
and unchanged-evaluator records at all 105 positions for both solvers in the
inspected fields: result and operand declarations, immediate operand opcodes,
operation attributes/backend configuration and immediate enclosing-fusion
configuration. This agrees with their bitwise-equal final outputs, but does not
establish internal numerical identity.

Original capture changes enclosing-fusion configurations at 13 product positions
in q/k/v. Two v_proj product result declarations differ only by an `S(1)` tag,
whose meaning is not interpreted here.
Matrix capture changes enclosing-fusion configurations at 16 positions across
six groups. The other inspected product fields match, aside from those two
original-branch result declarations. This comparison does not establish an
explanation for all failed final directions: the original capture's o_proj and up_proj directions
failed numerically despite matching all inspected product fields in those groups.
Upstream values and connections were not included in that equality claim.

Each graph retains 13 normalization division records. Following immediate
operand-producing fusions maps both first-product operands to a normalization
division for q/k/v, preserving repeated operand positions. The remaining four
groups are unresolved at this boundary; an empty direct-scope match does not
mean normalization is absent. The plan records fixture ordering/count checks;
the [comparison validation](compiler-capture-comparison-validation.json) records
a targeted operand-shape mutation and the q/k/v operand mapping checks.

Next, trace the remaining normalization producers and caller inputs using the
captured bodies and entry records, listing any missing bodies explicitly. Any
needed extraction stays on regional CPU. Both numerical capture gates remain
failed, so their intermediates cannot establish untouched hidden values. No
precision/tiling intervention or new numerical submission is selected. The CPU
log observer ended naturally; no experiment or queue remains active.

## 2026-11-03 09:18 UTC — Recover missing first-product caller bindings

The saved compiler inventory lacks the enclosing computation bodies for all
42 first polar products across six graphs. Without those bodies, the parser
cannot bind each nested parameter to its caller input. A [targeted extraction](compiler-capture-bindings-plan.json)
will collect those named bodies and nested callees on CPU in us-west4, verifying
the six optimized-HLO hashes. Large HLO files remain regional.

The new tracer preserves tuple-leaf selections and repeated operand positions,
follows explicit fusion parameter bindings, and stops at normalization divisions
or named opaque boundaries. Local fixtures passed nested binding, tuple selection,
combined all-reduce independence, normalization stopping and explicit missing-body
reporting. The file-processing CLI passed on six fixtures. No optimizer kernels ran.

Publish the source and plan before one CPU extraction with default preemption
retries and no wall-clock limit. Both numerical capture checks against their
untouched references remain failed; these static paths cannot establish hidden
original values or a numerical cause.

## 2026-11-03 09:22 UTC — Caller bindings extend normalization mapping

The [targeted CPU extraction](compiler-capture-bindings-completed.json) finished
on attempt 0 with exit 0. It verified all six optimized-HLO hashes in us-west4
and captured 18 named or nested computation bodies per graph. Large HLO stayed
regional; the selected-body report is 156,417 bytes.

[Following fusion parameter bindings](compiler-capture-normalization-bindings.json)
now reaches a normalization division from 46 of 84 first-product operand
positions across the six graphs. This includes all 36 q/k/v positions, one
o_proj position in every graph plus the second position in matrix capture,
and one gate_proj position in each matrix graph. Repeated operands remain separate
positions. Seventeen positions stop at named missing computation bodies and
21 stop at ConcatBitcast custom calls. These are parser-coverage counts, not
evidence that normalization is absent or that custom-call runtime semantics
are known.

[Binding fixtures](compiler-capture-normalization-bindings-validation.json)
passed tuple selection and combined-collective independence. An additional
end-to-end tracer probe on the older synthetic seven-subgraph fixture rejected
its nested callers because the tracer currently requires first-product callers
in the entry computation. No assertion was relaxed. All six actual graphs
passed that structural check and the reported coverage checks, retaining the
unresolved positions above. Their q/k/v mappings agree with the earlier analysis.

Next, inspect the recorded slice producers behind the opaque boundaries and
recover the explicitly listed missing bodies if needed. Both numerical captures
still fail their untouched-reference checks. The static paths do not establish
hidden original values or a numerical cause. No numerical experiment or
optimizer intervention is selected. The CPU log observer ended naturally;
no experiment or queue remains active.

## 2026-11-03 09:29 UTC — Trace declared slice inputs behind opaque calls

All 21 unresolved ConcatBitcast boundaries consume four slice-done results.
Each recorded slice-done references a slice-start with one declared source.
Following those declared sources conservatively reaches normalization divisions;
the custom calls remain marked opaque, and no runtime semantics are inferred.
The trace preserves each custom-call operand position and its slice attributes.

Local fixtures passed independent slice inputs, repeated operand contexts,
opaque-boundary retention and rejection of an invalid slice-done association.
The optional traversal leaves the default opaque-stop behavior unchanged.
The remaining 17 missing computation bodies are listed in the
[targeted CPU plan](compiler-capture-slices-plan.json). Publish this source and
plan before extracting those bodies and nested callees in us-west4, with default
preemption retries and no wall-clock limit. No accelerator or numerical
experiment is selected; both capture-reference checks remain failed.

## 2026-11-03 09:34 UTC — Preserve opaque collective boundaries during extraction

The [CPU slice-body extraction](compiler-capture-slices-completed.json) succeeded
on attempt 0 with exit 0 and verified all six optimized-HLO hashes regionally.
The 17 recovered bodies terminate at AsyncCollectiveDone custom calls, each
with one explicit all-gather operand. Tracing that operand separately exposes
17 further missing enclosing bodies. The opaque result and its other arguments
remain recorded; the all-gather branch is not assumed to determine that result.

The [next CPU extractor](compiler-normalization-closure-plan.json) follows
missing-body references iteratively until no requested record is absent or the
tracer stops at an explicit opaque boundary. A local fixture recovered two
missing layers and preserved repeated product operands; its file-processing
CLI passed. Slice fixtures and auxiliary all-gather fixtures passed, and the
previous default mapping retained all 84 operand-position results.
Publish this source and plan before the regional CPU extraction. No accelerator
or numerical experiment is selected, and both numerical capture gates remain
failed.

## 2026-11-03 09:39 UTC — Record gaps resolved; opaque semantics remain

The [iterative regional CPU extraction](compiler-normalization-closure-completed.json)
succeeded on attempt 0 with exit 0 and verified all six optimized-HLO hashes.
Three extraction rounds per graph recovered every body requested by the tracer.
The 1,246,589-byte report retains the remaining opaque boundaries.

Normalization divisions are reached at 67 of 84 first-product operand positions.
This includes 21 paths that conservatively follow declared ConcatBitcast inputs.
The other 17 positions end at AsyncCollectiveDone; their separately traced
all-gather inputs end at opaque AsyncCollectiveStart results. No missing
computation record remains, but semantic provenance is incomplete.

For each solver, untouched-versus-evaluator and untouched-versus-capture
[comparisons](compiler-normalization-closure-comparison.json) share 11 mapped
operand positions. Their ordered normalization result declarations, opcodes and
operation attributes preceding the metadata field in the HLO instruction text
match. The comparison excludes unresolved
endpoints and does not compare operand values, caller bindings or backend
configurations. A [targeted declaration mutation](compiler-normalization-closure-validation.json)
was detected; removing an endpoint excluded it from equality rather than treating
it as a match.

The prior six-output captures failed their reference checks while unchanged
evaluators matched bitwise. The next design tests whether appending only the
existing q_proj normalization denominator preserves all 28 outputs against each
solver variant’s own untouched reference.
This would test instrumentation sensitivity; no implementation, numerical
submission or optimizer intervention is selected. It requires its own online
CPU validation and published source before one sequential TPU diagnostic.
Matching final outputs would still not establish hidden intermediate identity.
The CPU log observer ended naturally; no experiment or queue remains active.

## 2026-11-03 09:44 UTC — Reduce capture to one existing denominator

The [single-denominator capture](optimizer-single-norm-capture-plan.json) appends
only the existing q_proj broadcast norm-plus-epsilon variable. The shared Jaxpr
helper retains every equation, input/constant variable and all 28 original
outputs; the capture has 29 output leaves. The source equations and precision
settings are unchanged.

Each solver keeps its untouched full-update reference, unchanged evaluator and
capture comparisons at rtol=atol=1e-5, with bitwise checks, fixed-executable
replays and input fingerprints. Failed reference agreement forbids interpreting
the captured denominator as an untouched internal value. Even matching final
outputs would not prove internal identity. This is an instrumentation control;
no optimization or performance claim is made.

Publish the implementation and plan before fresh online CPU logging and numerical
checks for the new mode and the existing six-output mode. A TPU submission
requires those checks and artifact readbacks to pass. Memory controls, full
matrix geometry and default preemption retries remain unchanged; no runtime
limit is selected.

## 2026-11-03 09:49 UTC — Single-output CPU preflight

The [CPU preflight](optimizer-single-norm-capture-preflight.json) passed on eight
virtual CPU devices. The new mode appends only the existing q_proj normalization
denominator. It completed 12 finite stages and all
13 comparisons; the existing six-output regression completed 12 finite stages
and all 18 comparisons. Each mode replayed four fixed executables bitwise, and
both evaluators and captures matched their own untouched 28-leaf references
bitwise. The new mode retained 29 output leaves; the regression retained 34.

Fresh online W&B runs finished for both modes and the separate logging check.
Both reports matched their GCS bytes, and all 28 compiler/Jaxpr objects passed
SHA, size and gzip readback checks. These small-matrix CPU checks do not establish
TPU behavior or hidden intermediate identity. Publish this evidence before one
sequential TPU diagnostic using the validated source and unchanged tolerances.

## 2026-11-03 09:50 UTC — Single-denominator TPU diagnostic started

The [single-output diagnostic](optimizer-single-norm-capture-started.json) was
accepted at 09:49:07 UTC from published revision
`95ae974e58e1c3f83c79574c57f89ee19bb3b9e6`. Attempt 0 is running on eight native
v5e chips in one JAX process. The loaded script and helper hashes match the CPU
validation. The fresh [online W&B run](https://wandb.ai/marin-community/marin/runs/ema-gram-20-attribution-8854b678361e)
passed its logging check before optimizer kernels.

This run exposes only the existing q_proj normalization denominator. All
28-leaf reference comparisons remain pending at startup. Failed agreement will
forbid interpreting the captured denominator as an untouched internal value; even
matching final outputs would not establish internal identity. No optimizer
acceptance or performance claim is made.

The task requests 96 CPUs, 128 GiB host memory and 50 GiB disk in us-west4, with
batch priority, default preemption retries and no runtime limit. Sole log
observer 85107 follows this task; PR observer 77099 remains live. Preserve the
job through compilation and collect both solver branches before selecting any
further diagnostic. Large compiler artifacts stay regional.

## 2026-11-03 09:58 UTC — Original evaluator replay passed

The [single-denominator diagnostic](optimizer-single-norm-capture-partial.json)
remains running on attempt 0. Five stages are finite, and the unchanged original
Jaxpr evaluator replayed bitwise on the same fingerprinted inputs. The run is
compiling the original solver's extra-output capture. Its all-28-leaf reference
checks and the matrix branch remain pending.

Schema-confirmed Finelog telemetry contains 72 host-memory samples through
09:57:05 UTC, with an observed cumulative peak of 58.13 GiB against the 128 GiB
reservation. Accelerator memory is null in these rows, so they do not measure
HBM. The original capture Jaxpr preserves 851 equations and has 29 output leaves;
its small artifact passed SHA, size and gzip readback. Large compiler artifacts
remain regional and have no new SHA readback claim here.

Preserve the job and sole log observer through compilation. No new diagnostic,
optimizer acceptance or warmed-performance claim follows from this partial
snapshot.

## 2026-11-03 10:15 UTC — Single-denominator capture preserves both references

The [single-output diagnostic](optimizer-single-norm-capture-v5e-completed.json)
succeeded on attempt 0 with exit 0. All 12 stages are finite; 11 of 13 comparisons
pass at rtol=atol=1e-5. Both unchanged evaluators and both single-output captures
match their own untouched full-update references on all 28 leaves, bitwise.
All four fixed executables replay bitwise. This contrasts with the earlier
six-output captures, which failed their own reference gates.

The captured q_proj normalization denominators pass tolerance across solvers but
are not bitwise equal: maximum absolute error is 2.91038e-10 and relative
Frobenius error is 2.65090e-7, relative to the original solver's captured value.
The matrix-versus-original comparisons of untouched outputs and captured full
outputs each still fail all seven directions while all 21 state leaves pass. Passing capture gates does not establish hidden
intermediate identity or show how much the denominator difference contributes
to the final optimizer discrepancy. No optimization is accepted.

The prior six-output report hash and current report hash were verified before
cross-job comparison. Input-bank fingerprints and both untouched full-update
output fingerprints match across those jobs. The 14 compiler/Jaxpr objects total
58,898,640 compressed bytes, with all sizes verified. Both small Jaxpr files
passed SHA and gzip readback; the 12 large compiler files remain regional and
were not newly SHA-audited. Online W&B finished with collection completed and
candidate acceptance false.

Finelog observed a 99.609375 GiB cumulative host peak over 215 samples through
10:12:52 UTC. The JAX allocator peak is 8,076,040,192 bytes per device; neither
this allocator value nor null accelerator telemetry is a total-HBM measurement.
Single-call timings include compilation and support no warmed-performance or
training-MFU claim. The log observer ended naturally; no experiment or queue
remains active.

The next design exposes only the existing q_proj pre-normalization inner matrix,
using the same Jaxpr helper and all-28-leaf reference gates. This asks whether a
single matrix output also preserves each reference and permits a separate
comparison of captured numerators. It does not combine values from separate
executables into a claim about one hidden path. The design requires fresh online
CPU validation and published exact source before one sequential TPU diagnostic;
no implementation or submission is selected yet.

## 2026-11-03 10:28 UTC — Capture one existing inner matrix

The [single-inner capture](optimizer-single-inner-capture-plan.json) appends only
the existing q_proj pre-normalization inner matrix to the full-update Jaxpr.
It preserves all equations, input/constant variables and 28 original outputs,
with the same source precision and sharding settings. The extra matrix output
is replicated, as in the earlier six-output capture; it adds no source-level arithmetic.

Both solver variants retain untouched full-update references, unchanged
Jaxpr evaluators and fixed-executable capture replays. All 28 reference leaves
are compared at rtol=atol=1e-5 with bitwise checks. Failed agreement forbids
original-path interpretation, and matching final outputs would not establish
hidden intermediate identity. Values from separate numerator and denominator
capture executables cannot be treated as one hidden path.

Publish this implementation and plan before fresh online CPU logging, the new
mode's numerical check and a regression of the single-denominator mode. Only
validated, published source may proceed to one sequential TPU diagnostic.
No optimizer intervention or performance claim is selected.

## 2026-11-03 10:31 UTC — Single-inner CPU preflight

The [online CPU preflight](optimizer-single-inner-capture-preflight.json) passed
for the new mode, which captures only the existing q_proj pre-normalization
inner matrix, and the single-denominator regression. Each mode
completed 12 finite stages, all 13 comparisons and four bitwise fixed-executable
replays. Both evaluators and captures matched their own untouched 28-leaf
references bitwise. Each capture retained 29 outputs. Compiler output signatures confirm the new
mode appends a matrix and the regression appends a denominator.

Fresh online W&B runs finished for both numerical checks and the separate
logging check. Both reports matched their GCS bytes; all 28 compiler/Jaxpr
objects passed SHA, size and gzip readback. These eight-virtual-device CPU checks
validate the harness and regression, not TPU behavior. Publish this evidence
before one sequential TPU diagnostic using unchanged source equations, precision
settings and rtol=atol=1e-5.

## 2026-11-03 10:32 UTC — Single-inner TPU diagnostic started

The [single-inner diagnostic](optimizer-single-inner-capture-started.json) was
accepted at 10:31:14 UTC from published revision
`9f8bb3320ceebedc2a999e19d9370f89d90aa307`. Attempt 0 is running on eight native
v5e chips in one JAX process. Loaded script/helper hashes match the validated
source. The fresh [online W&B run](https://wandb.ai/marin-community/marin/runs/ema-gram-20-attribution-ee0ff400e3c3)
passed its logging check before optimizer kernels.

This diagnostic appends the existing q_proj pre-normalization inner matrix.
All 28-leaf reference gates remain pending at startup. Failed agreement forbids
original-path interpretation; matching final outputs would not prove hidden
intermediate identity. Separate inner and denominator capture executables cannot
be treated as a single hidden path. No optimizer or performance claim is made.

The task requests 96 CPUs, 128 GiB host memory and 50 GiB disk in us-west4, with
batch priority, default preemption retries and no runtime limit. Sole log
observer 54894 follows this task; PR observer 77099 remains live. Preserve the
job through compilation and assess both solver branches before another numerical
candidate. Large compiler artifacts remain regional.

## 2026-11-03 10:45 UTC — Original single-inner reference gate passes

The [single-inner diagnostic](optimizer-single-inner-capture-partial.json)
remains running on attempt 0 and has entered the matrix branch. Seven stages
are finite and all five comparisons recorded so far pass at rtol=atol=1e-5.
The unchanged original Jaxpr evaluator matches its untouched full-update
reference bitwise. The single-inner capture passes all 28 reference leaves but
is not bitwise identical: only v_proj has nonzero measured error, with maximum
absolute error 8.94070e-8 and relative Frobenius error 1.06813e-7. The other
27 leaves have zero measured absolute error. Both original fixed executables
replay bitwise.

Schema-confirmed Finelog telemetry records 96 host-memory samples through
10:41:50 UTC, with an observed cumulative peak of 68.33 GiB against the 128 GiB
reservation. Accelerator-memory telemetry is null and does not measure HBM.
The small original Jaxpr artifact passed SHA, size and gzip readback and retains
851 equations and 29 output leaves. Large compiler files stay regional without
a new SHA readback claim.

Preserve the job and sole log observer while the matrix branch completes.
Passing the original reference gate does not establish hidden intermediate
identity. Matrix reference gates, final artifacts and terminal status remain
pending; no optimizer acceptance or warmed-performance claim is made.

## 2026-11-03 10:58 UTC — Single-inner capture passes both reference gates

The single-inner diagnostic completed successfully on attempt 0. It adds only
the existing q_proj pre-normalization matrix to each full optimizer update's
outputs. Both solver variants pass all 28 output-leaf checks against their own
untouched references at rtol=atol=1e-5. Both unchanged Jaxpr evaluators match their
references bitwise; neither extra-output capture does. The original capture has
nonzero measured error only in v_proj (maximum absolute error 8.94070e-8).
The matrix-solver capture has nonzero measured error in q_proj, k_proj and
v_proj (maximum 1.04308e-7). Other leaves have zero measured absolute error,
which does not establish per-leaf bitwise equality.

All 12 stages are finite, 11 of 13 comparisons pass, and all four fixed
executables replay bitwise on identical fingerprinted inputs. The captured
q_proj inner matrices pass tolerance across solvers but differ bitwise:
maximum absolute error 1.55183e-11 and relative Frobenius error 5.79009e-6,
using the original solver's captured matrix as denominator. The two failing
comparisons compare matrix versus original full updates, with and without
capture; each fails all seven direction leaves while all 21 state leaves pass.
These results do not accept an optimizer change or establish the mismatch's cause.

All 14 compiler/Jaxpr object sizes are verified (58,898,587 compressed bytes).
The two small Jaxpr files passed SHA and gzip readback; the 12 large compiler
files remain in us-west4 without a new SHA audit. Observed host memory peaked
at 94.10 GiB across 202 Finelog samples through 10:53:32 UTC. The JAX allocator
peak of 8,122,035,712 bytes per device is not total HBM. No warmed-performance
or training-MFU claim is made.

The next control is design-only: append q_proj's existing inner matrix and
norm-plus-epsilon denominator together, retaining every source equation and
all 28 original outputs. Separate single-output captures cannot establish
values along one hidden computation. A paired capture still needs its own
online CPU validation, published source and all-leaf reference gates before
interpretation. Matching final outputs would not prove hidden intermediate
identity. No new numerical job has been submitted.

Completed evidence: [single-inner capture](optimizer-single-inner-capture-v5e-completed.json).
Source `9f8bb3320ceebedc2a999e19d9370f89d90aa307`; canonical success was
verified at 10:57:01 UTC. The [online W&B run](https://wandb.ai/marin-community/marin/runs/ema-gram-20-attribution-ee0ff400e3c3)
finished with collection complete and candidate acceptance false. Log observer
54894 ended naturally with exit 0; no experiment or queue remains. PR observer
77099 remains the sole PR observer.

## 2026-11-03 11:13 UTC — Capture q_proj normalization operands together

The paired-input capture mode appends q_proj's existing pre-normalization
inner matrix and norm-plus-epsilon denominator to the full update's outputs.
It preserves all source equations, inputs, constants and 28 original outputs,
yielding 30 output leaves. The two added values come from the same instrumented
executable. Separate single-value captures previously passed their own
reference gates, but do not establish values from one hidden computation.

For each solver, the harness compares the untouched update, unchanged Jaxpr
evaluator and paired capture on identical fingerprinted inputs. Each evaluator
and capture is compiled once and replayed. All 28 reference leaves use fixed
rtol=atol=1e-5 plus bitwise checks. Each cross-solver captured-value comparison
records both reference gates. Failed agreement forbids original-path
interpretation; matching final outputs would not prove hidden internal identity
or explain the complete optimizer discrepancy.

Validation is pending: a fresh online logging check, a paired CPU run, and
single-inner and single-denominator regressions. The expected paired result is
12 finite stages, 14 comparisons, four fixed executable replays and 14 compiler
or Jaxpr artifacts. Validation will check output order and shapes as well as
artifact checksums. No numerical job has been submitted to Iris. One sequential
TPU diagnostic is planned only after this validation and source publication.

The [paired-input plan](optimizer-paired-input-capture-plan.json) preserves
full geometry, source precision, one executable/device bank at a time, host
offload and cache clearing. No precision, tiling or optimizer change is selected.

## 2026-11-03 11:17 UTC — Paired capture passes CPU validation

The paired q_proj capture passed its online CPU harness: 12 finite stages,
14 passing comparisons and four bitwise fixed-executable replays. Both solver
captures and unchanged evaluators match their own untouched 28-leaf references
bitwise. The single-inner and single-denominator regressions each passed
12 finite stages, 13 comparisons and four bitwise replays from the same source.
All recorded cross-solver captured-value comparisons are bitwise equal on CPU.

Compiler signatures verify 30 output leaves for the paired mode, with the inner
matrix followed by its broadcast denominator; the regression modes retain
29 leaves. All reports match their GCS readbacks, and all 42 compiler/Jaxpr
artifacts across the three runs passed SHA, size and gzip checks. A separate
fresh online logging check also passed. These checks use eight virtual CPU
devices and small matrices and do not establish TPU behavior.

The next diagnostic uses eight native v5e chips in one JAX process, the original
full geometry, source precision and fixed rtol=atol=1e-5. It retains both
untouched references and all-leaf instrumentation gates. Collection will
continue through expected TPU comparison failures; it cannot accept an
optimizer change. No precision or tiling override is introduced. No optimizer
acceptance, performance improvement or training result is claimed.

CPU source `a72a212baf0d7da978678c12b175635434520460`. Online runs: paired
`5fe1e8e63a90`, single-inner `76fb3a7a5d53`, single-denominator `99c8f79b2921`;
separate logging `f5964be75819`. All finished. The [preflight record](optimizer-paired-input-capture-preflight.json)
contains source hashes, exact report references and artifact checks.

## 2026-11-03 11:20 UTC — Paired capture starts on v5e

The paired q_proj capture diagnostic is running on eight native v5e chips in
one JAX process. Iris accepted it at 11:17:30 UTC from published revision
`ae5ca330717fad85f6e764a6b9f085329cc04fab`. Loaded script and helper hashes match
the CPU-validated source. Its fresh online W&B run passed the logging check
before optimizer kernels.

The mode appends only the existing q_proj inner matrix and norm-plus-epsilon
denominator, retaining all source equations and 28 original output leaves.
The 30-leaf capture is compared against each solver's untouched full update
and unchanged evaluator at rtol=atol=1e-5, with bitwise checks and fixed-executable
replays. All TPU reference gates remain pending. Failed agreement forbids
original-path interpretation; matching outputs would not establish hidden
intermediate identity or the complete optimizer discrepancy's cause.

The task requests 96 CPUs, 128 GiB host memory and 50 GiB disk in us-west4,
with batch priority, default preemption retries and no runtime limit. Preserve
this job through compilation and assess both solver branches before another
numerical candidate. Sole log observer 39235 follows the task; PR observer
77099 remains the sole PR observer. Large compiler files stay regional.

[Startup evidence](optimizer-paired-input-capture-started.json) records attempt UID
`4dbb9dabf624` and fresh [online W&B run](https://wandb.ai/marin-community/marin/runs/ema-gram-20-attribution-cfb9fbb20ed7).
Three stages are finite at startup: zero initialization, initialization and the
untouched original full update. Numerical capture gates remain pending.

## 2026-11-03 11:31 UTC — Paired capture passes the original reference gate

The paired capture exposes q_proj's pre-normalization inner matrix and
norm-plus-epsilon denominator. It passes the original solver's 28-leaf
reference gate at rtol=atol=1e-5. It is not bitwise identical to the untouched full update:
v_proj alone has nonzero measured error, with maximum absolute error
8.94070e-8 and relative Frobenius error 1.06813e-7. The other 27 leaves have
zero measured absolute error, which does not establish per-leaf bitwise equality.
The unchanged Jaxpr evaluator matches the untouched reference bitwise, and both
original fixed executables replay bitwise.

Seven stages are finite and all five recorded comparisons pass. The job remains
running on attempt 0 and has entered the matrix-solver branch. Its reference
gate and the cross-solver paired-value comparisons remain pending. Passing the
original gate does not establish hidden intermediate identity or explain the
full optimizer discrepancy; no optimization or performance claim is made.

Schema-confirmed Finelog telemetry records 92 samples through 11:27:41 UTC,
with an observed cumulative host-memory peak of 67.71 GiB against the 128 GiB
reservation. Accelerator-memory telemetry is null. The small original Jaxpr
artifact passed SHA, size and gzip readback; capture metadata records
851 preserved equations and 30 output leaves. Large compiler files remain in
us-west4 without a new checksum readback. Preserve the existing job and sole
log observer through the matrix branch; no concurrent candidate is selected.

[Partial evidence](optimizer-paired-input-capture-partial.json) records the
report snapshot and canonical running state verified at 11:29:31 UTC. Sole
experiment log observer 39235 and PR observer 77099 remain active.

## 2026-11-03 11:46 UTC — Paired capture completes with both reference gates passing

The paired q_proj capture completed successfully on attempt 0. Both solvers'
30-leaf captures pass their own untouched full-update references on all 28
original leaves at rtol=atol=1e-5, but neither is bitwise identical. The original
capture has nonzero measured error only in v_proj (maximum absolute error
8.94070e-8). The matrix capture has nonzero measured error in q_proj, k_proj and
v_proj (at most 1.04308e-7). Zero measured error on the remaining leaves does not
establish per-leaf bitwise equality. Both unchanged evaluators match their own
untouched references bitwise, and all four fixed executables replay bitwise.

The numerator and denominator now come from the same capture executable for
each solver. Both cross-solver value comparisons pass tolerance but are not
bitwise equal: numerator maximum absolute error is 1.55183e-11 and relative
Frobenius error is 5.79009e-6; denominator errors are 2.91038e-10 and 2.65090e-7.
Each relative error uses the corresponding original-solver captured value as
its reference. These measurements do not establish untouched internal identity
or either operand's causal contribution to the optimizer discrepancy.

All 12 stages are finite and 12 of 14 comparisons pass. The two failures compare
matrix against original full updates, once untouched and once captured; each
fails all seven directions while all 21 state leaves pass. No optimization,
warmed performance, training MFU or checkpoint-resume result is accepted.

Canonical job and task status were verified terminal SUCCEEDED, exit 0, at
11:41:39 UTC. Online W&B finished with attribution completed and candidate
acceptance false. All 14 artifact sizes were verified (58,899,076 compressed
bytes), as were SHA and gzip readbacks for the two small Jaxpr files. The 12
large compiler files remain in us-west4 without a new SHA audit. Finelog records
206 samples through 11:40:17 UTC and an observed host-memory peak of 98.56 GiB.
Accelerator-memory telemetry is null; the JAX allocator peak of 8,122,036,224
bytes per device is not total HBM. The sole experiment log ended naturally with
exit 0; no experiment or queue remains. PR observer 77099 remains the sole live
PR observer.

The next step is design review of a conditional full-update substitution
control. It would replace only the operand uses at q_proj's existing
normalization division, retaining the outer Gram-power multiplications and
all other equations. For each solver, one fixed executable would test the four
original/matrix numerator-denominator combinations. Each substitution executable, supplied with its own solver's captured pair,
must first match all 28 untouched reference outputs at rtol=atol=1e-5; failed
agreement forbids original-path interpretation. Fresh paired captures, online CPU validation,
operand-use preservation checks and published exact source are required before
one sequential TPU diagnostic. This control is not implemented or submitted.

[Completed evidence](optimizer-paired-input-capture-v5e-completed.json) records
report SHA `499b25a92d4a2cd43d9f10e3f3dcfe79b8278bb9374ba70f8074e1e417d6c993`.
The [attribution plan](optimizer-divergence-attribution-plan.json) records the
proposed control and its interpretation gates.

## 2026-11-03 12:00 UTC — Implement a division-operand substitution control

The new diagnostic replaces only the two operand uses at q_proj's existing
normalization division with external numerator and denominator inputs. Every
other traced equation, the target division primitive and its parameters,
original inputs, constants and 28 outputs are retained. Other uses of the
numerator and denominator variables are unchanged. Each solver has one fixed
substitution executable for all four combinations of original/matrix solver
numerators and denominators. Its own-pair replay uses both operands from that
solver's capture.

Fresh paired captures retain untouched and unchanged-evaluator references.
Before interpreting substitutions, both capture gates and the substitution
executable's own-pair comparisons against its untouched and captured 28-leaf
references must pass at rtol=atol=1e-5. Bitwise status is recorded separately.
Failed gates remain evidence and forbid original-path interpretation. Passing
gates does not establish hidden internal identity or the complete cause.

The [plan](optimizer-paired-input-substitution-plan.json) specifies 22 finite
stages, 26 comparisons, six fixed executables and bounded device residency.
Online CPU logging, numerical fixtures and the paired-capture regression are
pending. No TPU job has been submitted. Source precision, full geometry,
default preemption retries and all training gates remain unchanged.

## 2026-11-03 12:03 UTC — Substitution CPU validation passes

The new substitution harness completed 22 finite stages and all 26 comparisons
passed at fixed rtol=atol=1e-5. All six fixed executables replayed bitwise. Both
fresh paired captures passed their reference gates, and each substitution
executable supplied with its own solver's captured pair passed all 28 untouched
and captured reference leaves at1e-5, but not bitwise. Both solvers show nonzero
measured error in q_proj direction (maxabs9.53674e-7, relative Frobenius4.80648e-7)
and its momentum state (maxabs2.91038e-11). The four bank combinations are
bitwise equivalent within each fixed CPU executable; that does not predict
TPU sensitivity. Own-pair substitution changes CPU outputs within tolerance;
the source of that difference remains unresolved.

The targeted fixture used distinct replacement values at a division whose two
operands originally shared one variable. It verified the replaced division's
result and an unchanged additional use of that variable. Invalid targets and
missing or ambiguous normalization mappings were rejected. Actual full-update
program assertions preserve all non-target equations and the original outputs,
inputs and constants. The paired-capture regression passed 12 finite stages,
14 comparisons and four bitwise replays, with all recorded bitwise checks equal.

Both numerical reports matched their GCS copies. All 34 compiler/Jaxpr artifacts
passed SHA, size and gzip readback: 887,003 compressed bytes for substitution
and 660,926 for regression. Substitution StableHLO signatures retain 28 output
leaves and include both external operand shapes. A separate fresh online
logging check also passed. Every process used its own online W&B identity.
These small-matrix, eight-virtual-device CPU results do not establish TPU
behavior or untouched internal identity.

Source `381e3a697009e750843ce5615a30e205a882a328`; numerical runs
`f277f8c932fc` and `1eb7ab54998f`, logging run `faf92ad2b105`.
The [preflight](optimizer-paired-input-substitution-preflight.json) records
exact reports and source hashes. One sequential full-size v5e diagnostic is
ready under the published plan, preserving all numerical gates and resource
constraints. No optimizer or training result is accepted.

## 2026-11-03 12:07 UTC — Full-update substitution starts on v5e

Iris accepted the single substitution diagnostic at 12:04:36 UTC from published
revision `206cf9561b6f4c974f46b03154b8615e97b7a5e0`. Job and task are running on
attempt 0, UID `b0627618bb81`, verified at 12:06 UTC. The loaded script and
helper hashes match the CPU-validated source. Fresh online W&B run
`ema-gram-20-attribution-e6b9c35d4798` reports logging success before kernels.
Three stages are finite: zero initialization, initialization and the untouched
original full update. All TPU capture and substitution reference gates remain
pending; single calls include compilation and are not warmed performance.

The task uses eight native v5e chips in one JAX process, with 96 CPUs, 128 GiB
host memory and 50 GiB disk in us-west4. Batch priority, default preemption
retries and no runtime limit are retained. Preserve this job through compilation
and assess both solver branches before any further numerical candidate.
Sole log observer 86562 follows the task; PR observer 77099 remains the sole
PR observer. Large compiler files remain regional.

The control substitutes q_proj's normalization numerator and denominator;
"own-pair" means both values captured from that same solver.
The [startup record](optimizer-paired-input-substitution-started.json) and
[preflight](optimizer-paired-input-substitution-preflight.json) retain the exact
source, run identities and checks. Both fresh capture gates and each
substitution executable's own-pair 28-leaf reference gates must pass fixed1e-5
before original-path interpretation. Bitwise status is separate; even passing
gates cannot establish hidden internal identity or the full discrepancy's cause.
No optimizer, training MFU or checkpoint-resume result is claimed.

## 2026-11-03 12:13 UTC — Substitution diagnostic passes its original capture gate

The fresh report at 12:11:47 UTC contains eight finite stages and five passing
comparisons. The original solver's unchanged evaluator matches its untouched
28-leaf reference bitwise. Its paired capture passes all 28 leaves at
rtol=atol=1e-5 but is not bitwise equal: only v_proj has nonzero measured error,
with maximum absolute error 8.94070e-8 and relative Frobenius error 1.06813e-7. Zero measured error on
the other leaves does not prove per-leaf bitwise equality. Both original fixed
executables replay bitwise.

The untouched matrix full update is also finite, and the current report stage
is matrix evaluator compilation. Its capture gate and both substitution
executables' own-pair reference gates remain pending. Passing the original
capture gate does not validate substitution or establish hidden internal
identity. Preserve the current job and sole log observer; no further numerical
candidate is selected.

Finelog records 69 samples through 12:12:24 UTC and observed cumulative
host peak 58.98 GiB against the 128 GiB reservation. Accelerator memory is
null. The small original Jaxpr passed SHA, size and gzip readback; capture
metadata retains 851 equations and 30 outputs. Large compiler files remain regional
without a new checksum audit. [Partial evidence](optimizer-paired-input-substitution-partial.json)
separates this fresh report from the last canonical RUNNING observation at
12:06 UTC. Sole log observer 86562 and PR observer 77099 were both live at 12:11 UTC. No optimization,
warmed performance, MFU or checkpoint-resume claim is made.

## 2026-11-03 12:28 UTC — Original substitution fails its same-pair reference gate

The original substitution executable, supplied with its own solver's freshly
captured numerator and denominator, fails the untouched 28-leaf reference gate
at rtol=atol=1e-5. All seven direction leaves fail; all 21 state leaves pass.
Direction relative Frobenius errors range from 0.006139 to 0.026861, relative to
the corresponding untouched outputs. The comparison against the paired-capture
outputs also fails all seven directions. Both fresh paired captures had passed
their reference gates, so capture agreement does not validate this substitution
executable. The failed same-pair gate forbids original-path interpretation.

Within that fixed altered executable, numerator-only, denominator-only and
paired cross-solver substitutions change q_proj by relative Frobenius errors
0.00700109, 0.00610410 and 0.00700086, respectively. Each uses the corresponding
own-pair output leaf as reference. The other 27 leaves have zero measured
absolute error, which does not prove per-leaf bitwise equality. These are
conditional differences within the altered executable; they do not establish
the cause of the untouched optimizer discrepancy. Its own-pair replay is bitwise.

The 12:26:40 UTC report contains 17 finite stages and 13 passing comparisons out
of 20. Both unchanged evaluators match their own untouched references bitwise;
both paired captures pass tolerance but are not bitwise equal. All five
completed fixed executables replay bitwise. The matrix substitution executable
is compiling, and its same-pair gate remains pending. Iris reports the same job
running at 12:26:38 UTC, with zero failures and preemptions. Preserve it through
the remaining branch; no new numerical candidate is selected.

All four small capture/substitution Jaxpr artifacts passed SHA, size and gzip
readback. Large compiler files remain regional without a new SHA audit. Finelog
records 204 samples through 12:27:18 UTC and an observed cumulative host-memory
peak of 91.55 GiB against the 128 GiB reservation; accelerator memory is null.
No optimizer, warmed performance, training MFU or checkpoint-resume claim is
made. Sole log observer 86562 and PR observer 77099 remain live.

[Original-gate evidence](optimizer-paired-input-substitution-original-gate.json)
retains the failed checks, report hash, operand-bank identities and pending gate.

## 2026-11-03 12:45 UTC — Both substitution reference gates fail; collection completes

The substitution diagnostic completed successfully on attempt 0 with task exit
0. Both executables fail their same-pair reference gates at rtol=atol=1e-5.
The original fails all seven direction leaves; the matrix version fails six,
with o_proj passing. All 21 state leaves pass in both cases. For the six failing matrix directions,
relative Frobenius errors range from 0.005995 to 0.029045 against its untouched
reference. Its gate_proj right-Gram and down_proj left-Gram states differ by
at most 1.77636e-15. These small state differences pass tolerance; zero measured
error on other leaves is not per-leaf bitwise proof.

Both fresh paired captures pass their own 28-leaf reference gates, not bitwise,
and both unchanged evaluators match their untouched references bitwise. The
failed same-pair substitution gates therefore remain decisive: neither altered
executable supports original-path interpretation. Passing capture agreement
alone did not validate substitution.

All 22 stages are finite and 14 of 26 comparisons pass. All six fixed executables
replay bitwise. The 12 failed comparisons comprise two full matrix-vs-original
comparisons, four substitution own-pair reference checks, and six swaps against
own-pair outputs. In the matrix substitution executable, numerator-only,
denominator-only and paired swaps change q_proj by relative Frobenius errors
0.00701153, 0.00613400 and 0.00700061, respectively, using that executable's
own-pair output as reference. The original executable's corresponding errors
were 0.00700109, 0.00610410 and 0.00700086. Other leaves have zero measured
absolute error in these swap comparisons. These are conditional differences
within altered executables, not an explanation of the untouched optimizer's
discrepancy. No optimization or training result is accepted.

Canonical job success was verified at 12:41:38 UTC and task success at
12:42:22 UTC. Online W&B finished with logging success, attribution completed
and candidate acceptance false. All 20 artifact sizes were verified, totaling
78,527,323 compressed bytes; all four small Jaxpr files passed SHA/gzip readback.
The 16 large compiler files remain in us-west4 without a new SHA audit. Finelog
records 274 samples through 12:35:04 UTC and observed host peak 91.55 GiB.
Accelerator memory is null; JAX allocator peak 8,128,457,216 bytes/device is not
total HBM. Log observer 86562 ended naturally with exit 0; no experiment or queue
remains. PR observer 77099 remains the sole live PR observer.

Next, compare the eight saved optimized HLO graphs on regional CPU, using the
untouched, unchanged-evaluator, paired-capture and substitution graphs for both
solvers. Map all seven groups by output dependencies, including changed non-q
groups and matrix o_proj as a passing control. The existing inventory needs
explicit 30-leaf paired-capture roles and external-input binding checks before
use. This is static design only: no extraction or numerical intervention has
been submitted. Static records cannot establish runtime values, precision,
summation order or cause. Both failed numerical gates remain in force.

[Completed evidence](optimizer-paired-input-substitution-v5e-completed.json)
records final report SHA
`bb2adee4365661f280afa0b24dc52c05d82083e109c716d1ac43fbaa7cac6e09`.
The [manifest](compiler-substitution-manifest.json) lists the eight regional
inputs; the [attribution plan](optimizer-divergence-attribution-plan.json)
records validation and interpretation requirements.

## 2026-11-03 13:12 UTC — Substitution changes declared paths across all seven groups

The regional CPU extraction completed successfully on attempt 0, task exit 0.
It verified the size and SHA of all eight saved optimized HLO graphs before
parsing. Each graph maps 105 polar products to seven output-dependent chains.
The external numerator and denominator are identified by source labels
`bank[1][0]` and `bank[1][1]`, then checked at q_proj's two normalization-division
operand positions. Both substitution graphs pass these binding checks;
conservative dependencies of non-q first products exclude the external inputs.

The unchanged evaluators match all 105 positions against their untouched graphs
in the inspected product declarations, operand shapes/opcodes, operation
attributes/backend records and enclosing fusion configurations. Substitution
changes enclosing fusion configurations at 62 original and 58 matrix positions,
spanning all seven groups. Matrix o_proj limits interpretation: its numerical
reference check passes despite changed enclosing configurations at seven
product positions. These counts alone do not explain the numerical failures.

In both solvers, q_proj normalization and first-product operand declarations
change from `f32[11,1024,128]` to `f32[11,1024,1024]`. The mapped entry all-reduces
group q+k with separate v in the untouched graphs, and k+v in substitution.
No mapped entry collective has q's first-product ancestry in substitution;
this does not prove that no nested reduction exists. The mapped k/v collective
algorithms remain RotatedPincerEmitter/UniDirection1DRingStrategy. Other backend
fields differ; the [comparison](compiler-substitution-comparison.json) records
them without interpreting runtime summation order or precision.

Five extraction rounds per graph recover all requested missing bodies.
Ninety-one of 112 first-product operand positions reach a normalization division;
opaque boundaries remain explicit. Missing-record closure does not establish
complete semantic provenance, and an absent mapping is not evidence of omitted
or dead source computations. Both numerical substitution reference gates remain
failed. These static records do not identify runtime values or cause.

The task used four CPUs, 8 GiB memory and 10 GiB disk in us-west4, with default
preemption retries and no wall-clock limit. Its worker name includes TPU, but
this task requested no accelerator and ran no optimizer kernels. Large HLO
stayed regional. The extracted report is 17,953,035 bytes, SHA
`1b395fca38d61e77cc56c2595adfdfdc9967b10718e6dbb5ff82b419bbfcf33b`.
[Completed evidence](compiler-substitution-completed.json) records source,
checksums and attempt UID `7bc0bb5d9519`. Log observer 67223 ended naturally
with exit 0. No experiment or queue remains; PR observer 77099 remains live.

Next, trace the q first-Gram result through its enclosing fusion and entry
consumers, and verify the k/v result projections after regrouping. Use existing
extracted records first; any missing bodies require published, targeted
regional CPU extraction. No numerical or precision/tiling intervention is
selected.

## 2026-11-03 13:24 UTC — First-Gram result paths close across all eight graphs

The targeted regional CPU extraction succeeded on attempt 0 with exit 0.
It verified all eight optimized HLO sizes and hashes and recovered 15 selected
or nested computation bodies per graph. A tuple-aware trace closes all 48
operand paths from each q/k/v second polar product back to its first Gram
product, across eight graphs and two operands. Every path reaches only its
intended group's first product through leaf 0 of that product's enclosing
fusion, with no missing record or opaque boundary on these paths.

In the untouched, unchanged-evaluator and paired-capture graphs, both q paths
cross an entry all-reduce. In both substitution graphs, neither q path crosses
a declared collective between the first and second products. The k/v paths
retain one all-reduce each. After regrouping, result projection 0 follows only
k's operand and projection 1 follows only v's operand; the independent
combined-reduction projections are preserved. This scope ends at the first
product and does not describe upstream computations or other consumers.

The trace establishes a declared path difference introduced by substitution.
It does not measure runtime summation order, precision or numerical causality.
Both substitution reference gates remain failed. No optimizer or training
result is accepted. Parser fixtures cover nested fusion/tuple bindings, reversed
collective operands, repeated operand edges, missing bodies and opaque stops.
A deliberate mutation of k's combined-reduction result projection in an actual
graph is rejected. [Validation](compiler-first-gram-paths-validation.json) and
[traced paths](compiler-first-gram-paths-comparison.json) retain this scope.

The task used four CPUs, 8 GiB memory and 10 GiB disk in us-west4 with default
preemption retries and no wall-clock limit. It requested no accelerator and
ran no numerical kernels. It was already terminal at the first canonical read;
no log observer was attached. All large HLO remained regional. The retrieved
body report is 195,969 bytes, SHA
`4e5351ff3b6dfdce01dbcc96dfbd4637c42216266436ae478a357e4d5bd86295`.
[Completed evidence](compiler-first-gram-paths-completed.json) records the
published source and attempt UID `bdf95e039f94`. No experiment or queue remains;
PR observer 77099 is still the sole live PR observer.

Next, review a diagnostic that places only the external full q inner matrix in
the existing q-direction column sharding, with the denominator replicated and
all original full-update inputs, outputs and source arithmetic unchanged.
This tests whether aligning external input placement can recover the full
28-leaf same-pair reference gate. It does not force kernel tiling or precision,
and matching declarations would not establish numerical agreement. The design
is not implemented or submitted. It requires fresh online CPU validation, a
replicated-input regression, exact published source and one sequential TPU
diagnostic. All original numerical and training gates remain in force.

## 2026-11-03 13:35 UTC — Column placement control passes its CPU preflight

The column-input substitution control changes only the external q_proj
numerator's input placement to the existing q-direction column sharding,
`PartitionSpec(None, None, "data")`, on the eight-device data mesh. The
denominator remains replicated. Global shapes remain 11×1024×1024 FP32 for
the numerator and 11×1×1 FP32 for the denominator. The Jaxpr substitution
helper, original full-update input/output shardings and source arithmetic are
unchanged. All 28-leaf reference checks retain rtol=atol=1e-5 and bitwise checks.

One fresh diagnostic process runs both the original and matrix eigensolver
variants. Each variant recaptures its paired normalization inputs with
untouched and unchanged-evaluator references. One fixed substitution executable per solver evaluates all four
combinations of the two captured numerators and denominators, then replays its
own pair. Both capture gates and each executable's own-pair checks against
untouched and captured full-update outputs must pass before original-path
interpretation. Failed checks remain in the report. Passing endpoints still
do not establish hidden internal identity or numerical cause.

Own online CPU validation from source `446be9950` completed with 22 finite
stages, 26 passing comparisons and six bitwise fixed-executable replays. The
replicated-input substitution regression also passed 22 stages, 26 comparisons
and six replays. Both runs' same-pair checks pass tolerance but are not bitwise.
The new mode has nonzero measured error only in q momentum, maxabs
2.9103830456733704e-11. The replicated control also has a q-direction difference,
maxabs 9.5367431640625e-7. These are small CPU matrices and do not establish TPU
behavior or explain the differences. Shared mapping, repeated-use, non-target
preservation, shape/order and rejection fixtures passed.

Explicit device-index checks verify eight disjoint column slices for the new
CPU numerator and replication for the denominator. StableHLO signatures retain
28 outputs, with the two extra inputs declared as 3×8×8 and 3×1×1 FP32 and the
intended shardings. All 40 artifacts passed size/SHA/gzip readback, totaling
891,581 bytes for the new run and 886,995 for the regression. Both reports match
GCS exactly and both online W&B runs finished. A separate fresh online logging
check also finished successfully. The [preflight](optimizer-column-input-substitution-preflight.json)
records identities and checksums; the [plan](optimizer-column-input-substitution-plan.json)
records the single sequential TPU launch. No TPU result or optimizer acceptance
is claimed. Large compiler artifacts will remain in us-west4.

## 2026-11-03 13:38 UTC — Column-input TPU diagnostic starts

This control column-shards the external q_proj normalization numerator while
keeping its denominator replicated. It was accepted at 13:35:15 UTC from
published source `406d595538a32a6e97df2869d2bb0438efecd465`. Canonical job and
task state were running at 13:37:36 UTC, attempt 0, UID `7a5f9038a997`.
It uses eight native v5e chips in one JAX process, 96 CPUs, 128 GiB memory and
50 GiB disk in us-west4, with default preemption retries and no wall-clock limit.
The script and unchanged capture-helper hashes match the CPU-validated source.

Fresh online W&B run
[6fa93c0837a0](https://wandb.ai/marin-community/marin/runs/ema-gram-20-attribution-6fa93c0837a0)
passed its logging check before kernels. Three stages are finite at startup:
zero initialization, prior-state refresh and the untouched original full update.
Capture and same-pair substitution reference gates are pending. Single-call
timings include compilation and are not warmed performance or training MFU.
Finelog records 19 samples through 13:37:28 UTC and an observed host peak of
34.919921875 GiB; this is not total accelerator HBM. No current compiler-artifact
readback is claimed. Large HLO remains regional.

Sole experiment log observer 9948 started at 13:35:26 UTC. Reuse it and preserve
the job through compilation; no duplicate or concurrent candidate is authorized.
PR observer 77099 remains the sole live PR observer. The active heartbeat will
collect both solver branches and all reference gates before the next decision.
[Startup evidence](optimizer-column-input-substitution-started.json) records
the exact job, command, source hashes and output prefix. No optimizer or training
result is accepted.

## 2026-11-03 13:46 UTC — Both fresh capture gates pass; substitution remains pending

This diagnostic captures q_proj's normalization numerator and denominator,
then column-shards the external numerator during substitution while keeping
the denominator replicated. Both fresh paired captures pass their own
28-leaf reference gates at rtol=atol=1e-5, but neither is bitwise identical.
Original capture differs in v_proj by maximum absolute error 8.940696716308594e-8.
Matrix capture differs in q/k/v_proj by 1.043081283569336e-7,
8.940696716308594e-8 and 1.043081283569336e-7, respectively. Other leaves have
zero measured absolute error, which is not per-leaf bitwise proof. Both
unchanged Jaxpr evaluators match their own untouched references bitwise, and
all four fixed capture/evaluator executables replay bitwise.

The 13:45:12 UTC report contains 12 finite stages and 11 comparisons, ten
passing. The recorded failure is untouched matrix versus original, with all
seven directions failing and all 21 state leaves passing. Cross-solver capture
comparisons and both column-input substitution own-pair gates are pending.
Passing capture checks do not establish untouched hidden internal identity;
the substitution gates remain necessary before interpreting swaps as evidence
about the untouched path. No
optimization or training result is accepted.

Finelog records 85 samples through 13:44:45 UTC and observed cumulative host
peak 48.6494140625 GiB against the 128 GiB reservation. This is not total HBM.
Both small capture Jaxpr artifacts pass SHA/size/gzip readback; large compiler
files stay regional without a new checksum audit. [Partial evidence](optimizer-column-input-substitution-partial.json)
records the report hash, comparisons and telemetry query. Canonical RUNNING
remains the 13:37:36 UTC observation; the fresh report and live log observer
show continued collection, without another scheduler poll before its regular
interval. Preserve the existing job, sole experiment log 9948 and PR observer
77099. All numerical and training gates remain unchanged.

## 2026-11-03 14:00 UTC — Original column-input substitution fails k and o reference leaves

With the external q_proj numerator column-sharded and its denominator
replicated, the original substitution executable's own captured pair fails
full-update checks against both untouched and paired-capture outputs at
rtol=atol=1e-5. The failing leaves are
k_proj and o_proj directions. Against the untouched reference, their maximum
absolute errors are 0.0039804131 and 0.0039047748; relative Frobenius errors are
0.0066678317 and 0.0067292870. All other direction leaves and all 21 state leaves
pass tolerance. q_proj passes with maxabs 1.1920928955078125e-7 and relative
Frobenius error 1.7284896441321625e-7. Nonzero measured errors also occur in
v_proj and gate_proj, within tolerance. Zero measured error elsewhere is not
per-leaf bitwise proof.

Passing q_proj does not satisfy the full 28-leaf gate. Both original
same-bank reference agreement and conditional interpretation flags are false.
These flags prohibit attributing swaps to the untouched path; differences
within the altered executable can still be measured. Within this altered fixed
executable, numerator-only, denominator-only and paired swaps fail q_proj at
relative Frobenius errors 0.0070278945, 0.0060659926 and 0.0070323970, using the
own-pair q output norm as denominator. Other 27 leaves have zero measured
absolute error in these swap comparisons. These conditional differences do
not establish the cause of the untouched optimizer discrepancy.

The 13:59:26 UTC report has 17 finite stages and 20 comparisons, 13 passing.
Both fresh paired capture gates pass tolerance but are not bitwise; both
unchanged evaluators match their own untouched references bitwise. All five
completed fixed executables replay bitwise. The matrix substitution executable
is compiling; its own-pair gate and final artifacts remain pending. Canonical
job state is RUNNING at 13:59:26 UTC with zero failures or preemptions. Preserve
the existing job and sole log observer 9948 through the matrix branch. No new
candidate is submitted and no optimizer or training result is accepted.

Finelog records 221 samples through 13:59:47 UTC and observed cumulative host
peak 82.7294921875 GiB against 128 GiB. This is not total accelerator HBM.
All four small capture/substitution Jaxpr artifacts pass SHA/size/gzip readback;
large compiler files stay regional without a new checksum audit.
[Partial gate evidence](optimizer-column-input-substitution-original-gate.json)
retains all leaf errors, fixed-executable replays and telemetry scope. PR observer
77099 and the heartbeat remain active; reuse both existing observers.

## 2026-11-03 14:18 UTC — Column-input collection completes; both full-reference gates fail

The executable substitutes captured q_proj numerator and denominator values
at the existing normalization division. Only the external numerator placement
changes to column sharding; the denominator remains replicated. This does not
recover the full 28-leaf substitution reference gate. The original executable fails k_proj and
o_proj directions; the matrix executable fails gate_proj and down_proj.
Both fail against their own untouched and paired-capture references at
rtol=atol=1e-5. All 21 state leaves pass. q_proj passes in both executables,
but that does not satisfy the full gate or permit original-path attribution.

Against untouched references, original k/o relative Frobenius errors are
0.00666783 and 0.00672929. Matrix gate/down errors are 0.02793480 and
0.02798260, with maximum absolute errors 0.00959935 and 0.00915089.
Matrix gate right-Gram and down left-Gram differ by at most 1.77636e-15,
within tolerance. Original q_proj has maximum absolute error 1.19209e-7;
matrix q_proj has zero measured absolute error against its untouched
reference. Zero measured error is not per-leaf bitwise proof.

Both fresh paired captures pass their own full reference gates, not bitwise;
both unchanged evaluators match their untouched references bitwise. All 22
stages are finite, 14 of 26 comparisons pass, and all six fixed executables
replay bitwise. The 12 failures comprise two full matrix-versus-original
checks, four own-pair reference checks, and six within-executable swaps.
Original numerator-only, denominator-only and paired swaps change q_proj by
relative Frobenius errors 0.00702789, 0.00606599 and 0.00703240. Matrix errors
are 0.00700319, 0.00613309 and 0.00703216. Each denominator is the norm of the
corresponding own-pair output leaf. Other 27 leaves have zero measured error
in these swaps. These conditional effects in altered executables do not
explain the untouched optimizer discrepancy. No optimization is accepted.

The current and prior replicated-input reports pass local SHA verification;
their input-bank fingerprints and both untouched output hashes match. This
supports comparing the recorded outcomes across jobs, without establishing
hidden intermediate identity. The prior replicated-input substitution failed
seven original directions and six matrix directions; the column-input case
fails two in each solver. Both still fail the required full-update gate.

Canonical job/task success was verified at 14:14:17/14:14:18 UTC, attempt 0,
task exit 0. W&B finished with logging success, attribution completed and
candidate acceptance false. All 20 artifact sizes were verified, totaling
78,542,301 compressed bytes; four small Jaxpr files passed SHA/gzip readback.
The 16 large compiler files remain regional without a new SHA audit. Finelog
has 277 samples through 14:05:59 UTC and observed host peak 96.97 GiB.
JAX allocator peak 8,122,036,224 bytes/device is not total HBM. Log observer
9948 ended naturally with exit 0. No experiment or queue remains; PR observer
77099 remains the sole live PR observer. No warmed-performance, training MFU
or checkpoint-restoration claim is made.

Next is static comparison of the eight saved current optimized HLO graphs
with their own controls and the prior replicated-input records. Verify actual
q normalization/product widths and the first-to-second-product collective
path; requested input placement alone does not establish those paths. Include
failed original k/o and matrix gate/down, passing q and other groups. Review
compiled local parameter shapes separately from global source shapes before
reusing the external-binding checks. This is design only, with no extraction
or new numerical submission. Any extraction must verify consumed HLO hashes
on regional CPU, keep large HLO in us-west4, and preserve tuple/caller bindings
and opaque limitations. Static records cannot establish runtime values,
precision, summation order or cause.

[Completed evidence](optimizer-column-input-substitution-v5e-completed.json)
records final report SHA
`43edf83f3b2712f43bbe74ea87a23bf9fdf9ea7b3d282869bb11f046fabf916d`.
The [manifest](compiler-column-input-manifest.json) lists regional inputs;
the [attribution plan](optimizer-divergence-attribution-plan.json) records the
next design and its validation requirements.

## 2026-11-03 14:43 UTC — Column input recovers the declared q collective path, not full numerical agreement

Regional CPU inspection verifies that both column-input substitution graphs
use q normalization and first-product operands declared f32[11,1024,128].
The external source matrix still has global shape 11x1024x1024. In both graphs,
the q first-to-second-product path crosses an all-reduce, matching the presence
of that collective in their untouched controls. The prior replicated-input
substitution had width 1024 at those operands and no declared collective on
that segment. This is a compiler-path comparison, not proof that the collective
change caused the numerical differences.

All 48 paths across eight graphs, three groups and two second-product operands
reach only their intended first Gram product through fusion result leaf 0.
Each crosses one all-reduce, with no missing record or opaque boundary within
this segment. The combined q/k reduction independently projects q and k;
v has a separate reduction. The original q collective record matches its
untouched control; the matrix record differs only in its aliasing-operands
field. Both declare RotatedPincerEmitter/UniDirection1DRingStrategy. These
records do not establish runtime summation order or precision.

Column substitution changes enclosing fusion configurations at 16 original
and 19 matrix product positions against untouched controls. Three original
product result declarations differ only by S(1) tags, whose meaning is not
interpreted here. Matrix gate_proj and down_proj fail numerically despite
matching all inspected fields at their 15 product positions, including parent
configurations. Those comparisons omit upstream values and connections.
Original k/o each has one changed parent configuration, but this association
does not establish causality. All six control graphs have identical compressed
HLO hashes across the column and replicated jobs, and their inspected product
records match. The two substitution graphs differ in parent configurations at
64 original and 53 matrix positions across input placements.

The initial CPU extraction failed its strict no-opaque-provenance guard at the
original substitution numerator. It had completed three control graphs. That
failed attempt is preserved. A separately published parser revision retains
ConcatBitcast as an explicit opaque boundary on the numerator route, while
checking the source label, local parameter shape and operand position and
rejecting unexpected boundary kinds. Both substitution numerators traverse
this boundary; the denominator routes do not. These are conservative source
roles, not complete semantic provenance. The separate first-to-second-product
paths above begin downstream of this boundary. Normalization closure recovers
all requested missing records in five rounds per graph, with 90 of 112 operand
positions reaching normalization; other opaque boundaries remain.

The revised CPU-only task succeeded on attempt 0 with exit 0, verified at
14:40:02 UTC. It used four CPUs, 8 GiB memory and 10 GiB disk in us-west4, with
default preemption retries and no wall-clock limit. All eight optimized HLO
files passed regional SHA/size verification before parsing. Large HLO stayed
regional; the extracted report is 18,035,069 bytes with SHA
`b20c7269986af1a9b4a30ce20309f595a5e7504b4206674c37fc21d2159d5a8a`.
The initial failed task and revised successful task are terminal; neither will
be restarted. Their log observers ended naturally. PR observer 77099 remains
live and sole. No experiment or queue remains.

Both numerical full-reference substitution gates remain failed. No optimizer,
training, performance or MFU result is accepted. Next is static design only:
trace pre-normalization and outer-Gram operand bindings for matrix gate/down,
with passing up/o controls, using existing extracted records first. Preserve
caller contexts, repeated operands and opaque boundaries; recover explicitly
missing bodies on regional CPU only if needed. Matching product fields alone
cannot establish identical hidden computations or explain the numerical cause.
No new numerical, precision or tiling intervention is selected.

[Completion](compiler-column-input-completed.json),
[comparison](compiler-column-input-comparison.json),
[tuple paths](compiler-column-input-first-gram-paths.json) and
[initial failure](compiler-column-input-initial-failed.json) retain the evidence.

### November 3, 15:00 UTC — Upstream and outer-Gram bindings remain only partly visible

The new CPU-only extraction compares normalization operands and final direction
outputs for matrix gate/down, which fail the numerical substitution gate, and
up/o, which pass it. All twelve compared unchanged-evaluator signatures match the untouched
signatures after source labels are normalized from the wrapper argument structure.
Seven paired-capture traces and eight column-substitution traces differ from
untouched controls. These are differences in rooted declarations and ordered
connections, not evidence of different runtime values.

For column substitution, gate_proj differs at all three roots: normalization
numerator, denominator and final direction. One corresponding upstream route
changes from a tuple projection to copy-done; collective channel IDs also differ.
The down_proj numerator trace matches but stops at opaque ConcatBitcast. Its
final-direction signature differs only in two collective channel IDs (329→326 and 331→328). This comparison stops
at opaque boundaries and does not identify a numerical cause. Passing controls
also differ: up_proj at the denominator and final direction, and o_proj at the
denominator. A differing signature therefore does not distinguish a failed
numerical gate from a passing one.

The final-output traces expose one direct gram_direction product for gate and
up, and two for down and o, per graph. Their immediate operand declarations are
recorded by position. Other parts of the outer-Gram computation remain behind
opaque AsyncCollectiveDone or ConcatBitcast operations; eigensolver while loops
are also boundaries. None of these traces reaches a polar-product anchor.
Recovering all requested records is not complete semantic provenance. The
signature comparison includes constant literals and canonical source roles,
retains channel IDs, and excludes instruction identifiers and metadata. Raw
traces retain caller contexts and repeated edges; recursive signatures do not
distinguish edge sharing.

The job succeeded on attempt 0 with exit 0, verified at 14:54:31 UTC. It used
four CPUs, 8 GiB memory and 10 GiB disk in us-west4, with default preemption
retries and no wall-clock limit. Exact source and manifest were published
before extraction. All four optimized HLO files passed regional size/SHA
verification; large HLO stayed regional. Nine closure rounds per graph recovered
99/99/100/101 requested bodies for untouched/evaluator/capture/substitution.
The 5,717,993-byte report passed exact GCS readback and has SHA
`81c89f0aaceed5bf7ff2a9ce09eefd1d554172299b58080438d909cf6269f8c4`.

Both full-update numerical substitution gates remain failed. No optimization,
training or MFU result is accepted. Next is static design only: inspect the
remaining outer-product async/ConcatBitcast boundaries and gate's changed
producer route using existing records first. Retain opaque semantics and
compare passing controls before selecting any numerical intervention.
The CPU log ended naturally; no experiment remains. PR observer 77099 remains
sole, with its existing 12-hour exit condition.

[Completion](compiler-outer-bindings-completed.json),
[comparison](compiler-outer-bindings-comparison.json) and
[validation](compiler-outer-bindings-validation.json) retain the evidence.

### November 3, 15:15 UTC — Opaque endpoint inputs and gate trace-scalar route

The new static extraction follows declared inputs of the previously recorded
AsyncCollectiveDone and ConcatBitcast endpoints. It preserves the original
outputs as opaque and stores these input dependencies as auxiliary traces.
Untouched/evaluator/capture/column-substitution contain 30/30/29/31 distinct
endpoints in the inspected contexts, with 176/176/180/188 operand positions.
All 176 unchanged-evaluator operand signatures match the untouched controls
after source-wrapper label normalization. Capture and substitution endpoints
are not aligned by text order or list index.

The four graphs contain 10/10/11/11 AsyncCollectiveDone endpoints. For every
such endpoint, all ten operand-position traces reach one common declared
AsyncCollectiveStart identity through recorded tuple projections. This is a
connection in the emitted records, not proof of exact start/done semantics.
The auxiliary traces still stop at opaque starts, other custom calls and solver
while loops; none reaches a polar-product anchor. Requested records are now
complete for this auxiliary traversal, but semantic provenance remains partial.

A separate analysis of the saved outer-binding report identifies the changed
gate route as a Gram trace scalar feeding the `mean > 0` predicate. All four
graphs select projection 3 of a combined all-reduce. The recorded scalar is
multiplied by 0.0009765625 (1/1024), compared with zero, then broadcast into the
Gram-power selection. The path has three `copy-start`/`copy-done` pairs in untouched/evaluator
and four in capture/substitution. The extra pair follows the all-reduce and
also occurs in the paired-capture control, whose full reference comparison
passes tolerance. Its presence alone cannot distinguish the failing
substitution from that passing control. S(1) tags and channel IDs are retained
without assigning them runtime precision or summation-order meaning.

The CPU task succeeded on attempt 0 with exit 0, verified at 15:11:57 UTC.
It used four CPUs, 8 GiB memory and 10 GiB disk in us-west4, with default
preemption retries and no wall-clock limit. Published source preceded launch.
All four HLO files passed regional size/SHA checks. Closure took 11/11/11/12
rounds and recovered 53/53/56/55 requested bodies. Large HLO stayed regional;
the bounded extracted report is 30,706,097 bytes, with exact GCS readback and SHA
`cbf8996a1579a9267501e402ccb379fe8dddaf07cb2148ea845ff31dd290c6cd`.
The initial `--tail` read exited during building and was not terminal evidence.
The subsequent `--follow --tail` observer ended after the completion record;
canonical job and task reads confirmed success.

Both numerical full-update substitution gates remain failed. No optimizer,
training, MFU or causal result is accepted. Next is static design only: inspect
matched start-record operands and their declared all-gather/product producers,
using the saved inventory and traces first. Preserve tuple/caller bindings and
opaque semantics, with passing up/o and paired-capture controls. No numerical
intervention is selected. No experiment remains; PR observer 77099 remains sole.

[Completion](compiler-boundary-inputs-completed.json),
[endpoint summary](compiler-boundary-inputs-summary.json),
[gate route](compiler-gate-scalar-route.json) and
[validation](compiler-boundary-inputs-validation.json) retain the evidence.

## 2026-11-03 15:31 UTC — Declared start inputs reach product records

The regional CPU extraction recovered the declared all-gather inputs for four AsyncCollectiveStart calls per matrix graph. These calls were selected from the immediate opaque boundaries of the final Gram-direction products' operand paths. Traversal stops at intervening products, so upstream covariance calls are excluded. The four groups are gate/down, whose column-substitution direction checks failed, and passing up/o controls. The untouched, unchanged-evaluator, paired-capture and column-substitution graphs are compared.

All four evaluator and all four paired-capture normalized input-route signatures match untouched. These signatures remove compiler identifiers and wrapper metadata, retain declared shapes, attributes, ordered connections and constant literals, and canonicalize source-wrapper roles; this is record normalization, not numerical input normalization. In substitution, every group differs in its all-gather channel and chain IDs. Gate also differs in S(1) tags on sibling tuple leaves 2 and 3 of a producer fusion; the traced path selects leaf 0, whose declaration is unchanged. Down/up/o have no other differences within these bounded signatures. These identifiers and tags are recorded without interpreting runtime precision, order or cause.

Gate/up each reach a product labeled with the source function gram_direction after six recorded nodes. Down/o each have twelve nodes and terminate at a polar-product record and an opaque ConcatBitcast. The product records match the B_X matrix product (B times X) in the fifth polar update in the existing dependency-mapped polar chains. Start/Done outputs remain opaque: examining their declared inputs does not establish exact custom-call semantics or complete left/right Gram-power provenance.

The task succeeded attempt 0/exit 0, UID b848e05f829c, verified November 3 at 15:27:57–58 UTC. It used published source ee1310afe12a9ccd0da6ba00133b28df3d67d89f, four CPUs/8 GiB/10 GiB in us-west4, default preemption retries and no wall-clock limit. Two closure rounds recovered eight bodies per graph; no requested records remain missing. All four optimized HLO SHA/size checks passed regionally. Large HLO stayed regional. The extracted report is 387,447 bytes, SHA 48b61b4e367b55789f002f357f8551fb662e71a24997fb369007a50bd0192435, with exact GCS readback.

Both full 28-leaf numerical substitution gates remain failed. No optimization, training, MFU or causal result is accepted. Next is static design only: inspect the newly reached gate/up direct-product operand bindings and the down/o polar-product-plus-ConcatBitcast input branches using saved records first. Preserve passing paired-capture controls and opaque boundaries; do not infer a cause from unselected tuple tags or channel renumbering. No new numerical intervention is selected. No experiment remains; PR observer 77099 remains sole and the heartbeat stays active.

[Completion](compiler-start-producers-completed.json),
[comparison](compiler-start-producers-comparison.json) and
[validation](compiler-start-producers-validation.json) retain the exact source,
report, declared records and fixture checks.

## 2026-11-03 15:43 UTC — Product operand routes retain numerical ambiguity

The regional CPU extraction recovered sixteen declared operand routes per matrix graph: eight inputs to four product records, plus eight inputs to two ConcatBitcast calls. All sixteen unchanged-evaluator routes and all sixteen paired-capture routes match the untouched graph's normalized signatures. The signatures compare declared shapes, attributes, constant literals and ordered connections after removing instruction/computation names and canonicalizing source-wrapper labels; channel IDs remain in the comparison; they do not compare numerical values.

Ten substitution routes match. Gate/up product operand 0 differs only in a recorded all-reduce channel ID (231→228 and 276→273) within these signatures. The four inspected o ConcatBitcast inputs encounter copy-done where the controls reach a tuple projection. The comparator stops comparing descendants at that opcode mismatch; it does not claim the remaining subtrees are equal. These four positions traverse the same changed route, not four independent copy changes. All six down routes match despite down's numerical direction failure, and o remains a passing numerical control despite its changed routes.

The polar source uses A = XXᵀ and B = bA + (cA)A. The existing chain map names the products BX and (cA)A as B_X and scaled_A_A, respectively. Gate/up operand 0 reaches the corresponding left-Gram state parameter, a Gram-refresh product and a relative_gram_power product, supporting a declared left-power route. Traversal stops at those product records; their inputs and runtime values are not established. Gate operand 1 reaches the fifth polar update's B_X product (B times X) and an opaque ConcatBitcast. Up operand 1 stops at an opaque ConcatBitcast. Down/o B_X operand 0 reaches the fifth update's scaled_A_A product and a ConcatBitcast; operand 1 stops at AsyncCollectiveDone. The four declared ConcatBitcast inputs for each group reach earlier B_X product records—steps 3/4 for down and 2/3/4 for o—plus another opaque ConcatBitcast. These are source-scope/dependency-map matches, not semantic identity across the opaque calls.

The task succeeded attempt 0/exit 0, UID 7bd00ca2a4c0, verified November 3 at 15:38:57–58 UTC. Published source a6feb8318abeff0da2d4fd171ed4c0021c3027fb ran on four CPUs/8 GiB/10 GiB in us-west4 with default preemption retries and no wall-clock limit. The worker name includes TPU, but the task requested CPU only and ran static parsing without kernels. Seven closure rounds recovered 27 bodies per graph, with no requested records missing. All four optimized HLO SHA/size checks passed regionally; large HLO stayed there. The 2,162,972-byte extracted report passed exact GCS readback, SHA 580d846d18b0aacd842a91b3b77b5826b1a9ce7a1c1dec7c562ed9b5fd605e04.

Both full 28-leaf numerical substitution gates remain failed. Static agreement does not establish runtime precision, summation order, hidden values or cause. No optimizer, training or MFU result is accepted. Next is static design only: inspect the declared producers behind the remaining down/o fifth-step operand-1 AsyncCollectiveDone boundaries and gate/up polar-input ConcatBitcast boundaries, using existing records first. Preserve passing paired-capture and up/o controls, caller contexts and repeated positions, and keep the custom-call outputs opaque. No numerical intervention is selected. The experiment is terminal; PR observer 77099 remains sole and the heartbeat stays active.

[Completion](compiler-product-inputs-completed.json),
[comparison](compiler-product-inputs-comparison.json) and
[validation](compiler-product-inputs-validation.json) retain the source identity,
report checksums, declared routes and fixture results.

## 2026-11-03 16:00 UTC — Polar boundary inputs preserve the unresolved numerical gates

The regional CPU extraction recovered thirty declared input routes per matrix graph: four ConcatBitcast inputs each for gate/up, ten AsyncCollectiveDone inputs each for down/o, and one input to each of the two matched AsyncCollectiveStart calls. The selected endpoints come from the preceding product-operand traces, with explicit group and operand positions. All thirty unchanged-evaluator signatures and all thirty paired-capture signatures match the untouched controls. These signatures compare declared shapes, attributes, constant literals and ordered connections after normalizing instruction/computation identifiers and source-wrapper labels. Channel and chain IDs remain in the comparison; the signatures do not compare runtime values.

Twenty-six substitution signatures match. Two down routes differ only in the recorded all-gather channel/chain IDs, 358→355 and 53→57. The corresponding two o routes differ in IDs 224→221 and 31→35; the o Start-input route also encounters the previously observed copy-done versus tuple-projection change. The comparison stops at that opcode mismatch without claiming equality of the remaining subtree. These are four route comparisons, not four independent changes. All eight gate/up ConcatBitcast-input signatures match, while o remains a passing numerical control despite its changed route.

For every selected down/o Done call, all ten operand traces reach one common declared Start identity. Their recorded Start tuple paths, in operand order, are [0], [1], [0], [2], [3], [4], [5], [6], [7], [8]; repeated leaf 0 is retained. Done operand 2 encounters an all-gather before reaching Start leaf 0. That record and the all-gather on the Start's single declared input are distinct records with matching channel/chain IDs. They are not merged or assigned exact custom-call semantics. The auxiliary traces terminate at product records and opaque ConcatBitcast or Start calls. Original Start/Done/ConcatBitcast outputs remain opaque.

The task succeeded attempt 0/exit 0, UID b42d5d31eb79, independently verified at 15:55:02–03 UTC. Published source 0175712e0e2878df4c99f2c0a451bad4cd645425 ran on four CPUs/8 GiB/10 GiB in us-west4 with default preemption retries and no wall-clock limit. The worker name includes TPU, but this was CPU-only static parsing without kernels. Four closure rounds recovered ten bodies per graph; no requested records remain missing. All four optimized HLO files passed regional SHA/size checks and stayed regional. The 1,775,955-byte report passed exact GCS readback, SHA 41fcd05ef632f18edbe5dd6f8924137a28e9caa406c50d5c504e5e599ea0a709. Fixtures cover all sixty control comparisons, explicit position ordering, common-Start identity, tuple projections, duplicate/missing records and channel mutations.

Both full 28-leaf numerical substitution gates remain failed. Static agreement does not establish hidden values, precision, summation order, semantic identity or cause. No optimizer, training or MFU result is accepted. Next is static design only: consolidate the saved product and opaque-boundary records by graph, caller context, computation, instruction and tuple path. Identify which remaining frontiers are already covered and which bindings remain unresolved before requesting more extraction. Preserve the passing paired-capture and up/o controls and repeated edges. No numerical intervention is selected. No experiment remains; the heartbeat stays active. PR observer 77099 reached its authorized twelve-hour timeout and exited with code 2. Its final snapshot reports PR #10790 open, mergeable and review required, with 16 comments and 7 reviews. Monitoring is handed off without a replacement observer or manual polling.

[Completion](compiler-polar-boundaries-completed.json),
[comparison](compiler-polar-boundaries-comparison.json) and
[validation](compiler-polar-boundaries-validation.json) retain source identity,
report checksums, declared routes and fixture results.

## 2026-11-03 16:12 UTC — Saved trace coverage exposes specific input gaps

A local consolidation of six checksummed reports separates repeated trace records from uninspected operand bindings. It joins records only within one optimized HLO graph, retaining caller context, computation, instruction and tuple path. It preserves ordered edges and repeated operand positions. Explicit traces of a custom call's inputs are recorded separately from the opaque semantics of its output. No new job, accelerator, numerical kernel or HLO transfer was used.

For untouched/evaluator/capture/column-substitution respectively, 8,526/8,526/8,187/8,884 record occurrences reduce to 3,084/3,084/3,144/3,092 distinct records with caller context and tuple projection retained. Boundary occurrences total 480/480/472/521, representing 193/193/200/201 distinct context-and-projection boundaries, or 92/92/93/94 instructions when projections of the same instruction are grouped. These are coverage counts, not cross-graph equality checks or runtime graphs.

Every graph has eighteen product frontiers: four have complete saved input traces and fourteen do not. Ten of the fourteen have roles in the existing dependency-mapped polar chains: gate B_X steps 4/5, up B_X step 5, down B_X steps 3/4 and scaled_A_A step 5, and o B_X steps 2/3/4 and scaled_A_A step 5. None of the fourteen products' 28 immediate operand records is present in the saved traces in the same caller context. Four operand declarations are available in the scoped inventory; their availability alone does not provide full caller bindings. The remaining four product frontiers are outside this polar-chain map.

Grouping output projections by instruction, untouched/evaluator each have 42 opaque frontiers whose declared inputs are fully inspected and 22 whose inputs are not; capture has 41/24 and substitution 43/20. The uninspected opaque instructions comprise 6/6/7/7 Start calls, 10/10/11/7 ConcatBitcast calls and six while loops per graph. All opaque outputs remain opaque, including those with complete input coverage. The index rejects reported missing records within supplied traces and conflicting declarations. Deliberately unexpanded frontier operands remain valid coverage gaps. It does not join different caller contexts or infer semantics from matching channel IDs.

Fixtures passed for repeated edges, caller-context and tuple-projection separation, retained opacity, conflicting declarations, duplicate positions, wrong operand roots and missing records. Reordering record and edge lists, and reordering operand-list entries while retaining their explicit position labels, preserves the summary and edge identities in all four actual graphs. All source report SHA/size checks and per-graph HLO identity checks passed. The full consolidated index remains local; the published source and manifest reproduce it from the existing bounded reports. No large artifact was transferred.

Both full 28-leaf numerical substitution gates remain failed. The next static question is confined to the fourteen product frontiers' operand producers, with gate/down failures and paired-capture/up/o passing controls retained. Reuse the available scoped declarations first; any further regional CPU extraction must publish exact source, manifest and checksums before launch. Keep opaque boundaries explicit and use output dependencies and caller bindings rather than text-order alignment. No numerical intervention or training acceptance is selected. The heartbeat remains active, with no experiment or observer running; PR monitoring remains handed off after its twelve-hour timeout.

[Coverage summary](compiler-record-coverage-summary.json),
[completion](compiler-record-coverage-completed.json),
[validation](compiler-record-coverage-validation.json) and
[reproducible source](consolidate_compiler_records.py) retain the evidence.

## 2026-11-03 16:30 UTC — Product-frontier inputs expose a deeper down route change

The regional CPU extraction traced both operands of fourteen previously uninspected product frontiers in each matrix graph. All 28 unchanged-evaluator signatures and all 28 paired-capture signatures match the untouched controls. Column substitution matches 17 of 28. These are static operand-route comparisons, separate from the full 28-leaf numerical reference gates, which remain failed for both substitution solvers.

The new down difference feeds operand 1 of the fifth polar update's scaled_A_A product, the source expression (cA)A. In untouched, evaluator and paired-capture graphs, the route reaches an AsyncCollectiveDone fusion directly from the consuming fusion's parameter. Substitution inserts a copy-done/copy-start pair on that route before reaching its own AsyncCollectiveDone fusion. The reached custom-call outputs remain opaque. The normalized comparator stops at the fusion-versus-copy-done mismatch; it does not establish equality of the remaining producer subtree. The other five inspected down operand routes match. The preceding report's six matching down routes stopped at shallower product frontiers and did not cover this new difference.

The other ten substitution differences include three relative_gram_power operand routes with only recorded all-reduce channel-ID changes within their bounded signatures; two gate Gram-refresh inputs with S(1) declaration tags; two up Gram-refresh inputs encountering an entry gradient parameter where controls have a copy; and three o polar-product inputs with S(1) tags or producer-route changes. Up and o remain passing numerical controls despite their changed routes. Recorded bf16 declarations and S(1) tags are compiler evidence, not a new source-precision override or proof of runtime arithmetic precision. Neither copies, channel renumbering nor tags establish the cause of a numerical failure.

Targets retain exact graph, caller context, computation, instruction and operand position from the coverage index. Ten products use the existing dependency-mapped polar roles; four use gate/up Gram-refresh and relative-power source scopes. Three closure rounds recovered forty bodies per graph, with no requested records missing. Traversal stops at the next product, source parameter or opaque boundary; recovering requested records does not complete semantic provenance. Signatures retain shapes, attributes, constants and ordered connections while normalizing instruction/computation identifiers and source-wrapper labels. Raw traces preserve caller contexts and repeated positions; recursive signatures do not distinguish sharing.

The CPU-only task succeeded attempt 0/exit 0, UID cbdaa50702c1, with canonical task/job status independently verified at 16:26:57–58 UTC. Published source 4ea554f09f2f7ccc5c07b1fb48585aca7e6b0fff ran on four CPUs/8 GiB/10 GiB in us-west4 with default preemption retries and no wall-clock limit. All four optimized HLO size/SHA checks passed regionally; large HLO stayed there. The 1,676,770-byte report passed exact GCS readback, SHA 9c6c9b75fab8ca0ca64dba66bbfa179c72858ed0fbf106f8cdef6a145d5e2e0f. Fixtures passed for exact target roles, missing/duplicate rejection, operand ordering, wrong-root/semantic-swap rejection, literal mutation and all 28 evaluator comparisons.

Next is static design only: use saved records first to compare the declared Done-to-Start input bindings behind this down route, retaining the paired-capture and passing o controls. Select endpoints by dependencies and exact caller context, preserve repeated tuple projections, and keep custom-call outputs opaque. Any needed regional extraction must publish exact source, manifest and checksums before CPU-only parsing with HLO identity checks. No numerical intervention or optimizer/training/MFU acceptance is selected. No experiment or observer remains; PR monitoring stays handed off after its twelve-hour timeout. The research heartbeat remains active.

[Completion](compiler-frontier-operands-completed.json),
[comparison](compiler-frontier-operands-comparison.json),
[down route records](compiler-frontier-down-route.json) and
[validation](compiler-frontier-operands-validation.json) retain the evidence.

## 2026-11-03 16:42 UTC — Collective input records retain the unresolved down failure

The regional CPU extraction inspected the declared inputs behind three selected operand frontiers: down's fifth-step (cA)A product and o's second/third-step BX products, all at operand position 1. Untouched, evaluator and paired-capture graphs have 26 input routes each: two Done calls with ten inputs and one matched Start input apiece, plus four ConcatBitcast inputs. Substitution has 19 routes because the selected o second-step frontier is ConcatBitcast rather than Done. These different endpoint kinds are retained without pairing their operands.

All 26 evaluator signatures match untouched. Paired capture matches 25; its o second-step Start-input signature differs only in recorded S(1) tags within the inspected tree. Substitution's eleven down signatures differ: nine Done-input routes differ only in S(1) declaration tags, Done input 2 also differs in all-gather channel/chain IDs 360→357 and 57→61, and the Start-input route differs only in those IDs. These are repeated views of shared declarations, not eleven independent changes. The Start's declared input is an all-gather whose producer trace stops at an opaque ConcatBitcast in every graph. The extra copy pair found on the consuming side in the preceding extraction remains recorded separately; this input inspection does not establish its numerical effect.

For each selected Done, all ten declared operand traces reach one common Start identity. Their Start tuple paths are [0], [1], [0], [2], [3], [4], [5], [6], [7], [8] in operand order, retaining repeated leaf 0. This is a declared binding result, not exact Start/Done semantics. Original custom-call outputs remain opaque.

The passing o control has broader changes. Its four third-step ConcatBitcast input signatures all differ: the selected producer tuple projection changes from leaf 0 with bf16 declarations to leaf 1 with f32 declarations, followed by different operand connections. The difference walker stops when those connections cease to align; it does not claim equality beyond that point. Substitution's four second-step ConcatBitcast input routes are reported without comparison to the control Done inputs. Matching endpoint roles or emitted dtype declarations alone does not establish semantic equivalence, runtime precision, or the cause of a numerical failure. Source precision and the numerical test remain unchanged.

Five closure rounds recovered nine bodies per control graph; three rounds recovered three bodies for substitution. No requested records remain missing, but product and opaque frontiers remain. The task succeeded attempt 0/exit 0, UID 1e4029c1c3ef, with canonical job/task status verified at 16:40:31–32 UTC. Published source 657ef87ffd57090cf85ccd2588ddef10d9f45535 ran on four CPUs/8 GiB/10 GiB in us-west4 with default preemption retries and no wall-clock limit. No accelerator or numerical kernels were used. All four HLO SHA/size checks passed regionally; large HLO stayed there. The 1,884,282-byte report passed exact GCS readback, SHA a75d20f108ca32b4c8388aca1fbb847255d492dca7dd212fc0bfca13b2361523. Fixtures cover explicit position ordering, missing/duplicate/wrong-caller roots, common-Start mutations, literal changes, evaluator comparisons and rejection of cross-kind operand alignment.

Both full 28-leaf numerical substitution gates remain failed. Next is static design only: inspect the four declared inputs of the down ConcatBitcast reached from Start's input, using existing records first. None has a saved endpoint-input trace in the same caller context among the seven inspected reports. Reuse already inspected o controls and preserve their different endpoint kinds. Any needed extraction requires published source/manifest/checksums before regional CPU parsing with HLO identity checks. No numerical intervention or optimizer/training/MFU acceptance is selected. No experiment or observer remains; PR monitoring stays handed off after its twelve-hour timeout, and the research heartbeat remains active.

[Completion](compiler-frontier-collectives-completed.json),
[comparison](compiler-frontier-collectives-comparison.json) and
[validation](compiler-frontier-collectives-validation.json) retain the evidence.

## 2026-11-03 16:57 UTC — Down Concat inputs expose a declared slice-order change

The regional CPU extraction recovered the four declared inputs of the down ConcatBitcast reached from the previously matched Start input. Each route follows slice-done/slice-start, producer fusion leaf 0 and a declared f32-to-bf16 conversion to the same fifth-step down X_Xt product within that graph. The existing dependency map identifies X_Xt as X times X transpose. Traversal stops at the product; its operand values and the original ConcatBitcast, Start and Done output semantics remain unestablished.

All four evaluator and all four paired-capture signatures match untouched at the same operand position. All four substitution signatures differ at the same position. Controls present leading-axis slice ranges [0:3], [3:6], [6:9], [9:11]; substitution presents [6:9], [9:11], [0:3], [3:6]. The other two slice axes remain [0:128] and [0:1024]. Positions 1 and 3 consequently exchange declared leading sizes 3 and 2. These slice attributes and associated sizes are the only differences in the bounded normalized input-route comparisons.

A separate check finds a unique mapping of complete substitution route signatures to untouched signatures: operand positions [0,1,2,3] map to [2,3,0,1]. This auxiliary mapping preserves the recorded operand order and does not change the four same-position mismatches. It does not reorder an executable, establish that ConcatBitcast concatenates in that order, or prove a runtime data permutation, precision change or numerical cause. The four positions are slices from one producer, not four independent product changes. Passing o controls retain the broader route changes documented in the preceding extraction.

One closure round recovered one body per graph, with no requested records missing. The task succeeded attempt 0/exit 0, UID e08bb9bd2e99; canonical job/task status was independently verified at 16:54:29 UTC. Published source 976638bcfd0cad58df38b2427e48d6588e808ac7 ran on four CPUs/8 GiB/10 GiB in us-west4 with default preemption retries and no wall-clock limit. It used no accelerator or numerical kernels. All four optimized HLO SHA/size checks passed regionally, and large HLO stayed there. The 181,140-byte report passed exact GCS readback, SHA 1c1881b19ef6be952f4ac4aecbb02b3404f92e3c4614da1605e3ca087d9fde66.

Fixtures passed for explicit operand ordering, missing/duplicate/wrong-caller roots, semantic operand swaps, dtype-declaration mutation, evaluator comparisons, unique signature matching and polar-role mutation. The summary retains each slice record, conversion and exact product identity. Both full 28-leaf numerical substitution gates remain failed; source precision and fixed tolerances remain unchanged.

Next is static design only: inspect the two declared operands of the reached down fifth-step X_Xt product, with corresponding passing up/o products and paired-capture controls. Use saved records and coverage first, select by dependency-mapped product roles and exact caller context, and stop at the next product/source/opaque boundary. Any needed regional extraction must publish source, manifest and checksums before CPU-only HLO identity-checked parsing. No numerical intervention or optimizer/training/MFU acceptance is selected. No experiment or observer remains; PR monitoring stays handed off after its twelve-hour timeout, and the research heartbeat remains active.

[Completion](compiler-down-concat-completed.json),
[comparison](compiler-down-concat-comparison.json),
[slice records and mapping](compiler-down-concat-slices.json) and
[validation](compiler-down-concat-validation.json) retain the evidence.

### November 3, 17:10 UTC — inspect fifth-step Gram operands (static preflight)

The down Concat audit reaches a fifth-step `X_Xt` product but stops before its operands. I selected that product and the corresponding passing up/o controls by dependency-mapped polar roles, then verified their unique entry callers. None of the six immediate operand records per graph appears in the eight saved traces with the same caller context. The down product itself agrees with all four prior Concat input routes.

The [plan](compiler-step5-gram-inputs-plan.json), [manifest](compiler-step5-gram-inputs-manifest.json), and [preflight](compiler-step5-gram-inputs-validation.json) preserve both ordered operands across untouched/evaluator/paired-capture/column-substitution graphs. Missing or duplicate roles, ambiguous callers, and a wrong down role are rejected. Only missing producer bodies will be read from SHA/size-verified HLO on regional CPU in us-west4. No accelerator, numerical run, or precision/tiling intervention is involved. Both full 28-leaf numerical substitution gates remain failed; the slice-signature mapping is not executable permutation semantics or a cause.

### November 3, 17:15 UTC — fifth-step Gram operand routes match within opaque boundaries

The fifth-step `X_Xt` operand audit found no normalized route difference in the six inspected positions: both operands of down, up, and o match the untouched graph in the evaluator, paired capture, and column substitution. This is static agreement only. Every route still reaches an opaque custom-call boundary, so the result does not establish equal runtime inputs or explain the failed numerical gate.

The CPU job `ema-gram-20-step5-gram-inputs-cpu-20261103` succeeded on attempt 0 with exit 0, verified from canonical job/task reads at 17:13 UTC. It ran the published source `f3f6164932c6e579913e5b988b40e5f7a2571af9` on four CPUs in us-west4, with default preemption and no wall-clock limit. All four optimized HLO files were SHA/size verified before regional parsing. One closure round recovered three computation bodies per graph; no requested records remain missing. The 278,520-byte report has SHA256 `3759e105fd024f26422a0481bb5164c72b67485cfeddbdd288dd6320575338f6` and passed exact GCS readback. Large HLO files stayed regional.

The [comparison](compiler-step5-gram-inputs-comparison.json) preserves both operand positions. Here “normalized” means that compiler instruction/computation identifiers, metadata, and source-wrapper labels are removed or canonicalized; declared shapes, attributes, channel/chain IDs, constant literals, and ordered connections remain. Raw traces preserve caller contexts and repeated edges, although recursive signature hashes do not distinguish sharing. Down operand 0 reaches down `B_X` products from iterations 3 and 4 plus an opaque `ConcatBitcast`; down operand 1 reaches opaque `AsyncCollectiveDone`. Up and o operand 0 each reach an opaque Concat, and their operand 1 each reaches an opaque Done. The recorded f32/bf16 declarations are compiler evidence, not a source precision change or proof of runtime precision. The earlier down Concat slice-signature mapping `[2,3,0,1]` remains separate evidence and is not executable permutation semantics.

The [frontier coverage](compiler-step5-gram-inputs-frontiers.json) joins complete prior input-trace roots by exact graph, caller context, computation, instruction, and tuple path. Both earlier down products and all ten declared inputs of the down/o Done endpoints are already covered. None of the four declared inputs of each newly reached down/up/o Concat, or the ten inputs of the up Done, has a matching saved input-trace root. Covered inputs do not make opaque outputs transparent. [Validation](compiler-step5-gram-inputs-validation.json) checks explicit positions, missing/duplicate/wrong-caller rejection, product roles, and declaration mutations.

Both full 28-leaf numerical substitution gates remain failed. Matching these bounded signatures does not establish semantic identity, values, reduction order, or cause. No numerical, precision, tiling, training, MFU, or restore claim is accepted. The next static decision is to inspect the twelve uncovered declared Concat inputs with paired-capture and passing up/o controls, reusing covered product/Done inputs. The uncovered up Done remains recorded for later coverage review. Any needed extraction still requires exact published source and regional CPU verification.

The follow observer ended naturally with exit 0. No experiment, queue, or observer remains active. PR monitoring stays handed off after its earlier 12-hour timeout; the research heartbeat remains active.

### November 3, 17:22 UTC — inspect uncovered fifth-step Concat inputs (static preflight)

The [previous coverage audit](compiler-step5-gram-inputs-frontiers.json) identifies twelve uninspected declared inputs behind the down/up/o Concat endpoints reached by fifth-step `X_Xt` operand 0. The [new selector and manifest](compiler-step5-concat-inputs-manifest.json) use those saved dependencies, check the operand root and caller context, and reject missing or ambiguous boundaries. Saved records resolve part of the traces; two computation bodies per graph remain missing at preflight.

The [plan](compiler-step5-concat-inputs-plan.json) and [validation](compiler-step5-concat-inputs-validation.json) preserve all four operand positions per group and the paired-capture and passing up/o controls. Exact source will be published before one CPU-only us-west4 extraction with HLO SHA/size checks, default preemption, and no wall-clock limit. This is static parsing, with no accelerator, numerical kernels, or W&B run. Both numerical full28leaf gates remain failed; opaque output semantics remain unresolved.

### November 3, 17:26 UTC — fifth-step Concat inputs reach earlier polar products

The twelve declared Concat input routes all match the untouched graph in the evaluator and paired capture. In column substitution, all eight down/up routes match, while the four o routes differ at the same copy-done versus tuple-projection record. The comparison stops at that opcode mismatch; it does not establish equality of the remaining subtrees. The four mismatches are repeated views of one changed record, and o remains a passing numerical control. This is not an explanation of the failing down direction.

The CPU job `ema-gram-20-step5-concat-inputs-cpu-20261103` succeeded on attempt 0 with exit 0, verified by canonical job/task reads at 17:24 UTC. It ran published source `88c92932779943bc58ac94485f6342856b817371` with four CPUs in us-west4, default preemption, and no wall-clock limit. All four HLO hashes/sizes were verified before parsing; one closure round recovered two computation bodies per graph. No requested records remain missing. The 1,198,574-byte report has SHA256 `b81cd2490ba9bbe4c9e00a9130388901bfeca4c2bd5cf37d53e5c6a70840a63f` and passed exact GCS readback. Large HLO stayed regional; the task used no accelerator or numerical kernels.

The [comparison](compiler-step5-concat-inputs-comparison.json) preserves all four declared input positions for each group. Each down route reaches down iteration-2 `B_X` and another opaque Concat; each up route reaches up iteration-4 `B_X` and another opaque Concat. Each o route reaches o iteration-2/3/4 `B_X` products plus another opaque Concat. These roles are checked against the dependency-mapped polar lineage, where `B_X` denotes `B @ X`; traversal stops at each product record before following its operands. Four routes per group share these producers, rather than establishing four independent product changes.

“Matching normalized signatures” means compiler identifiers, metadata, and source-wrapper labels have been removed or canonicalized while shapes, attributes, channel/chain IDs, literals, and ordered connections remain. Raw traces retain caller contexts and repeated edges; recursive hashes do not distinguish sharing. All original Concat/Start/Done/while outputs remain opaque. These static records do not establish runtime values, precision, reduction order, semantic identity, or cause. Both full 28-leaf numerical substitution gates remain failed, with no optimization, training, MFU, or restore acceptance.

The [coverage summary](compiler-step5-concat-inputs-frontiers.json) finds no complete saved input-trace roots for the two operands of down iteration-2 `B_X` or up iteration-4 `B_X`, using exact caller/computation/instruction/tuple-path identities. The corresponding o iteration-2/3/4 product inputs are already covered. Newly reached down/up Concat inputs remain uncovered; the new o Concat inputs are covered only in substitution, not in the three control graphs. The previously recorded up Done gap also remains. [Validation](compiler-step5-concat-inputs-validation.json) checks positions, ambiguous selection, wrong callers, product roles, declaration mutations, and the shared identity of the four o mismatch records.

The next static decision is to inspect the four uncovered input positions of down iteration-2 and up iteration-4 `B_X`, reusing the covered o product controls and retaining paired capture. Other uncovered Concat/Done inputs remain recorded, not silently treated as resolved. Any needed extraction requires published exact source and regional CPU HLO verification. No new numerical or precision/tiling intervention is selected.

The sole follow observer ended naturally with exit 0. No experiment, queue, or observer remains active. PR monitoring remains handed off after its earlier 12-hour timeout; the research heartbeat stays active.

### November 3, 17:37 UTC — inspect earlier B@X operands (static preflight)

The [latest frontier coverage](compiler-step5-concat-inputs-frontiers.json) leaves four immediate operands uninspected: down iteration-2 `B_X` and up iteration-4 `B_X`. The [new selector](trace_earlier_bx_inputs.py) requires all four saved Concat routes to agree on each product's exact caller context and declaration, and cross-checks the dependency-mapped group/iteration/product. It rejects missing or duplicate boundaries, caller disagreement, duplicate positions, and wrong roles.

The [plan](compiler-earlier-bx-inputs-plan.json), [manifest](compiler-earlier-bx-inputs-manifest.json), and [validation](compiler-earlier-bx-inputs-validation.json) reuse eleven bounded inputs including inventory. Two scoped bodies per graph must be read again to recover operand records absent from the saved traces; already-covered o product controls will be reused without extraction. Publish exact source before one CPU-only regional us-west4 extraction, with HLO SHA/size checks, default preemption and no wall-clock limit. Both full28leaf numerical gates remain failed, and opaque outputs remain opaque.

### November 3, 17:42 UTC — earlier B@X operand routes match within opaque boundaries

All four newly inspected operand-route signatures match the untouched graph in evaluator, paired capture, and column substitution: two operands of down iteration-2 `B_X` and two of up iteration-4 `B_X`. Each operand-0 route reaches the corresponding `scaled_A_A` product and an opaque Concat; each operand-1 route reaches an opaque Done. Traversal stops at product records before following their operands. These bounded matches do not establish equal runtime inputs or explain the failed numerical gate.

The CPU job `ema-gram-20-earlier-bx-inputs-cpu-20261103` succeeded on attempt 0 with exit 0, verified by canonical job/task reads at 17:40 UTC. It ran published source `77d6e3182e614b2099bbbdc2e782eb4019d11368` on four CPUs in us-west4, with default preemption and no wall-clock limit. Two closure rounds recovered six computation bodies per graph after all four HLO hashes/sizes were verified. No requested records remain missing. The 234,859-byte report has SHA256 `9c1456d09d0387dc5ea0758d4ea807e1409fa331e2ceb15a9662d1e26761772f` and passed exact GCS readback. Large HLO stayed regional; no accelerator or numerical kernels were used.

The [comparison](compiler-earlier-bx-inputs-comparison.json) retains both explicit operand positions and the raw caller contexts. “Normalized” removes or canonicalizes compiler identifiers, metadata, and source-wrapper labels; shapes, attributes, channel/chain IDs, literals, and ordered connections remain. Recursive hashes do not distinguish sharing; raw repeated edges remain available. Product roles are checked against dependency-mapped polar lineage, where `scaled_A_A` denotes `(cA)A`. All original Concat/Start/Done/while outputs remain opaque. The records establish neither runtime precision/order/values nor semantic identity or cause. Both full28leaf numerical substitution gates remain failed.

The [reused o controls](compiler-earlier-bx-inputs-reused-controls.json) come from the prior frontier-operands report, with exact report SHA/size and same-HLO checks. Its six o iteration-2/3/4 input routes retain all prior evidence: six evaluator and six paired-capture signatures match; three substitution signatures match and three differ. Those are existing static control observations, not new extractions or numerical measurements. O remains a passing numerical control in the earlier numerical report.

The [frontier coverage](compiler-earlier-bx-inputs-frontiers.json) identifies 32 declared input positions without complete saved input-trace roots per graph: four product operands, eight Concat inputs, and twenty Done inputs across the two selected groups. This count preserves repeated operand positions and is not a count of unique values. It describes new frontiers, not missing records from the requested extraction. [Validation](compiler-earlier-bx-inputs-validation.json) covers ambiguous selection, explicit ordering, wrong caller/role rejection, declaration mutation, and rejection of wrong report/HLO identities for reused controls.

The next static decision is a combined binding audit of those 32 uncovered positions, using current and saved records first. Match each Done's declared inputs to a common Start identity only where unambiguous, and record that Start's declared input separately; preserve custom-call output opacity. Retain paired capture and reuse already-covered o controls. Other earlier Concat/up-Done gaps remain recorded. Any needed extraction still requires exact published source before regional CPU parsing; no numerical, precision, or tiling intervention is selected.

The sole follow observer ended naturally with exit 0. No experiment, queue, or observer remains active. PR monitoring remains handed off after its earlier 12-hour timeout; the research heartbeat stays active. No optimizer, training, MFU, or restore acceptance is claimed.

### November 3, 17:53 UTC — inspect earlier product and collective inputs (static preflight)

The [previous frontier coverage](compiler-earlier-bx-inputs-frontiers.json) identifies 32 uninspected positions behind down iteration-2 and up iteration-4 `B_X`: four `scaled_A_A` product operands, eight Concat inputs, and twenty Done inputs. The [selector](trace_earlier_bindings.py) checks their saved output dependencies, exact caller contexts, declared arities, and mapped polar roles. All 32 positions lack complete input-trace roots in the eleven prior reports. Six computation bodies per graph remain missing from the local record set at preflight.

The [plan](compiler-earlier-bindings-plan.json) and [manifest](compiler-earlier-bindings-manifest.json) preserve explicit repeated positions and paired capture. Each Done's ten inputs must reach one common Start identity before its single declared input is traced separately. Original custom-call outputs remain opaque. Already inspected o controls will be reused with the same HLO identity checks. Publish exact source before CPU-only us-west4 parsing with HLO SHA/size verification, default preemption and no wall-clock limit. No numerical intervention is selected; both full 28-leaf gates remain failed.

### November 3, 17:57 UTC — earlier input bindings differ only in recorded collective IDs

The combined audit recovered 32 declared input positions behind down iteration-2 and up iteration-4 `B_X` (`B @ X`), plus two auxiliary Start inputs. All 34 evaluator and paired-capture signatures match the untouched graph. Column substitution matches 30. Its four differences are confined to all-gather channel/chain IDs: down changes 340→337 and 34→38; up changes 308→305 and 46→50. For each group these differences occur in Done operand 2 and the matched Start's single input. Those are distinct all-gather records sharing IDs, not four independent changes or evidence of execution order. Up remains a passing numerical control.

Each Done's ten declared inputs reaches one common Start identity, with tuple paths [0], [1], [0], [2], [3], [4], [5], [6], [7], [8] in operand order. Repeated leaf 0 is retained. This establishes a declared binding only; Start and Done outputs remain opaque. Each Start input trace reaches an earlier `B_X` product (down iteration 1 or up iteration 3) and an opaque Concat. Each group's four Concat input traces reach the same current-iteration `X_Xt` (`X @ X.T`) product. With `A = X @ X.T`, `scaled_A_A` denotes `(cA) @ A`; its operands stop at opaque Concat and Done boundaries. Product roles come from dependency-mapped polar lineage; tracing stops at product records before following their operands.

The CPU job `ema-gram-20-earlier-bindings-cpu-20261103` succeeded on attempt 0 with exit 0, UID `a52082d3ebf5`, with canonical job/task status verified at 17:55 UTC. Published source `4f0224fba79803df5118c6823a6e3645cd033083` ran on four CPUs in us-west4 with default preemption retries and no wall-clock limit. All four HLO hashes/sizes were checked before parsing. Five closure rounds recovered fifteen computation bodies per graph; no requested records remain missing. The 1,839,091-byte report has SHA256 `083d6a087ddd67da206994cacf846c29a20f6b2fca81d5e769288989405d6aea` and passed exact GCS readback. Large HLO stayed regional; no accelerator or numerical kernels were used.

The [comparison](compiler-earlier-bindings-comparison.json) retains shapes, attributes, literals, channel/chain IDs and ordered connections after removing or canonicalizing compiler identifiers, metadata and source wrappers. Raw traces retain caller contexts and repeated edges; recursive hashes do not distinguish sharing. All original custom-call and while outputs remain opaque. Matching bounded signatures does not establish equal runtime inputs, precision, reduction order, semantic identity or cause. Both full 28-leaf numerical substitution gates remain failed.

The [coverage summary](compiler-earlier-bindings-frontiers.json) joins complete input-trace roots from the current and eleven prior reports by exact graph, caller, computation, instruction and tuple path. It identifies 36 further uninspected declared positions per graph: eight operands across four products, eight inputs across two Concat endpoints, and twenty inputs across two Done endpoints. The products are down iteration-2 `X_Xt`/iteration-1 `B_X` and up iteration-4 `X_Xt`/iteration-3 `B_X`. These are deliberate trace frontiers, not missing requested records. Inputs of the two newly matched Starts and two other encountered Concat endpoints are already covered. Covered inputs do not resolve opaque output semantics.

[Validation](compiler-earlier-bindings-validation.json) covers explicit position ordering, duplicate/missing/wrong-root rejection, semantic operand swaps, dtype-declaration mutation, common-Start identity and tuple-projection mutations. Six [reused o controls](compiler-earlier-bindings-reused-controls.json) retain their original same-HLO observations: all evaluator/capture routes match and three substitution routes differ. No control was re-extracted or numerically rerun.

The next static decision is to inspect the 36 uncovered positions together, using saved records first and retaining paired capture and passing controls. Select exact caller/output dependencies, preserve repeated positions and tuple paths, and inspect any newly matched Start input separately. Reuse covered inputs and keep opaque outputs opaque. Any required regional extraction needs published exact source/manifest/checksums before CPU-only HLO parsing. No numerical, precision, or tiling intervention is selected, and no optimizer, training, MFU or restore acceptance is claimed.

The sole follow observer ended naturally with exit 0. No experiment, queue or observer remains active. PR monitoring remains handed off after its earlier twelve-hour timeout; the research heartbeat stays active.

### November 3, 18:07 UTC — inspect upstream product and collective inputs (static preflight)

The [current coverage](compiler-earlier-bindings-frontiers.json) leaves 36 declared positions uninspected: eight operands of down iteration-2 `X_Xt`/iteration-1 `B_X` and up iteration-4 `X_Xt`/iteration-3 `B_X`, eight Concat inputs, and twenty Done inputs. The [selector](trace_upstream_bindings.py) preserves exact saved caller dependencies, deduplicates repeated views, and checks product roles. Complete input-trace roots from twelve saved reports exclude already covered endpoints. A partially covered input set is rejected for explicit position-level review.

The [plan](compiler-upstream-bindings-plan.json), [manifest](compiler-upstream-bindings-manifest.json) and [preflight](compiler-upstream-bindings-validation.json) preserve paired capture and reused o controls. Match each Done's ten inputs to one Start only where unambiguous, then inspect its declared input separately. Exact source will be published before regional CPU-only parsing with HLO SHA/size checks, default preemption and no wall-clock limit. No numerical intervention is selected; both full28leaf gates remain failed and opaque outputs remain opaque.

### November 3, 18:12 UTC — upstream routes reach down normalization's combined reduction

The upstream audit recovered 36 declared input positions plus two auxiliary Start inputs. All 38 evaluator and paired-capture route signatures match untouched. Substitution matches 30. Four differences concern all-gather channel/chain IDs: down changes 342→339 and 38→42; passing up changes 310→307 and 49→53. Each pair appears in distinct Done-operand-2 and Start-input records. The other four differences are shared views of down normalization's selected combined all-reduce projection: control leaf 5 becomes substitution leaf 6, the tuple declaration changes, and channel 144 becomes 134. The original comparison stops when the selected operand positions cease to align and makes no claim about equality beyond that point.

A [separate selected-input comparison](compiler-down-norm-projection.json) follows the one declared input at each observed all-reduce tuple projection. Its bounded producer-subtree signatures match in all four graphs. This comparison preserves the original four route mismatches; it does not redefine tuple positions or collective semantics. All four down Concat routes share one exact all-reduce projection per graph. The producer subtree records elementwise multiplication of a product output by itself, followed by summation over dimensions 1 and 2 for the declared normalization denominator. It stops at that nonpolar `gram_direction` product before its operands. Its presence and matching declarations do not establish equal runtime values, reduction order, precision or a numerical cause.

Each Done's ten inputs reaches one common Start identity with tuple paths [0], [1], [0], [2], [3], [4], [5], [6], [7], [8]; repeated leaf 0 remains explicit. Start/Done/Concat/while outputs remain opaque. Source polar roles use `A = X @ X.T`, `B_X = B @ X`, and `scaled_A_A = (cA) @ A`. Normalized signatures retain shapes, attributes, literals, channel/chain IDs and ordered connections after removing or canonicalizing compiler identifiers, metadata and source wrappers. Raw caller contexts and repeated edges remain available; recursive hashes do not distinguish sharing.

The CPU job `ema-gram-20-upstream-bindings-cpu-20261103` succeeded on attempt 0 with exit 0, UID `d2da0ac904b5`, verified from canonical job/task reads at 18:10 UTC. Published source `d3140912cd415d3e4168a80381f4faf5f6bba19e` ran on four CPUs in us-west4 with default preemption retries and no wall-clock limit. All four HLO hashes/sizes were checked before parsing. Four closure rounds recovered sixteen bodies per graph, with no requested records missing. The 2,484,911-byte report has SHA256 `247bbe8afcb3265f027caf712f719184547ea90f6b5999b043aa29a7fe960fca` and passed exact GCS readback. Large HLO stayed regional; no accelerator or numerical kernels ran.

The [coverage summary](compiler-upstream-bindings-frontiers.json) finds 48 further uninspected positions per graph: eight operands across four products, twenty inputs across five Concat endpoints, and twenty inputs across two Done endpoints. The products include down iteration-1 `scaled_A_A`, up iteration-3 `scaled_A_A`, up iteration-2 `B_X`, and the nonpolar down product reached through normalization. These are deliberate trace frontiers, not missing requested records. Coverage joins complete current and twelve prior input-trace roots by exact caller/computation/instruction/tuple-path identities. Covered inputs leave opaque output semantics unresolved.

[Validation](compiler-upstream-bindings-validation.json) covers missing-product/wrong-role/partial-coverage rejection, explicit position ordering, missing/duplicate/wrong-root and semantic-swap rejection, dtype-declaration mutation, common-Start identity and tuple-projection mutations, and the separate selected-input comparison. Six [reused o controls](compiler-upstream-bindings-reused-controls.json) retain their existing observations and three substitution mismatches; no new control extraction or numerical run occurred.

The next static decision is to inspect the 48 uncovered positions together using saved records first. Identify the nonpolar down product through its actual normalization dependency and source scope; do not assign it a polar role from names or order. Preserve exact callers, repeated positions, tuple projections, paired capture and passing controls. Inspect newly matched Start inputs separately, reuse covered inputs, and keep all opaque outputs opaque. Publish exact source/manifest/checksums before any needed regional CPU extraction. No numerical, precision or tiling intervention is selected. Both full 28-leaf numerical gates remain failed; no optimizer, training, MFU, restore or causal acceptance is claimed.

The follow observer ended naturally with exit 0. No experiment, queue or observer remains active. PR monitoring remains handed off after its earlier timeout; the research heartbeat stays active.

### November 3, 18:34 UTC — inspect normalization and earlier polar inputs (static preflight)

The [upstream coverage](compiler-upstream-bindings-frontiers.json) leaves 48 declared input positions uninspected. The [new selector](trace_norm_polar_inputs.py) uses exact saved dependencies for three polar products, the nonpolar down product reached through normalization, five Concat endpoints and two Done endpoints. It identifies the nonpolar product by the selected normalization reduction and repeated multiplication of its output, and checks the polar products against mapped source roles. Multiple Concat endpoints are distinguished by explicit parent operand routes, not compiler ordinals.

The [plan](compiler-norm-polar-inputs-plan.json), [manifest](compiler-norm-polar-inputs-manifest.json) and [preflight](compiler-norm-polar-inputs-validation.json) verify fourteen bounded input identities and exclude covered input roots. Eight computation bodies per graph remain missing locally. Publish exact source before CPU-only us-west4 extraction with HLO SHA/size checks, default preemption and no wall-clock limit. Done-to-Start matching remains a declared binding, and every original opaque output remains opaque. Both numerical gates still fail; no numerical intervention is selected.

### November 3, 18:39 UTC — normalization inputs reach right-Gram producer declarations

The audit recovered 48 declared input positions plus two auxiliary Start inputs. All 50 evaluator and paired-capture route signatures match the untouched graph. Column substitution matches 45. The second operand of down normalization's nonpolar `gram_direction` product reaches the declared right-Gram state (`states[6].right`, with corresponding bank wrappers), a Gram-refresh product and a `relative_gram_power` product. Its only bounded signature difference is all-reduce channel 329→326. The first operand reaches another direct `gram_direction` product. Traversal stops at each product before inspecting its operands, so these declarations do not establish the product's runtime inputs or values.

The other four differing routes are Done operand 2 and the matched Start input for down iteration-1 and up iteration-3 `B_X` (`B @ X`). Both groups change all-gather channel/chain IDs: down 334→331 and 24→28; up 302→299 and 40→44. The down Start input also traverses the same exact normalization all-reduce record inspected previously, including the control leaf-5 versus substitution leaf-6 projection. The comparison stops where selected edge positions no longer align. This preserves the mismatch without treating another view of the same record as an independent change. Up remains a passing numerical control.

Each Done's ten inputs reaches one common Start identity with tuple paths [0], [1], [0], [2], [3], [4], [5], [6], [7], [8]. Repeated leaf 0 remains explicit. Done operand 2 and the Start input are distinct all-gather records sharing IDs. These are declared bindings only; original Start/Done/Concat/while outputs remain opaque. Normalized signatures remove or canonicalize compiler identifiers, metadata and source wrappers while retaining shapes, attributes, literals, channel/chain IDs and ordered connections. Raw caller contexts and repeated edges remain available; recursive hashes do not distinguish sharing.

The CPU job `ema-gram-20-norm-polar-inputs-cpu-20261103` succeeded on attempt 0 with exit 0, UID `ac3bd97c77b6`, verified by canonical job/task reads at 18:37 UTC. Published source `4ab92f0d91d601489a6212a642ce829f8c7ff297` ran on four CPUs in us-west4 with default preemption and no wall-clock limit. All four HLO hashes/sizes were verified before parsing. Four closure rounds recovered seventeen bodies per graph; no requested records remain missing. The 3,167,377-byte report has SHA256 `7753d1d505e2df7ee084992000b3ed0fc24c7ecf5f7d85618b564ffc9a061cce` and passed exact GCS readback. Large HLO stayed regional; no accelerator, numerical kernels or W&B run was used.

The [coverage summary](compiler-norm-polar-inputs-frontiers.json) retains every record and repeated position in compact JSON. Current and thirteen saved reports leave 52 further uninspected positions per graph: fourteen operands of seven products, eight Concat inputs and thirty Done inputs. Four products have mapped polar roles: down iteration-1 `X_Xt`, up iteration-1 `B_X`, up iteration-2 `scaled_A_A`, and up iteration-3 `X_Xt`. Here `X_Xt` denotes `X @ X.T`, and with `A = X @ X.T`, `scaled_A_A` denotes `(cA) @ A`. The three other products lie on the down direct-direction, Gram-refresh and relative-power routes above. These are deliberate frontiers; covered input roots do not resolve opaque output semantics. Six [reused o controls](compiler-norm-polar-inputs-reused-controls.json) retain their previous same-HLO observations, with three substitution mismatches and no new numerical measurement.

The [comparison](compiler-norm-polar-inputs-comparison.json) and [validation](compiler-norm-polar-inputs-validation.json) preserve explicit positions, source roles and caller identities. The next static decision is to audit the 52 uncovered positions using saved records first, selecting the three nonpolar products through their actual output dependencies and scopes. Reuse covered inputs and passing controls, match Done to Start only where unambiguous, and keep opaque outputs opaque. Exact source must be published before any regional CPU extraction. Both full 28-leaf numerical gates remain failed. No runtime precision/order/identity/cause, optimization, training, MFU or restore acceptance is established.

The sole follow observer ended naturally with exit 0. No experiment, queue or observer remains active. PR monitoring remains handed off after its completed twelve-hour wait; the research heartbeat stays active.

### November 3, 18:43 UTC — inspect Gram-source and earlier polar inputs (static preflight)

The [current frontier coverage](compiler-norm-polar-inputs-frontiers.json) leaves 52 declared positions uninspected. The [selector](trace_gram_source_inputs.py) identifies three nonpolar down products by their actual normalization-operand dependencies and source scopes: direct direction, Gram refresh and relative power. Four other products are checked against the mapped polar lineage. Two Concat and three Done endpoints retain explicit parent operand labels, caller contexts and repeated positions.

The [plan](compiler-gram-source-inputs-plan.json), [manifest](compiler-gram-source-inputs-manifest.json) and [preflight](compiler-gram-source-inputs-validation.json) verify fifteen bounded input reports and exclude already covered input roots. Eight computation bodies per graph remain missing locally. Exact source will be published before one CPU-only us-west4 extraction with HLO SHA/size checks, default preemption and no wall-clock limit. Match each Done to a Start only where unambiguous and trace its single declared input separately. All opaque outputs remain opaque; both numerical gates remain failed and no numerical intervention is selected.

### November 3, 18:48 UTC — Gram-refresh operands reach the declared gradient source

The Gram-source audit recovered 52 declared input positions plus three auxiliary Start inputs. All 55 evaluator routes match the untouched graph. Paired capture matches 54: its remaining difference is confined to `S(1)` tags on sibling leaves 3 and 4 of a combined all-reduce tuple; the route selects unchanged leaf 5. Column substitution matches 42 of 55. The passing capture mismatch remains part of the evidence and does not explain the failing numerical direction.

Both operands of the down right-Gram refresh product reach the same declared gradient parameter within each graph (`gradients[6]`, with corresponding bank wrappers). Their normalized route signatures match across all four graphs. This establishes a declared input binding, without proving equal runtime arrays across graphs. The direct-direction product's first operand reaches left-Gram state/refresh/relative-power declarations in the controls; substitution retains an opaque Concat instead of reaching that state parameter. Its second operand reaches momentum state and Concat in the controls, but stops at an opaque Done in substitution. The right relative-power product's inputs reach opaque Done and while boundaries. Product traversal still stops before newly encountered operands.

Substitution's thirteen route mismatches comprise the two direct-direction inputs, one right relative-power input, six Done/Start collective routes and four shared up normalization routes. Direct-input differences include recorded tags/IDs and unaligned producer connections: the difference walker stops at each opcode or edge mismatch and does not assert equality beyond it. The simple collective routes differ only in recorded channel/chain IDs. The four up routes share one combined all-reduce projection: control leaf 4 becomes substitution leaf 5, with channel 144→134 and a changed tuple declaration. Up remains a passing numerical control. Repeated views, compiler tags and IDs do not establish runtime precision, execution order or cause.

Each of the three selected Done endpoints has ten declared inputs reaching one common Start identity, with tuple paths [0], [1], [0], [2], [3], [4], [5], [6], [7], [8]. Done operand 2 and the matched Start input are distinct all-gather records sharing IDs. Original Start/Done/Concat/while outputs remain opaque. Normalized signatures retain shapes, attributes, literals, channel/chain IDs and ordered connections after canonicalizing identifiers, metadata and source wrappers. Raw contexts and repeated edges remain available; recursive hashes do not distinguish sharing.

The CPU job `ema-gram-20-gram-source-inputs-cpu-20261103` succeeded on attempt 0 with exit 0, UID `0daf4c28c4a4`, verified by canonical job/task reads at 18:46 UTC. Published source `95dc5c31203586bfa8f190d5a1701ab89e1085b8` ran on four CPUs in us-west4 with default preemption and no wall-clock limit. All four HLO hashes/sizes were checked before parsing. Five closure rounds recovered twenty-one computation bodies per graph; no requested records remain missing. The 4,462,509-byte report has SHA256 `c58edc9ed50ab3b785b003f08869e56443e4d0c9a27e7f214e7406b947622253` and passed exact GCS readback. Large HLO stayed regional; no accelerator or numerical kernels were used.

The [coverage summary](compiler-gram-source-inputs-frontiers.json) uses current and fourteen saved reports. It leaves 35 further declared input positions per graph: ten operands across five products, four Concat inputs, twenty Done inputs and one while initializer. The polar products are up iteration-1 `scaled_A_A = (cA) @ A` and iteration-2 `X_Xt = X @ X.T`, where `A = X @ X.T`. The other products lie on the down Gram-refresh/relative-power and up direct-direction routes. There are 51 unique context/projection frontiers in each control and 49 in substitution; these counts do not establish cross-graph identity. All 27 encountered Start projections have covered declared inputs. Coverage remains separate from output semantics.

The [source-route supplement](compiler-gram-source-inputs-source-routes.json), [comparison](compiler-gram-source-inputs-comparison.json) and [validation](compiler-gram-source-inputs-validation.json) retain the passing-control mismatch and reject wrong source roles, callers, tuple projections and selected-leaf mutations. Six existing same-HLO o controls are reused with their three prior substitution mismatches; no numerical measurement was repeated.

The next static decision is to inspect the 35 uncovered positions using saved dependencies first, including only the while's declared initializer while preserving the while output as opaque. Identify nonpolar products by source scopes and exact output dependencies, retain paired capture and passing controls, and reuse covered inputs. Publish exact source before any regional CPU extraction. Both full 28-leaf numerical gates remain failed, with no runtime identity/cause, optimization, training, MFU or restore acceptance.

The follow observer ended naturally with exit 0. No experiment, queue or observer remains active. PR monitoring stays handed off after its completed twelve-hour wait; the research heartbeat remains active.

### November 3, 18:59 UTC — inspect Gram-state and while-initializer inputs (static preflight)

The [current coverage](compiler-gram-source-inputs-frontiers.json) leaves 35 declared positions uninspected: ten product operands, four Concat inputs, twenty Done inputs and one while initializer. The [selector](trace_state_initializer.py) identifies the three nonpolar products by exact saved dependencies and source scopes, checks two polar roles, and preserves caller contexts and ordered repeated positions. Six computation bodies per graph remain missing locally.

The existing tracer rejects whole tuples without a selected leaf. A [scoped initializer tracer](trace_initializer_tuple.py) follows the declared whole tuple's constituents, preserving repeated edges and stopping at products, entry parameters and opaque operations. Fixtures check nested tuples, repeated sharing, unchanged selected-leaf behavior and opacity of unselected collective outputs. The while output projection `[1]` stays opaque; no loop body or recurrence is interpreted.

The [plan](compiler-state-initializer-plan.json), [manifest](compiler-state-initializer-manifest.json) and [preflight](compiler-state-initializer-validation.json) verify sixteen bounded inputs. Publish exact source before CPU-only us-west4 extraction with regional HLO SHA/size verification, default preemption and no wall-clock limit. Reuse covered inputs and o controls, retain paired capture, and inspect newly matched Start inputs separately. Both numerical gates remain failed; no numerical intervention is selected.

### November 3, 19:05 UTC — a whole initializer reaches right-Gram state and an earlier opaque loop

The audit of the four matrix-path graphs (untouched, evaluator, paired capture and column substitution) recovered 35 declared input positions plus two auxiliary Start inputs. All 37 evaluator and paired-capture signatures match the untouched graph; column substitution matches 21. Both full 28-leaf numerical substitution gates remain failed. Matching bounded declarations does not establish equal runtime inputs, precision, order, semantic identity or cause.

The newly inspected while initializer is a tuple of thirteen declared inputs. Positions 5 and 7 reference the same constant record. Position 3 reaches right-Gram state (`states[6].right`, with bank wrappers in the other graphs) and a Gram-refresh product. Positions 6 and 12 reach the same earlier opaque while output at projection `[6]`. The initializer route differs in substitution only in recorded all-reduce channel 329→326. The original while output projection `[1]` stays opaque. No loop body or recurrence was interpreted, and traversal stops at product records before their operands.

Substitution's sixteen mismatching routes comprise two down relative-power inputs, the whole initializer, eleven views of one up Done/Start group and two views of another up Done/Start group. Forty changed shape fields across the first group's ten Done-input routes differ only by `S(1)` tags; these repeated views are not forty independent changes or runtime precision evidence. Its Start input also reaches the previously observed up normalization projection, control leaf 4/channel 144 versus substitution leaf 5/channel 134. The comparison stops at that unaligned edge. Other bounded differences are collective IDs. Up remains a passing numerical control.

Each Done's ten declared inputs reaches one common Start identity through tuple paths [0], [1], [0], [2], [3], [4], [5], [6], [7], [8]. Done operand 2 and Start input are distinct all-gather records sharing IDs. All original Start/Done/Concat/while outputs remain opaque. Normalized signatures retain shapes, attributes, literals, channel/chain IDs and ordered connections after canonicalizing identifiers, metadata and source wrappers. Raw caller contexts and repeated edges remain available; recursive hashes do not distinguish sharing.

The CPU job `ema-gram-20-state-initializer-cpu-20261103` succeeded on attempt 0 with exit 0, UID `7956a0cbcd23`, verified by canonical job/task reads at 19:02 UTC. Published source `c0747b30d15903277b5411a626b1fddae5f2921d` ran on four CPUs in us-west4 with default preemption and no wall-clock limit. All four HLO hashes/sizes were checked regionally. Four closure rounds recovered seventeen computation bodies per graph; no requested records remain missing. The 2,825,072-byte report has SHA256 `8f367ffadca75acf0d564707dfcaad361964df31ee4eef98ccd5c5e62ccc45ab` and passed exact GCS readback. Large HLO stayed regional; no accelerator or numerical kernels were used.

The [coverage summary](compiler-state-initializer-frontiers.json) uses current and fifteen prior reports. It leaves fourteen uninspected input positions per graph: two operands of up iteration-1 `X_Xt = X @ X.T`, ten inputs of a Done reached from up iteration-1 `scaled_A_A = (cA) @ A`, and two while initializers. The remaining `[1]` frontier is a different while on the down left-relative-power route; it is not the right-relative-power while whose initializer was just inspected. The `[6]` frontier is the earlier while reached from that right-relative-power initializer. Their exact caller/instruction identities remain in the coverage summary, and both outputs stay opaque. All eighteen encountered Start projections have covered inputs. The thirty-four unique context/projection frontiers per graph are coverage counts, not evidence of cross-graph identity. Earlier gaps remain recorded, and input coverage does not resolve opaque output semantics.

The [initializer supplement](compiler-state-initializer-sources.json), [comparison](compiler-state-initializer-comparison.json) and [validation](compiler-state-initializer-validation.json) preserve all mismatches and reject wrong callers, roles, tuple positions and repeated-edge changes. Six same-HLO o control routes are reused with their three prior substitution mismatches; no control extraction or numerical measurement was repeated.

The next static decision is to inspect those fourteen uncovered positions from saved dependencies first, including only the declared while initializers. Retain paired capture, passing controls, ordered repeated positions and opaque outputs; reuse covered inputs. Publish exact source before any regional CPU extraction. No numerical, precision or tiling intervention is selected, and no optimizer, training, MFU or restore acceptance is claimed.

The sole follow observer ended naturally with exit 0. No experiment, queue or observer remains active. PR monitoring remains handed off after its completed twelve-hour wait; the research heartbeat stays active.

### November 3, 19:13 UTC — inspect first-Gram and two spectral initializers (static preflight)

The [current coverage](compiler-state-initializer-frontiers.json) leaves fourteen declared input positions uninspected: up iteration-1 `X_Xt` operands, ten up Done inputs and two separate while initializers. The [selector](trace_polar_initializers.py) uses exact saved output dependencies, checks the polar product role and preserves the while projections `[1]` and `[6]`. It rejects missing, duplicate, wrong-arity, wrong-projection and already-covered targets. Four computation bodies per graph are missing locally.

The [plan](compiler-polar-initializers-plan.json), [manifest](compiler-polar-initializers-manifest.json) and [preflight](compiler-polar-initializers-validation.json) verify seventeen bounded inputs. The existing whole-tuple initializer tracer is reused without modification; loop bodies and recurrence remain uninterpreted. Publish exact source before CPU-only us-west4 extraction with regional HLO SHA/size verification, default preemption and no wall-clock limit. Preserve paired capture and passing controls, reuse covered inputs and keep original outputs opaque. Both full numerical gates remain failed; no numerical intervention is selected.

### November 3, 19:20 UTC — two declared initializers leave one Concat input gap

The audit of four matrix-path graphs (untouched, evaluator, paired capture and column substitution) recovered fourteen declared input positions plus one auxiliary Start input. Bounded signatures match the untouched graph in all fifteen evaluator routes, fourteen paired-capture routes and eleven substitution routes. Both full 28-leaf numerical substitution gates remain failed. These static comparisons do not establish runtime input equality, precision, order, semantic identity or cause.

Both newly inspected while initializers contain twelve tuple constituents. Positions 5 and 7 reference the same constant record; only position 3 reaches a product, source or opaque boundary. The initializer reached from down left-relative-power operand 0 reaches `states[6].left` and a Gram-refresh product in the controls. Substitution instead reaches opaque Concat boundaries and the refresh product. The other initializer is reached through positions 6 and 12 of the previously audited down right-relative-power initializer. Its position 3 reaches `states[5].left` and a Gram-refresh product in all four graphs. This describes the consuming dependency and declared source; it does not assign ownership or semantics to loop values. Both original while outputs, projections `[1]` and `[6]`, remain opaque. No loop body or recurrence was traversed.

The passing paired-capture mismatch is confined to `S(1)` tags on sibling leaves 3 and 4 of a combined all-reduce tuple in the left initializer; selected leaf 5 is unchanged. Substitution has four mismatching routes. Two are distinct Done-operand-2 and Start-input all-gather records, with channel 292→289 and chain 28→32. The newly inspected initializer reached through positions 6 and 12 of the previously audited right-relative-power initializer differs only in all-reduce channel 276→273. The left initializer includes channel 143→133, all-to-all channel 323→320, `S(1)` tags and unaligned copy/tuple/source connections. Comparison stops at unaligned opcodes or edges and makes no claim about the remaining subtrees. Up remains a passing numerical control; neither IDs nor tags establish execution order or cause.

The Done's ten declared inputs reach one common Start identity through tuple paths [0], [1], [0], [2], [3], [4], [5], [6], [7], [8]. Repeated leaf 0 remains explicit. All original Start/Done/Concat/while outputs remain opaque, and product traversal stops before operands. Normalized signatures retain shapes, attributes, literals, channel/chain IDs and ordered connections after canonicalizing identifiers, metadata and source wrappers. Raw caller contexts and repeated edges are retained; recursive hashes do not distinguish sharing.

The CPU job `ema-gram-20-polar-initializers-cpu-20261103` succeeded on attempt 0 with exit 0, UID `ee02cf152c49`, verified by canonical job/task reads at 19:16 UTC. Published source `bd35d932d32d9e7123ca868284c2b8ba65e07d82` ran on four CPUs in us-west4 with default preemption and no wall-clock limit. Four closure rounds recovered fourteen bodies per graph, with no requested records missing. Source/manifest checksums and all four HLO hashes/sizes were verified regionally. The 1,658,615-byte report has SHA256 `f08c5d403d7f9bddc9f2bbbc8c8e55ccb134ccebc2c6275873f6e5be108ccdee` and passed exact GCS readback. Large HLO stayed regional; no accelerator, numerical kernels or W&B run was needed.

The [coverage summary](compiler-polar-initializers-frontiers.json), using current and sixteen prior reports, leaves four uninspected input positions per graph on one Concat boundary reached through position 3 of the left initializer. It records eighteen unique context/projection frontiers in each control graph and seventeen in substitution. These are coverage counts, not cross-graph identity or output semantics. Older gaps remain recorded. The [initializer supplement](compiler-polar-initializers-sources.json), [comparison](compiler-polar-initializers-comparison.json) and [validation](compiler-polar-initializers-validation.json) retain all mismatches, exact callers and repeated tuple positions. Six same-HLO o control routes are reused with their three prior substitution mismatches; no control extraction or numerical measurement was repeated.

The next static decision is to inspect those four Concat inputs using saved dependencies first. Preserve paired capture, passing controls, ordered repeated positions and opaque outputs; reuse covered inputs and retain older gaps. Publish exact source before any regional CPU extraction. No numerical, precision or tiling intervention is selected, and no optimizer, training, MFU or restore acceptance is claimed.

The sole follow observer ended naturally with exit 0. No experiment, queue or observer remains active. PR monitoring remains handed off after its completed twelve-hour wait; the research heartbeat stays active.

### November 3, 19:33 UTC — saved records resolve four left-initializer Concat inputs

All four evaluator input-route signatures match untouched. Paired capture and column substitution each match zero of four at the same declared operand positions. The four graphs are matrix-path untouched, evaluator, paired capture and column substitution. Paired capture still passes its numerical reference gate, while both full 28-leaf substitution gates remain failed. The static mismatch therefore does not establish the cause of numerical failure.

This audit selects the previously uncovered Concat through position 3 of the initializer on the down left-relative-power route. Every declared input follows slice-done/start to the same within-graph all-gather. The leading slices, in operand order, are [0:3], [3:6], [6:9], [9:11]; both other axes span [0:1024]. Untouched and evaluator then reach a tuple projection and producer tree containing declared `states[6].left` (with the evaluator bank wrapper), a Gram-refresh product and an opaque Concat. The product and Concat inputs already have complete exact-context traces in saved reports; their outputs are not thereby semantically resolved.

Capture and substitution instead stop at one shared opaque Concat per graph where untouched has a tuple projection. The comparator stops at that opcode/connection mismatch, without claiming equality of the remaining subtrees. Substitution additionally changes all-gather channel 325→322; capture retains channel 325. Four input routes are repeated views of shared declarations, not four independent changes. The passing capture control prevents treating this Concat-versus-projection difference alone as the failure cause. Declared shapes, tags and IDs do not establish runtime precision, values or execution order.

All required computation bodies were already present in the eighteen SHA/size-checked bounded inputs: the inventory and seventeen saved reports. This was local static parsing only, with no new job, HLO download, accelerator, numerical kernels or W&B run. Per-graph HLO identities were joined to the previously verified regional inventory; no new HLO audit is claimed. The [complete local report](compiler-left-initializer-concat-report.json) is 448,760 bytes with SHA256 `06591fc7a0f9c6ce88ecee73db86a5ecea268076c518bd5301abaa5c3603f1ef`. No requested records are missing.

The [coverage summary](compiler-left-initializer-concat-frontiers.json) leaves no new uninspected positions on the current untouched/evaluator frontiers. Capture and substitution each expose four uncovered inputs on their newly reached Concat. Their endpoint identities remain separate; no counterpart is invented for the untouched/evaluator tuple projection. Older gaps remain recorded. All original while/Start/Done/Concat outputs stay opaque, and traversal stops at product records before their operands.

The [comparison](compiler-left-initializer-concat-comparison.json) retains all original mismatches. Normalized signatures retain shapes, attributes, literals, channel/chain IDs and ordered connections after canonicalizing compiler identifiers, metadata and source wrappers. Raw caller contexts and repeated edges are retained; recursive hashes do not distinguish sharing. The [validation](compiler-left-initializer-concat-validation.json) checks exact selectors, initializer position, explicit operand order, missing/duplicate/wrong-root rejection, dtype mutation, shared records, source scopes and coverage. Six same-HLO o controls are reused with their three prior substitution mismatches; no control measurement is repeated.

The next static decision is to inspect the four new Concat inputs in each of capture and substitution using saved dependencies first, retaining the passing capture control and reusing covered untouched/evaluator routes. Do not pair different endpoint kinds. Publish exact source before any necessary regional CPU extraction. Both numerical gates remain failed at rtol=atol=1e-5; no numerical, precision or tiling intervention, optimization, training, MFU or restore acceptance is selected.

No experiment, queue or observer is active. Old jobs remain terminal and PR monitoring remains handed off; the research heartbeat stays active.

### November 3, 19:47 UTC — capture and substitution inner Concat routes reach covered frontiers

None of the four new substitution input-route signatures matches paired capture at the same declared operand position. This comparison is between two matrix-path graphs: paired capture and column substitution. Untouched/evaluator outer routes are reused from the previous audit, with their original comparisons preserved; no counterpart is invented for their different endpoint kind. Paired capture remains a passing numerical control. Both full 28-leaf substitution gates remain failed, and these declarations do not establish the cause.

The inspected inner Concat is the opaque endpoint reached through the all-gather on each previously traced input of the Concat at position 3 of the down left-relative-power initializer. Each graph's four inner Concat inputs follows slice-done/start into a shared producer tree. Capture reaches declared `bank[1][6].left`, a Gram-refresh product and an opaque Concat. Substitution reaches a Gram-refresh product and an opaque Concat without reaching an entry state parameter on these bounded routes. Each graph's four routes share the same exact product and Concat records. Traversal stops at the product before its operands; all original Concat and while outputs remain opaque.

Capture's leading slice intervals are [0:3], [3:6], [6:9], [9:11], while substitution's are [6:9], [9:11], [0:3], [3:6]. The other axes span [0:128] and [0:1024]. Positions 1 and 3 consequently have different leading declared sizes. This is a difference in slice declarations, not evidence of executable permutation or Concat output semantics. Unlike the earlier down-step-5 slice audit, no full-signature correspondence is claimed here: the producer trees also differ.

Two tuple-shape records differ only by an `S(1)` tag on sibling leaf 6 of a fourteen-element tuple. Other recorded differences include additional shape tags, all-to-all channel 323→320, capture tuple-projection versus substitution copy-done, and capture opaque Concat versus substitution tuple-projection. The difference walker stops at each of the two unaligned connections; it does not compare the remaining subtrees there. Repeated views of shared records are not independent changes. Tags, IDs and copy declarations do not establish runtime precision, values, execution order, equality or cause.

All required bodies were already present in nineteen SHA/size-checked bounded inputs: the inventory and eighteen saved reports. This was local static parsing only, with no new job, HLO download, accelerator, kernels or W&B run. All four prior graph HLO identities were checked against the saved inventory, including the reused untouched/evaluator controls; this is not a new regional HLO audit. The [complete bounded report](compiler-capture-initializer-concat-report.json) is 362,383 bytes with SHA256 `cd961dbdc66714be814845b900540276a9ced5dced85b17e003def4a27921f27`. No requested records are missing.

The [coverage summary](compiler-capture-initializer-concat-frontiers.json) finds complete saved input-root traces for both operands of each reached Gram-refresh product and all four inputs of each reached Concat. There are no further uncovered positions on this audit's current frontiers. Capture has three unique context/projection frontiers (state, product, Concat); substitution has two (product, Concat). This closes the current branch's input gaps only. Older gaps across the accumulated reports still need reconciliation, and covered inputs do not resolve opaque output semantics.

The [comparison](compiler-capture-initializer-concat-comparison.json) explicitly uses capture as its reference. Its source-wrapper normalization was checked against the prior comparator after canonicalizing the reference's state label; both difference walkers agree. Fixtures also check exact caller/dependency selection, ordered operands, wrong roots, duplicates, missing records, dtype mutations, slice declarations, shared identities and zero new input gaps. [Reused outer controls](compiler-capture-initializer-concat-reused-outer-controls.json) retain four evaluator matches and four prior capture mismatches and four prior substitution mismatches. Six o control routes are also reused with their three prior substitution mismatches. No new numerical control measurement was made.

The next static decision is to reconcile exact-context input coverage across all accumulated reports before selecting another extraction. Preserve caller identities, output dependencies, ordered repeated positions and tuple paths; distinguish covered inputs, deliberate gaps and opaque outputs. Retain paired capture and passing controls, avoid re-expanding covered inputs, and keep unlike endpoint kinds unaligned. No numerical, precision or tiling intervention is selected. Any necessary regional parse requires exact published source/manifest/checksums first, default preemption and no wall-clock limit. The full numerical tolerance stays rtol=atol=1e-5.

No experiment, queue or observer is active. Old jobs remain terminal and PR monitoring stays handed off; the research heartbeat remains active.

### November 3, 20:02 UTC — accumulated coverage retains older product and collective gaps

Reconciliation across nineteen saved reports finds 46, 46, 48 and 43 frontier instructions with uncovered declared inputs in matrix-path untouched, evaluator, paired capture and column substitution, respectively. They account for 171, 171, 174 and 154 operand positions. The prior initializer branch's zero-new-gap result was local to that branch; these older gaps were not erased. Both full 28-leaf numerical substitution gates remain failed. Coverage counts do not establish cross-graph identity, runtime values or cause.

Each graph has fourteen uncovered product frontiers, with both operands uninspected: thirteen dependency-mapped polar products and one nonpolar `gram_direction` product. The polar roles are o iteration-1 `B_X`; o iteration-2 `scaled_A_A`; o iterations 3/4 `X_Xt` and `scaled_A_A`; down iterations 3/4 `scaled_A_A`; gate iterations 4/5 `X_Xt` and `scaled_A_A`; and up iteration-5 `scaled_A_A`. Here `A = X @ X.T`, `B_X = B @ X`, and `scaled_A_A = (cA) @ A`. The nonpolar product lies on saved o iteration-2 `B_X` dependencies; its exact source scope and caller identity are retained without assigning a polar role. Only two of the 28 immediate operand records appear in saved traces at the same caller context, and neither is a complete input-root trace.

The remaining gaps are eight Done endpoints per graph, six/six/seven/seven Start endpoints, thirteen/thirteen/fourteen/nine Concat endpoints, and five while initializers. Each Done has ten declared positions; each Start and while has one. Concat arities are retained individually, including two-input endpoints in capture and substitution. These are deliberate frontier gaps, not reported missing records from completed extractions. Input coverage never makes the original Start/Done/Concat/while output semantically transparent.

The join requires the exact graph, caller context, computation, instruction and empty tuple projection of a complete saved input trace root. A record merely appearing somewhere in a trace does not cover its inputs. Ordered edges and repeated operand positions remain in the local index, and the published gap summary retains source declarations, tuple projections and output-dependent report paths. The union of saved edges is not a runtime graph. Reordering report, record and edge lists while remapping record indices preserves all four actual summaries. Fixtures also reject conflicting declarations, reported missing records, invalid roots and duplicate outgoing positions, and preserve gaps under wrong-caller, wrong-computation and wrong-projection joins.

Twenty bounded inputs—the inventory plus nineteen reports—passed SHA/size checks. Each available graph's HLO identity matches the prior regionally verified inventory. The latest report contains only capture and substitution; its absence for untouched/evaluator is explicit, not treated as new evidence for those graphs. No new job, HLO download, regional HLO audit, accelerator, numerical kernels or W&B run was needed.

The [published summary](compiler-reconciled-coverage-summary.json) is 334,404 bytes with SHA256 `37b60b7fef30a99314a97576e48fb8c47d6149fd6d522fe9154f2d194dad3826`. The full index stays local at `scratch/20261103-reconciled-coverage-report.json`: 36,408,848 bytes, SHA256 `ea1210b29221534f921a69ed70a7f7146ab776345c3caf86bcb43c9a95650d06`. It can be regenerated from the [source](reconcile_compiler_coverage.py) and [manifest](compiler-reconciled-coverage-manifest.json); it was not transferred across regions. Record occurrences are 12,624/12,624/12,297/12,984, yielding 5,571/5,571/5,640/5,480 unique context/projection records. These are inventory counts only.

The next static decision is to inspect the 28 declared operands of those fourteen products per graph, using saved dependencies and source scopes first. Select the nonpolar product through the recorded o output dependency, not compiler ordinals or a guessed polar role. Preserve passing o/up and paired-capture controls, exact caller contexts, ordered repeated uses and tuple projections. Retain the other older gaps; do not repeat covered expansion or align unlike endpoint kinds. Any necessary regional CPU parse requires exact published source/manifest/checksums and HLO SHA/size verification first, default preemption and no wall-clock limits. No numerical, precision or tiling intervention is selected; full numerical tolerances remain rtol=atol=1e-5.

No experiment, queue or observer is active. All old jobs remain terminal and PR monitoring remains handed off. The research heartbeat stays active.

### November 3, 20:15 UTC — inspect reconciled product gaps (static preflight)

The [reconciled coverage](compiler-reconciled-coverage-summary.json) identifies fourteen products with twenty-eight uncovered operands per matrix graph. The [selector](trace_remaining_products.py) joins exact saved output dependencies to caller/instruction identities, verifies thirteen polar roles and selects the remaining nonpolar `gram_direction` product through its o iteration-2 dependency. It rejects missing, duplicate, wrong-context, wrong-role, wrong-projection and already-covered targets. Target selection is invariant to report and frontier order.

All twenty bounded input hashes/sizes and available graph HLO identity joins passed. Thirteen required computation bodies per graph are absent locally. The [plan](compiler-remaining-products-plan.json), [manifest](compiler-remaining-products-manifest.json) and [preflight validation](compiler-remaining-products-validation.json) specify one CPU-only us-west4 extraction after exact source publication. Verify each HLO hash/size regionally before parsing; use default preemption and no wall-clock limits. Reuse the existing product-operand tracer and comparator, preserve passing up/o and paired-capture controls, and keep all original opaque outputs opaque. Both full numerical gates remain failed; no numerical intervention is selected.

### November 3, 20:19 UTC — remaining product operands match outside the passing o control

The audit recovered both declared operands of fourteen previously uncovered products in each matrix-path graph. All 28 evaluator signatures match untouched, as do 27 paired-capture and 23 substitution signatures. All fourteen inspected gate/down/up routes match in both capture and substitution. The five substitution mismatches lie on o routes, which remain a passing numerical control. Both full 28-leaf numerical substitution gates remain failed. Static matches do not establish equal runtime inputs; static differences do not establish cause.

The passing-capture mismatch is o iteration-1 `B_X` operand 1: two shape declarations lose `S(1)`, followed by a fusion-versus-copy-done mismatch where comparison stops. Substitution has that same mismatch and four others. Its o iteration-1 `B_X` operand 0 changes two tuple-shape declarations, including sibling dimensions, beyond `S(1)` alone. Both o iteration-3 `X_Xt` operands encounter Concat-versus-tuple-projection connections; operand 1 also changes all-gather channel 212→209. The nonpolar o-direction product's operand 1 changes tuple shapes, `S(1)` tags and a copy-versus-copy-done connection. The comparator stops at each unaligned opcode/edge; it claims no equality of the remaining subtrees. Shape declarations and IDs do not establish runtime precision, values or execution order.

That nonpolar product's operand 0 reaches another direct `gram_direction` product. Operand 1 reaches declared right-Gram state `states[3].right` (with bank wrappers), a Gram-refresh product, a relative-power product and an opaque Concat in all four graphs. These are source declarations, not evidence that runtime arrays are equal. Product traversal stops before operands. All original Start/Done/Concat/while outputs remain opaque. Normalized signatures canonicalize compiler identifiers, metadata and source wrappers while retaining shapes, attributes, literals, channel/chain IDs and ordered connections. Raw caller contexts and repeated edges remain available; recursive hashes do not distinguish sharing.

The CPU job `ema-gram-20-remaining-products-cpu-20261103` succeeded on attempt 0 with exit 0, UID `78604323a445`, verified by canonical job/task reads at 20:16:41/42 UTC. Published source `92bf08b7928d25f54a5e194a8e39aa5990d73ba7` ran on four CPUs in us-west4 with default preemption and no wall-clock limit. All twenty bounded input identities and four HLO hashes/sizes were checked. Three closure rounds recovered 23 bodies in each untouched/evaluator graph and 22 in each capture/substitution graph. No requested records remain missing. The 1,476,105-byte report has SHA256 `32d2b7430a01c09bdf1ecbd4d0f2e6a9fac02c9dbbec567143f2bdb6641c511c` and passed exact GCS readback. Large HLO stayed regional; no accelerator or numerical kernels were used.

The [updated accumulated coverage](compiler-remaining-products-frontiers.json) joins the new report with all nineteen prior reports. All fourteen former product gaps are now input-covered; five newly reached products per graph leave ten further operands uninspected. These are o iteration-1 `X_Xt` and `scaled_A_A`, plus the nonpolar direct-direction, Gram-refresh and relative-power products selected through the actual o dependency. Other gaps remain: Done endpoints 16/16/15/15, Start 6/6/7/7, Concat 17/17/18/12 and five while initializers per graph. In untouched/evaluator/capture/substitution order, total uncovered instructions are 49/49/50/44 and positions 249/249/242/218. Totals grew because this expansion exposes further boundaries; no previously covered input became uncovered. These are deliberate coverage gaps, not missing requested records or output semantics.

The [comparison](compiler-remaining-products-comparison.json), [source supplement](compiler-remaining-products-sources.json) and [validation](compiler-remaining-products-validation.json) retain all mismatches and exact identities. Fixtures check target selection, explicit operand order, missing/duplicate/wrong-root rejection, dtype declaration mutation, actual right-state/source scopes, the shared capture/substitution fusion-versus-copy mismatch, old-gap coverage and new counts. The next static decision is to inspect the ten newly uncovered product operands per graph from saved dependencies first, preserving paired capture and passing controls. Retain all other collective/Concat/while gaps and keep unlike endpoint kinds unaligned. Publish exact source before any required regional extraction. No numerical, precision or tiling intervention is selected; rtol=atol=1e-5 remains fixed.

The sole follow observer ended naturally with exit 0. No experiment, queue or observer remains active. All old jobs stay terminal; PR monitoring remains handed off and the research heartbeat stays active.

### November 3, 20:29 UTC — inspect five o-dependent products (static preflight)

The [current coverage](compiler-remaining-products-frontiers.json) leaves ten operands across five o-dependent products per matrix graph. The [selector](trace_o_source_products.py) uses actual o iteration-1 `B_X` operand-0 dependencies for the two polar products and the saved nonpolar product's operand-0/1 dependencies for direct-direction, Gram-refresh and relative-power products. Source scopes and exact caller identities distinguish these roles; compiler ordinals do not select them.

All twenty-one bounded input hashes/sizes and available graph HLO joins passed. Two required computation bodies per graph remain absent from the saved bounded records. The [plan](compiler-o-source-products-plan.json), [manifest](compiler-o-source-products-manifest.json) and [validation](compiler-o-source-products-validation.json) specify one CPU-only us-west4 extraction after exact source publication, with regional HLO SHA/size checks, default preemption and no wall-clock limits. Fixtures reject missing/duplicate/wrong-context/role/projection and already-covered targets. Preserve paired capture and passing controls, reuse covered inputs and retain all other gaps. All original opaque outputs stay opaque; both full numerical gates remain failed.

### November 3, 20:36 UTC — o source operands retain differences in passing controls

The ten newly inspected product-operand routes match untouched in all ten evaluator comparisons, five paired-capture comparisons and one column-substitution comparison. All routes belong to the o dependency chain, whose numerical substitution control still passes. The whole optimizer's two full 28-leaf substitution gates remain failed. These static differences do not establish a failure cause, and matching routes do not establish equal runtime inputs.

The five products were selected through saved output dependencies and exact caller contexts: o iteration-1 `X_Xt` and `scaled_A_A`, plus nonpolar direct `gram_direction`, Gram-refresh and relative-power products. Product traversal stops before operands of the next product. In substitution, both inspected refresh operands reach the same declared `gradients[3]` parameter; untouched, evaluator and capture instead stop at a shared opaque Concat within each graph. The direct-direction operand-0 route reaches a declared `states[3].left` parameter only in substitution, alongside refresh/relative-power products and an opaque Concat. The other three graphs reach those product scopes and an opaque Concat without an entry state parameter on these bounded routes. This does not establish that their runtime state inputs are absent or different.

Capture's five mismatches comprise three routes with only `S(1)` shape-tag changes and two routes with unaligned connections: o1 `X_Xt` operand 1 has fusion versus all-gather, and relative-power operand 0 has all-gather versus copy-done. The three shape-only routes are o1 `X_Xt` operand 0, o1 `scaled_A_A` operand 1, and direct-direction operand 0. Capture still passes its numerical gate. Substitution matches only direct-direction operand 1. Its other nine routes retain shape/tag changes, tuple sibling dimension changes, collective IDs and unaligned copy, tuple-projection, Concat and parameter connections. The difference walker stops at every unaligned opcode/edge; no equality is claimed for the remaining subtrees. Repeated views are not independent changes, and declared tags/IDs do not establish runtime precision or order.

The relative-power operand-0 boundaries also differ: untouched/evaluator reach Concat and an opaque while projection [1]; capture reaches Concat and opaque Done; substitution reaches opaque while projection [2] and Done. Substitution's operand 1 reaches that same while projection [2], while the other graphs stop at Concat. Unlike endpoint kinds and tuple projections remain unaligned. No while body or recurrence was interpreted. All original Start/Done/Concat/while outputs remain opaque.

The CPU-only regional job `/kaiyuew@stanford.edu/ema-gram-20-o-source-products-cpu-20261103` succeeded on attempt 0, exit 0, UID `889ee6865bb0`; canonical job/task checks completed at 20:32:56/58 UTC. It used four CPUs, 8 GiB memory and 10 GiB disk in us-west4, default preemption and no wall-clock limits. The worker name includes TPU, but this task used no accelerator, numerical kernels or W&B. Exact published source `c6cfd10c383768ecd369815e2d7ced7413b04d93`, its manifest, twenty-one bounded inputs and all four HLO SHA/size identities were verified regionally. Each graph needed two extraction rounds and three computation bodies; no requested records are missing. Large HLO stayed regional. The [report](gs://marin-us-west4/ema-gram-20-attribution/2026.11.03.o-source-products/o-source-products-c8c5565b9a05/report.json) is 985,463 bytes, SHA256 `6e338c6bd6f0e2dc49ff746f1b9407da857cdb900bc8d42e848c0b7a7c55e685`, with exact GCS readback.

The [accumulated coverage](compiler-o-source-products-frontiers.json) now covers all five previous product gaps and exposes two further nonpolar products, each with two uncovered operands. Both are reached through direct-direction operand 0; their exact source scopes are `update_gram_state` and `relative_gram_power`, without guessed polar roles. The four graphs retain 47/47/48/42 uncovered instructions and 253/253/246/222 operand positions. Other gaps remain: Done 17/17/16/16, Start 6/6/7/7, Concat 17/17/18/12 and five while initializers per graph. These are deliberate frontier gaps, not requested missing records. Complete input-root coverage is distinct from record presence and opaque-output semantics.

The [comparison](compiler-o-source-products-comparison.json) retains declared shapes, attributes, literals, collective IDs and ordered connections while canonicalizing compiler identifiers, metadata and source wrappers. Raw callers and repeated edges remain recorded; recursive signatures do not distinguish sharing. [Source summaries](compiler-o-source-products-sources.json) reproduce the boundary records. Fourteen prior gate/down/up routes are reused with report SHA/size and same-HLO checks; all match across the four graphs, with no new numerical control measurement. Fixtures passed selection/order/missing/duplicate/caller/scope/projection/role mutations, comparator root/swap/dtype checks, source scopes and shared refresh endpoints, unaligned stops, coverage counts, and reused SHA/size/HLO rejection.

The next static decision is to inspect the four declared operands of the two newly exposed nonpolar products, selected through the exact direct-direction operand-0 dependencies. Use saved reports and inventory first, preserve passing capture/up/o controls, ordered repeated positions and tuple paths, and retain all other global gaps. Reuse covered inputs and keep unlike endpoints unaligned. Any needed regional CPU parse requires exact published source/manifest/checksums and HLO SHA/size verification first, default preemption and no wall-clock limits. No numerical, precision or tiling intervention is selected; full numerical tolerances remain rtol=atol=1e-5.

The sole follow observer ended naturally with exit 0. No experiment, queue or observer remains active. Old jobs stay terminal, PR monitoring stays handed off, and the research heartbeat remains active.

### November 3 — prioritize training loss and measured speed

The user clarified that the practical criterion is comparable training loss, rather than strict numerical equivalence. The research priority is now a paired training comparison of the original and candidate implementations. The existing `rtol=atol=1e-5` checks remain recorded diagnostics; failing that equivalence threshold alone no longer blocks a training-quality experiment. This changes the acceptance criterion, not the measured results, source precision, full Gram geometry or every-update refresh.

The latest direct candidate-versus-original comparison has relative Frobenius errors of approximately 0.70% for q/k/v/o update directions and 2.90% for gate/up/down, while all 21 state outputs pass the numerical threshold. These are single-update differences, not measured training-loss degradation. The input-replacement diagnostic feeds each path its own captured intermediate values and checks whether it reproduces that path's untouched outputs. Those self-checks fail, which is a separate comparison and still prevents causal claims about the untouched executable. No loss-equivalence or training-speed improvement is established yet.

Defer further compiler-boundary expansion unless it becomes necessary to explain an observed training problem. The interrupted four-operand o-product audit reached only local preflight: saved records suffice, but no validated result was published and no job was launched. Its draft selector is preserved under scratch. Existing opaque-output and causal limitations remain valid.

Next, prepare the paired training experiment from published implementations. Match model, global batch 8, data ordering, seed, optimizer hyperparameters and initialization or a verified full checkpoint restore. Preserve the original 7,630-step schedule; do not silently shorten it or introduce queue/task/training/watchdog wall-clock limits. Compare training and validation loss at matched steps and progress against measured wall-clock time. Record actual steady training throughput/MFU; the performance target remains an absolute steady training MFU above 5%, not a 5% relative improvement. Do not infer loss quality from update-norm error or substitute synthetic timing for training MFU. Quantify any loss gap and its variability before declaring it acceptable; no numerical loss-equivalence margin has been specified.

The remaining preparation is operational validation: confirm the selected published entrypoints and configurations, verify full model/optimizer/data-position restore if resuming, and run the required online logging check and CPU harness for any new numerical implementation. Keep finite-output, configuration, restoration and logging checks. Report numerical-equivalence failures transparently rather than treating them as training outcomes. Preserve fresh online W&B runs for fresh processes, default preemption/retries and evidence across attempts. Keep large checkpoint arrays in their existing region. Any GPU launch requires an allocation-retention holder.

No training run was launched by this priority change. No experiment or observer is active; all old jobs remain terminal and must not be repolled or restarted. PR monitoring remains handed off without replacement or manual polling. The existing research heartbeat stays active and follows this loss-first priority.

### November 3 — paired continuation configuration and preemption recovery

The original and candidate training configurations are prepared for a matched continuation from the same step-923 checkpoint in us-central2. No training has been submitted and no new loss result exists. The original uses the published replicated eigensolver layout; the candidate uses the published matrix layout, both with the default eigensolver implementation. The model, global batch 8, seed/data seed 0, data configuration, optimizer hyperparameters and 7,630-step final schedule match. The only resolved configuration differences are the layout and ten run/output/checkpoint identity fields. A continuation starts at the restored step; it does not run 7,630 additional steps.

Preparation exposed a preemption-recovery bug in the continuation recipe: `load_checkpoint=False` forced every new process back to the shared initial checkpoint, even when that run had saved later progress. The recipe now uses automatic checkpoint selection (`None`), preferring its own permanent or temporary checkpoint roots and falling back to the shared source on first start. Two configuration/recovery tests pass, including first start, own temporary-checkpoint recovery and a missing-source failure. These are mocked checkpoint-selection tests, not evidence that the real trainer state has been restored. No observed loss of running-job progress is claimed.

A fresh read of the small source metadata and manifest at 21:00:50 UTC matched the saved identities. They declare step 923 and 57 arrays across model, optimizer state, step and training PRNG state, including 21 FP32 Gram/momentum arrays totaling 3,460,300,800 bytes. No checkpoint arrays were downloaded or read on TPU. Metadata presence does not verify array contents, full restore or data position.

The [prepared configurations](loss-pair-prepared.json), [generator](prepare_loss_pair.py), [checkpoint metadata evidence](loss-pair-checkpoint-metadata.json) and [validation](loss-pair-preparation-validation.json) make this setup reproducible. The first local test invocation accidentally used another checkout's installed packages and failed; explicit worktree `PYTHONPATH` corrected the environment, and the two relevant tests then passed. Required lint passed. Preparation performed no numerical kernels and launched no W&B run or compute job.

Next, validate full trainer restoration and host/device memory in us-central2 before continuing the pair. Verify model, optimizer, step and PRNG state, plus the data stream's position derived from the restored step and matched data configuration. Keep the unchanged-state and synthetic-gradient optimizer checks. Run the required online logging check before numerical execution, publish exact source, and use fresh online W&B for each process. The proposed v4-32 configuration is not proof that full training fits; the earlier synthetic host-memory failure remains relevant. Keep default preemption/retries, all attempt evidence and no wall-clock limits.

Compare training and held-out loss at matched steps and progress per measured wall-clock time. Keep numerical-equivalence failures as diagnostics, without using them alone to veto the loss experiment. The absolute steady training MFU target remains above 5%; no acceptable loss margin or training speedup is claimed. Static compiler expansion remains deferred. All old jobs remain terminal, no experiment/observer is active, and PR monitoring stays handed off.

### November 3 — full trainer restore audit ready for regional validation

A non-training restore audit now follows the native LM initialization path and intercepts entry into its training loop. It compares every locally addressable shard of the complete saved trainer state against an independent TensorStore read, requires the exact manifest path set and restored step, and checks finite values. Each process must produce a successful report before the distributed restore can be accepted. The audit includes model, optimizer, step and training PRNG state; it does not selectively load only the Gram arrays.

The data check requires the iterator to start at the restored step and hashes the first batch against a second iterator at that same position. This verifies reproducibility of the configured stream at the restored offset. It cannot compare against a historical batch that was never saved. The probe executes no training update and does not save an advanced checkpoint. Its memory report covers restoration and verification, including temporary host copies; it does not establish full-training memory fit.

The own online logging check and generated CPU checkpoint harness passed. The CPU harness restores all 57 arrays through `Trainer.initial_state`, compares all bytes to storage and the known original state, verifies a known data-offset oracle, rejects a changed PRNG state and confirms that a synthetic optimizer update is finite without mutating its input. This uses a tiny generated model and local checkpoint, so it does not validate the original TPU checkpoint. An initial harness attempt failed before model kernels because its fixture version did not satisfy the repository naming rule; the corrected version is retained with the failed attempt evidence.

The proposed regional check uses one sequential v4-32 job with four hosts, 120 CPUs, 256 GiB memory and 50 GiB disk per host in us-central2. The request raises the per-host budget from 128 to 256 GiB to accommodate verification copies; actual usage remains to be measured. It does not change model geometry, source precision or numerical algorithms. Use default preemption/retries and no wall-clock limits. Publish the exact source and validation before submission. Each execution host first performs a separate online logging check; the native trainer then creates fresh online tracking before model kernels. Verify source, checkpoint metadata and manifest identities before loading arrays. Keep all large checkpoint data regional.

After successful regional reports, full training still needs measured memory, loss and speed evidence. Numerical-equivalence failures remain diagnostics and do not by themselves veto the paired loss experiment. Static compiler expansion stays deferred; old jobs remain terminal and PR monitoring remains handed off.

### November 3, 21:20 UTC — regional full restore audit accepted

Iris accepted `/kaiyuew@stanford.edu/ema-gram-20-full-restore-v4-32-20261103` at 21:20:10 UTC from published source `e045d6919861359860847d6b0044cc79f475239b`. The job requests one v4-32 slice, four hosts with 120 CPUs/256 GiB/50 GiB each in us-central2, default preemption/retries and no wall-clock limit. Source SHA256 is `51341f7c170f84fe84eea64f1956ffe28c0bf2bd445f769c1ce811e5d8221f6e`. The [preflight validation](full-restore-preflight-validation.json) records finished online logging and CPU harness runs with that exact source hash. Canonical job/task checks at 21:22:22–24 UTC show all four tasks pending, with no attempts yet. The sole follow observer is session 6883. Submission and queueing do not establish successful restoration; canonical success and all four process reports are still required.

The probe does not train or advance the saved checkpoint. Its four-host restore audit is the sole active experiment; monitoring belongs to the current thread. After restore validation, retain the original saved-state synthetic-gradient and unchanged-state checks, full-training memory/finite checks and matched execution resources before the paired continuation. Old jobs remain terminal and PR monitoring stays handed off.

### November 3 — match both continuation memory requests to the restore audit

Both prepared training configurations now request 256 GiB per host, matching the queued restore audit. Relative to the previously published configurations, the only payload change within each variant is `resources.ram`, from `128g` to `256g`. The shared recipe exposes this as `--host-memory`; its existing default remains `128g`. No numerical optimizer code, model, data or precision setting changed. Neither training variant has been submitted, and the queued restore worker retains its previously published source.

The saved configuration for the checkpoint's source run, version `2026.10.30.23`, exactly matches the new original configuration's model, data specification, data seed, sequence length, loss weighting, original optimizer fields, global batch, final step, training seed, mixed-precision policy and the configured `trainer.per_device_parallelism` value. That matching field does not mean the effective sharding layout is unchanged. The new original explicitly records `eigh_layout=replicated` and `eigh_implementation=default`, which were absent from the older serialization. Execution differences remain explicit: the new mesh is data4/model4 instead of data=-1/model1, profiling is at step940 for one step, and run/output/restore identities differ. Configuration matching does not prove historical cache contents or runtime batch identity.

The [resource and source-protocol comparison](loss-pair-memory-validation.json) records these checks. The two continuation configuration/recovery tests pass again. Full regional restoration remains pending; no training loss, training memory fit or speed improvement has been measured. After successful restore evidence, retain the saved-state synthetic-gradient/unchanged-state checks and full-training finite/memory checks before the paired continuation. Numerical-equivalence failure alone remains diagnostic, not a veto.

### November 3 — restore audit exposed an incorrect step-label assumption

The regional full-restore audit failed before its independent array and data checks. All four processes loaded a trainer state with counter 924 from the checkpoint named `step-923`. The audit incorrectly required that counter to equal the metadata label 923. Native `StepInfo.step` is `state.step - 1`, and the checkpoint callback saves the post-update state using that completed-update label. The observed difference is consistent with the native writer's convention; it does not establish checkpoint corruption or an optimizer failure.

Canonical status is failed: task 1 exited 1 on attempt 0, UID `ebad5d996d00`; tasks 0, 2 and 3 ended as coscheduled siblings. No preemption occurred. The sole follow observer ended naturally with exit 0. All four execution-host online logging checks passed, but no successful per-process audit report was produced. Full restoration, data replay and training memory fit remain unverified. This job is terminal and will not be restarted.

The corrected audit compares the loaded counter to an independent TensorStore read of the saved `step` scalar and reports the checkpoint label separately. Its CPU regression uses the native `StepInfo` convention to save label 1 with state counter 2; it also rejects a mutated restored counter. The fresh online logging check and CPU harness passed with the corrected source. The harness restored all 57 arrays, verified data offset 4 and known batch [4,5], rejected changed step/PRNG values, and retained finite synthetic-update and unchanged-input checks. The original CPU fixture used equal label and counter values, which failed to exercise this convention. The correction changes validation only; no optimizer algorithm or precision changes.

Both prepared training configurations now request 256 GiB per host. Their model, data specification and original optimizer settings match the saved source-run configuration; effective mesh differences remain explicit. Neither training variant has launched. The next regional audit must complete all four array/data reports before the paired continuation. Data replay follows the verified stored counter, not the checkpoint's label; the final schedule remains 7630 steps. The research priority remains matched training/validation loss and measured speed, with numerical-equivalence failures retained as diagnostics.

The [failure record](full-restore-failed.json) preserves canonical status and the assertion. The [corrected validation](full-restore-counter-validation.json) records fresh finished W&B runs and exact source hashes; CPU success does not validate the original TPU checkpoint. The [prepared follow-up](full-restore-counter-prepared.json) uses a new output version, `2026.11.03.24`, and a new job identity after source publication. It remains a non-training audit with one v4-32 slice spanning four hosts, 120 CPUs/256 GiB/50 GiB each in us-central2, default preemption/retries and no wall-clock limits. All previous jobs remain terminal.

### November 3, 21:41 UTC — corrected counter audit accepted

Iris accepted `/kaiyuew@stanford.edu/ema-gram-20-stored-counter-audit-v4-32-20261103` at 21:41:06 UTC from published source `b55656b4c2fd364292b75aeaee765698cbd9fc56`. Source SHA256 is `49f551e2c101ea2b9fde522a651312e95df8d53aa5c0eb3a2b55ac70faac2011`. The [submission record](full-restore-counter-submitted.json) identifies output version `2026.11.03.24`, unchanged audit resources and the sole follow observer, session 37396. Canonical job/task reads at 21:42:50–52 UTC show all four tasks pending, with no attempts, failures or preemptions. The initial check began 17 seconds before the requested two-minute wait; this monitoring deviation is recorded, and no duplicate check was made.

This is the sole active job. It performs no training update and must produce all four successful state/data reports before restoration is accepted. The old failed audit remains terminal. The next routine canonical check is no earlier than 21:52:22 UTC; asynchronous failure or completion evidence may justify earlier diagnosis. Paired training remains unsubmitted, and PR monitoring remains handed off.

### November 3, 21:52 UTC — specify paired loss and speed measurements while restore is queued

The sole corrected restore audit remains pending on all four tasks, with no attempts, failures or preemptions at the canonical 21:52:42–44 UTC check. Follow session 37396 remains the observer; the next routine canonical check is no earlier than 22:02:14 UTC. Queueing is not a failure and no replacement job is authorized by the wait.

The [measurement protocol](loss-pair-measurement-protocol.md) specifies complete matched-step loss histories, separate attempt/checkpoint ancestry, and steady-step versus end-to-end time. Static source/configuration checks confirm both prepared runs enable native NaN and Inf loss guards. If stored counter 924 is independently verified, the continuation has 6,706 updates remaining, starts at data example offset 7,392 and logs new losses at labels 924 through 7629. Evaluation callbacks use the next-step counter, so the first scheduled evaluation label is 1525; final forced hooks may repeat label 7629 and must not be silently merged.

The first designated steady-step window is labels 1000–1099, subject to the documented process/recompilation warmup rule. It is a measurement window, not a training limit. The full schedule remains 7630. The protocol retains the existing model-FLOP definition and aggregate hardware peak, reports restart/restore/compile/evaluation overhead separately, and treats single-seed variation as descriptive rather than seed-level uncertainty. No loss-equivalence margin is invented.

No numerical kernels, new job or training were launched for this preparation. The original saved-state check still needs both a direction on unchanged saved Gram/momentum and a full update with fixed synthetic gradients, retaining input immutability. The historical cost script's strict equivalence assertions require adjustment before reuse under the loss-first criterion; no such numerical change is implemented here. All four regional restore/data reports and full-training memory/finite checks remain outstanding.

### November 3, 22:02 UTC — full checkpoint restore and configured data replay verified

The corrected regional audit succeeded on all four tasks, attempt 0, exit 0, with no failures or preemptions. Canonical job/task checks completed at 22:01:52–22:02:16 UTC. All four process reports were retrieved and SHA/size checked. Each process verified all 57 manifest arrays across its four addressable devices: 228 local shard records per process, covering all 16 devices. Restored model, optimizer, step and PRNG bytes match independent TensorStore reads, all checked values are finite, and replicated records with the same path/index have consistent hashes. The [completion record](full-restore-counter-completed.json) preserves report URIs, checksums, task/attempt identities and per-process evidence.

The independently stored trainer counter is 924; checkpoint metadata label is 923. Every process's configured iterator starts at counter 924, example offset 7,392, and its first batch matches a separately constructed iterator at that position. This verifies replay of the configured stream, not identity with an unsaved historical batch. The earlier label/counter assertion failure remains preserved. No training update or checkpoint advancement occurred in either restore audit.

Restore/audit peak process RSS was 29.99, 29.88, 30.09 and 29.99 GiB for processes 0–3. The largest reported per-device JAX allocator peak was 4,340,666,880 bytes; this is not total HBM usage. These measurements include verification work and do not establish full-training memory fit. All four fresh execution-host logging checks and the native process-zero W&B run finished successfully. Source `b55656b4c2fd364292b75aeaee765698cbd9fc56` and its exact SHA were checked before execution. Large checkpoint arrays stayed in us-central2; only bounded reports were retrieved.

The sole follow observer, session 37396, ended naturally with exit 0. No experiment, queue or observer remains active; all old jobs are terminal and will not be restarted. Next prepare the saved-state optimizer diagnostic: directions from unchanged saved Gram/momentum, plus a full refresh/update with a fixed synthetic gradient, retaining finite-output and input-immutability checks. Numerical-equivalence failures remain reported diagnostics rather than a veto on the paired loss experiment. A new numerical entrypoint still requires its own online logging check, online CPU harness and exact publication before one sequential TPU diagnostic. Full-training memory/finite checks and the sequential loss/speed comparison remain outstanding; neither training variant has launched.

## Final decision

Pending pilot, confirmation and checkpoint-probe evidence. No exponent is
selected and no training advantage is claimed. The implementation PR remains
open; its code has not been merged.
