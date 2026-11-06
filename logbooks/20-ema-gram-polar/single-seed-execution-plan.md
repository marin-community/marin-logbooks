# Single-seed execution of the EMA-Gram exponent experiment

**Current execution amendment (2026-11-04 23:35 UTC):** The original-data
campaign below failed before training and remains historical. Execute all 30
pilots from scratch on the fixed HQ/code/math subset described in the final
section. No old pilot is credited for selection on the new data.

The user requested execution of Ali's original scientific comparison on
2026-11-04, using one seed and the highest available parallelism. The immediate
objective is to finish the scientific grid. Further optimizer acceleration
and the unlaunched 64-device synthetic diagnostic are no longer prerequisites.
The protocol is [experiments/ema_gram/README.md](../../experiments/ema_gram/README.md).
This plan preserves the scientific comparison while changing seed count and
execution concurrency.

## Fixed experiment

Use the original Delphi 447M architecture, tokenizer, regional data caches,
data order, context 4096, and global batch 8. Seed 0 controls model initialization
and data order. Preserve full Gram matrices, FP32 Gram/momentum state, every
update refresh, trace normalization, five-step production polar, the default
spectral solver, global clipping 0.1, momentum 0.95, Gram horizon 100, weight
decay 0.1, fixed nonmatrix AdamW, and the original schedule shape. Use the
validated independent-bank `matrix` execution layout; grouped and power
diagnostics are not promoted to the training baseline.

| Alpha | Initial matrix LR values | Ridge values | Pilots |
|---|---|---|---:|
| 0 | 0.01 × 2^(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5) | 0.0001 | 6 |
| 1/8 | 0.0025, 0.005, 0.01 | 0.0001, 0.01 | 6 |
| 1/4 | 0.001, 0.002, 0.004 | 0.0001, 0.01 | 6 |
| 1/3 | 0.0005, 0.001, 0.002 | 0.0001, 0.01 | 6 |
| 1/2 | 0.00025, 0.0005, 0.001 | 0.0001, 0.01 | 6 |

Each pilot runs all 7,630 updates (250,019,840 tokens). The completed alpha=0,
LR=0.007071067811865476 pilot is reused with its original evidence. The other
29 cells are enumerated in [single-seed-pilots.json](single-seed-pilots.json).
The reused pilot is documented in the notebook entry of 2026-10-30 21:43 UTC:
seed 0, all 7,630 finite training rows, final C4-en loss 3.91522193, successful
native attempt and finished online run, and final permanent checkpoint
metadata. Its original configuration and evaluation schedule are retained;
configuration agreement does not prove identical historical cache or initial
array bytes. The alpha=0 direction bypasses spectral powers. Its original
replicated execution remains recorded rather than being relabeled matrix.
The completed 924-update matrix prefix is not a completed pilot. Its cell
starts a new full run from random initialization with a new identity and path;
no terminal job is extended or restarted.

## Parallel execution

All 29 remaining pilots are scientifically independent and may run at once.
There is no artificial one-job or four-job concurrency cap. Assign each free,
compatible allocation to a different unfinished cell, prioritizing nonzero
cells on v4-32 and alpha=0 on v4-8. A v4-32 allocation contains 16 JAX devices
and four hosts; it is not 32 chips. The verified matrix training mesh is
data=4, model=4. Each host requests 120 CPUs, 256 GiB RAM and 50 GiB disk.

The first dispatch supports the established training topologies, v4-8 and
v4-32. Other ready TPU topologies should be qualified for full training while
these jobs run; do not count unvalidated hardware as productive capacity.
A large slice is not assumed to split into independent smaller slices. The
dispatcher must not change or release another job's allocation. GPU execution
requires a compatible data/runtime setup and an allocation holder that survives
worker success or failure before submission; no GPU launch is prepared here.

The user explicitly authorized cold queueing on 2026-11-04: enqueue all
29 remaining cells at normal batch priority, prioritizing elapsed completion
time. This supersedes the earlier ready-only restriction. Inspect ready
capacity first, then submit every remaining cell without waiting for a free
snapshot. The initial requests are 24 v4-32 jobs and five v4-8 jobs, totaling
404 v4 JAX devices if every requested job is allocated simultaneously. This
is requested capacity, not an allocation or an ETA. Preserve any queued job;
do not cancel or resubmit to chase a newer snapshot. Queued jobs do not count
as running experiments or completed progress.

Run `submit_single_seed.py --source-commit CURRENT_EXACT_PUBLISHED_HEAD --launch --allow-cold-queue`
from an immutable campaign worktree with its library PYTHONPATH. The fixed
source branch is `codex/20-ema-gram-single-seed-20261104`; the research branch
continues to receive notebook and outcome updates. Record the absolute
execution-worktree and ledger paths in monitoring state before starting. It refreshes fleet
capacity, fills every eligible slot in that snapshot and submits selected
cells concurrently. `scratch/20261104-single-seed-ledger.json` owns all
per-cell guards. The current thread owns dispatch; the initial ready-only watcher was stopped under its idle dispatch lock
after cold queueing was authorized, before any submission. The full-grid
dispatch has no wait for ready capacity. The existing heartbeat inspects
the ledger without starting duplicate submissions or replacing active jobs. Idle v4-8 slots serve alpha=0 first and may then serve nonzero cells
with the source-defined data=4, model=1 mesh. Timing on that fallback is
measured separately from the 16-device baseline.

Use one stable job and artifact identity per cell. Record source commit,
resolved configuration, exact submission command, topology, task/attempt IDs,
online W&B URLs and observed restored counter. Write a launch-started guard
before each submission. An interrupted submission is reconciled against its
canonical identity before any further action; never blindly retry it.

No queue, task, training, wrapper or watchdog wall-clock limit is added.
Preserve default scheduler preemption and retries. Every fresh process has a
fresh ONLINE W&B run, including execution-host non-training logging before
training. Initialize native tracking before optimizer steps. A retry may
restore its own full native checkpoint; only verified model, optimizer,
counter, RNG and data-position restoration supports a resumed-progress claim.

## Selection and confirmation

Retain all pilot results, failed cells and attempts. Select each alpha/ridge
group by final C4-en validation loss using `plan.evaluated_groups`, `expand`
and `lock_settings`. Nonfinite or short runs cannot win. Ties use the
existing deterministic plan order; retain the tied values in the report.
Infrastructure failures remain unresolved cells pending diagnosis; they
are not scientific evidence of a bad hyperparameter. A reproducible
numerical failure may be recorded explicitly as failed with its evidence,
as supported by the selection code. Any replacement requires a distinct
execution version/path and preserved predecessor record. Automatic scheduler
preemption retries stay within the original job. Cross-ridge and best-exponent
selection use the minimum eligible final loss. If expansion remains monotonic
or all added boundary points fail, report the unresolved optimum rather than
claiming an interior winner. If its winner is on the LR-grid boundary,
add a factor-of-two point beyond that boundary and run it to 7,630 updates.
Repeat until the winner is interior. Expansion count is data-dependent and
is not included in the initial 29-cell estimate. Do not prune cells based on
early loss or choose settings using confirmation outcomes.

After all pilot groups and expansions resolve, publish the existing protocol's
validated selection lock: tuned LR/ridge, pilot evidence, digest and one common
held-out loss target chosen before confirmation. Generate confirmations through
`plan.confirmations(lock)` and select only seed 0. Run alpha=0, 1/4 and 1/2,
plus the best pilot exponent if it is 1/8 or 1/3: three or four full runs, in
parallel, each from scratch for 37,335 updates (1,223,393,280 tokens).
Do not run seeds 42 or 62746 automatically. Do not continue pilot checkpoints
into confirmation, whose learning-rate schedule has a different horizon.

Use the alpha=1/4 confirmation as the reference for checkpoint probes,
selected now before confirmation outcomes. Preserve the native checkpoints
at counters nearest 3,733, 18,665 and 33,597 (the saved filename can use
counter minus one); do not silently substitute a different stage. Run
`experiments.ema_gram.probe` under the original protocol using full native
checkpoints. Probes can run alongside other confirmations when their
reference checkpoint and independent held-out batches are available. Compare norm-matched
directions on identical independent held-out batches and a common step-size
grid; retain measured batch losses. Compare the production polar against SVD
on an attention and an MLP matrix. Horizon ablations and 998M scaling remain
conditional follow-ups after the core comparison; their compute is not
included in the estimate below. A single seed cannot establish cross-seed
reproducibility.

## Completion and reporting

A cell is complete only after canonical Iris success, full requested update
count, final evaluation and checkpoint metadata, and finished online W&B
evidence agree. For a resumed run, join verified counter intervals from
predecessor attempts and the successful final attempt; do not require one
W&B curve to contain all updates or hide failed predecessors. A running status or first loss is insufficient. Preserve
incremental local and regional metric logs. Retain failures as evidence;
investigate OOM, distributed and data failures before any new execution.

Use one monitoring owner and one follower per job. First canonical check is
120 seconds after submission; subsequent checks are 570 seconds apart.
Publish actual running jobs/devices separately from queued jobs, completed
cells/updates, failure counts and an ETA based on measured progress. Report
loss versus tokens and time and device-hours with actual hardware counts.
Do not infer a layout-only speedup, training quality or multi-seed uncertainty
from synthetic diagnostics.

At the observed 3.70021 seconds/update on 16 v4 devices for nonzero alpha and
an assumed 0.35 seconds/update on four devices for alpha=0, the remaining
initial pilots plus single-seed confirmations cost approximately 4,269–4,883
v4-chip-hours before overhead and LR expansions. These rates are not yet
measured across all cells. Four simultaneous nonzero runs would take about
85.4 hours of pure training across pilot and confirmation stages, if capacity
stays available and the alpha=0 runs finish alongside them. This assumes additional concurrent alpha=0 capacity and excludes probe
time, overhead and expansions. This is a scenario,
not a reservation or delivery promise. More compatible parallel allocations
shorten the pilot stage; confirmation still depends on completed selection.

## Recovery from deleted training caches (2026-11-04)

The initial campaign failed while loading seven missing Nemotron cache roots.
Their original objects are soft-deleted. The earliest recorded object expires
on 2026-11-11 at 12:56:58 UTC; later objects expire after their respective
seven-day retention windows. StarCoder, ProofPile and all 23 evaluation components are
still readable. No pilot from this campaign has been validated as complete.

[The failure record](single-seed-cache-failure.json) preserves the original jobs,
object inventory hashes, deletion times and the failed recovery request. The
current service account's bulk restore request returned HTTP 403 because
`storage.buckets.restore` was denied. No object was restored. Do not retry through
another identity or API route without an authorized access change.

An authorized storage administrator can apply the seven request bodies in
[single-seed-cache-restore-requests.json](single-seed-cache-restore-requests.json)
to the recorded `POST` endpoint. Each request selects one exact cache prefix and
its November 4 deletion window, with `allowOverwrite=false`. Record the returned
operation names and wait for successful completion before data validation.
[Google's bulk restore reference](https://cloud.google.com/storage/docs/json_api/v1/objects/bulkRestore)
lists the required `storage.buckets.restore`, `storage.objects.restore`, and
`storage.objects.create` permissions. Restoration stays inside the original
regional bucket; it assigns new generations to the original bytes.

Then run, from the research worktree:

```bash
uv run --no-sync python logbooks/20-ema-gram-polar/check_campaign_caches.py \
  --full --output scratch/campaign-cache-restored-validation.json
```

This must pass all 32 components against the original name/size/CRC catalog.
The initial negative check passed 25 existing components and rejected exactly
the seven missing ones. A passing metadata check must be followed by a real
production-data loading check with online tracking before a replacement launch.
Submission now runs the full catalog check; workers repeat bounded metadata
readback and first/last chunk identity checks before the logging check and trainer.
These checks detect missing data but do not reserve immutable data or prevent
a later external deletion.

Preserve the original source, guards, job identities, failed attempts and W&B
runs. The original observer has now confirmed all 29 jobs FAILED and ended naturally
with exit 0; all followers are also terminal. The observer record is
`/Users/kaiyuew/Downloads/Project/marin/ema-gram-single-seed-20261104/scratch/20261104-single-seed-observers.json`.
After data recovery and validation, publish a new source and artifact version,
then submit only failed cells under new job identities and a separate ledger.
Do not rerun the original guarded dispatcher or treat infrastructure failure
as a scientific result when selecting learning rates.

### Requested minimal IAM grant

The user authorized granting recovery permission to the current service account.
A permission check confirms it already has `storage.objects.restore` and
`storage.objects.create`, but lacks `storage.buckets.restore` and
`storage.buckets.setIamPolicy`. Reading the bucket IAM policy also returned 403;
no IAM changes were made. An existing authorized administrator can grant the
missing permission immediately. The agent cannot grant it with current credentials.

An administrator should create a custom role containing only
`storage.buckets.restore` in the bucket's owning project, then bind it only to
`gs://marin-us-central2`. Replace `BUCKET_PROJECT_ID` with that verified project;
current credentials cannot read bucket metadata to verify ownership.

```bash
gcloud iam roles create emaGramCacheBulkRestore20261104 \
  --project=BUCKET_PROJECT_ID \
  --title="EMA Gram cache bulk restore" \
  --permissions=storage.buckets.restore --stage=GA
gcloud storage buckets add-iam-policy-binding gs://marin-us-central2 \
  --member=serviceAccount:iris-controller@hai-gcp-models.iam.gserviceaccount.com \
  --role=projects/BUCKET_PROJECT_ID/roles/emaGramCacheBulkRestore20261104
```

These commands follow Google's [custom-role creation](https://cloud.google.com/sdk/gcloud/reference/iam/roles/create)
and [bucket binding](https://cloud.google.com/sdk/gcloud/reference/storage/buckets/add-iam-policy-binding) interfaces.
If the role already exists, inspect it before reusing it; do not overwrite an
unrelated role. This grant is not yet applied.

The production-data check remains to be implemented after restoration. Its
required pass condition is successful native loading of every configured
training and evaluation component and a deterministic global batch of eight
4096-token examples (32768 tokens total) at the unchanged seed and mixture, with online
tracking initialized before numerical execution. Publish its source and
evidence before recovery submission. It must retain the original tokenizer,
packing, document boundaries, shuffle and data order.

## Fixed-subset execution amendment (2026-11-04 23:35 UTC)

The user requested a fixed-size retokenized subset and autonomous execution.
The original regional source probe failed HTTP503 and a tokenized mirror failed
exact decode/reencode. Use raw text from Fredithefish/Nemotron-CC-HQ-20B pinned
at revision 1bcdb7f17b10c759bd70ce33f2921ad28cc28505. This changes the original
seven-way web-quality mixture to one HQ component. It is explicitly disclosed
as a changed recipe, not verified byte-identical original data. The choice is
an implementation decision, not an explicit user selection of this repository.

Freeze the source/task manifest in subset-inputs.json. Build 56 distinct web
prefixes of 34,725,120 native Llama tokens each, plus 45,404,354 original code
and 9,988,958 original math tokens. Retain whole final documents and record
overshoot. Preserve aggregate web/code/math shares (10.70716/.25/.055).
All tasks use the same pinned native tokenizer and regional native cache
writer; each task performs full token readback/hash verification. Use the
experiment-owned gs://marin-us-central2/ema-gram-20-fixed-subset/2026.11.04.01
prefix, outside shared raw/ and tokenized/. No bucket/IAM changes.

The distinct hq01 regional probe precedes 58 concurrent normal-batch CPU build
jobs. Each guard is permanent even on ambiguous submission. Exactly one log
follow per accepted job; short CPU data stages use canonical checks after
120 seconds then60 seconds. Training checks remain120 then570 seconds.
Require canonical success and ONLINE W&B success before ledger assembly.
After assembly verify native mixture replay, all23 unchanged evaluation caches,
and per-component capacity for the entire37,335-update confirmation schedule.
No model training starts before this actual regional loader check passes.

Restart all30 seed0 pilots, including alpha0, on this immutable shared dataset.
Keep model, optimizer numerics,4096 context, global batch8, learning-rate grid,
7,630 pilot steps and37,335 confirmation steps unchanged. Old losses remain
context only. Normal batch queueing and all compatible parallel allocations
are authorized. The new request is24 v4-32 plus6 v4-8 jobs (408 requested JAX
devices, not actual allocation). Version2026.11.04.03 and new job/ledger
identities must be used. Never reset old failed guards or extend the924prefix.
After pilots, apply the same adaptive LR boundary expansion, selection lock,
common-loss target and3–4 seed0 full confirmations from scratch.

No queue, task, training, or watchdog wallclock limits; retain default
preemption/retries, all attempts, and fresh online tracking per process.
GPU allocation retention remains mandatory before any GPU launch.

## Qualified subset launch (2026-11-05 00:21 UTC)

All58 builds, native assembly and regional26-component loader qualification
passed. The exact data and source bindings are in subset-training-validation.json.
Launch from detached worktree `/Users/kaiyuew/Downloads/Project/marin/ema-gram-subset-seed0-20261105`,
with its own `scratch/20261104-subset-training-ledger.json` and observer state.
Publish the launch commit on `codex/20-ema-gram-subset-seed0-20261105`; keep
this worktree/source immutable while research notes continue on the research branch.
Normal batch queueing is already authorized for all30 cells. Requested408 v4
devices do not imply allocation or an ETA. Existing failed campaigns and all
data-stage identities/guards remain terminal and are never reused.

## Submission transport recovery (2026-11-05 03:04 UTC)

The first subset launch failed before any job was created; canonical
reconciliation of all30 exact IDs returned no jobs. Its16b4aa568 worktree and
guards remain immutable. Use new execution worktree
`/Users/kaiyuew/Downloads/Project/marin/ema-gram-subset-seed0-r02-20261105`
and publication branch `codex/20-ema-gram-subset-seed0-r02-20261105`.
New jobs are `ema-gram-20-subset-seed0-r02-NN-20261105`, with ledger
`scratch/20261105-subset-training-r02-ledger.json`. Upload at most8 request
packages at once; this does not cap queued/running experiment concurrency.
Dataset, numerical code, cells and artifactversion2026.11.04.03 are unchanged.

## Settled baseline preparation amendment — 2026-11-05 18:35 UTC

Predeclare **4.0** as the common held-out C4-en loss threshold for all seed0
37,335-update confirmations. This round threshold is below the best validated
initial-pilot loss, 4.1497836113. It is chosen before any long-run outcomes;
it is not an estimate that every exponent will reach the target. Each run must
finish all 37,335 updates even if it crosses early. A run that never crosses
is reported as not reaching the threshold within its schedule, without an
extrapolated crossing time. The threshold does not select the remaining pilot
settings or stop a run.

The six complete alpha0 pilots have an interior winner at LR0.003535533905932738.
That exact seed0 configuration may be prepared for independent early execution
while recursive expansions determine other settings. The
[frozen limited protocol](alpha0-long-run-protocol-draft.json) retains its
original filename for continuity; its status is now `frozen`. This amendment
permits only the settled alpha0 baseline. It does not declare a global selection
lock or permit confirmation outcomes to influence pilot selection. Preparation
is complete through controller and logging qualification, protocol declaration
and [baseline adoption checks](alpha0-long-run-adoption-preflight.json).
No long run has been submitted. Execution still needs its clean, published
immutable worktree, distinct campaign identity, full regional catalog check,
permanent launch guard and one observer.

After all recursive expansions validate, the full confirmation submitter still
requires every original global selection and completion check. Supply
`--adopt-baseline <tracked-receipt.json>` to both the submitter and observer.
The receipt contains the original `source`, `campaign`, relative `paths` for
`bundle`, `completions` and `protocol`, and a frozen copy of its accepted `ledger`.
The adapter reads those inputs from the original Git commit, replays baseline
selection, regenerates the final selected baseline at its original artifact
version, and verifies exact configuration, common target and accepted command.
A missing receipt after the early-start amendment, ambiguous acceptance or
changed source input rejects submission. No receipt exists before a real
baseline is accepted.

The final campaign ledger records `confirmation_accounting`: one adopted
baseline with its actual source, configuration and job, plus the newly submitted
confirmations. The adopted baseline keeps its existing observer. Its acceptance
is not scientific completion; all long-run metric, checkpoint, ONLINE and
terminal-attempt checks remain required. Use that accounting record when
collecting final results rather than the unused alpha0 candidate in the new
campaign's configuration bundle.

## Baseline execution and recursive expansion update — 2026-11-06

The user authorized execution of the settled baseline. Alpha0 seed0 was accepted
once and its original attempt, execution-host ONLINE logging and finite native
training were validated. The earlier preparation-only status above is historical.
Use [the accepted receipt](alpha0-long-run-accepted.json) for exact-once adoption;
retain the baseline's original observer and complete37,335 updates at target4.0.

R01 through R04 are validated and their observers retired naturally. Reuse their
completion caches. R04 leaves alpha0.5/ridge0.0001 at its lower LR boundary;
[accepted R06](subset-expansion-r06-launched.json) tests LR0.0000625 under the
same pilot schedule. R05 still requires terminal validation. Global selection,
new nonbaseline confirmations and the prescribed scientific probes remain gated
on the complete recursive selection protocol.

At03:12 UTC, R05 completion validated the alpha0.5/ridge0.01 group's interior
winner at LR0.001 (C4=4.3975324630737305). No further expansion for that ridge
is required. R01 through R05 are complete; only accepted R06 remains in the
recursive pilot search. R06 has finite native training progress. The full global
selection gate remains pending its outcome and any resulting boundary expansion.

At 07:00 UTC the baseline's original observer recorded success. The full
37,335-step completion subsequently validated with final C4 3.792296886444092;
first recorded target crossing is label 33,999 (34,000 updates), C4
3.9549849033355713. Reuse [its completed evidence](alpha0-long-run-completed.json)
and original accepted receipt for exact-once adoption. Never repoll or restart
completed baseline identities. Only R06 remains active; the global selection,
remaining confirmations and prescribed scientific probes are still pending.

On2026-11-06 the user authorized immediate parallel startup of all three
remaining planned long runs. The frozen parallel-long-runs-r01-amendment.json
supersedes the global-wait requirement for these three starts only. Use the
current completed-pilot choices for alpha0.125/0.25/0.5; alpha0.5 is provisional
while R06 continues. Keep the predeclared target4.0 and complete37,335 updates
from scratch. Exclude long-run outcomes from pilot selection. Reconcile every
accepted identity before any later confirmation submission; never duplicate the
completed alpha0 or a matching accepted nonzero configuration. The final global
selection record and prescribed scientific probes remain required.

2026-11-06 10:42 UTC execution: all three amended seed0 long runs are ACCEPTED and canonically PENDING (four tasks each, zero failures/preemptions). Execution source923d79510475df3ed0fef53d40b3290ae4c90fce; sibling ema-gram-confirm-seed0-r01-20261106; campaign ema-gram-20-confirm-seed0-r01-20261106; sole observer99712/PID96635. Reuse its snapshots and existing followers. Startup collector scratch/collect_parallel_r01_startup.py is prepared; run once only after RUNNING plus finished host checks and native URLs. Reconcile the accepted receipt before any future global submission to avoid duplicates.

2026-11-06 11:39 UTC: R06 completion passed (C4=5.104428291320801;
7,630 finite metric rows, stored counter 7,630, 57 array metadata entries, four
original successful attempts, finished ONLINE). All 37 pilots now replay to
interior group winners with no further expansion. Alpha0.5 remains ridge0.01,
LR0.001, matching its accepted long run. Reuse completed R01–R06 and baseline
caches; no active pilot helper is needed. Publish final selection/accounting
without duplicating any accepted long run. The original long-run observer
records alpha0.5 RUNNING and alpha0.125/0.25 PENDING; startup validation and
three prescribed reference probes remain pending.

2026-11-06 11:43 UTC: final pilot-only selection is locked in
subset-final-selection-lock.json, with explicit alpha0.125 amendment and all four
accepted identities in subset-final-confirmation-accounting.json. Exact original
source/configuration/target/command replay passed. Alpha0.5 is no longer
provisional. New submissions and new observers required: zero. The ordinary
global accounting path rejects an existing parallel acceptance receipt. Continue
only the original long-run observer, startup verification, full completion
validation and the three prescribed alpha0.25 scientific probes.

2026-11-06 13:55 UTC: alpha0.25 and alpha0.5 are canonically RUNNING on four
tasks each; alpha0.125 remains capacity-queued, with zero failures/preemptions.
See parallel-long-runs-r01-two-running.json. Alpha0.25 native startup remains
unverified after its original follower exited with Bad Gateway while queued.
Do not replace that follower. Obtain bounded source/attempt-bound evidence
before startup validation; preserve the collector all-three-running gate and
one-time caches. No new submission or observer is required.

2026-11-06 17:57 UTC: all three nonzero long runs are canonically RUNNING.
Startup validation passed for 12 original attempt 0 task/worker joins, 12 finished
ONLINE host checks and exact native source/configuration. Finite training
labels are 7/3489/5680 for alpha 0.125/0.25/0.5. See
parallel-long-runs-r01-started.json and cached
scratch/20261106-parallel-r01-startup/validated.json. Do not repeat startup
collection or finished-host APIs. Reuse observer 99712/PID 96635 and prepared
scratch/observe_parallel_r01_online.py for unfinished native progress only.
Full 37,335 outcomes and three alpha 0.25 checkpoint probes remain pending.

2026-11-06 18:15 UTC: reference-probe-config-repair.json records a probe
configuration decoding repair and offline validation. Existing source-bound
probe qualification does not cover the edited source; revalidate it and audit
four-host restore/SVD array access before launch. The first alpha0.25 prescribed
checkpoint has validated permanent metadata and 57-array schema; stored counter
and full restore remain pending. All three scientific probes remain unlaunched.

2026-11-06 18:24 UTC: first reference checkpoint stored counter3733 passed a
cached read-only scalar check. Probe direction comparisons now reduce global
arrays in JAX before host scalar conversion. Requalify this edit with the
extended sharded CPU oracle before scientific launch; no four-host probe has
run. Reuse completed b33ed91cf logging/SVD qualification caches without APIs.

2026-11-06 18:27 UTC: edited probe logging/SVD and eight sharded scalar
comparison cases passed on four CPU devices at source843b2446b. Updated
reference-probe-svd-preflight.json binds exact source hashes and finished ONLINE
runs. Actual four-host TPU/full checkpoint restore remain untested. Next prepare
immutable scientific request inputs, permanent submission guards and one observer;
launch the available counter3733 probe when all execution preflight gates pass.

2026-11-06 18:34 UTC: first prescribed counter3733 probe accepted once as
ema-gram-20-reference-seed0-r01-20261106-00, source2fbdb4290, v4-32/four hosts.
See reference-probe-r01-accepted.json. Permanent guard
scratch/20261106-reference-counter-3733-launch.json must never be reset or reused.
Reuse sole observer91651/PID70306 and follower70493; startup/restore/probe outputs
remain unvalidated. Remaining counter18665/33597 probes have not been launched.

2026-11-06 19:06 UTC: counter3733 reference probe startup validated once;
see reference-probe-r01-started.json and
scratch/20261106-reference-r01-startup/validated.json. Four
original attempt0 UIDs/workers, four finished ONLINE logging checks and four
native ONLINE runs match source/config/request through original task-tagged
follower URLs. W&B config has no task UID; reconcile wrapper attempt archives
at completion. Full restore and scientific outputs remain unvalidated. Reuse
observer91651/PID70306/follower70493, never repeat unchanged startup/host reads.
