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

2026-11-06 20:54 UTC: first prescribed counter3733 probe COMPLETED_VALIDATED;
see reference-probe-r01-completed.json and reference-probe-r01-terminal-audit.json.
Four original attempt0 successes, four finished ONLINE native runs, exact
wrapper/archive UID/source/config joins, generation-bound diagnostic readbacks,
27 finite paired-loss rows and 18 finite decomposition rows per host passed.
All four hosts returned identical loss and direction diagnostics over the same
eight batches; they are not independent scientific samples. Full TrainerState
load with allow_partial=False and restored counter3733 is validated for this
probe, without an independent all-array finiteness or training-resume claim.

Observer91651/PID70306 is naturally terminal. NEVER repoll/restart/resubmit this
completed probe, its tasks/native runs/hosts/follower/archives. Reuse research
scratch/20261106-reference-r01-completion/verified.json and execution completion
cache; retain consumed launch guards and historical transport/health evidence.
Continue only the three unfinished long runs under observer99712/PID96635 and
scratch/observe_parallel_r01_online.py once per wake. Later counter18665/33597
probes remain unlaunched and require exact checkpoint/request validation and
fresh permanent guards. Remaining scientific work: all three nonzero full37335
outcomes, two later probe results and final conclusions. The first probe's small
paired loss changes alone do not establish a preferred exponent.


2026-11-07 11:29 UTC: second prescribed counter18665 probe accepted once;
see reference-probe-r02-request.json and reference-probe-r02-accepted.json.
Exact checkpoint metadata, stored counter, unchanged qualified probe source and
full 26-component catalog passed. Immutable execution source
6ec7fa98b842cdff03b46986d00e3ee4dc735615; campaign
ema-gram-20-reference-seed0-r02-20261107, four hosts/v4-32. Permanent guard
scratch/20261107-reference-counter-18665-launch.json is consumed; never resubmit.
Sole observer51423/PID27858/follower28032 owns canonical observation.
Startup, actual four-host restore and scientific outputs remain unvalidated.

Prepared scratch/collect_reference_r02_startup.py; run its local readiness mode
only after the original observer saves its first canonical snapshot. Execute
--collect once only when canonical RUNNING4 and all four task-tagged finished
host logging/native URL joins are ready. Then use
scratch/observe_reference_r02_online.py for unfinished native records, sequentially
with observe_parallel_r01_online.py once per wake (both write machine state).
Both new helpers passed syntax checks; neither has executed live collection yet.
An initial local readiness check found no saved snapshot and failed before any
API; its raw error is preserved. The collector now exits explicitly while
waiting for the original observer snapshot.
Never query completed first-probe identities or qualification records. The three
long runs retain observer99712/PID96635; third counter33597 probe remains
unlaunched. Full long-run outcomes, both remaining probe results and final
conclusions remain required.


2026-11-07 11:44 UTC: counter18665 probe STARTUP_VALIDATED_RESULTS_PENDING.
See reference-probe-r02-started.json. The one-time collector passed four original
attempt0 UID/worker joins, four finished ONLINE host logging checks and four
running native ONLINE records against exact source/config/checkpoint/request.
Reuse scratch/20261107-reference-r02-startup/validated.json; never repeat startup
task or finished-host API reads absent changed attempt/failure lineage. W&B
config has no task UID, so exact wrapper/archive joins remain required at completion.
Full actual restore and scientific outputs remain unvalidated; native summaries
were empty at 11:44. Original observer51423/PID27858/follower28032 remain sole owners.

observe_reference_r02_online.py passed its first live invocation. Run it and
observe_parallel_r01_online.py sequentially once per wake with the prescribed
PYTHONPATH and .venv/bin/python; both write machine state. Completed first probe,
pilots, baseline and qualifications remain cached without API calls. Third
counter33597 probe remains unlaunched. Full three nonzero training outcomes,
second/third probe results and final conclusions are still required.


2026-11-07 11:52 UTC: counter18665 probe COMPLETED_VALIDATED (second of three).
See reference-probe-r02-completed.json and reference-probe-r02-terminal-audit.json.
Four original attempt0 successes, zero failures/preemptions, four finished ONLINE
native records, exact wrapper task/UID/source/config/checkpoint/counter joins,
four bounded task archives and 24 permanent generation-bound readbacks passed.
Each host has 27 finite paired-loss rows over the same eight held-out batches
and 18 finite polar/SVD comparisons. Statistics were independently recomputed;
all loss/direction diagnostics agree exactly across hosts, which replicate the
same batches. Full TrainerState allow_partial=False/counter18665 restore is
validated for this probe; no independent57-array finiteness or resume claim.

Observer51423/PID27858 and follower28032 exited naturally. NEVER poll/restart/
resubmit this completed probe or query its tasks/native/hosts/archives. Reuse
scratch/20261107-reference-r02-completion/verified.json and execution completion
cache; observe_reference_r02_online.py already exits from completion cache before
owner/API checks. No probe helper is needed each wake. Preserve all consumed
guards, immutable sources and raw evidence. Continue ONLY
observe_parallel_r01_online.py once per wake under observer99712/PID96635.
Third prescribed alpha0.25 counter33597 probe remains unlaunched. Three nonzero
full37,335-update outcomes, third probe results and final conclusions remain
required; do not select a preferred exponent from this checkpoint alone.


2026-11-08 05:03 UTC — third prescribed checkpoint probe accepted

The third alpha0.25 checkpoint probe, at counter33597 (zero-based label33596),
was accepted once by Iris at 05:02:58 UTC. Submission acceptance does not
establish startup or successful restore. The permanent checkpoint passed the exact
57-array metadata schema check and a read-only stored-counter check. All 26
regional data-catalog checks passed. The probe uses the unchanged qualified
source and configuration bound by reference-probe-svd-preflight.json,
with eight held-out Wikipedia batches, 27 paired-loss
comparisons and 18 attention/MLP polar-versus-SVD comparisons.

Execution source109598125c54094a5f921efeb3dbd4c29ad5140b is published in the
dedicated ema-gram-reference-seed0-r03-20261108 worktree. Campaign
ema-gram-20-reference-seed0-r03-20261108 has one job requesting v4-32 across
four hosts. It retains default preemption retries and has no queue, task,
training or wrapper wall-clock limit. See reference-probe-r03-request.json
and reference-probe-r03-accepted.json for exact inputs and launch arguments.
The counter33597 launch guard is consumed; never make a duplicate submission.
Iris retains its default retries after preemption.

Sole observer session91778/PID62794 polls canonical job state; original
follower PID62975 streams logs tagged by task and attempt from all four hosts.
Startup validation, actual full TrainerState restore and scientific outputs
remain pending. The prepared scratch/collect_reference_r03_startup.py reads only local
readiness by default. It uses execution scratch/ema-gram-20-reference-seed0-r03-20261108-observers.json.
Invoke --collect once after that saved canonical snapshot shows all
four tasks running and the original follower records each task’s completed
ONLINE logging check and native run URL. Do not repeat completed startup reads
unless an attempt changes or the scheduler records a new failure.

The first two probes remain completed and use cached evidence without remote
queries. The three long runs retain observer99712/PID96635. At 05:00 UTC their
training labels were32637/33621/34858 for alpha0.125/0.25/0.5, with finite
losses and no investigation flags. Each long run must complete 37,335 updates and its frozen outcome-validation gates. The
third probe results and final scientific conclusions remain required.


2026-11-08 07:24 UTC — third prescribed checkpoint probe startup validated

The third alpha0.25 reference probe, at stored counter33597 (training label33596),
passed startup validation. The original observer saved four RUNNING tasks with
zero failures or preemptions. Four original attempt0 task UIDs map to four distinct
workers. Each worker completed its ONLINE logging check and started a native
ONLINE probe run. Source109598125c54094a5f921efeb3dbd4c29ad5140b, the logged
training configuration, checkpoint path and probe request match the frozen inputs.
See [reference-probe-r03-started.json](reference-probe-r03-started.json) for task
identities, run URLs and validation evidence.

The one-time startup collector saved
scratch/20261108-reference-r03-startup/validated.json. Reuse its task snapshots
and finished ONLINE logging-check records unless a new attempt or failure is
recorded. W&B configs omit task UIDs; the original task/attempt-tagged follower
joins the runs to tasks. Completion validation must match each archive's wrapper
attempt UID to its original scheduler attempt UID, then verify the permanent
objects at the storage generations recorded by the wrapper.

Native summaries were empty during startup collection. The first native-only
check at 07:24 UTC found all four runs finished with probe/success=1 and cached
the records. Scheduler SUCCEEDED status, successful original task attempts and
complete output validation remain pending. The claimed full TrainerState restore
and paired-loss and polar/SVD outputs still need verification from those archives.

Observer91778/PID62794 continues scheduler polling; original follower62975 retains
log collection. At each five-minute automation wake, run
scratch/observe_reference_r03_online.py and observe_parallel_r01_online.py
sequentially for native W&B progress. They reuse terminal native records.
The long runs remain healthy under observer99712/PID96635; at 07:23 UTC their
training labels were34881/35646/36768 for alpha0.125/0.25/0.5. Each must reach
37335 total updates (final label37334) and pass outcome validation. The first two
probes remain complete and use cached evidence. The third probe results and
final scientific conclusions remain required.


2026-11-08 07:31 UTC — third prescribed probe completed and validated

The third reference checkpoint from training with Gram exponent alpha=0.25,
at stored counter33597 (zero-based training label33596), passed the prescribed four-host probe. Iris recorded four
original attempt0 successes, zero failures and zero preemptions. All four native
W&B runs finished ONLINE. Exact wrapper task/attempt UIDs, source, configuration,
checkpoint and counter match the startup records. Four bounded task archives
and 24 permanent objects passed generation-bound readback and checksum checks.
See [reference-probe-r03-completed.json](reference-probe-r03-completed.json) and
[reference-probe-r03-terminal-audit.json](reference-probe-r03-terminal-audit.json)
for results, identities, receipts and validator source.

Each host returned 27 finite paired-loss rows: nine diagnostic exponent/ridge
settings at three perturbation sizes, evaluated on the same eight held-out
Wikipedia batches. These settings vary only the diagnostic directions; the
checkpoint remains fixed. Each also returned 18 finite production-polar-versus-SVD
comparisons: attention and MLP matrices at nine settings. These compare the
production Newton–Schulz polar approximation with an SVD calculation of the
optimizer direction. Paired mean, standard deviation and standard error were
independently recomputed over eight batch-level candidate-minus-baseline losses; the offline validator
rejected eight corrupted-output cases. Loss and direction diagnostics are exactly
equal across the four hosts. The hosts replicate the same batches and provide no
additional independent samples. Timings and device-memory records are specific
to each process.

This validates full TrainerState restore with allow_partial=False and counter33597
for the four-host probe: the verified frozen source requires strict checkpoint
loading and checks the restored counter before producing the completed report.
It does not establish an independent finiteness audit of all checkpoint arrays,
an optimizer update, data-position replay or training resume. Baseline mean
Wikipedia loss was 3.5759398341178894. Every tested perturbation had a positive
candidate-minus-baseline mean loss change, ranging across the 27 settings/size combinations from 0.0000848770 to
0.0007255375. These local perturbation results alone do not establish a preferred
training exponent. Polar-versus-SVD cosine ranged from 0.8638303 to 0.9997517;
relative direction error ranged from 0.0227070 to 0.5213540. Relative error is the
Frobenius norm of the direction difference divided by the SVD direction norm.

All three prescribed reference probes at counters3733,18665,33597 are now
completed and validated. The third probe's sole observer (session91778, PID62794) and
original log follower (PID62975) exited naturally. Reuse
scratch/20261108-reference-r03-completion/verified.json and the sibling
ema-gram-reference-seed0-r03-20261108 worktree cache at
scratch/ema-gram-20-reference-seed0-r03-20261108-00-completion/verified.json; never query completed probe jobs, tasks, native records,
logging checks or archives again, or restart/resubmit them. Source
109598125c54094a5f921efeb3dbd4c29ad5140b and the consumed counter33597 launch
guard remain unchanged.

Continue observe_parallel_r01_online.py once per wake under the original
running observer (session99712, PID96635). At 2026-11-08 07:30 UTC, alpha0.125/0.25/0.5 training reached
labels34986/35742/36858 with finite losses and no investigation flags. Each run
must finish 37,335 updates (final label37334) and pass outcome validation, including complete finite metric histories,
final held-out C4 loss, stored counter and checkpoint metadata, exact source/config,
finished ONLINE records and original task success. The
three nonzero-alpha training outcomes and final scientific conclusions remain required.

## 2026-11-08 08:25 UTC — alpha0.5 failed during shutdown; artifact audit pending

The original observer's saved 08:17 UTC snapshot records the seed0 alpha=0.5
long run as failed at 08:10 UTC. Iris recorded task2 exit139/SIGSEGV and three
coscheduled sibling failures, with zero preemptions. Its task2 error also quotes
stderr raising signal6; both records are preserved without resolving that discrepancy.
All four task records have their original attempt0 UIDs, matching startup evidence.
See [parallel-long-runs-r01-alpha05-failed.json](parallel-long-runs-r01-alpha05-failed.json)
for scheduler and task records, native metadata, bounded attempt logs and checksums.

The logs report saving the final checkpoint at zero-based label37334 and completing
final evaluation. Process0's JSON logger records held-out C4 loss3.9583799839019775
at that label. This is a recovered local metric: the native W&B run is crashed,
and its summary stops at label37333. Full metric history, final checkpoint metadata
and stored counter still require separate audit. The existing completion requirements
of scheduler success and a finished ONLINE native record are not satisfied.

JAX reported that only three of four processes reached its shutdown synchronization
barrier; process0 was missing. Process0's main-thread stack was waiting for a thread
in Finelog.close, while sibling processes were in JAX shutdown. These observations
show a shutdown synchronization failure; the initiating cause remains unresolved.
They do not establish optimizer numerical instability. The frozen source places
tracker.finish after trainer.train; the running execution source has not been changed.

The same observer (session99712, PID96635) remains active. Its saved 08:17 UTC
snapshots show jobs00/01 (alpha0.125/0.25) RUNNING with four tasks each and zero
failures/preemptions. The 08:19 native helper queried those runs before rejecting
the changed failure count of job02; it did not persist those two native responses
and was not rerun that wake. The latest persisted native labels remain35659/36367
from08:14 UTC. The helper now skips the exact cached failed job02 before API calls.
Four offline checks verify that skip, rejection of changed or unrecognized failure
evidence, and preservation of the running-job observation path.

Continue the native helper once per wake for jobs00/01 and retain the original
observer. Reuse all cached terminal evidence for job02; never restart or resubmit
it. Next, audit recoverable scientific artifacts separately and investigate a
future teardown repair with runtime validation before any new execution. Preserve
all submission guards and completion requirements. The three prescribed probes
at counters3733,18665,33597 remain completed and cached, as recorded in
reference-probe-r01-completed.json, reference-probe-r02-completed.json and
reference-probe-r03-completed.json. Final scientific conclusions remain pending.

## 2026-11-08 08:32 UTC — failed alpha0.5 artifacts recovered with separate provenance

The failed seed0 alpha0.5 execution now has all five audited training metrics for all37,335
updates, with explicit mixed provenance. W&B history supplies37,334 finite rows
for all five training metrics at labels0..37333 and37 C4 evaluations. The original
process0 attempt0 JSON log supplies the missing final label37334 row and final
C4=3.9583799839019775, adding a distinct38th evaluation. Matching repeated local metric events were deduplicated;
the merge found no conflicting values. The combined history passes the existing
metric audit, and all five missing-final-metric negative tests were rejected. This does not
change the scheduler FAILED or W&B crashed states, and does not satisfy the
existing success/finished-ONLINE completion gates. See
[parallel-long-runs-r01-alpha05-recovered.json](parallel-long-runs-r01-alpha05-recovered.json)
for provenance, checksums, collection scripts, checkpoint receipts and audit results.

Read-only checks verified the final permanent checkpoint's generation-bound
metadata listing the expected57-array schema and scalar counter37335. This is not a full
checkpoint byte-finiteness audit or restored training run. The first observed W&B
C4 evaluation at or below4.0 was label35999:36,000 updates,1,179,648,000 tokens,
C4=3.997650384902954,recorded W&B _runtime154756.717511608seconds. No interpolation is
used. Native runtime includes compilation and evaluation, excluding queue time.
The final local C4 event has no W&B runtime; none is inferred.

All new history pages, metadata and counter reads have permanent guards and caches
under scratch/20261108-alpha05-artifact-audit; reuse them without querying completed
evidence again. The repaired monitor passed its first live invocation at2026-11-08 08:29 UTC,
querying only unfinished alpha0.125/0.25; labels35925/36495 had finite training
losses and no investigation flags. Original observer99712/PID96635 was active at that observation.
The shutdown cause and whether the recovered failed-run evidence is sufficient
for the scientific comparison remain unresolved. The prescribed alpha0.25 probes
at counters3733,18665,33597 remain complete. Final conclusions await alpha0.125
and alpha0.25 outcomes and explicit treatment of the failed alpha0.5 execution.

## 2026-11-08 08:36 UTC — alpha0.25 W&B crashed state conflicts with ongoing training

On2026-11-08 at08:34 UTC, the alpha0.25 native W&B record was observed as crashed,
with last logged zero-based update label36495. The original observer's new08:36
snapshot still records four tasks RUNNING, zero failures and zero preemptions.
Four bounded original-attempt log snapshots show continued finite training:
process0 (Iris task3) logged update label36667 at08:35:11, after the crashed-state
observation. The four inspected tasks show training progress at08:34; their
checkpoint-write logs show that temporary checkpoint label36592 committed and
passed its synchronization barrier at08:30. This checkpoint has not been audited.
These observations establish a discrepancy between tracking state and observed
training activity. The initiating cause is unresolved; the inspected intervals do
not establish a training-process exit. See
[parallel-long-runs-r01-alpha025-native-crashed.json](parallel-long-runs-r01-alpha025-native-crashed.json).

The native summary query is cached. The monitor now skips crashed, failed or
killed native records before further API queries, while retaining the original
scheduler observer. Four offline branch checks passed; no second live helper
invocation was made for this wake. Source/config checks passed at the native
observation; the log queries target original attempt0 identities. No tasks were
restarted or resubmitted. All four bounded log snapshots and query guards are
retained under scratch/20261108-parallel-r01-alpha025-native-crash. Future diagnosis
must use new evidence intervals and saved observer snapshots, not repeat these reads.

Completion of the prescribed37,335 updates and the final execution outcome remain
unverified. No offline fallback or replacement W&B curve has been introduced.
Alpha0.125 was running at label35984 with finite training loss at08:34. The
alpha0.5 execution remains failed with recovered artifacts; the three prescribed
reference probes at counters3733,18665,33597 remain complete. Continue the
original observer99712/PID96635 and native observation only for alpha0.125 while
investigating alpha0.25 through new scheduler/log evidence.


## 2026-11-08 09:26 UTC: alpha0.125 tracking crash with continued training

At 09:24:23 UTC, alpha0.125's native W&B record was observed crashed at its
last recorded training label 36593. Labels are zero-based: label 36593 corresponds
to 36,594 updates. The monitor verified the original source and configuration
identity at that observation. These checks do not establish tracking health.
The original scheduler observer's 09:24 snapshot still records all four
alpha0.125 tasks running, with zero failures or preemptions. Four bounded
original-attempt0 log reads show training activity. Task1, the process0 JSON
logger, reached label 36830 with finite loss 3.15659761428833 at 09:25:11, after
the W&B crash observation. The other three task logs show training at 09:24:19.
This establishes a discrepancy between W&B state and continued training;
its initiating cause remains unresolved.

The crashed native record and all four log reads are cached. Seven malformed
JSON events remain preserved in the raw logs and are excluded from the local
metric extraction. The existing native helper now skips the cached crashed
records for alpha0.125, alpha0.25 and the failed alpha0.5 execution before API
calls. The separate scheduler observer99712/PID96635 remains the sole owner of
canonical polling. Further diagnosis uses new bounded log intervals and saved
scheduler snapshots. No task was restarted or resubmitted, and no offline
fallback or replacement W&B curve was introduced.

For alpha0.25, newly cached process0 logs reached final training label 37334,
reported saving the final checkpoint at 09:23:20, and showed final evaluation
in progress at 09:24:05. Its W&B record remains cached crashed. The reported
checkpoint has not been verified; complete metric history, final evaluation
and execution outcome remain unaudited. These logs do not establish scientific
completion. See
[parallel-long-runs-r01-alpha0125-native-crashed.json](parallel-long-runs-r01-alpha0125-native-crashed.json)
for scheduler snapshots, native identity evidence and the bounded task logs.


## 2026-11-08 09:32 UTC: alpha0.25 shutdown abort observed

By 09:32 UTC, four new bounded original-attempt0 logs for alpha0.25 had been
cached. They record a JAX shutdown-barrier failure around 09:27:14 and abort
messages. The barrier reports 3/4 participants and names missing task3; the
service's numbering is retained without equating it to JAX process indices.
The process0 logger is Iris task3. Its background W&B tracker reported
abandoning its finish thread after 120 seconds at 09:27:13. The fatal stack
shows the main thread in Finelog metrics close; background W&B finish also
raised HandleAbandonedError. These observations establish a teardown
synchronization failure. The reason W&B finish did not complete, and its
relationship to the shutdown-barrier failure, remain unresolved. No
optimizer-instability conclusion follows from these logs.

The process0 JSON log records held-out C4 loss 3.844011068344116 at the prescribed
final training label 37334. This is a recovered local metric, not a validated
finished-ONLINE result. A prior log reported completion of final checkpoint
saving at 09:23:20. The complete five-metric history, checkpoint metadata and
stored counter remain unaudited. The latest saved scheduler snapshot predates
the abort and still marks four tasks RUNNING. Terminal scheduler outcome and
original attempt exits await the existing observer. No extra scheduler polling
was introduced. See
[parallel-long-runs-r01-alpha025-shutdown-abort.json](parallel-long-runs-r01-alpha025-shutdown-abort.json).

All four bounded log intervals and the crashed native record are cached; do not
repeat them. Alpha0.125 continues through original task1 logs, reaching label
36905 with finite loss 3.3658785820007324 at 09:29:42. Six malformed JSON events
in that new interval were preserved and excluded from metric extraction.
The original observer, launch guards and execution source remain unchanged;
no task was restarted or resubmitted. The alpha0.5 recovered-artifact evidence
and all three completed reference probes remain cached.


## 2026-11-08 09:39 UTC: alpha0.25 failed with partial artifact recovery

The original observer saved alpha0.25's canonical FAILED outcome at 09:33:54 UTC;
the scheduler finish time is 09:27:20.518. Scheduler task2 reports exit139/SIGSEGV,
while cached stderr also reports signal6. Their relationship remains unresolved.
Three siblings are COSCHED_FAILED. All four terminal task reads match their
original attempt0 UIDs and startup times. Cleared worker references reconcile
with cached startup assignments. The inspected run has zero preemptions and
no additional attempts; no restart or resubmission was performed. No usable
output archive is reported. Previously cached shutdown logs, crashed native
metadata and finished host checks were reused without querying them again.
See [parallel-long-runs-r01-alpha025-failed.json](parallel-long-runs-r01-alpha025-failed.json).

A separate guarded artifact audit validates generation-bound permanent
checkpoint metadata for 57 arrays and stored scalar counter37335. This does
not establish payload completeness, all-array finiteness, restoration or
training resume. One targeted W&B history collection returned 37,088 finite
complete five-metric rows at labels0..37087 and 37 C4 evaluations. Identical
page-boundary duplicates were deduplicated and isolated missing labels read
once. W&B history lacks final label37334. Combining that history with previously
cached complete local JSON events supplies the final label and yields 37,223
finite five-metric rows and 38 evaluations. Labels37140..37251 remain missing
for all five metrics; the unchanged full-history validator rejects the combined
record. The failed 09:14 log collection preserved a local credential-refresh
transport error; its interval was not recovered. No history or tracking state
was rewritten.

Final C4=3.844011068344116 comes from the local label37334 event and has no W&B
runtime. The first observed target4.0 crossing was label33999: 34,000 updates,
1,114,112,000 tokens, C4=3.985281467437744 and reported native runtime
142442.128474992 seconds. Native runtime includes compilation and evaluation
and excludes queue time; no interpolation is used. The failed scheduler state,
crashed native state and incomplete metric history remain explicit. The record
retains scientifically_complete=false and does not meet the prescribed
success/finished-ONLINE/full-history gates. See
[parallel-long-runs-r01-alpha025-partial-recovery.json](parallel-long-runs-r01-alpha025-partial-recovery.json).

Alpha0.125's original task1 reached label36992 with finite loss3.309948444366455
in the 09:34 wake's bounded log snapshot. The sole scheduler observer remains
session99712/PID96635, with a saved 09:33:59 RUNNING snapshot for alpha0.125.
The 09:34 native helper rejected the newly failed alpha0.25 canonical state
before any native API call; its error is preserved, and it was not rerun.
The exact failure is now registered for the existing cache-skip branch.
Remaining work is alpha0.125's outcome, failed-run scientific adjudication,
teardown investigation and qualified future repair, and final scientific
conclusions. Completed native, task, log, history and checkpoint reads stay cached.


## 2026-11-08 10:12 UTC: alpha0.125 completed and validated

This seed0 EMA-Gram run uses alpha0.125, ridge0.01 and learning rate0.0025;
four TPU hosts execute one distributed training run. The original Iris
monitoring process recorded it as SUCCEEDED at 10:02:52 UTC. The
scheduler's job finish time is 10:01:17.446. Four guarded terminal task reads
confirm four successful original attempt0 executions, zero failures and zero
preemptions. Their UIDs, start times and worker assignments match the cached
startup records. No task was restarted or resubmitted.

The earlier 09:24 W&B crashed observation remains preserved. New Iris scheduler
success, four task successes and process0 logs reporting W&B synchronization
completion justified one W&B state read protected by a persistent read guard
and cache. It returned finished on the original W&B run identity, with matching source and scientific
configuration. The startup cache verifies fresh ONLINE tracking with resume
set to never and four finished execution-host logging checks. No tracking
state or history was rewritten. The cause of the earlier tracking interruption
remains unresolved.

A targeted history audit recovered all 37,335 W&B rows for training loss,
update duration, model FLOP utilization, matrix learning rate and Adam learning
rate, plus 38 C4 evaluations. Labels0..37334 count completed updates from zero;
the stored counter37335 counts the same updates from one. All required values are finite; the
unchanged full-history validator passed. Identical pagination duplicates were
deduplicated and isolated boundary omissions were queried once. No local
metric rows were needed to complete the native history. Final held-out C4 loss
is 3.784024715423584, matching the complete local final JSON event. The first
observed C4<=4.0 evaluation was label32999: 33,000 updates, 1,081,344,000 tokens,
C4=3.9806714057922363 and native runtime128269.778556803 seconds. The final
native evaluation runtime is 144768.645205913 seconds. These runtimes include
compilation and evaluation and exclude queue time; no interpolation is used.

Reads pinned to the stored object generations validate permanent checkpoint
metadata for 57 arrays against the reference shapes, types and chunk layout,
and a read-only scalar check returns counter37335. This does not establish all-array byte
finiteness, a full-state restore or training resume. The source commit and frozen input hashes match the accepted submission,
including its configuration and data recipe. The original task lineage, finished ONLINE state,
complete finite history and final checkpoint metadata/counter gates pass.
See [parallel-long-runs-r01-alpha0125-completed.json](parallel-long-runs-r01-alpha0125-completed.json).

All three parallel jobs are now terminal. The original observer
session99712/PID96635 exited naturally; do not restart it or its followers,
or run the live native helper again. Completed task, native, history and
checkpoint reads are cached. The research program remains active: alpha0.25
and alpha0.5 retain their failed-execution classifications, and alpha0.25
retains its 112 missing metric labels. The remaining work is to determine which comparisons the failed runs support
without relaxing the original gates, investigate their shutdown failures,
validate a proposed future repair before any new execution, and write the
final single-seed conclusions. No new run or runtime change was made.
