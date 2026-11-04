# Single-seed execution of the EMA-Gram exponent experiment

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

## Fixed-subset rebuild alternative (2026-11-04)

The user proposed rebuilding a bounded subset after providing a discussion
about deleting shared GCS raw and tokenized data. This opens a second recovery
route; administrator restoration is no longer the only option. Preserve the
original Nemotron v1 categories and mixture unless the user explicitly chooses
a different corpus. The current regional DataKit discovery contains v2/v2.1
normalized sources, not a verified original v1 copy.

Prepare approximately 2 billion tokens once, with component targets proportional
to the original nine mixture weights. A pilot consumes 250,019,840 tokens and a
confirmation consumes 1,223,393,280 tokens. All configurations reuse the same
immutable subset and data seed; dataset preparation cost does not multiply by
the number of experiments. Verify sufficient per-component coverage under the
native mixture schedule, not just the aggregate token count. Keep complete
documents and record the actual final counts rather than cutting a document to
make an exact headline budget. Freeze source files, sampling seed, tokenizer
bytes, document processing and output identities before training. Use a dedicated
experiment prefix in us-central2 outside shared raw/ and tokenized/ directories.

Re-run all 30 seed0 pilots, including alpha0, on the new subset. The old alpha0
pilot cannot serve as its paired baseline. Keep model, optimizer, global batch,
update counts and evaluation data fixed. The later LR boundary expansion,
selection lock and 3–4 full confirmations remain unchanged.

The original public Nemotron source index is readable and lists 31,279 files.
Sample file metadata is readable, but body requests returned HTTP 503 SlowDown.
This establishes a source path, not successful tokenization or a runtime estimate.
Use bounded request-level backoff and validate actual source records from a
regional CPU worker before scheduling bulk processing. Do not impose a job or
training wallclock limit. Production subset construction and actual native loader
qualification are outstanding. No replacement training job has been submitted.
See single-seed-subset-preparation.json for quotas, evidence and launch conditions.
