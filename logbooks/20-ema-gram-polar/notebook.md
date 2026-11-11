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

**Current direction (2026-11-08, latest):** The user authorized minimal-compute
checkpoint supplements and excluded cross-seed work. Execute the bound
[supplement request](supplemental-probe-r02-request.json); do not rerun failed
longruns. Shutdown packaging and larger-model/horizon campaigns remain deferred.

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

Earlier entries: [October 30–31 archive](notebook-archive-20261030-31.md).

Earlier entries: [November 1–3 archive](notebook-archive-20261101-03.md).

### 2026-11-04: grouped-Gram execution starts

The grouped-Gram acceleration diagnostic has started. Canonical checks on
November 4 at 07:34:16 UTC show all four tasks running on attempt 0, with no
failures or preemptions. The reserved v4-32 slice that was previously booting
now hosts these tasks. No cancellation, resubmission or allocation change was
needed. Log-follow session 92765 remains the sole live observer; the next
routine canonical check is due no earlier than 07:43:46 UTC.

All four execution-host non-training W&B checks finished successfully, and all
four numerical workers are online. Their W&B configurations and
generation-pinned regional reports record the four expected source hashes and
published execution commit `03d5bad644d570fc29a43781a6bdcb7362f44185`.
Iris tasks 0–3 map to JAX processes 0,3,2,1. Each report identifies the original
checkpoint and its 21 selected Gram/momentum arrays; this is not a full
trainer restore.

All four partial reports contain directions computed from unchanged saved
state for both matrix and matrix_grouped, plus the matrix baseline's synthetic
gradient update. In these completed portions, outputs are finite and input
fingerprints are unchanged before versus after execution. Three repeated
outputs match one another bitwise within each case/layout/process.
The fixed-state matrix_grouped versus matrix direction comparisons fail
rtol=atol=1e-5 locally on all four processes; this diagnostic alone is not a
training veto. Full-update matrix_grouped results are still absent and every
report has completed=false. Global replica agreement and numerical errors
have not been aggregated. No speedup or training/MFU conclusion follows from
this partial snapshot. All raw timing samples remain in the regional reports;
fixed baseline-first order and three samples limit their interpretation.

[Startup evidence](grouped-gram-startup.json).
### 2026-11-04: startup delay traced to new cloud capacity

The grouped-Gram job waited 134m19s from submission to Iris's first task
assignment. GCP created its queued-resource request after 3m55s, but the
corresponding TPU node was created another 125m24s later. The remaining
node-creation-to-assignment interval was 4m59s. Thus 93.4% of this wait falls
between cloud request creation and node creation. First task setup logs appear
at 07:28:02 UTC; all four workers have online W&B by 07:28:49. The observed task setup and W&B initialization ran after assignment and
cannot explain the preceding two-hour delay.

Three of the user's other jobs received their first assignment in 0.309s
(v4-8), 5.070s (v4-64), and 3.511s (v4-512). Their assigned worker slices were
already ready in the earlier 06:16 fleet snapshot, before those jobs were
submitted. Our required v4-32 topology could not use those other shapes; the
four ready v4-32 slices in that snapshot were occupied. These selected examples
support a distinction between reusing ready capacity and waiting for a new
slice, not a general rule that smaller or larger TPU shapes start faster.

The earlier tier_blocked diagnosis was accurate for that routing snapshot but
incomplete as an explanation of the entire wait. The cloud request already
existed at 05:16:56, and the task-specific routing blockage was absent by 06:50 while the task remained
pending. Iris's booting label can include a queued resource whose TPU VM does
not exist yet. The exact GCP reason within the 125-minute allocation/provisioning
interval remains unverified; there is no evidence here establishing exhausted
quota, fragmentation, or a particular maintenance cause.

For the next distinct experiment, inspect compatible ready capacity before
selecting the TPU shape. Validate memory and full-Gram behavior for any changed
mesh; preserve the existing allocations and record the device count in speed
comparisons. No priority, cluster configuration, job or allocation was changed
for this diagnosis. Raw controller SQL was denied, controller logs timed out,
and Echo access was denied; these evidence limits are preserved.

The same read-only investigation found the grouped-Gram job terminal:
all four tasks succeeded on attempt 0, exit 0, without failures or preemptions.
Sole log-follow session 92765 ended naturally with exit 0. Final numerical reports and
online run completion still require validation; Iris terminal success does not
establish acceleration. Do not poll either terminal job or observer again.

[Diagnosis](grouped-gram-startup-diagnosis.json) ·
[Canonical completion](grouped-gram-canonical-completed.json).

### 2026-11-04: grouped-Gram final comparison

Keep matrix/default as the training baseline. Grouping Gram banks did not
establish a consistent optimizer speedup in this diagnostic, so matrix_grouped
is not promoted. Here matrix/default means the matrix layout with the default
eigensolver. The candidate combines equal-size Gram matrices across weight
banks into one eigensolver batch per size, then splits the eigenpairs back.

Four JAX processes compare seven weight-group direction tensors in two cases:
directions from fixed saved optimizer state, and refreshed Gram/momentum state
plus directions from a fixed synthetic gradient. Each timed replay starts from
the same inputs; returned states are not fed into the next replay, and no model
parameters are updated.

All four tasks succeeded on attempt 0 without failures or preemptions. All four
host logging checks and four numerical W&B runs finished successfully. Final
regional reports passed exact generation-pinned readback, source/checkpoint
identity, finite-output and input-immutability checks. Three repeated outputs
matched bitwise within every case/layout/process. The summary checks that repeated copies of each output shard have identical
hashes and error statistics before counting each shard once: 112 direction
shards, or 133 records when the 21 state outputs are included. The adapted aggregator passed 24 corruption fixtures,
known-answer norm/timing fixtures and actual report-order invariance.

For directions from unchanged saved state, per-process means were
3.307–3.332 seconds for matrix and 3.843–3.872 for matrix_grouped; baseline over
candidate ratios were 0.857–0.861. For a full synthetic-gradient update, means
were 3.569–5.529 and 3.597–5.337 seconds respectively, with same-process ratios
0.810–1.129. All three samples per case/layout remain included, including the
slower first grouped call. Host durations are never summed. Fixed baseline-first
order, only three samples and local completion synchronization limit timing
interpretation. There is no training throughput or MFU result.

All seven direction comparisons fail rtol=atol=1e-5 in both cases. Relative
Frobenius errors are 0.554–0.662% for fixed state and 0.587–0.701% for the full
update; maximum absolute errors are 0.025151 and 0.016699. All 21 state outputs
pass tolerance: 15 have matching hashes, while six Gram outputs have tiny
nonzero errors, at most 2.8422e-14. Directions are not applied to model
parameters in this diagnostic. These differences do not establish loss
degradation, and tolerance failure alone is not the decision against adoption.

The user reiterated that any suitable machine is authorized and then clarified
that TPU should remain the default. The v4-32 restriction was our submission
choice. Future experiments will first inspect suitable ready TPU capacity and
choose a compatible topology; GPU remains an option when justified, not the
new default. A November 4 08:01 UTC snapshot advertised one available reserved
v4-32 slice, 164 free H100s at cw-rno2a, 248 at cw-us-east-02a, and GB200
capacity. These snapshots are not allocations or start guarantees. No new job
has been submitted. GPU work, if selected later, requires allocation retention
through success and failure. Large saved checkpoints remain regional.

An initial local completion observer looked for a nonexistent W&B summary key
`completed`; the harness actually logs `grouped_gram/success`. The observer was
corrected, and all four runs passed the actual success-key check. This was an
observer error, not an execution failure.

[Completion](grouped-gram-completed.json) · [Summary](grouped-gram-summary.json) ·
[Validation](grouped-gram-final-summary-validation.json).

### 2026-11-04: prepare more full-matrix eigensolver shards

The next acceleration trial increases matrix-parallel device count. The prior
16-device grouped run left at most three of the 33 large matrices assigned per
shard, the same maximum as three independent banks. Grouping over 32 or 64
devices lowers that source assignment maximum to two or one. This is a workload
distribution hypothesis, not a measured speedup or kernel schedule. The existing
optimizer implementation and default solver remain unchanged.

The new harness supports ready v4-64 or v4-128 in us-central2, retaining the
original full checkpoint geometry and all three warmed samples. It uses the
same two cases and matrix/default reference. No training or checkpoint progress
is requested. The fleet snapshot at 08:13 UTC had no free qualifying slice;
the launcher refreshes capacity and submits nothing without one. A free v5p-8
in another region does not supply the 32/64-device distribution being tested.
TPU remains the default, with flexible hardware allowed for later interventions.

Fresh online logging and eight-/sixteen-device CPU fixtures finished and were
API verified against the four pinned sources. Both fixtures are finite, leave
state/gradients unchanged, pass checkpoint roundtrip and alpha-zero checks, and
repeat each output bitwise three times. Fixed-state directions match exactly.
The eight-device full-update comparison passes rtol=atol=1e-5; the sixteen-device
comparison fails for directions, with maximum absolute error 0.00635338. All 21
state outputs are exact in both fixtures. The threshold remains unchanged and
is not a sole training veto. Prior 17 online optimizer regressions are reused
only after exact optimizer/test hash checks.

Static aggregation now accepts the selected process/device count, preserving
the old 16-device result exactly. At each of 16, 32 and 64 devices it rejects
24 corrupt fixtures and passes known-answer and report-order checks. These
checks do not establish accelerator speed, peak memory or training quality.
No accelerator job has been submitted at this preparation checkpoint.

[Design](scaled-gram-design.md) · [Preflight](scaled-gram-preflight-validation.json) ·
[Summary checks](scaled-gram-summary-validation.json).

After publication of `4ca4ff8846ea77bf95595d3827c9265f3bb622a9`, the launcher refreshed fleet
capacity at 2026-11-04T08:22:49.015080+00:00. Neither eligible topology had an available
ready slice, so it submitted no job. All old jobs and observers remain terminal.
[Selection evidence](scaled-gram-prepared.json).

## 2026-11-04: Submit the 32-device Gram scaling trial

Iris accepted `/kaiyuew@stanford.edu/ema-gram-20-scaled-gram-v4-64-20261104`
at 16:31:48 UTC from published source `588a02243e29f78e5a7e8522542682cb47acacab`.
The launcher found an available ready v4-64 slice in us-central2 and selected
a request for 32 JAX devices across eight hosts, with mesh axes data=4 and
model=8. Allocation and execution have not yet been checked. This tests whether
grouping the 33 large Gram matrices across more devices reduces optimizer
latency. The independent-bank matrix layout with the default eigensolver remains the
training baseline; no performance result
is available yet. The 21 Gram/momentum arrays come from the original regional step-923 checkpoint;
three Gram arrays each contain eleven large matrices. Full matrix geometry,
precision and solver match the prepared design.

The submit RPC timed out once after 180 seconds, then its built-in retry returned
the job ID. This is a submission transport retry; worker attempts and restarts
remain to be checked. The launcher has recorded its guard against a second launcher submission.
One log follower is attached. No wall-clock limits or preemption retry overrides
were added, and no other allocation was changed. The original 08:22 capacity
snapshot remains preserved separately. See [launch record](scaled-gram-launched.json)
and [design](scaled-gram-design.md).

The first canonical check after the required 120-second wait reports eight
pending tasks, zero failures and zero preemptions. The ready-capacity observation
was advisory and did not reserve the slice. Keep this one submitted job and its
follower while it waits; do not cancel or resubmit it. No worker W&B identity or
performance result is available yet.

## 2026-11-04: Eight scaled-Gram tasks are running

At the 16:44 UTC canonical check, job
`/kaiyuew@stanford.edu/ema-gram-20-scaled-gram-v4-64-20261104` had eight
running tasks, zero completed tasks, zero failures and zero preemptions. The
existing log follower was still active at that check. Each task's stdout identifies attempt 0
and separate W&B URLs for its logging check and numerical worker. These links
have not yet been API-verified, and worker attempt UIDs remain unverified.
No complete performance result is available. Preserve the single submitted
job and collect all eight numerical process reports before comparing optimizer latency.
[Running observation and W&B links](scaled-gram-running.json).

## 2026-11-04: Verify execution-host identity and partial progress

The 16:57 UTC check of `ema-gram-20-scaled-gram-v4-64-20261104` still reports
eight running tasks, zero completed tasks, zero failures and zero preemptions.
Canonical task records identify one attempt per task, all attempt 0. The
Iris task-to-JAX process mapping is 0→7, 1→4, 2→2, 3→6, 4→3, 5→0, 6→5, 7→1.
The assigned slice differs from the launcher's earlier available-slice snapshot;
that snapshot was advisory. No allocation was changed by this monitor.

W&B API checks find all eight host logging runs finished with
`logging_check/success=1`, and all eight numerical runs running. Their recorded
source hashes and commit match the four pinned files and submitted revision.
Generation-pinned regional config/report downloads agree with those identities.
The reports' checkpoint URI, metadata, 21 saved-array paths, byte count and
manifest hash match the prior checkpoint record; this is not a fresh audit of
checkpoint array bytes. Each partial report contains both layouts and comparison
fields for the fixed-state case, but lacks the completed full-update comparison.
Keep the independent-bank matrix layout with the default eigensolver as the
training baseline. Wait for complete numerical process reports, global replica
checks and canonical success before judging speed. [Startup verification](scaled-gram-startup.json).

## 2026-11-04: Complete the 32-device Gram scaling trial

Keep the independent-bank matrix layout with the default eigensolver as the
training baseline. On 32 JAX v4 devices, the grouped layout had lower measured local mean times
for state refresh plus direction computation on all eight processes, with same-process baseline/grouped ratios of 1.130–1.728. Fixed-state
direction timing was mixed: grouping was faster on three processes and slower
on five. This diagnostic establishes neither training speed nor loss quality, so the
user’s adoption criterion of comparable loss and faster training is not yet met.

The Iris resource name v4-64 corresponds here to 32 JAX devices across eight
host processes. The baseline solves each weight group’s Gram banks separately;
the grouped layout combines banks of equal matrix size and dtype before
distributing whole matrices across devices. Seven weight groups provide seven
directions and 21 saved arrays (momentum, left Gram, right Gram); three Gram
arrays each contain eleven large matrices, giving 33 large solves. The full
case refreshes these states from fixed synthetic gradients and computes
directions. It excludes the model forward/backward pass, parameter application,
data loading and training hooks.

| Case | Matrix mean per process (s) | Grouped mean per process (s) | Same-process matrix/grouped ratio |
| --- | ---: | ---: | ---: |
| Directions from saved state | 3.314–3.399 | 2.710–3.677 | 0.910–1.242 |
| Refresh with fixed synthetic gradients, then directions | 3.729–6.264 | 3.234–4.835 | 1.130–1.728 |

These are ranges across eight processes, each with three warmed samples.
Compilation/first execution is separate, all samples are retained, and host
times are never summed. Each timed call blocks until its local returned outputs are ready.
The baseline always ran first; local output completion
does not align timing entry across hosts. The eight processes are not independent
experimental seeds.

This trial uses twice the devices of the completed 16-device experiment.
Multiplying each process's grouped mean by 32 gives 86.72–117.65 device-seconds
for saved-state directions and 103.49–154.73 for the synthetic full update.
The prior 16-device matrix baseline gives 52.91–53.32 and 57.10–88.46 by the same
calculation. These are local-time cost proxies across separate trials, not
end-to-end cost measurements or paired cross-trial estimates. They provide no
evidence of improved device efficiency.

All eight tasks succeeded on attempt 0 with exit 0, zero failures and zero
preemptions; Iris finished at 17:11:31.968 UTC on November 4, 2026. Source is
`588a02243e29f78e5a7e8522542682cb47acacab`; job is
`/kaiyuew@stanford.edu/ema-gram-20-scaled-gram-v4-64-20261104`. The sole follower
87880 ended naturally with exit 0. All sixteen host-check/worker W&B runs
finished successfully and match the recorded source identities. Regional
report/config generations, checksums and exact readbacks were verified.

Every process and layout is finite, preserves state/gradient inputs and
repeats each output bitwise three times. Replica hashes and statistics agree
before deduplicating 224 direction records and 245 full-update records.
The grouped outputs are compared with independent-bank matrix outputs from
the same saved state and, for refresh, the same synthetic gradients. Relative
Frobenius error is ||grouped − matrix||_F / ||matrix||_F; ranges span the seven
weight-group directions after global shard aggregation.
Direction relative Frobenius errors are 0.557–0.664% for saved-state directions
and 0.586–0.698% for the full update; all seven directions fail the unchanged
elementwise criterion |grouped − matrix| <= 1e-5 + 1e-5 |matrix| in each case. Maximum absolute errors are 0.01817089
and 0.02316278. All 21 state outputs pass tolerance: 15 have identical hashes
and six have tiny differences, at most 5.68434e-14. Directions were not applied
to model parameters, so these differences do not establish loss degradation.
Under the user’s loss/speed criterion, this diagnostic tolerance alone does
not decide training adoption.

The original regional checkpoint supplied 21 Gram/momentum arrays; this is
not a full trainer restore. Full geometry, every-update refresh, source precision
and default solver remain unchanged. The static join passed actual report/row
order reversal and reused the existing 24 mutation fixtures per topology.
A local validation typo (`comparison.records` instead of `comparison.shards`)
was corrected before the order check passed; no numerical worker failed.

The 64-JAX-device topology (Iris v4-128) remains an untested acceleration hypothesis: assigning
33 large matrices would reduce the grouped maximum from two matrices per shard
to one. This is a source assignment count, not a runtime-kernel or speed claim.
Any next trial needs a distinct identity and published launch plan; preserve
the completed job and its guard. No further job or training continuation has been
submitted, and the completed 924-update training baseline is not extended.

[Completion](scaled-gram-completed.json) · [Summary](scaled-gram-summary.json) ·
[Validation](scaled-gram-final-summary-validation.json) · [Design](scaled-gram-design.md).

## 2026-11-04 17:27 UTC — prepare the distinct 64-device Gram trial

The [completed 32-device trial](scaled-gram-completion-note.md) reduced grouped
full synthetic-update local means on all eight processes, with mixed fixed-state
results and no demonstrated device-efficiency gain. Keep `matrix/default` as the
training baseline. Test the remaining source-assignment hypothesis on 64 JAX v4
devices: the grouped maximum falls from two large matrices per device to one.
This count does not promise a runtime gain. Device count doubles again, so cost
proxies and latency must be reported together.

The [new plan](scaled64-gram-design.md),
[launcher](submit_scaled64_gram.py) and
[preflight record](scaled64-gram-preflight-validation.json) use a separate job,
regional report prefix and launch guard. All four numerical sources are unchanged;
reuse their finished online CPU8, CPU16, logging and 17 regression checks. The
CPU16 direction tolerance failures remain recorded. The new launcher accepts only
a ready available v4-128 with sixteen healthy idle hosts, verifies the published
HEAD, and preserves default preemption/retries without wall-clock limits. Static
checks cover the 64-device command and seven rejected capacity fixtures.

The initial fleet snapshot contained no available ready TPU slice. No new job
was submitted, and all old jobs and observers remain terminal. The completed
32-device guard remains untouched. Publish this preparation, then refresh
capacity through the distinct launcher; submit nothing while v4-128 is unavailable.

The published launcher (`476f9102b68daf5c7fcbe2bda80833511c47dc16`) refreshed
capacity at 17:34:18.818690 UTC and returned without submission: no eligible
ready available v4-128. Its new selection record has `launched=false` and no
`launch_started`; the original completed guard is unchanged. Evidence:
`scratch/20261104-scaled64-selection.json` and
`scratch/20261104-scaled64-1727-submit.log`. No worker or observer was started.


### 2026-11-04 21:49 UTC — Switch to the original single-seed scientific campaign

The user requested a strict plan and immediate execution of Ali's original
experiment, then emphasized maximum parallelism because time is tight.
[single-seed-execution-plan.md](single-seed-execution-plan.md) specifies the
30-cell original grid, reuse of the completed alpha=0 pilot, 29 remaining
full 7,630-update pilots, boundary expansion and locked single-seed
37,335-update confirmations. The training baseline remains matrix/default.
The unlaunched 64-device synthetic diagnostic is superseded and removed
from the heartbeat's launch instructions. No old job or follow was queried.

The new campaign binding calls the existing native full-schedule training
entrypoint. Its own ONLINE logging and tiny native CPU/checkpoint harness
finished successfully with API-verified source hashes; the 17 existing
optimizer regressions were reused by exact hashes. Static preparation checked
58 cell/topology configurations, preserving seed0, batch8, full7,630 schedule
and fresh initialization. All 29 new artifact roots were absent. The
capacity selector rejected seven ineligible cases and preserved unique
commands, default retries and no wall-clock limit for all29 cells. Required
lint passed. A separate cold reader reviewed the plan; dispatch cadence,
failure handling, pilot reuse, confirmation evidence across retries and
checkpoint probe selection were clarified. See single-seed-validation.json.

An initial local preparation failed because the candidate artifact version
was not a calendar version; it was corrected to 2026.11.04.02 before any
training submission. Formatting/lint findings were fixed. No numerical
source was modified, and no TPU training has been submitted at this point.
Fleet snapshots at21:34 and21:45UTC had no ready available TPU slices. The
previous no-cold-queue constraint was explicitly raised to the user; pending
an answer, the dispatcher remains ready-only. Source publication and a
single persistent60-second capacity dispatcher follow; all29 cells may run
concurrently when compatible capacity is available. Initial supported
variants are v4-8 andv4-32; larger topology qualification remains outstanding,
and unsupported capacity is not counted as training capacity.


### 2026-11-04 21:53 UTC — Scientific campaign dispatcher started

Published campaign source939fe916d07c581777b45d4e6dc0720097f92972
on the research branch and fixed branchcodex/20-ema-gram-single-seed-20261104.
A separate immutable worktree runs the sole ready-capacity dispatcher every
60seconds, with concurrent submissions and persistent per-cell guards.
Its first21:49UTC refresh found no eligible slot and submitted no job.
The dispatcher is active; training is not yet running. Cold queueing remains
a pending clarification of the user's earlier explicit restriction.

The companion read-only observer owns one log follower per accepted job,
checks canonical state first after120seconds and then570seconds after each
check, and stops polling each terminal job. A simulated29-job fixture passed
without any real job RPC. Its completion state is not scientific acceptance:
full training counters, online metrics and checkpoints still require review.
The standing heartbeat now follows this campaign and must not launch the
superseded64-device synthetic diagnostic or any old terminal job.
See single-seed-started.json for the initial observation.


### 2026-11-04 21:52 UTC — Cold queue authorized; all 29 submissions started

The user explicitly answered “允许全部并行排队，优先缩短总时间”.
This supersedes the earlier ready-only restriction. Under the dispatch lock,
confirmed zero existing cell submissions and stopped only the idle local
capacity watcher; no Iris job or allocation was canceled or released.
Started the published launcher with --allow-cold-queue. All29 distinct
launch-started guards are written:24nonzero cells request v4-32 and5alpha0
cells request v4-8, normal batch priority, 404v4devices if all are allocated.
Submission acceptance and actual allocations remain to be observed. The
companion monitoring owner is active, ready to attach one follower and
canonical checks to each accepted job. No scientific training result is
claimed by these submission guards.


### 2026-11-04 21:54 UTC — All 29 scientific pilot submissions accepted

All29 submissions returned exit0 with their unique canonical Iris job IDs;
the submit process ended naturally. The observer owner attached one follower
to each accepted job and scheduled first canonical checks after120seconds.
The accepted requests comprise24v4-32 jobs and5v4-8 jobs:404requested v4JAX
devices and101host tasks if all are allocated concurrently. Requested capacity
is not actual allocation. No training completion is claimed; first canonical
allocation and online training signals remain pending. Source, exactcommands,
cell parameters and persistent launch-started guards are preserved in
[single-seed-submitted.json](single-seed-submitted.json).


### 2026-11-04 21:56 UTC — First canonical campaign observation

All29 jobs are PENDING with101pending tasks, zero failures/preemptions and
no started_at timestamps. Allocated training devices are therefore0 in this
first observation; requested devices remain404. Scheduler reasons include
insufficient available host memory and autoscaler quota-pool tier blocking.
These are current scheduling reports, not a verified underlying cloud cause
or an ETA. Keep the accepted jobs and their guards unchanged. All29
followers are live with no exit failures; the owner schedules later checks
570seconds apart. See single-seed-first-observation.json.

The accepted campaign was published in issue20comment6643115554. Updating
the issue body through the GitHub integration returned403(Resource not
accessible by integration); no retry or alternative access path was used.
The latest plan and current evidence remain available in the published
research branch and the permitted issue comment.


### 2026-11-04 22:24 UTC — Campaign blocked by deleted training caches

The first campaign's workers failed during native data loading: seven original
Nemotron cache roots were soft-deleted today between 12:56 and 16:59 UTC.
The metadata inventory preserves 45,220 distinct objects with size, CRC and
generation. StarCoder, ProofPile and 23 evaluation components remain live;
the first broad-prefix inventory incorrectly marked the former two as empty.
These training paths match the successful matrix prefix's data configuration.
No completed scientific pilot or effective optimizer progress is established.

A scoped, no-overwrite bulk restore request for the first missing prefix
returned GCS HTTP 403: the current account lacks `storage.buckets.restore`.
No restoration succeeded and the remaining restore requests were not issued.
The user was asked to arrange authorized restoration or a permission change.
The seven exact restore request bodies and original object inventory hashes
are in `single-seed-cache-restore-requests.json` and
`single-seed-cache-failure.json`. No other identity/API route was attempted.

Added `check_campaign_caches.py` and an original-content manifest. Submission
checks full object catalogs before allocating new work; execution hosts check
metadata readback and first/last chunk identities before starting the trainer.
The real negative test rejected all seven missing components and passed the
other 25. The numerical entrypoint, optimizer, model, mixture and existing
immutable launch source remain unchanged. Full recovery still requires
restoration, complete content-identity checks and actual data-loader validation.
Original job guards and observer ownership remain intact. Recovery jobs require
new identities after original canonical termination; none have been submitted.

At 22:26 UTC the sole observer confirmed all 29 original jobs FAILED and exited
naturally with code 0; all 29 log followers are terminal. Follower exit 0 does
not indicate training success. The user chose to grant restoration permission
to the current account. IAM capability checks show object restore/create are
already allowed, while bucket restore and bucket IAM modification are absent.
Bucket IAM and bucket metadata reads returned 403. No IAM mutation succeeded
or was attempted after discovering missing authority. An administrator grant
is required; the exact one-permission custom-role commands are in the plan.


### 2026-11-04 22:33 UTC — Existing-copy search and Stanford permission check

The user requested an existing-cache search and logged in with their Stanford
account for a read-only permission check. That account has object get/list/create,
but neither object restore, bulk restore nor bucket IAM modification. No IAM
change or restoration was performed.

The seven configured GCS regions have no live objects at the original Nemotron
cache prefix. Broader regional directory listings found Delphi cache aliases
containing provenance metadata only, plus an unrelated Nemotron cache with
metadata and no token chunks. Backup and nested-prefix candidates did not
yield a verified intact copy. Original prefixes have no readable noncurrent
versions. CoreWeave and R2 could not be searched with current configured
credentials. An arbitrary-directory name query is still running; its result
will be recorded separately. `single-seed-cache-alternate-search.json` preserves
the scope, limitations and evidence hashes. No large data was downloaded or
transferred between regions.


### 2026-11-04 22:49 UTC — Fixed-subset reconstruction route

The user proposed retokenizing a fixed subset, supported by a discussion
screenshot about shared raw/tokenized cleanup and DataKit copies. The plan now
includes a roughly2B-token, original-mixture subset reused by all configurations.
All30 pilots, including alpha0, must be paired on that new data version. This
retains model/optimizer/tokenizer intent, update counts and evaluation protocol,
but does not preserve the original document sequence.

The public original Nemotron v1 index was read (31,279 paths; SHA recorded).
One file HEAD returned200 and342,647,944bytes; GET samples across seven categories
and an independent first-file sample returned503 SlowDown. No raw payload corpus
has been downloaded or tokenized. Regional DataKit discovery found v2/v2.1
normalized sources but no v1 root, so no equivalence is asserted. Native
per-shard sampling and Levanter cache writers are available, but a frozen
production subset builder and regional native loader check remain outstanding.
The implementation must count real tokens, preserve document boundaries, and
check full-schedule per-component coverage before any new training submission.

The earlier alternative FineWeb catalog checker incorrectly treated the Zarr
reserved shape as logical token count. Its generated missing-index list was
discarded and replaced with an explicit invalid-result record. That check does
not show missing token chunks or establish a usable replacement dataset. The
24-prefix candidate search completed; the broad original-name query remains
owned by session42678. No data-source switch or replacement job was launched.

### 2026-11-04 23:00 UTC — bounded subset worker ready for regional source probe

Implemented the native Llama 3.1 tokenizer and Levanter cache writer in
`experiments/ema_gram/subset_data.py`. The frozen source manifest selects 32
crawl/kind strata per original Nemotron component, with four source prefixes
per CPU task, plus bounded copies of the intact code/math caches (58 tasks).
The approximately 2B-token budget keeps whole documents, records actual token
counts and stream hashes, and never silently accepts source exhaustion.
This is a new fixed subset, not the original training stream; all 30 pilot
cells, including alpha zero, must use the same rebuilt dataset.

The final source passed an ONLINE CPU check for exact native token IDs,
deterministic replay, Unicode/empty/long documents, whole-document limits,
source-exhaustion rejection and a native cache write/read round trip.
W&B run `ema-gram-subset-self-check-da6e394c0822` and the prior non-training
logging check are API-verified FINISHED/success. The three tokenizer files
are SHA256-pinned in `subset-inputs.json`; source hashes and evidence are in
`subset-preflight.json`. Lint passed. A separate guarded launcher requests
2 CPU / 8 GB in us-central2, normal batch, no accelerator or wallclock limit,
and unchanged default preemption/retries. Build submission requires a
successful regional probe first. Old failed training identities stay terminal.

Local public source GETs returned HTTP 503 SlowDown, while the index and
object HEAD worked. The next published-source CPU probe tests actual original
source decoding in the training region. No data or training completion is
claimed from local preflight.

### 2026-11-04 23:29 UTC — use a raw-text HQ subset after original endpoint and token mirror fail

The first regional CPU source probe (source `24dbba52932df1940c8074b2e77cba206d406494`,
job `ema-gram-20-subset-probe-20261104`) failed on attempt 0 after the shared
HTTP client's retries received only 503 SlowDown. Its sole follower ended
naturally; task UID `1b694c1508d4437b` retains the failed report. This is a source
service failure, not unavailable compute. The original guard remains set.

A public 65B-token Nemotron mirror contains all seven source classifications,
but stores custom token IDs. Its 300 filenames match the original index.
The author's `dvruette/nemotron-cc-bpe` tokenizer adds prefix spaces and NFC
normalization; actual samples failed exact decode/re-encode. The candidate
was rejected by the online check (`ema-gram-subset-self-check-f748213d9416`),
with diagnosis `ema-gram-subset-decoder-diagnosis-85b625e16702`. No jobs or
training used that decoded text. Earlier optimism about that mirror was
superseded by the payload/schema and roundtrip checks.

To execute the user's requested fixed-subset route without waiting on cache
restore permissions, the next dataset uses actual raw JSONL text from
`Fredithefish/Nemotron-CC-HQ-20B`, pinned revision
`1bcdb7f17b10c759bd70ce33f2921ad28cc28505`. Two actual documents were decoded
successfully. The publisher describes sampled Nemotron-CC-HQ crawl slices;
this is not independently established as byte-identical to the deleted
Marin mix. The design deliberately replaces the seven-way web-quality mix
with that fixed HQ subset while retaining the aggregate web/code/math shares.
All 30 seed-zero pilots including alpha zero restart on this new dataset;
old losses are contextual and cannot be pooled into selection. Model,
optimizer, numerical precision, global batch, schedules, and 23 eval caches
remain unchanged. This changes the data recipe, not just its storage path.

`subset-inputs.json` selects 56 distinct raw files by a fixed hash ordering,
34,725,120 tokens per web task (whole-document overshoot recorded), plus
45,404,354 code and 9,988,958 math tokens. The native Llama tokenizer bytes
remain fixed. The same 58 guarded CPU tasks share one resulting dataset.
Actual text prefixes are streamed in-region; HF commit and LFS pointer are
checked, and every written token is read back and hashed. We do not claim
full compressed-source SHA verification when only a prefix is consumed.
The final CPU check `ema-gram-subset-self-check-d939f0b8e172` passed exact
native tokens, source exhaustion rejection, full native copy/readback,
sharded cache concatenation, and flat reads crossing shard boundaries;
W&B FINISHED/success is API-verified. The new `hq01` probe has a distinct
identity; production builds remain gated on its success. The draft native
training binding uses version `2026.11.04.03` and is not yet launch-qualified.

### 2026-11-04 23:35 UTC — regional raw HQ probe verified; release CPU builds

The hq01 probe succeeded canonically, one task/attempt0/exit0, no failures
or preemptions, native finished23:30:46.735UTC. Its bounded1054-byte output
archive, generation/CRC, exact source and tokenizer pins were verified.
Host logging c98fd0e6f4ab and probe2b1d454dfc0c are ONLINE FINISHED/API-success.
The sole observer30598 and its follower ended naturally; no further polling.
See subset-regional-probe.json. The published launcher may now submit all58
CPU builds with permanent guards and one observer/follower per identity.

Native training binding's online logging7a6dc8d5b0c9 and CPUc8bfa177fac9
passed and are API-verified; subset-training-cpu-validation.json records
the exact unchanged numerical sources. This is not real-model TPU or actual
regional data-loader qualification. All30 new pilots remain gated on data.

### 2026-11-04 23:42 UTC — all58 CPU builds accepted and running

All58 hq01 build submissions from source24a029e61 were accepted; submit60948
ended naturally exit0. Sole observer74503 owns all follows and canonical
checks. Its first canonical checks observed58RUNNING; no training allocation
is implied by these CPU tasks. subset-build-launched.json freezes the exact
identities, source, commands and guard evidence. Two log followers (tasks2,54)
ended with Finelog Bad Gateway; this is not worker failure and no replacement
followers were created. Both logging identities were retained in their logs.

The58 tasks use native cache writing, full token readback, and unchanged
pinned tokenizer bytes. Collection, regional assembly and real native loader
qualification are prepared. All30 static pilot configs and distinct output
paths were checked: seed0/fromscratch/7630steps/globalbatch8/4096context;
24nonzero usev4-32 and6alpha0 usev4-8. No pilot has been submitted yet.

### 2026-11-04 23:52 UTC — repair truncated source transfers and incremental GCS writes

The next canonical checks found web build failures with HTTP IncompleteRead,
including task0 after23,136,410 of358,218,755compressed bytes. Native GCS
small-batch writing had kept the producer blocked for minutes. This establishes
a broken transfer during processing; CDN idle behavior is a plausible mechanism,
not a verified server-side diagnosis. Math task57 succeeded and will be retained.
All failed guards/attempts remain; no queued/running identity is replaced.

The repaired worker downloads each pinned raw file completely to regional
scratch before parsing, checks full compressed-source SHA256, and restarts
a truncated download cleanly. Native cache writing now stages on regional disk
and uploads completed cache objects once, with ledger last. Full remote token
readback remains required. The first local staging check exposed a missing
parent directory and failed; the corrected final online selfcheck917ae563c699
passed native tokens, cache/readback/merge, and injected truncated-response
recovery without duplicated records. W&B finished/success is API-verified.
The manifest and selected tokens are unchanged. A distinct hq02 regional
probe must pass before repairing only canonically failed build indices.

### 2026-11-04 23:56 UTC — repaired regional download/write probe passed

Distinct hq02 probe from9dd8457ed succeeded canonically, attempt0/exit0, no
failures/preemptions. It downloaded all358,218,755compressed source bytes,
verified the exact pinned LFS SHA256, and produced identical sample token
hashes to hq01. Native local staging, GCS upload, and full remote readback
passed for1,293tokens. Host logging and probe2f23561d85cf are ONLINE
FINISHED/API-success; bounded artifact identity is recorded in
subset-regional-probe-hq02.json. This clears new hq02 repair submissions only
for observer-confirmed failed indices. Successful math57 (9,990,772tokens,
3,323documents) is fully verified and reused, not rebuilt. Other original
identities stay under observer74503 until their next owned canonical check.

### 2026-11-05 00:01 UTC —54 repaired builds accepted; code and math retained

All54 currently canonical-failed web indices were admitted once under hq02
from the same source2d357613c; submit sessions46276/94775/80975 ended0.
Observer2647 owns these follows/checks. Original indices25/46 remain under
observer74503, last-known RUNNING, so they were not replaced or duplicated.
Code56 succeeded with45,411,940tokens/22,006documents; math57 succeeded with
9,990,772tokens/3,323documents. Both full native readbacks, successful task
archives, exactsource/manifests and host/workerONLINE W&B were verified.
The repaired web workers are already reporting successful full builds; final
canonical checks remain owned by the existing observer. No training yet.

### 2026-11-05 00:04 UTC — faster checks for short CPU data stages

The fixed570-second monitoring interval unnecessarily delayed data assembly
after minute-scale workers completed. Local observer owners74503/2647 were
retired (exit143) and replaced under the same exclusive state locks by68096
(hq01) and81427(hq02), using60-second canonical checks after the initial120s.
Existing log follower PIDs were adopted; none were restarted or duplicated,
and no remote job or allocation was signalled. If an adopted follower vanished,
its exit code is explicitly unavailable rather than fabricated as0. Terminal
jobs remain excluded from canonical polling. Training cadence remains unchanged.

### 2026-11-05 00:10 UTC — retire the last two obsolete CPU writers

All54 hq02 repairs succeeded canonically. Original25/46 still used the slow
incremental GCS rewrite path and were actively writing, not stuck or failed.
To apply the already validated I/O fix promptly, their exact current CPU
attempts were explicitly operator-failed with a recorded migration reason.
This is a repair decision, not a wall-clock cutoff or spontaneous worker
failure; no accelerator allocation was involved. Their partial objects and
attempt evidence remain. Only after observer-confirmed termination will the
same pinned source/tokenizer/token budgets run under new hq02 guards.
The54 earlier IncompleteRead failures remain separately classified.
Collector now recovers bounded exact-attempt host logging IDs if a Finelog
follower failed, without creating replacement followers.

### 2026-11-05 00:14 UTC — all58 fixed-subset builds verified

58 unique successful build reports cover2,000,188,501tokens and3,043,111
documents. The188,469-token overshoot retains complete final documents.
All reports match the pinned source/task/tokenizer manifest, full native
token readback, successful exact task attempt/archive, and FINISHED ONLINE
host logging plus worker runs. Final repaired25/46 succeeded attempt0; old
operator-stopped attempts have no output archive and retain partial GCS objects
and canonical reason separately. The ordered results are now ready for native
ledger assembly; actual regional mixture/validation loading remains required.

### 2026-11-05 00:14 UTC — notebook size maintenance

The all-builds publication crossed the repository500KB notebook limit.
Moved October30–31 entries verbatim to the linked archive; no historical
evidence removed. This corrects the file-size check before assembly launch.

### 2026-11-05 00:18 UTC — native assembly verified

Assembly from29edd33f2 succeeded attempt0/exit0 and both execution-host
logging and workerf9dfd2595448 are ONLINE FINISHED/API-success. Exact
bounded archive/checksum and dataset/source/tokenizer identities verified.
Three merged ledgers reference precisely the58 successful shard paths in
manifest order; row/token totals match the successful reports. A local
comparison initially assumed absolute finished_shards; native ledgers use
relative names. Corrected the comparison and it passes, with no worker or
data change. Failed partial outputs are preserved but not training shards.
The frozen26-component catalog includes825 objects, retaining the original23
evaluation identities. Actual regional loader qualification is next.

### 2026-11-05 00:21 UTC — actual regional loader gate passed

Data-check from12a25a61b succeeded canonically, attempt0/exit0. Hostlogging
5c280d7c5e48 and data-check077aa98e32ef are ONLINE FINISHED/API-success.
Native loader reads all3 training components and23 original validation sets;
seed0 first/pilot-end/confirmation-end sequence hashes replay exactly.
Available sequences web474801/code11086/math2439 exceed the native
confirmation-block bounds web290832/code6716/math1460. Loaded token/row
counts equal the assembled ledgers. Exact numerical hashes reuse the prior
online CPU harness and regressions; no unchanged numerical tests rerun.
Published subset-training-validation.json binds all sources and data manifests.
Next dispatch all30 fresh pilots from a detached immutable execution worktree
ema-gram-subset-seed0-20261105, version2026.11.04.03,24 v4-32/6 v4-8.
The launcher must pass its full825-object catalog gate before submission.

### 2026-11-05 03:04 UTC — recover failed pre-job submission transport

The full825-object/26-component catalog gate passed, but30 simultaneous
18.3MB launch RPCs timed out at180s and later exited1 with DNS/no-route-to-host.
No queued/running training is claimed. At03:01:42UTC after connectivity
returned, a canonical batch GetJobState for all30 exact IDs returned no jobs.
Original guards remain set; empty observer54154 is retired with no followers.
The parent launcher incorrectly returned0 despite30 child failures; r02 fixes
that reporting. New r02 identities/ledger will reuse unchanged numerical
sources, validated data and artifactversion2026.11.04.03 (no original job
was created). Limit only simultaneous package-upload RPCs to8, with no cap
on concurrent experiments once accepted. The slow615-object web metadata
loop now uses16 parallel checks; full catalog identity, generation-bound JSON
readback and edge checks remain. Seven static positive/negative fixtures pass.
No network/IAM/security settings were changed and no remote job was cancelled.

### 2026-11-05 03:07 UTC — all30 r02 scientific pilots accepted

The parallel full catalog gate passed all26 components/825 objects, followed
by30 accepted unique job submissions from immutable2bca5a112. Submit94148
ended naturally0; each per-cell guard remains launch_started/launched.
Sole observer17065 owns30 follows and first canonical checks after120s,
then570s. Finelog snapshot-deadline failures affected some follows; they
are not worker failures and no replacement followers were created.
24nonzero v4-32 and6alpha0v4-8 requests total408 JAXdevices, not actual
allocation. Source/worktree and full launch evidence are recorded in
subset-training-r02-launched.json. Await owned canonical allocation and
startup evidence before claiming running models or scientific progress.

### 2026-11-05 03:09 UTC — first owned canonical campaign checks

All30 first checks completed:29PENDING,1RUNNING(index01 alpha0), with
101pending tasks and1building task. No failures or preemptions. This is
startup, not evidence of a model update or a completed pilot. Scheduling
reports insufficient available host memory and quota-pool tier blocking;
these are not a wait-time estimate. Sole observer17065 nextchecks are
recorded in subset-training-r02-first-observation.json. No resubmissions,
allocation changes, priority changes or replacement followers.

### 2026-11-05 03:14 UTC — first native training and online loss confirmed

Cell01 alpha0, task0/attempt0 UID343a04a0171ae325, is executing on its
assigned v4-8 worker. Bounded native loss labels109–118 are finite and match
ONLINE W&B exactly. Native run
ema-gram-pilot-3e18-a0-seed0-4b5ff8c72615-2026.11.04.03-d315857e7875
is RUNNING; execution-host logginge69c1fd6b24f is FINISHED/success. Both
API configs identify exactsource2bca5a112, and host numerical hashes match
the frozen validation. Initial local range assertion expected10rows from
inclusive109..119; corrected to109..118 without training changes. This
confirms actual new-data training startup, not pilot completion or an
optimizer-speed result. Other cells retain their owned scheduling checks.

### 2026-11-05 04:31 UTC — first two fixed-subset pilots fully validated

Cells01 and03 (alpha0, LR0.0035355339 and0.0141421356, seed0) completed
7630 updates from scratch. Both canonical task0/attempt0 succeeded, exit0,
with no failures/preemptions. All7630 native labels0..7629 have finite ONLINE
loss/duration/MFU/two LR metrics; final C4 losses are4.1497836113 and4.2295851707.
Both native W&B runs are FINISHED, exact frozen subset configs/source2bca5a112
verified, and exact task-attempt/run joins retained. Their permanent final
checkpoints have generation/CRC/readback-verified metadata and57-array manifests;
independent scalar reads confirmed stored counter7630 (directory label7629).
This does not claim a full checkpoint array audit or restored training state.
Evidence: subset-training-r02-first-completions.json. Two of30 pilots validated;
no hyperparameter selection or scientific-core completion yet.

Local collection corrections are preserved: Draccus coerced a jax_config bool
into a string; comparison now uses the exact frozen JSON mapping. W&B pagination
repeated seven identical boundary rows and omitted seven other labels; bounded
exact-label reads recovered all seven, with raw responses retained. An empty
min=max range was corrected. TensorStore's GCS driver could not parse the local
ADC format; the documented read-only HTTP driver read only the two stored step
scalars using the same ADC identity. No IAM change, credential output, worker
failure, numerical change, new launch or restart resulted from these corrections.
Sole observer52357 remains alive and retains every existing follow/launch guard.
Remaining pilots continue. After grid/expansion validation, lock selection and
launch seed0 confirmations with the required reference checkpoints and probes.

### 2026-11-05 04:37 UTC — alpha0 pilot group complete; no LR expansion

The remaining alpha0 cells00/02/04/05 have canonical task0/attempt0 success,
exit0, and no failure/preemption. As for01/03, exact frozen source/config and
native attempt/run joins, finished ONLINE tracking, all7630 finite short-metric
rows, final C4 evaluation, generation/CRC-bound permanent checkpoint metadata
and57-array manifests passed. Scalar-only reads independently confirmed stored
native counter7630 in all four final checkpoints. Existing01/03 results were
reused without polling their jobs or finished W&B runs.

| Cell | Matrix LR | Final C4 loss |
|---|---:|---:|
| 00 | 0.00176776695 | 4.1904120445 |
| 01 | 0.00353553391 | 4.1497836113 |
| 02 | 0.00707106781 | 4.1689982414 |
| 03 | 0.01414213562 | 4.2295851707 |
| 04 | 0.02828427125 | 4.3551721573 |
| 05 | 0.05656854249 | 4.4796938896 |

The preregistered evaluated_groups/expand functions select cell01, an interior
LR, and add zero alpha0 expansion cells. This resolves this group's LR search;
it does not lock cross-exponent selection or authorize an outcome-dependent
change to the protocol. Six of30 pilots are validated. All24 nonzero cells keep
their accepted identities; the owned snapshot has9RUNNING/15PENDING, no failures
or preemptions. Evidence: subset-training-r02-alpha0-completed.json. Continue
the remaining groups and their required expansions before global selection.

### 2026-11-05 04:44 UTC — qualify online reference-probe entrypoint locally

Published c4422dd51358b97a6fb6a33a8abc70a787bd74c6 adds tracked_probe.py,
which initializes a fresh ONLINE Levanter tracker before numerical work, records
resolved subset config/source hashes and measured paired loss summaries, and
keeps a separate output directory per process. It normalizes the frozen config
wrapper and checks reference alpha/seed/horizon. The existing numerical probe
now rejects a mismatched restored counter before loading evaluation batches;
its direction/loss/SVD formulas are unchanged. Training source2bca5a112 and all
accepted jobs remain unchanged. No production probe or additional training job
was launched.

Fresh ONLINE nontraining logging3693634ecd70 and CPU15eff3ffc5e6 both finished
successfully; API configs and source hashes were verified. The existing tiny
Qwen regression verifies matrix norm matching, fixed nonmatrix parameters and
paired held-out quadratic loss changes against direct updates. This is local
probe qualification, not full-size checkpoint/TPU proof. SVD-specific validation,
a validated confirmation lock/reference checkpoint, and fresh regional host
logging/complete restore remain required before production probe results.
Evidence: reference-probe-preflight.json. Initial lint exposed formatting and
deliberately delayed numerical imports; corrected before publication/checks.
A direct black command was unavailable, while the required repository pre-commit
formatter completed successfully; no dependency workaround was needed.

### 2026-11-05 04:50 UTC — SVD diagnostic CPU qualification passed

Published b90b7aa1d2279592935c9f9fc357f81584821da4 adds a dedicated CPU
SVD oracle behind the already-online tracked probe wrapper. It exercises all18
attention/MLP alpha/ridge rows, compares the SVD directions to independent
NumPy float64 eigendecomposition/SVD construction, and checks the reported
cosine/relative errors against that reference. Three zero/rank-deficient
rectangular cases and exact optimizer-state immutability also passed.
Maximum absolute SVD-reference difference was6.5373167013e-06.
CPU fixture timings are diagnostic only, not accelerator or training speed.

Fresh ONLINE logging8b005ee44e81 and SVDcheck5ce418f21a16 both FINISHED/success;
API source/config hashes verified. The prior norm-matching/paired-loss oracle
was reused with unchanged numerical-source hashes, without rerunning it.
Training code and accepted job source remain unchanged. No new training job,
reference-checkpoint probe or allocation change occurred. Evidence:
reference-probe-svd-preflight.json. A validated confirmation selection/reference
checkpoint, fresh regional host logging, complete restore and regional output
validation remain before production probe conclusions.

### 2026-11-05 04:56 UTC — prepare seed0 confirmation configurations

Added prepare_subset_confirmations.py to materialize the three or four seed0
confirmation configurations from a digest-validated scientific selection lock.
It reuses the qualified fixed-subset data and matrix/default numerical sources,
starts each 37335-update run from scratch, and retains alpha=.25 reference
checkpoints at native counters3733/18665/33597 (directory labels one lower).
The final checkpoint is forced by the native final hook. This tool does not
submit jobs, and no actual selection lock has been made.

Static fictional selection fixtures verified both the three- and four-cell
branches, seed0 filtering, unchanged model/data, and the native checkpointer's
exact retained labels. Tampered lock digests and the reused pilot artifact
version were rejected. The initial fixture used an invalid artifact version;
only its fixture version was corrected to the supported dev form. No kernels,
data loading, W&B numerical checks or training submissions were repeated.
Evidence: subset-confirmation-config-preflight.json. Fresh campaign identities,
full catalog checks and regional host logging remain launch requirements.

At04:56 UTC the sole observer remained alive and its cached canonical results
showed6 SUCCEEDED,11 RUNNING,13 PENDING, with no failures or preemptions.
The nine previously training nonzero cells all advanced to labels910–1332;
two newly running cells13/16 had online identities but no observed loss yet.
They are startup observations, not proof of optimizer updates. All six alpha0
scientific completions were reused without terminal-job or W&B repolling.

### 2026-11-05 05:03 UTC — bind active training to exact task attempts

Collected bounded task status and native W&B links once for the eleven
canonically running nonzero cells06–16. All44 tasks are attempt0 with distinct
UIDs; each cell occupies four distinct hosts on one named v4-32 slice.
Each host logging ID matches its already API-verified ONLINE source record,
and exactly one native training run per cell is joined to its emitting task.
Evidence: subset-training-r02-attempt-identities.json. This records assignment
and process identity, not a separate runtime JAX device-count audit or final
completion. The progress helper reuses these joins while canonical failure and
preemption counters stay unchanged, and requests refresh when they change.

At05:00 UTC all nine established nonzero runs advanced to labels985–1415.
Cells13/16 and the newly visible ONLINE identities17/18 had no loss summary
yet; scheduling snapshots and online discovery need not arrive simultaneously.
The sole canonical observer remains active; no jobs or followers were created,
restarted or canceled. The obsolete cell06 compilation-gap note was removed
from current progress; its historical evidence remains preserved.

### 2026-11-05 05:09 UTC — prepare regional reference-probe execution

Added run_reference_probe.py as a process wrapper around the already-qualified
tracked probe. It checks the configuration digest, exact permanent checkpoint
path/counter, regional destination and qualified source hashes; runs fresh
ONLINE host logging before the tracked numerical process; and preserves only
bounded diagnostic files under a unique regional attempt directory. Uploads
use generation-zero preconditions and generation-bound readback, with a
manifest of sizes, CRCs and SHA256s. Partial diagnostic outputs are retained
on numerical or logging failure, while the process still fails. No queue or
execution wall-clock limits were added.

Static subprocess/storage fixtures exercised success, logging failure and
probe failure, including ordering and partial-artifact preservation. Wrong
checkpoint labels and cross-region references were rejected. These checks
ran no kernels, online numerical processes, GCS writes or jobs. Existing
numerical checks were reused unchanged. Evidence: reference-probe-regional-preflight.json.
Actual reference checkpoints, fresh regional logging, complete restore and
real upload/readback remain required; no production probe has been launched.

At05:05 UTC the owned snapshots showed6 SUCCEEDED/14 RUNNING/10 PENDING.
The nine established nonzero runs advanced to labels1065–1494; cells13/16
now had native label7 and cell18 label4, while cell17 had no loss yet. These
startup updates establish actual optimization only for the cells with losses;
all six validated alpha0 completions are reused.

### 2026-11-05 05:18 UTC — prepare the confirmation native-config adapter

Added confirmation_worker.py to reconstruct the seed0 confirmation bundle
from its selection lock, require complete encoded-config equality, and call
the unchanged native run_training path. The adapter restores frozen jax_config
JSON types after Draccus decoding; four static fictional-lock configurations
round-tripped exactly, and a changed schedule was rejected. A separate fresh
ONLINE host-logging mode records the config, lock digest, task/attempt and
source identity before the future native process. No kernels or training ran.
The adapter still needs its own nontraining ONLINE logging qualification and
a separately guarded submission controller; no scientific selection lock or
confirmation submission exists. Evidence: confirmation-worker-preflight.json.

At05:15 UTC owned snapshots showed6 SUCCEEDED/15 RUNNING/9 PENDING; all13
previously observed nonzero loss streams advanced, and14 nonzero cells now
had native steps. Cell20 was still startup without an observed loss. Local
attempt joins now cover56 tasks in cells06–19. One cell17/task2 Finelog
snapshot deadline was recovered by one bounded read; its original error is
preserved and was not a worker failure or follower restart.

### 2026-11-05 05:20 UTC — correct fresh-process checkpoint-path decoding

The first nontraining confirmation logging invocation rejected its config
before tracking or kernels: Draccus decoded the list-valued own-checkpoint
search path as a string in a fresh process. The in-process static fixture
had not exposed this behavior. The adapter now restores that frozen union
field as well as jax_config, retaining complete encoded equality. All four
saved fictional configs passed reconstruction in a new process. Original
failure: scratch/20261105-confirmation-worker-logging.log. No pilot worker,
training source or numerical formula changed.

Fresh nontraining ONLINE logging a21f2a914c93 then FINISHED successfully
from published fcee69ce7. The API source, bundle/config digests, selection
digest and reconstructed configuration were verified. W&B omitted two empty
configuration dictionaries; only those known-empty mappings were restored
for comparison, and the original complete config digest matched online.
This used an explicitly fictional selection fixture, not a scientific lock.
No training or numerical CPU check was run; native numerical sources remain
unchanged. The separately guarded submission controller is still outstanding.

### 2026-11-05 05:24 UTC — prepare guarded parallel confirmation submission

Added submit_subset_confirmations.py, reusing the existing Iris batch command
resources and defaults. It validates the exact locked bundle and full pilot/
expansion completion coverage before any submission, then requires tracked
inputs, a clean immutable source matching its published ref, and a full cache
catalog check. Each distinct confirmation campaign has a locked ledger bound
to source/input hashes; per-cell launch_started is durably written before
its RPC. Submission errors retain that guard for canonical reconciliation.
All three or four fresh seed0 configurations submit concurrently, with fresh
host logging and bounded cache checks before the native worker. No wall-clock
limits, preemption overrides or changes to current pilots were introduced.

Static fixtures checked both TPU commands, logging order, wrong/incomplete
completion evidence, and guard retention across a persisted ambiguous failure.
The actual six-completion record was rejected as an incomplete scientific grid.
No cluster submission or numerical process ran. Evidence:
confirmation-submitter-preflight.json. The real global selection lock, common
loss target, full completion evidence, fresh artifact/campaign identities and
sole confirmation observer still depend on finishing pilots and expansions.
Scientific failures, if encountered, require explicit adjudication before this
launcher can accept them; infrastructure failures never count as poor settings.

At05:21 UTC all fifteen scheduled nonzero cells had native update evidence.
Thirteen previously active streams advanced; cell19 was still at its startup
label7 and cell20 first reached7. This is one unchanged startup observation,
not a cancellation trigger. Six completed alpha0 results remain validated.

### 2026-11-05 07:26 UTC — prepare terminal task evidence collection

Added a terminal-task collector for the fixed-subset seed0 pilot campaign.
It selects results awaiting validation only after the existing observer records
a terminal job, binds each request to its launch-ledger job ID and immutable source commit,
and saves one final task snapshot containing all its recorded attempts. An exclusive lock and
persistent read-start record prevent concurrent or automatic repeated reads
after an ambiguous error; another read requires explicit reconciliation. Raw
responses and all attempt identities are retained.

Mock checks cover cache reuse, transport failure, and failed, preempted and
successful attempts. The saved observer snapshots produce zero new requests: six completed results
are reused, and the other24 jobs are still running or pending.
No live terminal read, training process or checkpoint read ran. Evidence is in
terminal-task-collector-preflight.json. Full configuration, metric, checkpoint
and online W&B finish validation remains required for the remaining24 pilots.

At07:23 UTC six pilots remain scientifically validated,17 are running and7
pending. All17 native training streams advanced, with maximum native training-step label3634 and
cell22 at196 after previously pausing at startup label7. The68 cached task-attempt joins remain
reusable because the observer snapshots still report zero failures/preemptions
across all30 accepted jobs. Continue the exclusive observer
and finish the nonzero artifact validator before the next pilot terminates.

### 2026-11-05 07:47 UTC — prepare nonzero pilot completion validation

Prepared verify_nonzero_r02.py for terminal nonzero pilots. It requires the
recorded immutable training source and frozen subset configuration, canonical
job success, and success of the original attempt on each of four distinct
hosts. It joins those attempts to their finished ONLINE host-logging runs and
the finished native training run. Each of five training metrics must contain
finite values at every native label0..7629, with final C4 loss at7629. Raw
pagination rows are retained; only identical duplicates are removed, and
missing labels are queried explicitly. Checkpoint checks read final permanent
metadata, compare the57-array schema with the validated reference, and read
stored scalar counter7630 using the same ADC and read-only HTTP driver. This
is not a full checkpoint byte audit or a verified resume.

Previously validated results are reused only after checking saved evidence
hashes and task identities, without querying their completed W&B runs. Jobs
with multiple attempts are rejected pending separate lineage adjudication.
The offline fixture accepted a synthetic four-host completion, rejected seven
invalid artifact cases, and detected changed cached evidence. It used fabricated
metric and storage responses; no live terminal validation or numerical training
ran. Required lint passed. Source hashes and the runnable fixture command are
in nonzero-completion-validator-preflight.json.

At07:38 UTC the six alpha0 results remain the only scientifically validated
completions;18 nonzero pilots are running and six pending. Seventeen established
streams advanced, reaching native training-step label3843; cell21 became the
eighteenth reporting stream at label7. All72 cached task-attempt joins remain
valid against unchanged failure/preemption counts. Continue the exclusive
observer. Global hyperparameter selection still awaits all pilots and required
LR boundary expansions; no selection has been finalized.

### 2026-11-05 08:08 UTC — prepare boundary-expansion configurations

Added prepare_subset_expansions.py. It reconstructs successive expansion
choices from validated results, separately for each alpha/ridge group. All
currently required points in a group must finish before its next choice:
halve the learning rate if the lower boundary wins, or double it if the upper
boundary wins. Incomplete groups wait while other complete groups can advance.
Completion records must pass the existing full-result checks and match accepted
source/job bindings. A previously guarded expansion is excluded from new
configuration proposals, including an ambiguous submission pending reconciliation.

The builder retains seed0,7630 updates, batch8, the shared subset and the
qualified matrix layout/default spectral implementation. It requires a distinct
calendar artifact version, records configuration/input hashes, and refuses to
overwrite its output file. The future submitter must enforce published immutable
source and input hashes. Offline fixtures exercised two independent nonzero
boundary groups, repeated expansion, incomplete groups, wrong predecessor source
and ambiguous guards. Native decode/re-encode preserved the generated configs,
including the intended concrete types of union fields. Required lint passed;
evidence is in subset-expansion-config-preflight.json.

The actual cached alpha0 optimum is interior and needs no expansion. No nonzero
group has finished. These fixtures use fabricated losses and establish no new
scientific result. No training process or job submission ran. The expansion
worker, separate-process reconstruction, fresh ONLINE nontraining check and
guarded submitter remain outstanding. Actual launches will also require a new
immutable execution worktree and one expansion observer; the existing pilot
observer continues unchanged. At08:03 UTC all18 active native streams advanced,
with maximum label4250; six other nonzero pilots remain queued.

### 2026-11-05 08:12 UTC — reconstruct expansion configs in fresh processes

Added expansion_worker.py. It reconstructs the frozen expansion bundle from
completed predecessor records and accepted launch bindings, then calls the
unchanged native trainer. Three saved fictional configurations (one v4-8 and
two v4-32) reconstructed in separate processes, each matching its saved frozen
configuration SHA-256. Evidence: expansion-worker-offline-preflight.json.

The first digest check exposed four RoPE values that Draccus converted from
integers to floats. The adapter now restores their frozen input types before
comparing digests. The adapter changes those fields' Python types; the native
trainer code is unchanged. The initial mismatch was detected before tracking
or training initialized. Fresh ONLINE nontraining logging
is still required after source publication. No expansion job was submitted.

The fresh nontraining ONLINE run ema-gram-expansion-host-a2b42911b5b0 finished
from published worker source d1bcec239. API checks matched the source, bundle
and configuration digests. W&B omitted two known-empty configuration
dictionaries; restoring those locally made its configuration values agree with
the saved input. W&B also serializes whole-valued floats as integers, so its
serialized bytes differ; the reported original configuration digest matches
exactly. Both the process and API finish checks passed. This synthetic fixture ran
no training. Evidence: expansion-worker-preflight.json. The guarded submitter
and dedicated expansion execution worktree/observer remain outstanding.

### 2026-11-05 08:34 UTC — prepare guarded boundary-expansion submissions

Added submit_subset_expansions.py. It reconstructs the frozen bundle with prepare_subset_expansions.py, requires
Git-tracked bundle/completion/accepted-launch inputs and HEAD equal to the
published source commit,
and runs the full cache catalog check before submissions. One shared permanent
registry blocks duplicate Cell.name identities across worktrees sharing it. Each
parallel request gets a durable launch_started guard before its job-submission
RPC; errors retain guards and block retries until reconciliation. An initialization marker makes a missing registry fail closed.
Campaigns also bind source, input hashes and a distinct artifact version.

Offline tests intercepted every external command and exercised source rejection,
failed cache checks, two parallel submissions with one error, duplicate attempts
across campaigns, and rejection of a missing initialized registry. Both TPU
command variants retain normal batch priority and the scheduler's default
preemption retries, impose no additional wall-clock limits, and configure fresh
ONLINE host logging before training. These are offline command checks. Required lint passed.
Evidence: expansion-submitter-preflight.json and check_expansion_submitter.py.
The first orchestration invocation accidentally imported an older scratch draft;
the fixture now loads the logbook controller by its explicit path, and passed.

No actual expansion was submitted; the real nonzero groups remain incomplete.
Launch still requires a dedicated immutable worktree with fresh campaign/artifact
identities and an expansion observer/terminal-validation workflow. At08:24 UTC,
21 nonzero pilots were running and three queued;18 established native streams
advanced and cells24–26 first reported label7. The maximum native training-step label among those21 pilots was4573. All six
alpha0 results remain the only scientifically validated completions.

### 2026-11-05 08:46 UTC — all nonzero pilots are running

The sole observer reports all24 nonzero pilots RUNNING; none remains queued.
One native training stream for each of23 existing pilots advanced since08:41, reaching
maximum training-step label4900. Cell29 completed all four host logging checks
and initialized its native ONLINE run but has no loss yet. No failures or
preemptions appear in the observer's current canonical job snapshots.

The cached startup evidence in subset-training-r02-attempt-identities.json now
joins all96 nonzero task identities to attempt0, host logging and recorded
native run identities, including cell29's run without a loss yet.
Previous observation timestamps and the earlier cell17/task2 transport error
remain preserved. These are startup assignment identities, not a separate
runtime JAX-device audit or completion evidence. The six alpha0 cells remain
the only scientifically validated completions. No nonzero group has the
validated completions required for LR expansion.

### 2026-11-05 08:53 UTC — prepare expansion observation

Added watch_subset_expansions.py for one exclusive observer per future expansion
campaign. It binds accepted registry entries to the frozen source, configuration
digest and canonical job identity. It durably records a guard before spawning
each follower. The first canonical check is scheduled120 seconds after discovery;
each subsequent check is scheduled570 seconds after the previous check returns.
Restoration preserves recorded follower identities without establishing process
liveness. Saved terminal snapshots are reused without polling those jobs again.
RPC errors retain the last successful snapshot and raw output. If a previously
initialized campaign's state file is missing, observation stops pending reconciliation.

Synthetic fixtures passed with all subprocesses, thread pools and time mocked,
plus real temporary file locks. They checked exclusive ownership, terminal
restoration, transport errors and interrupted follower creation. Required lint
passed. This qualifies local orchestration only; no expansion observer, follower
or job has been launched. A real eligible group, fresh immutable campaign
worktree and expansion scientific terminal validator remain required.
Evidence: expansion-observer-preflight.json.

All24 nonzero pilots remain running, with six alpha0 results validated. At08:51,
one native stream for each of23 established pilots advanced, reaching maximum
training-step label4981. Cell29 remains in ONLINE startup before its first loss.
Its canonical snapshot is unchanged since the previous observation; this is not
a second scheduled check without progress. Continue the existing pilot observer.

### 2026-11-05 09:22 UTC — prepare expansion terminal validation

Added finish_subset_expansions.py to plan reads from accepted registry entries
and saved canonical terminal snapshots, then collect and verify terminal tasks.
The existing collector permits at most one automatic read attempt per task
identity and preserves ambiguous reads for reconciliation. Scientific validation
requires one successful attempt per host, matching source, task and ONLINE run
identities, the exact frozen configuration, all five required finite training
metrics at labels0–7629, final C4 loss, finished ONLINE runs, permanent checkpoint
metadata and independently read stored counter7630. Failed terminal outcomes
remain collectable evidence; this validator rejects failed or retried jobs pending
separate diagnosis and lineage adjudication. Previously validated completions
are reused only after recorded evidence hashes and saved canonical outcomes agree.

Offline one-host and four-host fixtures exercised planning, cached task collection,
full artifact validation and completed-result reuse without further external reads.
The validator rejected all ten invalid-artifact cases per topology, including
wrong counters, missing final evaluation, conflicting history, metadata and
source/tracking/host identity errors. Changed terminal owner or canonical evidence
also blocked reuse. Git source lookup and worker reconstruction were mocked in
the integration fixture, as were W&B and storage services. Separate worker decode
and ONLINE nontraining checks are recorded in expansion-worker-preflight.json.
Required lint passed. Evidence: expansion-terminal-validator-preflight.json.

This prepares orchestration and validation; no expansion has launched and no
full checkpoint restore is claimed. A complete validated predecessor alpha/ridge
group must authorize each expansion before a fresh immutable campaign is created.
The published R02 validator is unchanged. At09:19 all24 nonzero pilots continued
advancing, reaching maximum native label5410; cell29 reached292. Six alpha0
results remain the only validated completions, with no failures or preemptions
in the canonical pilot snapshots. Keep the existing observer and all live jobs.

### 2026-11-05 11:51 UTC — validate the first two nonzero pilots

Cells 06 and 09 completed their 7,630-update schedules on the immutable R02
source. Both use alpha 0.125 and learning rate 0.0025. Final C4 losses are
4.168757438659668 for cell 06 at ridge 0.0001 and 4.160338401794434 for
cell 09 at ridge 0.01. Each cell used four hosts.
The sole observer recorded canonical success; eight terminal task snapshots
confirm one successful attempt per host, exit code 0, and the original task,
worker and ONLINE run identities. Each native history contains all 7,630 finite
rows for loss, duration, MFU, optimizer learning rate and Adam learning rate,
plus final C4 evaluation at label 7629. The separate native training and host
logging W&B runs have verified finished states. Permanent checkpoint metadata contains
57 arrays, and independent read-only scalar reads returned native counter 7630.
Evidence: subset-training-r02-nonzero-completed.json.

The first validation attempt rejected both manifests because it required exact
chunk equality with the one-host alpha0 reference. All array paths, shapes and
dtypes match, but 16 chunk layouts differ. Levanter's plan_array_write derives
chunks from sharding and writer replicas. The corrected R02 and expansion
validators require identical array schemas and formats, validate chunk rank and
positive divisor geometry, and preserve the differing layouts in their evidence.
The original validation errors remain saved. Regression fixtures rejected missing
or duplicate arrays, changed shapes or dtypes, and invalid chunk geometry.
Full mocked R02 and one-host/four-host expansion artifact fixtures passed with
a differing chunk layout, along with required lint. Both real completion caches
were reused in validator reruns with W&B API access disabled; training used ONLINE
tracking throughout. Evidence: checkpoint-manifest-layout-preflight.json.
These checks do not establish a full checkpoint byte audit or full-state restore.

Eight pilots are now scientifically validated; 22 remain canonically running
in the observer's 11:46–11:47 UTC snapshots, with no failures or preemptions.
The 11:46 UTC ONLINE observation preceded this validation and showed progress
in all 24 then-unvalidated streams.
Neither alpha 0.125 ridge group is complete, so no new expansion is eligible.
The six completed alpha0 pilots retain their interior LR winner,
0.003535533905932738. Global selection is not yet authorized by complete results, and confirmations and scientific probes remain outstanding.
Evidence: subset-training-r02-completion-readiness.json. Continue the existing
observer and all live jobs; exclude the eight validated runs from future queries.

### 2026-11-05 12:08 UTC — finish the alpha 0.125 grid and prepare LR expansions

All six alpha 0.125 pilots are scientifically validated at 7,630 updates.
For ridge 0.0001, final C4 losses at learning rates 0.0025, 0.005 and 0.01
are 4.168757438659668, 4.192734718322754 and 4.221353054046631.
For ridge 0.01, the corresponding losses are 4.160338401794434,
4.1679792404174805 and 4.230307579040527. The lowest observed loss in each ridge group occurs at LR 0.0025,
the lower tested boundary. The preregistered factor-two rule therefore requires one new
LR 0.00125 pilot for each ridge, each from scratch for 7,630 updates.
Evidence: subset-training-r02-nonzero-completed.json and
subset-training-r02-completion-readiness.json. Twelve initial pilots are
validated; 18 remain running in the sole observer's 11:56 UTC snapshots.
Global selection and confirmations still require the remaining groups and
any further boundary expansions.

Sixteen new terminal task snapshots confirmed canonical success, one original
attempt per host and exit code 0 for cells 07, 08, 10 and 11. Cells 08 and 10
initially failed validation because seven terminal worker references were
empty. The empty fields are consistent with Iris remove_worker, which clears task
and attempt worker references. The saved
startup task snapshots match the terminal task IDs, original attempt UIDs and
start times, and retain the assigned worker IDs. Validation now records these
historical assignments separately while preserving the empty terminal fields.
A nonempty conflicting worker, changed attempt UID/start time or missing startup
evidence still rejects the result. The controller supports this deletion behavior;
no separate worker-deletion event was queried. Portable assignment proofs are
also checked by the complete-group selection and confirmation submission gate.
Regression fixtures, full mocked R02 and one/four-host expansion validation,
the expansion submitter fixture and required lint passed. Initial errors remain
saved. Evidence: terminal-worker-reconciliation-preflight.json.

All four new results also passed the exact subset/source/ONLINE joins, full
finite metric-history and final C4 checks, finished W&B states, permanent
57-array checkpoint metadata and independent stored-counter reads of 7630.
Subsequent selection and expansion gates reuse saved validated results
without new remote queries. No full checkpoint byte
audit or full-state restore is claimed.

Frozen expansion inputs are subset-expansion-r01-bundle.json,
subset-expansion-r01-completions.json and subset-expansion-r01-accepted.json.
They contain the two eligible cells, all 12 predecessor results and no prior
expansions, with fresh artifact version 2026.11.05.10. Both configurations
reconstructed exactly through the qualified worker without numerical kernels.
Publish these inputs and create a dedicated immutable execution worktree before
submission. Use normal batch priority, default preemption/retries, permanent
launch guards and one expansion observer; leave all existing pilot jobs intact.

### 2026-11-05 12:10 UTC — submit the two lower-LR expansions

The full regional catalog check passed for all 26 components. Both alpha 0.125,
LR 0.00125 expansions were accepted at normal batch priority, at ridges
0.0001 and 0.01. Submission ended with exit code 0. The immutable execution source is
1dffb9eb1c1e8b58487cf3716ed70549a859694b on branch
codex/20-ema-gram-expand-seed0-r01-20261105, in worktree
/Users/kaiyuew/Downloads/Project/marin/ema-gram-expand-seed0-r01-20261105.
Campaign ema-gram-20-expand-seed0-r01-20261105 uses fresh artifact version
2026.11.05.10. Each job requests v4-32 with four hosts; acceptance does not
establish allocation or training progress. Default preemption/retries remain
in effect, with no queue, task, training or wrapper wall-clock limits.

The shared permanent expansion registry records both launch guards, exact
source/config/input bindings, job identities and successful submission results.
A dedicated observer owns these two jobs, with one guarded follower per job.
Its first canonical checks are scheduled 120 seconds after discovery, followed
by 570-second intervals. Existing pilot jobs and their observer are unchanged.
Evidence: subset-expansion-r01-launched.json. Scientific completion still
requires all terminal, ONLINE, history and checkpoint-counter gates; global
selection and confirmations remain pending.

### 2026-11-05 12:11 UTC — validate the first alpha 0.25 pilot

Cell 12, alpha 0.25, ridge 0.0001 and LR 0.001, passed full pilot validation:
one successful original attempt on each of four hosts, exact source/config and
ONLINE joins, all 7,630 finite metric rows, final C4 loss 4.276551723480225,
finished native/host W&B runs, permanent 57-array checkpoint metadata and
independently read stored counter 7630. Its terminal task records retain all four assigned worker IDs.
Evidence: subset-training-r02-nonzero-completed.json. Thirteen initial pilots
are validated; 17 were observed running in the 12:05–12:06 UTC owner snapshots.
This alpha 0.25 group is incomplete. The complete-group replay finds no new
unguarded expansion: the two alpha 0.125 lower-LR cells are already accepted.
The expansion execution worktree retains the 12-result input snapshot that
authorized its launch; cell 12 was validated afterward.
No global selection, full checkpoint restore or confirmation completion is claimed.

### 2026-11-05 12:15 UTC — reconcile expansion observer output

The expansion observer's first two RPCs succeeded at 12:11 UTC and reported
both jobs RUNNING with four running tasks each, zero failures and zero
preemptions. The observer merged stderr diagnostics into stdout, then failed
to parse the mixed text as JSON. The raw responses were preserved. Its checker
now stores stdout JSON and stderr separately and retains both paths on a parse
error. Full-loop mocked regression and exclusive-owner/cadence/follower guards
passed, as did required lint.

After verifying the old observer's command and held lock, only that observer
process was stopped. Its exit code was 143. Under the released exclusive lock,
the two successful responses were decoded locally into separate JSON records;
original mixed output and errors remain preserved. The replacement observer
records its new process identity immediately. Both original follower PIDs and
next-check times were verified unchanged, with no new RPC or follower during
recovery. The next checks remain scheduled for 12:21:14 UTC. Training jobs,
allocations and the original pilot observer were untouched.
Evidence: expansion-r01-observer-recovery.json. This was a monitoring parser
failure; the saved canonical responses show running workers.

The expansion startup record now joins all eight assigned tasks to original
attempt 0, matching attempt UIDs and finished ONLINE host logging runs. These
short, non-training host checks record source/configuration identity before
the native training process starts; their completion is expected. Host
checks match the source, frozen configuration, bundle and qualified worker
digests. Both fresh native ONLINE runs passed exact configuration validation;
neither had a loss in the initial saved API snapshot. These joins establish
four distinct assigned workers per job, not a separate runtime JAX-device audit.
Evidence: subset-expansion-r01-started.json. Iris supplies IRIS_TASK_ID as
`task_id:attempt_id`; the initial startup matcher assumed a bare task ID.
The matcher was corrected using cached records without requery, and the
expansion terminal validator and its one/four-host fixtures now require the
same exact wire format. No training source or job was changed.

### 2026-11-05 12:30 UTC — validate cell 15 and observe expansion updates

Cell 15 (alpha 0.25, ridge 0.01, LR 0.001) passed full pilot validation at
7,630 updates, with final C4 loss 4.254374980926514. All four hosts succeeded
on their original attempt. Exact source/config/ONLINE joins, 7,630 finite
metric rows, finished W&B runs, permanent 57-array checkpoint metadata and
independently read stored counter 7630 passed. The cumulative nonzero record
now contains eight results; with six alpha0 results, 14 initial pilots are
validated. Sixteen initial jobs were RUNNING in the 12:25 UTC owner snapshots.
Both alpha0.25 ridge groups remain incomplete; replay found no new expansion
beyond the two already accepted alpha0.125 jobs. Global selection remains pending.
Evidence: subset-training-r02-nonzero-completed.json and
subset-training-r02-completion-readiness.json. No full checkpoint restore is claimed.

Both alpha0.125 expansions at LR 0.00125 now have finite native training metrics.
At 12:28 UTC, ridge 0.0001 reached native step label 41 with training loss
11.39132022857666; ridge 0.01 reached label 7 with loss 12.216241836547852.
These are early training observations, not final evaluation results. Fresh
ONLINE configuration/source checks passed. Task/attempt identities relied on
the saved startup joins; no new task-status reads were made. The current owner
snapshots retained the same job/source bindings and failure/preemption counts. Both jobs had four running tasks and zero
failures/preemptions in the owner's 12:21 UTC snapshots. The corrected observer
parsed those responses successfully. Its next checks are scheduled for
12:30:51 UTC. The original expansion00 log follower ended with Finelog
`StatsError: Bad Gateway`; its log is preserved and it was not restarted.
The independent ONLINE metrics establish training progress; the follower
error alone does not establish worker failure. Evidence: subset-expansion-r01-first-progress.json.

The R02 progress helper initially imported a superseded scratch validator
because its script directory preceded PYTHONPATH. That local cache check
rejected a previously reconciled worker identity before any ONLINE query.
The helper now inserts the published logbook directory before importing the
validator. The corrected invocation reused 13 validated results and observed
progress in all 17 then-unvalidated native streams; cell 15 passed full
validation afterward. Original errors and corrected observations are retained.

### 2026-11-05 12:59 UTC — validate cell 14

Cell 14 (alpha 0.25, ridge 0.0001, LR 0.004) passed full pilot validation at
7,630 updates, with final C4 loss 4.359232425689697. All four original task
attempts succeeded with exit code 0. Exact subset/source/ONLINE identity joins,
7,630 finite metric rows, finished W&B runs, permanent 57-array checkpoint
metadata and independently read stored counter 7630 passed. All four terminal
worker references were empty; [the published reconciliation rule](terminal_worker_identity.py)
verified the original attempt UIDs and start times against saved startup assignments,
retaining the empty raw fields and separate historical worker identities.
No new startup queries or full checkpoint restore were used.

Fifteen initial pilots are scientifically validated. Fifteen others were
RUNNING in the owner's 12:53–12:54 UTC snapshots, with no failures or preemptions.
The alpha0.25/ridge0.0001 group still lacks its LR0.002 result. Replaying all
groups while leaving incomplete groups unresolved found no new expansion.
The two accepted alpha0.125/LR0.00125 expansions continue, with finite native
step labels 484 at ridge0.0001 and 443 at ridge0.01 at 12:58 UTC. Their saved
startup identities were reused; fresh ONLINE source/configuration checks passed.
Global selection remains pending. Evidence: subset-training-r02-nonzero-completed.json,
subset-training-r02-completion-readiness.json and
scratch/20261105-expansion-r01-125801-progress.json.

### 2026-11-05 13:44 UTC — validate cell 16

Cell 16 (alpha 0.25, ridge 0.01, LR 0.002) completed all 7,630 updates with
final C4 loss 4.211307048797607. The canonical job and all four original
attempt-0 tasks succeeded with exit code 0. The validator matched the run's
configuration and source to the frozen subset configuration and accepted launch,
checked all 7,630 finite metric rows and finished online W&B runs, and verified
metadata for 57 arrays in regional checkpoint storage. It independently read
stored counter 7630. Empty terminal worker references on all four tasks were
reconciled against saved startup attempt UIDs, start times and assignments using
[the published rule](terminal_worker_identity.py). Raw empty references and
historical assignment proofs remain separate. No startup queries, full checkpoint
byte audit or full checkpoint restore were performed.

Sixteen of the 30 initial pilots now pass these completion gates; fourteen remain
RUNNING in the owner's 13:41–13:42 UTC snapshots, with no failures or preemptions.
The alpha0.25/ridge0.01 group still needs cell 17 (LR0.004), so no group decision
uses its partial results. Applying the preregistered factor-two boundary rule to
complete groups found no new expansion. The two accepted alpha0.125/LR0.00125
expansions advanced to native training step labels 1161 at ridge0.0001 and 1121
at ridge0.01 at 13:42 UTC. Global LR/ridge/exponent selection awaits all pilot
groups and their required expansions. Evidence:
[completed results](subset-training-r02-nonzero-completed.json),
[group readiness](subset-training-r02-completion-readiness.json), and local
scratch/20261105-expansion-r01-134257-progress.json.

### 2026-11-05 13:54 UTC — validate cell 18

Cell 18 (alpha 1/3, ridge 0.0001, LR 0.0005) completed all 7,630 updates with
final C4 loss 4.432610034942627. Its canonical job and four original attempt-0
tasks succeeded with exit code 0. The validator matched the resolved configuration to the frozen subset configuration
and the source revision to the accepted launch ledger, checked all 7,630 finite
metric rows and finished online W&B runs, and verified metadata for 57 arrays in
regional checkpoint storage. An independent scalar read returned stored counter
7630. All terminal task and attempt worker IDs matched their cached startup worker IDs.
No startup queries, full checkpoint byte audit or full checkpoint restore were
performed. The four new terminal snapshots bring the cached terminal-task total for alpha greater than zero
to 44; existing validated results were reused.

At 13:54 UTC, seventeen of 30 initial pilots pass these completion gates. Thirteen remain
RUNNING in the owner's 13:51 UTC snapshots, with no failures or preemptions.
The alpha1/3/ridge0.0001 group still needs LR0.001 and LR0.002; its partial results
are excluded from group selection. Applying the preregistered factor-two
boundary rule to complete groups found no additional expansion required by the
currently complete groups. The accepted
alpha0.125/LR0.00125 expansions advanced to native training step labels 1325 at
ridge0.0001 and 1281 at ridge0.01 at 13:52 UTC. Global LR/ridge/exponent selection
awaits all groups and required expansions. Evidence:
[completed results](subset-training-r02-nonzero-completed.json),
[group readiness](subset-training-r02-completion-readiness.json), and local
scratch/20261105-expansion-r01-135259-progress.json.

### 2026-11-05 14:04 UTC — complete both alpha 0.25 groups

Cells 13 and 17 passed all 7,630-update completion gates. Cell 13 (ridge0.0001,
LR0.002) has final C4 loss 4.284524440765381; cell 17 (ridge0.01, LR0.004) has
4.241844654083252. Both canonical jobs and all eight original attempt-0 tasks
succeeded with exit code 0. Resolved configurations and source revisions match
the frozen subset configurations and accepted launch ledger. All 7,630 finite
metric rows per run, finished online W&B runs, checkpoint metadata covering 57 array schemas and chunk geometries,
and independently read stored checkpoint counters 7630 passed. Terminal worker IDs match
the saved startup identities. No full checkpoint byte audit or restore was
performed. Nineteen of 30 initial pilots are validated; eleven remain RUNNING
in the owner's 14:01 UTC snapshots, with no failures or preemptions.

At alpha0.25/ridge0.0001, LR0.001 wins the complete three-point grid with
C4 loss 4.276551723480225. This is the lower boundary, so the preregistered
factor-two rule requires a new LR0.0005 pilot. At ridge0.01, LR0.002 wins with
C4 loss 4.211307048797607 and is interior; that group needs no expansion.
The [new expansion bundle](subset-expansion-r02-bundle.json) freezes one
seed0, 7,630-update run from scratch at alpha0.25/ridge0.0001/LR0.0005 on v4-32,
using artifact version2026.11.05.11. Its selection-evidence inputs contain all19 completed initial
results and the two already accepted alpha0.125 expansions. The latter retain
their existing job identities and permanent launch-started guards. The new campaign will use a separate
immutable worktree and the shared permanent expansion registry. This entry
records preparation; no new launch has occurred yet. Global alpha/ridge/LR selection awaits all initial pilots and required expansions. See [completed results](subset-training-r02-nonzero-completed.json)
and [group readiness](subset-training-r02-completion-readiness.json).

### 2026-11-05 14:08 UTC — launch the alpha 0.25 boundary expansion

Iris accepted one new pilot required by the lower-boundary winner reported above:
alpha0.25/ridge0.0001/LR0.0005, campaign
`ema-gram-20-expand-seed0-r02-20261105`, job suffix 00. Its immutable source is
`1363a08b76d442fbbeb17659cec8f99ea21064a3` and artifact version is 2026.11.05.11.
The full 26-component cache metadata check passed before submission. The submitter
ended with exit 0 after saving a permanent launch-started guard in the shared
registry to prevent duplicate submission of these settings. It requested
v4-32/four hosts/16 JAX devices, normal batch scheduling, default preemption/retries
and all 7,630 updates from scratch. Requested topology does not establish allocation.
No wall-clock limit was added.

Observer session 51045/PID43252 holds the exclusive monitoring lock for this
campaign; followerPID43256 reads its job logs. The first authoritative Iris job
status check is scheduled for 14:09:27 UTC. Job state and matches between the
startup W&B runs and task attempts are still pending. The initial-pilot observer
and expansion R01 observer were not changed. All three required expansion settings
now have permanent guards, so replaying completed-group decisions proposes no
additional submission. See [launch evidence](subset-expansion-r02-launched.json)
and [group readiness](subset-training-r02-completion-readiness.json).

At 14:09:32 UTC, the new campaign's sole observer recorded RUNNING with four
tasks and zero failures/preemptions. The startup audit then matched all four
original attempt-0 UIDs and distinct assigned worker IDs to their finished
nontraining online W&B logging checks. The native online training run matches
the frozen configuration and immutable source; no optimizer-step metric has
appeared yet. Saved follower logs provide the task/run joins. Each task was read
once with a durable read-start guard, and the five-run API snapshot is cached.
These assignments are not a separate runtime JAX-device audit. The next owner
check is 14:19:02 UTC. See [startup evidence](subset-expansion-r02-started.json).

### 2026-11-05 14:14 UTC — complete the alpha 1/3, ridge 0.0001 grid

Cells 19 and 20 passed all 7,630-update completion gates. At alpha1/3 and
ridge0.0001, LR0.001 has final C4 loss 4.415025234222412 and LR0.002 has
4.65962028503418. Both canonical jobs and all eight original attempt-0 tasks
succeeded with exit code 0. The validator matched each resolved configuration
to its frozen subset configuration and its source revision to the accepted
launch ledger. All 7,630 finite metric rows per run, finished online W&B runs,
checkpoint metadata covering 57 array schemas and chunk geometries, and
independently read stored checkpoint counters 7630 passed. Empty terminal task and attempt worker_id fields
on all eight tasks were reconciled with cached startup task/attempt UIDs,
start times and assigned worker IDs using [the published rule](terminal_worker_identity.py).
Raw blanks and historical assignment proofs are retained. No new startup reads,
full checkpoint byte audit or restore were performed.

Together with cell18 at LR0.0005 (C4 loss4.432610034942627), these results complete
this three-point grid. LR0.001 is the interior winner, so the preregistered
factor-two boundary rule requires no expansion for this group. Twenty-one of30
initial pilots pass the completion gates; nine remain RUNNING in the owner's
14:10–14:11 UTC snapshots, with no failures or preemptions. The alpha0.125/LR0.00125 expansions advanced to native step label1612 at
ridge0.0001 and1571 at ridge0.01 at14:13 UTC. The new alpha0.25/ridge0.0001/
LR0.0005 expansion retains its verified online identity and four running tasks; its first
optimizer-step metric is still pending. Global alpha/ridge/LR selection awaits
the remaining initial groups and required expansions. Evidence:
[completed results](subset-training-r02-nonzero-completed.json),
[group readiness](subset-training-r02-completion-readiness.json), and local
scratch/20261105-expansion-r01-141307-progress.json and
scratch/20261105-expansion-r02-141307-progress.json.

### 2026-11-05 14:21 UTC — first training metric from the alpha 0.25 expansion

The additional LR0.0005 pilot for alpha0.25/ridge0.0001 reported zero-based
optimizer-step label7 and finite training loss12.216623306274414 in the14:21 UTC
online observation. Its W&B run matches the frozen configuration and immutable
source `1363a08b76d442fbbeb17659cec8f99ea21064a3`. The sole observer for campaign
`ema-gram-20-expand-seed0-r02-20261105` recorded job00 RUNNING with four tasks
and cumulative failure/preemption counts of zero at14:19 UTC. The unchanged
counts permit reuse of the saved mappings between startup tasks, original
attempt0 identities and online runs. No startup queries were repeated.

This is an early training metric. The pilot must still finish all7,630 updates,
final C4 evaluation and the [full completion checks](verify_expansion_artifacts.py).
The alpha0.125/LR0.00125 pilots advanced to labels1743 at ridge0.0001 and1699
at ridge0.01. All nine unfinished initial pilots also advanced;21 initial pilots
remain fully validated. [Complete-group readiness](subset-training-r02-completion-readiness.json)
requires no additional expansion now. Global selection awaits the unfinished
pilots and required expansions. See [first progress](subset-expansion-r02-first-progress.json).

### 2026-11-05 14:41 UTC — validate cell23

Cell23 (alpha1/3, ridge0.01, LR0.002) passed all7,630-update completion checks,
with final C4 loss4.260484218597412. The canonical job and all four original
attempt0 tasks succeeded with exit0. Exact frozen subset/configuration/source,
finished online W&B runs, all7,630 finite metric rows,checkpoint
metadata covering57 array paths, shapes, dtypes and valid chunk geometries and independently read stored counter7630 passed. Empty terminal
task and attempt worker IDs were reconciled against cached startup task/attempt
UIDs, start times and assigned workers using [the published identity rule](terminal_worker_identity.py);
raw blanks remain preserved. No new startup queries, full checkpoint byte audit
or checkpoint restore were performed.

Twenty-two of30 initial pilots are now validated. This alpha/ridge group still
awaits LR0.0005 and0.001 (cells21/22), so no group winner or additional expansion
is selected. The eight remaining initial pilots and three previously selected expansions
continue. Initial-pilot owner snapshots at14:39 UTC and expansion owner snapshots
at14:35–14:38 UTC report cumulative failure/preemption counts of zero.
See [completed results](subset-training-r02-nonzero-completed.json) and
[complete-group readiness](subset-training-r02-completion-readiness.json).

### 2026-11-05 15:36 UTC — prepare the confirmation observer

The [confirmation observer](watch_subset_confirmations.py) is prepared for the
future seed0 full-training campaign. Before observing jobs, it verifies the
committed input bytes, selection lock, complete pilot results, configuration
bundle and submission ledger. Each accepted job must match the exact source,
configuration digest and submission command. An unresolved submission remains
excluded until its acceptance is explicitly reconciled.

The observer reuses the expansion observer's polling loop. Local fixtures passed
for one observer per campaign, first checks after120 seconds and subsequent checks
after570 seconds, recovery preserving existing log-streaming processes, and
preservation of RPC errors and separate stderr. Restarting the observer after
jobs finish creates no new queries or log-streaming processes.
A confirmation CLI fixture also exercised delayed acceptance and rejected a
changed accepted command before any new query, follower or state change.
[Validation evidence](confirmation-observer-preflight.json) uses fictional
results and mocked external processes; no Iris request, numerical kernel or
training run was started. Existing live observers were not restarted.

Global selection and real confirmation submissions still await all initial
pilots and required recursive LR expansions. Held-out, norm-matched direction/loss probes and attention/MLP production-polar
versus SVD comparisons remain required after the reference confirmation
checkpoints become available; training alone does not complete the core study.

### 2026-11-05 15:39 UTC — validate cell22

Cell22 (alpha1/3, ridge0.01, LR0.001) passed the full7,630-update completion
checks with final C4 loss4.281885147094727. The canonical job and all four
original attempt0 tasks succeeded with exit0. The observed subset,
configuration and source matched the frozen campaign inputs and launch record;
online tracking finished. All7,630 metric
rows were finite, checkpoint metadata covered57 arrays, and the independently
read stored counter was7630. All four terminal worker identities matched the
cached startup assignments. Validation reused the startup evidence without
additional startup queries; no full checkpoint byte audit or restore was performed.

Twenty-three of30 initial pilots are now validated. This alpha/ridge group
still awaits cell21 (LR0.0005), so group selection remains pending. The seven
unfinished initial pilots and three accepted expansions continue. The latest
initial-pilot owner snapshots at15:37 UTC report zero cumulative failures and
preemptions. See [completed results](subset-training-r02-nonzero-completed.json)
and [complete-group readiness](subset-training-r02-completion-readiness.json).

### 2026-11-05 16:09 UTC — complete the alpha1/3 high-ridge group

Cell21 (alpha1/3, ridge0.01, LR0.0005) passed the full7,630-update completion
checks with final C4 loss4.395976543426514. The parent job and its four worker
tasks succeeded on their original attempts with exit0. All7,630 finite metric rows, finished
online tracking, resolved configuration and source revision matching the frozen manifest,
57-array checkpoint metadata
and independently read stored counter7630 passed validation. Terminal worker
identities matched cached startup assignments. Validation reused startup evidence;
no additional startup query, full checkpoint byte audit or restore was performed.
Twenty-four of30 initial pilots, cells00..23, are now fully validated.

The completed alpha1/3/ridge0.01 group has final C4 losses4.395976543426514,
4.281885147094727 and4.260484218597412 at LR0.0005,0.001 and0.002. The lowest
loss occurs at the upper tested LR0.002, so the prescribed factor-two rule
requires a new LR0.004 pilot. The prepared
[R03 bundle](subset-expansion-r03-bundle.json) contains that one seed0 pilot,
using the same7,630-update schedule and v4-32 topology. Its
[completion inputs](subset-expansion-r03-completions.json) freeze24 validated
predecessors, and its [accepted inputs](subset-expansion-r03-accepted.json)
map the three existing expansion configurations to their submitted job IDs. Campaign
`ema-gram-20-expand-seed0-r03-20261105` and artifact version`2026.11.05.12`
are fresh. The bundle, completion inputs and accepted inputs will remain fixed in the
dedicated execution worktree.

The six alpha0.5 initial pilots and three accepted expansions are running. The new
LR0.004 pilot is prepared but has not yet been submitted. Global learning-rate/ridge selection,
37,335-update confirmation runs and held-out direction/loss and polar-versus-SVD
checkpoint probes remain pending. See
[completion evidence](subset-training-r02-nonzero-completed.json) and
[group readiness](subset-training-r02-completion-readiness.json).

### 2026-11-05 16:16 UTC — accept the alpha1/3 high-ridge expansion

Iris accepted the required alpha1/3, ridge0.01, LR0.004 seed0 pilot after all
26 data-cache components passed preflight. R03 uses immutable source commit
`282ba9da0ae0ee71aa43847ab19b61e022dbcdef` and fresh artifact version
`2026.11.05.12`. A permanent guard prevents duplicate submission. One monitoring
process owns the accepted job; the first scheduled Iris status query and checks
linking online W&B runs to the expected workers are pending.

Four expansions are now accepted, and every currently required expansion has a
submission guard. The initial validated count remains24/30. Global LR/ridge
selection,37,335-update confirmation runs, and held-out direction/loss and
polar-versus-SVD checkpoint probes remain pending. See the
[accepted launch](subset-expansion-r03-launched.json).

At16:18 UTC, the first scheduled Iris query reported four pending tasks and zero
failures or preemptions. The scheduler is waiting for a compatible four-worker
allocation; the autoscaler reports demand-routed workers not yet ready. Training
and startup online identity checks have not begun. The next owner check is
16:27:49 UTC. The accepted job remains queued without resubmission or cancellation.

### 2026-11-05 16:28 UTC — verify fourth expansion startup

The sole monitoring process reports the alpha1/3, ridge0.01, LR0.004 pilot
RUNNING with four tasks, zero failures and zero preemptions. Four original-attempt
task IDs, attempt UIDs and assigned workers match the identity fields recorded
in four finished online W&B host-logging checks. The trainer's own W&B run
records the expected configuration and source revision from the frozen manifest.
Its latest summary has no optimizer step or training loss yet; cached worker
logs show compilation. These are startup checks, not training completion or an
independent runtime JAX device audit. The initial validated count remains24/30.
All four expansions now have running parent jobs; six initial alpha0.5 pilots
remain running. See [startup evidence](subset-expansion-r03-started.json).

### 2026-11-05 16:42 UTC — accept an independent restore qualification job

Iris accepted a separate regional v4-8 job to read the validated alpha0.25,
ridge0.01, LR0.002 pilot checkpoint at stored counter7630. This advances
infrastructure preparation while pilot selection remains incomplete. It is
not a confirmation run or a prescribed scientific reference checkpoint. The
frozen request retains the full model and optimizer configuration; only the
restore mesh changes from four hosts to one host with four TPU devices.

Fresh ONLINE logging and a full-size abstract TrainerState check passed. The
template matches the checkpoint manifest's57 array shapes and dtypes. Actual
checkpoint reads and hardware restore remain unverified. The new worker refuses
cross-region reads and partial restores. It checks metadata.json and manifest.json
against their validated object generations and hashes, verifies the stored
counter, and tests every restored array for finiteness. The restored state
includes model, optimizer, counter and training PRNG; no data-position replay
is verified. The job makes no optimizer updates or checkpoint writes.

A permanent guard prevents duplicate submission, and one monitoring process
owns the accepted job. Existing scientific runs continue unchanged. GPU runtime
and numerical validation, GPU allocation retention, and larger-TPU full-training
validation remain outstanding. See the [frozen request](restore-qualification-request.json),
[worker](qualify_checkpoint_restore.py) and [launch evidence](restore-qualification-launched.json).

### 2026-11-05 16:45 UTC — pass actual native checkpoint restoration

The qualification job and its sole original-attempt worker succeeded. The saved
regional report confirms that Levanter's native checkpoint loader, with
`allow_partial=False`, restored the complete TrainerState on four v4 TPU devices.
All57 arrays are finite, the model and optimizer are present, counter7630
matches, and the training PRNG has the expected uint32[2] shape. Both fresh
online W&B runs finished; their recorded source, request, task and attempt
identities match the frozen request and accepted Iris job. The794-byte archive
containing host-logging.json and restore-report.json passed its recorded SHA256
and a readback restricted to the recorded object generation.

This closes one-host native restore qualification. It does not verify data-position
replay, resumed training, or full-size direction, loss and SVD calculations. The
scientific probes still require the three prescribed alpha0.25 confirmation
checkpoints. No checkpoint was modified and no optimizer update ran. Reuse
[completion evidence](restore-qualification-completed.json) without querying the
completed job, task, followers or W&B runs again.

### 2026-11-05 16:48 UTC — fourth expansion begins training

The fourth learning-rate expansion, Gram exponent alpha=1/3, ridge0.01,
LR0.004, has begun training. Its online W&B run records native optimizer-step
label99 and finite training loss8.38945961 with the expected source and
configuration. Six unfinished initial pilots and all four additional expansions
now show training progress. The initial validated count remains24/30; early
training loss does not satisfy final evaluation or checkpoint completion gates.
See [first-progress evidence](subset-expansion-r03-first-progress.json).

A separate worker is prepared to exercise the original direction, paired-loss
and decomposition probe functions on the full447M model. It will restore the
validated alpha0.25 pilot's complete model, optimizer, counter7630 and training
PRNG in the checkpoint's region. The frozen qualification request uses two
held-out Wikipedia batches and common perturbation step sizes1e-5,3e-5,1e-4.
Success requires27 finite paired-loss rows and18 finite attention/MLP
production-polar versus SVD comparisons. Intermediate reports preserve completed
stages. Actual execution remains unverified. Any results qualify infrastructure
only; scientific reference probes still require the confirmation checkpoints
at counters3733,18665,33597 specified in the
[execution plan](single-seed-execution-plan.md). See the
[frozen request](fullsize-probe-qualification-request.json) and
[worker](qualify_fullsize_probe.py).

### 2026-11-05 16:59 UTC — accept full-size probe qualification

Iris accepted the independent full-size probe job, requesting one v4-8 TPU
allocation in us-central2. Fresh ONLINE W&B preflight runs passed logging and
abstract TrainerState shape/dtype checks against the57-array checkpoint
manifest at source0b8bced61. These checks do not establish an actual restore.
One monitoring process checks the job; a permanent submission record prevents
duplicate launches. Its first scheduled check is17:01:38 UTC on2026-11-05.
The worker is configured to validate the26-component regional data-cache
catalog before reading held-out batches, then restore state and run the probes
described above. Actual hardware execution and probe results remain unverified.
This submission did not modify existing training jobs.
See [launch and preflight evidence](fullsize-probe-qualification-launched.json).

At17:01 UTC, the first scheduled check reports one running task and zero
failures or preemptions. Its original attempt UID matches the fresh ONLINE
host and probe runs. The host logging check finished; the probe run reports
four TPU devices on one host and is still running. This verifies startup,
not successful probe calculations. Reuse the
[startup evidence](fullsize-probe-qualification-started.json) and cached task
identity without repeating startup queries.

### 2026-11-05 17:05 UTC — validate two alpha0.5 initial pilots

Two initial pilots with Gram exponent alpha=0.5 and ridge0.0001 passed all
completion gates: LR0.00025 has final C4-en validation loss4.9130744934, and
LR0.0005 has loss5.0308866501. Each has7,630 finite training metric rows,
stored checkpoint counter7,630, the expected57-array checkpoint metadata,
a finished ONLINE W&B run with exact source/configuration identity, and four
successful original-attempt workers. Each completed worker matches its saved
startup task/attempt identity. This validates26 of30 initial pilots. The
alpha0.5/ridge0.0001 group still awaits LR0.001. The expansion rule requires
all three final evaluations before selecting a boundary winner, so no further
expansion is selected yet. These completion checks do not establish full checkpoint-byte
restoration. Reuse [completion evidence](subset-training-r02-nonzero-completed.json)
and [group readiness](subset-training-r02-completion-readiness.json).

### 2026-11-05 17:17 UTC — validate the next alpha0.5 pilot

The alpha=0.5, ridge0.01, LR0.00025 initial pilot passed completion validation.
Final C4-en validation loss is4.5720028877. All7,630 training metric rows are
finite. Stored checkpoint counter7,630 matches the requested update count,
and57-array metadata matches the frozen schema. The ONLINE W&B run finished
with the frozen source/configuration, and all four workers succeeded on
original attempt0. Four empty terminal worker references were joined to the
intended tasks using saved startup UIDs, start times and assignments; the raw
empty fields are retained. No new startup query or checkpoint restore was
performed. Initial completion is27/30. Remaining alpha0.5 pilots are
ridge0.0001/LR0.001 and ridge0.01/LR0.0005,0.001. Both ridge groups remain
incomplete; the expansion rule requires each group's final evaluations before
selecting a boundary winner. See [completion evidence](subset-training-r02-nonzero-completed.json).

### 2026-11-05 17:28 UTC — prepare the settled baseline and next expansion

Two further alpha0.5 pilots passed all completion gates. Ridge0.0001/LR0.001
has final C4-en validation loss5.5128903389; ridge0.01/LR0.0005 has4.4287953377.
Each has7,630 finite metric rows, stored counter7,630,57-array metadata matching
the frozen schema, a finished ONLINE source/configuration match and four
original-attempt successful workers. Empty terminal worker references were
matched using cached startup identities. Initial completion is29/30; only
ridge0.01/LR0.001 remains. The completed ridge0.0001 group selects the lower
boundary LR0.00025, so the protocol requires a new LR0.000125 pilot. Its
[frozen expansion inputs](subset-expansion-r04-bundle.json) contain29 completed
results and the exact configuration/source/job bindings of four accepted
expansions. The LR rule halves a lower-boundary winner selected by final C4-en
loss. See [completion evidence](subset-training-r02-nonzero-completed.json).

The settled alpha0 baseline long-run configuration is now prepared separately:
LR0.003535533905932738, seed0,37,335 updates from scratch, unchanged447M model
and fixed subset, global batch8/context4096, native ONLINE tracking, v4-8,
zero-based final checkpoint label37334. Validation of the
[six saved baseline completion records](subset-training-r02-alpha0-completed.json)
confirms the interior final-C4 winner within the alpha0 group; no model
evaluation was repeated. Extracting the shared configuration builder preserves
the complete serialized configuration in the saved confirmation fixture. This is a prepared
configuration, not a global selection lock or a launched run. A common held-out
loss target for comparing time-to-loss, an amendment if the baseline starts
before global selection, and an exact source/job submission record with one
monitoring owner remain to be fixed before launch. See the
[prepared configuration](alpha0-long-run-prepared.json) and
[preparation checks](alpha0-long-run-preparation-preflight.json).

The required alpha0.5/ridge0.0001/LR0.000125 expansion was accepted at17:32 UTC
from source dc00e1440f9baa3cb6eeda23ff6454865afbbe48. Its submission completed
with exit0 after the full26-component catalog check. The submitter enforces a
permanent duplicate-submission guard, and one observer owns scheduled status checks. Allocation
and training startup are not yet verified. See
[accepted launch](subset-expansion-r04-launched.json).

A bounded read of the independent probe's live report found phase
`loss_probes_complete` with27 rows comparing baseline and perturbed-model losses
on the same held-out batches, and no decomposition rows yet. The worker emits this
phase after the full restore succeeds and the loss calculations return. This
establishes phase progress despite the unchanged ONLINE summary. The optional process listing failed because the container lacks `ps`;
that diagnostic error does not establish worker failure. The job remains under
its existing observer. Finite output validation, SVD results and terminal/archive
checks remain pending. See [phase evidence](fullsize-probe-qualification-phase-progress.json).

### 2026-11-05 17:58 UTC — complete the initial pilot grid

All30 fixed-subset initial pilots are validated. The final cell, alpha0.5,
ridge0.01, LR0.001, has final C4-en validation loss4.3975324631. It passed
7,630 finite training rows, stored counter7,630,57-array checkpoint metadata,
finished ONLINE configuration/source matching and four original-attempt worker
successes. Reuse [completion evidence](subset-training-r02-nonzero-completed.json).
The initial campaign's sole observer exited after all30 canonical successes;
its scope is complete, and expansions have separate monitoring. Expansions
remain required before global selection.

The completed alpha0.5/ridge0.01 group selects its upper-boundary LR0.001.
The next factor-of-two point is LR0.002, prepared in the
[R05 bundle](subset-expansion-r05-bundle.json), which incorporates all30 initial
pilot results and the exact configuration/source/job identities of five
accepted expansions. This preparation does not establish
submission, allocation or training.

R04 startup checks join all four original task attempts to distinct assigned
workers and four finished ONLINE W&B host logging preflight checks. The native
training run matches its frozen configuration and source, with an observed finite loss12.21659
at zero-based update label7. This is early training progress, not final evaluation.
Reuse the [startup evidence](subset-expansion-r04-started.json); no additional
startup queries are needed unless task attempts change.

R05 was subsequently accepted after the full26-component data catalog check.
Its submit process exited0, and one observer owns the new campaign. The permanent
submission guard is retained; allocation and startup identity await verification.
See [accepted R05 launch](subset-expansion-r05-launched.json).

### 2026-11-05 18:14 UTC — qualify independent baseline orchestration

The settled alpha0 long run now has an independent
[execution and monitoring entrypoint](alpha0_long_run.py). It replays the six
validated baseline results, decodes the exact37,335-update configuration and
requires a frozen early-start protocol with a held-out C4 validation-loss
threshold shared by all confirmation runs before execution. The [actual protocol](alpha0-long-run-protocol-draft.json) remains a
draft with no target, so it cannot launch training. Existing global confirmation
selection checks are unchanged.

Local CLI integration checks used temporary committed inputs and a simulated
scheduler. An accepted submission produced one launch, one follower and one
terminal status query; reopening the observer reused terminal evidence. A
simulated transport interruption retained its permanent launch guard and
prevented resubmission. A protocol file modified after its source commit was
frozen was rejected before submission. See
[controller checks](alpha0-long-run-controller-preflight.json). The initial
fixture misclassified a submission command containing the data-check shell
fragment as a standalone cache check; that fixture error was corrected and
its original failure log retained. No real job was submitted by these checks.

A fresh ONLINE W&B non-training logging check finished under exact source
416f96119dae2a996c4cc916ee1ee3dee488f9ec. Its API record matched configuration,
protocol digest, entrypoint hash and local qualification identity. This used a
clearly fictional protocol fixture, not a research target declaration; it ran
no model initialization, kernels or optimizer updates. See
[runtime logging evidence](alpha0-long-run-runtime-preflight.json). Before any
early baseline launch, freeze the shared validation-loss threshold and record
the amendment allowing the selected alpha0 baseline to start before global
selection. Also implement and validate adoption of this exact baseline in the eventual
confirmation set so it is not submitted twice. A clean published execution
worktree and full regional cache check are also required.

At18:16 UTC, R05 startup validation joined four original task attempts to four
distinct assigned workers and finished ONLINE W&B host preflight runs. The native
run matches its frozen source and configuration, with an observed finite training
loss12.246804 at label2. All six accepted expansions now have observed
training progress. This is not final evaluation or expansion completion. Reuse
[R05 startup evidence](subset-expansion-r05-started.json) without repeating the
four task reads or five ONLINE startup queries.

### 2026-11-05 18:35 UTC — prepare baseline adoption and predeclare the target

The independent alpha0 long run can now be accounted for in the eventual full
confirmation set without a second submission or observer. The
[adoption adapter](adopt_alpha0_long_run.py) verifies its original committed
inputs, accepted ledger, exact selected configuration and common target. The
full confirmation controller retains complete-grid and recursive-expansion
selection checks. Its ledger identifies the adopted baseline separately from
new submissions, and acceptance remains distinct from scientific completion.
Temporary Git and fake-scheduler integration checks passed submission, observation
and reopening with one launch and follower per remaining confirmation. Missing
receipts, ambiguous baseline acceptance, changed inputs and a different target
were rejected. The ordinary full-confirmation observer path also passed. See
[validation evidence](alpha0-long-run-adoption-preflight.json), including retained
fixture setup errors and their corrections. No numerical kernels or real jobs
ran in these checks.

The [limited early-start amendment](single-seed-execution-plan.md#settled-baseline-preparation-amendment--2026-11-05-1835-utc)
predeclares a common C4-en loss threshold of4.0, below the best validated
initial-pilot loss4.1497836113. It was chosen before any long-run outcomes.
Every confirmation still runs all37,335 updates; a run that never crosses is
reported as not reaching the threshold within that schedule. The
[protocol](alpha0-long-run-protocol-draft.json) is frozen, while global selection
still awaits recursive expansions. No long run is submitted. The remaining
execution steps are a clean published immutable worktree, fresh campaign,
full regional catalog check, permanent guard and one observer.

At18:31, all six expansion streams advanced: R01 labels5598/5557, R02 label3728,
R03 label1659, R04 label488 and R05 label99. Saved canonical snapshots show no
failures or preemptions. At18:33, the independent full-size probe still reported
`loss_probes_complete` with no SVD completion. Its process CPU counter increased
from5,504,245 to7,569,680 ticks since the previous diagnostic, and the sampled
process state was running. This shows CPU activity; it does not establish the
cause of the missing SVD output or successful probe qualification. Preserve the
existing observer and bounded diagnostic evidence without cancellation.

At18:41 UTC, the [independent execution preparation](alpha0-long-run-execution-prepared.json)
fixed source f99298782188e8f1c4e1ad8a1b2a507693d653e8 in a separate clean,
published worktree for campaign `ema-gram-20-alpha0-seed0-r01-20261105`.
Its dry plan and full26-component regional catalog check passed. Exact submit
and observer arguments are recorded; neither was executed. No launch ledger,
observer, allocation or training run exists for this baseline campaign yet.

### 2026-11-05 18:52 UTC — complete full-size probe qualification

The full-size probe qualification passed on the validated alpha0.25 pilot
checkpoint. It restored all57 finite model, optimizer, counter and PRNG arrays
with `allow_partial=False`, checked counter7,630, and produced all27 finite
paired-loss rows and18 finite production-polar versus SVD comparisons. The
loss probes cover nine alpha/ridge settings at three common step sizes. The
decomposition comparisons cover the same nine settings on one attention matrix
of shape1,024×1,024 and one MLP matrix of shape4,096×1,024. Paired delta means, standard deviations and standard errors
were independently recomputed from the saved two-batch losses. See
[completion evidence and outputs](fullsize-probe-qualification-completed.json).

The original single task attempt succeeded with exit0 and matched its cached
startup UID and worker. The original observer recorded canonical success at
18:47 and exited naturally. The native ONLINE run finished with matching source,
request, checkpoint, task and attempt identity, and reported27 loss rows and18
decomposition rows. Its finished host logging record was reused from startup.
A29,653-byte output archive passed generation-bound readback and SHA-256 checks;
only its bounded reports were downloaded. The initial config comparison exposed
two known W&B omissions of empty mappings. Restoring only those mappings gave
exact request equality; the initial failure log is retained.

The earlier absence of intermediate SVD output is resolved by the complete
terminal report. No specific cause of the long runtime was established. Reuse
this completed evidence and retire its polling; do not restart its observer or
repeat task, tracking or archive queries. This qualification used pilot cell16
on one host with four TPU devices. It does not replace the scientific probes
on the three prescribed alpha0.25 confirmation checkpoints, establish training
resume or data-position replay, or change hyperparameter selection. Recursive
expansions and full confirmations remain unfinished.

### 2026-11-05 20:54 UTC — validate both alpha0.125 expansions

Both R01 lower-LR expansions passed all pilot completion gates. At LR0.00125,
final C4 loss was4.224646091461182 for ridge0.0001 and4.2394490242004395
for ridge0.01. Each has7,630 finite rows for all five training metrics, stored
counter7,630, the exact57-array checkpoint metadata, matching source/data/config,
a finished native ONLINE run, and four original attempt0 task successes.
All eight finished host logging records were reused from startup caches.
The sole observer recorded both canonical successes at20:49 UTC and exited
naturally; do not restart it or query these completed identities again. The
20:50 progress helper stopped before any API call because that owner had exited;
its raw error is retained. See [validated results](subset-expansion-r01-completed.json).

Replaying all30 initial results plus these two expansions closes both alpha0.125
ridge groups: LR0.0025 is now an interior winner for each. Their winning C4 losses
are4.168757438659668 (ridge0.0001) and4.160338401794434 (ridge0.01), so
ridge0.01 is the better alpha0.125 setting in this seed0 pilot comparison.
No additional alpha0.125 expansion is required. The other four already accepted
expansions remain running with finite progress; no new eligible cell was found.
The global selection lock and full confirmations remain pending. These checkpoint
checks establish metadata and stored-counter consistency, not full byte restore.

### 2026-11-05 23:03 UTC — validate alpha0.25 expansion and settle its settings

R02 passed all pilot completion gates. At alpha0.25, ridge0.0001 and LR0.0005,
final C4 loss was4.404330730438232. Its7,630 rows are finite for all five
training metrics; stored counter is7,630 and all57 checkpoint array metadata
entries match. Source, data and configuration match the accepted identity.
The native ONLINE run finished, and all four original attempt0 tasks succeeded
without failures or preemptions. Four finished host logging records were reused
from startup caches. The sole observer recorded canonical success at22:57 UTC
and exited naturally. Reuse the [validated result](subset-expansion-r02-completed.json);
do not restart that observer or query these completed identities again.

Replaying33 completed pilots closes both alpha0.25 ridge groups. Ridge0.0001
has an interior winner at LR0.001 with C4 loss4.276551723480225; ridge0.01
has an interior winner at LR0.002 with C4 loss4.211307048797607. Thus
ridge0.01 and LR0.002 are the selected alpha0.25 seed0 pilot settings.
No further alpha0.25 expansion is eligible. The three remaining candidates are
already accepted R03-R05 jobs and continue with finite progress. The global
selection lock, full confirmations and prescribed scientific probes remain
pending. The alpha0 long run remains prepared and unsubmitted. These checkpoint
checks establish metadata and stored-counter consistency, not full byte restore.

### 2026-11-06 01:37 UTC — validate alpha1/3 expansion and settle its settings

R03 passed all pilot completion gates. At alpha1/3, ridge0.01 and LR0.004,
final C4 loss was 4.3078436851501465. All 7,630 rows are finite for each of
five training metrics; the stored counter is 7,630 and all 57 checkpoint array
metadata entries match. Source, data and configuration match the accepted
identity. The native ONLINE run finished, and all four original attempt0 tasks
succeeded without failures or preemptions. Four finished host logging records
were reused from startup caches. The sole observer recorded canonical success
at 01:34 UTC and exited naturally. Reuse the
[validated result](subset-expansion-r03-completed.json); do not restart the
observer or query these completed identities again.

Replaying 34 completed pilots closes both alpha1/3 ridge groups. Ridge0.0001
has an interior winner at LR0.001 with C4 loss 4.415025234222412; ridge0.01
has an interior winner at LR0.002 with C4 loss 4.260484218597412.
Thus ridge0.01 and LR0.002 are the selected alpha1/3 seed0 pilot settings.
No further alpha1/3 expansion is eligible. Only the two accepted alpha0.5
expansions R04 and R05 remain. The global selection lock, full confirmations
and prescribed scientific probes remain pending. The alpha0 long run remains
prepared and unsubmitted. These checkpoint checks establish metadata and
stored-counter consistency, not full byte restore.

### 2026-11-06 01:42 UTC — launch the settled alpha0 long run

Following the user's direction to start the prepared long run, the exact
alpha0 seed0 baseline was accepted once as
`ema-gram-20-alpha0-seed0-r01-20261105-00` on v4-8 in us-central2.
It trains from scratch for 37,335 updates at LR0.003535533905932738,
with the unchanged 447M model, context4096, batch8 and fixed subset.
The predeclared C4 target remains4.0; training finishes the full schedule
regardless of an earlier threshold crossing. No wall-clock limit was added,
and default preemption and retry policy remain intact.

The submitter repeated the full26-component catalog gate, verified the clean
published source f99298782188e8f1c4e1ad8a1b2a507693d653e8 and unchanged logging
qualification, and persisted its permanent guard before submission. Submission
returned0. Sole observer7631/PID55651 owns canonical polling and follower55837.
Acceptance is not startup or scientific completion: task allocation, execution
host ONLINE identity and native training progress still require observation.
The [accepted receipt](alpha0-long-run-accepted.json) freezes original source,
input paths and ledger for exact-once adoption by final confirmations. Preserve
the prepared bundle and dedicated execution worktree. Do not submit a duplicate
baseline or start another observer. Global selection and nonbaseline confirmation
gates remain unchanged while R04/R05 finish.

### 2026-11-06 01:45 UTC — verify alpha0 long-run startup and finite progress

The accepted baseline is canonically RUNNING with one task and zero failures
or preemptions. Its original attempt0 UID9fb79df70f232c10 and assigned worker
join to the execution-host ONLINE check8770c3614a7e, which finished successfully.
The exact source, worker and bundle hashes, seed0, LR0.003535533905932738,
batch8 and37,335-update native configuration passed startup validation.
The [native ONLINE run](https://wandb.ai/marin-community/marin/runs/ema-gram-confirmation-3e18-a0-seed0-994a222dc7b9-2026.11.05.14-0c1aa5489b5c) is running and reached label38 with finite
training loss11.99289321899414. This establishes actual training progress,
not merely acceptance. See [startup evidence](alpha0-long-run-started.json).

Reuse the startup task and finished host caches. Sole observer7631/PID55651
continues canonical polling; the native progress helper queries only this
unfinished run. No duplicate submission or observer is needed. Full completion,
common-target accounting and final confirmation adoption remain pending.

### 2026-11-06 02:57 UTC — validate R04 and continue its lower-boundary search

R04 (alpha0.5, ridge0.0001, LR0.000125) finished with C4 loss
4.876398086547852. All 7,630 rows of five training metrics are finite,
the stored counter is7,630, and all57 checkpoint array metadata entries match.
Exact source, data, configuration and native ONLINE finish passed. All four
original attempt0 tasks succeeded with zero failures or preemptions; four
finished host records were reused from startup caches. The original observer
recorded canonical success at02:55 UTC and exited naturally. Reuse the
[completion evidence](subset-expansion-r04-completed.json); no completed
job, task, native run, host run or follower should be queried again.

The replay of35 completed pilots leaves this group's best loss at the lower
LR boundary, so the frozen factor-two rule requires LR0.0000625 with the same
alpha, ridge and seed0. R05 is still awaiting terminal validation. No global
selection lock is ready. The alpha0 long run is active and reached label8544
with finite training loss3.9630069732666016; this is a training loss, not a
held-out target crossing. Local OAuth endpoint DNS failures delayed observer
queries; original observers retained their scheduled retries, and R04 recovered
without a worker restart. Raw transport errors and the helper's terminal-PID
race are retained. Checkpoint metadata verification is not full byte restore.

R06 preparation freezes35 validated predecessors and six accepted expansion
bindings at artifact version2026.11.06.01. The planning snapshot retains all
validator fields and binds each original full completion report by path and
SHA256; raw evidence maps remain in those reports. This avoids duplicating
large evidence indexes in the new snapshot. Numerical and launch gates are
unchanged. The only new candidate is alpha0.5, ridge0.0001, LR0.0000625.

### 2026-11-06 03:01 UTC — accept recursive R06 expansion

R06 was accepted once as `ema-gram-20-expand-seed0-r06-20261106-00`, with alpha0.5,
ridge0.0001, LR0.0000625 and seed0. The unchanged7630-update v4-32
worker passed the full26-component catalog gate; source `08a1e57a552f42344ec57ddd6977fa4c255c7c31`
is frozen in a dedicated clean published worktree. The shared permanent registry
retains its launch guard. Sole observer94605/PID24368 owns polling
and its original follower. [Acceptance evidence](subset-expansion-r06-launched.json)
is not startup or scientific completion. Preserve default preemption/retries and
the complete update schedule, with no wall-clock limit.

### 2026-11-06 03:04 UTC — validate R06 startup identities

R06 is canonically RUNNING on four assigned workers, with zero failures or
preemptions. All four original attempt0 identities join to finished ONLINE
host logging checks; exact source, worker and bundle hashes, native configuration,
seed and learning rate passed [startup validation](subset-expansion-r06-started.json).
The native ONLINE run is `ema-gram-pilot-3e18-a0.5-seed0-9fd59fb65fba-2026.11.06.01-6adfdf90b19e`. It is compiling its first
step; the startup snapshot has no training step or loss yet. Finite progress
remains pending. Reuse these startup and finished host caches; the native-only
helper06 and local progress recorder now cover this campaign.

### 2026-11-06 03:12 UTC — validate R05 and settle its ridge-group learning rate

R05 (alpha0.5, ridge0.01, LR0.002) passed all completion gates with final
C4 loss4.459773540496826. All7,630 rows are finite for each of five
training metrics, stored counter7,630 and57 checkpoint array metadata entries
match, and exact source/data/configuration and native ONLINE finish passed.
All four original attempt0 tasks succeeded with zero failures or preemptions;
four finished host caches were reused. The sole observer recorded canonical
success at03:08 UTC and exited naturally. Reuse the
[completion evidence](subset-expansion-r05-completed.json); never poll or
restart completed R05 identities. Metadata checks do not establish byte restore.

Replaying36 completed pilots places this ridge group's winner at the interior
LR0.001, C4=4.3975324630737305. No further upper-LR expansion is
eligible. Only accepted R06 (alpha0.5, ridge0.0001, LR0.0000625) remains in
the recursive search, so global selection and nonbaseline confirmations remain
pending. R06 has now produced finite native progress at label7 with training
loss12.217096328735352. Alpha0 long-run label10233 has finite training
loss3.8518803119659424; these training losses do not establish held-out
C4 target crossing. Both original observers and launch guards remain intact.

### 2026-11-06 07:06 UTC — validate the complete alpha0 long run

The early alpha0 seed0 baseline completed all 37,335 updates from scratch with
final held-out C4 loss **3.792296886444092**. Every label 0–37,334 has
five finite training metrics; the permanent final checkpoint stores counter
37,335 with the expected 57 array metadata entries. Exact source, fixed data,
configuration and ONLINE finish passed. The original attempt0 succeeded with
zero failures or preemptions, and the finished startup host cache was reused.
The original observer recorded success at 07:00 UTC and exited naturally.
The monitoring helper's live-PID check then failed before any API call; its raw
log is retained, and this is not a worker failure.

The first recorded C4 below the predeclared 4.0 target is **3.9549849033355713**
at label 33,999: 34,000 updates and 1,114,112,000 tokens. Its recorded native
runtime is 17,173.085236197 seconds (about 4.77 hours), including compilation
and evaluation and excluding queue time. This is an observed evaluation point,
not an interpolated crossing. All 38 evaluation points and their timestamps are
in [the completion report](alpha0-long-run-completed.json). Training continued
to the full schedule; final evaluation runtime is 18,964.008649538 seconds.

The shared terminal validator now checks the explicitly requested schedule
against the frozen config. Offline fixtures passed for the 37,335-step baseline
and the existing four-host 7,630-step pilot, with ten rejection cases and cached
completion reuse without API access. W&B pagination's identical duplicates and
37 missing boundary labels were preserved and repaired with exact-label queries.
Metadata validation does not establish full checkpoint byte restore.

Never query completed baseline identities again or restart its observer.
Keep the original accepted receipt for exact-once adoption after global selection.
R06 remains active (latest label 3,483); its outcome and any further boundary
expansion still gate the selection lock and remaining confirmations. The required
scientific probes and full scientific conclusions remain pending.

### 2026-11-06 10:31 UTC — authorize immediate parallel long runs

The user explicitly directed immediate startup of all remaining planned long
runs. This supersedes the earlier requirement to wait for every recursive pilot
before nonbaseline submission. Freeze the current pilot-only choices: alpha0.125,
ridge0.01, LR0.0025; alpha0.25, ridge0.01, LR0.002; alpha0.5, ridge0.01, LR0.001.
Here alpha is the EMA-Gram exponent, ridge is its regularization parameter,
and LR is learning rate. All three use seed0 and start from scratch for37,335 updates, with native W&B ONLINE
tracking, batch8 sequences/context4096 tokens and the existing447M model,
fixed-subset data and validated optimizer, default scheduler
preemption/retry, and no wall-clock limit. Each requests v4-32 in us-central2.

The alpha0.5 choice is provisional because ridge0.0001 still has active
recursive pilot R06. R06 continues under its original owner. Long-run outcomes
are excluded from pilot selection; if the launched configuration differs from the eventual pilot winner,
preserve its long-run result as provisional and reconcile the finally selected configuration separately. Alpha0.125
has lower held-out C4 loss than alpha1/3 among completed pilots.
The common held-out C4-en loss target remains4.0,
with full schedules continuing after crossing. Alpha0.25 retains reference
checkpoint counters3733/18665/33597 optimizer updates.

The [frozen amendment](parallel-long-runs-r01-amendment.json),
[36-pilot snapshot](parallel-long-runs-r01-completions.json), and
[three configurations](parallel-long-runs-r01-bundle.json) record this change.
The completed alpha0 baseline stays outside the new submission and observer
set. At final selection, match the original accepted baseline receipt and this
campaign ledger by exact configuration, and count matching jobs in the final
comparison without submitting or observing them again. No global selection lock is claimed.

The existing confirmation worker, submitter and observer now accept this explicit
amendment as an alternative to the original global-lock path. Config decoding,
full completion validation, source/input binding, permanent pre-RPC guards and
regional cache checks remain required. Preparation is complete; acceptance and
startup will be recorded after execution.

### 2026-11-06 10:40 UTC — three parallel long runs accepted

All three authorized seed0 long runs were accepted once after the full26-component regional catalog passed. The execution source is `923d79510475df3ed0fef53d40b3290ae4c90fce`; campaign `ema-gram-20-confirm-seed0-r01-20261106` maps jobs00/01/02 to alpha0.125/0.25/0.5. Each requests v4-32 for37,335 updates from scratch. The permanent ledger records three successful submissions. Sole observer99712/PID96635 owns all three job snapshots and followers. Startup and finite numerical progress remain to be verified. The completed alpha0 baseline is excluded. See [accepted receipt](parallel-long-runs-r01-accepted.json).

Before any final confirmation submission, reconcile this accepted campaign by exact configuration and source; do not call the ordinary submit path to duplicate these jobs. Alpha0.5 remains provisional until pilot-only selection finishes. R06 continues under its existing owner.

At10:42 UTC, the sole observer saved all three canonical states as PENDING, each with4 pending tasks and zero failures/preemptions. The scheduler reports insufficient capacity for a four-host coscheduled group (CPU/memory) and autoscaler quota-pool tier blocking. This is a resource queue, not a remaining pilot-selection gate. Preserve accepted identities and default scheduling policy. [Saved queue evidence](parallel-long-runs-r01-queued.json); startup collector is prepared but has not queried tasks or native runs.

### 2026-11-06 11:39 UTC — complete recursive pilot selection

R06 (alpha0.5, ridge0.0001, LR0.0000625) completed all 7,630 updates
with final held-out C4 loss **5.104428291320801**. All five training
metrics have 7,630 finite rows, the stored counter is 7,630, and 57 checkpoint
array metadata entries match. Exact source/data/configuration, finished ONLINE
tracking and all four original attempt0 successes passed. Four finished host
logging records were reused. The sole observer recorded canonical success and
exited naturally; never poll or restart these completed identities. See
[completion evidence](subset-expansion-r06-completed.json). Metadata checks do
not establish full checkpoint byte restore or training resume.

Replaying all 37 validated pilots leaves every exponent/ridge group with an
interior learning-rate winner and no further expansion candidate. For alpha0.5,
ridge0.0001 settles at LR0.000125, C4=4.876398086547852; ridge0.01
with LR0.001 wins across ridges at C4=4.3975324630737305. This matches
the already accepted alpha0.5 long run. No long-run outcome entered selection.
The global selection record and exact-source accounting of accepted long runs
remain to be published; they must not cause duplicate submission.

The existing long-run observer now records alpha0.5 RUNNING on four tasks,
with zero failures/preemptions. Its original follower contains four host logging
URLs and a native ONLINE URL. Alpha0.125 and alpha0.25 remain resource-queued.
Full startup identity validation remains pending; the campaign collector has
not executed. The three prescribed alpha0.25 checkpoint probes remain required.

### 2026-11-06 11:43 UTC — lock pilot selection and reconcile accepted long runs

The [final selection lock](subset-final-selection-lock.json) replays all 37
validated pilots with the unchanged C4 target4.0. Every ridge group has an
interior LR winner. Selection reads only pilot completion reports; the already
completed baseline's long-run loss is excluded. The original global rule selects
alpha0, alpha0.25 and alpha0.5 because alpha0 has the best pilot loss overall.
The explicit parallel-start amendment also selects alpha0.125, the best other
exponent versus alpha1/3. Both rules are recorded without changing the original
core selection function.

[Final accounting](subset-final-confirmation-accounting.json) regenerates each
selected configuration at its original accepted artifact version, reads frozen
inputs from the original Git source, and verifies submission commands, input
hashes, target and permanent acceptance guards. All four accepted jobs match:
alpha0 LR0.003535533905932738; alpha0.125 ridge0.01 LR0.0025; alpha0.25
ridge0.01 LR0.002; alpha0.5 ridge0.01 LR0.001. Alpha0.5 is no longer
provisional. No new submission or observer is required. Existing observers and
completed caches retain ownership; acceptance accounting is separate from full
scientific completion and startup verification.

The ordinary global submit/observe accounting path now rejects a workspace
containing the parallel accepted receipt before any remote call. Local replay
and six rejection cases passed, including altered commands, ambiguous acceptance,
changed source/input, incomplete pilot history and duplicate global accounting.
The earlier baseline-adoption integration fixture passed with fake Iris; its
pre-parallel fixture excludes the later receipt. Initial local schema/import
errors were corrected and retained in scratch logs. See
[validation evidence](subset-final-accounting-preflight.json). No training
numerics or immutable execution worktree changed.

Long-run startup validation, complete schedules and the three prescribed
alpha0.25 checkpoint probes remain pending. Use the final accounting's original
job/config/source identities for result collection. Never submit the core lock's
configurations again.

### 2026-11-06 13:55 UTC — reference long run allocated

The original observer now records alpha0.25 RUNNING on four tasks;
alpha0.5 also has four running tasks. The scheduler reports alpha0.125 pending
four-host CPU/memory capacity. All three
have zero recorded failures and preemptions. Alpha0.5's original follower has
finite training progress through label2020 (loss5.092146873474121); this is a
training loss, not a held-out target result. See
[saved observer evidence](parallel-long-runs-r01-two-running.json).

Canonical allocation does not establish native startup validation. The alpha0.25
follower previously exited with a Bad Gateway transport error while queued;
the follower exit does not establish a worker failure and it will not be
restarted. Alpha0.25 lacks cached evidence joining its assigned tasks to host
logging checks and its native training run. Collect that evidence with exact
source and attempt identity checks before declaring startup validated. The startup collector remains unexecuted, with its existing
all-three-running gate and one-time guards preserved. Keep the single observer session99712/PID96635
and [accepted identities](subset-final-confirmation-accounting.json). Full
schedules and the three alpha0.25 checkpoint probes prescribed in the
[execution plan](single-seed-execution-plan.md) remain required.

### 2026-11-06 17:57 UTC — all three long-run startups validated

The Iris observer's saved 17:54 UTC snapshots report RUNNING for all three
nonzero EMA-Gram exponent settings: alpha 0.125, 0.25 and 0.5. Each run has four
distributed training tasks, with zero recorded failures or preemptions.
Startup validation joined all 12 original attempt 0 task UIDs and assigned
workers to separate, finished ONLINE host logging checks. Each native W&B
training run matches its configuration in parallel-long-runs-r01-bundle.json
and Git source 923d79510475df3ed0fef53d40b3290ae4c90fce. See
[startup evidence](parallel-long-runs-r01-started.json).

The source-filtered W&B snapshot records finite training losses:

| Alpha | Zero-based training label | Training loss |
| --- | --- | --- |
| 0.125 | 7 | 12.237302780151367 |
| 0.25 | 3489 | 4.54071569442749 |
| 0.5 | 5680 | 4.19376277923584 |

Held-out C4 target 4.0 and full-schedule outcomes remain pending. Startup
validation establishes task identity, configuration, logging and finite
training progress; it does not establish scientific completion.

The all-three-running gate passed before startup collection. A single bounded,
non-following alpha 0.25 log snapshot recovered its missing W&B URLs; the
collector checked those against original task UIDs, host checks and native
configuration. Its 12 task reads and source-filtered ONLINE snapshot are cached
under scratch/20261106-parallel-r01-startup. Preserve the collection guards
there and reuse these caches. Both earlier Bad Gateway exits affected log
followers; the new scheduler and W&B evidence shows training continued.
No replacement follower or observer was created.

Observer session 99712/PID 96635 retains Iris status polling. Native progress
monitoring can now use the validated W&B run identities without repeating
finished host checks or startup task reads. Each run's required schedule remains
37,335 updates. The three alpha 0.25 checkpoint probes at completed-update
counters 3,733, 18,665 and 33,597 remain required: held-out norm-matched direction
and paired-loss comparisons, plus attention/MLP polar-versus-SVD comparisons.

### 2026-11-06 18:15 UTC — reference-probe configuration repair

The checkpoint probe decoded `jax_softmax_custom_jvp`,
`jax_threefry_partitionable` and the checkpoint-path list as strings. It also
omitted application of the JAX settings recorded in the training configuration.
The probe now restores those field types, rejects differences between the
re-encoded configuration and its input, and applies the JAX settings before
reading a checkpoint or constructing model arrays. An offline check against the
[accepted alpha 0.25 configuration](parallel-long-runs-r01-bundle.json) passed
the full configuration round trip and verified the JAX setter values. This
check executed no numerical kernels and read no checkpoint bytes. See
[repair validation](reference-probe-config-repair.json).

The [prior logging and SVD qualification](reference-probe-svd-preflight.json)
remains historical evidence for its original source; it does not qualify this
edited source for launch. Revalidate the affected qualification and audit
four-host restoration and whether each process can read its SVD result arrays.
The [three prescribed probes](single-seed-execution-plan.md) remain unlaunched.
The accepted training jobs and their source are unchanged.

The alpha 0.25 native run reached zero-based label 3737 at 18:12 UTC. Its first
prescribed permanent checkpoint, label 3732 (expected stored counter 3733),
now has generation-bound metadata and a valid 57-array manifest matching the
reference schemas. These checks establish metadata availability only; the
stored scalar counter and full checkpoint restoration remain unvalidated.

### 2026-11-06 18:24 UTC — scalar comparisons for sharded probe directions

The reference-checkpoint probe converted complete direction matrices to NumPy
arrays to calculate cosine similarity and relative error. JAX rejects this
conversion when a matrix spans devices outside the current process. The probe
now calculates these global reductions in JAX and transfers only scalar results
to the host. The production polar and SVD functions are unchanged; timings still
cover each synchronized direction calculation after its warm-up call.
No actual four-host probe has run. Numerical validation of this edit remains
pending; the CPU qualification now includes row- and column-sharded inputs,
rectangular matrices, zero references and replicated scalar-output checks.

The two ONLINE logging/SVD qualification runs at source
b33ed91cf84bb2730ed0fa1a6869e62c94ca1d80 passed before this edit. Their cached
results remain in scratch/20261106-reference-probe-requalification-r01; reuse
them without querying those finished runs again. They do not validate this
new edit. A read-only scalar read separately verified stored counter 3733 in
the first alpha0.25 reference checkpoint. Its cache is
scratch/20261106-reference-probe-preparation/counter-3733-metadata/counter.json.
Full checkpoint restoration and the three prescribed probes remain pending.

### 2026-11-06 18:27 UTC — sharded scalar comparison qualification passed

At source 843b2446b12652ea96a13b80758b0a8449c87e6f, a fresh W&B online logging
check and CPU SVD qualification run both finished successfully. The numerical
check passed 18 attention/MLP polar-versus-SVD comparisons and three
zero/rank-deficient cases. Eight additional row- and column-sharded rectangular
matrix cases on four CPU devices matched NumPy reference calculations within
the unchanged tolerances, with all diagnostic outputs replicated scalars.
These checks qualify the edited probe code's logging and small-array numerics.
Four CPU devices in one process do not validate four-host TPU execution or full
checkpoint restoration. [Qualification evidence](reference-probe-svd-preflight.json)
now records these results, previous qualification references and raw cache hashes.
All three prescribed scientific probes remain unlaunched.

### 2026-11-06 18:34 UTC — first prescribed scientific probe accepted

The first prescribed scientific probe was accepted by Iris at 18:32:55 UTC as
ema-gram-20-reference-seed0-r01-20261106-00. It targets the alpha0.25 seed0
long-run checkpoint at completed-update counter3733 (directory step-3732), using
the unchanged recorded training configuration and requested v4-32 topology with
four hosts. Source and request were published at
2fbdb4290a95ac7b7cad839d49569d258812bb5f before submission. All 26 regional
component catalogs matched the recorded object names, sizes and CRCs. Exact
config/source checks, stored counter and 57-array metadata checks passed. See [acceptance evidence](reference-probe-r01-accepted.json).

Submission returned exit0. The permanent counter3733 launch guard records this
accepted identity. On the local research workstation, sole observer session91651/
PID70306 and original follower70493 own observation. Their state lives in the
ema-gram-reference-seed0-r01-20261106 execution worktree. Do not create replacements
or resubmit this identity. The observer state captured in the acceptance record
had not yet stored its first scheduled canonical snapshot. The first scheduled check at 18:35 UTC subsequently recorded four pending tasks,
zero failures/preemptions, and insufficient memory for four-host coscheduling;
the autoscaler also reported quota-pool tier blocking. Acceptance does not
establish native startup, full checkpoint restoration or successful probe results.

The probe requests 27 direction/step-size settings, each evaluated on the same
eight held-out Wikipedia batches with paired baseline losses. All nine direction
settings use step sizes 1e-5, 3e-5 and 1e-4. Separately, it requests 18 attention/
MLP polar-versus-SVD matrix comparisons. It performs no optimizer update or
checkpoint write. The other prescribed probes
at counters18665 and33597 remain unlaunched. The three accepted long-training
runs continue under their original observer; completed campaigns remain cached.

The previous issue20 status comment approached GitHub's size limit. Its entire
body is preserved in [the exact JSON archive](issue20-comment-history-20261106-1827.json)
with its SHA256 and a [Markdown reading copy](issue20-comment-history-20261106-1827.md).
The current issue comment links to this published history and summarizes active work.

### 2026-11-06 19:06 UTC — first scientific probe startup validated

The counter3733 alpha0.25 scientific probe is canonically RUNNING with four
tasks and zero recorded failures or preemptions. The one-time startup collector
cached four original attempt0 task UIDs and four distinct assigned workers,
four finished ONLINE host logging checks, and four running native ONLINE probe
runs. Each run matched the published source files, exact resolved training
configuration and its requested mode. Native requests also matched checkpoint
step-3732/counter3733, eight held-out Wikipedia batches at offset0, and the
three common step sizes. See [startup evidence](reference-probe-r01-started.json).

The original task/attempt-tagged follower lines join each task to its host and
native W&B URLs. W&B configs do not themselves contain task UIDs; the wrapper's
generation-bound attempt archive must reconcile those exact UIDs at terminal
collection. Startup validation does not establish successful full restoration
or scientific results. All four native summaries are empty in the cached startup snapshot. Preserve the original observer91651/PID70306 and
follower70493. Never repeat startup task reads or finished host APIs absent an
attempt/failure change; reuse scratch/20261106-reference-r01-startup/validated.json.

The three nonzero long runs advanced at 19:05 UTC to log labels963,4553,6651
for alpha0.125,0.25,0.5, respectively, with finite training losses and no
investigation flags. Full 37,335-update outcomes, all three scientific probe
results and final conclusions remain required. The later counter18665/33597
probes are unlaunched.

### 2026-11-06 20:54 UTC — first scientific checkpoint probe completed and validated

The first prescribed alpha0.25 probe, at completed-update counter3733
(checkpoint step-3732), passed terminal validation. Iris recorded four original
attempt0 successes, exit0 and zero failures/preemptions, finishing at 20:44:34 UTC.
All four native ONLINE runs finished with probe/success=1. The wrapper's archived
task IDs, attempt UIDs, source2fbdb4290, configuration digest and checkpoint
matched the original startup joins. Four bounded task archives and all 24
permanent diagnostic objects (five artifacts plus a manifest per host) passed
generation-bound readback and SHA256 checks. Finished host logging records were
reused without API calls. See [completion evidence](reference-probe-r01-completed.json)
and [the collection and offline audit scripts](reference-probe-r01-terminal-audit.json).

Each process returned 27 finite direction/step-size loss rows over the same eight
held-out Wikipedia batches, plus 18 finite attention/MLP polar-versus-SVD
comparisons. Paired means, sample standard deviations and SEMs were independently
recomputed. Loss rows and direction diagnostics match exactly across all four
processes; timings and device memory remain process-specific. Eight corrupted
report cases were rejected by the offline validator. The four hosts replicate
the same batches and do not increase the scientific sample size from eight.

The frozen source loaded the full TrainerState with allow_partial=False and
verified restored counter3733 before these computations. This validates actual
four-host checkpoint restoration for the probe. It does not independently audit
finiteness of all 57 restored arrays, perform an optimizer update, replay the
training data position or establish training resume.

The held-out baseline mean was 4.535014629364014. Mean candidate-minus-baseline
loss changes ranged from -2.8014183e-5 to +6.8068504e-5; every negative mean was
smaller in magnitude than its paired SEM. These small changes do not establish
a preferred exponent at this checkpoint. Across the two selected matrices and
nine exponent/ridge settings, production polar versus SVD cosine ranged from
0.9489101 to 0.9994838, and relative direction error from 0.0335550 to 0.3197872.
Relative error is the Frobenius norm of the direction difference divided by
the SVD direction norm. These are
measured approximation differences, not a claim that production polar equals
SVD or that one checkpoint determines the final scientific conclusion.

The original probe observer91651/PID70306 exited naturally after canonical
success. Its original follower70493 had already exited on a log-transport
timeout; neither was replaced. A temporary local DNS interruption recovered on
the original observers' scheduled retries. Earlier read-only task0 health
samples showed CPU activity while report files were absent; they did not prove
scientific progress or identify the cause of the runtime. Complete outputs now
resolve the pending output-validation question. Preserve those raw observations.

Reuse scratch/20261106-reference-r01-completion/verified.json and the execution
worktree's completion cache. Never repoll this completed probe's tasks, native
runs, finished hosts or archives, or restart its observer/follower. The permanent
counter3733 launch guard remains consumed. At 20:51 UTC the three nonzero long
runs had advanced to labels2576/6170/8130 for alpha0.125/0.25/0.5, with finite
training losses and no investigation flags. Continue their original observer
99712/PID96635 and unfinished-native helper only. Counter18665 and33597 probes
remain unlaunched; all three full 37,335-update outcomes, the remaining two
probes and final conclusions are still required.


## 2026-11-07 11:29 UTC — second prescribed checkpoint probe accepted

Iris accepted the second prescribed scientific probe of the alpha0.25 confirmation
run, using checkpoint counter18665 (18,665 completed updates, zero-based step18664). The checkpoint's 57-array metadata
schema and stored scalar counter passed read-only validation. The full
26-component data catalog also passed. Numeric probe code, exact training
configuration, eight held-out Wikipedia batches and common step sizes remain the
same as the completed counter3733 probe recorded in reference-probe-r01-completed.json.
The probe perturbation sizes are 1e-5, 3e-5 and 1e-4. Actual four-host checkpoint restore and scientific
outputs remain unvalidated.

The immutable execution source is 6ec7fa98b842cdff03b46986d00e3ee4dc735615 on branch
codex/20-ema-gram-reference-seed0-r02-20261107. Campaign
ema-gram-20-reference-seed0-r02-20261107 has one job, suffix00, requesting v4-32
across four hosts with Iris default retry/preemption policy and no wall-clock
limit. The permanent counter18665 launch guard is consumed. Submission
session7005 exited0. Sole observer51423/PID27858 and its original follower28032
own observation; never resubmit this identity or replace either observer or
follower. See reference-probe-r02-request.json and reference-probe-r02-accepted.json
for the exact command, identities and hashes.

Collect startup evidence once when the saved canonical snapshot is RUNNING with
four tasks and all four original task-tagged host/native logging joins are ready.
Reuse completed first-probe caches without APIs. The third prescribed probe at
counter33597 remains unlaunched. Three nonzero confirmations still require their
total schedule of 37,335 updates each; at 11:24 UTC alpha0.125/0.25/0.5 had finite training losses at
labels16203/18697/20326. These training losses do not establish the held-out C4
target result. Final long-run outcomes, results of this second probe and the unlaunched third probe and scientific
conclusions remain required.


## 2026-11-07 11:44 UTC — second prescribed probe startup validated

The alpha0.25 confirmation checkpoint at counter18665 (18,665 completed training
updates, zero-based label18664) is now being probed by four running tasks,
indices0–3, representing the four hosts of one distributed probe.
One-time startup collection verified four original attempt0 task UIDs on four
distinct workers, four finished ONLINE logging checks and four running native
ONLINE probe records. Their logged source hashes, frozen training configuration,
checkpoint, expected counter, eight held-out Wikipedia batches and probe
perturbation sizes match reference-probe-r02-request.json. See
reference-probe-r02-started.json and
scratch/20261107-reference-r02-startup/validated.json for the evidence.

Startup validation does not establish successful checkpoint restore or scientific
results. All four native summaries were empty at collection and at the first
unfinished-native helper check at 11:44 UTC. W&B configuration does not contain
task UID; original task/attempt-tagged follower lines link each task to its
logging and native run URLs. Completion still requires exact wrapper attempt UID
and archive reconciliation, full TrainerState restore with allow_partial=False
and expected counter18665, all prescribed loss/direction outputs, finished ONLINE
records and canonical success.

Reuse sole observer session51423/PID27858 and follower PID28032; do not repeat
remote startup-task or finished-host logging-record reads absent attempt/failure
changes; local cached evidence remains available. Execution
source6ec7fa98b842cdff03b46986d00e3ee4dc735615 and the permanent launch guard remain
unchanged. The unfinished-native helper completed its first live check; run it
sequentially with the long-run helper once per wake. The first counter3733 probe
remains completed and uses cached evidence only. At 11:43 UTC, the three
confirmation runs alpha0.125/0.25/0.5 advanced to training labels16508/18973/20601
with finite losses and no investigation flags. All three still require total
schedules of 37,335 updates. The second probe uses the frozen checkpoint while alpha0.25 training continues.
The second probe's results, the third probe at alpha0.25 counter33597 and final scientific conclusions remain outstanding.


## 2026-11-07 11:52 UTC — second prescribed probe completed and validated

The alpha0.25 confirmation checkpoint at counter18665 (18,665 completed training
updates, zero-based checkpoint label18664) passed the prescribed four-host probe.
Iris recorded four original attempt0 successes, zero failures/preemptions, and
canonical success at 11:46:32 UTC. All four native W&B runs finished ONLINE.
Exact wrapper task/attempt UID, source, configuration, checkpoint and counter
match startup evidence. Four bounded task archives and 24 generation-bound
permanent objects were read back and verified. Each host returned 27 finite
paired-loss rows over the same eight held-out Wikipedia batches and 18 finite
production-polar-versus-SVD comparisons. Each loss row summarizes eight paired
batches for one exponent/ridge setting and perturbation size: nine settings
times three sizes gives 27 rows. Direction comparisons cover two selected
matrices (attention and MLP) times nine settings. Paired mean/std/SEM were independently
recomputed; eight corrupted-output cases were rejected by the offline validator.
Parsed numeric loss and direction diagnostic values are exactly equal across
four hosts; timings and device-memory records are process-specific. They are
replicas over the same batches, not independent scientific samples.

This validates actual full TrainerState restore with allow_partial=False and
restored counter18665 for the four-host probe. It does not establish an independent
57-array finiteness audit or training resume. See reference-probe-r02-completed.json
and reference-probe-r02-terminal-audit.json for outputs, identities, receipts and
validator source.

The baseline mean held-out Wikipedia loss was 3.835104763507843.
Candidate-minus-baseline mean loss changes ranged from -0.0002271533 to
+0.0004232228. The most negative row used diagnostic exponent alpha=1/3, ridge=0.01 and
perturbation size3e-5
and paired SEM0.0001049152. This checkpoint alone does not establish a preferred
exponent; the third prescribed checkpoint and complete confirmation results
remain required. Across attention/MLP matrices and nine exponent/ridge settings,
polar-versus-SVD cosine ranged from 0.8948413 to 0.9995929 and relative direction
error from 0.0295548 to 0.4707010. Relative error is the Frobenius norm of the
direction difference divided by the SVD direction norm.

The sole observer session51423/PID27858 and original follower PID28032 exited
naturally. Never query this completed probe's jobs/tasks/native records,
finished-host records or archives again, or resubmit/restart it. Reuse research
scratch/20261107-reference-r02-completion/verified.json and the execution
worktree completion cache. The consumed counter18665 guard and immutable source
6ec7fa98b842cdff03b46986d00e3ee4dc735615 remain unchanged. Continue the three long
runs under observer99712/PID96635; at 11:48 their zero-based training labels
were16589/19013/20669 for alpha0.125/0.25/0.5, with finite losses and continued
progress. Only the third prescribed alpha0.25 probe at counter33597 remains
unlaunched; all three nonzero full 37,335-update outcomes and final conclusions
are still required.


## 2026-11-08 05:03 UTC — third prescribed checkpoint probe accepted

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


## 2026-11-08 07:24 UTC — third prescribed checkpoint probe startup validated

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


## 2026-11-08 07:31 UTC — third prescribed probe completed and validated

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


## 2026-11-08 10:26 UTC: seed0 comparison and failed-run scientific adjudication

The [seed0 findings](single-seed-findings.md) report compares all four
confirmation records and all three prescribed checkpoint probes, with
[scalar provenance](single-seed-comparison.json) and reproducible loss/token,
paired-loss and elapsed-time figures. This analysis used local caches only.
Alpha0 and alpha0.125 remain the two validated completions. Alpha0.125's final
C4 is lower by0.008272171020507812 nats and its first target crossing uses
1,000 fewer updates (33,000 versus34,000). Its deployed native time to target
is7.469233209559757 times the baseline and device-hours are29.876932838239032
times. Baseline uses four v4 devices with data4/model1; nonzero runs use16
with data4/model4. These figures do not isolate matched-hardware optimizer
overhead or establish cross-seed reproducibility.

Failed alpha0.25 and alpha0.5 are explicitly admitted only as supplementary
recovered endpoint/observed-crossing evidence. They remain outside completed-run
claims; their failed/crashed gates are unchanged. Alpha0.25 retains112 missing
metric labels and cannot support full-trajectory finite-history certification.
Both failed final C4 values have no native runtime. No missing values were
imputed, and no execution failure was attributed to optimizer instability.
The probes show checkpoint-dependent local responses, with four hosts repeating
the same eight batches. They do not establish a universal training exponent.

The report concludes that seed0 gives a small loss/token benefit for the tuned
alpha0.125 setting at higher observed deployment cost. A reproducible best
exponent, matched profile attribution and a peak-memory comparison remain
unestablished. No horizon or998M follow-up is triggered. The bounded scientific
interpretation is published; research remains active for shutdown-cause
investigation and qualification of a future repair before new execution.
No runtime/source hotpatch, new job or repeated terminal remote read occurred.


## 2026-11-08 10:52 UTC: local shutdown repair candidate

The [shutdown repair design](shutdown-repair-design.md) records an isolated
prototype with 19 passing local checks: eight telemetry-receipt checks, five
rank-exchange checks and six finalizer checks. Fake sinks and an in-memory
coordinator exercise delayed drain, lost updates, failed receipts, missing
peers and coordinator/shutdown errors. Two checks use actual local Python
process exits to verify that an atexit failure produces a nonzero status.
No numerical kernels, remote experiment reads or cluster jobs were used.

The candidate exposes local drain evidence, exchanges rank receipts before
JAX shutdown, and preserves required-telemetry failures. It keeps auxiliary
Finelog diagnostics separate. The inspected complete_job API can mark
unfinished tasks successful and stop them; the candidate instead leaves
status aggregation to natural task exits. The deployed controller revision
was not established, and this design does not reclassify historical runs.

Asymmetric telemetry drain remains a mechanism to test, not a proven initiating
cause. The prototype is not installed in an execution path. Required W&B
receipt integration, original application-error handling, hook replacement,
missing-peer cleanup and real multi-host ONLINE qualification remain pending.
Frozen source, launch guards and completion/integrity gates are unchanged.
Research remains active for the authorized local integration work; no new
training or qualification job is authorized by this publication.


## 2026-11-08 11:33 UTC: isolated shutdown source integration

The [source integration report](shutdown-source-integration.md) publishes a
reviewable patch and source hashes for candidate commit
`f6ef3108962a06bd3b39db50970a4d4a434ad372`. Thirty-six portable local checks
pass with fake service/JAX boundaries; source lint and type checks pass.
Tracker reports now reach an explicit outer entrypoint that replaces the
default exit hook after final device-memory use. The candidate preserves
application errors, records local drain failures, and serializes producer
admission with shutdown. These source findings do not establish the historical
initiating cause.

The patch remains isolated and unqualified. No launch recipe selects it, no
accepted source changed, and no new job or terminal experiment read occurred.
Local success does not certify remote history. Producer quiescence, real
package initialization, ONLINE multi-host behavior, missing-peer/partial-init
cleanup and the deployed controller still require qualification. No new
training or qualification job is authorized. Scientific findings and historical
failed-run gates remain unchanged.


## 2026-11-08 11:38 UTC: candidate finalization-read race correction

The [correction record](shutdown-read-ack-correction.md) documents a race in
the isolated candidate: the coordinator rank could shut down after reading
finalization records while another rank was still fetching them. A four-thread
fake coordinator reproduces the failure on exact candidate f6ef3108962a06bd3b39db50970a4d4a434ad372.
Candidate 689fe209f8102a588ebac1bc59cf04289137a3ad adds a read-completion
rendezvous before shutdown. The same simulation and nine affected exit-hook
checks pass; 27 unchanged local checks were reused.

This corrects the undeployed candidate, not the historical experiment record.
Actual JAX coordinator lifetime, missing-peer cleanup, live online multi-host
tracking and deployed controller aggregation remain unqualified. No new job,
numerical kernel or remote experiment read occurred. The prior candidate patch
and its limitations remain archived; apply the correction after that patch.


## 2026-11-08 12:14 UTC: local shutdown lifecycle observation

The [observation record](shutdown-lifecycle-observation.md) archives isolated
candidate `2674b5d3975e5c1ecac05fb4310814c90ec736e1`. An optional observer
records exit-hook and protocol boundaries without replacing the JAX client.
Observer write failures preserve normal cleanup and force local nonzero exit;
application exit codes retain precedence. Registration failure leaves the old
handler intact. Seventeen hook checks (including four Python atexit subprocesses),
three persisted lifecycle-log checks and nine log-contract checks pass within
fake external-service boundaries. Source and archive lint pass.

No runtime job, numerical kernel, terminal experiment read or accepted-source
change occurred. Real ONLINE multi-host behavior, deployed controller aggregation,
missing-peer cleanup and accelerator retention remain unqualified. The concrete
non-training runtime entrypoint and execution scope remain local preparation
work; no new job is authorized. Scientific findings and historical classifications
remain unchanged.


## 2026-11-08 13:11 UTC: prepared CPU shutdown diagnostic

The [CPU diagnostic preparation](shutdown-cpu-entrypoint.md) archives a concrete
non-training worker CLI, source-bound manifest and CPU deployment mapping for
candidate `2674b5d3975e5c1ecac05fb4310814c90ec736e1`. The default CLI performs
read-only preflight. An explicit execution path exists but requires new job
authorization. The proposed first stage has separate clean and injected-error
scenarios with four CPU tasks and four scalar ONLINE records per rank.

Twenty-seven distinct local checks have scoped passing evidence (12 entrypoint,
11 manifest, four bootstrap), with unchanged checks reused and two portability
checks also recorded. Review corrected the diagnostic's all-rank W&B failure
policy; four affected/new controls pass. Source/archive lint passes. These are
fake-service and source checks, not a fresh full runtime suite.

No job, real runtime service, numerical kernel, terminal experiment read or
accepted-source change occurred. CPU image/lock, tracking entity, region/placement,
deployed controller and independent runtime evidence remain unresolved. The
300-second per-rendezvous health budget is an unmeasured proposal, not a scheduler
or wrapper limit. Accelerator retention remains unresolved for future GPU/TPU
qualification. Scientific findings and historical classifications are unchanged.


## 2026-11-08 14:01 UTC: CPU source packaging and evidence preparation

The [packaging record](shutdown-runtime-packaging.md) archives a preparer for a
self-contained candidate Git checkout, a publication-bound diagnostic harness
and a cached dependency split for a proposed Linux x86_64/CPython 3.12.12 image.
The new local source copy passes read-only eight-file preflight with
`launch_ready=false`. The frozen lock export selects 160 registry requirements
and eight workspace packages; build tools, wheel availability and the base image
remain unresolved. Ten distinct local checks have scoped passing evidence
(seven splitter, three Git fixtures), with unchanged checks reused. Source and
archive lint pass; the new context preflight is recorded separately.

Source review also establishes an evidence limit: the inspected endpoint-list
API lacks attempt, registration/lease-time and deletion-reason fields. A final
absent endpoint cannot distinguish application unregister from expiry or other
cleanup. Independent controller/coordinator evidence acquisition must be arranged
before a future authorized attempt. No image build, package installation, real
runtime import/service, numerical kernel, job or terminal experiment query
occurred. Candidate and frozen training source, scientific findings and failed-run
classifications remain unchanged. Runtime execution and accelerator retention
remain unqualified.


## 2026-11-08 17:12 UTC: build-tool inputs and Iris packaging corrections

The [build-input record](shutdown-build-inputs.md) pins and caches eight build-tool
wheels, including Hatchling's extra editable dependency. Their hashes, Python
3.12.12 requirements and declared dependency closure pass a metadata-only audit.
The resolver's available-interpreter warning is retained; no real build ran.

Source review found that the Iris hook tests the target name instead of the
editable version. Six method-AST/fake controls reproduce current stamping and
check a proposed one-line guard correction; the patch is archived but not applied.
Read-only local Git queries also show that the prepared shallow copy reports an
Iris date of 2026-11-08 while full history at the same candidate yields 2026-10-29.
The earlier source-byte preflight did not check this derived identity. These are
packaging findings, with no claim about the historical shutdown cause.

Source/archive lint and patch applicability checks pass. Candidate and frozen
execution source remain unchanged. No package installation, image build, real
runtime service, numerical kernel, job or terminal experiment query occurred.
Runtime wheels, base image/OS/installer, integration of the guard/date policy and
independent controller/coordinator evidence remain unresolved. No new execution
is authorized; scientific classifications and original success gates are unchanged.

## 2026-11-08 17:25 UTC: answer the scientific questions; no longrun reruns

The user directed that the failed longruns must not be rerun. Active work now
answers the hypothesis and requested readout in Marin #10784 from the existing
results; shutdown repair and runtime-packaging work are deferred. No new
experiment is authorized by the list of remaining evidence needs.

[Question-by-question answers](ali-question-answers.md) retain alpha0 and 1/8
as validated completions and quarter/half power as supplementary recovered
measurements. The quarter endpoint is worse than alpha0 and better than half
power in this seed0 record. No full-trajectory or cross-seed claim is added.
The original scientific synthesis and plots are reused without regeneration.

New scalar analysis directly pairs the cached norm-matched quarter-power loss
with zero/half-power losses on the same eight batches. Quarter power is lower
than alpha0 in 4/6, 4/6 and 5/6 settings across the three checkpoints, and lower
than half power in 2/6 at each checkpoint. All 36 contrasts and batch-level
uncertainty are retained; settings/hosts are not independent replicates and
counts are not significance tests. Batch alignment and arithmetic checks pass.

The answer marks fresh-batch directional tests, production-versus-SVD loss
controls, paired new seeds and matched-hardware cost/memory evidence as
conditional gaps. Horizon/998M work remains deferred. These are not a launch
queue; no failed-run replacements, new jobs, package/runtime changes or
terminal experiment reads occurred. The alpha1/4 missing 112 labels and both
failed execution statuses remain explicit.


## 2026-11-08 17:55 UTC: minimal-compute supplements authorized

The user requested filling the remaining evidence gaps with minimal compute and
explicitly excluded cross-seed work. Failed longruns remain closed. The new
[supplement request](supplemental-probe-request.json) combines all three saved
reference checkpoints in one v4-32 job so compilation and input loading can be
reused. It evaluates fresh held-out Wikipedia batches64–79 (old scientific
probes used0–7; infrastructure qualification used16–17), alpha0/0.25/0.5,
pilot-selected ridge0.01 and the original three step sizes. Exact-SVD full-model
loss controls are restricted to the prespecified middle checkpoint18665.
There are30 loss rows per rank,16 paired batches each; hosts are replicas.

At the middle state, warm forward/reversed-order measurements compare matrix
direction kernels on equal hardware. These exclude gradient calculation, Gram
updates and full training; cumulative device-memory snapshots are not a
comparative training peak. Recorded logical matrix-state bytes quantify storage.
No source-matched full-training profiles were found in the cached profile
inventory; older November4 profiles are excluded from this comparison.

A new tiny NumPy oracle must pass under ONLINE tracking before submission.
The prepared worker also runs a non-training ONLINE check on each execution
host. Source/config are published before execution; attempts and W&B identities
are fresh. Prior reference source and shutdown candidate stay unchanged.
One job only; no scheduler/wrapper wall-clock limit, default preemption/retries,
regional storage and no cross-region checkpoint movement. TPU execution uses
existing Iris fleet policy; no GPU allocation is requested or released.
No larger-model/horizon/cross-seed campaign is authorized.


### 2026-11-08: isolate supplemental tracking from shared-project saturation

The local nonnumerical ONLINE check initialized in project `marin`, but its
file stream repeatedly returned429, explicitly identifying the per-project
request limit. The local check was interrupted before kernels or TPU submission;
its run `ema-gram-supplement-logging-check-ce017eb01b21` is not a successful check.
The new supplement uses project `ema-gram-20-supplements` in the same entity.
Its ONLINE check must pass independently; no offline fallback or old-run rewrite.


### 2026-11-08 17:58 UTC: minimal supplement accepted once

One four-task v4-32 supplemental job was accepted from source
`684236952eb4adce89d3d8a3aa008e794be93222`. The local ONLINE logging check and
new numerical oracle both finished online;14 small-model matrices match the
NumPy SVD oracle, the production path matches the prior probe, nonmatrices and
optimizer state are unchanged, and paired summary arithmetic passes.
[Launch and check evidence](supplemental-probe-accepted.json).

The live owner follows only this new job; all historical terminal reads remain
cached. Acceptance does not establish startup or scientific completion.


### 2026-11-08 18:06 UTC: fit the unstarted probe to ready host memory

The new r01 supplement was still pending on all four tasks, with empty worker
IDs, current_attempt_id=-1 and no attempts. Its256GiB host-memory request
prevented use of a ready v4-32 group whose limiting host had240.5GiB available.
The unstarted request was withdrawn and an r02 request prepared with224GiB per
host; TPU count, CPU, source kernels, data, checkpoints and scientific settings
are unchanged. No scientific worker had run and no GPU allocation was involved.
The earlier local ONLINE/numerical evidence is reused for identical source
hashes; every new execution host still runs its own logging check.

[Cached checkpoint storage](supplemental-logical-storage.json) shows the same
7,350,834,212 logical array bytes (about6.85GiB) in alpha0 and alpha0.25. Both
include2,722,103,296 bytes of left/right Gram state. This implementation retains
and updates Grams for alpha0 as well; it is not a memory-minimal generic Muon
baseline. Logical array storage does not establish peak host/device allocation.


### 2026-11-08 18:16 UTC: supplement data range corrected before loss measurements

The r02 supplement was accepted from `d21c0d91efe406185600de0c32c6a8b764770ef5`.
All four logging checks finished ONLINE; all four probe runs failed with
`Insufficient fresh held-out batches` before checkpoint restoration or loss
measurements. Canonical FAILED and four original task/attempt/source/native
joins, along with generation-verified partial artifacts, are cached in
`scratch/20261108-supplement-r02`. [Failure evidence](supplemental-probe-r02-failed.json).

The requested range64–79 was invalid: bounded offsets metadata establishes
580,527 tokens,141 context4096 sequences and17 complete batches of8. This was
an experimental setup error, not a scientific result or shutdown failure.
The replacement [r03 request](supplemental-probe-r03-request.json) fixes the range
to8–15: eight paired batches, disjoint from original probe0–7 and infrastructure
probe16–17. No losses from the failed request informed this selection. These are
fresh for the checkpoint probes, not a new untouched training-validation corpus.

The loader now checks complete-batch capacity before fetching and seeks directly
to the selected batch. Five scoped fake-loader regressions pass; AST comparison
shows all five numerical helpers unchanged, reusing the previously finished
ONLINE numerical oracle. [Correction evidence](supplemental-data-correction.json).
Three checkpoints, one selected ridge, three step sizes, and the middle-checkpoint
SVD loss control are retained, with half the originally proposed evaluation work.
No cross-seed, failed-longrun rerun, full-training timing claim or shutdown repair
candidate deployment is included.


### 2026-11-08 18:18 UTC: corrected eight-batch supplement accepted

Iris accepted r03 from `0019824a13ee93eeedd89652793428f8d7c85c3f`.
[Exact request and command](supplemental-probe-r03-accepted.json) bind the
corrected data range. Follow only this fresh job through evidence collection;
acceptance is not scientific completion. r01 was withdrawn before any attempt;
r02 failed during data selection and produced no loss measurements.


### 2026-11-08 19:07 UTC: recover two stages; isolate the remaining supplement

The r03 job failed after44.02 minutes. The controller records container-memory
OOM killing task0's numerical subprocess; siblings lost the JAX coordinator.
This occurred after21 loss rows and before the bulk tuple benchmark returned.
The all-bank compiled benchmark was a preparation mistake; it is removed from
the follow-up. This does not establish a training-memory limit or an optimizer
instability. There were no preemptions.

[Recovered partial evidence](supplemental-probe-r03-partial.json) binds task0's
small archive and permanent objects to hashes/generations. Early and middle
checkpoint rows and the middle SVD control are finite; only one full replica
artifact is recovered, and the native run still reported running on its one-time
terminal check. Keep FAILED/unqualified status; no completed-run claim or native
state rewrite. The other terminal attempt responses have no archive descriptor.
The interrupted log follower was reconnected once from its last timestamp;
its transport timeout did not establish a worker failure.

The [r04 request](supplemental-probe-r04-request.json) runs only the missing late
checkpoint and nine losses. It reuses the already warmed norm-matched direction
path for two forward/reverse-order timing repetitions, serializing banks as in
the working probes. It does not compile a tuple of all matrix-bank outputs.
[Source/fake checks](supplemental-timing-correction.json) preserve numerical
helpers and verify timing order/synchronization. The local numerical oracle is
reused through those unchanged functions; no partial early/middle row is rerun.
Timing now explicitly includes the norm reference/rescaling and is only a probe
path comparison. Full-training overhead and comparable peak memory remain open.


### 2026-11-08 19:08 UTC: late-only supplement accepted

Iris accepted r04 from `7e917a2be5c8744a4ec70742496b43b82a4e557a`. The
[launch record](supplemental-probe-r04-accepted.json) binds the late-only request.
No early/middle measurements are repeated; no full-budget training restarts.


### 2026-11-08 19:26 UTC: minimal supplement findings, with failed-run provenance preserved

The late-only r04 supplement completed: four successful attempt1 tasks and four
finished ONLINE probe runs, nine identical finite loss rows per host, strict
counter33597 restore, and generation/hash-bound artifacts. Attempt0's worker
reconciliation failure and controller retry count1 remain visible; no provider
preemption cause or training resume is inferred.

[Supplement findings](supplemental-findings.md) report all18 quarter-versus-control
contrasts over eight fresh batches. Late step3e-5 favors quarter on8/8 batches
against both controls (−0.000685364/−0.001069099 nats), but other settings are mixed.
The21 early/middle/SVD rows remain single-artifact recovery from the OOM-failed
r03 job; no finished ONLINE or four-host agreement is claimed for them. Exact
SVD effects change sign with step size and do not explain a full-training ranking.

The validated warm norm-matched probe path takes0.09675s/39.80455s/39.80278s for
alpha0/.25/.5. This includes norm-reference/rescaling in the probe layout, not
full training. Logical checkpoint storage is equal; comparable peak memory and
full-training attribution stay open. All supplement job spans total a conservative
16.316 device-slot-hour bound, including failures and rescheduling wait, not billing.

The user-scoped minimal supplement is concluded with these evidence limits.
No cross-seed, failed-longrun rerun, horizon/998M campaign or shutdown-engineering
execution is queued. All completed new task/native/artifact reads are cached;
do not repeatedly query them or launch a follow-up automatically.

### 2026-11-10: frozen-checkpoint Hessian pilot

The user requested the real checkpoint experiment from the Hessian-power package.
Start with counter 18,665 and the first q_proj and gate_proj slices, then repeat
at early/late counters after derivative and data checks pass. The five supplied
NumPy synthetic checks pass; they contain no Marin measurements.

The new runner uses the existing complete-state restore and matrix_state mapping,
FP32 parameters/forward/derivatives, highest matmul precision, vanilla attention
and unfused cross entropy with configured z-loss. It measures 12 distinct Gram
eigenvector products across eight disjoint sampled sequence groups, retains all
projected entries and computes leakage from averaged full block HVPs. Fresh
sampling uses the fixed training caches, not a claim of unseen training data.
Each group retains the native valid-token-normalized objective and equal group
weight; valid token counts and token hashes are recorded.

[The frozen pilot request](hessian-middle-request.json) records the input hash,
measurement selection and resource plan. Source/configuration will be published
before submission. Local ADC reads confirm permanent checkpoint metadata remains
available; complete restoration and actual counter validation occur on the host.
No new training, shift, GGN substitution or GPU allocation is involved.

The pilot was submitted at 16:52 PST as
`/kaiyuew@stanford.edu/ema-gram-hessian-middle-20261110-01`, using source
`be4552103096bec8b9da94d4190d112af3bbc600`. The 16 MB bundle excludes historical
logbooks and documentation while retaining the exact frozen configuration.
[The launch record](hessian-middle-launched.json) preserves the command, flags,
and validation. Two numerical tests and lint passed; real results remain pending.

The first attempt restored counter 18,665 completely and loaded eight groups
of 32,768 tokens (32,760 valid weights each), with no duplicate sequences.
It stopped before curvature collection: the weakest selected q_proj direction
had finite-difference relative errors 0.0501, 0.2489, 0.3163 at steps 0.1, 0.03,
0.01, while bilinear symmetry error was 2.65e-10.
[Diagnostics](hessian-attempt01/attention-derivative-checks.json) and the data
manifest are preserved. This is a derivative-validation failure, not an
indefinite-Hessian result or a completed power fit.

The increasing error as steps shrink motivates a declared wider sweep
{3, 1, 0.3, 0.1, 0.03, 0.01, 0.003}. The next attempt checks low, middle and
high Gram modes, retaining HVP and gradient-difference norms. The 2% tolerance
is unchanged and now must hold at two adjacent scales in every tested mode.
No curvature data from the failed gate will be used.

### 2026-11-10 17:12 PST: real Hessian pilot blocked at strong-mode validation

The revised middle pilot ran from `bbf7847b72fc8f44e85b1bea56695a75e5e24e8f`.
It again restored counter 18,665 and the same eight disjoint data groups. The
HQ/code/math object catalogs match their archived checksums. Low and middle
Gram modes passed the unchanged 2% adjacent-scale test; the strongest mode
failed, with best relative error 0.8357 at the smallest tested step 0.003.
Its HVP norm is 12.5594, versus 0.000103842 for the weak mode. Smaller strong-mode
steps remain untested; no cause is established by this result alone.

[Pilot findings](hessian-pilot-findings.md) retain both failures, data validation,
the second launch and generation-bound diagnostics. Iris reports terminal failed,
one failed task and three coscheduled sibling failures, with zero preemptions.
No full curvature export or power fit exists; early/late runs were not launched.
Further launches are stopped under the repeated/unclear failure rule in the
Iris monitoring reference. A smaller-step diagnostic is the concrete next study,
without relaxing the gate or treating symmetry alone as validation.

### 2026-11-10: authorized smaller-step derivative investigation

The user explicitly authorized continuing numerical debugging after the second
failure. Extend the existing declared step sweep through 0.001, 0.0003, 0.0001,
0.00003, 0.00001, 0.000003, and 0.000001, keeping the same 2% adjacent-scale
requirement and the same frozen checkpoint, basis, data and precision. The
existing fail-closed runner collects curvature only after every tested mode
passes. This tests whether the strongest mode has a convergence window at
smaller perturbations; it does not assume that explanation is correct.

### 2026-11-10 21:16 PST: middle pilot completed; early and late submitted

Attempt 3 from `579fc3ade` passed both derivative gates and completed all 192
HVPs. The strong q_proj mode passed at 1e-4, 3e-5 and 1e-5 with errors 0.00339,
0.00192 and 0.00402. All four Iris tasks succeeded, and all four online W&B
measurement runs are `finished`. The publishing run is
[035397d82a6b](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-035397d82a6b).

The [attention analysis](hessian-middle/attention-fit.json) rejects both proxies:
the unshifted mean has minimum eigenvalue -0.07669 and leakage 0.9274. All 1,000
conditional batch resamples remain indefinite. The [MLP analysis](hessian-middle/mlp-fit.json)
passes the supplied numerical positivity gate: raw r=1.1941, rho=0.91595,
alpha_log=0.38353 (conditional 95% interval [0.38136, 0.38627]); ridged
alpha_log=0.41190. All 1,000 fit resamples are positive. MLP leakage is 0.2968,
so this is a projected-subspace descriptor, not a full-block or training optimum.

MLP's minimum/maximum eigenvalue ratio is 4.23e-8: a global relative gate of
1e-7 would reject it. Diagonal scaling gives minimum eigenvalue 0.835, indicating
the large condition number primarily reflects scale differences. Neither that
diagnostic nor batch bootstrap establishes a calibrated floating-point error
bound. These sensitivity limits accompany the numerical fit.

The early and late repetitions were accepted at 21:15 PST from published source
`72acf6036`, with identical numerical code, seed, mixture and thresholds.
[Early launch](hessian-early-launch.json) and [late launch](hessian-late-launch.json)
retain their exact commands. They restore counters 3,733 and 33,597 respectively;
each must pass its own derivative and data checks before exporting curvature.

### 2026-11-10 21:29 PST: requested three-checkpoint Hessian experiment completed

The early and late jobs both succeeded on all four hosts, with all eight online
W&B measurement runs verified `finished`. Together with middle, this yields six
8-by-12-by-12 curvature exports. Token hashes match across all checkpoints.
Both blocks pass the 2% adjacent-scale derivative checks at every checkpoint.

[Final findings](hessian-findings.md) preserve the original postprocessor's
acceptance decisions. Early/middle attention are indefinite. Late attention is
rejected by the separate input symmetry gate (batch 7: 1.55066e-5 > 1e-5);
its symmetric-part spectrum is only a diagnostic, not an accepted fit.
No shift, eigenvalue flooring, absolute-value replacement or relaxed tolerance
was used to obtain a result.

MLP raw alpha_log is 0.3766, 0.3835 and 0.4101 at early/middle/late; production
ridging gives 0.4052, 0.4119 and 0.4365. Late has 228/1,000 invalid bootstrap
replicates, so its interval is conditional on the 772 valid replicates. Leakage
is 0.4905, 0.2968 and 0.3824: none is a full-block training-power recommendation.
All MLP minimum/maximum eigenvalue ratios are below 1e-7; resolution sensitivity
and diagonal-scaling diagnostics accompany the default-gate numerical fits.

The requested measurement stage is complete with these failures and uncertainty
limits explicitly retained. No additional training, fresh-Gram campaign or
automatic follow-up is launched. Durable artifact manifests bind exact GCS
generations, sizes, CRC32C and local SHA256 values; bounded reports are archived
on the research branch, and six NPZ exports accompany the local results package.

### 2026-11-11: motivation-aligned follow-up authorized

The user requested additional experiments wherever the initial measurements
did not address the original motivation. The report remains pending until the
follow-up evidence is incorporated. The motivating question is why alpha=1/8
performed well in the previous single-seed comparison, not whether a sparse
quarter-power projection admits a numerical matrix logarithm.

First measure the actual eighth-power trajectory, using the archived exact
training configuration and regional checkpoints. Storage lists available labels
3733, 18665 and 33597, one update later than the prior quarter-power labels.
Metadata and config round trips are checked locally; full execution-host
restoration must verify the actual counters before measurement. The local
TensorStore client cannot parse the current ADC credential format, although
ordinary storage SDK metadata reads succeed; no access restriction is bypassed.

Add projected finite-difference checks for every direction and every data group.
Use the largest adjacent pair from the declared 3-to-1e-6 sweep satisfying the
unchanged 2% full-HVP criterion. Reuse the first group's scales when they pass;
otherwise apply the same sweep to that group. Save both projected references.
For positive mean curvature, apply diagonal congruence scaling and require its
minimum eigenvalue to exceed ten times the largest spectral discrepancy among
the HVP matrix and the two finite-difference references. This is an empirical
agreement screen, not a rigorous bound on all systematic errors. Retain the
original symmetry gate and all unmodified eigenvalues. Failed screens remain
unresolved; indefinite curvature remains a valid negative finding.

After the middle eighth-power pilot, repeat at the available early/late states
and apply the same stronger validation to the quarter-power middle control.
Further controls will compare two independent frozen-weight fresh Gram histories
against the stored history in one fixed basis, compare a nested wider subspace
against the original twelve directions, and measure held-out loss changes of
the actual polar matrix-update rule. Each control needs its own predeclared data
partition and validation before launch. Frozen proposals do not constitute a
new training trajectory or identify the globally best training exponent.

### Frozen controls and nested subspaces

The follow-up controls use the original model attention configuration and mixed
precision for first derivatives. Two separately randomized histories each collect
512 accepted batches at frozen weights, with the native global clip and beta=0.99.
Their finite EMAs are bias corrected. Entire candidate batches containing repeated
sequence hashes are rejected; accepted source positions and rejection counts are
recorded. Curvature, history, update-gradient and evaluation data are disjoint
within this measurement, not claimed unseen during prior training. All projected
Gram entries are retained, including off-diagonal entries. The controls select a
nested 48-direction basis so both its original 12-direction restriction and wider
projection can be analyzed without rebuilding the histories.

Full native optimizer candidates use alpha=0, 0.125, 0.25, 0.375, 0.5 with the same
saved optimizer state, one fresh gradient and native LR/decay. Sixteen paired
batches evaluate fractions 0.1, 0.3 and 1.0 of each update. A separate diagnostic
matches the entire update norm to alpha0; this also rescales AdamW and decay and
must not be described as isolating only matrix direction. Evaluation uncertainty
conditions on the single update gradient. No training trajectory is advanced.

Expanded Hessian bases retain the original 12 directions. Preliminary derivative
checks use actual Gram-product ranks, not appended-array positions. Leakage for
the original 12 HVP inputs is evaluated against both the original and expanded
output projections, with the same denominator. This avoids conflating a changed
input distribution with improved response capture.

Validation: five numerical tests passed; a tiny-model end-to-end control test
exercises all 30 update branches, forced duplicate rejection, common evaluation
batches, equal norms in the matching arm and unchanged source model/optimizer
state. A separate native-update comparison agrees with take_train_step for all
five alphas. These checks do not substitute for execution-host validation.

Scientific acceptance remains separate from artifact export. The unchanged
package can label a numerical fit valid even when the new empirical precision
screen is unresolved. Follow-up summaries must require both checks before calling
a power fit accepted; negative or unresolved measurements remain in the archive.

### Submitted aligned campaign and acceptance postprocessing

Seven jobs are bound to dated launch JSON records: eighth early/middle/late;
eighth and quarter frozen controls; eighth and quarter nested 48-direction
Hessians. The middle pilot uses e5acf9e20; the other six use c194c6872. All retain
default scheduler preemption/retries, request v4-32 in us-central2, and impose no
wall-clock limit. Every measurement has a separate online tracking identity and
a CPU logging preflight before TPU initialization. These are frozen measurements,
not restarted or resumed training runs.

The middle eighth pilot fully restored counter18666. Its attention block has
positive mean curvature and passes the declared empirical screen (margin29.203,
threshold10). This is preliminary block evidence while the job remains running.
Raw descriptive r=1.24226, rho=0.974253 and alpha_log=0.392129. The new
postprocessor reapplies both numerical and empirical gates to every bootstrap
sample: 619/1000 accepted, 178 indefinite and203 empirically unresolved. These
conditional intervals must not be described as unconditional uncertainty.

The acceptance postprocessor preserves the original supplied-script results and
adds a separate accepted_positive_power decision. Synthetic validation accepts
an exact fourth-power case and rejects a numerically valid fit with unresolved
empirical precision. Fresh-proxy comparisons require matching basis archive size
and CRC before sharing coordinates. Native local-update comparisons use paired
batch bootstrap intervals, conditional on the one update-gradient batch and the
saved state; no multiple-comparison or training-seed claim is made.

### Native-control precision correction

Both already-submitted controls failed at their first fresh-gradient compilation,
before producing a Gram history or optimizer-loss result. The TPU Mosaic error
reported a bf16 attention matmul with an FP32 contraction precision and `Bad lhs
type`. The shared Hessian entrypoint had unconditionally forced
jax_default_matmul_precision=highest, despite controls-only restoring the native
attention backend. The frozen training configs do not specify that override.
Local Splash attention source uses lax.dot_general without an explicit precision
argument, so this global setting propagates into the kernel.

The correction applies highest precision only to curvature measurements; native
controls retain the training-config/default setting. A numerical-configuration
artifact records the effective precision and backend. This restores the intended
native control protocol and does not weaken the 2% HVP or empirical resolution
thresholds. Continue the focused numerical correction under the user's existing
“Continue numerical debugging” authorization and request for aligned experiments.
Preserve both failed attempts. Relaunch from the original complete frozen state
with fresh tracking identities; no training-resume claim applies.


### November 11 — Verified the motivating trajectory at three stages

The eighth-power early/middle/late measurements all completed on their original
attempts: canonical Iris success (4/4 tasks), completed GCS artifacts, and eight
finished online tracking runs per job including logging preflights. Complete
restores verified counters 3734, 18666, and 33598. All six original/follow-up
12-direction data manifests are identical, including sequence hashes and weights.

Accepted mean fits under both numerical and empirical requirements are early
attention and both middle blocks. Raw fit-derived alpha values are 0.37008,
0.39213, and 0.40598, respectively. Empirical precision margins are 10.905,
29.203, and 36.418 (required >10). Strict bootstrap acceptance is 511, 619, and
585 out of 1000. Intervals remain conditional on acceptance and the fixed proxy.
Early MLP and late attention fail positivity. Late MLP passes the original
numerical gate but its empirical margin is 4.784; its raw alpha 0.42436 is
unresolved and must not be included among accepted fits. Negative signs are not
separately calibrated by the screen, which exits on nonpositive mean curvature.

The native-precision correction is validated through fresh-gradient execution:
quarter-controls attempt02 completed its first 512-batch history and progressed
through the second. The first two failed control attempts remain preserved and
produced no fresh Gram or optimizer-loss result. Eighth-controls attempt02 was
then submitted from published source 9ca969f8a, with no wall-clock limit and no
preemption override. Both 48-direction measurements remain running. No training
trajectory is advanced by any of these measurements.

### November 11 — Completed all seven aligned measurements

All seven added experiments are now verified complete: four successful Iris
hosts, completed durable artifacts, and all eight online tracking runs finished
per experiment. Both controls use corrected source 9ca969f8a; the two failed
native-control attempts remain archived. No new training trajectory was run.
The eight curvature data manifests (original three plus five follow-ups) are
byte-identical. Both widened twelve-direction restrictions reproduce their
original middle curvature matrices exactly. Four fresh histories each contain
512 batches; the two controls have identical accepted and rejected sequence
manifests, 8,392 unique complete sequences, and 25 rejected candidate batches.
This excludes duplicate complete sequences, not shared fragments/documents.

The quarter middle MLP now passes the empirical screen at dimensions12 and48
(margins193.133 and42.011). Saved raw alpha is0.38353 and0.38467; ridged alpha
is0.41190 and0.41283. Strict bootstrap acceptance falls from1000/1000 to663/1000
at48 (247 indefinite,90 unresolved). Quarter attention has negative measured
mean curvature and one direction/group lacking a valid derivative reference;
no accepted fit. Eighth middle attention passes at12 but fails positivity at48
(minimum-5.156e-7, sign not separately calibrated). Eighth MLP passes at12 but
is unresolved at48 (margin7.100<10). No thresholds were relaxed.

With the same original12 HVP inputs, enlarging only the output projection to48
barely reduces leakage: quarter attention0.927401->0.926993, quarter
MLP0.296787->0.296575, eighth attention0.387614->0.387059, eighth
MLP0.513949->0.513878. These are norm ratios, not squared-energy fractions.

Fresh proxies agree more closely with each other than with the saved proxy in
both dimensions and for raw/ridged definitions. At48, raw centered-log distances
fresh/fresh versus saved/fresh are respectively0.178 versus0.838/0.869
(quarter attention),0.264 versus1.741/1.754 (quarter MLP),0.151 versus
0.627/0.665 (eighth attention), and0.156 versus1.144/1.155 (eighth MLP).
This suggests history dependence; two windows do not identify a noise/drift
variance decomposition. Accepted fresh raw fits yield alpha0.431–0.477 and
r1.042–1.142, not either proposed one-eighth pattern.

All60 full-optimizer hypothetical branches completed. At the quarter state,
all five full native mean loss changes are positive; negative means at smaller
steps have conditional intervals spanning zero. At the eighth state, native
candidate1/4 versus1/8 has paired loss difference-0.000276
(conditional95% interval[-0.000361,-0.000183]). After matching whole-update
norms, their mean changes are nearly identical (-0.00050436,-0.00050458),
with paired difference interval[-0.000165,+0.000169]. Every candidate improves
at0.1 and0.3 native fractions in that state, with conditional intervals below0.
One gradient/state is fixed per control; no multiple-comparison correction or
cross-seed/long-horizon conclusion is claimed. Fresh diagnostic histories were
not substituted into these optimizer controls.

The report explains the motivation, actual measurements, numerical rejection,
leakage, fresh statistics, and full-update controls. The geometric hypotheses do
not explain the earlier training ranking in these selected measurements.
Repeated controlled training remains a future causal test, not a claimed result.
