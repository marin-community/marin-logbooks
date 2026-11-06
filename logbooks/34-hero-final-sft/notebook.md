---
title: "Compact SFT on the final Hero pretraining checkpoint"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/34"
research_branch: "codex/34-hero-final-sft"
marin_base_revision: "33189166cfb244bf284f3d893e58ce0272215eac"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: heldout-preparing
---

# Compact SFT on the final Hero checkpoint

## 2026-11-05: source verification and repeat design

Repeat the [earlier compact chat SFT](https://github.com/marin-community/marin-logbooks/issues/2)
on final pretraining step 390251, skipping checkpoint averaging. The prior run used
64 updates on intermediate step 294000 and demonstrated basic turn-taking on
100 reviewed responses. Its held-out evaluation and independent merge parity
were incomplete. The small training dataset was reused for about 47.6 corpus
token-volume equivalents, so training-source EOT acquisition is insufficient
for selecting this repeat's checkpoint.

Read-only Iris inspection confirmed all 176 training tasks succeeded with exit
0, and the coordinator succeeded with exit 0. The final checkpoint metadata
records step 390251, timestamp `2026-11-05T15:54:51.998349`, and
`is_temporary: false`. [source-verification.json](source-verification.json)
records exact checkpoint, metadata, manifest and source-configuration digests.
The [final-save incident](https://marina.oa.dev/echo/wiki/929) records the prior
JAX shutdown and S3 write failures; this investigation made no production-run,
checkpoint-retention or cluster changes. Live W&B completion is not verified.

The research base is the exact final-training revision, rather than a later
model implementation. [source-model.json](source-model.json) contains only the
saved model configuration, extracted from the run artifact without runtime
environment values. It retains context capacity 65536, QK multiplier 1.66,
receiver capacity factor 1.5 and pooled transport capacity factor 1.15. No YaRN
extension is introduced.

### Implementation plan

1. Restore the archived compact-SFT interfaces and selectively port LoRA and
   vocabulary hooks into the final-training model. Preserve the frozen-weight
   short-convolution backward from Marin #10556. Verify adapter gradients,
   compact optimizer state, checkpoint recovery and merged-weight behavior.
2. Bind the final native checkpoint and safe model record by metadata,
   manifest and configuration digests. Apply the five original donor output
   column substitutions in memory before the BF16 base conversion. This avoids
   writing a separate roughly 2.14 TB FP32 donor checkpoint. Keep effective
   routing bias in FP32.
3. Reuse the exact original no_robots JSONL, split, tokenizer and completed
   token cache after verifying their manifests and fingerprints. The split has
   4806 training and 1186 held-out conversations; tokenizer revision is
   `a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`.
4. Train rank-16/alpha-16 attention/shared-projection LoRA and five input/output
   vocabulary deltas. Freeze ordinary base parameters and routing control.
   Preserve peak adapter/vocabulary learning rates 1e-4/1e-5, two warmup
   updates and seed 20261021. Pack 64 sequences of 16384 positions per update,
   with expert64/context1/replica1 on 64 GB200 GPUs.
5. Configure the optimizer horizon as 128 from the start. Stop the smoke at
   update 8; continue its complete compact Adam state, counters and data cursor
   to update 128 without changing the learning-rate schedule. The cumulative
   position limit is 134217728, twice the previous 64-update run. At similar
   non-padding density this is about 95 training-corpus token-volume
   equivalents; it is not additional unique data.
6. After all gradients finish, evaluate compact updates 64, 96 and 128 on the
   same frozen held-out assistant-target/EOT panel. Avoid alternating shorter
   diagnostics with training executables, which failed in the prior campaign.
   Among candidates with EOT top-1 count at least that of step 64, select the
   lowest assistant-target NLL; exact ties choose the earlier checkpoint. Step
   64 remains eligible. Record every score and any numerical or routing issue.
7. Write one canonical native BF16 export of the selected checkpoint, verify
   adapter-to-merged numerical behavior, and review all 100 frozen math, code
   and fixed-history responses. Preserve the 65K capacity declaration while
   explicitly reporting that this short panel does not test 65K behavior.

The planned launch is one 16-task, four-process-per-task GPU gang. Launch
requests will use unique names, output roots and receipts, zero automatic
application-failure retries and bounded preemption recovery. Compact states
are approximately 0.878 GB per update, including Adam; retaining 128 states
costs approximately 112.4 GB before overhead. Put training attempts and
intermediate compact states under a region-local 30-day lifecycle prefix.
Keep the selected compact checkpoint, one approximately 1.07 TB native export,
resolved requests and compact reports under the caller-owned durable output
root. Exact roots and launched source/request identities must be recorded
before submission. Babysitting uses one heartbeat after distributed startup.

The compact modules were restored from archived source
`c39c6f2f5b1b1c5aaa29d51a39304b40c7612dca`; the frozen-weight short-convolution
implementation and its existing tests come from Marin #10556 at
`467b8393773a143ad465fb421a98de4278b62faa`. The final-training model receives
only the adapter and embedding-delta hooks needed by this experiment. The
held-out materializer, compact evaluator and bounded export comparison are new.

The fixed selection panel contains the first 256 valid held-out conversations,
48013 assistant targets and no truncated conversations. Its SHA256 is
`545e4e2fa893875fdd84641cde4dee83e85b33843a260a25fa25591eeaa7f862`.
The entire train and held-out JSONL hashes and all 18 historical cache objects
(2478098 bytes) matched fresh S3 readbacks. The cache fingerprint remains
`1f405933`, with 1385205 training tokens. Safe model configuration and the panel
were published with fsutil verified-copy under
`s3://marin-us-east-02a/marin/users/dlwh/hero-final-sft/2026.11.05-v1/inputs`.

The preflight validated the final source metadata/configuration/layout and
resolved that exact completed cache. Required lint and type checks passed.
Focused source tests passed 18 cases with 37 accelerator-only skips. Independent
source review found no smoke blocker. It found that create-only export
comparison reports do not survive transparent export retries; the launcher now
requires zero preemption retries for selected export. An interrupted export
needs inspection and a fresh destination. The broad affected run passed 3913 tests, skipped 48 and had six expected
failures. Twenty failures read CI files deliberately removed from the archive;
all twenty pass on unchanged source with its workflows present. One Linux-only
bubblewrap failure reproduces on unchanged source. The isolated Harbor suites
passed 1028 tests. No test expectations or numerical tolerances were relaxed.
[validation.json](validation.json) records the checks and limitations. GPU startup, memory fit, 64-rank transport and full-size
kernel behavior remain unverified.

The port and all new evaluation code are research artifacts. No Marin
contribution or production adoption is implied. No SFT job has launched.


## 2026-11-05: direct-controller admission correction

The first smoke submission from source `83b709982db97380ba3f6e6de9919ef7d3fd0eaa`
was rejected before job creation. Its resources carried `target_cluster` equal
to the already selected direct controller, so admission interpreted the pin as
a federation request to an unconfigured peer. A canonical-prefix job listing
confirmed no job existed. The exclusive pending receipt and rejected request
remain intact; no GPU work was launched by this attempt.

The corrected v2 request omits the self federation pin, retains direct controller
`cw-us-east-08a`, and uses a fresh job name, output root and receipt. The launcher
now rejects this configuration locally before making a submission. Model,
optimizer, data, hardware count and step bounds are unchanged. The corrected
request will receive its own reviewed, immutable publication before submission.


## 2026-11-05: corrected smoke accepted

Iris accepted `/dlwh/hero-final-sft-smoke-20261105-v2` on `cw-us-east-08a`
from published source `cee9e7f52712a4ba99795c438a93f9ec57e2b9dd`. The exact
request SHA256 is
`e6777fc2bf342dc4dc566182b329ce52be0113f3be63ecb64748fdfa69b82973`.
[smoke-launch.json](smoke-launch.json) records the configuration and immutable
request URI. The submitted receipt is preserved at
`s3://marin-us-east-02a/marin/users/dlwh/hero-final-sft/2026.11.05-v1/receipts/smoke-v2/receipt.json.submitted.json`.

The launch uses 16 tasks with four GB200 processes each, interactive priority,
a two-hour per-task runtime limit and one-hour scheduling limit. There are no
automatic application-failure retries. It stops at eight updates on the fixed
128-update optimizer schedule; the cumulative non-padding token cap is 8388608.
No averaging or separate donor checkpoint was performed. Outputs and compact
states are under the 30-day lifecycle prefix named in the launch record. W&B
is disabled; telemetry and exact checkpoints provide progress evidence.

The user is the DRI. The current chat owns babysitting through the
`finish-final-hero-compact-sft` heartbeat every ten minutes. It may continue a
verified successful smoke to 128 cumulative updates without changing the
schedule/data/model, then perform the fixed held-out selection, one selected
export and 100-response review. It may not extend beyond that bound. At 20:55:22 UTC all 16 tasks were running with zero failures or
preemptions. Recent regional startup logs showed successful TensorStore
checkpoint barriers across ranks at 20:55:39 UTC. Actual gradient telemetry and
smoke completion are still pending verification.


The leader attempt is `d361647db22a79a3`; its attempt record matches the exact
v2 request SHA256, semantic request digest
`76856da1ff064fb9197b40fd56fe589667f780150ff84da125d704807352cf7f`
and launched source. It records no recovery parent. This binds later telemetry
and checkpoints to this execution rather than the rejected v1 identity.


## 2026-11-05: smoke passed; preserve full state for 128 updates

The smoke succeeded: all 16 tasks exited 0, with no application failures or
preemptions. All eight updates had finite supervised loss, gradient norm and
routing metrics. The optimizer audit contained only the intended compact state,
and all 22 returned gradient leaves belonged to LoRA or the five-token
vocabulary deltas. Peak HBM was 74077436232 bytes per GPU (about 69.0 GiB).

Update 8 recorded loss 1.64170, gradient norm 0.91174, mean routed-assignment
overflow 0.08299 and router z-loss 12.4396. Loss endpoints used different packed
batches. These are runtime/training-source measurements; they establish no
held-out or generation quality. [smoke-result.json](smoke-result.json) retains
exact identities, counters and checkpoint pins.

The completed compact checkpoint is step 8 with metadata digest
`40d55592efd21db14c8092e151fef2c3de46910080ab9bb94b3aa4418fe0f0df`,
8252608 processed and 5314600 supervised tokens. Its completion record matches
the published request and actual attempt. A fresh continuation is prepared to
perform 120 additional updates, ending at 128 total, while preserving full
Adam, schedule count, deterministic data position and the fixed 128-update
horizon. [continuation-template.json](continuation-template.json) pins the exact
parent request and checkpoint; its source placeholder is resolved to the
published commit at launch. The continuation has not been submitted yet.


## 2026-11-05: continuation accepted

Iris accepted `/dlwh/hero-final-sft-long-20261105-v1` on `cw-us-east-08a`
from published source `e8ddebe1305ae8c543501fa4b8210e584425f5b3`. Its request
SHA256 is `b9d37c34c5e3cebe34f6bf196ed9dbcc4d23b8e5ac2321df119178df9f448c3e`;
[continuation-launch.json](continuation-launch.json) records the immutable
request URI, output root and hardware/runtime limits. Preflight selected exactly
the verified step-8 compact checkpoint and counters. It will perform 120
additional updates, ending at 128 cumulative updates on the unchanged schedule.

The continuation uses the same 64 GB200 GPUs and direct controller, with a
three-hour runtime limit, one-hour scheduling limit and zero application-failure
retries. Full compact state, 16K packing, 65K model settings, source/data/cache
pins and peak learning rates remain unchanged. The existing heartbeat now owns
this exact continuation; no second continuation should be submitted. Actual
restoration/progress beyond update 8 and final completion remain pending.


## 2026-11-05: continuation restored and advanced

At 21:06:50 UTC all 16 continuation tasks were running with no failures or
preemptions. Attempt `0a3328348b80b978` records the exact requested step-8
recovery source. Its first new update is 9: loss 1.63636, gradient norm 1.24963,
9286178 processed and 5981570 supervised tokens. This verifies real progress
beyond the restored state. [continuation-start.json](continuation-start.json)
pins the source/request/attempt, compact parent metadata and raw update hash.

Further monitoring is delegated to the single existing ten-minute heartbeat.
The next evidence gates remain completion at 128 total updates, fixed held-out
selection among 64/96/128, one selected export, durable compact preservation and
review of all 100 response cases. No held-out or response-quality result exists
yet. The logbooks index update remains for the completed experiment handoff.


## 2026-11-05: 128-update training completed

The continuation succeeded: all 16 tasks exited 0, with no failures or
preemptions. The 120 new update records cover exactly steps 9 through 128;
all supervised losses, gradient norms and recorded routing metrics are finite,
and the compact optimizer audit passed. The final handoff and permanent
step-128 metadata match the exact request and execution identity.
[training-result.json](training-result.json) records counters, final measurements,
checkpoint pins and the fixed 64/96/128 selection candidates. These training
losses use different packed batches and do not establish held-out improvement.

Next, run the predeclared 256-conversation held-out panel against the three
compact states. [heldout-template.json](heldout-template.json) pins actual
checkpoint metadata and the original panel SHA256. It retains batch 64,
physical padding 16384, expert64/context1/replica1 and the same 65K forward
configuration. The evaluator runs only forward passes, with one separate
64-GB200 job, a one-hour runtime bound and zero application/preemption retries.
A failed or interrupted evaluator requires inspection before using a fresh
execution identity. Its source placeholder is resolved after publication.
No held-out result, selected export or response-quality result exists yet.


## 2026-11-05: held-out evaluator accepted

The fixed 64/96/128 comparison is running as
`/dlwh/hero-final-sft-heldout-20261105-v1` on `cw-us-east-08a` from
published source `e81d24d4328a1046043fdcf90dde8453f8d562ec`. The request
SHA256 is `05564dd2edcc05f7ccf3c39c0fbdf8e579f5f56816e09afd081b9ec2d6f7bbb4`.
[heldout-launch.json](heldout-launch.json) records the exact immutable request
publication, accepted receipt, command and output. At 00:18 UTC on November 6,
all 16 tasks were running with no failures or preemptions; this scheduler state
does not establish numerical evaluation progress or a completed report.

The worker uses the pinned 256-conversation panel with 48013 assistant targets,
batch 64 and physical padding 16384. The minimum assistant NLL among candidates
whose EOT top-1 count is at least step 64 selects the export; exact ties select
the earlier step. One existing heartbeat owns monitoring. No more gradient
updates or duplicate evaluator launch is authorized. Report validation, one
selected native export, durable compact preservation and the frozen
100-response review remain pending.
