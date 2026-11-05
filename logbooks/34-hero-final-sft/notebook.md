---
title: "Compact SFT on the final Hero pretraining checkpoint"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/34"
research_branch: "codex/34-hero-final-sft"
marin_base_revision: "33189166cfb244bf284f3d893e58ce0272215eac"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: preparing
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
