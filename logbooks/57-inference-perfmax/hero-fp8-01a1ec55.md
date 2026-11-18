# Hero BF16 and online FP8 campaign

Campaign: `s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-fp8-01a1ec55/`.
Source worktree: `hero-fp8-01a1ec55`, local branch `codex/hero-fp8-01a1ec55`.
The published branch remains `codex/57-inference-perfmax` in marin-logbooks.

## 2026-11-18 01:56 UTC — Scope and hypotheses

The dispatch delegates experiment choices and provisional FP8 bounds. Start at
research head `3f0c06de9f02d1778a9ef7d64119bd7885094501`. Keep runtime
`0ac9dcb157ee86ccb447bc717259159c41ad456a`, TP1, BF16 KV, the step400407
export, tokenizer and saved Levanter goldens fixed. Request interactive priority;
accept normal Batch fallback. Own at most 32 concurrent GB200 GPUs and no H100s.
The parent owns the cache-contract correction. Qualification measurements wait
for its exact source revision.

The [handoff](handoff.md) has no Hero AgentX window. The
[prior throughput study](https://github.com/marin-community/marin/issues/11430)
calls BF16 decode step-latency bound and identifies this runtime as throughput
neutral against its previous wheel. That synthetic evidence is a starting
hypothesis; AgentX includes long histories, prefix reuse and replay pauses.

1. Online FP8 reduces routed-expert weight memory and may free KV capacity.
   Expect lower resident weight memory and potentially fewer cache evictions.
   Check actual versus theoretical prefix reuse and KV occupancy before
   attributing a throughput change to cache capacity.
2. FP8 can reduce matrix time, but launch and communication costs may dominate
   at low batch sizes. A four-GPU, C16 diagnostic tests full-model loading and
   numerics before a sixteen-GPU comparison. The online implementation uses
   meta weights and materializes layers during loading, which makes this small
   allocation plausible for the 535B model.
3. Use sixteen-GPU C64 BF16 and FP8 as the first matched load. Choose neighboring
   loads from measured queueing, KV use and throughput. Prefer completed output
   tokens/s/GPU, then lower latency when throughput differs by less than 5%.
   The 5% band is a campaign heuristic. It is not a latency requirement.

The [FP8 policy](analysis/hero-fp8-01a1ec55/fp8-policy.json) is a separate
provisional engineering screen selected before FP8 observations. It deliberately
requires the same numerical envelopes as the current BF16 screen. This is a
conservative campaign choice, not approval of those numbers as an FP8 quality
budget. It has no FP8 calibration evidence. Exact cache persistence remains zero.
Identity mismatches, non-finite scores and demonstrated state corruption fail.
Keep failed observations and do not change bounds to explain a bug. RL learning
and trainer probability ratios remain outside this campaign.

The model configuration, export manifest and tokenizer resolve correctly on CPU.
Reference lookup reuses all fixed saved histories and scores. CPU workload
preparation runs before GPU allocation. Native model inspection will retain the
loaded online FP8 methods; KV precision is checked separately by the shared
deployment check. No runtime override is planned initially.

The initial provider read saw 206 GB200 nodes, 205 ready and schedulable. Five
looked idle in Iris's snapshot. This is not a reservation. Refresh both Iris and
provider state immediately before submitting. The compact
[inventory](analysis/hero-fp8-01a1ec55/inventory-initial.json) retains its digest.

The runner currently includes the full numerical policy hash in comparison
controls. Separate BF16/FP8 policy identities therefore form separate automatic
report groups even when their numerical limits match. Preserve this restriction;
inspect actual campaign evidence before deciding whether a small reporting fix
is justified.

## 2026-11-18 02:03 UTC — Keep corpus preparation off devbox

Manual preparation encountered a cold Hero workload cache. Its native aiperf
processor reached 6,314,932 KiB RSS while devbox had about 4.3 GiB available at
startup and full swap. The shell bridge became unavailable twice. This does not
establish an OOM kill. The owned preparation tree was stopped; the retained
failure is exit 143. Both this session and the parent verified those processes
were gone. No GPU was submitted. Use the runner's existing 32-GB CPU coordinator
for the cold corpus. The local attempt is retained in the
[failure receipt](analysis/hero-fp8-01a1ec55/local-preparation-failure.json).

The first broad lint invocation formatted fifteen old JSON artifacts solely by
adding final newlines. Those exact EOF-only edits were verified and reverted.
Subsequent lint calls name only this campaign's files. Pyrefly's initial host
result reports 32 missing optional imports in unrelated Marina apps; retain
that limitation without changing the runtime lock or those files.

## 2026-11-18 02:15 UTC — Compare equal criteria across weight precisions

The BF16 and FP8 fixtures use equal numerical limits and probes, but their full
policy hashes differ because applicability records the weight quantization.
The report treated them as separate comparison groups. The runner now retains
a second hash for numerical criteria while keeping each complete policy identity
and hash. The report compares this criteria hash for new results; older records
still require their full policy identity. Changing a threshold or probe still
blocks a baseline recommendation. The table now shows weight precision.

Twenty-three focused report, scope and recovery cases pass, including equal
criteria across BF16/FP8, changed criteria, and older records without the new
hash. The [test output](analysis/hero-fp8-01a1ec55/report-criteria-tests.txt)
is retained. This fixes reporting from actual campaign use. It changes no
numerical limit, serving setting, golden or checking function.

The first diagnostic coordinator is
`/romain/knotmeter-20261118t021016z-f29854d8472b`, from source
`334cbc9147b8914b5292137cef70660c11d258b5`. Its trial is
`s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-fp8-01a1ec55/trials/20261118T021016Z-f29854d8472b`.
Board row 2220 records its four-GPU share. The CPU coordinator was admitted and
running before any GPU child was submitted.

## 2026-11-18 02:37 UTC — Full online FP8 fails the fixed-reference screen

The first allocation was preempted during setup. The permitted retry ran on
`s5jtxs64`, native UID `6998c7c95448168a`; its pod priority was `iris-batch` after
requesting interactive. Each worker loaded 136.02 GiB in about 65.3 seconds.
Compilation and CUDA graphs stayed enabled. FlashInfer explicitly reported
BF16 query and KV dtypes. The native loaded-model inspection passed for online
FP8 linear and expert methods.

The [numerical summary](analysis/hero-fp8-01a1ec55/diagnostic-a1-check-summary.json)
fails the fixed-reference and decode screens. Short reference p99 is 0.32200,
p90 0.09737 and 14.50% exceed 0.075. Long p99 is 0.22086, p90 0.09307 and 14.76%
exceed 0.075. Maxima are 0.83593 and 0.91789. Decode/recompute maximum is 0.30268
and p99 0.29097. Batch maximum is 0.03664. Changed-prefill cache maximum is
0.08218. All ten exact cache controls are zero. The failure is not evidence of
cache-state corruption. Bounds stay fixed; this configuration is unqualified.
The short diagnostic replay can continue as labeled performance evidence.

The parent supplied cache-corrected source
`644ab8767a4426ea56e6ad435baec8dcc69824f6`; it is merged before further
qualification runs. The [cache handoff](cache-retention-handoff.md) and its
56-case result define that correction. The BF16 baseline remains the published
sixteen-GPU C64 profile at retention 128, with a normal one-hour window.

The next FP8 diagnostic changes only the native MoE backend from Triton W8A8 to
Marlin W8A16. It keeps the existing online FP8 weight method, BF16 KV, numerical
policy and four-GPU C16 load. The pinned native oracle accepts this weight scheme
on Marlin. Its expert kernel consumes BF16 activations. This tests whether
activation quantization explains the large decode/recompute tail, and whether
reference disagreement falls enough to keep pursuing this candidate. The model's
smaller linear projections still use the starting online FP8 method.

PR #127's padding fix is already an ancestor of pinned runtime `0ac9dcb`. It is
not a missing patch to reapply. Its Snowball evidence remains a useful warning,
not Hero qualification. No runtime source override has been introduced.

## 2026-11-18 02:59 UTC — Retain the short screen and a setup failure

The Triton FP8 diagnostic completed six successful profiling requests with zero
errors. Its reported output throughput is 5.87152 tokens/s/GPU over an 86.47681-
second successful-request span. The requested screen was 60 seconds. Native
protocol validity is false for this short diagnostic, and numerical acceptance
fails separately. This is not the required one-hour comparison. Its
[result](analysis/hero-fp8-01a1ec55/diagnostic-a1-result.json) retains exports,
checks, attempt archives and allocation evidence. Total cost is 2.304403 GB200-
hours, including 0.162181 for the preempted setup. The successful attempt used
2.142222 GB200-hours. Both GPU attempts are terminal; the final pod was observed
Succeeded before collection completed.

The first sixteen-GPU BF16 C64 baseline failed before numerical checks or
measurement. One host timed out fetching pinned `tokenspeed_mla==0.1.9` ARM wheel
metadata from PyPI after three retries in 130.5 seconds. Its siblings stopped as
coscheduling failures. The
[failure result](analysis/hero-fp8-01a1ec55/bf16-c64-a1-failed-result.json) retains
all four attempts and complete cost: 1.389160 GB200-hours. All four GPU pods were
gone in the provider projection. Retry the unchanged profile after another live
inventory read. No numerical or performance conclusion follows from this
package-fetch failure.

The cached workload was reused by both new coordinators. Marlin's four-GPU
server is healthy and checking fixed references with compilation and graphs
active. Its initial KV capacity is 480,540 tokens per engine; precision remains
BF16. No tolerance or checkpoint changed.
