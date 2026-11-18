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
