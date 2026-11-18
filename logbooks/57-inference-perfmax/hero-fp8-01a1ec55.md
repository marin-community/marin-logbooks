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

## 2026-11-18 03:04 UTC — Screen the remaining activation quantization

Marlin expert activations leave reference tails nearly unchanged: short p99 is
0.31912 versus 0.32200, and long p99 is 0.22057 versus 0.22086. The smaller
projections still quantize activations. The pinned runtime also exposes native
`linear_backend=marlin`. The next four-GPU numerical screen uses both Marlin
backends, leaving all activations BF16 while keeping the same per-tensor FP8
weight method. Fixed goldens and every numerical limit stay unchanged.

Use the existing shared correctness launcher for this screen. It runs the same
endpoint checks with graphs and compilation enabled, and retains allocation
receipts without an AgentX warmup. A candidate that passes still needs the actual
sixteen-GPU AgentX configuration and a normal one-hour window. A numerical-only
screen cannot replace that comparison. Its separate correctness-run URI and cost
will be linked in the campaign ledger.

The skill and runner README now say preparation runs in the runner's CPU
coordinator. This corrects the ambiguity behind this session's failed local
corpus preparation. The later real CLI invocations already demonstrate the
intended CPU-first workflow; no duplicate evaluation run is needed.

## 2026-11-18 03:33 UTC — Prepare native selective FP8 component checks

The runtime patch is published in
[244247767c](https://github.com/yonromai/vllm/commit/244247767cef50ef1670a732d416065015e12f01),
based on unchanged runtime `0ac9dcb`. It admits native
`--quantization online --quantization-config '{"moe":"fp8_per_tensor"}'`.
The native selector dispatches this static-tensor weight key to the same
`Fp8PerTensorOnlineMoEMethod` used by legacy `Fp8Config`, with the same per-expert
weight scales. Linear projections stay BF16. The router stays FP32 under both
config classes. Targets, explicit activation overrides and other weight schemes
remain rejected; TP1, TPU and KV restrictions remain intact. The
[patch](analysis/hero-fp8-01a1ec55/runtime-selective-fp8.patch) preserves both source
and meaningful test changes. PR115's already-merged tests are copied into this
older baseline, then extended for selective and full native configurations.

The parent inspected the narrow runtime change and authorized a one-GB200
component diagnostic with the pinned compiled base and one explicit, hash-checked
Python-module override. This is not candidate-wheel qualification. The
[input](analysis/hero-fp8-01a1ec55/selective-component-input.json) pins source,
archive, model and bootstrap hashes. It uses the native isolated runtime and
`run-wheel-tests` helper. Tests compare exact legacy expert weights/scales and
outputs, independent BF16 Hero math at one/seven tokens, loaded weight dtypes,
FP32 routing, and rejection of unsupported schemes. GPU results are pending.
A real candidate wheel and manifest remain required before full serving tests.

The Marlin-experts screen fails strict cache persistence under the corrected
checker. Replayed-decode-branch maximum is 0.0581454 despite equal prepared and
repeat hits at 384 tokens; sibling reuse is 128. Ordinary settled branches have
smaller nonzero differences too. Coverage and matched-repeat checks pass. These
remain failures. No evidence currently explains them as different prefill
geometry. The fixed-reference and decode failures also remain. Further screens
keep all exact limits unchanged.

## 2026-11-18 03:49 UTC — Keep the completed screens and fix component packaging

The sixteen-GPU BF16 C64 run passed the corrected complete suite. Every one of
its 19 exact cache comparisons is zero. Reference short/long p99 are
0.0889078/0.0557193; decode maximum is 0.0686349. Its normal one-hour AgentX
measurement is running. These numerical results do not yet establish completed
performance evidence.

The all-Marlin FP8 screen improves fixed-reference errors, but does not qualify.
Short/long p99 fall to 0.247622/0.153125. Decode passes, with BF16 activations;
reference and strict cache checks still fail. The
[summary](analysis/hero-fp8-01a1ec55/marlin-all-a1-check-summary.json),
[receipt](analysis/hero-fp8-01a1ec55/marlin-all-a1-receipt.json) and
[execution](analysis/hero-fp8-01a1ec55/marlin-all-a1-execution.json) retain the
native report and command. Its complete allocation cost is 1.207778 GB200-hours.
The Marlin-experts AgentX screen also completed: 6.48384 output tokens/s/GPU,
short diagnostic protocol invalid, reference/decode/cache failures retained,
complete cost 2.327778 GB200-hours. Its
[result](analysis/hero-fp8-01a1ec55/marlin-experts-a1-result.json) preserves native
exports and allocation evidence. Both GPU tasks were observed terminal.

Selective component attempt a1 did not run a test. My minimal source archive
omitted `tests/ci_envs.py`, imported during native pytest collection. The pinned
compiled wheel and overridden model hashes verified before collection failed.
The [failed execution](analysis/hero-fp8-01a1ec55/selective-component-a1-failed-execution.json)
retains its 84 GPU-seconds (0.023333 GB200-hours). Retry a2 includes the complete
native tests tree from the same runtime commit, rather than trimming helper
imports one by one. The runner now archives component outputs on failure as
well as success. This corrects test packaging, not runtime behavior. The runtime
source and model-module hash are unchanged.

## 2026-11-18 03:54 UTC — Correct the explicit module override

Component a2 reached pytest, then failed the first CPU guard test because my
explicit import placed `grugmoe` in `sys.modules` without attaching it to the
parent package. Native string-based monkeypatch lookup therefore could not find
the module. The bootstrap now reproduces normal import binding on the parent.
No runtime code or test assertion changed. Retry a3 keeps the same source and
archive, with the corrected bootstrap hash. Both failed attempts remain in the
cost ledger. The parent confirmed the documented native precompiled build path
for this Python-only patch; a CPU builder and hash comparison will precede cold
installation of the actual candidate wheel.

## 2026-11-18 04:03 UTC — Preserve native measurement after a numerical failure

The runner previously allowed continuation only through `--diagnostic-seconds`,
which also adds AgentX's unsafe override. That cannot supply the goal's normal
one-hour comparison when numerical checks remain unqualified. The new explicit
`--measure-failed-checks` option retains those failures while keeping the native
command, 3600-second window, count-driven warmup, cache reset and validity. Its
report says `diagnostic performance (failed checks)` separately from AgentX's
validity. Existing all-checks-passed acceptance rules still block recommendation.
The command-equivalence and recommendation regression test passes.

Experimental wheels can now live at a private durable cloud URI. The worker
materializes the current architecture once under a stable hash-based task-local
path, checks its SHA256, and records the source URI/hash without changing the
saved manifest. Normal HTTPS releases still use the native installer. Tests
cover stable identity, corrupted bytes and HTTPS. All 27 runner tests pass;
scoped required lint passes. The parent requested these exact controls.

Component a3 stopped before installation on a bootstrap hash mismatch: lint had
reordered imports after I froze that input. It cost 23 GPU-seconds. The runner
now checks the local bootstrap hash before submitting a job. All three failed
overlay preparations are retained and supply no runtime-test qualification.
Rather than allocate another overlay retry, build the actual wheel on a CPU
job and cold-install it for the focused GPU tests. The
[build input](analysis/hero-fp8-01a1ec55/selective-build-a1-input.json) pins all
source bytes and the native precompiled base. The builder uses unmodified native
`setup.py` through `python -m build --wheel --no-isolation`, then compares every
ELF member against the pinned wheel and checks the changed Python module hash.
No native binary is rebuilt. No release or adoption pin is changed.

## 2026-11-18 04:06 UTC — Native candidate wheel built on CPU

The CPU build succeeded in 98 allocated seconds on ARM host s2zpxs64, with eight
requested CPUs and no GPU requests. The actual source is `244247767c`; the
candidate version is `0.0.0.dev20261118+marin.244247767cef.cu132`. All 19 ELF
payloads match the pinned `0ac9dcb` wheel exactly. The changed model module also
matches its source hash. The wheel is 227,698,667 bytes, SHA256
`c3c2ec10f43c07606cc86a70c08885538f63480f64d5779b984947f8eb2da5a2`.
The native build emits a CPython3.12 `linux_aarch64` wheel; this experiment does
not claim a new manylinux portability qualification. The
[manifest](analysis/hero-fp8-01a1ec55/selective-build-a1-manifest.json),
[build proof](analysis/hero-fp8-01a1ec55/selective-build-a1-build.json),
[receipt](analysis/hero-fp8-01a1ec55/selective-build-a1-receipt.json) and
[execution](analysis/hero-fp8-01a1ec55/selective-build-a1-execution.json) preserve
all bytes and identities. Cold GPU installation and focused tests remain pending.

The selective full-model screen keeps Marlin experts with BF16 activations and
BF16 linear projections. Earlier all-Marlin evidence improved reference errors
and passed decode, while quantized projections and cache persistence still
failed. This chooses the native expert method and BF16 projections without
inventing another quantization scheme. Its separate policy changes only identity
and quantization applicability; all numeric criteria and fixed goldens stay the
same. The loaded-method check now requires FP8 experts and no FP8 linears for
this exact selector. Method inspection still does not read tensor values; the
actual-wheel component tests directly check weight and scale tensors.

## 2026-11-18 04:12 UTC — Actual wheel tests pass; adopt the stronger cache probe

Cold installation of the immutable candidate passed all 18 native focused tests
in 20.38 seconds on GB200, with zero skips. The imported model and package both
come from installed `site-packages`; no module override is used. Tests directly
verify exact expert weights/scales and outputs, BF16 projections in experts-only
mode, FP32 router weights, the independent Hero BF16 math at one/seven tokens,
and unsupported-scheme/platform rejection. The
[component identity](analysis/hero-fp8-01a1ec55/selective-wheel-tests-a1-components.json),
[JUnit](analysis/hero-fp8-01a1ec55/selective-wheel-tests-a1-components.xml),
[receipt](analysis/hero-fp8-01a1ec55/selective-wheel-tests-a1-receipt.json) and
[execution](analysis/hero-fp8-01a1ec55/selective-wheel-tests-a1-execution.json)
retain the results. Complete cost is 91 GPU-seconds (0.025278 GB200-hours).
The pod is Succeeded and resource row2234 is closed.

Merged the parent's stronger cache source
`dff1c8812335eb9047d6da0cf6ef279e588a71cb` before the next qualification. It
explicitly creates sibling checkpoint K and original checkpoint E using K+1/E+1
prompts, then requires sibling hit K and unchanged original hit E. The existing
BF16 C64 window and earlier FP8 reports retain their v1 probe scope. The next
selective screen and BF16 neighboring load use v2. A v1 pass does not establish
coverage of this stronger generated-history fork.

Use four GB200 GPUs for the selective numerical screen. Dense projections add
roughly a dozen GiB per GPU relative to all-FP8 at EP4, while expert weights keep
the same native per-tensor method. The prior all-FP8 deployment had 35.37 GiB of
BF16 KV space per engine. This is the smallest plausible full-context allocation;
record its actual capacity and retain any capacity failure before expanding.

## 2026-11-18 04:21 UTC — Keep physical precision distinct from its native selector

The selective plan exposed a metadata error: native `quantization: online`
would label the physical weights `online` in upstream exports. The optional
`weight_precision: fp8` field now supplies that label while retaining native
`--quantization online` and its exact selector. A boundary test verifies both;
all 28 runner tests and scoped lint pass. Prior submitted plans stay unchanged.

The full selective screen is submitted at
`s3://marin-us-east-02a/marin/inference/correctness/runs/20261118T041443Z-65a5fdd270e2`.
Its four-GPU pod is still SchedulingGated, so no allocation has begun. Requested
interactive priority became Batch as expected. Its source reports dirty because
I prepared the three future load profiles during CPU reference lookup. The
tracked tree hash `1777ab6ade` is the same as the clean dry run; no serving,
checker or submitted profile bytes changed. This commit preserves those future
profiles. The actual candidate wheel and full plan remain immutable.

The planned neighboring load is C128. Compare both precisions on sixteen GB200
GPUs with normal one-hour windows and the stronger cache probe. The existing
BF16 C64 is the lower neighboring point. A sampled C64 interval had about85%
KV use and67% actual versus95% theoretical cache hits, with no steady queue.
Extra lanes could improve utilization or worsen cache pressure. These are
hypotheses, not a measured causal explanation. Add selective C64 if its outcome
could change the recommendation. No latency ceiling is inferred.

The [cost ledger](analysis/hero-fp8-01a1ec55/cost-ledger.json) sums all completed
owned receipts, including failures and the CPU build. Its current7.318286
GB200-hours excludes the live BF16 window and queued selective screen. Native
profile timestamps split setup, checks, warmup, profiling/native grace and
exports/shutdown where available. Unsplit auxiliary jobs retain their full cost.

## 2026-11-18 04:26 UTC — Reduce an oversized RAM reservation

Kueue's exact admission reason says no node fits:69 nodes lack requested CPU,
136 lack requested memory, and one lacks GPUs. This screen requested32 CPUs and
768 GiB. The live BF16 task's cgroup peak is114,597,363,712 bytes (106.73 GiB),
with68,100,161,536 bytes current use. Keep32 CPUs and reduce the next request to
256 GiB. The parent inspected and supported this correction. Both upcoming C128
precision points use the same32 CPU/256 GiB requests.

Cancelled the screen while PodScheduled=False and nodeName was empty. The
[native execution](analysis/hero-fp8-01a1ec55/selective-fp8-marlin-a1-execution.json)
remains killed, with no assigned node or scheduling timestamp. The generic cost
helper reports null for such a timestamp; the separate
[queue-cost evidence](analysis/hero-fp8-01a1ec55/selective-fp8-marlin-a1-queue-cost.json)
retains the admission reason, before-cancel state and zero allocated GPU time.
This is a resource-fit correction, not a response to a log/observation timeout.
The candidate wheel, kernel, model, CPU count, GPU count and numerical criteria
are unchanged. Retry a2 keeps a new native identity.

## 2026-11-18 04:54 UTC — Complete the BF16 window and choose a lower neighbor

BF16 C64 completed a valid native one-hour window. The
[client summary](analysis/hero-fp8-01a1ec55/bf16-c64-a2-window-summary.json)
records submission_valid=true, no cancellation, and TTFT/ITL duration coverage
both1.0 against the required3600 seconds. It completed1841 requests and
1,096,749 output tokens over a3629.952-second successful-request span. P90 TTFT
is312.165 seconds; p90 completion is324.189 seconds. Final native processor and
allocation collection are still running on the CPU coordinator. All four GPU
pods are Succeeded.

The exporter warns about counter decreases on server metric series. The native
gang IDs stayed fixed; sampled leader EngineCore process ages exceeded6330
seconds. Do not infer restarts or trust counter-derived absolute rates without
resolving this. Independently, client usage fields give57.395% cache reads,
consistent with the observed gap to the roughly95% theoretical trace reuse.
This supports testing less cache pressure without claiming its cause is proved.

Choose BF16 C32 as the next neighboring load and selective FP8 C64 for the
matched precision comparison. Both new points request32 CPUs and256 GiB per
host on sixteen GB200 GPUs. BF16 C32 also supplies fresh positive128 cache-v2
qualification. Keep the prepared C128 profiles as a possible later expansion;
no C128 GPU job has been submitted. This change follows the completed window,
not a latency ceiling. Numerical failures may continue to a normal one-hour FP8
performance window through the explicit measure-failed-checks flag, while
remaining ineligible for recommendation.


## 2026-11-18 05:38 UTC — Completed baseline and selective failures

BF16 C64 completed a valid native one-hour window. The processor reports
18.94936 output tokens/s/GPU, 1841 profiling successes, zero profiling errors,
P90 TTFT of 312.16459 seconds and P90 completion of 324.18918 seconds. Protocol,
submission, duration coverage and gang checks all pass. Its server completion
usage totals 1,100,564 tokens. The separate harness output-token counter totals
1,096,749; the earlier notebook entry quoted that counter. Keep both meanings
distinct. One warmup request returned an empty response and failed. The window
still passes the native validity rules. The counter-decrease warning remains
unresolved; use the client export for this comparison. The
[full receipt](analysis/hero-fp8-01a1ec55/bf16-c64-a2-result.json) gives 103,852
GPU seconds including setup, checks, warmup, profiling, drain and shutdown.
All four GPU pods are Succeeded and resource row 2226 is closed.

The selective actual-wheel screen finished on cache v2. Long reference,
batch and decode pass. Short reference fails only maximum: 0.696609 exceeds
0.65, while P99 of 0.122859, P90 of 0.0330411 and fraction of 0.0259455 pass.
Exact cache comparisons fail with maximum 0.0134047 despite all reuse/branch
coverage groups passing; matched repeats are exact. These remain numerical
failures. No bound changed, and no causal explanation is established. The
[summary](analysis/hero-fp8-01a1ec55/selective-fp8-marlin-a2-check-summary.json),
[receipt](analysis/hero-fp8-01a1ec55/selective-fp8-marlin-a2-receipt.json) and
[execution](analysis/hero-fp8-01a1ec55/selective-fp8-marlin-a2-execution.json)
retain the source, wheel, raw archive and both attempts. One allocated attempt
was deleted after 49.889 seconds; its native automatic retry completed checks.
Complete cost is 4451.5559998 GPU seconds (1.23654333 GB200 hours). Both pods
are terminal and row 2237 is closed.

BF16 C32 passes the entire stronger v2 suite at retention 128, including all
19 exact comparisons equal to zero. Its normal one-hour profiling window began
at 05:32:08 UTC after native warmup. The sampled leader cgroup peak is
118,896,197,632 bytes (110.73 GiB). Its 32 CPU/256 GiB request differs from the
completed C64 baseline's 32 CPU/768 GiB request; the latter's observed peak was
106.73 GiB. This comparison covers two serving configurations, with this request
difference recorded. The FP8 C64 candidate also requests 32 CPUs/256 GiB.
New runner receipts retain each host's cgroup peak before final artifact upload.

## 2026-11-18 05:40 UTC — Actual storage check before the FP8 window

The parent identified a qualification gap: get_model_inspection reports module
and quantization-method names, without tensor dtypes. Existing inspection
records prove method selection only. The new
[worker extension](analysis/hero-fp8-01a1ec55/hero_weight_metadata.py) uses the
pinned native worker-extension/collective_rpc path. It reads metadata and never
replaces a runtime model module or copies tensor values.

Native Marlin repacks E4M3FN FP8 weights as int32 words. The check requires the
actual packed tensors to occupy one byte per logical expert weight, with the
native FP8 consumer, exact scale shapes and all 48 expert modules covered. It
also checks every dense linear projection is CUDA BF16 and every router weight
and bias is FP32. Actual typed attention KV and short-convolution state views
are checked separately as CUDA BF16; their shared backing storage sizes are
not added. Every DP rank writes metadata into its native host archive. The
pinned DP utility broadcasts to all engines, propagates any worker failure and
returns only its first result. The inspector records that precise scope.

All three future selective profiles load the extension through native
--worker-extension-cls with a campaign-only PYTHONPATH. Storage inspection
runs before numerical checks. The wheel and numerical criteria are unchanged.
The CPU regression distinguishes real one-byte FP8 packing from BF16 storage
reinterpreted as int32. It and 28 runner tests pass. Scoped lint passes. Actual
full-model storage proof will come from the new deployment, not these CPU tests
or earlier method-name records.


## 2026-11-18 05:52 UTC — Verify all loaded ranks before profiling

The primary selective FP8 C64 comparison is
`/romain/knotmeter-20261118t053943z-0eded6360d5c/serve`, with
[trial artifacts](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-fp8-01a1ec55/trials/20261118T053943Z-0eded6360d5c).
It uses clean research source `bd82567837`, native runtime `244247767c`,
the immutable candidate manifest and the normal 3600-second protocol. Only
--measure-failed-checks is set; no short diagnostic override is used. Failed
checks remain recommendation blockers. The four GPU hosts are Running at
Batch priority, as allowed. Together with BF16 C32 this uses the assigned
32 GB200 GPUs. Resource row 2240 retains the job identities.

The new native storage check passed before profiling. A separate read of every
host's saved metadata confirms all 16 DP ranks, with the identical extension
hash. Each rank holds 48 FP8 expert modules with 24 local experts each and
32,614,907,904 expert weight bytes. Their int32 packed storage occupies exactly
one byte per logical FP8 weight, and native consumers declare E4M3FN. All 820
linear projections per rank are BF16. All 48 routers per rank have FP32 weights
and biases. Actual attention KV and short-convolution state views are BF16.
The [all-rank summary](analysis/hero-fp8-01a1ec55/selective-fp8-c64-a1-weight-storage-summary.json)
records these observations. Full metadata files stay in each native host archive.
The shared numerical suite is running before warmup.

The parent caught a summary-export error: recursive filtering removed the
statistics fields. The selective and C32 summaries now preserve each raw
statistics map exactly, verified against the saved full reports. No measurement
or numerical result changed, and no GPU rerun was used for this report fix.


## 2026-11-18 06:12 UTC — Preserve main-point failures and prepare the contribution

The selected 16-GPU FP8 C64 deployment passed actual storage inspection on all
ranks. Its full shared numerical suite then failed reference and cache checks.
Short-reference maximum is 0.719764 and P99 is 0.125869, beyond unchanged bounds
of 0.65 and 0.125. Long reference passes (maximum 0.288436, P99 0.057456).
Batch and decode pass; decode maximum is 0.0500023. Exact cache comparisons
fail at maximum 0.0346905. The
[main-point summary](analysis/hero-fp8-01a1ec55/selective-fp8-c64-a1-check-summary.json)
preserves the raw statistics. Normal native warmup is running. Performance will
remain diagnostic because these checks fail.

At the parent's request, the runtime contribution now has a separate clean
worktree and branch against current Marin main `47a8e415bff17b9376f02d9b92edcc67e962eaee`:
`/home/romain/work/marin-community/vllm/hero-fp8-selective-contribution-01a1ec55`,
branch `codex/hero-fp8-selective-contribution-01a1ec55`, revision
`0474521acd8093ea4cbf884dd435f59d5ecd7d5f`. It reuses PR #115's existing tests,
adding only the new selection/support checks. The tested `244247767c` branch,
wheel and both live trials stay frozen. No new GPU run was used for this step.

The [provenance](analysis/hero-fp8-01a1ec55/runtime-contribution-provenance.json)
and [module diff](analysis/hero-fp8-01a1ec55/runtime-contribution-versus-tested-model.patch)
make the validation boundary explicit. The whole model module differs because
current main includes newer ShortConv output-buffer and graph changes. The
quantization class/helper and focused test behavior match the tested source.
CPU Ruff passes. A single advisory lint-catalog pass reported one naming issue;
its follow-up calls the test mode routed-experts-only, since shared experts stay
BF16. The review's plain output omits observed model and native child identity,
so these remain unknown. Its records are in this session's
`reviews/LINT-20261118T060400Z/`. Opening a draft PR awaits the final High review.
The [draft body](analysis/hero-fp8-01a1ec55/runtime-pr-body.md) is prepared.

The parent's bounded audit found no known correctness fix for this cache failure
in the newer ShortConv changes. PR #130's full graph mode is unused here;
PR #107's allocation/copy change has retained equality evidence; the null-row
fix is already in the tested base. These findings do not justify another GPU
run on that lead while the separate reference failure remains.

The cost ledger now separates setup, checks, warmup, the accepted-request span,
post-request native drain/export, and shutdown/upload. It uses the leader's
phase timestamps for followers only when native gang IDs match. Each allocation
is intersected with those phase times, and a conservation check confirms every
GPU second is counted exactly once. Unsplit failures retain their entire cost.
The completed BF16 C64 allocation includes 6.29349 GB200 hours of warmup and
1.32255 hours after the accepted-request span while the harness drains/exports.
Those two native activities are not timed separately. Completed owned cost
remains 37.402607 GB200 hours, excluding both live normal windows.


## 2026-11-18 06:48 UTC — Complete the lower-load BF16 comparison

BF16 C32 completed a valid normal one-hour window. The native processor reports
17.93673 output tokens/s/GPU, P90 TTFT of 166.45803 seconds and P90 completion of
171.59327 seconds. All 1513 profiling requests succeeded, with zero profiling
errors. Submission, duration coverage and unchanged-gang checks pass. Two warmup
requests returned no response content and raised InvalidInferenceResultError;
they remain preserved separately from profiling. The native window is valid.
Client-reported cache reuse is 85.237972%, independent of the unresolved server
counter-decrease warnings.

Compared with BF16 C64, throughput is 5.3439% lower, P90 TTFT is 46.6762% lower,
and P90 completion is about 47% lower. C32 is a defensible balanced choice, with
C64 retained as the throughput neighbor. The earlier 5% similarity heuristic
was an agent-chosen guide, not a user-approved acceptance cutoff. It should not
create an artificial decision boundary at this observed 5.34% difference.
There is only one normal window per point, so repeat variability is unmeasured.
No repeat is needed solely to resolve that invented boundary.

The [complete result](analysis/hero-fp8-01a1ec55/bf16-c32-a1-result.json),
[export summary](analysis/hero-fp8-01a1ec55/bf16-c32-a1-window-summary.json) and
[native receipt](analysis/hero-fp8-01a1ec55/bf16-c32-a1-native-receipt.json)
preserve the measurements and source. The 333,918,262-byte leader archive was
downloaded once and verified against SHA256
`910046785d86677c01c99809ae392293693dc0296908954dc08e0834d70ed9fb`.
Complete allocated cost is 97,348 GPU seconds (27.041111 GB200 hours).
The [memory observations](analysis/hero-fp8-01a1ec55/bf16-c32-a1-memory-observations.json)
retain all four hosts' observed cgroup peaks and 256 GiB limits. They are live
high-water marks, not final shutdown measurements.

The [identity comparison](analysis/hero-fp8-01a1ec55/bf16-c64-v2-evidence-reuse.json)
compares actual retained reports and profiles. Model, tokenizer, goldens,
policy, runtime, GPU type/count and every serving setting match exactly between
C64 and C32. Only client concurrency and the host RAM reservation differ.
All 19 exact cache-v2 comparisons on C32 are zero, and every supported reuse
and branch group passes. This supplies the stronger positive-retention evidence
for the matching C64 serving setup while retaining its original v1 report.

Iris root and serve jobs are Succeeded. All five pod names are now absent; four
GPU pods had already been removed, and the remaining terminal CPU coordinator
was deleted after evidence retention. The
[cleanup receipt](analysis/hero-fp8-01a1ec55/bf16-c32-a1-cleanup.json) and
[Iris states](analysis/hero-fp8-01a1ec55/bf16-c32-a1-iris-terminal.json) retain
proof. Row 2239 is closed. Completed owned cost is now 64.443718 GB200 hours,
including every retained failure and setup. The only live owned GPU allocation
is the frozen 16-GPU FP8 C64 window, which reaches its admission boundary near
07:25 UTC.


## 2026-11-18 07:18 UTC — Continue under the overnight amendment

The amended goal extends autonomous work to 17:00 UTC. The current normal FP8
hour stays frozen. Final High review and PR opening wait for the overnight work.
The parent owns reconciliation of the older 50.71015-token recipe and has
assigned an independent BF16 search to another child. This child keeps its
32-GB200 share and zero H100 share, with no overnight questions or contact to
Romain or LA1.

I read the older selected recipe and handoff. It used another checkpoint,
tokenizer/runtime, compact KV, retention zero, C128 and explicit deterministic
compiler settings. The current runtime does not read grug_logical_kv_cache, and
current NUMA binding is false. The parent's baseline reconciliation records
these distinctions; the old result is not an isolated regression or a promise
for the fixed current checkpoint. PR #110 remains outside this campaign.

The next cheap numerical control targets native Marlin EP filtering and
intermediate-buffer history. Existing component tests cover EP1, while the
full-model failures occurred with EP4 and EP16. The
[component script](analysis/hero-fp8-01a1ec55/marlin_history_bootstrap.py) uses the
actual installed candidate wheel, native per-tensor FP8 quantization/packing,
24 local experts out of 384, hidden/intermediate size 3072 and eight routed experts per token. It
checks both ends of the global expert map, all-local/mixed/unmapped/padding
routes, and kernel-size boundaries through 2048 tokens. Inputs and weights are
chosen so the weight quantization is exactly representable. An independent
FP32 reference checks local contributions. A→B→A reuses buffers after different
work and requires exact repeated A outputs; unmapped outputs must be zero.

This one-GPU component can reject a stale-buffer hypothesis cheaply. It cannot
qualify serving or explain cache-state failures by itself. Reference tolerances
are fixed before this component run at 0.02 absolute/relative for BF16 outputs;
exact history comparisons stay zero. No model policy or golden changes.
Mechanical lint and CPU syntax checks pass before payload freezing. The native
candidate wheel, model module and Marlin module hashes will be recorded.


## 2026-11-18 07:42 UTC — Complete the initial FP8 hour and release its allocation

The frozen selective FP8 C64 hour completed with valid native protocol,
submission, coverage and gang controls. It scored 2079 successful requests and
zero profiling errors. Output throughput is 22.62167 tokens/s/GPU, P90 TTFT is
263.60362 seconds and P90 completion is 278.60459 seconds. Against BF16 C64,
throughput is 19.3796% higher, TTFT is 15.5562% lower and completion is 14.0611%
lower. Client cache reuse is 73.929736%. The reference and exact cache checks
still fail, so this is diagnostic performance and blocks an accepted FP8 choice.

The final export/JSONL retain one empty-content warmup request with
InvalidInferenceResultError. This corrects the earlier progress-log report of
zero warmup errors. The native export's profiling count remains zero errors.
All three normal windows' empty-content warmup failures are retained separately.
Server counter decreases remain unresolved; rates and cache read fractions here
come from client exports.

The [initial comparison](analysis/hero-fp8-01a1ec55/comparison-initial.md) and
[machine-readable controls](analysis/hero-fp8-01a1ec55/comparison-initial.json)
include the BF16 neighbor tradeoff, exact source-pinned commands and qualification
limits. The [complete FP8 result](analysis/hero-fp8-01a1ec55/selective-fp8-c64-a1-result.json),
[export summary](analysis/hero-fp8-01a1ec55/selective-fp8-c64-a1-window-summary.json) and
[native receipt](analysis/hero-fp8-01a1ec55/selective-fp8-c64-a1-native-receipt.json)
retain raw archive URIs/hashes. The leader's 342,472,507-byte archive was read
back against SHA256 `a759f66170444a0b43e108f4bb45b761f1286f4844c38205b809181331ef42d7`.
Its final recorded cgroup peak is 120,561,532,928 bytes (112.28 GiB) under a
256 GiB limit. Other hosts' recorded peaks are retained in the full result.
The model uses 56.07 GiB/GPU and offers 1,514,130 BF16 KV tokens/rank.

This window cost 106,100 GPU seconds (29.472222 GB200 hours). Completed owned
cost, including the next component's setup failure, is now 93.932329 GB200 hours.
The [native job census](analysis/hero-fp8-01a1ec55/initial-owned-job-census.json)
finds all 21 owned submitted jobs terminal. The
[provider cleanup](analysis/hero-fp8-01a1ec55/initial-cleanup.json) deletes two
remaining terminal pods and verifies zero matching owned pods. All 15 resource
rows are [closed at this checkpoint](analysis/hero-fp8-01a1ec55/initial-board-closed.json).
Overnight experiments will use distinct new jobs and rows.

The first Marlin component supplied no kernel evidence. A local invocation
initially lacked repository PYTHONPATH and allocated no GPU. The corrected
submission then stopped on an import: ensure_current_vllm_config is a test
helper, not a public native export. Its [execution](analysis/hero-fp8-01a1ec55/marlin-history-a1-execution.json)
and [receipt](analysis/hero-fp8-01a1ec55/marlin-history-a1-receipt.json) preserve
59 allocated GPU seconds. The pinned native equivalent is
set_current_vllm_config(VllmConfig()), verified from source before the retry.

The parent's upstream lead, issue #52525 and Python-only PR #52532, describes
Marlin W13 differences under equivalent within-expert token orders. Its pinned
source audit finds no canonicalization here; both GEMMs already disable atomic
output addition and enable FP32 reduction. Hardware and weight-format differences
leave applicability to Hero FP8 unproven. The next cheap control will compare
full expert-region orders with separately zeroed output/workspace, alongside
A→B→A history, before justifying another full-model setup.


## 2026-11-18 07:53 UTC — Separate stale-buffer and routed-order controls

The corrected one-GPU probe uses verified native
set_current_vllm_config(VllmConfig()). A CPU preflight checks every imported
native symbol against pinned source, compiles the script, and validates the
upstream canonicalizer's active-region ordering with hostile inactive-tail data.
These are setup checks, not GPU numerical evidence.

The independent EP/history oracle keeps exactly representable weights. Its
A→B→A comparisons and unmapped/padding outputs stay strict. A separate order-only
phase uses native-quantized random BF16 weights and varied column magnitudes,
with the same per-tensor FP8 quantizer and BF16 activations. It compares ascending
and descending legal orders across each entire expert region, with independent
zeroed output, intermediates and workspace. W13 and final contributions are read
back. Identical-order repeats distinguish ordinary repeat variance from order
sensitivity. Canonicalized repeats are a paired diagnostic control.

The order phase targets concentrated hot local/mixed routing at 16, 115, 128,
129, 256, 512 and 2048 rows, with actual Hero inner dimensions 3072, local24 and
global384. This can establish or reject applicability at those shapes, scales and
routes. A negative synthetic result will not prove full-model determinism.
The canonicalizer is retained from upstream PR #52532 with its exact PR identity;
no installed runtime module is patched for this component.

The initial comparison is also saved under the campaign URI as
comparison-initial.md/json, with remote SHA readback in
[the receipt](analysis/hero-fp8-01a1ec55/comparison-initial-receipt.json).

## 2026-11-18 08:19 UTC — Retain completed controls; run order phase separately

The second component completed 64 EP/history cases before a local readback
bug stopped serialization of the first order row. The local integer `maximum`
shadowed the readback helper. All 64 A→B→A differences are exactly zero, and
unmapped/padding filtering passes. Four independent FP32-reference comparisons
remain failures at the declared `atol=0.02, rtol=0.02`: both local offsets at
2048 rows for all-local and negative-padding routing, each maximum 0.046875.
Those failures are retained without changing the bound.

Three CUDA graph controls at 1, 128 and 2048 rows each record a zero maximum
difference. The last row lacks its final status field because the old writer
flushed before assigning it. The [partial native JSON](analysis/hero-fp8-01a1ec55/marlin-order-a2-partial.json)
and [summary](analysis/hero-fp8-01a1ec55/marlin-order-a2-summary.json) preserve
that distinction. No order row survived; the order hypothesis stays pending.
The job cost 75 allocated GPU seconds and its resource row is closed.

The next one-GPU attempt reuses these completed phases by exact wheel and
native-module identities. It runs only the pending order phase. Its
[CPU preflight](analysis/hero-fp8-01a1ec55/marlin-order-a3-cpu-preflight.json)
executes the actual order-row expression with the former shadowing trigger,
round-trips its JSON, validates the canonicalizer with hostile inactive-tail
data, and verifies that the weight/input/probability initialization AST is
unchanged from the completed attempt. These checks prevent a repeat of the
serialization failure; they provide no GPU numerical evidence.

Completed owned cost is now 338,231.384 GPU seconds (93.953162 GB200 hours).

## 2026-11-18 08:27 UTC — Native FP8 order sensitivity reproduced

The order-only component completed on one GB200 (SM100, 152 multiprocessors)
using the installed selective wheel and unchanged native payloads. Eighteen of
28 legal-layout comparisons change. All 14 varied-magnitude, native-quantized
cases change at 16 through 2048 rows: maximum W13 difference 0.0009765625 and
maximum routed contribution difference 0.000030517578125. Four of 14
exactly representable cases also change, with contribution maximum 0.015625.
Identical-order and canonical-order repeats are exact throughout. The
[complete component record](analysis/hero-fp8-01a1ec55/marlin-order-a3-components.json),
[summary](analysis/hero-fp8-01a1ec55/marlin-order-a3-summary.json) and
[verified native archive](analysis/hero-fp8-01a1ec55/marlin-order-a3-receipt.json)
retain the outputs and source controls. This establishes component applicability
of the order lead to FP8/EP on GB200; full-model attribution remains open.

The separate runtime branch `codex/hero-fp8-marlin-order-01a1ec55` starts from
frozen selective source `244247767cef50ef1670a732d416065015e12f01`.
It ports upstream PR #52532 at `e8a07dcccf8e48dd4b9b42a355fc6d5b9db59073`:
the unchanged Python canonicalizer and call, plus its existing-file regression
test. The [concrete patch](analysis/hero-fp8-01a1ec55/marlin-canonical-runtime.patch)
changes no quantizer, scale layout, Grug module or native source. The
[CPU check](analysis/hero-fp8-01a1ec55/marlin-canonical-cpu-validation.json)
executes the hostile-tail regression and verifies that the helper AST matches
the successful component control. Native precompiled packaging, actual-wheel
cold installation and full shared model checks precede any new normal hour.

This component cost 80 GPU seconds. Completed owned cost is 93.975384 GB200
hours. Both component pods are gone and resource rows 2243/2244 are
[closed](analysis/hero-fp8-01a1ec55/marlin-order-a3-cleanup.json).

## 2026-11-18 08:36 UTC — Actual canonical wheel and graph path pass

Runtime `467e3fc14427012aa9114127fb41f9c8273ca80a` was packaged through
the native precompiled path on eight requested CPUs for 77 allocated seconds,
with zero GPUs. The [normalized manifest](analysis/hero-fp8-01a1ec55/marlin-canonical-build-a1-manifest.json)
pins wheel SHA256 `c4da6586c454efd499b178fc03077fa5dab9747ed79fcc8c952501a8a370d1d6`.
All 19 ELF payloads match the immutable 0ac base. The Grug module remains
unchanged; the wheel's Marlin Python module matches the ported source hash.

A separate cold-install job then exercised the actual patched fused path.
All ten FP8 cases (1, 16, 115, 128 and 2048 rows, local/mixed EP routing)
match separately buffered canonical lower-level contributions exactly and
repeat exactly. Fresh CUDA graphs of this installed fused path at 128 and
2048 rows match eager output exactly. The same current tensors still show
the old legal-order difference when sent directly to the unchanged lower-level
kernel. The hostile inactive-tail regression also passes on the installed
helper. [Native outputs](analysis/hero-fp8-01a1ec55/marlin-canonical-component-a1-components.json)
and the [archive receipt](analysis/hero-fp8-01a1ec55/marlin-canonical-component-a1-receipt.json)
retain the evidence. No installed source overlay was used.

The cold component cost 87 GPU seconds. Its pod and the CPU build pod are
deleted; resource rows 2245/2246 are closed. Completed owned cost is now
93.999551 GB200 hours. All earlier full-model and four component-reference
failures remain failures.

The next four-GPU deployment uses a
[separate campaign node](../../tests/cluster/vllm/test_hero_fp8_canonical.py)
and [profile](analysis/hero-fp8-01a1ec55/selective-fp8-canonical-4gpu.yaml).
It reuses the native correctness deployment, checks actual loaded storage,
then runs the unchanged shared cache functions at the known failing
retention128/PIECEWISE/2048 geometry. A cache failure stops before the longer
suite. A pass proceeds to the full shared suite in the same deployment.
Only the inactive logical-KV argument is removed in this new profile;
historical profiles stay source-pinned. CPU collection and required lint pass.

## 2026-11-18 08:56 UTC — Full-model admission and separate contribution

The cache-first model diagnostic is
`/romain/knotmeter-20261118t083746z-cc0ba58c46ea`, with durable native evidence
under `s3://marin-us-east-02a/marin/inference/correctness/runs/20261118T083746Z-cc0ba58c46ea`.
It queued with no assigned node, then admitted four GB200s on `s2zpxs64`
at normal Batch priority. The original Interactive request is preserved.
Model loading is healthy; the cache and full-suite outcomes remain pending.

The [native plan](analysis/hero-fp8-01a1ec55/marlin-canonical-model-a1-submitted-plan.json)
keeps `dirty=true`. The sole untracked file at source capture was the new
provider-before observation JSON. The [source audit](analysis/hero-fp8-01a1ec55/marlin-canonical-model-a1-source-audit.json)
proves every tracked source, test, profile and policy matched published
`38908a22a2`; its file blobs are retained. This metadata flag is not rewritten.
Future before-state snapshots will stay in ignored scratch until source capture.

During the genuine queue wait, a separate current-main contribution was
prepared on `codex/hero-fp8-marlin-canonical-contribution-01a1ec55` at
`b0d1a26b6e365fb9fc5ee1f991ac064a477e855b`, based on freshly fetched
`47a8e415bff17b9376f02d9b92edcc67e962eaee`. That base lacks canonicalization.
The required read-only advisory catalog completed once and found one prose
issue: native aligners currently sentinel-fill the inactive tail. Only the added
docstring and comment were clarified. The test file remains byte-identical to
tested runtime `467e3fc`; the entire Marlin executable AST matches after excluding
docstrings. CPU regression and required lint pass. [Validation](analysis/hero-fp8-01a1ec55/marlin-canonical-contribution-validation.json)
keeps whole-current-main GPU qualification explicitly absent.

[Review evidence](analysis/hero-fp8-01a1ec55/marlin-canonical-contribution-review.json)
identifies all seven exact native sessions by user-prompt hashes. Each observed
model is `claude-opus-5-5`; requested effort was xhigh, while observed effort is
not recorded. Tool actions were inspected for reads and source state remained
unchanged apart from the author's prose fix. The earlier selective contribution's
six native sessions were recovered the same way, correcting its prior unknown
model/ID fields without rerunning review or changing `04745`. That
[dated readback](analysis/hero-fp8-01a1ec55/selective-contribution-review-native-readback.json)
preserves the original record. Both contributions await the parent's final High
GOAL review; no PR is opened.

## 2026-11-18 09:20 UTC — Canonical port does not resolve the model failure

The [cache-first model screen](analysis/hero-fp8-01a1ec55/marlin-canonical-model-a1-summary.json)
fails unchanged exact bounds. Five of 19 strict comparisons are nonzero:
maximum 0.0184222980 and p99 0.0170459551. All storage checks, deployment
identity and supported reuse/branch coverage pass. Changed-split comparisons
pass, maximum 0.0274308726; the matched repeat is zero. The later reference,
batch and decode suites were skipped after this rejection. This is not a
full-suite pass or a proof of the serving failure's cause.

The [raw cache report](analysis/hero-fp8-01a1ec55/marlin-canonical-model-a1-cache-screen.json)
and [native archive receipt](analysis/hero-fp8-01a1ec55/marlin-canonical-model-a1-receipt.json)
preserve all responses and the five failures. The archive SHA256 is
`13d0ad6dd7dfb11c1c5b2c1de882fb6548d24adc9a9ce48e3e10e0d398eb30f2`.
All four ranks' actual storage metadata passes. Model memory is 147.25 GiB/GPU,
with 328,295–328,310 BF16 KV tokens/rank. Host peak before teardown/upload is
92.26 GiB under 256 GiB. This attempt cost 2,604 allocated GPU seconds
(0.723333 GB200 hours). Its pod is gone and board row 2247 is
[closed](analysis/hero-fp8-01a1ec55/marlin-canonical-model-a1-cleanup.json).
Completed owned cost is now 94.722884 GB200 hours.

The next cheap hypothesis is zero-weight padding geometry. Native Grug routing
zeros padded hidden rows and combine weights, but computes their expert IDs
from the supplied router logits. Those IDs may still change block layout.
The [new component](analysis/hero-fp8-01a1ec55/marlin_padding_bootstrap.py)
uses the existing immutable wheel, actual Grug router and native forward
context. It asserts identical real inputs/routes and zero padded inputs/weights,
then varies only padded expert IDs. It compares exact real-row outputs,
same-route repeats, restored A→B→A outputs and fresh raw CUDA graphs. Local
expert counts 24 and 96 cover the selected 16-GPU and current four-GPU shapes.
Any negative result remains limited to those weights, shapes and routes.
The [CPU readback preflight](analysis/hero-fp8-01a1ec55/marlin-padding-a1-cpu-preflight.json)
and required lint pass before allocation.

A [post-hoc comparison](analysis/hero-fp8-01a1ec55/initial-reference-precision-comparison.json)
also reuses the original BF16/FP8 reports. At book04 position1016, the BF16
reference error is 0.1228700 and FP8 is 0.7197637; their direct probability
difference is 0.5971223. This is distinct from the cache persistence failure.
It is a comparison of the two serving configurations and their saved top-k
probability bounds, not KL or a new numerical acceptance policy.

## 2026-11-18 09:35 UTC — Zero-weight padding routes affect real outputs

The [actual-router padding component](analysis/hero-fp8-01a1ec55/marlin-padding-a1-components.json)
completed in 84 allocated GPU seconds. Twenty of 28 cases change real-row MoE
outputs when only padded expert IDs change, despite exactly zero hidden rows
and combine weights for padding. The maximum raw MoE output change is
0.00006103515625. All same-route and restored A→B→A comparisons are zero.
Sixteen fresh CUDA graph cases retain that sensitivity, with exact same-route
and restored controls. Separately zeroed lower-level buffers reproduce W13
and routed-contribution differences. This isolates a component layout effect;
it does not establish the full-model failure's cause. [Summary](analysis/hero-fp8-01a1ec55/marlin-padding-a1-summary.json)
and [native receipt](analysis/hero-fp8-01a1ec55/marlin-padding-a1-receipt.json)
retain both local-expert geometries (24 and 96). The pod is deleted and board
row 2251 is closed. Completed owned cost is 94.746218 GB200 hours.

The [PR127 source check](analysis/hero-fp8-01a1ec55/grug-fp8-pr127-identity.json)
confirms that its hidden-row/weight masking is already merged into both the
tested base and current main. The new source therefore extends that existing
path. Runtime `a399f9a0f1582897a44cb4bac690221004516125` is a separate
467-based experiment: two model lines clear padded router logits out of place
before top-k. Real-row logits and the caller's tensor are preserved. The
regression requires stable valid padded IDs, unchanged real routes/weights,
and zero padded weights; it does not prescribe a particular padding expert set.
The [patch](analysis/hero-fp8-01a1ec55/padding-router-runtime.patch) and
[CPU old-fail/new-pass check](analysis/hero-fp8-01a1ec55/padding-router-cpu-validation.json)
are retained. This CPU check supplies the context mask through a small facade;
actual native helper, wheel and graph integration remain pending.

The model module SHA256 is now
`65e35957606714c57fcc3a19b5ba104b268d3b48d3cbad4f9b849586395748ac`;
the canonical Marlin module stays byte-identical. No weight method, scale layout,
router storage precision or KV precision changes. Native CPU packaging will
verify both Python modules and all 19 pinned native payload hashes before
cold component and cache-first model checks. No new hour is launched.

## 2026-11-18 09:50 UTC — Actual padding wheel closes the component failure

The native CPU build succeeded in 99 allocated seconds on eight requested CPUs,
with zero GPUs. Its [manifest](analysis/hero-fp8-01a1ec55/padding-router-build-a1-manifest.json)
pins actual wheel SHA256
`453138ae87b639ef7cac30069aeaaf6a8963cc9f6ae5ae7af896027e113a21ef`.
All 19 ELF payloads equal the original 0ac base, and the built Python model and
Marlin hashes match source.

The separate cold-installed-wheel job then passed all 19 focused native tests,
with no failures, errors or skips. All 28 paired padding cases and 16 raw CUDA
graph controls are now exact zero; padded IDs are stable, and same/restored
routes remain exact. [Raw component outputs](analysis/hero-fp8-01a1ec55/padding-fixed-a1-components.json),
[native JUnit](analysis/hero-fp8-01a1ec55/padding-fixed-a1-components.xml) and
[summary](analysis/hero-fp8-01a1ec55/padding-fixed-a1-summary.json) retain the
proof. This fixes the measured component behavior. Full-model attribution
and numerical qualification remain open.

The component cost 117 allocated GPU seconds at Batch priority. Its pod and
the CPU build pod are released; board rows 2252/2253 are closed. Completed
owned cost is 94.778718 GB200 hours. All previous failures remain retained.

The next model screen uses
[the new profile](analysis/hero-fp8-01a1ec55/selective-fp8-padding-4gpu.yaml),
changing only the tested runtime from the preceding canonical screen. The
padding flag's effective-setting basis is explicit: pinned
`VLLM_MOE_SKIP_PADDING` defaults to true, and neither the profile nor Marin
launchers override it. The component observed true before its native pytest;
that test also sets it explicitly. For serving this is a source/default
inference, not a separately sampled per-worker value. The same cache-first
and full-shared gates run before any normal hour.

The padding contribution is separate from the canonical and selective branches.
Its current-main revision `139b7521151c580f2e524691b53be00127992e64` adds only
the two model lines and regression. Its router/test ASTs match candidate `a399`;
broader current-main GPU behavior is not claimed. The required advisory catalog
completed once with no findings. The parent's final High GOAL review remains
pending, and no PR is opened.

## 2026-11-18 09:59 UTC — Matching padding model gate is live

The new job is `/romain/knotmeter-20261118t095444z-639ba7d4875a`, with native
artifacts under `s3://marin-us-east-02a/marin/inference/correctness/runs/20261118T095444Z-639ba7d4875a`.
It admitted four GB200s at Batch on `s3jtxs64`, attempt
`b34b842f195d7635`. The [actual source plan](analysis/hero-fp8-01a1ec55/padding-model-a1-submitted-plan.json)
is clean published `03d1f1a63e`, tree `a5f4e8e023`; before-state observations
stayed in ignored scratch until after source capture. Source/code controls and
all numerical bounds match the preceding screen except the declared runtime
padding change. The [before-record](analysis/hero-fp8-01a1ec55/padding-model-a1-before.json)
retains the padding flag's source/default inference. Numerical outcomes remain
pending; no AgentX hour is running.

The standalone padding [advisory record](analysis/hero-fp8-01a1ec55/padding-contribution-review.json)
retains six exact native sessions. Every observed main model is
`claude-opus-5-5`; requested effort is xhigh and observed effort is absent.
The inspected actions are source/Git reads, with no tests or source changes.
No findings were reported. The broader final High GOAL review remains separate.

## 2026-11-18 10:22 UTC — Persistence passes; reference still rejects FP8

The actual a399 wheel passes both the focused and full-suite cache screens:
all 19 exact comparisons are zero in each. All supported reuse and branch
coverage passes. Full-suite changed-split maximum is 0.0452303596. Batch
(maximum 0.0035133274) and decode (0.0728239648) pass. The fixed reference still
rejects short histories: maximum 0.6728747848 exceeds the unchanged provisional
0.65, while p99 0.1222981009 passes 0.125. The worst position is book03/769.
Long histories pass, maximum 0.2777403530. No acceptance bound is changed.
This resolves the previously measured cache failure in this deployment, not
all FP8 numerical differences or RL qualification.

[Summary with statistics](analysis/hero-fp8-01a1ec55/padding-model-a1-summary.json),
[raw first cache report](analysis/hero-fp8-01a1ec55/padding-model-a1-cache-screen.json)
and [native receipt](analysis/hero-fp8-01a1ec55/padding-model-a1-receipt.json)
retain both outcomes. Archive SHA256 is
`85ae22048fa6a64ae92791e96059a2f76bf7493d2bdabbabe5abd425d7552cf1`;
its full `report.json`, every response and server logs remain native evidence.
All four actual storage records pass. Host peak before teardown is 94.13 GiB
under 256 GiB. This attempt cost 5,084 allocated GPU seconds (1.412222 hours).
Its pod is absent and board row2254 is closed. Earlier canonical and original
failures remain unchanged.

The next hypothesis uses native online quantization's ignore list. Keep the
first12 and last12 routed-expert layers BF16; quantize the middle24 with the
same per-tensor FP8 weight method. Half the expert layers gives meaningful
memory savings (about15.2 GiB/GPU at16 GPUs, or30.4 GiB at eight, before scales), unlike a token FP8
fraction. This is a provisional layer choice protecting early routing and late
scores, not a demonstrated optimum. Global Marlin rejects unquantized MoE;
Triton natively accepts both. Its FP8 path also dynamically quantizes activations,
which is a deliberate change from Marlin W8A16 and will be recorded separately
from BF16 KV. A small actual-wheel loader/output/graph component must pass
before a model screen. The unchanged full shared gates precede any new hour.

## 2026-11-18 10:33 UTC — Mixed configuration preparation

The first one-GPU mixed component stopped before constructing layers: the tiny
Hero configuration requires `num_shared_experts > 0`. This was a setup mistake,
not kernel or numerical evidence. It cost68 GPU seconds. Its native log/receipt,
failed execution and [cleanup](analysis/hero-fp8-01a1ec55/triton-mixed-a1-failure.json)
are retained; pod absent and row2256 closed. The corrected configuration uses the
same shared-expert values as the existing native Hero regression. A
[CPU preflight](analysis/hero-fp8-01a1ec55/triton-mixed-a2-cpu-preflight.json)
executes the exact pinned config class, native tiny helper and actual bootstrap
expression without CUDA imports. It reproduces the old rejection and accepts
the corrected configuration. Required lint passes before retry.

The [prepared eight-GPU model profile](analysis/hero-fp8-01a1ec55/mixed-fp8-triton-8gpu.yaml)
keeps the same source wheel, model, goldens, bounds, graphs and retention128.
Native Triton is the declared backend change. Exactly24 explicit ignore patterns
keep outer expert layers BF16. New metadata requires the expected native method,
actual float8/BF16 tensor dtypes, shape, payload/storage bytes, scales and W8A8
consumer metadata for every layer. It separately checks BF16 projections, FP32
routers and typed BF16 attention/ShortConv cache views. Full per-rank records
remain native artifacts. The campaign cache-first test now takes its inspector
from the profile's existing serving check; shared checking code is untouched.
No model allocation or performance hour is authorized by a component setup exit.

Correction to the first estimate:24 FP8 layers save about15.2 GiB/GPU at16 ranks,
or30.4 GiB at eight, based on measured full-FP8 expert payload bytes. The earlier
30 GiB-at16 estimate counted the whole expert saving rather than half.

## 2026-11-18 10:37 UTC — Match native distributed initialization order

The second mixed component failed during `initialize_model_parallel`, before
layer construction: the dtype-only test model facade was assigned too early
and lacks `is_moe`. The pinned native fixture initializes parallel groups with
the default model config, then the native layer test assigns its dtype facade.
The retry now follows that same order. The [failure record](analysis/hero-fp8-01a1ec55/triton-mixed-a2-failure.json)
and native log preserve64 allocated GPU seconds, zero kernel evidence, absent
pod and closed row2257. The previous CPU construction check covered the runtime
config class, not distributed context initialization; its narrower claim remains.

The eight-GPU model dry run resolves the explicit24/24 selector and immutable
a399 manifest without submission. No full-model mixed result is claimed.
The comparison label will be **24 of48 expert layers in FP8**, with native
W8A8 activations on those layers, BF16 on the remaining expert layers and KV.
