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
