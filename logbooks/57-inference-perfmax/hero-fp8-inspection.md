# Hero campaign execution inspection

The initial BF16/FP8 comparison is complete. The overnight BF16 search and
targeted FP8 investigation continue; final selection and review remain pending.
This record covers actions inspected so far.

The child started without the parent's implementation conversation. Its native
session is `01a1ec55-0bb6-7cf0-b8d5-f7c917b7a913`. The parent inspected actual
native commands, frozen inputs, runtime changes, component results and the
retained BF16 client export. The
[campaign notebook](hero-fp8-01a1ec55.md) contains reproducible commands and receipts.

The child read the skill, prior handoff, policies and pinned source before GPU
submission. It checked relevant vLLM PRs and distinguished layer tests from
full-model qualification. It kept the checkpoint, tokenizer and goldens fixed,
chose a separate provisional FP8 policy before measurement, and retained rejected
screens. The first tests used four GPUs. The all-Marlin numerical screen reused
the correctness launcher and avoided another AgentX warmup.

Local corpus preparation was a poor placement choice. The processor reached
about 6 GiB RSS while devbox had about 4.3 GiB available and full swap. The owned
process was stopped, with exit 143 and no GPU allocation. This does not establish
an OOM kill. The guide now directs uncached preparation through the existing
32-GB CPU coordinator before GPU allocation.

The module-overlay route became a small detour. Three attempts supplied no
runtime qualification: missing test helpers, incomplete import binding and a
stale bootstrap hash. Their 230 GPU-seconds remain charged. The child then used
native precompiled packaging on CPU. Every one of 19 native payloads matched the
pinned base, and the actual installed wheel passed
[18 focused tests](analysis/hero-fp8-01a1ec55/selective-wheel-tests-a1-components.xml).
The contribution README now explains that build path and actual-wheel validation.
The campaign wrappers remain research artifacts.

The child found two runner defects: failed numerical checks could continue only
under a short unsafe protocol, and private experimental wheel URLs could not be
installed directly. The promoted fixes preserve a normal one-hour measurement
after explicit diagnostic authorization, keep failed checks blocking acceptance,
and verify task-local wheel bytes while retaining their durable identity.
The native selector `online` also has a separate physical precision label.
The 68 combined checker, serving and runner cases pass after the review fixes below.

The completed BF16 C64 result has valid native measurement controls, zero
profiling errors and one retained warmup error. Its numerical report uses cache
probe v1. The BF16 C32 trial now supplies passing v2 coverage on the same model,
runtime and serving settings, including all 19 exact controls. Its lower RAM
limit and benchmark load remain explicit when reusing that evidence.
The completed C64 rate is 18.94936 output tokens/s/GPU, with p90 TTFT
312.16459 seconds and completion 324.18918 seconds. Its complete allocation cost
is 28.847778 GB200-hours. The parent read the durable
result and original client export. Client cache accounting independently shows
57.395% reuse against 95.331% theoretical reuse. Server counter decreases remain
unexplained; they are not evidence of process restarts. The lower C32 neighbor
tests a capacity hypothesis. It does not establish the cause of the cache gap.

The [C32 neighbor](analysis/hero-fp8-cache-validation/hero-bf16-c32-parent-inspection.json)
is now complete: 17.93673 output tokens/s/GPU, p90 TTFT 166.45803 seconds and
completion 171.59327 seconds. It gives up 5.3439% throughput while nearly halving
latency. Native controls pass with 1,513 profiling successes and no profiling
errors; two warmup empty-content errors remain recorded. The parent compared
the original native export and numerical report with the retained summary.
The 5% similarity heuristic is not a user-approved cutoff. C32 is a defensible
balanced choice, with C64 retained as the higher-throughput neighbor. Neither
point has a repeat that quantifies measurement noise.

Reducing the future RAM request from 768 to 256 GiB follows a measured BF16 peak
near 107 GiB. Both primary points are configured for 16 GB200s and 32 CPUs. Preserve
the candidate's peak and the allocation difference when reusing C64. The result
compares serving configurations; it is not a precision-only ablation. A queued
screen was cancelled after an admission failure. Its retry later received a
native PodDeleted failure after allocation; that attempt must also be charged.

Parent inspection found that the existing full-model weight check reports method
names without tensor dtypes. Earlier reports therefore establish method selection.
The [native worker-extension query](analysis/hero-fp8-01a1ec55/hero_weight_metadata.py)
now verifies actual storage on all 16 DP ranks before the main FP8 window.
Each rank reports 48 routed expert layers with one byte per FP8 weight packed
into INT32 words, an E4M3FN consumer, 820 BF16 projections and 48 FP32 routers.
Attention and short-conv cache views are BF16. The
[all-rank summary](analysis/hero-fp8-01a1ec55/selective-fp8-c64-a1-weight-storage-summary.json)
retains the extension hash and rank coverage. A real-storage test rejects a BF16
tensor reinterpreted as INT32. The short skill now explicitly requires loaded
storage evidence and separate KV precision. A cold reader prompted an AgentX
definition and links to operational skills. The guide now has 89 lines,
including the later requirement to audit prior baselines.

Before the overnight amendment, a bounded fork audit found no applicable newer
correctness fix for the observed failure.
[PR130](https://github.com/marin-community/vllm/pull/130) enables an unused full-graph
mode and adds a BF16 Triton table. [PR107](https://github.com/marin-community/vllm/pull/107)
changes allocation/copying, with retained output/cache equality evidence. The
null-row correctness fix, PR126, is already in the tested base. These findings
gave little support for another GPU run on those changes while the reference
failure also remained. That audit did not cover the later upstream Marlin lead.

The prepared runtime contribution `0474521acd` applies the same quantization
logic to current main `47a8e415bf`. Its whole module differs from tested `2442477`
because main includes newer ShortConv/full-graph support. Current-main full-model
qualification remains pending. The branch's required advisory catalog review
completed once; it clarified the routed-expert-only test label. The final High
goal review is still pending.

The contribution's [advisory pass](analysis/hero-fp8-cache-validation/marin-extension-advisory.json)
completed once. Small fixes align recorded correctness-launch limits with actual
submission, reuse constants and correct docstrings. Existing lifecycle routines
and JSON-boundary records remain; the disposition explains each finding.

Four outstanding [PR comments](analysis/hero-fp8-cache-validation/pr-feedback-resolutions.json)
also led to CPU-only fixes. Concrete golden bindings moved to experiment files
with identical bytes and successful real lookups. External numerical commands
and evidence now affect comparison identity. Rejected GPU submissions retain a
recoverable result; uncertain admission retains unknown cost. Named runtime flags
replace the preset selection. All 68 focused cases and required lint/types pass.
Running trials and numerical bounds were unchanged.

The [completed FP8 hour](analysis/hero-fp8-cache-validation/hero-fp8-c64-parent-inspection.json)
measures 22.62167 output tokens/s/GPU, 19.38% above BF16 C64. Its p90 TTFT is
263.60362 seconds and completion is 278.60459 seconds. All native validity
checks pass, with 2,079 profiling successes and zero profiling errors. The parent
read the original client export, JSONL and numerical report. The final JSONL
contains one warmup empty-content error despite an earlier progress log showing
zero. All four recorded task-memory peaks stay below the 256-GiB limit; the
leader peaks near 112.28 GiB before final artifact upload. Total cost is
29.472222 GB200-hours.

Short-reference maximum/P99 and exact cache replay still fail. Coverage of
reuse and branches passes. The [real-result report check](analysis/hero-fp8-cache-validation/real-fp8-rejection.json)
uses the unchanged trial with every non-numerical acceptance condition passing.
It renders a diagnostic report and rejects explicit selection. No accepted
profile or decision is written. The
[reproduction script](analysis/hero-fp8-cache-validation/check_real_recommendation_blocking.py)
takes the retained result, a new local campaign directory and a receipt path.
The numerical failure remains distinct from valid performance measurement.

A second fresh-context child, `01a1ed83-c2a5-74b1-b98d-b48055085778`, now owns
BF16 configuration experiments under a separate 32-GB200 ceiling. The parent
inspected its actual guide, old-recipe and source reads, plus isolated worktree
creation. Two historical fixture aliases failed in CPU preparation with zero
GPU allocation. Contribution `fe89246124` now catches this during plan resolution;
the current preset plans successfully. Its 34 affected runner cases pass.
Both corrected retention-zero trials now run the shared checks. A zero-GPU
NUMA dependency probe passes without changing shared packages or images.

The FP8 child separately probes expert mapping, workspace history and legal
token-order changes. Its first component attempt stopped on an import error,
charging 59 GPU-seconds without kernel evidence. Parent research found upstream
[issue #52525](https://github.com/vllm-project/vllm/issues/52525) and
[PR #52532](https://github.com/vllm-project/vllm/pull/52532): Marlin can change
results under equivalent within-expert layouts. Pinned `0ac9dcb` passes alignment
output directly onward and already uses FP32 reduction without atomic addition.
The upstream evidence uses other shapes and quantization formats. It is a lead
for a cheap control, not a demonstrated cause of Hero's mismatch.

The [order control](analysis/hero-fp8-01a1ec55/marlin-order-a3-summary.json)
now finds different outputs for equivalent token layouts in 18 of 28 cases.
All 14 varied-weight cases are sensitive; their largest final contribution
difference is 0.000030517578125. This establishes component order sensitivity,
not its contribution to the full-model failure. The earlier component attempt
also retains four failed FP32-reference comparisons at 2% bounds. The next test
does not erase those failures or change their bounds.

The child ported the upstream canonicalization to a separate runtime, `467e3fc144`,
leaving the original `244247767c` frozen. A CPU build preserved all 19 native
payload hashes. The [cold installed-wheel check](analysis/hero-fp8-01a1ec55/marlin-canonical-component-a1-components.json)
passes ten native FP8 cases, two CUDA-graph cases and the inactive-tail regression.
The parent inspected these records and the full-model test node. It first checks
loaded storage and cache replay, then runs the unchanged shared suite only after
those pass. This avoids another setup if the cache result is promising. The
four-GB200 screen used this sequence; component success does not qualify its model.

That [screen](analysis/hero-fp8-cache-validation/hero-fp8-canonical-parent-inspection.json)
fails five of 19 strict comparisons, with maximum probability
error 0.0184222980. Storage, deployment and all reuse/branch coverage pass.
Matched repetition is exact; changed-split checks pass. The test correctly stops
before reference, batch and decode after this rejection. Its complete cost is
0.723333 GB200-hours. Canonicalization therefore has component evidence but does
not resolve Hero's full-model failure. A one-GPU padded-route control followed.

That control found real-output changes in 20 of 28 cases when only padded routes
changed, including CUDA graphs. Same-route and restored A→B→A controls were exact.
The child masked padded router logits while preserving real rows.
The [follow-up](analysis/hero-fp8-cache-validation/hero-fp8-canonical-parent-inspection.json)
uses a cold-installed `a399f9a0f1` wheel: 19 focused tests pass without skips, and
all 28 paired cases plus 16 graphs now have stable padded IDs and zero output
differences. The parent inspected the patch, native JSON and pytest XML.

The same four-GPU model then passes deployment, storage, batch, cache and decode.
All 19 exact cache differences are zero. Fixed-reference short maximum remains
0.6728747848 against 0.65; P99 0.1222981 passes 0.125, and long reference passes.
The parent read the original full report and retained its hash. Complete cost is
1.412222 GB200-hours. This is a useful cache correction with a remaining reference
failure; it does not qualify FP8 or provide performance evidence for the fixed
runtime. The child next chose native ignored-layer controls with meaningful FP8
coverage, keeping every numerical bound unchanged.
Changing the native backend can also change activation precision. The short
guide now requires it to be recorded alongside weight storage and KV precision.

The second child also exposed a misleading successful CLI exit after failed CPU
preparation. Contribution `3a1a01b2ea` makes synchronous run and recovery preserve
the failed report and return failure. Its regression exercises the actual recovery
command after rejected submission; the 34 affected runner cases pass. Native
measurement validity, numerical status and accepted selection remain separate.

Both [retention-zero BF16 hours](analysis/hero-fp8-cache-validation/hero-bf16-retention0-parent-inspection.json)
pass the full suite and all 19 exact cache controls. The parent streamed both
original client JSONLs, checked report hashes and matched the exported latencies.
C32 reaches 19.29162 output tokens/s/GPU at p90 TTFT 145.36914 seconds; C64 reaches
23.60952 at 276.38157 seconds. Client reuse is 94.1941% and 81.6861%, respectively.
C64 also reaches 99.904% reported peak KV use. These observations support a
larger-capacity allocation test without proving cache capacity is the sole cause.
The child chose DP/EP32 at C128, preserving offered sessions per GPU. NUMA remains
prepared. It avoided spending an hour on both choices before seeing these data.
