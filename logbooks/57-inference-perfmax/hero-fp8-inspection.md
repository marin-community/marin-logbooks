# Hero campaign execution inspection

The campaign is still running. The matched FP8 window, final choice and cleanup
remain pending. This record covers actions inspected so far.

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
The corresponding HTTP/report tests and 28 latest runner cases pass.

The completed BF16 C64 result has valid native measurement controls, zero
profiling errors and one retained warmup error. Its numerical report uses cache
probe v1. The BF16 C32 trial now supplies passing v2 coverage on the same model,
runtime and serving settings, including all 19 exact controls. Its lower RAM
limit and benchmark load remain explicit when reusing that evidence.
Its rate is 18.94936 output
tokens/s/GPU, with p90 TTFT 312.16459 seconds and completion 324.18918 seconds.
Complete allocation cost is 28.847778 GB200-hours. The parent read the durable
result and original client export. Client cache accounting independently shows
57.395% reuse against 95.331% theoretical reuse. Server counter decreases remain
unexplained; they are not evidence of process restarts. The lower C32 neighbor
tests a capacity hypothesis. It does not establish the cause of the cache gap.

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
Each rank reports 48 FP8 expert layers with one byte per logical weight packed
into INT32 words, an E4M3FN consumer, 820 BF16 projections and 48 FP32 routers.
Attention and short-conv cache views are BF16. The
[all-rank summary](analysis/hero-fp8-01a1ec55/selective-fp8-c64-a1-weight-storage-summary.json)
retains the extension hash and rank coverage. A real-storage test rejects a BF16
tensor reinterpreted as INT32. The short skill now explicitly requires loaded
storage evidence and separate KV precision; it remains 85 lines.
