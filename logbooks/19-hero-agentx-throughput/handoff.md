# Hero AgentX performance handoff

Use BF16 Hero-535B on sixteen GB200 GPUs, with the qualified logical KV cache and released Router. [The confirmed full hour](results/guarded-retention-agentx-c128-r1-comparison.json) gives **811.36246 output tokens/s total, 50.71015 per GPU**, with 4,545 scored successes and zero scored errors. C128 means 128 concurrent live conversation trees across the group; each tree can issue several HTTP subrequests. All 393 pinned source sessions remain eligible. The native send window is 3,600 seconds. The pinned processor’s first-request-start to last-completion denominator is 3,628.39070 seconds.

[The recipe and launch command](owner-integration/hero-logical-kv-recipe.md), [exact selection record](owner-integration/hero-selected-recipe.json) and [completion/scoring commands](owner-integration/hero-logical-kv-completion.md) are ready for the benchmark owner. The executed manifest pins guarded vLLM source `a0be4d10f1`, private ARM wheel `23dd9d42018c`, the BF16 checkpoint, tokenizer, dependency constraints, AgentX 1.0.12 and corpus. Use TP1/PP1/DP16/EP16, prefix caching, prefix-state retention interval 0, a 2048-token scheduler budget, 64 sequences per rank, GPU-local NUMA binding and memory fraction 0.95. Keep native ShortConv graphs NONE, compilation mode 3 with the recorded compiler settings, FlashInfer attention and Triton experts. Interval 0 retains sliding-window attention and ShortConv state at replay boundaries and shared branch junctions, without adding periodic interior checkpoints. This prefix-state policy is separate from the frozen model-weight checkpoint. No shared defaults or owner worktrees change.

## Measured explanation

The starting recipe inherited disabled prefix caching and a 512-token scheduling budget from a context-fit test. C8/C16/C32 supplied only 8/16/32 live conversation trees to sixteen GPUs. Its long multi-turn inputs averaged roughly 70K tokens at C32; repeated history required fresh prefill. The qualified cache/routing path reuses that history. More offered load and a larger scheduling budget also change the work available per step. These contributions cannot all be assigned one combined speedup ratio.

A C128 comparison separates the combined prefix-cache/2048-budget change before later NUMA and memory tuning: 16.952 to 29.124 output tokens/s/GPU. Subsequent combined CPU placement, memory and routing settings reach 33.267/GPU. Those changes do not isolate a NUMA-only gain. With a matched compiler profile, storing six distinct global-attention KV heads once instead of repeating them to twelve stored heads raises runtime token capacity from 1,118,056 to 2,095,445 per rank. The corresponding C128 hours measure 34.531/GPU with the flag off and 46.789 on, with API cache reads rising from 33.49% to 65.74%. Differing neighboring activity limits attribution of the full observed 35.50% ratio. The final guarded cache-on hour confirms 50.710/GPU under another run’s conditions; it is not an additional algorithmic gain over the earlier cache-on hour.

Full-model state and source checks preserve exact token IDs and top20 logprobs at matching recorded global batch shapes. Native grouped-decode tests also match an independent FP32 reference through context 262144 at the original tolerances. [The qualification record](results/guarded-logical-cache-policies-r1-qualified.json) names those checks and their raw readbacks. Precision, checkpoint architecture, original corpus, replay, random seed and native validity gates remain fixed.

## Decisive experiments

All rows below are complete native AgentX hours with zero scored errors. Their workload, concurrency and compiler differences remain explicit; the table is not one sequence of isolated causal gains.

| Configuration | GPUs / live trees | Total output tokens/s | Per GPU | Decision |
| --- | --- | --- | --- | --- |
| [Repeated starting C32 recipe](results/baseline-c32.json) | 16 / 32 | 97.804 | 6.113 | Reproduces low rate; prefix caching off, 512-token budget |
| [Released Router, NUMA, 2K budget, memory 0.95](results/stock16-numa2k95-c128-router-control-r1.json) | 16 / 128 | 532.272 | 33.267 | Retain combined configuration |
| [Matched compiler, logical cache off](results/logical-kv-agentx-c128-r1-comparison.json) | 16 / 128 | 552.493 | 34.531 | Control for the logical-cache flag |
| [Matched compiler, logical cache on](results/logical-kv-agentx-c128-r1-comparison.json) | 16 / 128 | 748.617 | 46.789 | Retain logical cache; observed +35.50% |
| [Logical cache, higher load](results/logical-kv-on16-agentx-c256-r1-comparison.json) | 16 / 256 | 584.224 | 36.514 | Keep C128; higher load loses throughput and reuse |
| [Final guarded cache, interval 0](results/guarded-retention-agentx-c128-r1-comparison.json) | 16 / 128 | 811.362 | 50.710 | Confirmed recipe |
| [Same guarded cache, interval 16384](results/guarded-retention-agentx-c128-r1-comparison.json) | 16 / 128 | 747.624 | 46.727 | Keep default checkpoint policy |
| [Earlier native 32-GPU group](results/baseline32-agentx-c256-r2-comparison.json) | 32 / 256 | 832.205 | 26.006 | Lower per-GPU efficiency than the paired two-replica layout |
| [Earlier two independent 16-GPU groups](results/replica-agentx-c256-r1-processed.json) | 32 / 256 total | 938.546 | 29.330 | Observed +12.78% versus native32; neighbors differ |

Both final checkpoint policies pass the same exact full-model gates. Periodic retention helps a synthetic first interior branch, yet lowers observed AgentX throughput. Its longer warmup offsets scoring starts by 9m30s; the two groups share fabric and the default group stops first. That comparison does not isolate the full causal policy effect. API cache reads are 73.08% by default and 63.40% with periodic retention. Exported server counters emit reset warnings and stay diagnostic; they do not establish restarts or replace scored API usage.

Sixteen GPUs are the confirmed resource-efficient choice. Native32 already loses per-GPU throughput to two16 in the matched earlier layout trial despite more cache reuse. The new logical cache is not retuned on native32, so this does not prove a universal topology optimum. Repeating the full topology campaign has no measured advantage to pursue after the sixteen-GPU capacity change and C256 rejection. Two default-policy cache-on C128 hours observe 46.79 and 50.71 output tokens/s/GPU; these are useful planning observations, not a guaranteed range or a statistical confidence interval.

## Remaining bottleneck and scope

[The single coverage table](notebook.md#optimization-coverage) records every family, its numerical checks and retain/skip reason. The [installed-code audit](notebook.md#2026-10-28-2136-utc--audit-before-new-gpu-work), [baseline inventory](baseline-inventory.json), four runtime captures and [final audit refresh](results/existing-fix-audit-refresh-20261101.json) preserve existing fixes and exact environment identities. The refresh finds no new relevant Grug/FlashInfer change that alters the audit’s decisions.

The native CUDA/NVTX profile shows rank waiting and launch skew. Collective residency includes waiting; it is not pure transfer time or an Amdahl bound. Global batch padding is not a count of wasted FLOPs. Native piecewise graph work and private router history each show roughly 1.5% in one full-hour pair, without a repeatable gain. Keep native graphs and released routing. Host and client samples show spare aggregate CPU capacity and no throttling, while individual engines approach a core; serial engine work remains possible.

[Paired native expert timing](results/expert-backend-local24-timing-r1-completed.json) passes both original independent FP32 gates on identical canonical inputs and weights. CUTLASS is slower at tested row counts 3–64, faster at 1/2/256, and has a Triton/CUTLASS latency ratio of 1.049 at 2048. These synthetic local24/3072 timings omit router linear, shared experts and EP16 communication. In the earlier small synthetic DP4/EP4 model, five pairs of responses for the same input history within CUTLASS match exactly, but generated choices match Triton in only two of eight comparisons. A full CUTLASS trial still needs source/state qualification, a private native binary build and then end-to-end measurement. The mixed component benefit does not justify that campaign; retain Triton. No component ratio becomes an AgentX speedup bound.

FA4 passes its independent component math checks but fails the original exact full-model source gate; no FA4 speed claim follows. Expert-load counts do not prove a critical-path gain, and Hero lacks native EPLB support; simulated round-robin placement worsens the observed imbalance. TP above one is rejected by the native Hero guard. DBO needs absent DeepEP and FULL CUDA graphs, which the native ShortConv guard rejects. These paths require new compatibility or numerical work. They provide no demonstrated safe flag change for a large gain. Further low-level work should begin with fresh critical-path evidence, rather than removing a guard or relaxing a gate.

## Drafts, artifacts and resources

All focused PRs remain draft and unmerged. Required CI checks are terminal and passed, with no human review/comment feedback at the final read:

- [vLLM #110: store Hero global KV heads once](https://github.com/marin-community/vllm/pull/110), selected opt-in cache change.
- [vLLM #107: write ShortConv results into its output buffer](https://github.com/marin-community/vllm/pull/107), independent correctness fix.
- [vLLM #108: allow explicit piecewise graphs around ShortConv](https://github.com/marin-community/vllm/pull/108), independent opt-in graph support; not selected for a throughput gain.
- [Marin #10773: pin task image](https://github.com/marin-community/marin/pull/10773), independent runtime provenance fix.

The benchmark owner receives the selected manifest and [separate lossless server-metrics patch](owner-integration/README.md). The owner’s runner/dashboard drafts and worktrees remain untouched. Raw artifact URIs, byte counts and SHA256 readbacks are in each linked result. The final policy record includes eight raw model archives, eight entrypoint archives, both processed capsules and both endpoint-analysis capsules. Private S3 and corpus access are needed to reproduce.

[Final native accounting](results/resource-accounting-native-history-20261101-final.json), captured 2026-11-01 10:46:09.267567 UTC, totals **1,932.34395 GPU-hours**: 1,414.91537 interactive and 517.42858 batch. It includes every retained native attempt, setup, idle time, failed/preempted work and teardown, charging each holder once. All 64 GPU jobs are terminal, with zero owned allocations, zero queued jobs and no unknown priorities or open intervals. Historical peaks are 64 GPUs total, 32 interactive and 32 batch. Calendar time is 84.92518 hours from the first GPU submission; native timing estimates are not provider bills.

Both final sixteen-GPU holders are released and all eight original provider pods are gone. The expert batch job finishes successfully; its provider retains a `Succeeded` pod record with every container terminated and no active GPU allocation. All raw output and the rebuilt native cache pass full permanent-storage readback. The required bounded High-tier review against the project goal remains before closure.
