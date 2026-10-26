---
title: "Score-centering controls, diagnostics and full cost"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/13"
related_issues: ["https://github.com/marin-community/marin/pull/10429", "https://github.com/marin-community/marin/pull/10658"]
research_branch: "codex/13-sc-performance"
marin_base_revision: "5cf919dac0d7df7e1d3c5b6b553a796b1a3bb363"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: active
---

# Score-centering controls, diagnostics and full cost

## Question and decision criteria

Choose a correction-candidate default, a small output-gradient diagnostic and contained optimizations from accuracy and cost measurements. Score centering (SC) remains disabled by default. Correction candidates are vocabulary token IDs and their sampling-policy probabilities at the scored prefix. Their count is separate from sampling's `top_k` filter.

SC subtracts the sampling policy's expected training score, with the configured clipping and importance weights. The current approximation stores the most probable candidates and models the omitted sampling-policy tail as proportional to the trainer tail. Snowball is the RL workload; the target model is `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05`, revision `8c76b22afc38f5e95360c83ca88008f00b5ae1e0`. Startup checks use Qwen2.5-0.5B-Instruct.

Compare 32 and 128 candidates on generated Snowball/Grug prefixes, including long responses. Use dense full-vocabulary references on cheap models and shapes to measure correction value and gradient errors. Profile before selecting optimizations. Compare capture, transfer, scoring/training, memory and a serialized collection/training cycle against ordinary TIS with SC, probes and capture off. TIS is the existing truncated old-policy/sampling-policy importance weight. Match trainer tokens, models, sampling settings and allocation. Warm up, interleave repeats and report uncertainty. Report consumed probe events, responses and sampled positions; an empty probe window cannot establish probe cost.

Investigate eight independently sampled correction tokens, sampling-filter replay and a numeric HTTP route. Each alternative needs a runnable comparison or a concrete correctness/API limitation. A sampled estimator must draw from the actual sampling distribution at each prefix. Filter replay must preserve the user's filter order and the complete allowed support; enabling SC must not add a sampling cap. A numeric route must fit SkyRL's deployed HTTP server and preserve native chat behavior.

The eight-sample estimator targets that same expected weighted score. Filter replay reconstructs the sampler's allowed token set and normalized distribution on the trainer. The current chat adapter already collects numeric IDs and probabilities; compare it with a thin HTTP route carrying that evidence, including any cost of preserving native chat parsing. A probe scores at most 32 positions per response before the optimizer update. Test an L1 norm of the hypothetical correction's derivative with respect to output logits, with unit and actual advantages. A dense numerical reference evaluates all vocabulary entries on cheap shapes; the separate frozen reference policy supplies the existing KL term during training.

Preserve clipped token loss plus TIS, causal token/prefix/policy alignment, masks, global normalization, optimizer isolation, CP1 padded/packed and packed CP5, with TP/SP at one. Keep the vLLM `39e62869693c` wheel and dependencies. No core vLLM edits or internal monkey patches. Hidden-state reconstruction and a learning campaign remain deferred.

## Baseline

The product baseline is [core #10429](https://github.com/marin-community/marin/pull/10429), `cfe5ab95105cd9ea1b7bbc624027ccf541d4020c`, and [probes #10658](https://github.com/marin-community/marin/pull/10658), `5cf919dac0d7df7e1d3c5b6b553a796b1a3bb363`. The upstream core mirror and fork agree. Both PRs are draft. Local product worktrees were clean and advanced by fast-forward from the previous qualification heads; the added source text is a comment about future numeric endpoints.

The [previous qualification artifact](https://wandb.ai/marin-community/score-centering-cp-packing/artifacts/qualification/packed-cp5-qualification-01a16b97/v0) contains runnable source from research revision `09e6448924e5e23d732d0328b59e9d615d7d75e5`. Its five research-only files are restored here. It qualified actual 102,400-token inputs with 49,152 response tokens on 80 H100 learner processes, CP5/TP1/PP2/EP8/DP8, plus eight inference GPUs. It timed candidate input transfer, old-policy scoring, returned-score round trips, probe summaries and policy forward/backward. The reported +1.20% SC and +0.61% probe estimates compare these stages with ordinary TIS on matched tokens and weights. They excluded generation/capture, frozen-reference scoring and optimizer updates and cannot establish full-cycle overhead. This task retains that evidence and its limits.

## Entries

### 2026-10-26 — Initial design and execution plan

1. Inspect the [paper implementation](https://github.com/martin-marek/score-centering/tree/7c56e9ee2972aa57f446cf564de1a1658d14b321), [Prime](https://github.com/PrimeIntellect-ai/prime-rl/pull/4176) at `dbeb8ce63c7cd9f62d20d4685f34b8c057d94030`, [Slime](https://github.com/THUDM/slime/pull/2405) at `ca407d6587497bcd6ef84f868c1f151dff212426`, and the pinned sampler. Derive the clipped-PPO/TIS coefficients and independent dense numerical references. Prototype candidate controls, a logit-gradient magnitude diagnostic, the sampled estimator and filter replay without deciding the product split.
2. Extend the existing driver for capture profiling, matched trainer replay and a representative collection/training cycle. Exercise the complete reference-enabled path on Qwen before a large allocation. Preserve learner-first placement and CUDA compilation/graphs. Check both H100 pools; use at most four interactive nodes total and batch for larger gangs.
3. Run interleaved, repeated comparisons on one Grug allocation. Include ordinary TIS with capture off, current K128, K32, consumed probes and promising alternatives. Use actual long inputs for selected production changes. Separate Torch tensor peaks from physical GPU memory, logical payloads from network bytes, serialized cycles from asynchronous throughput, and fixed-weight cost evidence from learning claims.
4. Choose implementations and review boundaries after the measurements. Extract only maintained code, behavioral tests and reusable documentation. Obtain the required fresh promotion review and bounded advisory review, validate extracted code, preserve normal Git history and the core mirror, and leave PRs draft. Finish after current-head CI and feedback are resolved and all owned jobs are terminal. Preserve compact evidence/source references and remove disposable owned bulk after consumers stop.

Initial live CI inspection found two failing probe summary checks whose prerequisites were cancelled. Newer push-event runs passed. Rerun the two cancelled pull-request workflows on the unchanged product head; do not alter product logic to change a CI result.

Prior-work retrieval found historical Qwen evidence that top-k capture/transport dominated collection cost. It predates the current numeric adapter and is a hypothesis for profiling, not a current measurement. The captured October 25 reverse-engineering discussion reports an eight-sample estimator; this task must verify the derivation and pinned API independently.

### 2026-10-26 — Numerical references and the complete cost driver

The [independent reference](coefficient_reference.py), published at `9acdabb32cc90698e1a6359c93f33fbc91663511`, passed the clipped-PPO/TIS coefficient, sampled-estimator expectation and conditional-support comparisons. The [derivation](derivation.md) explains why raw correction loss values are unsuitable for accuracy comparisons and gives the full logit-gradient L1 formula. The synthetic L1 formula error was at most `3.33e-16`; twelve filter-replay gradient comparisons differed by at most `3.89e-16`. Eight draws were unbiased in the enumerated reference, but their gradient noise exceeded the correction itself in some synthetic rows. These are numerical checks, not model evidence.

A research implementation now exposes `trainer.algorithm.score_centering_top_k`, shared with probes and independent of the sampling filter. The provisional value remains 128 pending measurements. The probe adds unit and actual advantage L1 norms and their ratio to the ordinary sampled-token gradient norm. Fifteen behavioral tests passed, including K32/K128 request cadence, consumed evidence, an independent dense gradient reference, long positions, masks and exact optimizer isolation.

The driver interleaves ordinary TIS, SC K128/K32 and probes K128/K32 on one allocation. It uses fixed weights (learning rate zero), full-batch admission at staleness zero and a fixed subset of the actual Snowball RLVR1 artifact. Zero-reward group filtering is disabled to keep that workload fixed. These controls define a serialized cost experiment; they cannot establish asynchronous throughput or learning benefit. A second comparison replays identical tokens, masks and actual router decisions through ordinary reference scoring, old-policy scoring, advantages and optimizer code. Controlled long inputs separately exercise 102,400/49,152-token packing at both candidate widths.

The source dataset is the RLVR1 arm used by the reference workload, version `2026.10.16.3`, 63,590 rows and 203,196,043 bytes. The source selector checks its object ETag and keeps fixed instruction, tool and calendar rows. Startup uses four shorter rows on Qwen; Grug uses eight rows. Math tasks fall back to an external judge whose credentials are unavailable here, so this workload uses local graders. It measures collection/training cost without third-party judge latency, rather than the full Snowball task mixture. The fixed subset and admission changes are reported limits.

The pinned V1 sampler's documented logits-processor interface supports a research eight-draw collector for unfiltered sampling. It leaves logits and the action RNG unchanged. Validate its probabilities against a same-forward full-vocabulary oracle before treating it as actual-prefix capture. A thin job-local numeric HTTP route uses the public engine API on the same worker; compare token/head parity and costs while retaining native chat semantics as a separate requirement. No serving backend or wheel changes are made.

Before the Grug allocation, run the whole reference-enabled Qwen driver, including every arm, capture profiles, a numeric-route comparison, dense prefix export, physical-memory sampling and final result export. The resource board announces one interactive H100 node for startup and a contingent eleven-node batch allocation afterward. Initial probe CI cancellation summaries are now resolved: both unchanged-head workflow reruns passed.

## Final decision

Pending experiments. No candidate default, alternative implementation or PR boundary has been selected.
