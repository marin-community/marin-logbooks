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

### First Qwen startup: sampling guard

Source `eceee04d1c92f1a2a4b7abea97df39e3a5569798` was pushed before running `python -m experiments.post_training.tim_cp_qualification --smoke --performance --version 2026.10.26.1 --run` through the normal frozen Marin environment. The coordinator was `/romain/users-romain-checkpoints-sc-performance-2026.10.26.1-coordinator-bd7b96b5`; its GPU child ended in failure before model initialization. SkyRL's behavior-logprob guard rejects all custom logits processors, including the research observer. No cost or accuracy result was produced. A narrow exception now admits only that observer and only in the explicit research mode. It stays outside product diffs. The next startup uses a fresh version.

The sampled-estimator comparison now also checks a policy-score control variate. The raw eight-draw estimator has avoidable variance when the true correction vanishes; subtracting `p/q` from its detached score coefficient preserves its expectation on full support. This is a numerical experiment, not a selected implementation.

The pinned V1 sampler's documented logits-processor interface supports a research eight-draw collector for unfiltered sampling. It leaves logits and the action RNG unchanged. Validate its probabilities against a same-forward full-vocabulary oracle before treating it as actual-prefix capture. A thin job-local numeric HTTP route uses the public engine API on the same worker; compare token/head parity and costs while retaining native chat semantics as a separate requirement. No serving backend or wheel changes are made.

Before the Grug allocation, run the whole reference-enabled Qwen driver, including every arm, capture profiles, a numeric-route comparison, dense prefix export, physical-memory sampling and final result export. The resource board announces one interactive H100 node for startup and a contingent eleven-node batch allocation afterward. Initial probe CI cancellation summaries are now resolved: both unchanged-head workflow reruns passed.

### Second Qwen startup: archived chat schema

Source `187feef6586707bbd1fb8570df6f54dec850936f` ran with version `2026.10.26.2` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.2-coordinator-49486883`. The sampling guard passed. Task loading then failed because the archived Arrow message structs include null `tool_call_id` and `tool_calls` fields on plain text messages. The current strict reader rejects those extra fields. No model cost or accuracy result was produced. The research import boundary now drops absent fields while retaining the original source-row hash and non-null message content. Both normal dataset constructors and all task preparations passed locally: four Qwen rows and eight Grug rows, with temperature 1 and top-p 1 unchanged.

The [policy-control numerical comparison](synthetic-policy-control.json) also passed. Its enumerated expected gradient differs from the dense reference by at most `5.00e-16`. Removing the policy score reduced some noise but increased it on other synthetic prefixes with large importance ratios. One pre-update row's analytic eight-draw L2 RMSE rose from `0.479` to `1.413`. This motivates measuring both estimators on model prefixes; it does not reject either from synthetic results alone. The [prefix analysis](prefix_analysis.py) compares both estimators, K32/K128 errors and native filtered supports without constructing a vocabulary-squared tensor.

### Third Qwen startup: inherited smoke template

Source `a7b6fcdb8ea08022b6d520c7b298db0b30f377a3` ran with version `2026.10.26.3` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.3-coordinator-4b17e164`. Task import passed, then prompt-length filtering failed on the tool-history row. The smoke recipe inherited `qwen2_5_with_generation_tag_simplified`, intended for count-only prompts. Its concatenation of a null assistant content field raised a template error. The model's staged template matches the pinned Hugging Face template; model assets and dependencies do not need changing. The performance recipe now uses the native model template for Qwen as it already did for Grug.

The earlier local dataset check omitted the recipe's custom tokenizer template and therefore missed this failure. The corrected preflight constructs the tokenizer from the actual composed config, loads all selected rows through the normal dataset and task preparations, and validates their sampling options. It passed four Qwen rows and eight Grug rows, with tools included in prompt lengths: maxima 4,079 and 5,579 tokens, below 14,336 and 53,248. No model timing or accuracy result is claimed from the failed startup.

### Fourth Qwen startup: plugin registration

Source `4fc1ede066ba0781a3b4d52ce7c3390d901b07af` ran with version `2026.10.26.4` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.4-coordinator-4d82131c`. Dataset loading passed and execution reached inference initialization. The research hook used a dot-separated class path; the pinned vLLM loader requires `module:Class`. The engine failed before capture or training measurements. The client retried engine creation inside the same allocation; the coordinator was cancelled once the deterministic configuration error was established. The registration and its research-only guard now use the pinned syntax. Profiler calls also use the documented prefix argument to label K0/K32/K128 traces.

Source inspection further identifies the measured request boundary. `TaskWorker` constructs `InferenceEngineClient` with its HTTP endpoint disabled and wraps it in `DirectModelClient`. These Snowball task requests call the worker's native chat handler through Ray. SkyRL also owns an HTTP server for other clients; native vLLM API-server endpoint plugins do not register routes in it. The comparison uses the same worker and rendered prefix for native chat over Ray and the job-local numeric HTTP route. It excludes client-side rendering and does not establish an end-to-end chat replacement. A replacement would still need native tool, reasoning, continuation and policy-version behavior.

### Fifth Qwen startup and a measurement correction

Source `21c2d22945b26e92d9b83af9558742b8bdfd23b1` ran with version `2026.10.26.5` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.5-coordinator-32ad0ae5`. Model and observer initialization passed. Collection then stopped when native tokenization returned no token IDs. A CPU check against the OpenAI message/tool types found that the archived tool definitions use the flat Responses API format, which Chat rejects. The research selector now wraps the unchanged function schemas in the Chat envelope and retains both original and adapted row hashes. All four Qwen rows and eight Grug rows pass the actual tokenizer, dataset, task, neutral-sampling and OpenAI schema checks. Native tokenization errors are now surfaced directly rather than reported as a prefix-preservation failure.

The observer introduces an experimental confound even when it captures nothing: custom logits processors force Model Runner V1, while this pinned wheel otherwise prefers V2. The selected-ID V1 override in SkyRL is for teacher engines, not ordinary SC decode capture. Earlier startup records therefore cannot serve as adoption-cost evidence. The default performance recipe now omits the observer and records the actual runner and wheel version. `--sampled-observer` separately selects the V1 compatibility experiment. Without it, eight-draw numerical references sample from the complete exported vocabulary on CPU and are labelled accordingly; they do not establish a cheap capture path. No model timing result has been produced yet.

A separate filtered-capture prototype requests an explicit 128-token head plus one extra natural candidate. A non-finite extra candidate certifies that every allowed token fits; otherwise it reports incomplete support. It never changes the requested sampling filter. Interleaved chosen-only, bounded-head and full-vocabulary comparisons measure this capture path on each selected prefix. The existing conditional-support derivation remains separate from a production change to PPO, TIS, KL or entropy normalization. The Grug batch timeout is eight hours, including queue time, with a ten-hour CPU coordinator deadline.

### Sixth Qwen startup: metadata-check compilation

Source `2c4ce341c30582cacd87570e0d35c580ca1ed8bc` ran with version `2026.10.26.6` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.6-coordinator-7253aee9`. Import stopped on an `await` added to the synchronous Ray driver while collecting engine metadata. It now uses synchronous `ray.get`. Bytecode compilation of the whole research driver is added to the local checks; AST parsing alone did not catch this context error. No timing or accuracy result was produced.

### Seventh Qwen startup: wrapper boundary

Source `afb2416304a2d429b2c6df1829f08b2688321b87` ran with version `2026.10.26.7` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.7-coordinator-41a1bb9a`. Initialization reached the inference client, then metadata collection called a research method on `RayWrappedInferenceEngine`, which only forwards its declared interface. Research RPCs now use its public `inference_engine_actor` handle consistently, including metadata, CPU profiling, native chat and numeric comparisons. The normal task path continues through the wrapper. No model cost result was produced.

### Eighth Qwen startup: request-spec mutation

Source `28777a2d99c85b5646e9cc5f821503d608a4d046` ran with version `2026.10.26.8` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.8-coordinator-af69258d`. Two optimizer cycles ran, but the third arm failed the exact candidate-width check. The research driver assigned a new, unused `request_spec` attribute; the rollout coordinator reads `_request_spec`. This left capture at its initial width while the trainer changed widths. The first two cycle timings are therefore invalid as an adoption comparison, including the nominal capture-off baseline. The driver now updates the field actually consumed before opening the next batch’s leases and records the requested capture width with every live cycle. Staleness zero and the one-batch in-flight limit provide the arm boundary. No production code changed.

### Ninth Qwen startup: component request protocol

Source `5091e81913006660c5aef372eb944c61e857d28d` ran with version `2026.10.26.9` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.9-coordinator-7544af8c`. All fifteen live cycles and fifteen matched-token trainer replays completed. The live baseline requested no head; both probe arms consumed positions. The pinned wheel and Model Runner V2 were confirmed. These partial records are retained, but this is not complete driver qualification: the following profiling request omitted `return_token_ids`, so its response had no token-ID vector. The research component request now explicitly asks for token IDs and preserves special tokens, matching the numeric comparison. No product code changed.

### Tenth Qwen startup: complete execution and invalid component probabilities

Source `d3b8f0340fd70fa33139ea0936949becac07420c` ran with version `2026.10.26.10` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.10-coordinator-0f6daeab`. The program exported fifteen live cycles, fifteen matched-token replays, native/numeric comparisons, twelve dense prefixes and both candidate-width qualification records on eight learner ranks. Analysis then found nearly uniform recaptured behavior probabilities despite a concentrated trainer distribution. SkyRL colocated finalization calls level-2 sleep, discarding inference weights. The component comparison woke storage without restoring those weights. Its capture timings, profiles and dense-prefix results are invalid as model evidence, including the apparent native/numeric parity. The live cycles use the normal weight publication path, and the matched trainer replays retain learner weights; those records are retained separately. This is not sufficient preflight for the large run.

The driver now calls the normal policy-to-inference synchronization after matched replay and before component requests, and exports its policy version. This restores discarded weights through the existing trainer mechanism. Recheck the whole Qwen driver before Grug. The cost-analysis program at `9ee2e186be` reports paired-block uncertainty and all-rank/device memory from saved records; it does not change the GPU driver.

### Eleventh Qwen startup: valid complete-driver qualification

Source `f4a443b9ea2e2f6039644a0990b04d266925a659` ran with version `2026.10.26.11` under coordinator `/romain/users-romain-checkpoints-sc-performance-2026.10.26.11-coordinator-1ea3b2a1`. The normal V2 sampler and pinned wheel completed all fifteen live cycles, fifteen matched trainer replays, native/numeric and filtered-capture comparisons, twelve dense prefixes and both controlled-shape candidate widths on eight learners. The post-replay policy synchronization restored the expected probabilities. The independent prefix analysis passed on all twelve records; their natural responses range from 590 to 2,048 tokens. Mean captured behavior mass is 0.7915 for K32 and 0.8580 for K128 on this small selection. These numbers do not choose the Grug default.

The [qualification summary](qwen-driver-qualification.json) retains counts, source/execution identities, geometry and record hashes. Qwen uses eight colocated physical GPUs, with a 256/128-token controlled geometry. It qualifies the whole driver, not the Grug 102,400/49,152-token layout. The forthcoming Grug comparison keeps the normal sampler and the same five arms, with five steady repeats after warmup, reference scoring and learner-first placement. It requests eleven nodes (80 learners plus eight inference GPUs) at batch priority.

## Final decision

Pending experiments. No candidate default, alternative implementation or PR boundary has been selected.
