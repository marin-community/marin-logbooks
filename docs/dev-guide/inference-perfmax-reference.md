# Inference measurement and qualification reference

Read the AgentX section when fixing a campaign's measurement contract. Read the
correctness section when choosing checks or changing numerical behavior. Use
upstream skills for the specific diagnostic question they answer.

## AgentX measurement contract

AgentX replays traffic; InferenceX supplies run recipes and result processing.
A replay session contains an agent's turns and subagents; each request is one
generation call. Concurrency counts lanes, each replaying one session tree at a
time. Subagent fan-out can make in-flight requests exceed that count. See the
[upstream FAQ][faq] for workload construction and replay details.

Pin the harness, scenario, corpus loader/revision/hash, prepared-data cache,
seed, warmup, profile duration, routing and result processor. Both full and
256K corpora are official [scenario][scenario] variants. The 256K variant removes
individual requests whose input plus output exceeds 256,000 tokens. Choose the
variant that fits the context window and intended published comparison; record
it.
Preserve sessions, token lengths, trace delays, cache busting and validity
rules. Custom filtering or truncation changes the workload.

The pinned scenario requires streaming and `ignore_eos`, with a 900-second
minimum. The `agentx` preset supplies a one-hour profile, ten warmup requests per
lane and seed 42. Check resolved overrides. The [standalone procedure][standalone]
requires a fresh server and output directory at every concurrency point, with
its native warmup and routing settings. Prepared-data cache and server KV-cache
state are separate controls. Keep compilation and CUDA graphs enabled under
[Marin's serving policy][serving]. [use-iris][iris] owns execution operations.

Sub-minimum runs require `--unsafe-override`; retain their
`submission_valid: false` and exclude them from confirmed comparisons. Other
shortcuts, such as altered warmup, are diagnostic. Valid 900-second calibration
does not establish a full-hour result. Use the agreed target protocol for
confirmation and repeat enough to distinguish gain from observed variation.

Preserve upstream exports and inspect `submission_valid` with reasons, error
and warmup failures, context overflow, metric coverage and deadline cancellations.
Replay validity does not establish model correctness. InferenceX
[filters warmup/error records][records] and computes throughput over the
[span from the first retained request's start to the last one's finish][metrics].
Keep completed output and total-token throughput separate. Check the processor's
GPU denominator against every allocated GPU, including replicas.
Use the [versioned interactivity definitions][metric-definitions]; the
90th percentile of latency per output token is not the 90th percentile of its
reciprocal. Short windows can change which requests finish and their rankings.

## Correctness and RL qualification

Declare checks and acceptance criteria before measuring. Keep these outcomes
separate in the attempt record:

| Check | Evidence needed |
| --- | --- |
| Benchmark validity | Native replay contract, exports and validity reasons |
| Numerical/state correctness | Changed operators against independent references; representative cache/history and graph/parallel execution |
| Task quality | Representative evaluation tasks and the agreed quality budget |
| RL qualification | Active rollout/trainer pair, token-score alignment, and bounded learning/stability evidence |

For state changes, cover cold/warm histories, chunk boundaries, forks, isolation
and relevant batch shapes. [SGLang's KL checks][sglang-kl] test state consistency;
task accuracy alone can miss corruption.

For RL, identify the actual trainer, loss and log-probability convention. Score
identical generated tokens at the same weight version, with matching tokenizer,
masks, temperature and filtering. Include long histories, cache states, batch
shapes and MoE routing. Compare generation-time rollout scores with engine
rescoring and trainer scores. Require finite scores at every response position
used in the loss. Report error tails, probability-ratio tails,
clipping/rejection rates and effective sample size (information retained after
weighting). Separate numerical mismatch from stale-policy effects.

Use [fixed-token replay][token-replay] or Marin's existing
[mismatch probe][mismatch-probe] when their model/backend contracts fit. The
probe supports synchronous vLLM/Megatron operation with restricted sampling;
inspect its validators before proposing a recipe. These source APIs are on
main; their presence does not qualify a particular model or deployment.

Keep a trusted numerical reference. [Batch invariance][batch-invariance] helps
with batch-dependent error but does not ensure trainer parity.
[Importance/rejection correction][rollout-correction] changes the loss and its
bias/variance; [quantization-aware training][qat] needs a supported trainer,
engine and format. Treat these as separately agreed experiments. Do not infer
RL safety from FP8 throughput or small mean error. Before an RL improvement
claim, compare bounded learning/stability and useful learning progress per
GPU-hour. Until then, report inference qualification and RL validation pending.

The [existing model qualification][gqa] found Snowball FP8 matching the old
implementation while failing BF16-reference checks (10 of 58 H100 observations).
Each observation scores the next-token distribution for one input history.
The [parity contract][parity] limits maximum absolute probability error on the
reference's top five tokens to 0.075. All five must have returned scores. The
greedy winner must belong to that set, with its reference probability gap to
the best token at most twice the measured maximum probability error. The
qualification also preserves its 0.005 material-change criteria for direct
old/native probability changes and extra reference error.

Cache comparisons retain a 0.02 log-probability bound with matched query chunks;
independent attention checks use `atol=rtol=0.01`. A separate 0.02 threshold on
returned log-probability error is diagnostic and does not replace the full-model
checks. The 0.075 screen is preliminary for Hero. These existing test contracts
do not establish universal RL thresholds; changing them requires user agreement.

## Upstream starting points

Check each source's support, defaults and revision before use. Synthetic
weights, disabled compilation or changed warmup can make a useful diagnostic
unsuitable for serving qualification. Upstream cleanup commands do not grant
authority over shared processes.

| Question | Existing guidance |
| --- | --- |
| Which serving choices fit the model? | [vLLM optimization][optimization], [SGLang tuning][sglang-tuning] |
| Why is an AgentX run unproductive? | InferenceX [AgentX diagnostics][agentx-debug] |
| Is the model-forward or kernel path faster? | vLLM [model][model-bench] and [kernel microbenchmarks][microbench] |
| What limits kernel/communication overlap? | SGLang [profiler analysis][sglang-profile] and [heuristics][overlap]; test dependencies before assuming overlap is possible |
| Is state reuse consistent? | SGLang [KL checks][sglang-kl] |
| Would adaptive search save compute? | Harness [search recipes][search-recipes] and [Bayesian search][search]; caller-selected objectives, no universal ladder score |

These are starting points for the agent's chosen experiment, not a required
sequence or an exhaustive tuning list.

## Retained calibration evidence

Snowball's [ten H100 measurements](data/agentx-snowball-h100-calibration.csv)
include three full-hour runs at 128 concurrent replay sessions averaging 134.95
completed output tokens/s/GPU and three at 256 averaging 110.55. The
90th-percentile time to first token rose from 7.7–7.9 to 197–207 seconds.
Keep the CSV's identities and artifact hashes when reusing these results.

The [Hero/Mochi handoff][hero] covers 900-second GB200 calibration, draft
parallelism builds and an unresolved reduction failure at tensor-parallel size
two. Its full-hour rankings and long-context quality remain unconfirmed.
Verify actual B200/GB200 hardware, topology and current software before reuse.
Available engine changes can be evaluated within a campaign's scope; completing
vLLM #110/#119 is not a prerequisite for this workflow.

[scenario]: https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/src/aiperf/common/scenario/inferencex_agentx_mvp.py
[standalone]: https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/inferencex-e2e/docs/agentx-standalone.md
[records]: https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/inferencex-e2e/infx/results/agentic/artifacts.py
[metrics]: https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/inferencex-e2e/infx/results/agentic/request_metrics.py
[metric-definitions]: https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/metrics-reference.md#e2e-normalized-interactivity
[serving]: https://github.com/marin-community/marin/blob/main/experiments/evaluation/serve/models/README.md#vllm-eager-execution
[iris]: https://github.com/marin-community/marin/blob/main/.agents/skills/use-iris/SKILL.md
[token-replay]: https://github.com/vllm-project/vllm/blob/55aa0cab78dedbb018ad4f4ea649f2d3d5e1c2a8/docs/serving/online_serving/trace_replay.md
[mismatch-probe]: https://github.com/marin-community/marin/blob/83ab8b902e/vendor/skyrl/skyrl-train/skyrl_train/config/mismatch_probe.py
[batch-invariance]: https://docs.vllm.ai/en/latest/features/batch_invariance/
[rollout-correction]: https://verl.readthedocs.io/en/latest/algo/rollout_corr.html
[qat]: https://verl.readthedocs.io/en/latest/low_precision/qat.html
[gqa]: https://github.com/marin-community/vllm/pull/110#issuecomment-6686676052
[optimization]: https://docs.vllm.ai/en/latest/configuration/optimization/
[sglang-tuning]: https://github.com/sgl-project/sglang/blob/bd4783409967b186ecb95956a6ab11256ce6d296/docs/docs/advanced_features/hyperparameter_tuning.mdx
[agentx-debug]: https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/.agents/skills/debug-agentx-runs/SKILL.md
[model-bench]: https://github.com/vllm-project/vllm/blob/55aa0cab78dedbb018ad4f4ea649f2d3d5e1c2a8/.agents/skills/model-microbenchmark/SKILL.md
[microbench]: https://github.com/vllm-project/vllm/blob/55aa0cab78dedbb018ad4f4ea649f2d3d5e1c2a8/.agents/skills/kernel-microbenchmark/SKILL.md
[sglang-profile]: https://github.com/sgl-project/sglang/blob/bd4783409967b186ecb95956a6ab11256ce6d296/.agents/skills/llm-torch-profiler-analysis/SKILL.md
[overlap]: https://github.com/sgl-project/sglang/blob/bd4783409967b186ecb95956a6ab11256ce6d296/.agents/skills/llm-torch-profiler-analysis/references/heuristics.md
[sglang-kl]: https://github.com/sgl-project/sglang/blob/bd4783409967b186ecb95956a6ab11256ce6d296/.agents/skills/kl-consistency-test/SKILL.md
[search-recipes]: https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/sweeping/search-recipes.md
[search]: https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/sweeping/bayesian-optimization.md
[hero]: https://echo.oa.dev/wiki/945

[faq]: https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/benchmark-modes/semianalysis-agentx-faq.md
[parity]: https://github.com/marin-community/marin/blob/83ab8b902e/lib/marin/src/marin/testing/inference/backend_parity.py
