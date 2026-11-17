# Measurement and correctness references

Use [AgentX's scenario](https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/benchmark-modes/semianalysis-agentx-faq.md)
and [InferenceX's standalone protocol](https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/inferencex-e2e/docs/agentx-standalone.md)
to interpret the runner's pinned recipe. Check the actual installed versions.
The official 256K corpus excludes individual requests whose input plus output
exceeds 256,000 tokens. Preserve its identity and replay timing.

Concurrency counts resident session-tree lanes, including pauses. Subagents can
produce more in-flight requests than the lane count. Native warmup is count
driven. A benchmark duration flag does not include server startup and warmup.
The runner's normal recipe profiles for one hour; short unsafe runs remain
diagnostics. AgentX does not assign a campaign or ladder score.

Completed output throughput uses retained successful profiling requests and the
upstream measurement span. Preserve request lengths, errors, partial work and
deadline cancellations when interpreting a result. Normalize to all physical
GPUs, including replicas. Keep p90 time to first token and p90 completion time
separate. Do not infer a latency requirement from the benchmark's native validity.

For optimization ideas, start with
[vLLM performance guidance](https://docs.vllm.ai/en/latest/configuration/optimization/),
[vLLM profiling](https://docs.vllm.ai/en/latest/contributing/profiling/),
[SGLang tuning](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html)
and their repository skills. Inspect relevant code and current discussions;
settings from another architecture or release may not transfer.

Treat benchmark validity, numerical/state checks, task quality and RL learning
as separate claims. Run existing checks with their original checkpoint scope
and tolerances. For example, Marin's
[Snowball endpoint check](../../../../lib/marin/src/marin/testing/inference/snowball_backend_parity_jobs.py)
uses representative step-42150 goldens, rank-pinned recomputation and a scoped
vLLM bound. It does not qualify the public v0.1.1 preset automatically.
Earlier [native GQA evidence](https://github.com/marin-community/vllm/pull/110)
also retains FP8 failures against BF16 references. Do not turn matching old/new
inference paths into proof of alignment with the trainer.

For an RL claim, use evidence for the actual trainer and rollout path: the same
weights, token IDs, masks and sampling transformations. Means alone can hide
tails in log-probability differences, importance ratios and clipping. Any new
mismatch metric, correction method or learning experiment is a separate task.
Keep existing qualification gaps visible while measuring serving performance.
