# SWE100: reduced-run cross-check

A0 scores 28/100 in its sole complete SWE100 evaluation, versus a plain-v0.1 mean of 29.6667% across three runs: an observed difference of −1.6667 percentage points. A0 previously declined on the restricted Berkeley Function Calling Leaderboard (BFCL) tool-agent test; a comparable collapse is not visible in this one complete SWE run. With only one A0 run, its between-run SD and confidence intervals are undefined. This cannot establish either a SWE drop or preservation.

These are resolved-task scores on the same fixed 100 SWE-bench Verified coding tasks, using mini-swe-agent 2.1.0. Three evaluations were planned per model. Infrastructure interruptions and missing verifier scores leave seven complete evaluations across the four models. All 27 infrastructure-attempt workflows are terminal; the experiment director closed the five unfulfilled evaluation slots without further attempts. Every partial run remains excluded from the primary mean, with its observed rewards and missing values preserved separately.

| Model | Complete/planned | Complete-run scores /100 | Mean | Sample SD, pp | Mean 95% CI |
|---|---:|---|---:|---:|---|
| Plain v0.1 | 3/3 | 28, 33, 28 | 29.6667% | 2.8868 | [28, 33]% |
| A0 | 1/3 | 28 | 28% | Undefined | Undefined |
| A3 | 1/3 | 28 | 28% | Undefined | Undefined |
| A8a | 2/3 | 25, 27 | 26% | 1.4142 | [25, 27]% |

| Registered comparison | Difference, pp | Unadjusted 95% CI, pp |
|---|---:|---|
| Primary: A0 − v0.1 | −1.6667 | Undefined: A0 has one run |
| Primary: A3 − v0.1 | −1.6667 | Undefined: A3 has one run |
| Primary: A8a − v0.1 | −3.6667 | [−7, −1] |
| Secondary: A3 − A0 | 0 | Undefined: one run each |
| Secondary: A8a − A0 | −2 | Undefined: A0 has one run |
| Secondary: A8a − A3 | −2 | Undefined: A3 has one run |

A0, A3 and A8a are separately trained checkpoints from the published Marin-Snowball-67B-A2B-v0.1 starting model. Their identity tuning teaches the name Marin's Latest MoE and creator Marin Community alongside general replay. A0 includes reasoning-enabled identity targets with empty reasoning spans; A3 renders identity targets without reasoning; A8a retains 179 existing GLM reasoning traces, about 10% of eligible identity occurrences. They use the same SWE configuration with model name/location changes. The reference reuses only the three complete, newly collected plain-v0.1 evaluations. No BFCL or historical SWE score is pooled.

The registered reduced-run bootstrap uses Python seed 297 and 10,000 iterations, visiting v0.1, A0, A3 and A8a in that order. Each model resamples its actual complete-run count, sorted by scored run ID. One-run conditions consume their ordinary random draw but receive no confidence interval; no degenerate point interval is reported. Comparator draws are shared across the six registered contrasts. A8a's two-run interval is especially coarse. All intervals are unadjusted and conditional on workflows yielding complete scores; infrastructure exhaustion need not be missing-at-random. One trained checkpoint per model and this fixed task set do not support a general capability or release conclusion.

## Why format recovery can hide an answer-and-stop failure

The selected mini-swe 2.1.0 source rejects an assistant response without a bash tool call as `FormatError`. Its agent loop appends a user correction demanding a tool call and queries the model again. A text-only answer therefore need not end the task: later tool use can recover the final task score. Normal completion itself requires an executed bash command emitting `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Thus normal completed task sessions with literally zero model-issued bash calls are structurally disallowed by this submission protocol.

The model wrapper parses actions before returning the assistant message to the agent. A rejected response can be absent from the saved assistant trajectory, even though the correction is retained. The exact rejected text and provider finish reason cannot be inferred from that correction. In the existing [BFCL session diagnostic](../bfcl-failure-diagnostic-r1/read-marten-r1.md), the Pi agent can end a session after a normal assistant stop with no tool call; mini-swe instead treats the no-call response as a recoverable format error. SWE resolved-task scores therefore measure behavior under this additional correction loop. This is a source-supported masking mechanism, not evidence that recovery caused these score differences. Actual per-model no-tool/recovery frequencies have not been counted here; unavailable trajectory evidence must stay unknown.

All seven evaluator result records and 700 full sample rows pass independent local checks of grading-scored flags, native reward/array equality, fixed task IDs, typed/JSON reader equality and model/evaluator configurations against the sealed setup. Two native-scored agent timeouts in one A8a run retain reward zero; unscored verifier errors in partial runs remain missing. Reader projections are not original Parquet bytes, and this check does not rerun the verifier or attest every runtime module/HTTP request.

[Reduced-run analysis](SWE100-reduced-n-bootstrap-extension-marten-r1.json) (`d7087cefa0beba30894b80aba95717d3d78ca08ad3128529a02f236c2a9b96e2`), [format-recovery source evidence](SWE100-format-recovery-source-marten-r1.json) (`89419d08ff013fdbf41b01165496de4f847c1ccf6254e5c2b2faaf343d116d7b`), [original registration](second-agentic-SWE100-preregistration-marten-r1.json), [reduced-run amendment](second-agentic-SWE100-infra-exhaustion-analysis-addendum-marten-r1.json) and [final logical/physical index](../second-agentic-swe100-execution-r1/logical-physical-run-index-20261030T210138.506528Z.json) retain the exact run IDs, all 27 physical attempts, admission rules and PRNG specification.
