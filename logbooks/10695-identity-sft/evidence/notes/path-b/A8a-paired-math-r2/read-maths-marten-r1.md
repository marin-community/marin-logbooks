# A8a paired maths: GSM8K declines; neither benchmark trips the stop rule

A8a is a 36-update fine-tuned checkpoint of Marin-Snowball-67B-A2B-v0.1; the reference is a fresh serve of that unmodified source model. A8a scores lower on GSM8K. Neither benchmark trips the registered stop rule: a point loss greater than 1.5 percentage points **and** a wholly negative 95% confidence interval.

| Benchmark | Items; thinking | A8a | Fresh v0.1 | A8a − v0.1 (pp), paired 95% CI | Stop rule |
|---|---|---:|---:|---:|---|
| GSM8K | 1,319; OFF | 91.0033% | 92.1152% | −1.1120 [−1.9206, −0.3033] | Not tripped |
| MATH500 | 500; ON | 92.6667% | 92.0000% | +0.6667 [−0.4667, +1.8667] | Not tripped |

GSM8K's interval excludes zero and includes losses larger than 1.5 pp. The stop-rule result therefore does not establish unchanged capability or non-inferiority. MATH500's interval includes zero. Neither result establishes release readiness.

Each score averages three decoding settings within each matched item: greedy, then seeds 17 and 29 at temperature 0.7/top-p 0.95, all capped at 8,192 generated tokens. The registered analysis resamples matched items 10,000 times with NumPy seed 297, retaining each item's three decode outcomes together; intervals are unadjusted. Correct counts in that order are GSM8K A8a **1199/1201/1201**, v0.1 **1216/1209/1220**; MATH500 A8a **460/466/464**, v0.1 **461/461/458**. All twelve model × benchmark × decode combinations have complete native scored coverage. Evaluator-marked MATH500 outcomes with no extractable answer remain in the scored denominator: A8a **23/20/24**, reference **28/24/29**; GSM8K has none. No missing-score imputation, best-decode selection or historical-reference pooling occurs.

Actual native records, item arrays, model/server-tokenizer logs and the first three finalized sampling-control markers for each unit agree with the registered settings. These markers are not full wire-HTTP captures. The configured/server context is 32,768 tokens; the client uses 32,704 through the frozen 64-token margin. The client loads no tokenizer; the server uses each model's bound HF tokenizer.

The launched parent job and its recorded evaluator-source archive are identified. Orchestration metadata for the child evaluation jobs omits executable, code-bundle and working-directory details, so it cannot independently establish every worker code file or the effective child retry policy. All physical attempts are retained; none retried. The source files and observed scientific settings support this comparison with that observability limit. This analysis authorizes no model run or storage action.

Evidence: [registered method](../bfcl-noise-r1/A8a-paired-math-preregistration-marten-r2.json), [full analysis and aligned arrays](paired-math-analysis-marten-r1.json), [native record/array joins](actual-twelve-done-and-array-joins-marten-r1.json), [runtime joins](actual-runtime-CLI-server-source-joins-marten-r1.json), [RPC-readback qualification](child-RPC-observability-clarification-marten-r1.json).
