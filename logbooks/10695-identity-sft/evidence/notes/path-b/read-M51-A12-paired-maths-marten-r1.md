# M51 and A12 paired maths

Neither model triggers the registered maths STOP rule. M51 has a negative GSM8K interval; its point estimate does not cross the rule’s −1.5 percentage-point threshold. A12’s intervals include zero on both benchmarks.

| Candidate | Benchmark | Candidate accuracy | v0.1 accuracy | Difference, pp | Paired 95% interval, pp | STOP |
|---|---|---:|---:|---:|---:|---|
| M51 | GSM8K, thinking OFF | 90.953% | 92.115% | −1.162 | [−1.971, −0.354] | No |
| M51 | MATH500, thinking ON | 92.867% | 92.000% | +0.867 | [−0.200, +2.000] | No |
| A12 | GSM8K, thinking OFF | 92.292% | 92.115% | +0.177 | [−0.708, +1.087] | No |
| A12 | MATH500, thinking ON | 92.067% | 92.000% | +0.067 | [−1.000, +1.133] | No |

M51 is the linear merge 0.49×A3 + 0.51×A8a, with the registered frozen-tensor and special-row preservation. A12 trains fresh from v0.1 on A8a’s first 18 data updates. Recovery re-exported its existing final native checkpoint without retraining. Each comparison uses the same admitted fresh-v0.1 reference outputs. Accuracy averages three decodes per item: greedy, seed17 and seed29. Intervals resample paired item-mean differences over all 1,319 GSM8K or 500 MATH500 items, using the registered NumPy seed297 and 10,000 draws. No M51–A12 contrast was added.

STOP requires both a point difference below −1.5 pp and an interval upper bound below zero. Its absence does not establish noninferiority, no loss or release acceptance. The M51 GSM8K interval includes losses larger than 1.5 pp. These item-resampling intervals do not measure variation across training runs or checkpoints.

All six benchmark×decode units per model have complete binary item arrays and matching native configs, coverage and done records. Runtime evidence joins the actual candidate and source archive, server HF/tokenizer, BF16/TP1/DP8/EP/grug configuration, each client’s thinking mode and the first three final sampling payload markers per unit. Greedy uses temperature0 without an HTTP seed; sampled decodes use temperature0.7, top_p0.95 and seed17/29. Generation is capped at 8,192 tokens; the server context is 32,768 and the client retains the inherited 64-token margin at 32,704. The 18 markers per model do not attest every request. Global federated child-status fields do not attest original peer policy or worker bundle contents: effective child retry/timeout settings and complete live child-code identity remain unverified. Sampling and server qualifications use the actual logs described here.

Evidence: [M51 final analysis](M51-paired-math-r1/paired-math-analysis-marten-r1.json), [A12 final analysis](A12-paired-math-r1/paired-math-analysis-marten-r1.json), [registered maths method](bfcl-noise-r1/A8a-paired-math-preregistration-marten-r2.json), [merge registration](bfcl-noise-r1/M25-M49-M51-M75-merge-sweep-preregistration-marten-r2.json), [A12/A13 registration](bfcl-noise-r1/A12-A13-science-and-analysis-preregistration-marten-r1.json). The final analyses bind the runtime qualifications and aligned item arrays. Previously sealed BFCL and maths score arrays and analysis files were not modified; the v0.1 maths outputs were reused. No reference serve, generation or score repair was added for this analysis.
