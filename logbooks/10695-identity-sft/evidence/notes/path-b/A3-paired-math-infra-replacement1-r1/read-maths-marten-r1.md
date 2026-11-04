# A3 paired maths

A3 triggers the registered GSM8K stop rule against plain v0.1. MATH500 does not trigger it. A3 continues training from v0.1 for 36 updates on replay data and identity answers that name Marin's Latest MoE and the Marin Community as its developer and trainer. All identity examples are rendered without reasoning. These evaluations use GSM8K thinking OFF and MATH500 thinking ON.

| Benchmark | Items | A3 accuracy | v0.1 accuracy | A3 − v0.1, pp | Paired 95% CI, pp | Registered stop |
|---|---:|---:|---:|---:|---:|---|
| GSM8K OFF | 1,319 | 90.3715% | 92.1152% | −1.7437 | [−2.6030, −0.9091] | Yes |
| MATH500 ON | 500 | 91.8000% | 92.0000% | −0.2000 | [−1.4000, +1.0000] | No |

Each item receives greedy decoding and sampled decodes with seeds 17 and 29. We average its three binary scores, then bootstrap the paired A3–v0.1 item differences: 10,000 resamples, NumPy seed 297, percentile intervals. The reference is the existing admitted six-unit plain-v0.1 maths evaluation; no new reference was run. The registered rule requires both a point loss greater than 1.5 percentage points and a confidence interval entirely below zero. GSM8K meets both conditions. Its interval does not establish that the true loss exceeds 1.5 points.

A3 correct counts, in greedy/17/29 order, are GSM8K 1196/1191/1189 and MATH500 460/459/458. Reference counts are 1216/1209/1220 and 461/461/458. All items were scored; MATH500's 20/22/21 unanswered A3 outputs remain in the scores. The first candidate execution was preempted before producing outputs; the one whole replacement supplied these six units.

Actual server logs identify the audited A3 model and tokenizer. All six client modes match the registration. The server context is 32,768 tokens; the client uses 32,704 after the inherited 64-token margin. Generation allows 8,192 tokens. Greedy requests have temperature zero and no HTTP seed; sampled requests use temperature 0.7, top-p 0.95 and seeds 17/29. Three finalized sampling-payload log entries per unit match these controls. These 18 entries are not a full HTTP capture. Stripped child-job metadata cannot establish live worker-file identity or effective retry policy. Item arrays are local native-reader projections checked against native completion records, coverage and headline scores, without an independent remote per-item verifier reread.

Runtime qualification therefore relies on the pinned source, saved owner records, sampled request-control logs and the operator's item-array extraction. It does not independently establish every live worker file or every request and verifier result. Undetected deviations outside those observations remain a limitation of the comparison.

This comparison covers one trained checkpoint, fixed public items and three decode settings. The intervals quantify paired-item uncertainty, not variation across independently trained models. The registered stop finding does not authorize cancellation or establish release acceptance.

Evidence: [registration](../bfcl-noise-r1/A3-A11-paired-maths-preregistration-marten-r1.json), [final analysis](paired-math-analysis-marten-r1.json) (`49d44ef3d5c6ad4efc5da0f4bdf06723c062ee30c8629863e0feee0a2e6084cb`), [runtime qualification](actual-runtime-qualification-r1/actual-runtime-qualification-margay-r1.json), [unchanged conditional statistics](paired-math-statistics-pending-runtime-marten-r1.json). Earlier A8a maths and BFCL analyses remain unchanged.
