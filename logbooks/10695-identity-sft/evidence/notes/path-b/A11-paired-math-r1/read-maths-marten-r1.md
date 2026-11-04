# A11 paired maths

A11 triggers neither registered maths stop rule against plain v0.1. GSM8K still shows a negative paired interval, and the evidence does not establish that its loss is within 1.5 percentage points.

A11 continues training from Marin-Snowball-67B-A2B-v0.1 for 72 updates on replay data and identity answers naming Marin's Latest MoE and the Marin Community. Identity training examples contain no reasoning. The reference reuses the existing plain-v0.1 maths evaluation, which passed the earlier source, runtime and coverage checks. It uses the same item IDs, benchmark/prompt-generation configurations, decoding settings and scoring procedure; no new reference was run.

| Benchmark | Items | A11 accuracy | v0.1 accuracy | A11 − v0.1, pp | Paired 95% CI, pp | Registered stop |
|---|---:|---:|---:|---:|---:|---|
| GSM8K, thinking OFF | 1,319 | 90.7758% | 92.1152% | −1.3394 | [−2.1481, −0.5307] | No |
| MATH500, thinking ON | 500 | 91.7333% | 92.0000% | −0.2667 | [−1.4000, +0.8667] | No |

Each item receives greedy decoding and sampled decodes with seeds 17 and 29. We average its three binary scores, then bootstrap paired A11–v0.1 item differences: 10,000 resamples, NumPy seed 297, percentile intervals. The registered stop requires **both** a point loss greater than 1.5 points and an interval entirely below zero. Neither benchmark meets both conditions. This does not establish noninferiority, absence of loss or release acceptance. No A3–A11 contrast was added.

A11 correct counts, in greedy/17/29 order, are GSM8K 1192/1203/1197 and MATH500 453/464/459. Reference counts are 1216/1209/1220 and 461/461/458. All items were scored; MATH500's 33/22/25 unanswered A11 outputs remain in the scores.

Actual server logs identify the audited A11 model and tokenizer, with BF16, tensor parallelism 1, data parallelism 8, expert parallelism and the inherited reasoning parser. Client settings match the registration for all six benchmark/decode combinations. Server context is 32,768 tokens; client context is 32,704 after the inherited 64-token margin. Generation allows 8,192 tokens. Greedy sampling has temperature zero and no HTTP seed; sampled decodes use temperature 0.7 and top-p 0.95. Three finalized sampling-payload entries per combination match these settings. These 18 entries are not full HTTP or all-request capture. The job-status API omits child code/workspace details, so it cannot attest all live worker files or effective retry policy. Item-score arrays were extracted locally by the operator's native-results reader and checked against completion records, coverage and headline scores, without an independent remote per-item verifier reread.

This comparison covers one checkpoint per condition, fixed public items and three decode settings. Intervals quantify paired-item uncertainty, not variation across independently trained models. [Final analysis](paired-math-analysis-marten-r1.json) (`0be6c5ff54e35fb26c2cd85352a0791a913557361578bb19de054a8bc07f5e2e`), [registration](../bfcl-noise-r1/A3-A11-paired-maths-preregistration-marten-r1.json), [actual runtime qualification](actual-runtime-qualification-margay-r1.json) and [unchanged conditional statistics](paired-math-statistics-pending-runtime-marten-r1.json) retain the evidence. Earlier maths and BFCL results remain unchanged.
