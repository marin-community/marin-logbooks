# A13 and remaining merge maths

M25 and M49 trigger the registered GSM8K STOP rule. A13 and M75 have negative GSM8K intervals but do not cross the point-estimate threshold. None triggers STOP on MATH500.

| Candidate | GSM8K difference, pp | Paired 95% interval | STOP | MATH500 difference, pp | Paired 95% interval | STOP |
|---|---:|---:|---|---:|---:|---|
| A13 | −1.112 | [−1.921, −0.329] | No | +0.200 | [−1.067, +1.467] | No |
| M25 | −1.542 | [−2.401, −0.708] | Yes | +0.400 | [−0.867, +1.733] | No |
| M49 | −1.592 | [−2.451, −0.758] | Yes | +0.067 | [−1.267, +1.400] | No |
| M75 | −1.213 | [−2.072, −0.379] | No | +0.800 | [−0.467, +2.067] | No |

Each difference is candidate minus the same admitted fresh-v0.1 reference. STOP requires both a point difference below −1.5 percentage points and an interval upper bound below zero. Its absence does not establish noninferiority, no loss or release acceptance. The A13 and M75 GSM8K intervals include losses larger than 1.5 pp. No job cancellation or new experiment follows automatically from this analysis.

A13 duplicates thinking-ON tool-replay appearances and repacks to 36 updates under the [A12/A13 registration](bfcl-noise-r1/A12-A13-science-and-analysis-preregistration-marten-r1.json). The merges use W(t)=(1−t)A3+tA8a at t=0.25, 0.49 and 0.75 with the registered tensor/row preservation. The [maths method](bfcl-noise-r1/A8a-paired-math-preregistration-marten-r2.json) averages greedy/seed17/seed29 scores within each item, then resamples paired item-mean differences using NumPy seed297 and 10,000 draws independently per benchmark. Intervals are the 2.5th and 97.5th percentiles of those bootstrap means, using NumPy’s default linear interpolation. All 1,319 GSM8K items use thinking OFF; all 500 MATH500 items use thinking ON. Intervals describe paired item variation, not independent training or merge replicates. No candidate-to-candidate contrast was added.

Each candidate has six complete benchmark×decode units, with 5,457 binary item scores and matching native done/config/coverage records. The independently checked runtime evidence joins the committed candidate, source archive, actual server HF/tokenizer/BF16/TP1/DP8/EP/grug settings, six client modes and 18 original final-sampling marker lines. Greedy uses temperature0 without an HTTP seed; sampled decodes use temperature0.7/top_p0.95 and seed17/29. Generation max is8,192; server context32,768/client32,704 retains the frozen64-token margin. Those first three markers per unit do not attest every request. Global child-status projections do not establish effective peer retry/timeout policy or complete worker-code identity.

Final analyses: [A13](A13-paired-math-r1/paired-math-analysis-marten-r1.json), [M25](M25-paired-math-r1/paired-math-analysis-marten-r1.json), [M49](M49-paired-math-r1/paired-math-analysis-marten-r1.json), [M75](M75-paired-math-r1/paired-math-analysis-marten-r1.json). They bind runtime qualifications and aligned item arrays. Existing reference outputs were reused; previously sealed BFCL/maths arrays and analyses were not changed. No new reference serve, generation or score repair was performed.
