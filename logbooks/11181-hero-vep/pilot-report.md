# Marin Hero VEP: 100-variant development pilot

This is an exploratory experiment on ten autosomal missense matched groups (10 positives and 90 controls), using the final Hero SFT step-400407 model. It is not a clinical validation or an official benchmark result. Code and evidence remain on branches; merging into main is out of scope.

## Results

| Method | AUPRC | Group-bootstrap SE | Pathogenic versus matched-control wins |
| --- | ---: | ---: | ---: |
| null | 0.213 | 0.045 | 63.3% |
| healthy | 0.302 | 0.093 | 64.4% |
| zero-shot | 0.068 | 0.011 | 21.1% |
| two-shot | 0.263 | 0.088 | 71.7% |
| zero-shot/mapping-average | 0.094 | 0.028 | 35.6% |
| two-shot/mapping-average | 0.114 | 0.029 | 52.8% |

All primary scores average both strands. DNA scores are log P(reference) minus log P(alternate); classification scores are log P(pathogenic letter) minus log P(benign letter). Letter swaps are semantically corrected before comparing scores. Mapping averages are secondary diagnostics. Ties count as half a pairwise win. Bootstrap uses 2,000 paired match-group resamples and seed 11181, through the pinned MarinDNA evaluator. Ten groups are below its 30-group threshold for standard subset reporting.

The healthy prefix improves pilot AUPRC by 0.089 over null, but the paired bootstrap interval spans −0.044 to +0.220. The pilot therefore does not establish a reliable conditioning benefit. Primary two-shot classification outperforms primary zero-shot here, but this comparison is strongly sensitive to the answer mapping: mapping-averaged two-shot AUPRC is only 0.114.

## Sensitivity and representative failures

- Swapping A/B assignments flips the sign for 94/100 zero-shot and 76/100 two-shot scores after semantic alignment. It reverses 50/90 and 36/90 strict positive-control pair orders, respectively; another 3 and 5 pairs involve ties. This is model/prompt sensitivity, not a missed sign correction.
- Mean absolute forward/reverse score difference is 5.393 log units for null DNA and 5.639 for healthy DNA, versus 0.158 for zero-shot and 0.181 for two-shot. Their score scales differ, so these magnitudes are not directly comparable as quality metrics.
- The six preselected explicit reference/alternate format checks shift the zero-shot score by +0.318 on average and flip its sign in four cases. Six cases do not support selecting a superior representation.
- Positive 9:36234105:C:T ranks last among its ten-member group under both DNA methods; control 9:100473674:C:G ranks highest. Positive 17:61859862:G:C ranks eighth under both.
- Primary zero-shot ranks positives 3:9942863:T:C and 11:533875:G:T last in their groups. Two-shot still ranks positive 13:113118831:T:G only 6.5th, counting ties. See per-variant scores and all group ranks in the companion JSON files.

## Execution and stability

Source: 4cbc70b7cbcda68b17d362cf1d468b74664e6262. All 2,448 panel scores passed matrix completeness and input-hash validation. The run took 3,977 seconds (66m17s), made 6,285 HTTP requests, all successful, and processed 928,974 prompt tokens. There were 2,760 completion requests, including six sentinel baselines and 306 repeats. Every repeat matched exactly, including the final check. All 248 targets shared with the prior paced trial also match exactly.

One client used a fixed rank and a one-second gap after each completion. Completion latency excluding pacing was 0.395 seconds median, 0.635 seconds p95 and 0.974 seconds maximum. The run remained below 10,000 HTTP requests and 90 minutes. This is evidence for the guarded sequential workaround; it is not a server-wide determinism guarantee or a measurement of impact on other users. No prefix-cache savings have been established. No deployment settings were changed.

## Frozen-method continuation

Retain the original four primary methods, exact templates, two demonstrations, both-strand averages, original score signs, and reserved 300-variant/30-group development manifest. Retain mapping averages as secondary diagnostics; do not select a mapping or invert a score based on these pilot outcomes. Keep the held-out test split unused. No further explicit-pair sweep is planned for validation.

The fresh validation panel has 7,200 targets and 906 sentinel calls. Its cap is 30,000 total HTTP requests and four hours, at unchanged rank, serialization, pacing and stop thresholds. Its initial sentinels and serving fingerprint must match the pilot. The expected runtime at the observed completion rate is approximately 3h15m. Validation has not started in this report.

## Reproduction

Aggregate the complete saved scores using aggregate.py, then run report_metrics.py with --locked --fork-strategy fewest --prerelease explicit, --bootstrap-replicates 2000 and --seed 11181. Set PYTHONPATH to the pinned MarinDNA evals_v2/src directory; report_metrics.py checks the native metrics source hash. report_sensitivity.py produces the strand, mapping, format and positive-rank summaries without model calls. The native metric calculation completed in 19.62 seconds with peak RSS 148,648 KiB and exit status 0.

Companion artifacts: pilot-100-summary.json, pilot-100-metrics.json, pilot-100-variants.json, pilot-100-sensitivity.json, pilot-100-overlap.json and validation-protocol-frozen.json. Raw HTTP, authentication/deployment context and service URLs remain private; raw score and guard hashes are recorded in the public summary.
