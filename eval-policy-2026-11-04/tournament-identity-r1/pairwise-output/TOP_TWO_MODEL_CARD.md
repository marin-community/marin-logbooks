# Snowball tournament finalists: head-to-head model card

This card compares the first- and second-ranked checkpoints from the completed uncertainty-aware Snowball/Grug tournament. It uses observed scores only.

## Models

| Tournament rank | Checkpoint | Mean pairwise win rate | Cohort W-L-T |
| ---: | --- | ---: | ---: |
| 1 | [open-athena/Snowball-Smartie-RL-2026.10.15.32-step14](https://huggingface.co/open-athena/Snowball-Smartie-RL-2026.10.15.32-step14) | 0.522 | 3-1-42 |
| 2 | [open-athena/Snowball-67B-A2B-Identity-PathB-A3-2026.10.30](https://huggingface.co/open-athena/Snowball-67B-A2B-Identity-PathB-A3-2026.10.30) | 0.511 | 2-1-43 |

## Direct result

The champion wins the direct matchup **1-1-21** across 23 benchmarks, for a paired win rate of **0.500**. Its decisive-win margin is **+0 benchmark**. The tournament title comes from performance against the entire candidate cohort, not from a dominant direct result against the runner-up.

A benchmark is decisive only when `abs(score difference) > 1.96 x hypot(adjusted SEMs)`. The adjusted SEM includes the policy factor `1 / sqrt(0.9)`; otherwise the benchmark is a tie.

## Decisive benchmarks

| Benchmark | Champion | Runner-up | Margin | Winner |
| --- | ---: | ---: | ---: | --- |
| mrcr | 0.211 | 0.241 | -3.0 pp | Runner-up |
| sotopia-hard | 0.659 | 0.631 | +2.9 pp | Champion |

## Complete selection panel

| Benchmark | Champion score (95% CI) | Runner-up score (95% CI) | Margin | Decision |
| --- | ---: | ---: | ---: | --- |
| math500 | 0.944 [0.923, 0.965] | 0.954 [0.935, 0.973] | -1.0 pp | Tie |
| humanevalplus | 0.909 [0.862, 0.955] | 0.841 [0.782, 0.901] | +6.7 pp | Tie |
| mbppplus | 0.770 [0.725, 0.815] | 0.743 [0.697, 0.790] | +2.6 pp | Tie |
| olympiadbench | 0.347 [0.290, 0.404] | 0.330 [0.274, 0.386] | +1.7 pp | Tie |
| gsm8k-0shot | 0.928 [0.913, 0.943] | 0.919 [0.903, 0.934] | +0.9 pp | Tie |
| winogrande | 0.765 [0.740, 0.789] | 0.759 [0.734, 0.784] | +0.6 pp | Tie |
| boolq | 0.834 [0.820, 0.847] | 0.846 [0.833, 0.859] | -1.2 pp | Tie |
| truthfulqa | 0.488 [0.458, 0.519] | 0.483 [0.452, 0.513] | +0.6 pp | Tie |
| aime24 | 0.730 [0.677, 0.783] | 0.703 [0.649, 0.758] | +2.7 pp | Tie |
| mmlu-pro | 0.772 [0.764, 0.780] | 0.767 [0.759, 0.775] | +0.5 pp | Tie |
| gpqa-diamond | 0.648 [0.608, 0.689] | 0.645 [0.604, 0.685] | +0.3 pp | Tie |
| cruxeval | 0.709 [0.686, 0.733] | 0.713 [0.690, 0.736] | -0.4 pp | Tie |
| ifbench | 0.263 [0.211, 0.316] | 0.260 [0.208, 0.312] | +0.3 pp | Tie |
| mrcr | 0.211 [0.191, 0.231] | 0.241 [0.220, 0.262] | -3.0 pp | Runner-up |
| nupa | 0.674 [0.660, 0.687] | 0.686 [0.672, 0.699] | -1.2 pp | Tie |
| tau3-pi | 0.091 [0.060, 0.122] | 0.096 [0.065, 0.127] | -0.5 pp | Tie |
| swebench-verified | 0.252 [0.212, 0.292] | 0.287 [0.245, 0.328] | -3.5 pp | Tie |
| ot-tblite-recovery | 0.225 [0.176, 0.274] | 0.180 [0.135, 0.225] | +4.5 pp | Tie |
| tb2-recovery | 0.092 [0.055, 0.128] | 0.106 [0.067, 0.146] | -1.5 pp | Tie |
| ds-1000-local | 0.240 [0.177, 0.303] | 0.245 [0.182, 0.308] | -0.5 pp | Tie |
| bfclparity-pi | 0.237 [0.154, 0.319] | 0.285 [0.200, 0.369] | -4.8 pp | Tie |
| bixbench-pi | 0.073 [0.036, 0.111] | 0.117 [0.071, 0.164] | -4.4 pp | Tie |
| sotopia-hard | 0.659 [0.652, 0.667] | 0.631 [0.623, 0.639] | +2.9 pp | Champion |

## Benchmarks excluded from selection

FinanceBench and PIQA were excluded as non-discriminating within the candidate cohort. TriviaQA was excluded because its answer extraction was overly sensitive to response format. Their point estimates are shown for completeness but did not affect either rank.

| Benchmark | Champion | Runner-up | Margin |
| --- | ---: | ---: | ---: |
| piqa | 0.820 | 0.819 | +0.1 pp |
| triviaqa | 0.570 | 0.564 | +0.5 pp |
| financebench | 0.460 | 0.560 | -10.0 pp |

## Interpretation and limitations

The champion's decisive advantages are sotopia-hard; the runner-up's are mrcr. A statistical tie is not evidence that two checkpoints are equivalent; it means this evaluation did not resolve their difference at the configured threshold. Benchmark metrics share a `[0, 1]` direction but are not otherwise interchangeable, so raw score margins should be interpreted per benchmark.

## Provenance

- Tracker SHA-256: `5dd06aac4724904dcf700466026fedd56b14c575207ca63005860ee1e5a3be0d`
- Recovered-metrics SHA-256: `e5780a952247456b2c0a8ad7dc58fcea29d3543600fd1f66f398895dede710d9`
- Selection panel: 23 benchmarks
- Candidate cohort: 3 checkpoints
- Detailed machine-readable comparison: `top_two_head_to_head.csv`
- Tournament source: `snowball_benchmark_pair_outcomes.csv` and `snowball_pairwise_summary.csv`
