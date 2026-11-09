# Snowball-lineage tournament

Top checkpoint: `open-athena/Snowball-Smartie-RL-2026.10.15.32-step14`.

Source tracker SHA-256: `5dd06aac4724904dcf700466026fedd56b14c575207ca63005860ee1e5a3be0d`. The tournament requires complete observed scores for all 3 candidates on 23 benchmarks. FinanceBench, PIQA, and TriviaQA are excluded. Comparisons use full-precision aggregates recovered from the durable trial evidence; tracker values are three-decimal presentation values only. A benchmark is a win only when the adjusted 95% confidence interval for its paired score difference excludes zero; otherwise it is a tie. The adjusted SEM includes the policy's infrastructure-failure tolerance factor of `1 / sqrt(0.9)`.

Finalist comparison: [`TOP_TWO_MODEL_CARD.md`](TOP_TWO_MODEL_CARD.md).

| Rank | Model | Mean paired win rate | Pooled win rate | W-L-T | Benchmark comparisons |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | S3 | 52.2% | 52.2% | 3-1-42 | 46 |
| 2 | S2 | 51.1% | 51.1% | 2-1-43 | 46 |
| 3 | S1 | 46.7% | 46.7% | 0-3-43 | 46 |

| Model | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| S1 | — | 47.8% (0-1-22) | 45.7% (0-2-21) |
| S2 | 52.2% (1-0-22) | — | 50.0% (1-1-21) |
| S3 | 54.3% (2-0-21) | 50.0% (1-1-21) | — |

| ID | Model |
| --- | --- |
| S1 | open-athena/Snowball-67B-A2B-Identity-PathB-A13-2026.10.30 |
| S2 | open-athena/Snowball-67B-A2B-Identity-PathB-A3-2026.10.30 |
| S3 | open-athena/Snowball-Smartie-RL-2026.10.15.32-step14 |

Each benchmark receives equal weight within a model pair. A win scores 1, a tie 0.5, and a loss 0. The selection criterion is the unweighted mean of each candidate's 2 pairwise rates. The matrix shows `win rate (wins-losses-ties)`.
