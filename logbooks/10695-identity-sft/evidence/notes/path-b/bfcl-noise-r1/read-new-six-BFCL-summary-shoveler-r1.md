# BFCL for the six new Path B models

All six registered BFCL analyses are written. Primary comparison: each model minus fresh v0.1 (five-run reference). Secondary: A8a for A12 and A13; A3 and A8a for the merges W(t) = (1−t)·A3 + t·A8a. Intervals are unadjusted run-cluster bootstrap 95% intervals; "n" is the number of complete 123-item runs out of five planned.

| Model | n | Mean % | − fresh v0.1, pp | − A3, pp | − A8a, pp | One-turn no-call stops (slot 1) | Identity intrusions (slot 1) |
|---|---:|---:|---|---|---|---:|---:|
| A12 | 5 | 14.80 | −12.85 [−16.42, −9.92] | | −11.38 [−14.15, −8.46] | 45 | 3 |
| A13 | 5 | 28.29 | +0.65 [−3.41, +4.39] | | +2.11 [−1.46, +5.69] | 16 | 1 |
| M25 (t = 0.25) | 5 | 30.41 | +2.76 [−0.81, +6.02] | −0.65 [−4.23, +2.93] | +4.23 [+1.30, +7.32] | 11 | 0 |
| M49 (t = 0.49) | 4 | 29.47 | +1.83 [−1.75, +4.88] | −1.59 [−5.28, +1.79] | +3.29 [+0.41, +6.18] | 14 | 0 |
| M51 (t = 0.51) | 4 | 23.78 | −3.86 [−7.93, +0.08] | −7.28 [−11.34, −3.13] | −2.40 [−5.73, +1.30] | 12 | 0 |
| M75 (t = 0.75) | 5 | 23.90 | −3.74 [−6.83, −1.30] | −7.15 [−10.41, −4.39] | −2.28 [−4.55, 0.00] | 13 | 1 |

What the numbers say:
- **A12** (18 updates) loses about 13 points on BFCL and returns to the A9/A10 level of one-turn no-call stops. Fewer updates did not protect tool use.
- **A13** (A8a plus doubled thinking-ON tool replay) shows no detectable loss against fresh v0.1 and keeps A8a's low stop count.
- **The merges split at t ≈ 0.5.** M25 and M49 sit at A3's level and above A8a; M51 and M75 sit at A8a's level and below A3. M49 and M51 differ by only 0.02 in t yet by 5.7 points in mean. No M49-versus-M51 interval is registered, so this is an observation from the table, not a tested difference.
- BFCL is one screen. Marten's paired maths finds that M25 and M49 trigger the registered GSM8K stop; A13, M51 and M75 do not.

Every result was replayed independently and exactly by margay (A12 by marten and margay). Two slots ended exhausted (M51 slot 1, M49 slot 1) after the original and both replacements were infrastructure-incomplete; they contribute no score and nothing is zero-filled. These are evaluations of one checkpoint per model; they do not estimate training variability, and none of them is a causal, noninferiority or release claim.

Notes: [A12 (marten)](read-A12-BFCL-marten-r1.md), [A13](read-A13-BFCL-shoveler-r1.md), [M25](read-M25-BFCL-shoveler-r1.md), [M49](read-M49-BFCL-shoveler-r1.md), [M51](read-M51-BFCL-shoveler-r1.md), [M75](read-M75-BFCL-shoveler-r1.md), [first-run diagnostic](../bfcl-failure-diagnostic-r1/read-new-six-first-original-diagnostic-shoveler-r1.md).
