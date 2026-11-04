# BFCL ablation against five fresh v0.1 runs

The five fresh unmodified-v0.1 reference runs average27.64% on123 tool-use tasks. A0 and A1 remain below this reference. A3's run-bootstrap contrast spans zero; A4's is positive. A2 has a smaller negative contrast whose unadjusted interval also excludes zero.

The reference is Marin Snowball67B-A2B v0.1 (`adc0078a`). Each trained arm has36 updates from those weights. Evaluations use the pinned123-task BFCL-Parity tool-use suite through CABA/Harbor/Pi (`caba0fa38b`), with source-derived effective temperature1.0 and no seed. Each admitted run has all123 unique binary rewards. The five reference counts, in logical-slot order, are34,32,41,31,32. Their average happens to equal the historical34/123 point; that historical run is excluded.

Identity ON examples in A0 contain an empty bounded reasoning span before the answer; OFF examples omit that span. A3 replaces each ON occurrence with the same original conversation's OFF version. Replay consists of teacher responses to general and tool-use prompts; A2 fills the fixed training budget with replay only. A1 makes the256 special-token embedding/output rows trainable while retaining A0's other parameter freezes.

| Condition | Runs | Mean % | Sample SD, pp | Range % | Difference from fresh v0.1, pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| v0.1, unchanged reference | 5 | 27.64 | 3.30 | 25.20–33.33 | — |
| A0: empty-think ON identity control | 6 | 15.58 | 4.92 | 11.38–23.58 | −12.06 [−16.45, −7.56] |
| A1: A0 without special-row freezes | 5 | 14.80 | 2.26 | 13.01–18.70 | −12.85 [−16.26, −9.92] |
| A2: replay only | 5 | 24.55 | 1.67 | 21.95–26.02 | −3.09 [−6.18, −0.65] |
| A3: same-source OFF replaces identity ON | 5 | 31.06 | 3.61 | 27.64–36.59 | +3.41 [−0.49, +7.15] |
| A4: historical T3 recipe extended to36 | 5 | 36.26 | 3.48 | 30.08–38.21 | +8.62 [+4.39, +11.87] |

The reference mean's unadjusted95% interval is[25.69,30.57]%. The pre-specified extension uses10,000 independent run-level bootstrap draws, Python seed297 and lexical actual run IDs. It reproduces the original26-run draw stream first, then continues that stream for the five-run reference resamples. Each contrast subtracts an independent reference draw, shared across arms. There is no task resampling or pairing by repeat number. The [original arm-only analysis](primary-run-bootstrap-marten-r1.json) (`06f71ba8`), including every interval, is unchanged.

Nine physical reference workflows are retained: five complete scored runs and four incomplete originals, with114/118/119/116 observed rewards. All missing rewards in those originals have infrastructure-timeout grading. The authorized whole-run replacements are linked; partial outcomes are neither zero-filled nor pooled. Diagnostics from the earlier stopped r5 training cohort are also excluded.

These unadjusted intervals use only five or six evaluation-run clusters, one trained checkpoint per arm and a fixed task set. The later reference batch is not time-randomized against the arms. A2 receives more replay than A0. A4 extends the earlier T3 training recipe: it uses unpacked examples, a much smaller identity dose, only router-bias full-field freezes and no special-row freeze. It changes several factors together. These results do not establish replicated-training effects, equivalence, or a release pass. The earlier screens still expose planning in42/52 A3 ON answers and50/52 A4 ON answers; A4 also has5/14 OFF third-person creator rewrites. See the [A3/prefill reading](../prefill-p1-r1/read-marten-r1.md) and [A4 reading](../ablation-identity-a4-r1/read-marten-r1.md).

[Complete extension and input hashes](fresh-v01-reference-extension-marten-r1.json) (`e6a6b5e2`); [pre-specified extension](reference-extension-spec-marten-r1.json) (`4e449bc7`); [five-slot/nine-workflow index](../bfcl-v01-reference5-r1/reference-run-index-20261030T082646.869642Z.json) (`69b3a03e`).
