# A10 BFCL

A10 averages **23.9837%** on four complete BFCL evaluations. The registered comparison with plain v0.1 is −3.6585 percentage points, with a 95% run-bootstrap interval of [−8.0488, +0.0407]. This interval does not establish preservation within five points. BFCL here means the frozen 123-task tool-calling parity slice of the Berkeley Function Calling Leaderboard, shared across all conditions; these percentages do not describe the full benchmark.

Plain v0.1 is the published Marin-Snowball-67B-A2B-v0.1 checkpoint. “Fresh” refers to its five new evaluation runs, not another training run; the older historical reference score is excluded. A3 and A8a both train from v0.1 for 36 updates: A3 renders all identity examples without reasoning, while A8a retains reasoning in a deterministic 10% subset of trace-eligible identity ON occurrences.

| A10 complete runs | Planned slots | Exhausted without score | Mean | Sample SD | Min–max | Mean 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 33, 30, 32, 23 correct /123 | 5 | 1 | 23.9837% | 3.6661 pp | 18.6992–26.8293% | [20.5285, 26.4228]% |

| Registered comparison | Difference, pp | Unadjusted 95% CI, pp |
|---|---:|---:|
| Primary: A10 − fresh v0.1 | −3.6585 | [−8.0488, +0.0407] |
| Secondary: A10 − A8a | −2.1951 | [−6.0976, +1.1789] |
| Secondary: A10 − A3 | −7.0732 | [−11.5041, −3.1707] |

All five logical evaluation slots are terminal. Four produced complete native-scored binary rewards for the same 123 tasks. Slot 2 exhausted its last permitted whole replacement after a task-repository fetch failure before tasks began. Its score remains absent. All 14 physical workflows, partial task results and failed attempts remain in the record; no valid score was rerun or selected from alternatives.

The [reduced-n amendment](BFCL-infra-exhaustion-analysis-addendum-marten-r1.json) uses the actual four complete run fractions, ordered by actual run ID, with 10,000 independent within-condition run resamples. It preserves the earlier Python seed-297 results and the reserved A9/A10/A11 random-stream block starts. A10 draws start from a copy of its reserved block state and sample four runs per draw. Comparator draws are shared across the three registered contrasts. No item bootstrap, repeat-number pairing, historical reference pooling or A9–A10 contrast was added. The reference, A8a and A3 each retain their existing five runs.

A10 continues training from v0.1 for 36 updates using 44 identity sources, four per family, each repeated ten times: nine answers without reasoning and one with its existing GLM reasoning trace. Identity sources are questions and answers teaching the name Marin's Latest MoE and development/training by the Marin Community. Their 11 families cover question types such as name, creator, training, maintenance and false-premise identity questions. Replay is the unchanged ordered stream of general chat, code, maths and tool-use training examples. Compared with A8a, identity final-text dose is 8,930 versus 126,492 tokens, identity reasoning is 1,313 versus 5,388, and replay final-text dose is 822,038 versus 616,153. Diversity, repetition, dose, placement and packing all change. The comparisons cannot isolate a reasoning-share effect.

The four actual native result records and all 492 scored sample rows were checked locally against the frozen reports and reward arrays, including grading flags and error semantics. Native evaluator/model configurations match the frozen profile except model name/location. Full typed and JSON sample-reader projections agree; they are not original Parquet bytes. Capability URLs remain in restricted source records and are omitted here.

These means describe complete-score survivors. Infrastructure completion need not be random, and four evaluation clusters give limited uncertainty estimates. One trained checkpoint and unadjusted intervals do not establish release acceptance or a universal mechanism. [Final analysis](A10-run-bootstrap-extension-marten-r1.json) (`37bc83d8411d62748be16e546fa70ec9aa73bc804fb302b3ad14770ceedcd62c`), [actual native readback](../A10-recovered-BFCL-five-r1/final-four-FULL123-native-record-and-full-samples-readback-margay-r1.json), and [registration](A10-run-comparison-preregistration-marten-r1.json) retain the evidence. Earlier published results remain unchanged.
