# A6: BFCL deficit remains

A6's five complete Berkeley Function Calling Leaderboard evaluations score 21, 16, 17, 23 and 18 of 123. Mean accuracy is 15.45%, sample SD 2.37 percentage points, range 13.01–18.70%; the unadjusted bootstrap 95% interval for the mean is [13.66, 17.40]%.

| Pre-specified contrast | Difference (pp) | Unadjusted 95% interval (pp) |
|---|---:|---:|
| A6 − fresh v0.1 | −12.20 | [−15.61, −9.27] |
| A6 − A3 | −15.61 | [−19.19, −12.36] |
| A6 − A0 | −0.14 | [−4.39, +3.66] |

A0, A3 and A6 each train from plain v0.1 for 36 updates. The fresh v0.1 reference comprises five new evaluations of the unmodified base model. ON/OFF denote thinking-enabled/disabled renderings. A0 uses empty thinking spans in identity-ON rows; A3 substitutes each source conversation's paired OFF answer. A6 inserts existing GLM-teacher reasoning before that identity answer where a trace passed the prior pool review; fallback rows are identical to A3's OFF rows. Here A6 does not restore tool-calling accuracy to A3 or plain v0.1. Its interval against A0 includes zero; this does not establish equivalence.

The [original-parser identity read](../ablation-identity-a6-r1/read-marten-r1.md) covers 52 core prompts and 14 third-person controls in both modes, 132 responses total. A6 ON has 52 own-name credits, 50 creator credits, zero foreign self-claims and 2 visible-planning format breaks among 52; OFF has 30 name credits, 32 creator credits and one foreign claim. Name means “Marin's Latest MoE”; the historical creator criterion accepts the Marin Community or Open Athena, excluding maintenance-only credit. Format breakage requires empty parsed reasoning plus exposed planning/instruction commentary, not emptiness alone. Eight controls per mode receive canned nonanswers; OFF also falsely attributes real third-party products to the wrong developer in one response. Identity credits do not establish a release pass.

The pre-specified analysis uses 10,000 independent run-level bootstrap draws, Python seed 297. It preserves all earlier A0–A4 and fresh-reference intervals exactly, then continues the same random stream for A6. All five A6 launches, audited-export bindings, native samples and 615 binary rewards join exactly; no replacement was needed. Only the five new reference runs enter this comparison; historical diagnostics remain separate.

There is one trained checkpoint per arm and only five or six evaluation-run clusters. Tasks are fixed, batches were not time-randomized, and intervals are unadjusted. A ledger occurrence is a scheduled training example; final tokens exclude reasoning, and replay comprises math, code, chat and tool examples. A6's 36-update cutoff consumes 175 fewer examples than A0/A3: identity final tokens 126,492 versus 129,641, replay final tokens 616,153 versus 635,510, plus 55,073 additional identity reasoning tokens. Of A6's ON-assigned identity occurrences, 1,791 use traces and 1,235 use OFF answers. This is not a dose-matched, pure reasoning-only causal comparison.

Sources: [frozen prospective specification](A6-run-comparison-extension-spec-marten-r1.json) (`c2cd0734`), [complete analysis and admission checks](A6-run-bootstrap-extension-marten-r1.json) (`1dbd6ba0`), [five-run index](../ablation-a6-r1/A6-run-index-20261030T094546.225808Z.json) (`0f474b94`), and [fresh-reference analysis](fresh-v01-reference-extension-marten-r1.json) (`e6a6b5e2`).
