# BFCL ablation: primary repeated-run analysis

A3's thinking-OFF replacement improves mean BFCL over A0 by **15.47 percentage points [10.76, 19.92]**. Removing special-token row freezes in A1 gives **−0.79 points [−5.09, +3.01]**, with no observed rescue. These unadjusted intervals do not establish equivalence or a unique cause.

All **26 planned score slots** have complete, hash-verified rewards for the same 123 tasks. All 41 physical workflows are retained, including 15 infrastructure non-scores and authorized replacements. No valid score was replaced. Original single-run comparisons are archived and **superseded for decisions**.

These are repeated stochastic evaluations of one trained checkpoint per arm, all at 36 completed updates. A0 is SFT from v0.1 with identity answers plus replay, targeting 10% raw identity-token share. It freezes entire Q/K projection, router and bias tensors as well as the 256 special-token embedding/head rows. A4 uses the earlier CD16-T3 data/recipe, including a much smaller identity dose and only router-bias freezes; exact configurations and data are in the [ablation packet](../bfcl-ablation-r1/PACKET.md).

| Arm | Change from A0 | Runs | Mean % | Sample SD, pp | Min–max % | Mean 95% CI % |
|---|---|---:|---:|---:|---:|---:|
| A0 | r5 B-base 10% control | 6 | 15.58 | 4.92 | 11.38–23.58 | 12.20–19.51 |
| A1 | Special-token row freezes removed | 5 | 14.80 | 2.26 | 13.01–18.70 | 13.50–16.75 |
| A2 | Replay only; no identity rows | 5 | 24.55 | 1.67 | 21.95–26.02 | 23.25–25.69 |
| A3 | Each identity ON occurrence replaced by the same source's OFF row | 5 | 31.06 | 3.61 | 27.64–36.59 | 28.46–34.15 |
| A4 | Historical CD16-T3 recipe extended to 36 updates | 5 | 36.26 | 3.48 | 30.08–38.21 | 33.17–38.05 |

All A0–A3 use unit loss weights. A1 retains the entire-tensor freezes. A3 preserves A0's 129,641 supervised identity final-answer lexical tokens, excluding reasoning/control tokens, and replay order, with different packing/raw counts. A2 has more replay exposure. A4 changes multiple recipe/data factors.

| Contrast | Difference, pp | Unadjusted 95% CI, pp |
|---|---:|---:|
| A1 − A0 | −0.79 | −5.09 to +3.01 |
| A2 − A0 | +8.97 | +4.82 to +12.44 |
| A3 − A0 | +15.47 | +10.76 to +19.92 |
| A4 − A0 | +20.68 | +15.85 to +24.91 |

Approved spec `636ddeb0`: 10,000 independent within-arm run resamples, Python `random.Random(297)`, lexical actual run IDs, and mean run fractions. Each draw samples six A0 runs and five per other arm with replacement, sharing A0 across contrasts without pairing repeat numbers. CI indices are 249/9749; intervals are unadjusted. Old-r5 runs are excluded. Historical v0.1's 34/123 is descriptive only; a separate five-run reference extension is pending.

Pinned client/server sources imply effective temperature 1.0 and no seed; no actual HTTP sampling payload was captured. This is consistent with the spread, without explaining every difference.

A3's separate read of 52 core identity prompts with thinking enabled found **42/52 ON format breaks**: empty parsed reasoning plus planning/system commentary in the answer. Broad name credit was 42/52, with only 33 closing named deliverables; nine credits came from planning alone. The BFCL advantage does not remove that identity-format defect or establish release suitability. See the [identity ledger](../ablation-identity132-r1/role-aware-ledger-marten-r1.json).

Five or six clusters give imprecise bootstrap uncertainty. These intervals cover evaluation variability on fixed tasks/checkpoints, excluding task-population and independent training-run uncertainty. A2 exposure and A4 recipe differences limit attribution. No new experiment or release is authorized.

Evidence: [primary bootstrap JSON](primary-run-bootstrap-marten-r1.json), SHA256 `06f71ba85ac51e23471bba15bcb6198b2a02528f74af302b986fae39c466d9fa`; final [41-physical / 26-logical index](../bfcl-missing7-go-r1/versioned-run-index-20261030T063233.665229Z.json), SHA256 `9c5365a722ca3c5e0302a388bec1abea52d4056624ff6fc5282cc37e71cb3a89`. The JSON preserves all admitted run IDs, per-run counts and input hashes. Reader: `/mnt/storage/karan-scratch/marten-bfcl-run-bootstrap-r3.py`. Method: pinned Caba `caba0fa38b780af42da54af2f188ba5bfcc28501`; one H100x8 inference service per evaluation, unchanged model/configuration/method across an arm's repetitions.
