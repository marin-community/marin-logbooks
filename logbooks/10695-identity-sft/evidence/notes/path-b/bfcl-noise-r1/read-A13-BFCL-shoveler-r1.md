# A13 BFCL

A13 shows no detectable BFCL loss against fresh v0.1. Its five full runs score 40, 39, 34, 32 and 29 of 123 (original slot order), for a mean of 28.293% and a sample standard deviation of 3.787 percentage points. The run-bootstrap mean interval is [25.366%, 31.220%].

| Registered comparison | Difference, pp | Unadjusted 95% interval, pp |
|---|---:|---:|
| A13 − fresh v0.1 | +0.650 | [−3.415, +4.390] |
| A13 − A8a | +2.114 | [−1.463, +5.691] |

The fresh-v0.1 comparison is primary; A8a is secondary, and no other contrast is added. Both intervals include zero, and the point estimate against fresh v0.1 is far above the owner's −5 pp line. These are five evaluations of one checkpoint; they do not estimate variation across training runs, and they do not establish that the doubled thinking-ON tool replay caused the difference from A8a.

The five logical slots took seven physical workflows. Slot 4's original (122 scored, 1 AddTestsDirError) and its first replacement (121 scored, 2 EnvironmentStartTimeoutError) were infrastructure-incomplete; both are retained with NULL outcomes, and slot 4's second authorized replacement supplies its 32/123. No partial score enters the mean, no missing outcome is zero-filled, and no valid score was rerun.

Method: the registered run-cluster bootstrap (d3c966d9, with the reduced-n rules of f44554ba, which were not needed at n = 5). The script replays marten's frozen prefix through A11 exactly, starts A13 at canonical state cde9ce3c, draws 10,000 means of five choices from the lexically ordered admitted run IDs with Python Random, shares each draw across both contrasts against the frozen comparator draws, and uses sorted endpoints 249/9749. Run-ID order equals job order here. The canonical stream leaves state c3a6106b reserved for M25. Before use on A13, the same script reproduced marten's A12 result exactly (mean, interval, both contrasts and the A13 start state).

Each physical workflow was joined independently: native record, launch, HF and readiness binding, full JSON and 23-column typed projections, 123 fixed task IDs, binary rewards, coverage and error counts. Reader projections are not original Parquet bytes or an independent verifier recomputation.

Evidence: [final analysis](A13-run-bootstrap-extension-shoveler-r1.json) (0a3a80c1), [index](../A13-BFCL-production-five-r1/A13-BFCL-index-A12-schema-margay-r1.json) (50cc4a78), [pre-registration](A12-A13-science-and-analysis-preregistration-marten-r1.json), script `/mnt/storage/karan-scratch/shoveler-BFCL-arm-run-bootstrap-r1.py` (e720afad). First-run diagnostic: 16 one-turn no-call stops and 1 validated identity intrusion in the original slot-1 workflow (separate note). This result authorizes no retry, model run or release.

Independent replay: margay's A13-independent-native-and-full-prefix-bootstrap-replay-margay-r1.json reproduces every number above exactly (current body 0f604ad8, with a clarified limits line; the original body e9878140 is preserved beside it), including the shared draws (7d97614b) and the next start c3a6106b, without importing this script.
