# M51 BFCL

M51, the merge W = 0.49·A3 + 0.51·A8a, finishes at four of five planned runs. Its four full runs score 30, 25, 35 and 27 of 123 (slots 2 to 5), for a mean of 23.780% and a sample standard deviation of 3.536 percentage points. The run-bootstrap mean interval is [21.138%, 26.829%].

| Registered comparison | Difference, pp | Unadjusted 95% interval, pp |
|---|---:|---:|
| M51 − fresh v0.1 | −3.862 | [−7.927, +0.081] |
| M51 − A3 | −7.276 | [−11.341, −3.130] |
| M51 − A8a | −2.398 | [−5.732, +1.301] |

The fresh-v0.1 comparison is primary; A3 and A8a are secondary; no cross-merge contrast is added. Against fresh v0.1 the point estimate is inside the −5 pp line and the interval just reaches zero. M51 is clearly below A3 and indistinguishable from A8a. With four run clusters the intervals are coarse. These are evaluations of one merged checkpoint, not of training variability, and they do not establish why the merge sits where it does.

Slot 1 is exhausted: its original (122 scored + 1 infrastructure NULL), first replacement (122 + 1) and second replacement (120 + 3, EnvironmentStartTimeoutError) were all infrastructure-incomplete, and there is no third replacement. Under the reduced-n addendum (f44554ba) the planned five are kept, n = 4, the slot contributes no score, nothing is zero-filled, and the bootstrap resamples the four actual runs, not five.

Method: the registered run-cluster bootstrap (merge pre-registration 108c2f65; reduced-n f44554ba). The script replays marten's frozen prefix through A11 exactly and advances the canonical stream through the reserved n = 5 blocks of A12, A13, M25 and M49 (whatever their own outcomes), so M51 starts at canonical state 181512e0. Each of 10,000 draws is the mean of four choices from the lexically ordered admitted run IDs (run-ID order equals job order), shared across the three contrasts against the frozen comparator draws; sorted endpoints 249/9749. The canonical stream then advances through M51's reserved n = 5 block and leaves 6007be3f for M75. Both block starts equal an earlier independent CPU computation of the chain. The same script reproduced marten's A12 result exactly and its A13 result was replayed exactly by margay.

All seven physical workflows were joined: native record, launch, HF and readiness binding, full JSON and 23-column typed projections, 123 fixed task IDs, binary rewards, coverage and error counts.

Evidence: [final analysis](M51-run-bootstrap-extension-shoveler-r1.json) (85709a7f), [index](../M51-BFCL-production-five-r1/M51-BFCL-index-A12-schema-margay-r1.json) (bf2c0e89), [pre-registration](M25-M49-M51-M75-merge-sweep-preregistration-marten-r2.json), script `/mnt/storage/karan-scratch/shoveler-BFCL-arm-run-bootstrap-r1.py` (e720afad). First-run diagnostic: 12 one-turn no-call stops, 0 validated identity intrusions. This result authorizes no retry, model run or release.

Independent replay: margay's M51-independent-native-and-full-prefix-bootstrap-replay-margay-r1.json (9eb2ef39) reproduces every number above exactly, including the shared draws (f7227a5a), the start 181512e0 reached through the reserved M25 (c3a6106b) and M49 (794bbc41) blocks without their outcomes, and the next start 6007be3f, without importing this script.
