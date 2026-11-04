# M49 BFCL

M49, the merge W = 0.51·A3 + 0.49·A8a, finishes at four of five planned runs and shows no detectable BFCL loss against fresh v0.1. Its four full runs score 39, 38, 32 and 36 of 123 (slots 2 to 5), for a mean of 29.472% and a sample standard deviation of 2.517 percentage points. The run-bootstrap mean interval is [27.236%, 31.301%].

| Registered comparison | Difference, pp | Unadjusted 95% interval, pp |
|---|---:|---:|
| M49 − fresh v0.1 | +1.829 | [−1.748, +4.878] |
| M49 − A3 | −1.585 | [−5.285, +1.789] |
| M49 − A8a | +3.293 | [+0.407, +6.179] |

The fresh-v0.1 comparison is primary; A3 and A8a are secondary; no cross-merge contrast is added. The primary and A3 intervals include zero; M49 scores above A8a. With four run clusters the intervals are coarse. BFCL is one screen: M49 separately triggers the registered GSM8K stop rule in its paired maths (marten's maths note), so this BFCL result is not a clean pass on its own. These are evaluations of one merged checkpoint; they do not estimate training variability or explain the merge's position.

Slot 1 is exhausted: its original (121 scored + 2 infrastructure NULLs), first replacement (SDK worker_failed after its pod was deleted while active; 122 + 1) and second replacement (121 + 2, EnvironmentStartTimeoutError) were all infrastructure-incomplete, and there is no third replacement. Slot 2 needed its second replacement and slot 5 its first. Under the reduced-n addendum (f44554ba) the planned five are kept, n = 4, slot 1 contributes no score, nothing is zero-filled, and the bootstrap resamples the four actual runs.

Method: the registered run-cluster bootstrap (merge pre-registration 108c2f65; reduced-n f44554ba). The script replays marten's frozen prefix through A11 exactly, advances through the reserved n = 5 blocks of A12, A13 and M25, and starts M49 at 794bbc41, the state M25's realized analysis left. Each of 10,000 draws is the mean of four choices from the lexically ordered admitted run IDs (run-ID order equals job order), shared across the three contrasts against the frozen comparator draws; sorted endpoints 249/9749. The canonical stream then advances through M49's reserved n = 5 block and reaches 181512e0, the M51 start already used and replayed, so the chain A12 → A13 → M25 → M49 → M51 → M75 is closed and consistent.

All ten physical workflows were joined: native record, launch, HF and readiness binding, full JSON and 23-column typed projections, 123 fixed task IDs, binary rewards, coverage and error counts.

Evidence: [final analysis](M49-run-bootstrap-extension-shoveler-r1.json) (6cdfba44), [index](../M49-BFCL-production-five-r1/M49-BFCL-index-A12-schema-margay-r1.json) (601ee8a2), [pre-registration](M25-M49-M51-M75-merge-sweep-preregistration-marten-r2.json), script `/mnt/storage/karan-scratch/shoveler-BFCL-arm-run-bootstrap-r1.py` (e720afad). First-run diagnostic: 14 one-turn no-call stops, 0 validated identity intrusions. This result authorizes no retry, model run or release.

Independent replay: margay's M49-independent-native-and-full-prefix-bootstrap-replay-margay-r1.json (bb4655e7) reproduces every number above exactly, including the shared draws (c604bf13), the start 794bbc41 and the next state 181512e0, without importing this script.
