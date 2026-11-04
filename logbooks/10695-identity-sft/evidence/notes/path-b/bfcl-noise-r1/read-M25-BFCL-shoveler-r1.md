# M25 BFCL

M25, the merge W = 0.75·A3 + 0.25·A8a, shows no detectable BFCL loss against fresh v0.1 and is level with A3. Its five full runs score 35, 38, 43, 37 and 34 of 123 (original slot order), for a mean of 30.407% and a sample standard deviation of 2.851 percentage points. The run-bootstrap mean interval is [28.455%, 32.683%].

| Registered comparison | Difference, pp | Unadjusted 95% interval, pp |
|---|---:|---:|
| M25 − fresh v0.1 | +2.764 | [−0.813, +6.016] |
| M25 − A3 | −0.650 | [−4.228, +2.927] |
| M25 − A8a | +4.228 | [+1.301, +7.317] |

The fresh-v0.1 comparison is primary; A3 and A8a are secondary; no cross-merge contrast is added. The primary and A3 intervals include zero; M25 scores above A8a. BFCL is one screen: M25 separately triggers the registered GSM8K stop rule in its paired maths (marten's maths note), so this BFCL result is not a clean pass on its own. These are five evaluations of one merged checkpoint; they do not estimate training variability or explain the merge's position.

The five logical slots took eight physical workflows. Slot 4 needed its second replacement (original 120 + 3 NULL, first replacement 121 + 2 NULL, both EnvironmentStartTimeoutError), and slot 5 its first (original 121 + 2 NULL). All incomplete workflows are retained with NULL outcomes; nothing is zero-filled, no valid score was rerun, and there is no third replacement.

Method: the registered run-cluster bootstrap (merge pre-registration 108c2f65; reduced-n rules f44554ba, not needed at n = 5). The script replays marten's frozen prefix through A11 exactly, advances through the reserved n = 5 blocks of A12 and A13, and starts M25 at c3a6106b, the state A13's analysis left and margay's replays confirmed. Each of 10,000 draws is the mean of five choices from the lexically ordered admitted run IDs (run-ID order equals job order), shared across the three contrasts against the frozen comparator draws; sorted endpoints 249/9749. The stream leaves 794bbc41 for M49, the start margay's M51 and M75 replays already asserted.

All eight physical workflows were joined: native record, launch, HF and readiness binding, full JSON and 23-column typed projections, 123 fixed task IDs, binary rewards, coverage and error counts.

Evidence: [final analysis](M25-run-bootstrap-extension-shoveler-r1.json) (b484c297), [index](../M25-BFCL-production-five-r1/M25-BFCL-index-A12-schema-margay-r1.json) (75898578), [pre-registration](M25-M49-M51-M75-merge-sweep-preregistration-marten-r2.json), script `/mnt/storage/karan-scratch/shoveler-BFCL-arm-run-bootstrap-r1.py` (e720afad). First-run diagnostic: 11 one-turn no-call stops, 0 validated identity intrusions. This result authorizes no retry, model run or release.

Independent replay: margay's M25-independent-native-and-full-prefix-bootstrap-replay-margay-r1.json (ab4350d6) reproduces every number above exactly, including the shared draws (7384a38b), the start c3a6106b and the next state 794bbc41, without importing this script. The realized n = 5 draw ends at the same state that margay's M51 replay reached by advancing the reserved M25 block without outcomes.
