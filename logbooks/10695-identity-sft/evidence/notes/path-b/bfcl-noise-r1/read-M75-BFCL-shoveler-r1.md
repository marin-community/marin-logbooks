# M75 BFCL

M75, the merge W = 0.25·A3 + 0.75·A8a, scores below fresh v0.1 with an interval that excludes zero, though its point estimate stays inside the −5 pp line. Its five full runs score 30, 27, 31, 28 and 31 of 123 (original slot order), for a mean of 23.902% and a sample standard deviation of 1.477 percentage points. The run-bootstrap mean interval is [22.764%, 25.041%].

| Registered comparison | Difference, pp | Unadjusted 95% interval, pp |
|---|---:|---:|
| M75 − fresh v0.1 | −3.740 | [−6.829, −1.301] |
| M75 − A3 | −7.154 | [−10.407, −4.390] |
| M75 − A8a | −2.276 | [−4.553, 0.000] |

The fresh-v0.1 comparison is primary; A3 and A8a are secondary; no cross-merge contrast is added. The primary interval lies wholly below zero. M75 is clearly below A3. Against A8a the upper endpoint is exactly 0.000 (the 9749th sorted draw), so that interval only touches zero; read it as indistinguishable from A8a at this resolution, leaning lower. These are five evaluations of one merged checkpoint; they do not estimate training variability or explain the merge's position.

The five logical slots took twelve physical workflows. Slots 1, 2 and 3 each needed their second replacement; their originals and first replacements were infrastructure-incomplete (EnvironmentStartTimeoutError NULLs) and are retained with NULL outcomes. Slot 5's original (122 + 1 NULL) was replaced once. Slot 2's original and first replacement also contain one native-scored AgentTimeoutError reward 0 each; those runs are not admitted, and the score stays a valid agent outcome in their records. The admitted slot-2 run has no errors. Nothing is zero-filled, no valid score was rerun, and there is no third replacement.

Method: the registered run-cluster bootstrap (merge pre-registration 108c2f65; reduced-n rules f44554ba, not needed at n = 5). The script replays marten's frozen prefix through A11 exactly, advances the canonical stream through the reserved n = 5 blocks of A12, A13, M25, M49 and M51, and starts M75 at 6007be3f, equal to an earlier independent CPU computation and to margay's M51 replay. Each of 10,000 draws is the mean of five choices from the lexically ordered admitted run IDs (run-ID order equals job order), shared across the three contrasts against the frozen comparator draws; sorted endpoints 249/9749. The stream leaves d39bf36d after M75's block. The same script reproduced marten's A12 result exactly; its A13 and M51 results were replayed exactly by margay.

All twelve physical workflows were joined: native record, launch, HF and readiness binding, full JSON and 23-column typed projections, 123 fixed task IDs, binary rewards, coverage and error counts.

Evidence: [final analysis](M75-run-bootstrap-extension-shoveler-r1.json) (dea2b167), [index](../M75-BFCL-production-five-r1/M75-BFCL-index-A12-schema-margay-r1.json) (2719ec5d), [pre-registration](M25-M49-M51-M75-merge-sweep-preregistration-marten-r2.json), script `/mnt/storage/karan-scratch/shoveler-BFCL-arm-run-bootstrap-r1.py` (e720afad). First-run diagnostic: 13 one-turn no-call stops, 1 validated identity intrusion. This result authorizes no retry, model run or release.

Independent replay: margay's M75-independent-native-and-full-prefix-bootstrap-replay-margay-r1.json (0e9bef47) reproduces every number above exactly, including the shared draws (6a6821c6), the start 6007be3f and the next state d39bf36d, without importing this script; it also checked all 1,476 sample rows and retained the 26 infrastructure NULLs of the seven non-admitted workflows.
