# Minimal-compute supplements to the EMA-Gram questions

The fresh late-checkpoint probe finds a local quarter-power advantage at one of the three prespecified step sizes. It does not establish a uniform exponent advantage or change the completed longrun ranking. No cross-seed, longrun replacement, horizon sweep or larger-model run was performed.

The new batches are Wikipedia batch indices8–15: eight global batches of8 sequences at context4096, disjoint from the original probe indices0–7 and infrastructure indices16–17. They are fresh for these checkpoint probes, not a new untouched validation corpus. Ridge0.01 and step sizes1e-5,3e-5,1e-4 were fixed before measurements. Each matrix direction is norm-matched to alpha0; nonmatrix weights and optimizer state remain fixed. Four hosts are replicas, not32 statistical samples.

## Evidence and execution status

- **Late checkpoint33597: completed and validated.** Four successful attempt1 tasks, four finished ONLINE probe runs, four finished host logging checks, generation/hash-bound permanent objects,16 records and9 finite loss rows per host. All four loss outputs agree exactly. Strict TrainerState restore returned counter33597; no training-resume claim.
- **Early3733, middle18665 and middle SVD: recovered supplementary measurements.** The r03 job failed later during the bulk timing stage. Its controller records a container-memory OOM killing task0 and three co-scheduling failures. One complete global-loss artifact supplies21 finite paired rows; there is no four-host artifact agreement or finished ONLINE claim for this failed job. Its native run still reported running at the saved one-time terminal read; no status rewrite or repeated query was made.
- r02 failed before checkpoint restoration because the requested batch range exceeded the17 complete batches in the dataset. The corrected range was selected from metadata without seeing losses. r01 was withdrawn before task attempts. These preparation failures remain in the record.
- r04 preserved Iris defaults: attempt0 hit a worker reconciliation failure and co-scheduling bounce; attempt1 ran with fresh native IDs. The controller counted one preemption/retry event and zero application failures. This does not establish a provider preemption or a training resume.

[Late validation](supplemental-probe-r04-completed.json), [partial recovery](supplemental-probe-r03-partial.json), [raw recovered rows](supplemental-probe-r03-partial-rows.json), [raw late rows](supplemental-probe-r04-rows.json), [paired analysis](supplemental-comparison.json), [analysis code](analyze_supplement.py).

## Does the norm-matched advantage persist on new batches?

At the late checkpoint and step3e-5, quarter power beats both controls on all8 paired batches. The quarter-minus-zero mean is−0.000685364 nats (paired SEM0.000137511); quarter-minus-half is−0.001069099 (SEM0.000096828). Its mean loss change from the unchanged checkpoint is−0.000572711 nats (SEM0.000089838). This is a clear descriptive local pattern on these batches.

The pattern is not uniform: quarter power loses to zero at step1e-5 at every stage and loses to half at the late step1e-4. The late mean advantage over both controls appears at the prespecified middle step, not all steps. Report all settings; do not select a new training LR or exponent from this probe. Reusing batches across stages and steps makes the settings dependent; SEMs are batch-conditional descriptions, not cross-seed uncertainty or a multiplicity-adjusted significance claim.

Values below are quarter minus comparator, mean±paired SEM in **micro-nats (1e-6 nats)**. Negative favors quarter power.

| Counter | Step | Versus alpha0 | Versus alpha0.5 | Evidence |
|---:|---:|---:|---:|---|
| 3,733 | 1e-05 | +15.62 ± 31.70 | -20.15 ± 36.26 | Failed-run recovery |
| 3,733 | 3e-05 | -24.26 ± 21.34 | -22.59 ± 29.08 | Failed-run recovery |
| 3,733 | 0.0001 | +2.09 ± 29.22 | +44.05 ± 29.22 | Failed-run recovery |
| 18,665 | 1e-05 | +96.17 ± 100.01 | -52.45 ± 85.73 | Failed-run recovery |
| 18,665 | 3e-05 | -218.27 ± 121.88 | -118.76 ± 72.89 | Failed-run recovery |
| 18,665 | 0.0001 | -319.90 ± 105.23 | -42.71 ± 87.80 | Failed-run recovery |
| 33,597 | 1e-05 | +75.46 ± 109.67 | -64.37 ± 101.67 | Validated |
| 33,597 | 3e-05 | -685.36 ± 137.51 | -1069.10 ± 96.83 | Validated |
| 33,597 | 0.0001 | -237.32 ± 98.92 | +164.30 ± 86.87 | Validated |

## Does exact SVD consistently improve local loss?

The full-model middle-checkpoint SVD control replaces only the polar approximation while preserving the same Gram powers, per-matrix norm targets and paired batches. Its direction helper passed the earlier14-matrix NumPy oracle; that unchanged numerical check was reused. These loss rows were recovered from the failed r03 job and retain that evidence limitation.

| Step | Exact SVD minus production polar, nats | Paired SEM |
|---:|---:|---:|
| 1e-05 | +0.000224471 | 0.000113488 |
| 3e-05 | -0.000054240 | 0.000101146 |
| 0.0001 | -0.000076801 | 0.000118252 |

Exact SVD is worse at the smallest step and has small negative mean differences at the other two steps. This does not show a uniform local improvement or explain the full-training exponent ranking. It is one checkpoint, one ridge and eight batches, with no completed-run replication across hosts.

## What is the measured cost?

The completed r04 comparison times the already-warm **norm-matched probe direction path** at counter33597, serializing matrix banks. It includes the alpha0 norm reference and rescaling, and excludes gradients, Gram updates, AdamW/nonmatrix updates, model loss and compilation. It uses the probe eigensolver layout, not the manual matrix-parallel training layout. Two repetitions use forward then reverse alpha order; each repetition takes the maximum rank time, followed by the median across repetitions.

| Alpha | Warm probe-direction seconds | Ratio to zero within this probe |
|---:|---:|---:|
| 0.0 | 0.096750 | 1.00 |
| 0.25 | 39.804553 | 411.42 |
| 0.5 | 39.802777 | 411.40 |

Nonzero powers add substantial cost in this probe implementation. The approximately411-fold ratio is **not** a full optimizer-step or training-speed ratio. A comparable full-training trace and alpha-specific peak-memory comparison are still missing. The earlier bulk all-bank benchmark produced no usable timing and is not pooled with these measurements.

Cached checkpoint metadata establishes equal logical storage for alpha0 and alpha0.25:7,350,834,212 array bytes each, including2,722,103,296 bytes of left/right Gram state. This implementation retains and updates Grams even at alpha0. Logical checkpoint bytes are not peak device/host allocation, nor a comparison with memory-minimal generic Muon. [Storage evidence](supplemental-logical-storage.json).

## Supplement resource accounting

| Request | Outcome | Job-span minutes | 16-device slot-hour bound |
|---|---|---:|---:|
| r02 | JOB_STATE_FAILED | 1.893 | 0.505 |
| r03 | JOB_STATE_FAILED | 44.019 | 11.738 |
| r04 | JOB_STATE_SUCCEEDED | 15.273 | 4.073 |

Total: **16.316 device-slot hours as a conservative job-span bound**. This charges16 devices for every second from each request's first start to terminal state, including setup, failed work and r04's rescheduling wait. It is not active-kernel utilization or a cloud bill. The unstarted r01 withdrawal and small local CPU checks are excluded. [Accounting](supplemental-compute-accounting.json).

## Remaining questions under the minimal-compute scope

- [x] New disjoint batch contrasts at the three saved stages:21 recovered early/middle rows plus9 validated late rows, with provenance kept separate.
- [x] Full-model SVD local-loss control at the middle checkpoint:three recovered paired rows; no uniform improvement.
- [x] Same-hardware warm probe-path timings and cached logical-state storage.
- [ ] Hardware-matched full-training cost attribution and comparable alpha-specific peak memory: still required for those performance claims; no new profiling job is queued.
- [ ] Fully successful early/middle/SVD replication: current values remain failed-run recovery. Repeating them solely to upgrade status was excluded to conserve compute.
- Cross-seed is explicitly omitted at the user’s request. No reproducibility claim is made. Horizon/998M experiments and failed-longrun reruns remain outside this supplement.

Historical alpha0.25/0.5 execution failures, missing alpha0.25 training rows and original completion gates are unchanged. The new local probes do not certify those failed longruns or establish their initiating shutdown cause.

Five new analysis fixture/rejection checks verify provenance, paired signs, rank aggregation, missing hosts, batch mismatches, replica disagreement and incomplete timing repetitions. All use scalar fixtures and cached data. No previously completed numerical oracle or old experiment collector was rerun.
