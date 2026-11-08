# Answers to the EMA-Gram experiment questions

This answers the hypothesis and requested measurements in
[Marin issue #10784](https://github.com/marin-community/marin/issues/10784).
[Logbooks issue #20](https://github.com/marin-community/marin-logbooks/issues/20)
holds the execution record. The earlier one-sided optimizer experiment in
[#6588](https://github.com/marin-community/marin/pull/6588) provides context;
its Qwen3/BPB results are not pooled with this two-sided Delphi/nats comparison.
The fetched issues describe the experiment, but do not contain a direct comment
from Ali spelling out a separate question list.

The user subsequently authorized minimal-compute supplements and explicitly
excluded cross-seed work. [The completed supplement report](supplemental-findings.md)
adds disjoint-batch contrasts, a recovered full-model SVD loss control, validated
late-checkpoint measurements and bounded probe timings. Early/middle/SVD values
retain failed-run recovery status. Failed longruns were not rerun; shutdown
packaging remains deferred and original completion gates are unchanged.

The matrix update uses
`D_alpha = A^(-alpha) polar(A^(-alpha) M B^(-alpha)) B^(-alpha)`,
where M is gradient momentum and A/B are the trace-normalized, ridge-regularized
EMAs of the left/right gradient Grams. Alpha=0 reduces this direction to
`polar(M)`; alpha controls both inner and outer preconditioning. This defines
the convention used below, rather than all variants called Muon.

## Does quarter power improve held-out loss after LR tuning?

The recovered quarter-power endpoint is better than half power and worse than
alpha=0 in this seed0 comparison. It does not support the hypothesis that
quarter power beats both controls. The two failed executions supply supplementary
measurements only; there is no fully validated quarter-versus-half completion
comparison or cross-seed conclusion.

| Alpha | Pilot-selected matrix LR | Ridge | Final C4 loss, nats | Evidence status |
|---:|---:|---:|---:|---|
| 0 | 0.003535534 | 0.0001 | 3.792297 | Validated completion; W&B |
| 1/8 | 0.0025 | 0.01 | 3.784025 | Validated completion; W&B |
| 1/4 | 0.002 | 0.01 | 3.844011 | Failed shutdown; recovered local final evaluation |
| 1/2 | 0.001 | 0.01 | 3.958380 | Failed shutdown; recovered local final evaluation |

Quarter power is 0.051714 nats above alpha=0 and 0.114369 below half power.
Alpha=1/8 is the best observed endpoint among these four trajectories and is
0.008272 nats below alpha=0. This is a comparison of independently tuned
configurations, not a fixed-LR causal estimate of the exponent alone. The
37 completed pilots, including seven boundary expansions, jointly selected LR
and ridge by final pilot C4 loss before confirmation. Each pilot used 7,630
updates (250,019,840 tokens). Confirmation included alpha=0, 1/4, 1/2 and the
best other pilot exponent, 1/8. Alpha=1/3 was not that best other exponent; its absence is not an unfinished cell
of the amended confirmation plan.

The user-amended scope is one seed, 447M parameters, 37,335 updates and
1,223,393,280 training tokens on the fixed HQ/code/math subset. The web-data
replacement means this is not a byte-identical reproduction of the original
Delphi data recipe. Model/data configuration agreement does not prove identical
initial array bytes across different hardware layouts. Full curves and source
bindings remain in the [published seed0 findings](single-seed-findings.md) and
[scalar comparison](single-seed-comparison.json); they were not regenerated.

The quarter-power execution failed during distributed shutdown: cached logs
show a 3-of-4 JAX shutdown barrier and process0 blocked in tracker finalization;
W&B background completion was abandoned after 120 seconds. One task records
exit 139/SIGSEGV while stderr also mentions signal 6, and siblings report
co-scheduling failure. The initiating cause and signal discrepancy remain
unresolved. These observations do not establish optimizer divergence or a
numerical failure caused by alpha=1/4.

## Does any advantage survive per-matrix norm matching?

There is no consistent quarter-power advantage over both controls across the
tested local perturbations. The three prescribed probes are complete. Each uses
the same eight held-out Wikipedia batches at an alpha=1/4 reference checkpoint,
with every matrix direction rescaled to that matrix's alpha=0 Frobenius norm.
Nonmatrix weights stay fixed, weight decay is omitted, and optimizer state does
not advance. The four hosts repeat the calculation and are not extra samples.

A new scalar analysis of those cached batch losses compares alpha=1/4 directly
with alpha=0 and 1/2 at each common step size. Nonzero comparisons also use the
same ridge. Negative paired mean loss differences favor quarter power.

| Reference update | Quarter power lower than alpha=0 | Quarter power lower than half power |
|---:|---:|---:|
| 3,733 | 4 of 6 settings | 2 of 6 settings |
| 18,665 | 4 of 6 settings | 2 of 6 settings |
| 33,597 | 5 of 6 settings | 2 of 6 settings |

Each six-setting group is two ridges (0.0001, 0.01) by three step sizes
(0.00001, 0.00003, 0.0001). These counts are descriptive, not independent wins
or significance tests. All 36 contrasts, eight batch differences per contrast,
paired standard deviations and standard errors are retained in
[the contrast data](ali-norm-matched-contrasts.json). The
[analysis script](paired_probe_contrasts.py) verifies batch alignment and checks
each mean against the difference of the cached means. Across the six settings,
the paired mean ranges (quarter minus comparator, nats) are:

| Reference update | Versus alpha=0 | Versus half power |
|---:|---:|---:|
| 3,733 | -0.00003469 to +0.00001705 | -0.00001150 to +0.00004333 |
| 18,665 | -0.00048798 to +0.00028893 | -0.00013706 to +0.00029406 |
| 33,597 | -0.00044149 to +0.00003025 | -0.00013366 to +0.00022858 |

The 36 per-contrast paired standard errors range from 0.00002472 to 0.00013039
nats; the data file pairs each mean with its own uncertainty. These are not
multiple-comparison-adjusted intervals. The original 27 settings per checkpoint
are nine exponent/ridge directions times three step sizes; the direct comparisons
above select quarter power against the two controls. Standard errors describe
batch uncertainty conditional on this reference trajectory, not seed uncertainty.

At the late checkpoint, all 27 original probe settings increased mean loss
relative to leaving the checkpoint unchanged. A lower loss than another
perturbation therefore need not mean an improvement over the checkpoint. These
local tests do not establish the outcome of norm-matched training, explain the
full-run ranking causally, or select a universal best exponent. Reusing the same
eight batches across stages also limits generalization to fresh held-out data.

## Could the production polar approximation change the answer?

It remains possible; the current evidence does not isolate that effect on
training quality. Across the three checkpoints, the production-versus-SVD
direction comparisons cover one attention and one MLP matrix at nine
exponent/ridge settings: 54 records. Cosines span 0.863830–0.999752 and relative
errors span 0.022707–0.521354. These are measured direction discrepancies, not
loss differences between production-polar and exact-SVD training. The Grams use
spectral powers while the production polar uses five quintic Newton–Schulz steps;
the two approximation questions must not be conflated. The diagnostic changes
only `polar_method` in `gram_direction` at the same stored Gram/momentum state,
alpha and ridge; it retains the same Gram-power computation and update scaling.
It does not evaluate paired production-versus-SVD losses.

## Is the method more efficient, and where does its cost come from?

There is a small observed token advantage for alpha=1/8, with higher measured
deployment time and device use. Quarter power reaches the fixed C4 <= 4.0 target
at the same observed update count as alpha=0; half power reaches it later.

| Alpha | First observed target-crossing updates | Native hours to target | Device-hours to target |
|---:|---:|---:|---:|
| 0 | 34,000 | 4.770 | 19.081 |
| 1/8 | 33,000 | 35.630 | 570.088 |
| 1/4, failed | 34,000 | 39.567 | 633.076 |
| 1/2, failed | 36,000 | 42.988 | 687.808 |

Crossings use recorded evaluations without interpolation. Native time includes
compilation/evaluation and excludes queue time. Alpha=0 used four v4 devices;
nonzero configurations used sixteen with a different model mesh. Thus the
1/8-to-zero ratios of 7.47 in time and 29.88 in device-hours describe these
deployments, not hardware-matched optimizer overhead. The two failed runs have
no observed native final-evaluation runtime. Source-matched attribution among
Gram updates, decompositions and exposed communication, and a comparable peak
host/device-memory measurement, remain unestablished. Old synthetic or
earlier-data profiles cannot fill those gaps.

## New minimal-compute evidence

At the late checkpoint33597 and the prespecified step3e-5, norm-matched quarter
power beats zero and half on all eight new batches. Mean differences are
−0.000685364 and−0.001069099 nats respectively; other steps do not show a uniform
advantage. The middle-checkpoint SVD control has mixed effects by step size and
is recovered evidence from a later OOM-failed probe. These results support a
conditional local direction effect, not a better full-training exponent.

Warm norm-matched probe directions take about0.097s for alpha0 and39.80s for
alpha0.25/0.5 on the same topology. This includes norm-reference/rescaling and
uses the probe layout; it is not a training-overhead ratio. Alpha0 and0.25 have
equal logical stored state in this implementation. Full-training cost attribution
and comparable peak memory remain unresolved. See [all settings, provenance,
cost accounting and remaining limits](supplemental-findings.md).

## Which experiments would resolve the remaining questions?

The table records the remaining claim boundaries after the minimal supplements.
No further job is queued; cross-seed remains explicitly excluded.

| Missing claim | Smallest useful additional experiment or analysis | Priority and scope |
|---|---|---|
| A quarter-power direction advantage generalizes beyond the eight probe batches | Repeat the fixed checkpoint/ridge/step contrasts on new disjoint held-out batches; report all paired differences | Supplemented at all three stages; early/middle are recovered evidence, late is validated |
| Polar approximation explains the quality ranking | At the same retained states, compare norm-matched production-polar and exact-SVD perturbation losses on identical fresh batches | Middle checkpoint supplemented using recovered SVD losses; effects are mixed, not a full-training explanation |
| A stable advantage across training seeds | New paired seeds with pilot-locked settings and unchanged controls | Explicitly omitted by the user; no cross-seed reproducibility claim |
| Optimizer cost or memory overhead on equal hardware | First inspect any already-cached matching profiles; if unavailable, use a short same-topology, same-source profiling/memory experiment | Probe-path timing and logical storage supplemented; full-training attribution and peak memory still missing |
| Better Gram horizon or scaling to 998M | Horizon 100/1,000/10,000 comparisons or the larger-model recipe | Deferred conditional follow-ups; current evidence does not satisfy the original reproducibility gate |

Alpha=1/3 full training would be a scope expansion only if a broader full-budget
exponent optimum is requested. No seed0 replacement runs are planned.
The recovered final evaluations use the configured C4 evaluator at the recorded
last update; checkpoint counters are 37,335 and configuration/source joins were
checked. This does not establish all-array restore integrity or a clean final
tracking lifecycle. Alpha=1/4 still lacks 112 training-log steps for all five
required fields: train/loss, throughput/duration, throughput/mfu,
optim/learning_rate and optim/adam_lr. That missingness cannot be repaired by
interpolation or by changing a completion label. Neither failed
job is declared successful. Their recovered endpoints and earlier target
crossings retain the supplementary status already adjudicated in the findings.
