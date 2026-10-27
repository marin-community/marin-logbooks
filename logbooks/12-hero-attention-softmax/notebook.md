---
title: "Hero attention softmax gradient diagnostic"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/12"
related_issues:
  - "https://github.com/marin-community/marin-logbooks/issues/10"
  - "https://github.com/marin-community/marin/issues/8506"
research_branch: "codex/12-hero-attention-softmax"
marin_base_revision: "8577d258214dd0dcb992c04458ee4eb10496d4e5"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: active
---

# Hero attention softmax gradient diagnostic

## Question and decision criteria

Compare actual native SM100 fused attention Q/K gradients with an FP64 mathematical
reference on Hero checkpoint inputs. The [output-CE diagnostic](https://github.com/marin-community/marin-logbooks/issues/10)
did not test attention. The [softmax paper](https://arxiv.org/abs/2610.31642) motivates
checking cancellation but supplies no measured result for this fused kernel.

Backward uses `dS_i = p_i (dot(dO,V_i) - dot(dO,O))`. The stored output O is BF16.
Measure actual fused gradient error and distinguish it from a reconstruction that
substitutes stored O into otherwise reference computations. The reconstruction is
an attribution control, not an observation of the fused kernel's internal dS.

Finish when all three checkpoints and ten selected layers have validated finite
results, matching identities, bounded event evidence, and a written conclusion.
No change to live Hero training is part of this run.

## Baseline and launch contract

- Diagnostic implementation commit: `8c31314d20ab501fa82dd985153a246d7726429d`.
- The launch archive commit adds this notebook/configuration and removes inherited CI.
  It will be supplied explicitly through `--source-revision` and recorded in results.
- Recorded production source: `eadf76a1bc24c2bdd9e0e8cbcd75b8da843551b0`.
- Permanent checkpoint steps: 270000, 300000, 330000 from `hero-ctx16k-step257k`.
  Complete URIs and canonical metadata digests are in [request.json](request.json).
  Metadata step, permanence, and digest were reverified locally before submission.
- Inputs, seed, mixture schedule, packing, and document masks match the fixed batch
  from #10: 256 sequences × 16384 positions at production data step330000 and
  production batch schedule2816 (sequence offset929280000).
- Hardware: one spare NVL72 rack, 16 GB200 workers × four GPUs, one process per GPU.
  Each worker requests120 CPUs, 890GB RAM, 1TB scratch. Expert axis64, replica1, context1.
- Unique run: `hero-attention-softmax-20261026-a1`; coordinator uses the same stem
  plus `-coord`; JAX port32961; batch priority; timeout7200 seconds.
- Output root:
  `s3://marin-us-east-02a/marin/users/dlwh/hero-attention-softmax/20261026-a1`.
- W&B disabled; no optimizer state or parameter-gradient tree, optimizer updates,
  checkpoint writes, or exports. New checkpoint retention0 and projected bytes0.
- Full-model input carries move to pinned host between layer passes. Estimated
  carry storage is36GiB/rank, 144GiB/node for48 BF16 carries, excluding restore
  staging and other runtime allocations. This is an estimate, not a measurement.
- Only compact numerical metrics/events are uploaded. Full Q/K/V, token contents,
  and cotangent tensors are not persisted. Temporary runtime state stays node-local;
  Iris owns normal logs/task artifacts. The user is the DRI and this chat owns monitoring.
- Dependencies come from the source tree's GPU extra and lockfile, including
  `flash-attn-4==4.0.0b28`; the run records actual fused GPU execution, not the
  earlier CPU illustration.

## Method

Restore authoritative weights and pending router biases, then cast to the production
FP32-parameter/BF16-compute policy. Keep the recorded capacity-limited EP64 MoE path.
Run all 48 blocks forward with frozen parameters. Reverse layer-by-layer with respect
to hidden states only, starting from the regularized output loss with z-loss1e-4.
This supplies real downstream raw-attention-output cotangents, including XSA, head
gating, output projection, branch convolutions and downstream MLP effects.

The cotangent normalization is the one-rack regularized mean loss. This is not a
bitwise replay of the live eleven-rack training backward. GPU fusion may also differ
between blockwise and scanned execution. Selected capture records its reconstructed
tail's maximum absolute output error and the original output RMS against the
ordinary Block call. Exact BF16 forward parity is required for both the captured
Block tail and head-group replay. A nonzero maximum discrepancy invalidates the
panel and stops the diagnostic; no tolerance is silently relaxed.

Selected zero-based layers are `[0,3,12,15,24,27,36,39,44,47]`: five local and five
global layers. Each rank selects one local sequence and one KV boundary group.
Each of four query strata selects an eligible query with at least two visible keys,
with explicit fallback counts. The same seeded selection is reused across checkpoints.

Replay one GQA4 group through the actual fused SM100 forward/VJP, retaining the
entire original key sequence, D128, scaling, causal/document/window bounds, and
the native kernel configuration. A sparse dO keeps only the four selected query
positions and four grouped query heads. Both compared dQ and accumulated dK are
derivatives of this same sparse objective, not full-step gradients.
Head-group replay records maximum output discrepancy against the original all-head
fused output. In global layers, six logical KV heads expand to twelve boundary
heads; report the expanded boundary's gradients, not the logical K projection's
post-duplication accumulation.

NumPy FP64 reference uses identical quantized Q/K/V and dO and all allowed keys.
It centers value vectors on the winning value before the dO dot product to preserve
small winning-score derivatives. Compare actual fused dQ/dK (and dV/output controls)
using squared-norm aggregated relative-L2 errors, absolute errors, norms, and
exact-zero actual versus nonzero-reference component/vector counts.

Store per-panel masks/counts, margins/entropy and bounded worst absolute-dQ,
winner-cancellation-control, and largest-margin events. Events retain layer, rank,
sequence/query/head identities and numerical correction/gradient evidence.
The stored-output controls retain FP64 reference probabilities with NumPy FP32 dP
and dO·O; they do not reproduce exact tensor-core accumulation, probability
reconstruction, BF16 dS, or gradient casts. Agreement is evidence consistent with
a mechanism, not causal proof. Exclude masked/single-key rows from saturation rates.

## Entries

### 2026-10-26 — Implementation and validation before launch

Nine focused tests passed: four FP64/reference/control tests, three capture parity
tests, and two sparse replay-selection tests. The reference matches independent
FP64 autodiff for GQA and causal/document/window masks and preserves tiny gradients
beyond the FP64 probability cliff. Capture tests compare original attention outputs,
input gradients, and complete Block w_q gradients reconstructed from captured dO.
Replay selection tests verify full-key retention and sparse cotangent support.

Required repository lint passed. Additional local checks verified squared-norm
aggregation instead of averaging relative errors, and bounded JSON process-gather
serialization. GPU-only native execution remains for the rack run.

The run first executes a small native fused two-key saturation control on every
rank. That result is kept separate from checkpoint measurements.

### 2026-10-26 19:43 UTC — Authorized launch submitted

- Published source/config shipped: `e71598bfb54216b9a0c555433d613d2e07bc6cd6`.
- Coordinator: `/dlwh/hero-attention-softmax-20261026-a1-coord`, submitted at
  2026-10-26 19:43:07 UTC (12:43:07 America/Los_Angeles).
- Expected child: `grug-train-hero-attention-softmax-20261026-a1` under that coordinator.
- Request file SHA256: `edfb99bd023beebf34ac2fb8ddd141c83d33d722d116b595ef6f7907cc0de774`.
- The implementation was tested at originating source revision `8c31314d20ab501fa82dd985153a246d7726429d`;
  its parity-gate change is preserved in archive commit `833280836c` with identical source content.
- Submit matched the reproduction command below, replacing `<launch-commit>` with
  the full shipped revision above. W&B is disabled; timeout7200; no new checkpoints.
- Native synthetic control and checkpoint measurements are pending. A running
  coordinator or successful CPU tests do not establish a successful fused GPU comparison.
- This chat owns heartbeat monitoring and will publish validated results before
  updating the issue's conclusion.

### 2026-10-26 19:44 UTC — Native fused GPU control reproduced cancellation

The actual SM100 fused GPU VJP completed on the controlled two-key case before
checkpoint restoration. [synthetic.json](synthetic.json) preserves the full numeric
comparison. The BF16-quantized score margin is9.9878832843 and the FP64 maximum
probability is0.9999540487, below the nominal FP32 probability cliff.

All four query-head gradient vectors were exactly zero in the actual fused VJP;
the FP64 reference dQ norm was8.122739e-6 (4.061370e-6 per head), giving relative
L2 error1.0. Actual dK had relative L2 error0.7071077 and lost one of two nonzero
reference key-gradient vectors. The stored BF16 output was1; its reconstructed
winner-score gradient was0 while the stable reference was4.594915e-5.

This is direct evidence of the mechanism in the actual fused kernel on synthetic
inputs. It does not estimate the prevalence or effect in checkpoint attention.
Coordinator and child were both running at the initial startup check; real-data
checkpoint comparisons remained pending. The heartbeat owns further monitoring.

### 2026-10-26 20:43 UTC — Checkpoint comparison completed

The actual native SM100 fused attention VJP was compared with a stable NumPy FP64
mathematical VJP using identical quantized BF16 Q/K/V and output cotangents (dO).
Q and K denote the query and key arrays at the fused-attention boundary.

The coordinator and all 16 child tasks succeeded with exit 0, no failures, and no
preemptions. The final result was written at 19:50:53 UTC, about eight minutes after
submission. [results.json.gz](results.json.gz) retains the complete compact output, including
per-layer summaries and numerical events. The launch source remains
`e71598bfb54216b9a0c555433d613d2e07bc6cd6`; no replacement run was needed.

All three checkpoints contain ten selected layers and 64 rank panels per layer.
Each rank panel selects one local sequence and one grouped-query-attention (GQA)
KV group, with four seeded stratified multi-key query positions and four query
heads sharing that KV group. It keeps all original keys and masks: 10,240
multi-key query-head observations per checkpoint, 30,720 across checkpoints.
There were no masked rows, single-key rows, or selection fallbacks. Both exact BF16
forward-parity gates passed with maximum absolute difference 0 in every panel:
original Block output versus the capture path rebuilt from the attention output,
and captured full-attention output versus the native one-GQA-group replay output
at selected queries.

The following errors concatenate gradients within each checkpoint and attention
category. Relative L2 is `sqrt(sum(error_squared_norm) / sum(reference_squared_norm))`,
not an average of per-panel relative errors. Absolute norms use the recorded
one-rack mean regularized-loss scaling (CE plus final-logit z-loss weight 1e-4),
with downstream dO propagated from the loss-weighted mean over the fixed
256 x 16384 batch (zero-weight target positions excluded from the denominator).
The replay supplies dO only at the four selected query positions. Both dQ and dK
therefore differentiate this same sparse objective; dK sums its selected
cotangents across the grouped heads.

| Step | Attention | dQ relative L2 | dK relative L2 | dQ error L2 | dK error L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 270000 | all | 0.5483% | 0.3965% | 8.655297e-10 | 1.113748e-09 |
| 270000 | local | 0.5417% | 0.4024% | 6.597493e-10 | 9.096125e-10 |
| 270000 | global | 0.5579% | 0.3854% | 5.602433e-10 | 6.426815e-10 |
| 300000 | all | 0.4775% | 0.3586% | 7.499159e-10 | 8.188381e-10 |
| 300000 | local | 0.4338% | 0.3453% | 5.263233e-10 | 5.872578e-10 |
| 300000 | global | 0.5357% | 0.3744% | 5.341887e-10 | 5.706349e-10 |
| 330000 | all | 3.1118% | 2.4963% | 5.996337e-09 | 8.668485e-09 |
| 330000 | local | 0.5530% | 0.4377% | 8.854715e-10 | 1.131595e-09 |
| 330000 | global | 5.5313% | 3.7069% | 5.930598e-09 | 8.594308e-09 |

| Step | dQ reference L2 | dK reference L2 | dQ max component error | dK max component error |
| --- | ---: | ---: | ---: | ---: |
| 270000 | 1.578537e-07 | 2.809102e-07 | 1.061798e-10 | 2.031454e-10 |
| 300000 | 1.570446e-07 | 2.283462e-07 | 6.043000e-11 | 1.030537e-10 |
| 330000 | 1.926939e-07 | 3.472469e-07 | 1.520135e-09 | 2.097472e-09 |

No actual fused dQ or dK vector was exactly zero when its FP64 reference vector was
nonzero: dQ 0/10,240 and dK 0/4,265,319 per checkpoint. The dK denominator contains
nonzero key vectors from the sparse objective; it is not a token-prevalence
denominator. Raw counts of exact-zero components with nonzero reference were dQ 0/1/0
and dK 200/194/228 at 270k/300k/330k. These component counts include arbitrarily tiny
reference values and do not measure model harm.

The 330k outlier is global layer 15 (zero-based): dQ relative L2 12.0572% and dK
13.1156%. Its largest absolute dQ-error event is rank 43, sequence 175, query 7194,
query head 41, expanded boundary KV head 10. The reference dQ norm is 1.315417e-8,
fused norm 1.813559e-8, and error norm 5.885925e-9 (44.7457% relative). Its maximum
attention probability is 0.9695174, aggregate winner-versus-tail margin 3.4596409,
and entropy 0.2978795 nats. This one row accounts for 96.3513% of all sampled
330k dQ squared error. This event occurs well below the nominal FP32 threshold.

The same event has reference winning score derivative 1.383374e-8. Reconstructing
`p*(dP-dot(dO,O))` with the actual stored BF16 output gives 1.966823e-8 and a
dQ norm of 1.812870e-8; using BF16-rounded reference output gives a dQ norm of
1.820124e-8. At layer 15, those controls have dQ/dK relative errors of
12.0305%/13.1221% and 12.1967%/13.2479%, respectively. This agreement is consistent
with error from the stored-output correction. The controls omit actual fused
internal arithmetic, so they do not establish causal attribution.

| Step | Maximum aggregate margin | Mean entropy (nats) | Rows with margin >=8 | Rows with margin >24 ln(2) | Stored-output zero-winner control |
| --- | ---: | ---: | ---: | ---: | ---: |
| 270000 | 11.578052 | 3.851748 | 37/10240 | 0/10240 | 4/10240 (0.0391%) |
| 300000 | 14.956124 | 3.854881 | 36/10240 | 0/10240 | 4/10240 (0.0391%) |
| 330000 | 15.421632 | 3.889846 | 29/10240 | 0/10240 | 4/10240 (0.0391%) |

Here margin means winner score minus logsumexp of all other visible scores; it
is not the top-two score gap. All 30,720 sampled rows were below `24 ln(2)`
(16.6355), the nominal round-to-nearest threshold where adding the aggregate
exponential tail to 1 in FP32 can round back to 1. The fused kernel has other
rounding operations; this threshold is not a universal kernel-collapse boundary.
The reconstructed zero winning-score derivative occurs 4/10,240 times
per checkpoint (0.0390625%), with a nonzero stable reference. This is a control
count; the fused internal score gradients are not exposed. It does not count
actual fused Q/K vectors lost to collapse. The observations reuse one batch
across layers and checkpoints and are correlated; no independent-token confidence
interval or whole-run bound follows from the zero-vector counts.

One concrete saturation-tail example at 300k (layer 3, rank 54, query 3945, head 38)
has margin 10.60836 and reference winning-score derivative -1.176382e-13. The
stored-output control returns exactly zero; actual fused dQ has norm 1.185765e-13
against reference 1.735565e-13, with error 1.176505e-13 (67.7880%). The numerical
event remains small in absolute norm even though its relative error is large.

| Step | Stored-output control dQ/dK relative L2 | BF16-reference-output control dQ/dK relative L2 |
| --- | ---: | ---: |
| 270000 | 0.5109%/0.3226% | 0.4687%/0.2972% |
| 300000 | 0.4253%/0.2717% | 0.3986%/0.2545% |
| 330000 | 3.0975%/2.4861% | 3.1440%/2.5119% |

The [native synthetic control](synthetic.json) had zero actual fused dQ for all
four query-head vectors against a nonzero FP64 reference, giving relative dQ
error 1.0 and dK error 0.7071. It demonstrates that the kernel can erase gradients.
Checkpoint attention shows measurable Q/K error against the FP64
reference, including a concentrated 330k global-layer outlier. This fixed panel
did not show complete loss of any sampled Q/K vector or any aggregate margin
above the nominal FP32 cliff. Those negative observations do not certify the
kernel or bound checkpoint-wide error. The experiment supports keeping attention
backward precision in the numerical follow-up; it does not measure a training-loss
or model-quality benefit from changing it. No live Hero change or Marin contribution
PR was selected, and no further accelerator work is part of this completed task.

The measured gradients are at the expanded fused-attention boundary, with dK
accumulated only over the four sampled-query cotangents and one GQA group.
Production global layers duplicate logical KV heads. Results do not include the
post-duplication projection gradient, dense full-step dK, or a comparison of full
FP64 model training. The native grouped replay has exact output parity with the
captured full-head attention at selected queries. Full-head versus grouped
backward bit parity was not separately measured, and this does not assert bitwise
identity with a live eleven-rack backward.

Large per-rank panels remain under
`s3://marin-us-east-02a/marin/users/dlwh/hero-attention-softmax/20261026-a1` as
`step-<step>/layer-<layer>-panels.json`. Compact results retain bounded worst events.
[validation.json](validation.json) records finite/count/parity checks, recomputed
aggregates, and 61 artifact SHA256 values. [validate_results.py](validate_results.py)
reproduces the checks when the per-layer JSON files are fetched alongside it.
[checksums.sha256](checksums.sha256) identifies the compact archived evidence.
Raw token IDs, Q/K/V, and cotangents were not published.

### 2026-10-26 — Authorized mitigation comparison

The checkpoint diagnostic above is complete. A follow-up request authorized
implementing and comparing mitigations on the spare rack, then selecting a
bounded paired continuation if accuracy and cost justify one. Live Hero remains
unchanged. This phase reuses the task issue and research branch.

Compare the untouched native SM100 backward with genuine FP32 pre-cast saved
output and a post-BF16 score-gradient projection. The projection follows the
zero-row-sum correction described in [Chen et al.](https://arxiv.org/html/2609.34272v1).
Its FA3 results do not establish FA4 accuracy or performance.

The FP32-output kernel has a different shared-memory layout and pipeline depth.
Its BF16 cast output and LSE differences will be measured. A separate fixed-forward
shadow candidate retains the untouched public BF16 output and baseline LSE while
using a genuine FP32 shadow output only for the backward correction. A control
uses the same correction implementation with the baseline output upcast from
BF16. This distinguishes saved-state precision from reduction-order changes.

The initial projection candidate instruments a private, hash-checked copy of the
pinned SM100 mainloop. It exports actual BF16 probability and score-gradient
operands at the selected query rows. The correction acts on decoded native FP32
Q/K accumulators before their final casts. Both Q and K receive corrections,
including the BF16(lambda*Q) arithmetic of the key pass. This is a native+XLA
hybrid for accuracy investigation. Its sampled export storage and timing do not
establish the cost of a fully fused dense implementation.

Retain the same three checkpoints, fixed batch, and ten layers. Expand replay to
all 48 query heads and 12 expanded KV heads for one selected sequence per rank,
with the same four stratified positions. All keys and original masks remain.
Both gradients use that sparse selected-query objective; dense full-step dK
accuracy remains outside this first comparison. The FP64 mathematical reference
uses identical represented inputs and real downstream cotangents.

Before checkpoint restoration, native smoke cases must compile and pass capture,
forward, and accumulator-decoder parity gates. Decoder acceptance compares native
postprocessing at scale 1 and the real softmax scale; no plain reshape of packed
MMA accumulators is assumed. Instrumentation-versus-baseline backward differences
are measured separately. Both candidate accuracy and absolute errors matter;
no improvement will be inferred from a zero-sum statistic alone.

Benchmark full-head dense-cotangent forward+backward separately at sequence
lengths 256, 2048, 16384, local/global layer masks, with warmup and repeated timings.
Report first-call compile-including latency and per-rank steady-state samples.
The shadow variant includes both forwards. Keep the projection hybrid's sparse
accuracy evidence distinct from dense production-cost claims. Record exact
shapes, device counts, flags, source identities, and saved-state byte estimates.

The first compute allocation is bounded to one spare 64-GPU rack with batch
priority and a two-hour timeout, including native smoke, checkpoint comparison,
and benchmarks. Stop on distributed/OOM/unclear failures. Small obvious script
exceptions may be corrected under distinct source/job/output identities; no
cluster mutation, unrelated cancellation, or production restart is authorized.
Source and resolved request will be published before submission. Only after
validated results will a paired training continuation be configured and logged;
no long-run quality claim can follow from this frozen comparison.

### 2026-10-26 — Mitigation implementation ready for native acceptance

Source implementation `16f78514bf3e39027343721c53bbe79d2181aeab` is preserved
in this archive with identical contents. Eight focused CPU tests pass, covering
saved-output correction precision, exact post-cast projection witnesses, packed
all-head selection, and missing/corrupt operand-export rejection. Required lint
passes. An independent preflight review found no FFI argument or scaling error;
its requested operand coverage/contraction gates were added before launch.

[mitigation-comparison-config.json](mitigation-comparison-config.json) records
public scientific settings. Resolved storage inputs and operational identities
remain in the private run record; the entry point accepts that private resolved
JSON. This redaction avoids publishing additional internal infrastructure
identifiers. Source and public configuration are published before compute.

The rack job will first execute two native smoke cases: a constructed two-key
cancellation case and a packed multi-head case. It must pass exact forward and
accumulator-decoder parity, valid probability coverage, masked-operand zeros,
and actual score-operand contraction checks. The latter use explicit analytic
FP32 rounding bounds rather than an adjustable empirical tolerance. Failure of
an accuracy gate invalidates the panel. The algebraic projected row-mass residual
is telemetry, not independent proof of correct projection.

No GPU acceptance, checkpoint mitigation result, or training improvement has been
established by these CPU checks. The optimized FP32-output candidate and the
fixed-forward FP32-shadow candidate remain distinct. The latter uses original
BF16 public output and baseline LSE; its cost includes both forwards. The native
projection hybrid is a sampled-query accuracy prototype; a scalable dense port
is still required before it could be used in training.

### 2026-10-26 16:51 America/Los_Angeles — Mitigation comparison submitted

The bounded rack comparison was submitted from published source
`8541bd0419acdd89a10c20d29739be4a1411976e`. It first runs native acceptance
smokes, then the three-checkpoint comparison and dense timing panel described
above. Eight focused CPU tests and required lint passed before submission.
GPU acceptance and mitigation results are pending.

The canonical launch command, resolved request checksum, execution identities,
and private output locations are retained in the local operational record.
A heartbeat owns monitoring and result validation. No optimizer updates or live
Hero changes are part of this frozen comparison.

### 2026-10-26 — Native smoke stopped at the decoder verification gate

The experiment compares FP32 saved attention output with a post-cast Q/K
projection. Its decoder reconstructs native packed matrix-multiply accumulators
as logical FP32 gradient arrays before the final BF16 cast.

The initial mitigation attempt failed during its first constructed two-key
native smoke (a small fused-attention GPU test with two allowed keys per selected
query), before checkpoint loading. The native kernels compiled and ran,
but the key-gradient decoder comparison returned maximum absolute difference
3.2281968742609024e-7 against the untouched BF16 postprocessor. The decoder gate has not passed, and whether the checker caused the failure
remains unverified. There are no checkpoint mitigation or performance results.

An independent check reproduced that difference exactly: unrounded FP32 scaled
gradient 0.001838361262343824 versus represented BF16 gradient
0.00183868408203125. GPU fusion may have removed the FP32-to-BF16-to-FP32 roundtrip in
the numerical-difference checker. This is an inference awaiting a GPU test;
no decoder-layout defect has been established.

The prepared correction observes BF16 integer representations and requires zero
mismatches, ignoring signed-zero differences as the original numerical gate did.
No tolerance was increased. A regression verifies the reproduced rounding case
and rejects an adjacent BF16 value. Full gate vectors will be preserved before
validation raises, improving failure evidence. Nine focused CPU tests and
required lint pass; GPU acceptance remains unverified.

A separate single-GPU acceptance entry point is ready for a bounded 15-minute
check of the two-key case and a second packed-sequence case with eight query
heads. The initial attempt stopped before reaching that second case. The
acceptance entry point loads no Hero checkpoints. Replacement
compute is awaiting user judgment under the job-monitoring failure policy; the
original rack job has ended. Further checkpoint comparison and any training
continuation remain pending. Operational details are retained privately.

### 2026-10-26 — Single-GPU acceptance retry authorized and submitted

The follow-up request explicitly authorized the prepared acceptance retry and
routine harness fixes needed to get the suite running. The single-GPU check was
submitted from published source `136b1c39f8fe2efb58a23700bca1c0089fac3921`,
with a 15-minute timeout. It loads no checkpoints and preserves the exact zero-
mismatch and operand-contract criteria. Both native cases must pass before the
bounded rack comparison resumes. GPU acceptance is pending.

Operational identities remain in the private run record. A heartbeat owns this
acceptance phase and the conditional transition to the checkpoint suite. Live
Hero remains unchanged.

### 2026-10-26 — Two-key acceptance passed; second case timed out

The strict BF16 representation checks passed the native two-key case, as did
all operand coverage, mask, finite-value, and contraction gates. This resolves
the previous gate failure for that case without changing the decoder or
increasing tolerances. [Partial summary](mitigation-two-key-partial-summary.json)
retains the measured Q/K errors.

On this constructed case, baseline dQ/dK relative errors were 100%/70.7108%.
FP32 saved output and its fixed-forward shadow gave 0.2032%/0.1604%; native
projection gave 0.2032%/0.2131%. All 16 baseline query vectors were zero against
nonzero references; both mitigation candidates retained all 16. The same
postcast-output control retained the baseline failure. These are synthetic
results and establish neither checkpoint prevalence nor training benefit.

The packed eight-head case did not finish before the 15-minute parent timeout.
The last cache-miss message precedes both compilation and execution, so the
existing log cannot identify the stalled phase. No numerical failure was
recorded for that case. The acceptance harness also omitted Hero runtime
defaults, including its existing command-buffer workaround. The next source
aligns those settings and persists lowering, compilation, dispatch, and
synchronization markers, with periodic Python stack dumps. Whether the runtime
mismatch caused the timeout remains unverified.

The optimized FP32-output forward comparison now observes BF16 bit patterns as
well. The previous float-roundtrip difference could be affected by the same
cast folding, so it is not evidence of represented BF16 output disagreement.
Nine CPU tests and required lint pass. The next acceptance attempt remains a
single-GPU check, now bounded to 30 minutes to distinguish slow compilation from
a stall. Both cases must pass before checkpoint comparison resumes.

### 2026-10-26 — Explicit lowering compatibility correction

The runtime-aligned acceptance attempt stopped at the first lowering marker with
`AttributeError: PjitFunction has no attribute _fun`. This came from Equinox's
explicit lowering path against the GPU environment's JAX version, before kernel
execution. It does not evaluate the runtime-alignment hypothesis or the second
native case.

The smoke helper now uses JAX's public `jit`/`lower`/`compile` API with static
shape arguments. A local compiled-call check and required lint pass. Native
inputs, mitigation arithmetic, and acceptance criteria are unchanged. The next
single-GPU attempt retains phase markers and the same 30-minute bound.

### 2026-10-26 — Worker import boundary correction

The public JAX `.lower()` call also raised `PjitFunction has no attribute _fun`
before kernel execution. This corrects the preceding attribution: the available
trace did not establish Equinox as the cause. The acceptance entrypoint was a
function in `__main__`, which Iris serializes by value together with its
referenced jitted smoke helper. A local CPU reproduction confirmed that a
JAX-jitted function executes after a cloudpickle roundtrip but its explicit
`.lower()` raises the same error. This reproduces with one JAX version; a
version mismatch is not required.

The acceptance and frozen-checkpoint comparison launchers now resolve their
worker function from its importable module before dispatch. This lets the worker
import its own JAX wrappers instead of receiving serialized wrappers. GPU
verification of the change is pending. Phase markers, input cases, mitigation
arithmetic, and acceptance gates are unchanged. The next acceptance attempt
retains the one-GPU, 30-minute bound. Both the two-key 128-token/four-head case
and the packed 256-token/eight-head case must pass before the three-checkpoint
rack comparison.

## Reproduction

The complete result is gzip-compressed to fit the repository artifact-size limit.
Decompress `results.json.gz` to `results.json` before validation; the compressed
artifact preserves the original S3 bytes. `validation.json` hashes the original
remote objects, and `checksums.sha256` hashes the archived compressed copy.

The post-run arithmetic validator needs the per-layer JSON directories fetched
from the recorded output root into this notebook directory. Then run
`uv run --no-sync python logbooks/12-hero-attention-softmax/validate_results.py`.
It checks counts, squared-norm aggregates, selected events, identities, and finite
values; it cannot reconstruct gradients without the original operand arrays.

Pure CPU checks:

```bash
uv run pytest experiments/grug/moe_hero_ep/test_attention_softmax_reference.py \
  experiments/grug/moe_hero_ep/test_attention_softmax_capture.py \
  experiments/grug/moe_hero_ep/test_attention_softmax_replay.py
```

Authorized accelerator submit from this published archive, with the concrete
archive SHA substituted for `<launch-commit>`:

```bash
uv run --no-sync iris --config lib/iris/config/marin.yaml job run --no-wait \
  --enable-extra-resources --target-cluster cw-us-east-08a --priority batch \
  --cpu 2 --memory 8GB --disk 32GB --timeout 7200 \
  --job-name hero-attention-softmax-20261026-a1-coord \
  -e IRIS_PORT_JAX 32961 \
  -- python -m experiments.grug.moe_hero_ep.attention_softmax_diagnostic \
     --request logbooks/12-hero-attention-softmax/request.json \
     --source-revision <launch-commit>
```

Iris forwards normal application credentials; no secrets are in this archive.
Repeating this command requires fresh run/output/job/port identities and separate
compute authorization. Sources and configuration are published before this launch.

## Final decision

The original diagnostic is complete. The two-key native acceptance case passed
strict parity and operand-contract checks, with both mitigation candidates
recovering its lost query gradients. The second packed case timed out without
a numerical verdict. Worker import boundaries now avoid the reproduced serialization failure in
explicit lowering; native verification of that correction is pending. Checkpoint comparisons, performance,
and any training winner remain pending.
