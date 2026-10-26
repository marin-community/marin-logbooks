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
Run all48 blocks forward with frozen parameters. Reverse layer-by-layer with respect
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

## Reproduction

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

Pending the native control and checkpoint results. No maintained Marin contribution
or production-kernel change has been selected. Interpret total fused/reference
disagreement separately from cancellation-control events and retain remaining
uncertainty about full training trajectories and model quality.

