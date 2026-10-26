---
title: "Hero checkpoint softmax-collapse diagnostic"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/10"
related_issues:
  - "https://github.com/marin-community/marin/issues/8506"
research_branch: "codex/10-hero-softmax-collapse"
marin_base_revision: "8577d258214dd0dcb992c04458ee4eb10496d4e5"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: completed
---

# Hero checkpoint softmax-collapse diagnostic

## Question and decision criteria

An external training report suggested softmax collapse affected more than 0.1% of tokens.
Check whether Hero has a comparable prevalence before considering a production loss change.
The external report's data and kernel were not reproduced; 0.1% is a comparison threshold,
not a prevalence result from the paper.

[Barron, Gnanapragasam, and Tschannen, A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642)
describes target gradients rounding to zero while non-target gradients remain positive.
For hard labels, define the confidence margin as
`m = z_target - logsumexp(z_non_target)`. The nominal FP32 cliff is
`24 ln(2) = 16.6355323`. A gap to the second-largest logit does not account for the vocabulary tail.

The decision criterion was measured prevalence through Hero's production CE path,
supported by a stable full-vocabulary margin calculation and a positive-tail check.
The stopping condition was a completed three-checkpoint panel with finite, validated
metrics. A matched optimizer continuation or a quality comparison would be separate work.

## Baseline and provenance

The original experiment was run on 2026-10-26 from local committed source. This archive
was requested afterward. Source was committed before submission but was first pushed
with this retrospective archive; a pre-launch archive publication did not occur.

| Identity | Value |
| --- | --- |
| Marin base | `8577d258214dd0dcb992c04458ee4eb10496d4e5` |
| Initial diagnostic source | `8ace3514b77ef759f0008536515f71a7cdbff1b4` |
| Successful launch source | `634bf63d5a3cb8f812cb5fbf836e14302361dbd4` |
| Recorded production launch source | `eadf76a1bc24c2bdd9e0e8cbcd75b8da843551b0` |
| Production run | `marin-community/marin_moe/hero-ctx16k-step257k` |
| Successful Iris coordinator | `/dlwh/hero-softmax-collapse-20261026-a2-coord` |
| GPU child | `/dlwh/hero-softmax-collapse-20261026-a2-coord/grug-train-hero-softmax-collapse-20261026-a2` |
| Submit time | 2026-10-26 18:13:29 UTC / 11:13:29 America/Los_Angeles |
| Child submit time | 2026-10-26 18:14:08 UTC |
| Completion verification | 2026-10-26 18:23 UTC: coordinator and child both succeeded; all result rows present |
| Tracker | W&B disabled for the diagnostic |
| Limits | Batch priority; coordinator timeout 7200 seconds; no task-failure retries |
| Owner | This investigation; monitoring heartbeat removed after verified completion |

The launch tree was clean. The published branch retains both source commits in its
ancestry. The source tree's `uv.lock` and GPU extra supplied dependencies; a separate
runtime package-version or container-image inventory was not captured.

The recorded production source is an ancestor of the diagnostic base. The CE streaming
implementation and reference forward had no diff between that recorded source and
the diagnostic base. The diagnostic uses the recorded model configuration; this is
checkpoint inference, not a bitwise replay of the complete production training step.

## Method

### Checkpoints and model

All checkpoints are permanent checkpoints from the same 16K-context run:

| Step | Canonical metadata digest |
| --- | --- |
| 270000 | `e5405b9822834f7676bf7d087a55b6163f970d247e60b56bcb7c9a50d7851369` |
| 300000 | `0571a717c4ee54c1d5753540a772a9d0c334aeaa9c8eb87568e5c0db701206f1` |
| 330000 | `cd18cfe449805b070ce7d5e0e77d8f914e02db9deeed314edcc8d87d736852e9` |

The common checkpoint root is
`s3://marin-us-east-02a/marin/grug/hero-ctx16k-step257k/2026.08.19.2/checkpoints/`;
append `step-N`. [request.json](request.json) records the full URIs and digests.
Restoration verifies permanent metadata and applies pending query-bias/router updates.
It loads authoritative weights only, with no optimizer state, training updates,
checkpoint writes, or canonical model export.

The recorded model has 48 layers, hidden width 6144, 384 routed experts with top-8
routing, two shared experts, 128256 vocabulary entries, and 16384 context length.
It uses `ragged_all_to_all`, training capacity factor 1.15, three expert waves,
and the SM100 attention implementation. Weights restore in FP32 and cast to the
recorded BF16 compute policy. The model uses the production capacity-limited EP64
route, rather than switching to dropless inference.

### Hardware and data

One spare GB200 NVL72 rack supplied 16 workers × four GPUs, with one process per GPU.
Each worker requested 120 CPUs, 890 GB RAM, and 1 TB scratch. The mesh had expert
axis 64 and replica axis 1. The production run uses 11 replica racks; this diagnostic
keeps one rack's production-shaped batch and does not update the live run.

The request preserves the production model/data configuration retrieved for
`hero-ctx16k-step257k`. Data comes from the Harrier store `store_4d2e363d`, its
recorded scheduled mixture, original seed 0, packing, and document attention masks.
It is a training-mixture sample, not held-out validation. The same batch is reused
across checkpoints, so later-checkpoint CE changes are descriptive paired results,
not independent estimates across batches.

The mixture dataset uses the production batch schedule of 2816 sequences per update.
Its fixed sequence offset is `330000 * 2816 = 929280000`. A diagnostic loader batch
of 256 starts at loader step `3630000`. It supplies 256 × 16384 = 4,194,304 positions,
with each sequence's last next-token position masked: 4,194,048 valid tokens.

For margin analysis, each 16-position stratum contributes a seeded random position.
The seed is 1729 and the same position selection is reused across rows and checkpoints.
There are 262,144 sampled positions per checkpoint, all valid in this realized sample.
This sampling has within-sequence and shared-position correlation.

### CE and gradient measurements

The all-token production calculation calls the actual `xla_fast_bwd` streaming
forward, with FP32 logits/log-sum-exp, vocabulary tile 4096, batch tile 65536,
and BF16 hidden/output-head operands. Its forward returns `lse - target_logit`.

Sampled full-vocabulary logits are computed in bounded blocks of 128 positions.
The stable CE is `softplus(-m)` and its target logit gradient is `-sigmoid(-m)`.
Dense FP32 comparison metrics are labeled separately because GEMM shapes and
reduction order may round differently from the production path.

For sampled production CE, the target gradient is reconstructed as
`exp(-production_ce) - 1`. The non-target mass sums
`exp(diagnostic_logits - production_lse)` with the target excluded.
A sampled collapse requires zero production CE, zero reconstructed target gradient,
and positive non-target mass. This is a logit-level reconstruction using dense
diagnostic projections; it does not execute a full parameter-gradient/optimizer
comparison. It precedes weighting and BF16 backward casts.

Hero's final-logit z-loss coefficient is `1e-4`. It is excluded from these pure-CE
diagnostics. The full regularized loss need not be zero on an affected token, and its
logit gradients need not sum to zero. No claim about the full regularized gradient
or optimizer trajectory follows from the pure-CE count.

## Entries

### 2026-10-26 — Synthetic susceptibility

Source inspection found `exp(logits - lse)` followed by target subtraction in the
streaming backward. A pure-CE synthetic check exercised that implementation with a
vocabulary tile smaller than the vocabulary; allowing a full-vocabulary tile would
dispatch the reference implementation instead.

| Logits, target 0 | Stable margin | Stable CE | Streaming CE | Target gradient | Each non-target gradient |
| --- | --- | --- | --- | --- | --- |
| [9, -9, -9] | 17.3068523 | 3.0459972e-8 | 0 | 0 | 1.5250407e-8 |
| [20, 6, 6] | 13.3068523 | 1.6630569e-6 | 0 | 0 | 8.3073974e-7 |

The second case demonstrates an additional cancellation point: an unshifted FP32
log-sum-exp can round back to the target logit below the nominal 16.64 cliff.
This motivated measuring the actual streaming path alongside margins.

[reproduce_synthetic.py](reproduce_synthetic.py) preserves this CPU/GPU probe.
It was rerun during archival and reproduced the values above.

### 2026-10-26 — Initial dispatch and correction

Attempt `hero-softmax-collapse-20261026-a1-coord`, source `8ace3514b77e`, failed before
GPU allocation. Its child redundantly pinned `cw-us-east-08a` as a federation peer
although the coordinator already ran there. The replacement source `634bf63d5a3c`
removed that child target and inherited the coordinator's cluster. This failure
yielded no numerical evidence. Attempt a2 is the sole checkpoint measurement used.

### 2026-10-26 — Completed checkpoint panel

Both a2 jobs succeeded. Result validation checked the three expected checkpoint
identities/digests, finite metrics, and equality of each margin histogram's count
to its sampled-token denominator.

| Step | Production zero CE / valid | Percent | Sampled zero CE with positive tail / sampled | Sampled margin above cliff |
| --- | --- | --- | --- | --- |
| 270000 | 5 / 4,194,048 | 0.0001192166% | 0 / 262,144 | 0 |
| 300000 | 2 / 4,194,048 | 0.0000476866% | 0 / 262,144 | 0 |
| 330000 | 7 / 4,194,048 | 0.0001669032% | 1 / 262,144 | 0 |

The full-batch zero-CE rates are 0.477–1.669 per million tokens, approximately
600–2100 times below 0.1% (1000 per million). The step-330k positive-tail sample
has its own denominator: 1 / 262144 = 0.00038147%. Do not interchange these rates.

| Step | Full-batch mean CE | Sample mean stable CE | Mean absolute dense CE error | Mean reconstructed production CE gradient imbalance |
| --- | --- | --- | --- | --- |
| 270000 | 1.12181167 | 1.12802804 | 1.10356e-7 | 5.08144e-8 |
| 300000 | 1.10425894 | 1.11028802 | 1.09339e-7 | 5.18338e-8 |
| 330000 | 1.07776962 | 1.08283377 | 1.11441e-7 | 5.13080e-8 |

These mean gradient imbalances include ordinary FP32 rounding on all sampled
positions and do not isolate the rare collapsed position. FP32 aggregate CE sums
can coincide despite the nonzero absolute per-token errors.

Margin histogram counts, with half-open intervals:

| Step | <0 | [0,4) | [4,8) | [8,12) | [12,14) | [14,16) | ≥16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 270000 | 92224 | 64838 | 48474 | 56597 | 11 | 0 | 0 |
| 300000 | 90971 | 64685 | 48804 | 57681 | 3 | 0 | 0 |
| 330000 | 89271 | 64415 | 50085 | 58371 | 1 | 1 | 0 |

No sampled margin exceeded the nominal cliff. All metrics were finite. There were
no invalid sample weights/targets, no nonfinite sample logits, and no nonfinite
production CE counts. [results.json](results.json) contains all additive statistics
and [summary.csv](summary.csv) gives recomputed checkpoint rows.
[analyze.py](analyze.py) validates and regenerates that CSV.

## Reproduction and artifacts

Compact immutable evidence:

- Resolved input: [request.json](request.json).
- Result manifest with request and metrics: [results.json](results.json).
- SHA256 identities: [checksums.sha256](checksums.sha256).
- Runnable implementation: [softmax_collapse.py](../../experiments/grug/moe_hero_ep/softmax_collapse.py).
- Stable diagnostics: [softmax_collapse_metrics.py](../../experiments/grug/moe_hero_ep/softmax_collapse_metrics.py).
- Regression checks: [test_softmax_collapse_metrics.py](../../experiments/grug/moe_hero_ep/test_softmax_collapse_metrics.py).

The original result root is
`s3://marin-us-east-02a/marin/users/dlwh/hero-softmax-collapse/20261026-a2`,
containing `results.json` and one `step-N.json` per checkpoint. Checkpoints and
tokenized datasets remain in their existing stores and require authorized Marin
storage access. No credentials, access-bearing links, raw text, or token samples
are in this archive.

Recompute the report without accelerator or storage access:

```bash
uv run python logbooks/10-hero-softmax-collapse/analyze.py
uv run python logbooks/10-hero-softmax-collapse/reproduce_synthetic.py
```

The original successful submit command, with credentials supplied through normal
Iris secret delivery:

```bash
uv run --no-sync iris --config lib/iris/config/marin.yaml job run --no-wait \
  --enable-extra-resources --target-cluster cw-us-east-08a --priority batch \
  --cpu 2 --memory 8GB --disk 32GB --timeout 7200 \
  --job-name hero-softmax-collapse-20261026-a2-coord \
  -e IRIS_PORT_JAX 32947 \
  --include scratch/hero-collapse-20261026/request.json \
  -- python -m experiments.grug.moe_hero_ep.softmax_collapse \
     --request scratch/hero-collapse-20261026/request.json
```

To repeat on accelerators, first copy the archived request to that scratch path and
change its `run_id`, `output_root`, and `source_revision` to fresh, explicit
identities. Use a fresh coordinator job name and JAX port. Preserve the other fields
for a matched checkpoint/data comparison. A source archive commit has the same
diagnostic implementation plus this notebook and archive-only workflow removals.
Do not rerun the original identity or imply that the archive revision was the
historical launch source. Launching a new rack job requires its own authorization.

Validation before launch: three focused behavior tests and required repository
lint passed. The tests cover tiny CE/gradient preservation, vocabulary-tail
aggregation, and masked/nonfinite rows. A singleton explicit-mesh probe exercised
the streaming 4096-vocabulary tile with a 4097-entry vocabulary. Archival validation
reran the synthetic probe and recomputed all CSV rates from the raw result.

### 2026-10-26 — Archival review correction: scope of the prevalence result

The full-batch table counts rounded-zero CE, which is narrower than all possible
target-gradient collapse. In FP32, `exp(-CE)` can round to one for a sufficiently
small positive CE, giving a zero reconstructed target gradient. The implementation
did not retain a separate production count for that condition. Consequently the
full-batch zero-CE counts are not a bound on the complete target-gradient-collapse
frequency, and the comparison with 0.1% applies specifically to zero CE.

The single sampled positive-tail event was retained only as an aggregate count.
Its individual production LSE, dense target logit, stable margin/CE, and non-target
mass were not saved. The aggregate checks therefore cannot quantify numerical
agreement or practical effect at that event. The archived record preserves the
measurement as performed; resolving either gap would require another diagnostic.
No further accelerator job was launched for archival.

This narrows the earlier low-urgency interpretation: rounded-zero CE is rare on
the fixed panel and the sampled margin tail is small, but a complete production
zero-target-gradient prevalence estimate and an event-level comparison remain open.

## Final decision

The kernel is susceptible. Rounded-zero CE occurred on 0.000048–0.000167% of the
tested tokens, below the 0.1% comparison threshold for that measured condition.
The complete target-gradient-collapse rate was not measured. Preserve this as research evidence and
close issue #10. No code is selected for a maintained Marin contribution and no
Marin contribution PR was opened. The archive index PR is separate documentation.

A production loss replacement was not implemented: the incidence measurement
does not establish a benefit to training. FP64 casting, higher precision, and a
stable-loss continuation were not benchmarked. The stable reference is a numerical
comparator, not an adopted production training path.

Remaining uncertainty: one batch, correlated token positions, three checkpoints,
capacity-limited inference, dense-vs-streaming GEMM rounding, reconstructed logit
gradients, no production count of zero target gradients with positive CE, no
individual values for the sampled event, and no parameter-gradient,
optimizer-continuation, or quality comparison.
Earlier phases and other data mixtures may differ. Z-loss may constrain logit growth,
but its causal protection was not tested. Reopen with new evidence if prevalence
rises or a matched continuation shows a material effect.

