---
title: "Gated DeltaNet-2 TPU kernels"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/42"
related_issues: ["https://github.com/marin-community/marin/pull/9350"]
research_branch: "codex/42-gdn2-tpu"
marin_base_revision: "acd9b1193678436b3c6b714b06f297ace586c50c"
archive_template_revision: "261c88369ab6bfe81d297a2b0f78710e139c99c3"
status: complete
---

# Gated DeltaNet-2 TPU kernels

Compare a single-device Gated DeltaNet-2 Pallas candidate with Atomic Ops. Accept a timing only after the output, final state, and gradients for all six token inputs and the initial state pass an independent FP32 CPU recurrence. The comparison uses the fastest accepted baseline tile for each shape and dtype.

## September 2026 sweeps

The candidate replaced off-diagonal pairwise exponentiation and reduction with boundary-centered matrix products. The diagonal calculation masked noncausal pairs before exponentiation. TPU v4 needed a feature-first diagonal score layout because its compiler rejected the feature-last reduction. The candidate used an exact triangular solve and accepted strong negative log-decays without clipping.

The preserved source snapshots, hardware settings, seeds, tile grids, timings, and rejected configurations are in the [measurement report](measurements/README.md). At batches 4 and 8 with 4096 tokens, six 128-dimensional heads, and FP32 or BF16 input, the snapshots measured 3.05–3.27× faster forward+backward on v5e/v5p and 3.12–3.22× on v6e. The v4 baseline failed to compile, and candidate BT256 backward exceeded VMEM. BT128 won on all four devices, with MB16 on v4/v5e/v5p and MB32 on v6e. Batches 5–7 were unmeasured.

To reproduce a historical row, check out the base revision in its source manifest and restore the manifest's `files` mapping at the recorded paths. Use the hardware, flags, and command in the report. The manifests preserve the source contents and digest. Restricted bundle locations are omitted from the public archive.

## November 2026 correctness check

The refactor moved the baseline to the published `atomic-ops==0.1.0` wheel, shared the candidate's forward stages and scan, and removed an unused backward operand. The benchmark and kernel source in this branch were extracted from [Marin commit `c856c4b8d0`](https://github.com/marin-community/marin/tree/c856c4b8d0cb39a3ff1f0e1c1de9a50104814a58). The smoke test ran the same executable sources at [`c82bff30aa`](https://github.com/marin-community/marin/tree/c82bff30aa4be02ccbf1be800fe4f3b36dc8bcac), with the intervening commit changing documentation only.

The v5e smoke test used shape `[1,256,1,128]`, BT128/MB16, FP32/BF16, one warmup, two repetitions, JAX 0.11.1, and `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`. All eight candidate and baseline forward and forward+backward comparisons passed the CPU oracle. All 12 stage probes passed. The local CPU interpretation matrix passed all 12 cases. See the [benchmark commands and tile guidance](benchmark.md).

## Decision

Keep the harness, tuning cache, stage probes, and measurement evidence here. [Marin #9350](https://github.com/marin-community/marin/pull/9350) carries the candidate, independent reference, parity tests, and their user-facing contract. Representative-shape performance, best tile choices, and v4 VMEM fit remain unverified for the refactored candidate.
