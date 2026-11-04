# Group same-size Gram banks across eigensolver calls

The user rejected further measurement-only refinement. The host-barrier timing
plan is superseded and will not be implemented or launched. Keep `matrix` as
the working training baseline and test `matrix_grouped` as a new numerical
execution layout.

The current scanned model has fourteen independent Gram banks: eleven banks
of eleven 1024-square matrices and three banks of eleven 4096-square matrices.
The new layout normalizes each bank using the same source expressions, groups
normalized full matrices by square size and dtype, then performs one
matrix-parallel eigensolver call per group. It splits the eigenpairs back into
the original bank order before applying the unchanged clipped-eigenvalue/ridge
power reconstruction and polar direction calculation. State refresh and model
parameter layout are unchanged; this is not whole-update sharding.

This changes fourteen source-level eigensolver shard-map invocations to two;
it is not a measured runtime kernel/collective count. On sixteen
devices, small-bank padded positions decrease from 176 to 128. Large-bank
padded positions remain 48 (33 real matrices), so this is not a claim that
large-matrix arithmetic has decreased. The intended gain is fewer separate
dispatches/collective boundaries and better scheduling of independent banks.
Batched convergence and larger live tensors could offset that gain; only the
new implementation's measured result can decide adoption. No speedup is assumed
from padding counts or prior summed profile durations.

Preserve full Gram geometry, every-update refresh, source FP32 arithmetic,
HIGHEST reconstruction multiplies, default eigensolver, zero-Gram identity,
eigenvalue clipping, ridge and polar iterations. No approximation, matrix-axis
tiling, precision override, or refresh-frequency change. Defaults stay unchanged.

Use a fresh online non-training logging check and online CPU harness first.
The CPU harness checks mixed square sizes, bank order/split boundaries, an
uneven grouped batch, zero and rank-deficient Grams, independent diagonal
power oracles and recorded random full-rank NumPy diagnostics, checkpoint roundtrip, both fixed-state and full-update
finite/input-immutable outputs, and equality with the baseline when preconditioning exponent alpha is zero. Extend the native
optimizer/checkpoint regression to select the new layout. Preserve numerical
rtol=atol=1e-5 as a reported diagnostic, not a training veto.

After exact source/manifest publication, one distinct sequential v4-32 job will
compare `matrix` and `matrix_grouped` using the [existing saved-state harness](power_reconstruction_check.py)
protocol. It is an acceleration experiment, not a rerun of `matrix_power` or a
measurement-calibration job. Preserve all three timed samples and known timing
limitations; no extra calibration phases or barriers are added. Require four
complete process reports from the same job attempt, finite outputs and unchanged
inputs, source/checkpoint identity, online logging and Iris job/all-four-task
success before acceptance.
Keep the original regional checkpoint and all large arrays in us-central2.
No training extension is implied by this diagnostic.
