# Reconstruct matrix powers before gathering

The working matrix baseline takes 3.70021 seconds per training update in the
[924-update comparison](prefix-train-comparison.json), run `2026.11.04.01`. The saved profile places substantial recorded device
work under `relative_gram_power`; it does not measure the isolated wall-time
cost of its reconstruction matmul.

The implementation in `lib/levanter/src/levanter/optim/ema_gram.py` normalizes full Grams, shards complete eigendecompositions,
gathers eigenvectors and eigenvalues, and reconstructs matrix powers with
replicated output constraints. The candidate `matrix_power` keeps each power's
clamping, exponentiation and reconstruction matmul inside the same full-matrix
shard before gathering powers. It retains the same normalization before the
shard and the same zero-Gram identity selection afterward. Full Gram refresh,
momentum, polar iterations, arithmetic precision and default eigensolver remain
unchanged. Here a matrix shard contains whole independent square matrices;
the leading matrix-batch dimension is distributed over the combined data/model
mesh axes, and neither square matrix axis is split by this shard map. The default `matrix` and `replicated` branches remain available.

The hypothesis is that avoiding gathered intermediate eigenpairs permits less
replicated reconstruction work. The largest gathered matrix payload remains
matrix-sized, so no large communication-volume reduction is assumed. Compiler
fusion, layout and floating-point rounding may change. Padding is unchanged;
the design does not rely on padding having caused the prior eigensolver-scope
split between eleven chips near 3 seconds and five near 3 milliseconds.

The diagnostic compares `matrix` then `matrix_power` sequentially on the same
original saved Gram/momentum state and fixed synthetic gradients. The source is
the [original checkpoint labelled step 923](loss-pair-checkpoint-metadata.json),
version `2026.10.30.23`, in us-central2; it is not the newer matrix checkpoint. It checks
unchanged-state directions (seven outputs) and complete refresh/update (seven
directions plus 21 states). The seven banks are q/k/v/o, gate, up and down
projections; each has momentum, left Gram and right Gram state. Every output must be finite; input-state and
gradient fingerprints must remain unchanged. Elementwise `rtol=atol=1e-5`
comparisons and repeated-output hashes are diagnostics, not loss-acceptance
vetoes. Preserve all four process reports and verify repeated replica hashes
before aggregating global norms. Replica agreement means equal leaf/index
regions, independent of host; replay hashes compare the same inputs and
compiled function on successive executions. A finite disagreement alone does
not reject the candidate, but must be reported. This harness cannot establish
training-quality acceptance, which requires a matched loss comparison. The selective state read is not trainer
restore, and this harness executes no training steps.

Each case reports compilation plus first execution separately from three
warmed, synchronized samples. Finite checks and host copies are outside the
sample timer. These samples measure the optimizer harness, not training MFU or
an end-to-end training speedup. Three samples in fixed baseline-then-candidate
order leave timing variability and order effects unresolved for small changes. Compare per-process records without summing
host durations or treating samples as independent seeds. A finite candidate
that is no faster gives evidence against this intervention for this saved state
and harness; it must not be
called a training improvement without a subsequent matched training measure.

CPU validation uses four virtual devices in a 2×2 data/model mesh, three layers
and reduced matrix widths. It includes checkpoint round-trip, the two cases,
zero-Gram identity, rank-deficient finiteness, an independent NumPy spectral
reference, and exact regression with Gram power exponent alpha=0 (passive Gram tracking). The first fixture used only a data
axis and failed the native polar mesh requirement; that failure is retained.
The corrected fixture and each new process require fresh online tracking.

After this implementation passes its logging/CPU checks and exact source publication, one TPU diagnostic
may use a v4-32 slice in us-central2, four hosts with 120 CPUs, 256 GiB RAM and
50 GiB disk each. It preserves default preemption retries and has no wall-clock
limit. Each attempt executes both cases, both layouts and three warmed samples
per layout. A preempted attempt restarts this finite diagnostic from the same
saved state; retain its partial reports separately, and accept only a complete
set of four reports from one attempt. Every host first performs an online non-training logging check; every
numerical process then owns a fresh online run through Levanter's tracker.
Keep checkpoint arrays regional and retrieve only bounded reports. No new
training or extension of the completed 924-update prefix is part of this test.
