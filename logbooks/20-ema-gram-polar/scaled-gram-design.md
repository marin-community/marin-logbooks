# Spread full Gram solves over more TPU devices

Test the existing grouped-Gram implementation on 32 or 64 JAX v4 devices,
using a ready v4-64 or v4-128 slice in us-central2. The working baseline remains
matrix/default: independent Gram-bank eigendecompositions with the default solver.
A bank is one weight group's batch of eleven matrices; a shard assigns whole
square matrices to one JAX device. matrix_grouped concatenates equal-size banks
before distributing them. No optimizer arithmetic changes are proposed.

On the completed 16-device trial, three banks each contained eleven large
4096-square Grams. Independent banks assign at most three large matrices to
each shard across the three calls. Grouping all 33 still assigns at most three
large matrices per shard, so that trial did not reduce this maximum. With 32
shards the grouped maximum falls to two; with 64 it falls to one. The
independent-bank maximum remains three. These are source assignment counts,
not verified kernel counts, serial execution counts or a promised speedup.
The total 33 real eigendecompositions is unchanged.

The next distinct diagnostic compares matrix and matrix_grouped on the selected
larger topology, using the original regional step-923 checkpoint's 21 Gram and
momentum arrays, identified in [checkpoint metadata](loss-pair-checkpoint-metadata.json).
Seven weight groups each supply momentum, left Gram and right Gram arrays;
three of those Gram arrays each contain eleven 4096-square matrices. Gradients
use NumPy seed 0 and the shapes/scaling in [the harness](scaled_gram_check.py).
One case computes seven weight-group directions from immutable saved state.
The other refreshes momentum and both Grams from the synthetic gradients before
computing those directions; every replay starts from the same saved inputs.
Each case runs matrix then matrix_grouped, with compilation/first call separated
from three timed calls. Each timer ends after local output completion; finite
checks, host copies and replay-hash comparisons occur outside it. It performs no model update or training continuation.
All full square matrices, normalization, full-update refresh, precision, default
eigensolver, ridge and polar computations remain unchanged. Each phase releases
its executable and temporary outputs before the next. Replicated saved inputs
remain about 3.46 GB per process; this does not establish peak accelerator or
host memory fit on the new topology.

Prefer an available ready v4-128 slice (64 devices, 16 hosts, 8 by 8 data/model
mesh); otherwise use ready v4-64 (32 devices, 8 hosts, 4 by 8 mesh). Each host
requests 120 CPU, 256 GiB RAM and 50 GiB disk. Both meshes divide width 1024.
No queue, task or wrapper wall-clock limit or preemption retry override is added.
A capacity snapshot is advisory and can change before assignment. The launcher
submits nothing if neither shape is ready and available, and refuses a second
launch attempt under its recorded identity in scratch/20261104-scaled-selection.json. It does not cancel or alter other
allocations. A November 4, 2026 08:13 UTC fleet snapshot had no free slice of either shape;
a free v5p-8 in us-central1 had four JAX devices, which cannot test the proposed
32/64-way distribution, and the saved checkpoint remains in us-central2.
TPU remains the default; this trial's shape and region are experimental choices,
not restrictions on future research or authorization to move large data.

The CPU harness uses fresh online runs on eight and sixteen virtual devices.
Its three-layer fixture gives nine large Grams, exercising the transition from
two to one assigned per shard. Finite outputs, input immutability, replay hashes,
checkpoint roundtrip, identity power for zero Grams, known powers for diagonal
Grams, mixed sizes, uneven padding and equality when alpha is zero are checked. Optimizer/test hashes remain
identical to the prior [single-device](grouped-gram-regression-single-validation.json)
and [matrix](grouped-gram-regression-matrix-validation.json) online regressions
(17 tests). CPU agreement is not TPU
performance or memory evidence; all rtol=atol=1e-5 failures remain recorded and are not a sole training veto.
Eight-device CPU comparisons pass; the sixteen-device full-update directions
reach maximum absolute difference 0.00635338 and fail that diagnostic threshold.
Both CPU fixtures are finite, preserve inputs and reproduce outputs bitwise
three times. Fixed-state direction comparisons have zero measured error.

Require every process report from one successful attempt, exact source and
checkpoint identities, finished online logging, finite immutable inputs and
replica hash/statistic agreement before global deduplication. At 32 devices,
expect 224 unique direction leaf/index records and 245 including states; at 64,
448 and 469. Each direction is divided into one slice per device, yielding
7 times the device count; the 21 state arrays are replicated and counted once
after hash/statistic agreement. Only correctness records are deduplicated.
Keep all raw times and separate compilation; never sum host times.
Report device-time cost with latency because this uses two or four times the
previous device count. A faster optimizer diagnostic does not establish training
throughput, MFU or quality. This changes execution parallelism; it adds no calibration phases or entry
barriers to the existing diagnostic.
