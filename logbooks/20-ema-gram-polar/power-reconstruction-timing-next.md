# Controlled timing design after the first power-reconstruction diagnostic

Keep `matrix` as the working training baseline. The completed `matrix_power`
diagnostic does not establish a speed benefit: fixed-state direction means are
slower on every process, and full-update mean ratios disagree across processes.
The fixed baseline-first order and three samples leave timing-order effects
unresolved. Numerical differences alone are not the reason to withhold adoption.

The next bounded question is whether the candidate reduces full optimizer-update
latency when hosts enter measured calls together. This is a measurement change,
not a new optimizer implementation. No code or job for this design is submitted.

The existing harness blocks on each local output, then copies and validates
shards outside the timer. It has no explicit cross-host entry barrier before
each timed call. Per-host copying/checking can therefore affect when a host
enters the next collective-containing call. This is a possible confound, not a
measured cause of the observed timing variation.

Prepare one diagnostic of the same full synthetic-gradient refresh/update on
the original step-923 saved Gram/momentum state on 16 v4 devices across four
hosts, preserving FP32 source arithmetic and the `matrix`/`matrix_power` layouts.
The [completed diagnostic](power-reconstruction-auth-completed.json) and
[source manifest](power-reconstruction-manifest.json) pin these inputs. Use phase
order `matrix`, `matrix_power`, `matrix_power`, `matrix`. Compile and warm each
phase outside measurement; retain phase identities and all compilation times.
Release each phase's compiled function and outputs before preparing the next,
so the design does not require simultaneous residency of both executables.

For each of ten measured calls per phase, use a named all-process barrier
before starting the local timer, block on all local output leaves, stop the
timer, and then use an exit barrier. Copying, finite checks, replay hashes and
input-state/gradient fingerprints remain outside the timer and precede the
next entry barrier. Retain every sample; do not discard slow first calls after
seeing results. Require the same complete sample/phase identifiers on all four
processes. Report raw per-host timings and the maximum of the four durations
for each aligned call; this is an operational latency measure, not a sum of
host times or proof of exact simultaneous start. Report spread and phase-order
dependence, without inventing a significance threshold.

Before one distinct sequential TPU diagnostic, validate the changed measurement
entrypoint with its own fresh online non-training logging check and online CPU
harness, then publish the exact source/manifest. CPU checks must cover barrier
and phase ordering, complete sample IDs, finite outputs, unchanged inputs and
retention of numerical tolerance failures. Preserve default preemption/retries,
all attempt evidence and no wall-clock limits. Keep all large state regional.
The completed job and failed log observer must never be restarted or replaced.

Only a repeatable optimizer benefit would motivate a separate native training
speed/loss check. Neither this design nor the completed synthetic diagnostic
changes the [accepted 924-update training result](prefix-train-completed.json), establishes training MFU, or
extends the completed training prefix.
