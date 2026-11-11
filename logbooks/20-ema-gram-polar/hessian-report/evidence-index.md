# Evidence index

This index accompanies the accessible report. The report states scientific conclusions; these records preserve configurations, numerical checks, execution evidence, and unsuccessful attempts.

## Read first

- `hessian-experiment-report.pdf`: rendered report.
- `hessian-experiment-report.html`: adjustable browser reading view, with native MathML equations.
- `hessian-experiment-report.md` and `.tex`: editable report sources.
- `scientific-results.json`: compact extraction from the detailed follow-up analyses.
- `optimizer-step-results.csv`: all sixty paired optimizer branches, including the applied update norm.
- `checksums.json`: SHA-256 hashes for the delivery files, excluding the checksum manifest itself.

## Motivating training result

`evidence/motivation/single-seed-comparison.json` contains the earlier evaluation scalars and provenance. `single-seed-findings.md` explains which training runs completed and which final evaluations were recovered from failed executions. The report's 3.784025 versus 3.792297 comparison comes from these existing runs; it is not a new training experiment.

The two compared configurations were tuned separately. Their learning rates differ. Their configured ridges also differ, but ridge is inactive at alpha zero. There is only one training seed. The recovered quarter- and half-power endpoints remain supplementary evidence from failed jobs.

## Measurements

The original `early`, `middle`, and `late` records use the quarter-power trajectory and twelve directions. Names beginning `eighth` use the one-eighth trajectory. `wide` measures 48 directions while retaining the original twelve. `controls` builds fresh frozen-weight histories and tests hypothetical complete optimizer updates.

| Record | Verified restored counter | Local evidence | Tracking |
|---|---:|---|---|
| early | 3,733 | [Records](evidence/hessian-early/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-ed298f04d98a) |
| middle | 18,665 | [Records](evidence/hessian-middle/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-035397d82a6b) |
| late | 33,597 | [Records](evidence/hessian-late/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-6537e4fd9860) |
| eighth-early | 3,734 | [Records](evidence/hessian-eighth-early/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-3213c9c34d80) |
| eighth-middle | 18,666 | [Records](evidence/hessian-eighth-middle/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-6361319350e2) |
| eighth-late | 33,598 | [Records](evidence/hessian-eighth-late/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-591d42d60870) |
| quarter-wide | 18,665 | [Records](evidence/hessian-quarter-wide/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-6d1c141ca4c8) |
| eighth-wide | 18,666 | [Records](evidence/hessian-eighth-wide/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-42256e0d131a) |
| quarter-controls | 18,665 | [Records](evidence/hessian-quarter-controls/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-acf65deca9cd) |
| eighth-controls | 18,666 | [Records](evidence/hessian-eighth-controls/completed.json) | [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-dc21859f4564) |

Each `completed.json` binds the restored counter, durable GCS root, and coordinator's online tracking URL. `wandb-verification.json` checks all measurement hosts and, for follow-ups, their logging preflights. New `job-verification.json` files record canonical scheduler success. Hosts cooperate on one calculation; they are not independent experimental repetitions.

`artifact-manifest.json` lists original GCS object URIs, generations, sizes, CRC32c checksums, and local SHA-256 checksums where downloaded. Large basis archives remain at the recorded GCS locations. They are not silently omitted measurements: the small projected matrices and basis-selection metadata needed to reproduce the fits are included locally.

## Which result files to use

- `attention.npz` and `mlp.npz`: group curvature matrices, raw/ridged projected Gram matrices, metadata, leakage, and the added projected finite differences where requested.
- `attention-fit.json` and `mlp-fit.json`: output of the user's original, unmodified analysis program. Its `valid` flag is a numerical logarithm gate, not the added empirical precision screen.
- `aligned-analysis.json`: follow-up mean acceptance and strict bootstrap results. Use `accepted_positive_power` and `empirical_bootstrap`; do not substitute the older bootstrap embedded inside `analysis`.
- `*-projected-checks.json`: perturbation sizes and errors for every checked group/direction. The 2% threshold is unchanged.
- `*-precision-resolution.json`: full-basis empirical screen. The 12-direction restriction of a 48-direction measurement is separately checked in `aligned-analysis.json`.
- `fresh-grams.json`: both fresh histories, projected statistics, effective sample count, and gradient records.
- `full-update-losses.json`: all candidate losses and baseline values on the paired evaluation batches.
- `paired-loss-analysis.json`: differences and conditional batch-bootstrap intervals. One gradient and one checkpoint remain fixed.
- `data-manifest.json` and control data manifests: sequence hashes, accepted source positions, and duplicate-rejection evidence.

No negative Hessian eigenvalues are made positive by flooring, absolute values, or a shift. A negative measured eigenvalue rejects this positive-matrix analysis. The empirical screen exits before calibrating the sign of such a value; rejection does not itself certify a tiny negative curvature.

## Exact sources and launch configurations

The supplied script is preserved unchanged in `analysis/fit_hessian_power.py`, with its original README and dependency list. Synthetic examples are validation cases, not checkpoint findings.

`evidence/*launch.json` contains exact source commits and launch arguments. The original frozen Marin configuration was based on source `14e9a9b2f94e1413cf6c0c60621c7041ea8a4057`. The one-eighth middle projected pilot used `e5acf9e2015018c85be3fa5e520296068728f646`. Other first follow-up measurements used `c194c68722ae258f96c58847c017c565e1b2d602`. The controls precision correction was published as `9ca969f8a3a68a81bcae43deb54b55b91e5783a9`.

`measurement-source/` contains the source used for the final measurement/control implementation and tests. Exact commit bindings remain authoritative for earlier attempts. `evidence/hessian-aligned-configs.json` contains both frozen training configurations; full state restoration and counter checks precede measurement. These jobs do not resume or extend training.

No requested experiment has a new wrapper or scheduler wall-clock limit. Default scheduler preemption/retries are retained. All new accelerators in this campaign are TPUs, not GPUs.

## Unsuccessful attempts

Original derivative-validation failures and the two first native-control compilation failures remain in `failed-attempts/`. Failed attempts are not additional scientific replicates. The first native-control attempts restored the model/state but failed before producing fresh gradients or optimizer losses.

The correction confines the highest-matmul-precision override to FP32 curvature measurements. Native controls retain the restored training precision setting. This fixes a concrete BF16 attention-kernel compilation incompatibility without relaxing the derivative or fit thresholds. A retry is a fresh frozen measurement, not resumed training.

## Reproduce the analysis

From a checkout of the recorded Marin source with its Python environment, run the supplied analysis on a saved `.npz` file. The original program's CLI is documented in `analysis/README.md`.

For the added gates and paired controls, run:

```bash
PYTHONPATH=. .venv/bin/python logbooks/20-ema-gram-polar/hessian_aligned_analysis.py \
  --measurement /path/to/evidence/hessian-eighth-wide \
  --controls /path/to/evidence/hessian-eighth-controls \
  --package /path/to/analysis/fit_hessian_power.py
```

Omit `--controls` for a standalone curvature analysis. Use only `--controls` for paired optimizer-loss analysis. The combined fresh/history comparison verifies that curvature and control basis archives have identical size and CRC32c before comparing coordinates.

Every added analysis records SHA-256 hashes of the postprocessor, empirical-screen source, and supplied script, plus Python and NumPy versions.

The empirical bootstrap uses 1,000 resamples and seed 20261110. Paired loss intervals use 10,000 resamples and seed 20261111. All intervals are conditional on the measured state and the other fixed experimental choices. They do not supply cross-seed training uncertainty.

## Report validation

The delivery includes independent terminology, reader-comprehension, and scientific-evidence reviews against the final report candidate, plus a PDF rendering check. Execution logs and intermediate reviews are separate from the scientific conclusions.
