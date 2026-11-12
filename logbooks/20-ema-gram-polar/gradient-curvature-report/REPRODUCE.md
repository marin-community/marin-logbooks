# Reproducing the gradient and curvature study

The reading guide explains the scientific questions and results. This file records how the artifacts are produced.

## Source and environments

Use the `codex/20-ema-gram-hessian-20261110` branch of the Marin research archive. The exact source commit for each job is in its launch JSON. The path-probe pilots and the later probes used different published commits; their source hashes and configurations are preserved separately. Do not infer a job's source from the current branch tip.

The experiment environment is the Marin checkout's Python 3.12 environment and locked dependencies. Iris builds the TPU environment from the submitted source. The report environment is separate: Python 3.13 with `pypandoc-binary`, PyMuPDF, Pillow, and Tectonic. The browser PDF uses Playwright and Chromium. Plotting uses Matplotlib in the experiment environment.

## Raw evidence

Each checkpoint and process attempt has its own output directory. Its configuration identifies the checkpoint, saved optimizer counter, precision, model, and reference seeds. `data-manifest.json` identifies the actual reference and update sequences by hashes. `measurements.jsonl` preserves incremental progress. A successful process writes its results and `completed.json`.

The three collections have separate roles:

1. `gradient-curvature-records`: loss, gradient, and curvature measurements along the candidate paths.
2. `gradient-curvature-calibration-records`: direct loss evaluations of the predefined step-rule candidates on calibration checkpoints.
3. `gradient-curvature-heldout-records`: separate direct evaluations of selected rules at the excluded checkpoints.

Scheduler success, a matching completion artifact, and a finished W&B run are verified together. Source/configuration and W&B URLs are recorded in launch/completion files. Queue waiting is not recorded as task-active compute.

## Analysis sequence

Run these commands from the Marin checkout, replacing the uppercase arguments with the six corresponding result files, one per checkpoint. Do not mix incomplete attempts or repeat a state.

```sh
.venv/bin/python -m experiments.ema_gram.gradient_curvature_analysis \
  SIX_PATH_RESULTS --output logbooks/20-ema-gram-polar/gradient-curvature-analysis

.venv/bin/python -m experiments.ema_gram.gradient_curvature_analysis \
  SIX_CALIBRATION_RESULTS \
  --calibration-plan logbooks/20-ema-gram-polar/gradient-curvature-analysis/calibration-plan.json \
  --output logbooks/20-ema-gram-polar/gradient-curvature-analysis

.venv/bin/python -m experiments.ema_gram.gradient_curvature_analysis \
  SIX_HELDOUT_RESULTS \
  --heldout-plan logbooks/20-ema-gram-polar/gradient-curvature-analysis/heldout-plan.json \
  --output logbooks/20-ema-gram-polar/gradient-curvature-analysis
```

The first command creates the curvature fits and frozen calibration plan. The second requires completed direct calibration measurements before selecting scalars and creating the held-out plan. The third validates each held-out result against that exact plan before computing paired summaries. Adaptive losses are measured directly, never interpolated from the path plots.

The plans are committed and pushed before their respective launches. The exact Iris argv is saved for every job; these records are reproduction evidence, not an instruction to submit duplicate jobs.

## Figures and documents

```sh
.venv/bin/python -m experiments.ema_gram.gradient_curvature_plots \
  SIX_PATH_RESULTS \
  --analysis logbooks/20-ema-gram-polar/gradient-curvature-analysis/analysis.json \
  --adaptive SIX_HELDOUT_RESULTS \
  --output logbooks/20-ema-gram-polar/gradient-curvature-analysis/figures
```

The figure generator reads the measured results and exports PNG, editable SVG, and PDF. It does not manufacture missing adaptive outcomes.

In the report environment, run `build_methods.py`, then `build_reading_guide.py`. Run `build_reading_pdf.cjs` with Node and Playwright to print the main guide. If using a bundled runtime, set `PLAYWRIGHT_MODULE` to its Playwright package and `CHROMIUM_EXECUTABLE` to its browser executable. The builders use their own directory as the report root.

Check the HTML's local links, figure loading, text-size controls, spacing, section mode, zoom, keyboard behavior, and mobile overflow. Check every final PDF page for missing text, broken equations, clipping, and readable figure labels. Manuscript terminology, reader comprehension, and factual evidence are reviewed independently; audit records identify the exact reviewed hashes.

## Interpretation limits

The reference text and update batches are shared across checkpoints. Holding out a checkpoint tests an excluded model state, not new text. The study has one training seed and separately tuned training learning rates. Its conditional intervals do not cover training-seed variability or scalar-selection uncertainty. Directional curvature is not a certificate of the full Hessian operator norm.
