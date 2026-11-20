---
title: "Hinted scaling laws on the hero ladder"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/67"
related_issues:
  - "https://github.com/marin-community/marin/issues/9541"
  - "https://github.com/marin-community/marin/issues/11866"
  - "https://github.com/marin-community/marin/issues/11867"
  - "https://github.com/marin-community/marin/issues/11868"
research_branch: "codex/67-hinted-scaling-laws"
marin_base_revision: "3fe82d09927215cc320276cce31aaa535d5daa73"
archive_template_revision: "0987f59713f018960c143749b9cbda1392e82cb0"
status: active
---

# Hinted scaling laws on the hero ladder

## Question and decision criteria

- Goal and why it matters: decide whether hinted scores on small hero-exact ladder checkpoints forecast Mochi's unhinted base accuracy well enough to use the ladder for SFT-recipe comparisons (marin-community/marin#9541).
- Baseline and alternatives: a two-parameter sigmoid on unhinted ladder scores, and a two-stage compute → bits per byte → accuracy fit.
- Evidence needed to choose: first, the eval and the fit code reproduce the paper's Delphi GSM8K results (marin-community/marin#11868). Then, on the hero ladder, the hinted Mochi forecast beats both baselines by more than the bootstrap interval (marin-community/marin#11870).
- Constraints and unresolved questions: Delphi is used only to validate code; conclusions about Hero come from hero checkpoints.

## Baseline

- Date: 2026-11-20
- Published source revision: the paper's code and data, [tatsu-lab/hinted-scaling-laws@c0490e53](https://github.com/tatsu-lab/hinted-scaling-laws/tree/c0490e53ba8eae58bf5375f6b56956f0d57dad30).
- Configuration, command, and external inputs: the paper's released per-problem rollouts in `data/results/rollouts/gsm8k__{no_hint,mask,truncate}.csv`.
- Result and evidence: the paper's Table 1 and the forecast errors that `experiments/forecasting/plot.py` and `two_stage.py` print.
- Limitations: the paper repository has no license file, so its data is not copied here.

## Entries

### 2026-11-20 22:30 UTC - Fit module reproduces the paper's Delphi fits on the paper's rollouts

- Question or hypothesis and reason for trying it: does `marin.scaling_laws.hinted_fit` (marin-community/marin#11891, for #11867) reproduce the paper's fits when given the paper's own rollouts? This isolates the fitting code from our eval, which is not built yet.
- Published source revision: this branch's base, `3fe82d09927215cc320276cce31aaa535d5daa73` (the head of marin-community/marin#11891).
- Command or material configuration, from the repository root with the paper checkout at `$HSL`:

  ```bash
  uv run --with pandas python -I logbooks/67-hinted-scaling-laws/scripts/reproduce_paper.py "$HSL"
  uv run --with pandas --with matplotlib python -I logbooks/67-hinted-scaling-laws/scripts/plot_delphi.py "$HSL" logbooks/67-hinted-scaling-laws/figures
  ```

- External inputs and artifact references: tatsu-lab/hinted-scaling-laws at c0490e53, `data/results/` (`rollouts/`, `models.csv`, `bpb_delphi.csv`). GSM8K uses the first 256 test problems; a rung's score is the mean over its training seeds (`original`, 42, 62746); 1e23 has only the original seed.
- Environment and hardware when relevant: laptop CPU. No rollouts were generated.
- Result, including failure or restart:
  - All ten Table 1 joint-law parameter sets (five model families × two methods) match to printed precision; the largest absolute difference is 0.0048. GSM8K × Delphi: truncation b1=1.73, b2=2.46, b0=−6.53; masking b1=0.89, b2=4.20, b0=−3.95.
  - Table 1's in-sample joint-fit MAE, averaged over the five families: 3.1 pp for masking and 3.3 pp for truncation, as in the paper. These are in-sample errors. They are not a Delphi forecast error.
  - Forecasting the 1e23 rung from the smallest 7 to 11 rungs, the sigmoid-baseline and ensemble absolute errors equal `plot.py`'s to four decimals.
  - The two-stage baseline errors equal `two_stage.py`'s: 0.1628, 0.2300, 0.2304, 0.2347 and 0.2364 for 9 down to 5 fitted rungs.
- Evidence:
  - [`figures/delphi_hint_curves.png`](figures/delphi_hint_curves.png): observed accuracy against hint strength for all 12 rungs (seed means), with the joint law fitted on all rungs.
  - [`figures/delphi_forecast_t9.png`](figures/delphi_forecast_t9.png): the 1e23 forecast at H=0 from the 9 smallest rungs (3e18 to 6e20).
  - [`figures/delphi_forecast_error.png`](figures/delphi_forecast_error.png): absolute 1e23 forecast error against the number of fitted rungs, ours and the paper's.

  ![Delphi hint curves](figures/delphi_hint_curves.png)
  ![Delphi 1e23 forecast from 9 rungs](figures/delphi_forecast_t9.png)
  ![Delphi forecast error against fitted rungs](figures/delphi_forecast_error.png)

- Interpretation and limitations:
  - Truncation gives a signal only from 1e22 up. Below that, every rung stays under about 0.1 accuracy even with the full reference as the hint.
  - Masking separates the rungs from 9e18 up. The 3e18 rung barely responds (0.005 at H=0.1, 0.014 at H=1), but the law predicts about 0.27 at H=1, the largest misfit in the figure.
  - With 9 fitted rungs, the ensemble forecasts 0.242 for 1e23 against an observed 0.259, and the sigmoid baseline forecasts 0.040. The ensemble is close only because two large errors cancel: masking alone forecasts 0.475, and truncation sits at its floor of 0.008. The hero ladder tops out near 1e21–1e22, where truncation is still flat, so the hero analysis should report each method's forecast as well as the ensemble.
  - This checks the fit code only. Our own eval has not produced any rollouts yet.
- Decision and next action: the fit code is accepted for the gate. Next, generate fresh Delphi GSM8K rollouts with the hinted eval (marin-community/marin#11866) on each rung's original seed, compare per-cell accuracies with the paper's rollouts, and rerun these fits and figures on our rollouts (marin-community/marin#11868).

## Final decision

- Selected approach and evidence:
- Rejected alternatives and why:
- Marin contribution PRs: marin-community/marin#11891 (fit module, open)
- Reproduction instructions:
- Remaining uncertainty:
