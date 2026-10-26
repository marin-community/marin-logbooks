---
issue: https://github.com/marin-community/marin-logbooks/issues/8
archive_template_revision: f54b3aa0e6ee533e988d2b97e9db0d1523f934dd
marin_base: 8577d258214dd0dcb992c04458ee4eb10496d4e5
---

# Hero Uncheatable forecast repair

Question: why does Hero outperform the Uncheatable forecast before the 4K→16K switch, while its original Paloma forecast tracks better?

Baseline: original Uncheatable fixed-floor 1.4 and October 3 shared-floor refits. Target: seven-subset unweighted dropless cross-entropy, nats/token, 4K packing. Exclude all 16K evaluations. Use Paloma as a control, preserving its original forecast.

## 2026-10-25: recovered sources and exploratory checks

Public inputs are reduced to loss histories, analytic compute totals, output-head fractions, run identities, and original forecast points in `inputs.json`. Configurations and histories were read from public W&B GraphQL on October 25, Pacific time. Frozen H100 histories were recovered from the October 3 forecast artifact, with their SHA256 hashes recorded in the input. The old d1536 source covers only 30%; new-mixture d1536 covers all fractions. Original source: [Paloma scaling code at d23e6e9](https://github.com/marin-community/marin/blob/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/plot_scaling_ladder.py). The analytic output head is `6 * hidden_dim * vocab_size * sequence_length` training FLOPs per example.

Exploratory CPU checks preceded this archive. Excluding output-head FLOPs and refitting the macro floor reduced later 4K Hero error substantially, but several individual subsets became too optimistic. A macro fit can hide opposing subset errors. An early-Hero effective-floor calibration and an equal-parameter offset comparison are required before selecting an update. This is retrospective validation, not a prospective pre-registration.

Plan: profile nonnegative floors with positive-amplitude scaling fits at each training fraction; keep total and body compute conventions separate. Fit ladder-only floors from d512–d1536. Fit effective floors from Hero evaluations through step 179,999; score steps 182,999–254,999. Check d1536 holdouts, derive macro forecasts from subsets, and retain the direct macro model as a diagnostic. A fitted effective floor is not a measurement of infinite-compute entropy. Hero and ladder routing, batch, optimizer, and parameter precision differ.

Reproduction after source publication:

```bash
uv run --no-project --with numpy --with scipy --with matplotlib python logbooks/8-hero-uncheatable-forecast/forecast.py
```

No accelerator work or live training change is part of this task. Final 4K predictions are conditional on the original schedule; they do not predict the later 16K or 262K metric.

## 2026-10-25: narrow the comparison to Paloma's method

User direction: first reproduce the poor Uncheatable forecast, then apply the method that worked for Paloma, and only then diagnose a further repair if needed. Early-Hero calibration and detailed subset model selection are deferred. The previously archived `forecast.py` preserves that unexecuted broader comparison; `compare_methods.py` is the selected experiment.

The Paloma source uses per-percentile log-OLS on `log(loss - floor)` against `log(compute)`, excluding output-head compute. Its floor is fixed at 1.5 for Paloma. Keep Uncheatable's existing 1.4 floor for the first transfer, because the corpus differs. Also test an empirical shared floor fitted from the ladder alone, retaining log-OLS at every percentile. Profile raw-loss SSE to choose this floor. Compare total/body compute and fixed/fitted floors as a 2×2; this distinguishes changing the compute convention alone from refitting the asymptote. Score actual published lines separately because they use different run coverage and sparse plotting points. Use the same 46 Hero points, steps 119,999–254,999, for all comparisons, and report the later 25 points separately. No Hero loss selects the floor or any scaling coefficient.

## 2026-10-25: Uncheatable method comparison

Executed source `b32cc37447` with Python3.12.11, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8 on CPU. Command from the task checkout: `uv run --no-sync python /tmp/hero-forecast-audit-20261025/research/logbooks/8-hero-uncheatable-forecast/compare_methods.py`. The script reads only adjacent frozen public inputs. No training, live evaluation, or context-switch adjustment occurred.

| Uncheatable method | floor | RMSE on46 Hero 4K points | RMSE onlater 25 points |
| --- | ---: | ---: | ---: |
| Published fixed floor/total compute | 1.4 | 0.031107 | 0.034682 |
| Totalcompute/log-OLS/free floor | 1.37444 | 0.021271 | 0.023595 |
| Paloma-style body compute/log-OLS/fixed floor | 1.4 | 0.061911 | 0.063326 |
| Bodycompute/log-OLS/free floor | 1.20343 | 0.006467 | 0.005324 |

Changing the compute axis while retaining 1.4 makes the error worse. Refitting the floor after changing the axis removes 79.2% of the published line's RMSE. At step 254,999, measured loss is 1.581697; the repaired prediction is 1.575678, against 1.619043 published. Its conditional 4K endpoint at the original 390,251-step budget is 1.496758. This is a retrospective diagnostic, with no Hero losses used to choose the floor or scaling coefficients. The ladder itself selected the lower floor. The fitted parameter is an effective floor within this model and compute convention; it does not establish infinite-compute entropy or eliminate training-recipe differences.

The comparison preserves constant H100 run coverage between axis/floor candidates. Original Paloma's actual TPU forecast, scored separately, has 0.008557 RMSE on these 46 points. H100 Paloma refits are controls using different training runs; they do not replace that baseline. Recovered old-mixture d1536 covers only 30%; new-mixture d1536 covers all 15 fractions. The macro is fitted directly; per-subset errors are not guaranteed to improve. See `method-results.json` and `method-comparison.png`.

## 2026-10-25: Paloma floor provenance and next comparison

The [original Hero script](https://github.com/marin-community/marin/blob/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/plot_scaling_ladder.py) hard-codes 1.5. Its [issue discussion](https://github.com/marin-community/marin/issues/8435#issuecomment-5344804457) distinguishes 4K's 1.5 from the assumed 8K floor 1.4. The older [67B record](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980) says prior large-scale 4K runs indicated 1.5. Thus 1.5 was an empirical historical prior, not a fit to the Hero ladder.

User direction expanded to fitting the original Paloma floor and comparing broader 67B/Delphi/ladder evidence. A ~2% miss is acceptable; matching Hero more closely is not sufficient reason to select a floor. Preserve the blind fitted model alongside the historical prior and examine cross-family transfer. Original TPU ladder and historical-data analyses are separate from the H100 control above.

## 2026-10-25: original Paloma ladder and fitted floor

The independent original-TPU-ladder analysis is in `paloma-ladder/`. It reproduces the published 2.039 endpoint as 2.038784599. A shared floor fitted to measured 30-80% ladder points is 1.291409, with a conditional final 4K forecast of 2.007205. Its RMSE on the same 46 Hero points is 0.023931, versus 0.008658 for the recomputed fixed-1.5 baseline. This is approximately 1.1% relative RMSE; the forecast's lower endpoint differs by approximately 1.6% from the original forecast. User guidance: this miss is acceptable and a blind fitted floor may be preferable to choosing a prior solely for its Hero accuracy.

The ladder itself prefers the lower floor: its SSE is 0.000217632 versus 0.000453138 at 1.5. The range with at most 10% additional SSE is 1.21388-1.36167, excluding 1.5. This is a sensitivity range, not a confidence interval. The free floor transfers less accurately to Hero despite fitting the ladder better. Fitting measured 5-80% yields 1.33698; adding the original extrapolated d2048 tail yields 1.35490-1.37438 depending on the progress window. Neither the observed Hero error nor the better local ladder fit identifies the true entropy floor.

The broader comparison uses historical data to test a common floor before adding architecture-specific intercepts. For matched evaluation distribution, tokenizer, packing, and context, entropy is shared. An empirical intercept can absorb distribution mismatch and functional-form error. The [cross-entropy decomposition and its asymptotic assumptions](https://arxiv.org/html/2010.14701v1#S1) clarify why a fit intercept need not equal entropy at available scales. Fit family-specific amplitudes/exponents first; use held-out-family performance to decide whether additional intercepts are supported.

## 2026-10-25: shared historical floor, with Hero left out

`paloma-history/` preserves 81 original Delphi endpoints, 22 retrained endpoints, selected trajectories, the separate 8K 67B check, and the normalized TPU ladder inputs. The primary shared-floor fit uses 44 measured TPU points at 30-80% progress and nine Delphi endpoint budget means from 27 runs. Each family receives half the MSE weight; each TPU progress fraction has separate amplitude/exponent coefficients and Delphi has its own endpoint coefficients. No architecture-specific floor offset is fitted. No Hero loss enters the objective.

The common floor is 1.424157. Scoring its TPU scaling predictions against the separate 46-point Hero 4K trajectory gives RMSE 0.008589, versus 0.008658 for the original fixed 1.5 model and 0.023931 for the TPU-only floor fit. Its conditional final 4K prediction is 2.026369. This supports a shared empirical floor, with family-specific curves, without selecting 1.5 for its Hero accuracy.

Sensitivity matters more than the point estimate: Delphi weights of 25% and 75% give floors 1.355680 and 1.499789. Raw-point pooling gives 1.334710 because the correlated TPU checkpoints dominate the nine Delphi budget means. Selecting observed endpoint minima from the full Delphi grid instead gives a 1.350975 equal-family floor, with selection bias. These are sensitivity checks, not confidence intervals. Related Llama 3 tokenizers and matching 16-subset definitions support approximate comparability, but cache bytes and ordinary-text tokenization were not independently audited. The result is conditional on that comparison.

An unrestricted single two-axis law fitted to the Delphi grid has an effective intercept above measured Hero loss, despite stable multistart optimization and reasonable local held-out-budget performance. Adding records does not remove misspecification. The separate 8K 67B native-evaluation analysis also prefers a floor that transfers worse to its main run; its evaluation sampling differs from the matched-anchor pre-registration. Neither check establishes a physical architecture-specific entropy gap.

Conclusion: the original Uncheatable floor was too restrictive within its extrapolation model; changing the compute convention alone is insufficient. The ladder-fitted-floor repair reduces pre-switch error by 79%. For Paloma, retain both the historical 1.5 prior and the data-fitted alternatives. The common 1.424 historical floor offers a useful primary fitted result, with explicit family-weight and data-selection sensitivity. Preserve the original forecasts and publish the alternatives as forecast-only runs. No training setting or original report changed.

## Publication and verification

[The comparison report](https://forge.coreweave.com/marin-community/marin_moe/reports/Hero-Uncheatable-forecast:-original-vs-Paloma-method--VmlldzoxODE4MjA0Ng==) contains both forecast alternatives. Public API reads verified 15 Uncheatable points, 20 shared-floor Paloma points, three report panels, finished forecast-run states, and `hero_loss_used=false` in each configuration. The final conditional 4K values are 1.496758 and 2.026369. Frozen inputs, results, source, and limitations are archived here. Formatting and lint checks passed on the selected scripts; independent regression checks and the historical multistart checks are included with the results.

[Index PR #9](https://github.com/marin-community/marin-logbooks/pull/9) adds this task to the archive index and remains open. It is monitored by an hourly heartbeat for up to 12 hours; no merge is authorized. The original Marin worktree is clean. No original forecast or live training configuration was changed.
