# Original Paloma ladder floor fit

The original fixed floor of 1.5 transfers to Hero better than the floor estimated from this ladder. Fitting the floor on the measured 30–80% portion gives 1.2914, lowers ladder fit error by 52%, and worsens pre-switch Hero RMSE from 0.00866 to 0.02393 nats/token. No Hero loss is used to fit a floor, amplitude, or exponent.

| Floor procedure | Fitted floor | Conditional final 4K Hero loss | Hero RMSE, all 46 points | Hero bias, all 46 points |
| --- | ---: | ---: | ---: | ---: |
| Original fixed floor | 1.5000 | 2.03878 | 0.00866 | +0.00569 |
| Fit measured 30–80% | 1.2914 | 2.00721 | 0.02393 | −0.02294 |
| Fit measured 5–80% | 1.3370 | 2.01345 | 0.01848 | −0.01720 |
| Fit 30–100%, including extrapolated d2048 points | 1.3549 | 2.01600 | 0.01633 | −0.01487 |
| Fit 5–100%, including extrapolated d2048 points | 1.3744 | 2.01883 | 0.01400 | −0.01229 |

The main fit uses only measured ladder points, 30–80%, so the extrapolated d2048 tail cannot select its floor. The conditional 100% prediction still uses the original source's d2048 tail extrapolation, for parity with its 2.039 prediction. The other fit windows are sensitivity checks, not independent validation runs.

On the main measured window, the profiled raw-loss SSE is 0.00021763247 at the optimum and 0.00045313843 at floor 1.5, a ratio of 2.0821. Floors with no more than 10% extra SSE range from 1.21387711 to 1.36166712. This is an arbitrary fit-sensitivity range, not a confidence interval. Floor 1.5 is outside it. The failure is therefore not explained by claiming both floors fit this ladder equally well. The lower floor fits this ladder better and transfers worse to Hero. The shared floor also changes with the progress window and the inclusion of extrapolated data.

## Method and sources

- Original method: [Paloma scaling script at d23e6e9](https://github.com/marin-community/marin/blob/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/plot_scaling_ladder.py). `COMPUTE` in the analysis copies its exact lm_head-excluded constants.
- Rungs: `rav-ladder-d768-v2`, `rav-ladder-d1024`, `rav-ladder-d1536`, and `rav-ladder-d2048-v3`, all in `marin-community/marin_moe` on W&B. The first three have 20 evaluated fractions, 5–100%; d2048 has 16, 5–80%. Their public query results were fetched by the parent analysis on October 25, 2026, Pacific time.
- `rav-inputs.json` preserves only rung identities, stop steps, and Paloma history values from those queries. It records SHA256 hashes of the parent's raw public W&B JSON files. No external fetch is required to reproduce this fit.
- The original script snaps evaluation fractions with `round(round(step / stop * 20) / 20, 2)`. This analysis preserves that convention exactly. It uses the last row in a duplicate bin.
- At each fraction, fit `log(loss − E)` against centered `log(compute)` by ordinary least squares. Keep amplitude and exponent independent by fraction; share one nonnegative floor `E` across the chosen fractions. Profile raw-loss SSE to estimate `E` and search its whole feasible interval before local refinement. This differs from optimizing log-residual SSE.
- d2048 fractions above 80% use the source's linear 60–80% extrapolation plus 0.005 loss. These points are excluded when selecting the main floor.
- Score the same 46 measured Hero evaluations, steps 119,999–254,999, as the parent Uncheatable comparison. The later slice contains 25 points, steps 182,999–254,999. Scoring interpolates the predicted 5% curve at `(step + 1) / 390251`. No 16K observation is included.

The fixed-floor endpoint is 2.038784599, reproducing the rounded published 2.039. Its scored RMSE differs slightly from the stored published line's 0.00855674 because the published values are rounded. This script recomputes the underlying curve from the frozen ladder values.

## Reproduce

Place this directory under `logbooks/8-hero-uncheatable-forecast/paloma-ladder/`, next to the parent's `inputs.json`. Then run:

```bash
uv run --no-project --with numpy --with scipy --with matplotlib python logbooks/8-hero-uncheatable-forecast/paloma-ladder/fit_original_paloma.py
```

`results.json` records curves, profiles, source counts, and checks. `results.png` plots the original fixed and fitted forecasts against Hero, plus the floor profile. Embedded numerical checks recover the fixed-floor curve independently with centered normal equations, verify the published endpoint, check the d2048 measured cutoff, and confirm the 46-point scoring window.

This is a retrospective comparison under an assumed power law. Its floor is an effective model parameter, not a direct measurement of the infinite-compute entropy of Paloma. Adjacent checkpoints are correlated. Cross-family inference should distinguish training regime and context rather than treating every checkpoint as an independent scaling rung.
