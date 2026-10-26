# Historical Paloma floor diagnostics

All fits use historical training runs; no Hero loss enters their objectives. This is exploratory CPU analysis, not a prospective pre-registration. Root owns publication and scoring against Hero.

## Shared 4K floor

`pool_shared_floor.py` fits one nonnegative floor E shared by the original Hero TPU ladder and Delphi endpoint curves. It fits a separate positive-amplitude power law at each of the TPU ladder's eleven measured progress fractions, 30–80%; Delphi has its own amplitude and exponent for its endpoint curve. At each candidate E, log-OLS fits `log(L-E)` against centered `log(C_body)`. The floor minimizes raw-loss residuals.

The primary objective is `0.5 * mean_squared_error(TPU's 44 points) + 0.5 * mean_squared_error(Delphi's 9 budget means)`. This is equal-family weighting, not raw-point pooling. Output-head compute is removed from every run. No architecture-specific floor offset is fitted.

The Delphi budget means use 27 runs: for the seven existing recipe budgets, original seed-0 plus new repeats at seeds 42 and 62746; for the two gap-filling budgets, new seeds 0, 42, 62746. New seed-0 controls at the first two budgets are excluded to avoid counting their originals twice. Exact source IDs are in `shared-floor-results.json:delphi_budget_means[].ids` and their metadata/source URLs are in `historical-inputs.json`.

| Delphi family weight | shared floor |
| ---: | ---: |
| 0 | 1.291410 |
| 0.25 | 1.355680 |
| 0.50 | 1.424157 |
| 0.75 | 1.499789 |
| 1 | 1.588385 |

Raw-point pooling yields E=1.334710 because 44 TPU points outweigh 9 Delphi points. If Delphi instead uses the lowest observed endpoint loss at each of seven original nominal compute budgets in the 81-run isoFLOP grid, the equal-family fit is E=1.350975; Delphi alone is E=1.460737. This minimum selection has winner's-selection bias and is a sensitivity, not the primary result. Its seven exact run IDs appear in `historical-fit-results.json:original_minima_C.ids`.

The endpoints support a practical shared floor around 1.4–1.6; they do not identify a universal entropy floor precisely. Weighting and choice of endpoint envelope matter. The shared fit has no Hero-dependent selection.

## Data contract and limitations

The 81 original Delphi grid runs and 22 retrained runs are dense Qwen3 at 4,096 tokens, use `meta-llama/Meta-Llama-3.1-8B`, and report unweighted 16-subset `eval/paloma/macro_loss` on the val split. Their old cache identities differ from the Hero/67B Marin-tokenizer cache identities. Marin's tokenizer source says it is derived from Llama 3 with chat-template and reserved-special-token renames. This is evidence of closely related tokenization, not a byte-for-byte cache parity audit. The cross-family shared-floor result is conditional on treating those eval contracts as comparable. Do not label cache bytes or ordinary-text tokenization proven identical.

The original TPU ladder is 4K with dropless evaluation and the Grug MoE architecture. Delphi is dense, uses its older training mixture, and uses ordinary Levanter evaluation with no MoE routing. These families are allowed separate curve coefficients. Pooling every endpoint against one compute-only curve would confound parameter count, token count, training mixture, and overtraining.

Delphi metadata spans nonembedding N from 25.2M to 10.74B, and D from 1.10B to 222.6B tokens. The lowest loss among all 103 records is 2.815649 at `delphi-seed-repeats-6e20-seed42-2026.09.22` (4.9401e20 body FLOPs). More records at similar scales do not automatically identify the far-extrapolated floor.

## Other historical checks

Endpoint-only body-compute fits:

- Seven original recipe endpoints: E=1.676352, RMSE .004548; E=1.5 gives .004838.
- Nine budget means of the 22 new repeats: E=1.570061, RMSE .003456; E=1.5 gives .003538.
- Reusing many within-run fractions changes the answer: E=1.784708 for repeat trajectories over 30–100%, E=2.072528 for selected original trajectories. Adjacent checkpoints are correlated and training schedules differ; these are functional-form sensitivities, not additional independent floor measurements.

`fit_history.py` also fits the standard two-axis form `L=E+exp(logA-alpha*log(N_body/1e9))+exp(logB-beta*log(D/1e10))` to the 81 original isoFLOP endpoints. It obtains E=2.516978, RMSE .021017, versus RMSE .043553 at E=1.5. Including all 103 records gives E=2.468015. This is a locally well-determined effective intercept of this particular law; it cannot be a universal 4K Paloma floor because measured Hero losses are below it.

`multistart.py` checked 30 starts (E=.1,1,1.5,2,2.5 and six alpha/beta pairs). All reached the same best SSE .035778079 and floor within 1.4e-7. Parameters are interior: lower bounds [0,-10,.005,-10,.005], upper [min(y)-1e-5,5,2,5,2] for [E,logA,alpha,logB,beta]. Best [2.5169784,-1.3064686,.3198866,-1.3704593,.4549123]. Thus the high intercept is not an observed multistart/local-minimum failure. Holding out the fourteen largest-budget (3e20) original points gives RMSE .037490 with the free floor, versus .068359 at fixed1.5, but both remain limited-range extrapolations. Do not interpret either as an architecture-dependent physical entropy measurement.

## 67B, separate 8K check

`fit_67b.py` fits the three native June-prep small-model histories at 8K, body compute, per-fraction log-OLS, shared E, then scores the 67B production 8K continuation separately. Over 30–80%, the small-rung floor is E=1.616184; its ladder RMSE is .001148. E=1.4 gives ladder RMSE .001717. On 24 native main-run points in the pre-mixture-swap window, free-floor RMSE is .043004 versus .017440 for fixed1.4. Changing the small-model fit window gives E=1.642–1.664. This result is a separate context-family comparison, not evidence for merging 8K and 4K raw losses.

The native small-model eval configuration uses eval_batch_size=64 and max_eval_batches=8; production uses a larger eval shape. These are not the matched 1024-sequence EP=1 reload evaluations in the original pre-registration. The larger 67B architecture, batch/optimizer changes, and eval sampling also differ from its small anchors. Later 65K cooldown results are excluded from this fit.

## Source links

- Delphi recipe reconstruction: https://github.com/marin-community/marin/blob/8577d258214dd0dcb992c04458ee4eb10496d4e5/experiments/references/train_delphi_seeds.py
- Original TPU forecast source: https://github.com/marin-community/marin/blob/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/plot_scaling_ladder.py
- Prior large-scale Paloma floor precedent and 8K assumption: https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980
- Original 67B matched-anchor fit: https://github.com/marin-community/marin/issues/6044#issuecomment-4819900707
- Public W&B run URLs are included per record in the sanitized inputs.

## Essential files and reproduction

Keep `historical-inputs.json`, `historical-trajectories.json`, `models.json`, `rav-inputs.json`, `67b-inputs.json`, and these analysis scripts/results. Raw public full configs in other JSON files are discovery artifacts and are unnecessary for reproducing the fits.

Run scripts with Python 3.12, NumPy, and SciPy; no network or accelerators required:

```bash
python fit_history.py
python curve_checks.py
python pool_shared_floor.py
python fit_67b.py
python multistart.py
```

Results are in `historical-fit-results.json`, `historical-curve-results.json`, `shared-floor-results.json`, `67b-fit-results.json`, and `multistart-results.json`. The `*_profile` arrays are fit sensitivity scans, not confidence intervals.
