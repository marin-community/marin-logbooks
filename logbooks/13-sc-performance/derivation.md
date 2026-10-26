# Correction coefficients and output-gradient diagnostics

Let `p` be the current trainer policy, `o` the stored policy before the optimizer update, and `q` the policy that sampled at this exact token prefix. The finalized advantage `A` is held fixed. All probabilities below use the same token IDs and policy/prefix provenance.

For vocabulary action `v`, define

```text
delta_v = log(p_v) - log(o_v)
r_v = exp(clamp(delta_v, -20, 20))
w_v = min(o_v / q_v, TIS_cap)
d_v = 1[abs(delta_v) <= 20]
      * (1[r_v <= 1 + clip_high] if A >= 0 else 1[r_v >= 1 - clip_low])
b_v = w_v * r_v * d_v
```

The ordinary clipped token loss has log-score gradient `-A * b_a` at sampled action `a`. Its score-centering correction has gradient

```text
A * E_(v~q)[b_v * grad(log p_v)].
```

An independent numerical reference differentiates the ordinary clipped surrogate for every vocabulary action, then takes its expectation under `q`. It does not call the production coefficient code. The probability weights and clipping decisions in the centering surrogate are detached.

## Captured head and proportional tail

For captured head `H`, the implementation models both omitted policies as proportional to `p` on the tail, preserving their measured tail masses. Thus `b_v` is constant over the modeled tail. With `alpha = (q_tail / p_tail) * b_tail`, define

```text
h_v = q_v * b_v - alpha * p_v,  for v in H
C_H = A * sum_(v in H) stop_gradient(h_v) * log(p_v).
```

This eliminates the modeled tail through `sum_v p_v * grad(log p_v) = 0`. Tail masses are floored at `1e-6` before computing `alpha`. Head-only and dense correction loss values differ by a zero-gradient offset. Compare their gradients or changes under a fixed detached coefficient; their raw scalar loss values are not an accuracy measure.

## A small exact logit-gradient norm for this surrogate

Let `S = sum_H h_v`. The full-vocabulary derivative with respect to an output logit `z_v` is

```text
grad_z C_H = A * (h_v * 1[v in H] - p_v * S).
```

All omitted entries have the same sign after factoring out `p_v`. Their L1 norm therefore needs only the actual omitted probability mass:

```text
unit_advantage_logit_L1 = sum_H abs(h_v - p_v*S) + max(0, 1 - sum_H p_v) * abs(S)
actual_advantage_logit_L1 = abs(A) * unit_advantage_logit_L1.
```

The existing [`hypothetical_score_probe`](https://github.com/marin-community/marin/blob/5cf919dac0d7df7e1d3c5b6b553a796b1a3bb363/vendor/skyrl/skyrl-train/skyrl_train/objective/score_centering_probe.py) runs before the optimizer update, where current and old policies match. PPO clipping is inactive for either advantage sign at this snapshot. Here `q_v*b_v = min(p_v, TIS_cap*q_v)` and `alpha = min(1, TIS_cap*q_tail/p_tail)`. The ordinary sampled token's logit-gradient L1 norm is `2*abs(A)*w_a*(1-p_a)`. Report both unit and actual advantage correction norms. The ratio is `sum(actual correction L1) / sum(ordinary sampled-token gradient L1)`. Omit it when the denominator is zero. Existing `hypothetical_loss_mean`, `hypothetical_loss_abs_mean` and unit-advantage loss metrics retain their meanings.

This adapts [Prime's L1 diagnostic](https://github.com/PrimeIntellect-ai/prime-rl/blob/dbeb8ce63c7cd9f62d20d4685f34b8c057d94030/src/prime_rl/trainer/rl/loss.py) to clipped PPO plus TIS. Prime uses IcePop acceptance; Slime selects REINFORCE. Neither objective replaces our configured loss. The diagnostic describes output sensitivity of the head/tail approximation. It does not measure parameter gradients or learning benefit.

## Eight independent correction draws

For independent `u_1 ... u_8 ~ q` at the scored prefix, use

```text
C_8 = A/8 * sum_i stop_gradient(b_(u_i)) * log(p_(u_i)).
```

Its expected gradient is the same dense correction above. Draws must be independent of the sampled action and its advantage conditional on the prefix. Duplicates are valid and must retain multiplicity. Drawing from a saved head renormalizes `q` and does not establish an unbiased estimator.

The loss adds positive `C_8` to the ordinary negative PPO surrogate. With unit REINFORCE coefficients, their dense softmax terms cancel exactly in the loss gradient. Clipping and TIS give unequal coefficients. The dense part of this loss gradient is then `A * (b_a - mean_i b_(u_i)) * p` and generally remains. Reported observations about eight nonzero logits in a different objective do not establish that property here.

The pinned vLLM wheel is `0.0.0.dev20260929+marin.39e62869693c.cu132`, [source `39e62869693c46402b1a95fde4fc55ca1aab9ae1`](https://github.com/marin-community/vllm/tree/39e62869693c46402b1a95fde4fc55ca1aab9ae1). Its sampler has no direct multi-draw correction option. Its documented logits-processor interface can observe the actual logits after temperature. A research collector uses a separate CUDA RNG, leaves logits unchanged, and records eight categorical draws. It rejects top-k/top-p/min-p filters because this hook runs before those filters. A same-forward full-vocabulary oracle must check its stored probabilities and prefix alignment on GPU. Capturing the full distribution and sampling on CPU is another compatible reference, with a vocabulary-sized payload per prefix.

## Sampling filters

For actual allowed support `M`, use the recorded normalized `q_M` and normalize `p` and `o` over that same fixed support. Apply the same clipped-PPO/TIS expression on `M`. If the full support is captured, centering is exact there and needs no proportional-tail approximation. Selecting a new trainer top-k set would define another distribution.

The numerical prototype checks top-k followed by top-p on untied logits. The native GPU sampler must establish the actual support, tie behavior and order. Prime's default top-p-before-top-k order and implicit top-k cap are not adopted. Top-p-only support may be wide; measure it before choosing scope. Native sampling-mask replay in this pinned wheel requires Model Runner V2 and excludes custom logits processors. These are separate alternatives, not interchangeable capture settings.

## Initial numerical evidence

[`coefficient_reference.py`](coefficient_reference.py) at source `9acdabb32cc90698e1a6359c93f33fbc91663511` produced [the retained numerical output](synthetic-reference.json). Float64 synthetic cases cover positive, negative and zero advantages at the pre-update snapshot and after policy movement. The L1 formula's maximum absolute error was `3.33e-16`. Twelve conditional-support gradient comparisons had maximum absolute error `3.89e-16`. Enumerating the sampled estimator's expectation matched the dense gradient; 20,000 repetitions agreed with its analytic variance.

In one pre-update synthetic row, the eight-draw estimator's analytic gradient RMSE was `0.399`, versus a dense correction L2 norm of `0.165`. This is a synthetic variance example, not model evidence or a learning result. Real-prefix accuracy, capture feasibility and cost remain pending.
