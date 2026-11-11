# Frozen-checkpoint Hessian pilot: validation blocked

Historical report for attempts 1 and 2. After the user authorized numerical
debugging, attempt 3 passed both blocks' derivative gates and completed. See
the subsequent notebook entry and `hessian-middle/` for the successful pilot.

The real middle checkpoint was restored and evaluated on a v4-32 TPU slice.
The pilot did not pass derivative validation, so it produced no projected
curvature export or power fit. Early and late checkpoint runs were not launched.

## Verified inputs and execution

- Complete checkpoint restore: `step-18664`, state counter 18,665.
- Frozen quarter-power, seed-0 training trajectory; no optimizer updates.
- Eight equal-weight groups, each with 32,768 tokens and 32,760 valid weights;
  no repeated token sequences across groups.
- HQ, code, and math cache object catalogs match the archived sizes and checksums:
  [cache validation](hessian-cache-validation.json).
- FP32 parameters and derivatives, highest matmul precision, vanilla attention,
  unfused cross entropy with the configured z-loss.
- Online W&B preflight succeeded on each host in both attempts. The four hosts
  execute one distributed calculation; they are not independent data samples.
- The selected q_proj Gram products span a factor of about 4.75e9. This spread
  makes a single perturbation scale inappropriate for checking every direction.

## Derivative checks

The first attempt failed on the weakest selected direction at perturbation
scales 0.1, 0.03, and 0.01. Its errors increased as the scale decreased.
The second attempt checked low, middle, and high Gram-product directions at
3, 1, 0.3, 0.1, 0.03, 0.01, and 0.003. Passing requires relative error below
2% at two adjacent scales in every tested direction.

| Direction | HVP norm | Best relative error | Best scale | Gate |
| --- | ---: | ---: | ---: | --- |
| Weakest (0) | 0.000103842 | 0.006971 | 3 | Pass |
| Middle (6) | 0.0393369 | 0.001830 | 0.01 | Pass |
| Strongest (11) | 12.5594 | 0.835707 | 0.003 | Fail |

The loss was 3.8717523 and normalized bilinear symmetry error was 7.19e-11.
These checks do not override the failed finite-difference comparison. The
strong direction's smaller perturbations have not yet been tested; a truncation
error explanation remains a hypothesis, not an established cause. This is not
evidence that the Hessian is indefinite, nor evidence for any preferred power.
The MLP derivative check and all full curvature measurements remain unexecuted.

## Evidence and next diagnostic

- [Attempt 1 launch](hessian-middle-launched.json) and
  [checks](hessian-attempt01/attention-derivative-checks.json).
- [Attempt 2 launch](hessian-middle-attempt02-launch.json),
  [checks](hessian-attempt02/attention-derivative-checks.json), and
  [generation/checksum manifest](hessian-attempt02/manifest.json).
- Attempt 2 source: `bbf7847b72fc8f44e85b1bea56695a75e5e24e8f`.
- [Iris job](https://iris.oa.dev/#/job/%2Fkaiyuew%40stanford.edu%2Fema-gram-hessian-middle-20261110-02)
  was terminal `failed`, one failed task and three coscheduled sibling failures,
  zero preemptions, checked at 17:12 PST on November 10.
- [Publishing host's W&B run](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-1db65afbf04a).
  Iris task 2 was JAX process 0; do not infer artifact ownership from task index.
- Durable diagnostics:
  `gs://marin-us-central2/ema-gram-20-hessian/20261110/middle/ema-gram-hessian-1db65afbf04a/`.

The next diagnostic would extend only the strong-mode finite-difference sweep
below 0.003 (for example, 0.001 through 0.000001), retain full norms and errors,
and require the same 2% adjacent-scale agreement before collecting curvature.
If no convergence window appears, compare differentiation paths on a reduced
model and inspect precision before further checkpoint measurements. Do not
relax the gate to obtain a fit.

Further launches are stopped under the repository Iris monitoring guidance for
repeated or unclear failures. Both attempts ended on application errors; no
GPU allocation was created, cancelled, or released.
