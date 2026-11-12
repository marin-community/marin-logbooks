# Gradient size, curvature, and the choice of optimizer power

This methods reference explains the experimental protocol. The companion results guide reports the measurements, interpretation, and completion status of each comparison.

## Why run this experiment?

A language model's **loss** measures how poorly it predicts the next token. Lower loss is better. Its **gradient** is a vector that describes how the loss changes when we change the model's parameters. The gradient's length is called its norm.

The **Hessian** describes how that gradient changes as the parameters move. This is one way to measure curvature. Curvature matters because the gradient at the starting point only describes the start of a step. The loss can bend along the rest of the step.

The study compares two optimizer settings, power **1/8** and power **1/4**. C4 is a separate text evaluation; lower loss is better. In the earlier comparison, power 1/8 recorded final C4 loss 3.784025. Power 1/4 had a recovered final evaluation of 3.844011 from a failed job. These runs used one training seed and separately tuned learning rates. This observation motivates the local tests, but does not isolate a causal effect of power. These settings specify the magnitudes of inverse matrix powers applied to statistics of past gradients. Their exact matrix transformation is defined in the matched-comparison section below. A power can change both the direction and the length of its proposed parameter update.

This study asks whether curvature grows when the gradient of a fixed evaluation loss is larger. It also asks whether step length helps explain the comparison between the two powers. These are separate questions. A relationship between gradient size and curvature would not, by itself, identify the better optimizer power.

The **Hessian operator norm** is the largest local rate of gradient change over all unit directions. Generalized smoothness bounds this norm by a function of gradient size, rather than one global constant. The proposed step-length scale is related to a radius around the starting parameters discussed in Lemma 3.3 and Remark 3.4 of [Li and colleagues, 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e8bb8d17bb1cb24dfe972a2f8ff2500-Paper-Conference.pdf). Here, we test a diagnostic based on measured directions. We do not establish the paper's assumptions for the full model.

## What is a matched checkpoint comparison?

A **checkpoint** stores the model's parameters and the optimizer's accumulated state. The accumulated state includes averages of past gradients and their matrix statistics. Every candidate update starts from the same saved parameters and optimizer state. It advances those averages once.

A training trajectory is one sequence of model states produced during training. The study uses six checkpoints: early, middle, and late from each of two trajectories. An update batch is text used to compute a candidate parameter change. Four update batches are sampled once and reused across all checkpoints. Both powers receive the same batch in each comparison.

Each update batch contains eight sequences of 4,096 tokens. Clipping limits the gradient’s norm. Momentum is an exponentially weighted average of past gradients, giving newer gradients more weight. The learning-rate schedule controls update size over training. AdamW is the optimizer used for parameters outside the matrix transformation. Weight decay shrinks parameters toward zero. The production optimizer retains these original operations and settings. Only the inverse matrix power changes within a paired comparison.

For a matrix parameter, the left and right gradient-Gram matrices are exponential averages of the clipped gradient matrix multiplied by its transpose, in the two possible orders. Each average is divided by its own mean eigenvalue. A ridge of 0.01 is added to its diagonal. Call the resulting matrices A and B. Let M be the momentum matrix after incorporating the new clipped gradient with weight 0.05 and the saved momentum with weight 0.95, and let α be the chosen power magnitude.

The polar factor keeps the matrix's singular vectors and replaces its nonzero singular values by one. We approximate this polar factor with the same five Newton-Schulz iterations as production training, denoted polar₅. The matrix direction, before learning-rate scaling and decay, is

\[
A^{-\alpha}\,\operatorname{polar}_5\!\left(A^{-\alpha} M B^{-\alpha}\right)B^{-\alpha},
\qquad \alpha\in\{1/8,1/4\}.
\]

An exact polar decomposition is not substituted for the production approximation.

The nominal learning rate is shared between the two powers within each checkpoint. The saved trajectories used different tuned learning rates. Therefore, differences between the trajectories cannot be attributed to the power alone.

## What loss is measured?

All measurements use the same **reference panel** of 64 sequences. The panel is divided into eight groups of eight sequences. Each sequence has 4,096 tokens. Cross entropy penalizes assigning low probability to the correct next token. Prediction logits are unnormalized token scores. The z-loss penalizes the squared log of the sum of their exponentials. The reference loss is the average next-token cross entropy plus the original z-loss term with coefficient 0.0000001. Weight decay is excluded from this measured loss.

The panel and update batches are drawn from the original training-data mixture. Complete duplicate sequences are rejected within this study. Reference and update batches share no complete sequence. This does not establish that the reference text was absent from the original training run.

A **Hessian-vector product** measures the local change of the full gradient along a chosen vector. Measurements use 32-bit floating-point parameters, loss, gradients, Hessian-vector products, and small parameter displacements called perturbations. Stochastic evaluation is disabled. Production updates retain the original training precision. Keeping these two calculations separate preserves the production direction while improving the numerical reliability of the measurements.

Group gradients are averaged as vectors, with weights proportional to the sum of their token-loss masks, which identify tokens included in next-token prediction. Only then is the norm taken. The same rule applies to Hessian-vector products. Taking each group's norm first would lose cancellation between groups and would measure a different quantity.

At the middle checkpoint from the power-1/8 trajectory, the reference panel is also doubled to 128 sequences. This checks how sensitive the measurements are to the reference sample.

## How are step lengths compared?

A **physical step length** is the Euclidean norm of the complete proposed parameter change. The primary comparison includes matrix updates, AdamW updates, and decay. For each checkpoint and update batch, the power-1/8 update supplies a reference length.

Both powers are evaluated at 0.1, 0.3, and 1.0 times that same reference length. Each direction is rescaled without changing its orientation. This removes a difference in update length from that comparison. The original, unscaled update norms are also retained.

The middle checkpoints have a separate control using only matrix-transform updates. It removes decay and nonmatrix updates, and holds the other parameters fixed. Its reference length is recomputed from the corresponding matrix-only power-1/8 update. Only the movement is restricted; reference gradients and Hessian-vector products still cover the full model. This control is reported separately from complete production updates.

## Which quantities distinguish direction from curvature?

Let x denote the vector of model parameters, and let F denote the fixed reference loss. Let g be its gradient at x, and let H be its Hessian there. Let s be the physical length. Let d be a unit vector pointing opposite the actual parameter change. The candidate move is x minus s times d.

The starting gradient norm is G = ||g||. It is measured without gradient clipping. The **alignment** is the predicted initial loss decrease per unit physical length. Write it as a = gᵀd. Positive alignment means that a sufficiently small move in the proposed direction decreases the reference loss.

The **signed directional curvature** measures how the loss bends along that direction. Write it as κ = dᵀHd; it can be negative. The **Hessian-product norm** measures how quickly the full gradient changes locally in that direction. Write it as Ldir = ||Hd||; it is nonnegative. These quantities answer different questions.

Write g(y) for the gradient evaluated at any parameter state y. The **gradient-change rate** is the norm of the gradient difference between the step’s endpoints, divided by its length: ||g(x − sd) − g(x)|| / s. It averages a change over the whole step. It is shown separately from the local Hessian-product norm.

The direction d stays fixed as the model moves along each path. Independently measured signed curvature along the path predicts the loss change. The prediction adds an alignment contribution and an integrated curvature contribution. Its curvature term is measured from Hessian-vector products. It is not reconstructed from the loss change being predicted. Let u be the fraction of the path traversed, and write H(y) for the Hessian evaluated at a parameter state y. The loss change satisfies

\[
F(x-sd)-F(x)=-sa+s^2\int_0^1(1-u)\,d^\top H(x-usd)d\,du.
\]

The numerical prediction approximates this integral from measured Hessian-vector products. The initial-curvature contribution is s²κ/2. The remaining curvature contribution integrates the difference between curvature along the path and its starting value. We subtract the corresponding contributions for the two powers.

Numerical validation compares representative Hessian-vector products with **central differences**: gradients at equal positive and negative displacements are subtracted and divided by twice the displacement length. Relative error is the norm of the difference from the Hessian-vector product, divided by that product’s norm (with a denominator floor of \(10^{-30}\)). The perturbation sequence is 3, 1, 0.3, 0.1, continuing in alternating factors of approximately three down to 0.000001. Two adjacent sizes must each have relative error below 2%. These representative checks use the first of the four update batches. They include both powers at the starting state and endpoint at one reference length, plus the starting states of the matrix-only controls.

The integral uses composite Simpson quadrature, which combines curvature measurements at evenly spaced positions. Refinement uses 2, 4, 8, then at most 16 intervals. The paired prediction error is the absolute difference between the predicted and observed power-1/8-minus-power-1/4 loss differences. Refinement stops when this error is at most the larger of 0.000002 and 20% of the observed difference's magnitude. Comparisons that still fail at 16 intervals are marked unresolved. Their curvature attribution requires that qualification.

## How are curvature models and step rules tested?

The curvature models predict the local Hessian-product norm from the gradient norm at the same path position. Thus G and Ldir are recomputed together at every position used for fitting. The alternatives are a constant, an intercept plus a multiple of G, and an intercept plus a multiple of G². Both intercepts and coefficients must be nonnegative. Each model targets the 90th conditional quantile: a curve intended to put about 90% of curvature measurements below it at a given gradient size. This is an empirical upper envelope, not a bound covering every direction or every point.

One complete checkpoint is held out at a time. All measurements at that checkpoint are excluded from fitting. Measurements using the same update-batch texts at the other checkpoints remain available for fitting. Each of the other five checkpoints has equal total fit weight. The primary fit pools both powers and all four update batches on complete-update paths measured with 64 reference sequences. Its seven fixed path fractions are 0, 0.05, 0.1, 0.15, 0.3, 0.5, and 1, each measured relative to the reference update length. The full-step numerical-validation endpoint is also one reference length. Within the curvature integral, u instead runs from 0 to 1 over the particular evaluated length s. Extra integration points, matrix-only controls, and the doubled-reference measurements are excluded from this primary fit.

**Coverage** is the fraction of measured values at the held-out checkpoint that lie at or below the predicted envelope. Prediction quality is also reported using absolute error and 90th-quantile loss. For each measurement, quantile loss is 0.9 times the underprediction magnitude or 0.1 times the overprediction magnitude. We average it over measurements within a held-out checkpoint, then weight checkpoints equally in the summary.

The comparison uses a constant physical length and two rules based on the fitted linear or quadratic envelope. Call that envelope ell, and let c be a scalar shared across the two powers. Here G is the gradient norm at the starting checkpoint. The proposed physical length is s = c G / ell(2G), evaluating the envelope at twice that norm. Evaluating ell at 2G can require extrapolation beyond the gradient norms used to fit it. The rule is tested as a diagnostic. It is not certified to produce a safe step.

For each held-out checkpoint, direct loss evaluations on the other five checkpoints choose c. For each calibration checkpoint, first take the median reference update norm over its four update batches. Call the median of those five checkpoint medians m. Each rule receives three calibration candidates, scaled so that the median proposed length across the five calibration checkpoints equals 0.1m, 0.3m, or m. The constant-length candidates are simply 0.1m, 0.3m, and m.

Selection minimizes mean loss change and gives equal weight to checkpoints, powers, and update batches. A separate evaluation then tests the selected scalar at the held-out checkpoint. The held-out loss cannot choose its own scalar.

All rules clip the proposed length to the same interval [0.01m, 3m] within a checkpoint holdout. Both the proposed and clipped lengths are saved, together with the actual displacement after floating-point rounding. Nonpositive alignment is flagged.

## What do the uncertainty intervals mean?

The reported difference is the power-1/8 loss minus the power-1/4 loss. A negative difference favors power 1/8. A positive difference favors power 1/4.

Each interval is reported separately for one checkpoint and comparison. It resamples paired update batches and shared reference groups with replacement. We use 10,000 bootstrap replicates and their 2.5th and 97.5th percentiles for a pointwise 95% interval. Within one bootstrap replicate, the same sampled reference-group indices are used for every sampled update batch and both powers. They describe uncertainty conditional on these checkpoints and the evaluated directions and lengths. They do not include variation across training seeds. Individual path points and TPU hosts are not treated as independent experimental replicates.

## What would the results establish?

The measurements can compare local update directions and physical lengths at the selected checkpoints. The loss decomposition can distinguish initial alignment, initial signed curvature, and further curvature change along the path. A reduction in the loss difference between powers under the tested step rules would support a contribution from step length in these local comparisons.

The full Hessian operator norm is the largest Hessian-product norm over all unit directions. We measure selected optimizer directions. Consequently, these measurements cannot certify a bound on the full Hessian norm. The six checkpoints come from two trajectories using one training seed. They cannot establish a causal explanation for the earlier training ranking or its reproducibility across training seeds.
