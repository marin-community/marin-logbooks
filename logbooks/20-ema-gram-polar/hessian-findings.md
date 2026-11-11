# Real Marin Hessian-power experiment

Completed the requested early, middle and late frozen-checkpoint measurements. All three jobs succeeded on all four TPU hosts, all 12 measurement W&B runs finished, and both blocks passed the unchanged derivative-validation gate at every checkpoint.

No attention-block power fit is accepted: early and middle are indefinite, and late fails the input symmetry check. The MLP block passes the supplied numerical positivity gate and admits the conditional subspace fits below. These are not measured training-loss optima.

The late MLP result is unstable to batch resampling: 228 of 1,000 replicates fail positivity. Its reported interval includes only the 772 valid replicates and is not an unconditional 95% uncertainty interval. The early and middle MLP fits have no invalid bootstrap replicates.

## MLP results

| Counter | Proxy | r | rho | alpha_log (conditional 95% interval) | Best direct-grid alpha | Leakage | Valid bootstrap / 1000 |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 3,733 | raw | 1.2449 | 0.9376 | 0.3766 [0.3705, 0.3842] | 0.375 | 0.490 | 1000 |
| 3,733 | ridged | 1.1627 | 0.9422 | 0.4052 [0.3989, 0.4129] | 0.405 | 0.490 | 1000 |
| 18,665 | raw | 1.1941 | 0.9159 | 0.3835 [0.3814, 0.3863] | 0.385 | 0.297 | 1000 |
| 18,665 | ridged | 1.1283 | 0.9295 | 0.4119 [0.4096, 0.4148] | 0.410 | 0.297 | 1000 |
| 33,597 | raw | 1.1339 | 0.9300 | 0.4101 [0.4078, 0.4195] | 0.410 | 0.382 | 772 |
| 33,597 | ridged | 1.0789 | 0.9419 | 0.4365 [0.4341, 0.4471] | 0.435 | 0.382 | 772 |

Conditional 95% intervals for r and rho, held-out split diagnostics and the complete congruence grid are retained in the linked fit JSON files.

| Counter | Proxy | Relative off-diagonal curvature | Relative log commutator | Fitted log residual | r=2 residual | r=4 residual |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3,733 | raw | 0.0697 | 0.00088 | 0.0624 | 0.4073 | 4.6549 |
| 3,733 | ridged | 0.0697 | 0.00091 | 0.0578 | 0.5463 | 5.6684 |
| 18,665 | raw | 0.0463 | 0.00142 | 0.0841 | 0.5012 | 5.1414 |
| 18,665 | ridged | 0.0463 | 0.00138 | 0.0705 | 0.6252 | 6.0914 |
| 33,597 | raw | 0.0260 | 0.00348 | 0.0700 | 0.6127 | 6.0123 |
| 33,597 | ridged | 0.0260 | 0.00354 | 0.0581 | 0.7446 | 6.9628 |

The ridge is the production factor-wise normalized ridge of 0.01. All fits use the full projected 12-by-12 matrices, including off-diagonal entries. Each interval resamples eight whole curvature groups 1,000 times and conditions on the saved EMA, checkpoint and selected subspace; it excludes EMA-history and training-seed uncertainty.

## Positivity and resolution diagnostics

| Counter | Block | Minimum mean eigenvalue | Maximum mean eigenvalue | Nonpositive bootstrap minima / 1000 | Leakage |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3,733 | attention | -0.00107474 | 0.902618 | 1000 | 0.419 |
| 3,733 | mlp | 6.45512e-07 | 9.15153 | 0 | 0.490 |
| 18,665 | attention | -0.0766894 | 2.11911 | 1000 | 0.927 |
| 18,665 | mlp | 1.58713e-06 | 37.4807 | 0 | 0.297 |
| 33,597 | attention | -0.271356 | 30.4675 | 1000 | 0.693 |
| 33,597 | mlp | 9.94963e-07 | 47.5177 | 228 | 0.382 |

The late attention block fails the original input symmetry gate: batch 7 has relative asymmetry 1.55066e-5, above 1e-5. Its spectrum and bootstrap in this table are diagnostics of the symmetric part, not an accepted input or power fit. The tolerance was not relaxed.

No negative eigenvalues were replaced by absolute values, floored, or shifted. The positivity threshold of 1e-12 is a numerical gate, not an empirical measurement-error bound. The files below include sensitivity gates from 1e-12 through 1e-6, individual-group spectra, bootstrap spectra and diagonal-scaling diagnostics. The MLP spectra span a large range; narrow batch intervals do not establish floating-point accuracy.

| Counter | MLP minimum/maximum eigenvalue | Pass at relative gate 1e-7 | Minimum diagonally scaled eigenvalue |
| ---: | ---: | --- | ---: |
| 3,733 | 7.05e-08 | False | 0.8669 |
| 18,665 | 4.23e-08 | False | 0.8352 |
| 33,597 | 2.09e-08 | False | 0.4832 |

Leakage is computed from the averaged full block HVPs. It is substantial, so the selected subspace is not approximately invariant enough to justify extrapolating these exponents to the full parameter block or recommending a training optimizer power.

## Procedure and validation

- Frozen alpha=0.25, seed=0 trajectory; checkpoint directories step-3732, step-18664 and step-33596.
- First flattened q_proj (1024-by-1024) and gate_proj (4096-by-1024) slices; raw saved gradient-Gram EMAs.
- Twelve distinct rank-one Gram eigenmodes spread over the resolved log-product spectrum. Modes are selected separately from each checkpoint's EMA, so subspaces change across checkpoints.
- Eight disjoint groups of 32,768 tokens each; 32,760 valid weights per group. Group token hashes are identical across checkpoints. These are fresh samples of fixed training caches, not data guaranteed unseen during training.
- FP32 parameters, forward and HVPs, highest matmul precision; FP64 eigendecomposition and postprocessing. Native token-weighted cross entropy plus z-loss 1e-7, vanilla attention and unfused loss; weight decay excluded.
- Low, middle and high Gram modes require agreement below 2% at two adjacent finite-difference scales. The expanded sweep from 3 to 1e-6 resolved the middle checkpoint's initial validation failure. The two failed pilot attempts remain preserved.
- Two local tests cover native-loss equivalence, dense-Hessian agreement, finite differences, orthonormality and raw/ridged Kronecker actions. Repository lint checks passed.
- No training updates, fresh-Gram experiment, GGN substitution, or GPU allocations were performed.

## Results and reproduction

- **Early, counter 3,733:** [attention fit](hessian-early/attention-fit.json), [MLP fit](hessian-early/mlp-fit.json), [resolution checks](hessian-early/resolution-diagnostics.json), [artifact generations and checksums](hessian-early/artifact-manifest.json), [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-ed298f04d98a).
- **Middle, counter 18,665:** [attention fit](hessian-middle/attention-fit.json), [MLP fit](hessian-middle/mlp-fit.json), [resolution checks](hessian-middle/resolution-diagnostics.json), [artifact generations and checksums](hessian-middle/artifact-manifest.json), [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-035397d82a6b).
- **Late, counter 33,597:** [attention fit](hessian-late/attention-fit.json), [MLP fit](hessian-late/mlp-fit.json), [resolution checks](hessian-late/resolution-diagnostics.json), [artifact generations and checksums](hessian-late/artifact-manifest.json), [W&B](https://wandb.ai/marin-community/marin/runs/ema-gram-hessian-6537e4fd9860).

The results archive contains the six NPZ exports and the supplied, unchanged `fit_hessian_power.py`. Reproduce each fit with:

```bash
python -m pip install numpy
python analysis/fit_hessian_power.py hessian-middle/mlp.npz --output mlp-fit.json --bootstrap 1000
python analysis/hessian_resolution.py hessian-middle
```

Launch records bind exact source revisions and commands. Full Gram bases remain in GCS; the artifact manifests record their generations and CRC32C checksums. The four distributed hosts are one calculation, not four independent statistical samples.
