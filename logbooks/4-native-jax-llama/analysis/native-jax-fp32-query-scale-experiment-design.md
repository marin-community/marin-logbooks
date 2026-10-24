# FP32 query-scale boundary experiment

The prepared local candidate is `f1e991207ceda752957938b1cfe8893027d3fed8`, based on `36e2a9a97fab22b76dfa5a03964c6600ec5713ac`. It adds an optimization barrier before native Splash query scaling only for FP32. BF16 remains at the baseline source behavior. No mathematics, tolerance, learning rate, attention backend, or model configuration changes.

The independent static audit of the actual normal FP32 Trainer HLO shows a concrete association difference: native materializes F32 `scale*cos` and `scale*sin`, then multiplies raw Splash dQ; named materializes F32 `dQ*scale`, then applies inverse RoPE. Both use the same FP32 scale `0.0883883461`. This is a testable explanation for some gradient nonidentity. Its effect on the failed 20-loss trajectory is unproven.

## Next bounded numerical gate

Prepare one independently reviewed standalone FP32 diagnostic before considering another full Trainer comparison. It should contain two complementary controls, with all arrays retained on the TPU host and only bounded JSON/HLO summaries archived.

1. **Actual common source-prefix A/B.** Reconstruct the pinned 1.53B initial model and batch using the existing benchmark initialization/import path. FP32 masters and compute, highest matmul precision, layer 27, B8/S2048/D2048/MLP5632/H16/KV4/head128/V65536, data2/model2. Capture actual FP32 source hidden values and incoming cotangent through a separately compiled source full block, checking the normal-source scan/hidden/loss and full/composed guards before attributing any result. Reuse frozen AST-proven source extraction/canonical import machinery, but create a new explicitly FP32 driver; the BF16 capture's five operand hashes cannot be reused or relabeled as FP32 normal-source hashes. The previously retained highest-FP32 control casts BF16-source operands/cotangents and is a distinct corroborating control.
2. **Same raw-dQ/trig two-order control.** From that common FP32 prefix, materialize one actual projected Q/K/V boundary and the context cotangent. Capture raw Splash dQ once for both order calculations. The source implementation makes this accessible: `_tpu_splash_attention` applies query scaling before calling its shard-map `wrap`, and `wrap` itself has no query scale. A scratch source-derived function can remove exactly the single `q_ = q_ * (1.0 / math.sqrt(D))` assignment from the clean baseline function, accept the already-scaled query as its input, retain every kernel/mask/sharding operation, and take a VJP with respect to that scaled input. Retain the source SHA, original and transformed AST hashes, and an exact one-node AST-edit assertion. No production monkeypatch, private kernel rewrite, division of an already-scaled gradient, or inferred dQ is acceptable.

The second control intentionally materializes a new standalone graph. It captures the actual kernel derivative for the fixed source inputs, not a promise that this kernel sees bit-identical hidden runtime operands in the uninstrumented full Trainer.

For common trig values, source-extract the existing `apply_rotary` coefficients in a separate scratch helper whose only AST change is returning its computed `cos, sin`. Keep native compile-time inverse frequencies, runtime positions, FP32 dtype, and physical partitioning. Guard source Q/RoPE forward and VJP against the existing public `apply_rotary` before using the helper. Capture these coefficients once; neither rounding order may recompute a different trig function.

## Two explicit rounding orders

Let `g` be the one materialized raw Splash dQ and let `c/s` be the one materialized FP32 cos/sin pair. Use the identical FP32 scalar `a = float32(1/sqrt(128))`.

- Scale-first: materialize `u = F32(g*a)` in its own compiled multiplication. Feed `u,c,s` to the inverse-RoPE calculation.
- Coefficient-first: materialize `c_a = F32(c*a)` and `s_a = F32(s*a)` in their own compiled multiplication. Feed `g,c_a,s_a` to the same inverse-RoPE calculation.

The shared inverse calculation is `g*c + rotate_transpose(g*s)` using the existing split-half convention. Check its ordinary unscaled result against a VJP of public `apply_rotary` on the same fixed coefficients/source positions before interpreting the two-order comparison. The standalone products must take runtime array operands and return materialized buffers; inspect optimized HLO to verify no compiler pass re-associates away the intended boundaries. Record whether final product/add fusions differ; do not assume optimized HLO disassembles every eventual floating-point instruction.

Report post-RoPE dQ max/mean absolute error, relative L2, different-element count, finite values, and byte hashes. Reuse the actual common Q-projection/inputnorm VJP with each post-RoPE cotangent to measure the downstream input/master-gradient effect. Hold every other cotangent contribution fixed. The plain two-order algebra is scratch diagnostic code, not a new maintained kernel or a checked-in test oracle.

## Guards and measurements

- Pin baseline and candidate imports in fresh children. Check every relevant source-file SHA, loaded attention-core path, actual static dtype predicate, no stale bytecode, and clean restoration of any disposable source-swap fixture.
- Check initial model and batch hashes against the retained large-source pins. Record new actual FP32 hidden/Q/K/V/cotangent/trig/raw-dQ hashes without pretending those values were retained by the earlier normal report.
- Compare source full/scan/hidden/loss controls independently. If a source guard fails, retain the failure and stop attribution at that boundary.
- Assert common argument shapes, dtypes, global shapes, partition specs, concrete device set, memory kind, local shard indices, and finite values. Auto/Explicit axis types may differ only as the intended variant mode.
- Preserve all named forward/VJP hashes and check all native forward hashes. Check all reported input and master VJPs; improving Q while introducing another difference is not a success claim.
- Save complete bounded HLO/StableHLO files with size and checksum, rejecting rather than truncating. Keep arrays/weights/checkpoints on the TPU host. A target cap of 64 MiB/128 diagnostic files is sufficient for planning but must be proven against the final artifact fixture.
- Preserve `--xla_allow_excess_precision=false`, highest matmul precision, FP32 masters/compute, model, optimizer, and approved criteria. Do not search seeds, rescale cotangents, change LR, or tighten/relax a threshold to manufacture a failing or passing test.

## Decision before a full Trainer run

Proceed only after independent review confirms that the source/import/operand/placement controls pass; the same-raw-dQ two-order control quantifies the expected FP32 discrepancy; candidate optimized HLO preserves the scaled-dQ boundary; and the common-source FP32 prefix has targeted input/Q/inputnorm discrepancies reduced without new mismatches. Record a null result if the changed HLO does not improve the numerical boundary. A small below-tolerance discrepancy is still valid localization evidence; it is not a strict gradient-gate failure.

Then a separately pinned full Trainer comparison may test whether the approved 20-loss-vector gate improves. Keep all existing normal-state/diagnostic checks and five-warmup/20-timed-step settings. Timings must use the normal no-hook executable, with profile capture after timing if requested. A numerically invalid case keeps null accepted speed ratios. An unchanged or worse full trajectory falsifies the sufficient-cause interpretation, even if the local gradient boundary improves. Full performance, BF16 integration, entry-point replacement, legacy deletion, and convergence remain separate gates.

No hardware diagnostic, RPC, job observation, submission, remote publication, or automation change was performed while preparing this candidate/design. Archive the source and this preparation in the research record before a future authorized hardware execution.
