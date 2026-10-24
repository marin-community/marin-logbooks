# Local embedding VJP preparation self-review

Prepared locally on clean source8aa; TPU execution, network, submission, archive, PR and push remain unperformed. Root and fresh review are pending.

The first observed run is preserved under `/private/tmp/native-jax-embedding-vjp-preparation-first-observation` with its six HLO files in `/private/tmp/native-jax-embedding-vjp-cpu-mechanics-1`. It failed an overstrict harness criterion after HLO writes and before JSON; its exact numerical metrics were not retained. Subsequent runs preserve metrics for the identical scientific-function ASTs/config/fixture. Only report/mechanics classification, numerical exit and root-authorized no-Git provenance were corrected.

Final observed small CPU proof: `/private/tmp/native-jax-embedding-vjp-cpu-mechanics-final/report.json`. Dimensions V64/D32/B4/S8. CPU mechanics passes while comparison_valid=false. Forward outputs are exact. BF16 weight VJP differs in 8/2048 coordinates: max abs0.0078125, mean abs2.002716064453125e-05, relative L20.0005118898582032608 under unchanged atol1e-5/rtol1e-4. FP32 scatter equals NumPy add.at for this fixture; both are higher-precision diagnostics, not exact mathematical oracles. Legacy/native vs FP32 max abs is0.015625. Shared operand bytes, global/local devices, shards, dtype/spec/memory and device-local layouts match except intended Auto/Explicit axis type.

The full main V65536/D2048/B8/S2048 forward/VJP shapes and BF16compute/FP32gradient dtypes were traced abstractly without model-sized arrays, lowering, compilation or execution. Legacy Auto eval_shape does not infer physical output sharding; that remains an actual compiled-runtime check. The original seed42 benchmark token/loss-weight digest was independently reproduced using only131072 bytes of token/loss-weight arrays, with no arrays saved.

The final runner passed an exact no-.git snapshot abstract regression, recording git_metadata_available=false and runtime_git_commit=null, plus wrong-pin/wrong-source-hash negatives before JAX import. All10 declared scientific module imports resolve to the guarded source snapshot and have exact hashes.

The numerical source/fixture/metrics/abstract/physical function ASTs are identical to first observation. See `self-review.json` for exact hashes, complete metrics, limits, correction record and proof inventory. Scientific runner/config bytes are ready for independent review; no claim of full-model localization or TPU order is supported.

Scope is lookup transpose only, excluding optional input-embedding normalization. Original full-Trainer provenance explicitly declares input_embedding_norm=false, matching the frozen legacy norm=None constructor and native fragment stop point. See input-norm-scope-proof.json. cpu-order-signature.json traces actual CPU gradient entry operands and BF16-rounding wrappers; it is not typed TPU-HLO reproduction.
