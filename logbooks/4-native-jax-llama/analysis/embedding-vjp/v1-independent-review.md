# Independent embedding VJP preparation review

One blocking finding: preserve structured evidence for unexpected nonfinite numerical outcomes before exit. The scientific design is runnable as a bounded hypothesis test, but this frozen harness is not ready for TPU dispatch under its stated retention contract.

`runner.py:311-312` raises on NaN/Inf before recording the variant at315 or writing `report.json` at337. `metrics()` likewise raises at159-160 before the report. Finite BF16 parity failures are handled correctly: report retained, comparison_valid=false, TPU exit1. Pure lookup on this fixed finite fixture makes arithmetic overflow implausible, but an unexpected numerical outcome would leave HLO/logs without the required structured source/config/input/device/partial-result record. Correct only this bounded harness path in a separately versioned freeze; keep scientific ASTs/config/fixture/thresholds unchanged. This finding does not require a generic import/lowering exception framework. Current frozen artifacts were untouched.

Reviewed hashes:

- final-freeze.json: `6c4316efa090c133cf05c1070ac357be8823c18b866e8ed9925aceb540ac8f26`
- runner.py: `b62bfe4d2b2508f6c5f742ffd3aa4766f3a8ff82f8e4a6d4c3095948a3ae0110`
- config.json: `9aa91221a40cda48a7154632f47908556de446c524a464a73ab208aa54508634`
- manifest.json: `de8ef8ab780672fd1916b08e165153eadb240f56e33bbfc682dc1cc880d22083`

All10 frozen files, all10 scientific source files, AST/statement/assignment guards, benchmark helper/token-generator ASTs and every listed proof artifact match. Original input_embedding_norm=false is independently verified in hashed compare.json. Legacy invokes production LlamaEmbedding.embed with norm=None and standard Haliax embedding. Native is explicitly the exact three guarded lookup assignments plus return, not a nonexistent public native embed method or full activations invocation. Both differentiate FP32 masters through production BF16 compute casts and restore FP32 storage gradients.

The seed42 benchmark token generator is seed44, and the retained original batch digest is reproduced in the proof. Both paths share one synthetic seed20261023 unit-RMS BF16 cotangent, deterministic synthetic weights, and exact bytes and physical placements apart from Auto/Explicit axes. No absent original full-model cotangent is claimed. Runtime version/flag/import/source checks are concrete; absent runtime Git is honestly false/null, while clean exact local submission identity remains root's separate verification responsibility.

Retained small CPU evidence (V64/D32/B4/S8) has exact forward outputs and rejected BF16 VJP parity:8/2048 coordinates, max abs0.0078125, mean abs2.002716064453125e-05, relative L20.0005118898582032608 under unchanged atol1e-5/rtol1e-4. FP32 scatter equals NumPy add.at for this local fixture; neither is an exact mathematical oracle. This numerical report used runner4bd38d; scientific functions/config/fixture ASTs match final b62bfe. Final b62bfe has an actual no-Git full-size abstract proof with all10 import paths/hashes verified. Main V65536/D2048/B8/S2048 tracing allocated no full-size arrays and did not lower, compile or execute them; Auto output sharding remains a runtime gate.

Actual retained CPU entry gradient dataflow shows native local-batch scatter then cross-data reduction and feature slice, versus legacy cotangent feature all-to-all plus gathered full-batch tokens then feature-shard scatter. It uses legalized F32 operations with explicit BF16 round-trips. It is not typed TPU-HLO reproduction. The original normal no-hook HLO and hook-enabled comparison are separate executables; original cotangents and diagnostic optimized HLO are absent. TPU minimal entrypoint HLO must receive an independent static review and non-reproduction must remain an allowed result. No full-model cause, numerical pass, adoption, or speed ratio follows.

The first preparation failure and all six HLOs remain preserved without changing fixture/scientific functions/config/thresholds. Its initial exact numerical metrics were not saved and are not inferred. Output policy remains bounded complete HLO plus JSON only; no arrays/checkpoints/profiles or transfer logic. Import/lowering failures have no explicit all-failure reporting promise; no broader defensive framework is requested.

This review used local stdlib reads, hash/AST inspection and review-file writes only. No hardware/JAX execution, Git command, network/remote query, archive/catalog action, production edit or frozen-artifact edit occurred. Before TPU: resolve R1 in a new freeze, review that delta, and complete root's exact clean-source and durable-archive gates.
