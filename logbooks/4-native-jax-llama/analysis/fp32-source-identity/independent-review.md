# Independent review of same-source FP32 identity control

**Zero review findings. The retained S1 exact-identity control failed and supports the bounded negative conclusion.** On this finite CPU fixture, canonical model reimport and added flattened-parameter constraints are not necessary to reproduce the forward discrepancy. This is not a claim of wrong mathematics, a particular compiler mechanism, TPU behavior or full-Trainer causality.

## Source, arguments and guards

All 24 manifest payload hashes/sizes match, totaling 2,158,477 bytes. All 79 protected old bundle/result files still match their recorded pre-execution hashes. Both pinned fixtures retain all thirteen scientific-source hashes. The new runner uses baseline source `36e2a9a97fab22b76dfa5a03964c6600ec5713ac`; candidate fixture preservation does not mean the new control was run on the candidate.

The runner extracts exact frozen AST bodies for scanned_capture, capture_last and isolated_full, removing decorators only for explicit compilation. Independent AST-hash checks agree. isolated_full calls the actual last source block obtained from the original stacked model. The value-plus-VJP wrapper differentiates that same function directly. There is no canonical model deserialization or forced flat weight-spec construction on this S1 path. The canonical digest helper is used only to verify initialization provenance, outside these scientific call arguments. Existing constraints introduced by the original source remain; the conclusion concerns the added flattened/canonical replay constraints.

The primal has thirteen argument records; the VJP has the identical thirteen-record prefix, including tree paths, shape/dtype/value hashes and complete recorded physical metadata, plus the retained source suffix cotangent. Initial model/batch hashes equal the prior fixture. All five source operand hashes and their full physical records match the prior run exactly. The full runtime snapshot also equals the prior CPU snapshot. Float32, highest precision, vanilla/reference attention, original two-layer dimensions and Auto data2/model2 mappings remain fixed. The output physical signatures match across the primal/VJP comparison. Recorded physical identity covers logical sharding/device placement, not unseen internal compiler buffer layouts.

The complete StableHLO entry points corroborate original stacked parameter shapes and layer extraction on both sides. The VJP adds the cotangent input, derivative/output context and original model-gradient outputs. Source-primal StableHLO is byte-identical to the prior source-full-separate StableHLO. This isolates the joint source/AD/executable-output-context comparison from canonical replay representation changes; it does not isolate AD alone or identify an optimized instruction.

## Actual failure and evidence

The isolated source primal equals the source scan exactly; their recorded value hash is `14403290e244cd65c870809c922d4d7156f0bfb407102a9454043cb8f5e01d79`. The source value-plus-VJP forward hash is `6ff1b3edfeb1c0cda0a9aa2aa1698b450a19ab613c2f90ecf5f873143d14de6c`, which also matches the earlier named canonical-replay full result. The full comparison record equals the earlier source-versus-replay discrepancy record:

- 460 of 1,024 forward elements differ.
- Maximum absolute difference: 7.152557373046875e-7.
- Relative L2: 7.539494840157132e-8.
- Worst index: `[3,2,22]`, values -2.3125529289245605 and -2.3125522136688232.

The runner executes compiled handles, synchronizes results, records complete graphs before execution, then raises on the real exact comparison failure. The execution receipt records exit 1; report fields remain completed:false, s1_exact:false and scientific_attribution_valid:false. No tolerance was weakened and no failed guard was bypassed. This genuine failure exercises the negative gate.

Twelve original model-gradient leaves and the input-gradient record are retained, including shape/dtype, value hashes, finite statistics and full physical signatures. The twelve comprise nine layer-stacked leaves plus final norm, embeddings and lm-head leaves; their retention is not an independent gradient-correctness proof. The final run retains eight complete StableHLO/optimized-HLO files totaling 714,170 bytes, all size/hash/non-truncation checks verified. No follow-up canonical or raw control exists on the runner's continuation path or final retained control list; S1 failure stops that path.

The first attempt's scalar-recording failure, source, partial complete graphs, log/report and receipt remain separately preserved. The source diff changes only record(): it replaces the inherited scalar-incompatible ndarray.view(uint8) digest with the same shape/dtype-prefixed array.tobytes digest and finite host statistics. Independent AST comparison after removing that helper body shows no other source change. This is a recording-only repair, not an altered scientific function or relaxed identity check.

## Interpretation and limits

The supported conclusion is that this CPU mismatch occurs with the original source-shaped model and unmodified block call, without canonical reimport or added flattened-parameter constraints. Therefore those representation changes are not necessary for this particular observed discrepancy. It does not show that canonical reimport has no further effect, prove a specific fusion/optimizer mechanism or wrong math, or establish identity of gradients, updated state, raw dQ, normal full-Trainer cotangents, TPU/Splash behavior, trajectory, performance or adoption. The fixed cotangent is from the source suffix, not an observed whole-Trainer derivative boundary.

The source audit's conditional sequence was followed: S1 failed, so later canonical/raw attribution controls were not pursued. Preserve this negative result and stop standalone full-graph attribution. A future observation method must first preserve ordinary loss, every relevant original/canonical gradient and state output in its own validated normal graph. This review does not imply every possible identity-preserving observation must fail and grants no new hardware, job, archival or publication authority. Root owns all jobs and subsequent decisions.

This review used static source/AST, logs and saved reports plus standard-library mechanical checks only. It did not run numerical/JAX code or independently recompute array differences from retained arrays; arrays were not retained. The recorded outcome is corroborated by exact source/runtime/argument/physical evidence, complete graphs and agreement with the old output hash and comparison record.

## Exact identities

- Handoff SHA256: `54623e7c97a02ad3b2891dc0264fa49db78f5d1686efb349c460bfe66f47761d`.
- Manifest SHA256: `ac936f5607989c61a644606a71d21f1e07f7a159867cebac4d18843b2ffc4257`.
- Source SHA256: `3e20ad352d683a0ee9d877afc25e88035ed2c91a0f4cc7188385bc9b16a1831a`.
- Config SHA256: `c9cc6810b4b0425dbf6608f0b2c0e70298cd11419634dbf4b90f7f26e149fb57`.
- Final report SHA256: `3cb61a80c929c6990d6221101f36bf988ab64b4c7b6ecb4bcea37de7f839132d`.
- Final execution receipt SHA256: `2cc80da597c55393216c9d8be7f7372f2f3f7b66e0db8a615bc487c628716369`.

Only this report and its JSON companion were written. No source edits, numerical execution, tests, thresholds, network, client/storage/job operations, publication or subagents were used.
