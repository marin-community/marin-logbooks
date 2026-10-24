# Query-scale barrier A/B audit

Source `36e2a9a97fab22b76dfa5a03964c6600ec5713ac` versus `8aa8674d10894728ed0b455ae14e14f46457cf9f`, one physical v5p-8 slice, actual1.53B layer27 operands, BF16 compute/F32 master weights, identical shared operands/cotangents and data2/model2 placements. Only the attention core file differs. Both actual child imports, file SHAs, barrier code presence, initial/batch hashes and frozen source boundary guards pass. Every recorded artifact size/SHA was independently checked.

The candidate makes prefix input VJP, Q-master gradient and inputnorm-master gradient **bit-exact** to named. Full/composed show the same improvement plus removal of the47-element V-master difference. All44 named and44 highest-FP32 result hashes and all4 native forward hashes remain unchanged. No newly nonexact output is observed.

|Output|Baseline full relative L2|Candidate full relative L2|
|---|---:|---:|
|Input VJP|0.000588515|0|
|Q-master gradient|0.004576706|0|
|Inputnorm-master gradient|0.005366241|0|
|V-master gradient|0.0000342844|0|
|K-master gradient|0.0000362533|0.0000362533|
|MLP down-master gradient|0.0000361012|0.0000361012|

The prefix inputnorm baseline is0.005653452; it also becomes zero. Tail stays exact for all11 outputs. Full/composed retain unchanged48-element K and680-element MLP-down differences.

Optimized native prefix HLO changes from rounded `scale*cos/sin` coefficients (baseline245–266) applied directly to raw Splash dQ to an explicit BF16 `dQ*scale` fusion (candidate552–559, call1077) feeding separate inverse-RoPE products (1079/1085). The same boundary appears in full and composed. This supports the intended rounding-order explanation for the removed differences; common raw-dQ control was not performed, so this is not final operator-level causality proof.

This result supports proceeding to an independently matched full Trainer test. It does **not** establish full Trainer backward identity, trajectory acceptance, performance or convergence. Named-source incoming cotangents and standalone graph/fusion limitations remain; staged comparisons also hold named intermediates fixed. No production source or acceptance threshold was changed during this audit.
