# Layer-0 o_proj static localization

The remaining BF16 first-gradient failure has a concrete projection-local compiler difference worth testing: the legacy VJP keeps head/head-size axes while native flattens them. This is a static hypothesis, not a measured mechanism. Both full-Trainer results remain rejected.

The failing diagnostic tensor is `model.layers.0.self_attn.o_proj.weight`, unchanged from8aa: max error1.52587890625e-5, mean6.926814677399307e-11, relative L2 3.578135510906577e-5. The retained HLO is the separate normal no-hook executable, so it cannot establish that the failing diagnostic executable uses identical local rounding.

| Path | Legacy | Native |
|---|---|---|
| Source | Named o_proj contracts heads and head_size | Reshape heads*128, then `...i,oi->...o` |
| StableHLO weight VJP | `[8,2048,2048] × [8,16,2048,128]`, contract `[0,1]×[0,2]` → BF16 `[2048,16,128]` | `[8,2048,2048] × [8,2048,2048]`, contract `[0,1]×[0,1]` → BF16 `[2048,2048]` |
| Optimized local kernel | conv137, `size=8x4 pad=7_7x0_0 rhs_reversal=1x0`, `[2048,8,128]` | conv124, `size=4`, `[2048,1024]` |
| Collective | BF16 add over `{{0,2},{1,3}}`, slice1024 output rows | Same dtype/groups/output partition |

Exact retained locations: legacy StableHLO1782/1787 and optimized3920–3945; native StableHLO3086–3098 and3230–3234, optimized3232–3257. JSON retains exact source operations, file hashes and line numbers. These shared reverse-loop body operations apply all28layers; the HLO is not a uniquely unrolled layer0 kernel. The selected optimized contraction signatures already exist in8aa.

Different local axis views select different typed convolution layouts/window forms before BF16 output and BF16 data reduction. That supports checking local contraction grouping/rounding, but HLO does not reveal every internal accumulator or emitted microkernel order. The equally unresolved alternative is that actual attention result X or incoming reverse cotangent G already differs. No operand arrays/hashes were retained, and matching surrounding parity summaries cannot prove operand identity.

The archived attention-order patch (`805b6689…`, base55ce8053) altered **reference attention** Q/K/V score/context dimension order. The Q-only patch (`a1a79424…`, basea3d7f220) articulated **q_proj forward** compute weights. Both retained FP32 failures on the older L8/D1024/S512 fixture and were rejected. Neither is an exact-common-operand Splash o_proj weight-VJP control. This plan does not revive either patch.

The smallest valid future diagnostic is one initial full-size BF16 diagnostic step per implementation, preserving the exact2b17 fixture. First capture the unmodified diagnostic all255 raw gradients/HLO. Tap layer0 actual BF16 X and G in an instrumented full graph and require its loss/all255 raw gradient bytes and failing coordinates to reproduce the untouched diagnostic. Compare canonical transpose/reshape-only global/per-shard operand bytes, dtype, layout and cast provenance. If X or G differs, inspect only its immediate producer; no local-kernel causal claim is valid.

Only after that identity gate, compare both projection VJP axis views on the **same actual X/G pair**, as shadow branches inside the same full-graph diagnostic context. Each branch must exactly reproduce its original local/after-collective result before the cross-view comparison is accepted; extra outputs can change compilation. Pre-collective disagreement narrows to local contraction, post-collective-only disagreement to collective/layout effects. Failed identity controls are inconclusive. Standalone FP32 or source-prefix synthetic controls that failed identity remain invalid.

No runner, patch, source edit, experiment, JAX import/lowering/compile, workload/test/job, client/network/download or catalog change was made. Future diagnostic execution requires separate authorization. Diagnostic localization would not by itself establish normal no-hook behavior or explain the20-step trajectory.
