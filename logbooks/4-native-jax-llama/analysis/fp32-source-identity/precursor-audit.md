# FP32 standalone identity audit

Do not launch the existing common-raw-dQ v2 on TPU. Its exact scientific guards fail on the retained CPU fixture. I found no demonstrated mathematical reconstruction error; I found definite changes in executable boundaries and source representation. Those changes prevent treating the standalone values as normal full-graph values. Their precise causal contribution is unproven, and the failures do not prove that every possible identity-preserving diagnostic must fail.

This audit imported no JAX and performed no numerical execution, job query, RPC, storage/network access, source edit, commit, publication, catalog review, threshold change, or BF16 callback work. It inspected frozen source and retained outputs, verified their identities with Python standard-library code, and wrote only these audit files.

## Exact inputs and checks

The frozen preparation is `/private/tmp/native-jax-fp32-common-dq-preparation-v2`, manifest SHA256 `2f91c521b719d855acee45272489578d3ee48de574a34c94f398468cb3d495d3`. All 19 manifest payload hashes and sizes match. All 13 relevant source-file hashes match each actual retained source fixture: baseline `36e2a9a97fab22b76dfa5a03964c6600ec5713ac`, candidate `f1e991207ceda752957938b1cfe8893027d3fed8`.

The retained CPU controller summary is `/private/tmp/fp32-common-dq-v2-final-cpu-proof/controller-summary.json`, SHA256 `d057d9277ad84ce91e2606c7a3233b01d13707f07f357e733f8bfcdcb9e8082a`. Its mechanics pass is explicitly not scientific validity. Exact input paths, sizes and hashes, the two source-file verification maps, and extracted guard results are in the companion JSON. I also read the independent v2 review and query-scale experiment design named there.

## What actually fails

1. **Raw-kernel forward:** `fp32_common_dq.py:254–284` first runs the public kernel as a separately compiled primal, materializes `scaled_q = qr * alpha` in another compiled executable, then runs a different forward-plus-Q/K/V-VJP executable with that scaled buffer. The CPU source reference originally scales Q inside the contraction graph, after reshaping. The standalone raw path removes that source operation and feeds a materialized scaled input. These are equal mathematical expressions but distinct rounding/fusion and AD/output contexts. The retained forward differs in 190/1024 elements, max `2.384185791015625e-7`, relative L2 `3.044029006130773e-8`. Public rotary forward and inverse VJP are exact, so those controls do not explain this failure.

   The source-derived TPU function removes precisely one scale assignment; reinserting it reconstructs the original function AST exactly (independently rechecked). That proves the source edit boundary, not floating-point identity after moving multiplication into another executable. The public-kernel supporting HLO was hashed but not retained; the raw derivative HLO is complete. Consequently these artifacts cannot prove a specific optimized public-versus-raw CPU instruction change. CPU does not execute Splash or the candidate TPU barrier path.

2. **Source full versus replay full:** the source side calls the original last block from the original stacked model in a primal capture. `BlockReplay.functions/run` instead reconstructs a block from flattened canonical parameters inside a fresh forward-plus-all-master-VJP graph, uses explicit weight constraints, changes parameter signatures, and returns gradients as well as the primal. The canonical roundtrip preserves values, but not the original argument shapes or compilation context. Retained StableHLO makes this concrete: original stacked parameters have leading layer dimensions and `_take` extraction; replay parameters are flat state-dict leaves with import divides/reshapes/constraints and AD operations. The output differs in 460/1024 elements, max `7.152557373046875e-7`, relative L2 `7.539494840157132e-8`.

3. **Source residual versus replay prefix:** the original capture returns residual/tail/suffix derivatives together; replay reconstructs canonical parameters and returns prefix plus all-master derivatives. It differs in 214/1024 elements, max `2.384185791015625e-7`, relative L2 `3.8125120197281005e-8`. The source tail cotangent versus replay tail control is exact.

The source scan/ordinary hidden/loss/separate-full/composed-forward guards all pass exactly. Each replay variant's full/composed control also passes; its full and composed StableHLO files are byte-identical in both children. Thus the retained failure is specifically across source/replay compilation boundaries, not evidence that the block's full/composed algebra is wrong. Both children repeat the same failures, which establishes repeatability of an invalid common control rather than validity. No retained comparison isolates canonical import, AD, output set, or scale materialization as the sole cause.

The manual block decomposition also assumes no hybrid post-attention/post-MLP norms and no active stochastic effect. Those assumptions must remain guarded if this harness is reused. The actual pinned vanilla configuration does not establish support for other configurations.

## Smallest useful next control

A passing new control cannot be promised from read-only inspection. Do not repair v2 by weakening exact checks, bypassing them, or relabeling the standalone materialized buffer as a normal-source buffer.

First separate the changes rather than rebuilding another broad diagnostic. Preserve the already exact source-primal captures. Add a same-source value-plus-VJP control whose inputs are the original source-shaped stacked model leaves, x and cotangent, whose function is the original unmodified source last-block call, and whose mesh/mappings/precision are unchanged. Compare its primal exactly against the retained source-primal capture, and retain the complete stacked/master-gradient and physical records. Do not perform canonical deserialization or force flattened parameter specs in that control. If it fails, AD/output context alone already prevents this standalone attribution; retain that outcome and stop.

Only if that control passes should a second control introduce canonical import with a primal-only signature, followed by the same import with value-plus-VJP. This small matrix distinguishes representation/layout changes from AD/output changes. It does not automatically recover full-Trainer cotangents.

For raw extraction plumbing, compare the original public function and the exactly AST-reinserted source under the same value-plus-Q/K/V-VJP signature, using original unscaled Q and scaling at the original statement. Require exact primal and all Q/K/V VJPs. This is a source/recompilation identity check only; it intentionally does not yield a raw dQ observation. Exposing raw dQ afterward is another graph change and must separately preserve all original outputs/gradients. In particular a separately materialized scaled-Q buffer cannot inherit that identity proof.

If these exact controls fail, keep this standalone diagnostic abandoned as a full-graph explanation. Its two-order result can remain explicitly labelled standalone rounding sensitivity, with no causal or performance claim.

## Evidence needed for the actual numerical question

Use a pinned normal full Trainer/loss graph and retain complete HLO/StableHLO, runtime/source/import identities, exact initial weights/batch, all physical placements, original loss and every canonical gradient/Adam/update result. Any capture graph must first match its own uninstrumented implementation exactly in loss, every canonical and original stacked gradient leaf, and ordinary updated-state outputs before comparing implementations. Keep any failure.

Observe the FP32 X/G and scale/RoPE operands or trustworthy bounded hashes directly in that validated graph. A suffix cotangent from a standalone source tail is not a normal whole-Trainer cotangent. Raw dQ must be directly observed at the derivative boundary; do not divide a scaled gradient or infer it. Finally, only a separately pinned approved full-Trainer comparison tests whether the 20-loss trajectory improves. Static association and standalone sensitivity cannot establish a sufficient cause.

No new control was implemented or executed here. Root retains ownership of source changes, hardware decisions, existing callback work, publication and heartbeat monitoring.
