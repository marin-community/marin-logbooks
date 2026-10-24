# FP32 common-raw-dQ control: prepared, not hardware-tested

This bundle prepares one scientific diagnostic for baseline `36e2a9a97fab22b76dfa5a03964c6600ec5713ac` versus isolated FP32 query-scale candidate `f1e991207ceda752957938b1cfe8893027d3fed8`. It does not submit jobs, change a production module, publish source, or establish numerical improvement. The candidate worktree remained clean at its original head.

The controller creates two fresh disposable source trees using `git archive` of those exact pins, verifies thirteen relevant source-file hashes and actual import paths, checks the loaded barrier variant, and starts fresh sequential Python children with fresh bytecode locations. No production monkeypatch or source substitution is used. The source repository must contain both pins. The scientific script/helper bytes are frozen by `manifest.json`; the proof-input manifest records the exact files used during validation.

## Scientific boundaries

The derived capture runner uses actual FP32 initial model values and FP32 incoming cotangents at highest matmul precision. It never casts the retained BF16 source operands to claim FP32 provenance. Large mode uses the existing 1,531,037,696-parameter configuration, layer27, B8/S2048/H16/KV4/head128/V65536, data2/model2, identical source initialization and batch. It requires the retained initial parameter hash `d79fb8c383f9af209fd9e3e32e00798c4aeb460d779a1444d9ea20ba9f453f9a` and batch hash `d65acfa301bad7db399ff9df86cbbc150c649317dd357b3a26c53c9eaea9caa4`. Source scan, normal hidden/loss, isolated last-block and composed-forward controls precede attribution. The separate-prefix/full/composed VJPs preserve source masks, modules and canonical imports. All named results and native forward hashes must remain unchanged across the A/B; introduced numerical mismatches are reported and fail controller guards.

`fp32_common_dq.py` makes three audited scratch extractions:

- The baseline `_tpu_splash_attention` AST loses exactly its single `q_ = q_ * (1.0 / math.sqrt(D))` assignment. Reinserting that node must reconstruct the original AST exactly. A VJP with respect to an already-scaled query obtains one raw kernel dQ, dK and dV, with no division of a previously scaled gradient.
- Public `apply_rotary` keeps its complete computation and replaces only its final return with `(cos, sin)`. Runtime trig is captured once, retaining the compile-time inverse frequencies and runtime positions. Public rotary forward and inverse VJP controls must be numerically exact.
- The actual native attention method retains its exact statement prefix immediately before the first public Q RoPE call and returns its projected q/k/v. The actual inputnorm and canonical FP32 masters remain in the downstream VJP.

Both rounding orders use the same one materialized raw-dQ buffer and one materialized cos/sin pair. Scale-first separately materializes `F32(dQ * alpha)`; coefficient-first separately materializes `F32(cos * alpha)` and `F32(sin * alpha)`. Both call the same split-half inverse calculation. The products take runtime array arguments; optimized HLO must contain multiplication and runtime parameters with no trig recomputation or dot. Selected scientific graphs have complete HLO and StableHLO files; supporting public-API control graphs retain size/hash metadata only. No saved HLO is truncated. Final graph HLO cannot prove the exact device instruction sequence or sufficient cause of an entire training trajectory.

Downstream results are the actual joint Q/K/V-projection plus inputnorm VJP, changing only the post-RoPE Q cotangent and holding dK/dV fixed. These arrays exclude the common residual, output-projection-weight and MLP contributions. They quantify the rounding differential, not an independently verified whole-prefix or full-Trainer gradient. Source masks, canonical masters, all other cotangents and physical partitioning remain fixed.

Every boundary/result records shape, dtype, byte hash, finite statistics, concrete device set, memory kind, partition spec and local shard indices. Difference fields use float64 host arithmetic for reporting; `exact` uses numerical equality and does not alone prove signed-zero identity. Separate byte hashes preserve that distinction.

## Validated scope and preserved failures

Final CPU4 mechanics proof: `/private/tmp/fp32-common-dq-final-cpu-proof/controller-summary.json`. Both children completed and all sixteen cross-child guards passed. Public rotary forward and inverse VJP agree exactly with the common-trig implementation. Both materialized-product optimized-HLO guards pass. The CPU fixture's two rounding orders differ by max `1.862645149230957e-9`, relative L2 `6.02942464373926e-8`, 299/1024 elements.

This is deliberately **not a scientific attribution pass**. Separately materialized scaled Q changes the CPU reference kernel forward by max `2.384185791015625e-7`, relative L2 `3.044029006130773e-8`. The named source-versus-canonical-replay full forward also differs by max `7.152557373046875e-7`, relative L2 `7.539494840157132e-8`. Both failed exact guards are preserved; CPU-small mode may finish bookkeeping and report mechanics with `scientific_attribution_valid: false`. No numerical tolerance was introduced. Non-small execution retains fatal attribution guards. The CPU reference path does not execute the TPU candidate path or prove its numerical benefit.

Final actual-large CPU abstract proof: `/private/tmp/fp32-common-dq-final-abstract-proof/controller-summary.json`. Both variants confirm the 1.53B model and trace actual FP32 B8/S2048/H16/KV4/D128 Splash primal/three-VJP graphs without compiling or executing a TPU kernel. Public rotary+Splash derivative graphs have zero barriers at baseline and two for the candidate. The common source-derived raw-kernel graph has zero barriers for both. This is graph-construction evidence only.

The final CPU fixture contains 115 diagnostic files and fits the configured 128-file/64MiB ceiling. The actual TPU artifact size is not yet measured; all writers and the controller reject excess size or file counts rather than truncate evidence. Arrays, checkpoints, weights and profiles stay out of artifacts. Diagnostic stdout is not part of the scientific archive.

## Reproduction after independent review

Set `PYTHON` to the validated workspace Python with the pinned workspace dependencies available. On CPU, set `JAX_PLATFORMS=cpu`, `JAX_NUM_CPU_DEVICES=4`, and `XLA_FLAGS=--xla_allow_excess_precision=false`. Run from the bundle directory:

```sh
"$PYTHON" fp32_common_dq_controller.py \
  --repository /path/to/repository-containing-both-pins \
  --fixture-root /fresh/disposable/source-fixtures \
  --output-dir /fresh/diagnostic-output \
  --expected-backend cpu --small
```

Use separate fresh directories and replace `--small` with `--abstract-only` for the actual-large CPU abstract proof. Large runtime mode omits both switches and requires exactly one process/four v5p devices. This README provides no submission command or permission to launch. The isolated candidate, frozen runner, source pins and reviewer-approved scientific design must first be preserved in the durable research archive. The parent owns any later hardware authorization and babysitting.

Before a full Trainer run, an independent reviewer must verify the actual TPU source/operand/physical-placement controls, the same-raw-dQ measurement, candidate optimized-HLO scale boundary, targeted prefix/inputnorm/Q improvement and absence of new mismatches. Null improvement or another exact source-control failure is valid negative evidence and must stop the sufficient-cause claim. Full trajectory, accepted performance, BF16 integration, entrypoint replacement, legacy deletion and convergence remain separate gates.
