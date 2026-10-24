# Independent review: FP32 common raw-dQ preparation

**Verdict: changes required before hardware.** The central source derivation and cotangent mathematics are sound within the stated standalone scope. Two guard-enforcement gaps need a new preparation revision. The existing CPU/abstract proofs establish mechanics and graph construction; scientific attribution remains false. This review authorizes no launch and supports no full-Trainer, speed, trajectory or convergence claim.

## Verified frozen evidence

I independently checked all 13 manifest payload files (201,019 bytes), their sizes and SHA256s. The manifest SHA256 is `6b98d8ed07bee5695025594f35352be5c1656ed7af351d1ab1c32752dba73c16`. Applying every guarded edit in `derivation.json` reconstructs the FP32 capture script byte for byte. Both sets of 13 source hashes match the exact local Git objects, and the retained baseline core is byte-identical to its baseline object. The candidate worktree is clean. Its complete relevant diff adds only the FP32 dtype-guarded optimization barrier immediately before query scaling; BF16 source behavior is unchanged.

Both proof summaries and every listed artifact hash/size match. The CPU summary SHA256 is `da084a0c3caa1132e577812ce62ddaae2a4a5507d17e546a6508f1960161deb6`; the abstract summary SHA256 is `a5c84b9e05d600488f9b7ae369e572f7a61e693a51d14e1904f74746d2beb68a`. Both point to the byte-identical frozen validation input manifest. CPU has 115 files / 7,507,872 bytes; abstract has 5 files / 42,724 bytes. The parent reports durable archive commit `944004a3b5157ff618012a682953f3a7a3016400`; I did not query that remote.

## Must fix before hardware

1. **[P1] Assert complete physical placement, rather than labeling partial checks exact.** `native_jax_fp32_common_capture.py:258-268` compares only byte hash, shape, dtype and partition spec, then writes `common_actual_fp32_operands_masters_and_physical_specs_exact: true`. The recorded `physical` objects contain memory kind, device set and shard indices, but these are not compared. Source records at lines 84-97 also lack a complete normalized physical signature in provenance, and controller lines 196-219 omit source/stage placement from A/B guards. The raw helper's A/B physical comparisons cover its own records, not the entire source/stage path. A partition spec cannot by itself prove the device mapping. Record and fatally assert complete source/common/stage argument and cotangent placement, allowing only the intended Auto versus Explicit axis-type difference. The inspected CPU named/native signatures agree in all four stages; this is an enforcement defect, not observed placement drift.

2. **[P1] Make failed non-small cross-child guards terminate the controller unsuccessfully.** `fp32_common_dq_controller.py:220-253` writes `passed: false` and `scientific_attribution_valid: false` when an A/B invariant fails, but then returns successfully and prints `completed: true`. For example, candidate native forward drift may leave each child internally consistent and only fail the cross-child forward hash guard. Preserve the failure summary, then raise/exit nonzero. All sixteen current CPU guards pass, so the proof has not exercised this path. The child source/raw/replay identity guards themselves already fail fatally in non-small mode.

Make these changes in a new revision; retain the frozen bundle and its negative evidence. Verify guard rejection locally using changed metadata/guard fixtures, then freeze and archive new hashes. No data, seed, precision or tolerance changes are warranted.

## Scientific/code conclusions

- **Actual FP32 provenance:** capture lines 309-365 initialize the pinned benchmark master model and run its FP32 source scan/last-block/suffix path at highest precision. FP32 hidden values and incoming cotangents are computed directly. Historical BF16 operand hashes are removed rather than reused. Dtypes, nonzero source cotangents, normal hidden/loss and source scan/full/composed controls are checked; their failures stop the child.
- **One raw kernel dQ:** `fp32_common_dq.py:50-68` removes exactly `q_ = q_ * (1.0 / math.sqrt(D))` from the clean baseline Splash AST. Exact reinsertion reconstructs the original AST, and the baseline must have no barrier. Independent AST reconstruction matches all recorded original/transformed hashes. Lines 274-289 materialize scaled Q and differentiate the derived function with respect to that already-scaled query, obtaining dQ/dK/dV once. There is no division of an already-scaled gradient.
- **One common trig pair and correct adjoint:** only `apply_rotary`'s final return changes to `(cos, sin)`. Compile-time inverse frequencies and runtime positions remain. Both orders use one materialized pair. For the split-half rotation `R([a,b])=[-b,a]`, its transpose is `[b,-a]`, so `g*cos + R_transpose(g*sin)` is correct. Public forward and inverse-VJP controls are exact in the CPU fixture.
- **Explicit orders:** scaled dQ and scaled trig are outputs of separate compiled, blocked multiplications with runtime arguments. Both downstream inverse calculations share the same expression. CPU optimized product HLO guards pass. Actual TPU optimized HLO and device instruction behavior remain unknown.
- **Actual downstream modules:** the extracted projection retains the actual native attention statement prefix before Q RoPE, including source Q/K/V projections/reshards. The joint downstream VJP includes actual inputnorm and canonical FP32 masters, changes only the post-RoPE Q cotangent, and holds K/V cotangents fixed. Its arrays exclude residual, output-projection-weight and MLP contributions, as disclosed.
- **Controls and evidence:** source capture controls, public RoPE controls, raw kernel identity and source/replay/full-composed identities fail fatally in non-small children. Selected scientific HLO is complete; overflow rejects rather than truncates. No actual TPU artifact size or performance result exists.

## Preserved negative evidence

Both CPU children have the same failed exact identities:

| Control | Max absolute difference | Relative L2 | Different elements |
| --- | ---: | ---: | ---: |
| Public versus separately scaled-Q raw kernel forward | 2.384185791015625e-7 | 3.044029006130773e-8 | 190/1024 |
| Source output versus named full replay | 7.152557373046875e-7 | 7.539494840157132e-8 | 460/1024 |
| Source residual versus named prefix replay | 2.384185791015625e-7 | 3.8125120197281005e-8 | 214/1024 |

The third row is an additional preserved CPU source/replay failure. Mechanics completion does not erase it. The two-order post-RoPE difference is max `1.862645149230957e-9`, relative L2 `6.02942464373926e-8`, with 299/1024 differing elements; it is not a validated attribution result because source/kernel controls fail. Numerical `exact` permits signed-zero differences; separate hashes retain byte identity information.

The large abstract proof confirms the 1,531,037,696-parameter shapes and actual B8/S2048/H16/KV4/D128 Splash primal/three-VJP construction. Public rotary+Splash graphs contain zero baseline/two candidate barriers; the source-derived common raw graph has zero in both. This is CPU abstract tracing with no TPU compilation or execution.

## Causal limits and root gates

Materializing Q, trig, context and cotangents changes compilation context. These operands are source-derived, but the helper does not establish their hidden bit identity inside an uninstrumented full Trainer. The incoming cotangent is from the named source suffix/tail, not an independently native incoming cotangent. Downstream arrays measure the QKV/inputnorm order differential, not complete absolute model gradients.

A nonzero order difference proves sensitivity to those two orders on one materialized buffer. It does not independently establish that the actual baseline equals coefficient-first or the actual candidate equals scale-first inside the compiled block. Nor does abstract barrier presence establish the optimized TPU scaled-dQ boundary. HLO alone cannot prove every eventual instruction. The controller's `scientific_attribution_valid` denotes control validity, not improvement: its no-new-mismatch guard only checks exact-to-inexact transitions, so already-inexact errors may worsen. Every targeted and untargeted result needs review.

Before hardware, resolve both guard defects in a new immutable revision, verify rejection locally, preserve the revised source/design/review in the durable archive, and pin/record the exact Python/JAX/jaxlib/runtime dependency environment. The present bundle records JAX but does not independently establish complete package/build identity. Correct non-small numerical invocation, one process/four v5p devices, highest precision and the exact XLA flags remain required. This review is not hardware authorization.

A separately authorized bounded TPU diagnostic must preserve all exact source/scan/hidden/loss/full/composed/replay/raw-kernel/RoPE and physical-placement controls, unchanged source/master/batch/raw operands, unchanged named results and native forward hashes, and complete bounded HLO. Any failed control is negative evidence and stops attribution. The CPU failures must not be tuned away.

Before any full Trainer comparison, independently inspect actual candidate TPU optimized HLO, quantify the common-raw differential, demonstrate targeted actual prefix/inputnorm/Q improvement, and inspect all results for new or worsened unacceptable errors. Null benefit stops the sufficient-cause claim. Only a separately authorized, separately pinned full Trainer comparison can then test the unchanged approved loss-vector criteria. Performance, BF16 integration, replacement/deletion and convergence remain separate gates. Earlier parent-provided BF16 full-Trainer evidence does not validate this FP32 hypothesis.

Scope: local read-only file/Git/stdlib audits only, apart from the two requested review outputs. No source edits, experiment execution, hardware/job/remote queries, secret access, object-storage operations or publication.
