# 1.53B Splash numerical assessment

Both source `36e2a9a97f` comparisons completed, but neither passes the unchanged numerical criteria. No speed ratio is accepted. This analysis reads saved reports only; physical placement verification is a separate audit.

The workload has 1,531,037,696 parameters: 28 layers, width 2048, MLP 5632, 16 query/4 KV heads, head size 128, vocabulary 65536, batch 8, sequence 2048, data2/model2, on four v5p devices. Masters and Adam moments are FP32. FP32 compute uses highest precision; BF16 compute uses default. Both use `--xla_allow_excess_precision=false`.

| Comparison | FP32 | BF16 |
|---|---|---|
| First loss | Exact | Exact |
| First gradients | All 255 pass; max 6.10948e-7 | 26/255 fail; max .00634765625 |
| Adam mu | Pass | Embedding fails; all other tensors pass |
| Adam nu/count | Pass | Pass |
| Updates | Every tensor passes approved .5% L2; worst .0293347% | 248/255 fail strict checks; worst tensor L2 13.9607% |
| Diagnostic next parameters | Every tensor passes approved .5% L2; worst .0205737% | 248/255 fail strict checks; worst tensor L2 9.79078% |
| Normal next parameters/Adam | Same pairwise error summaries as diagnostic | Same pairwise error summaries as diagnostic |
| 20 measured losses | L2 .366196%, exceeds approved .1% | L2 3.64930%; fails original elementwise checks |

FP32 fails **only** the trajectory, at 3.662 times its approved relative-L2 limit. Maximum absolute loss difference is .0926728. Final measured losses are 7.61323309 legacy / 7.68892336 native. All first-step acceptance groups pass, including parameters and Adam state from the normal executable.

BF16 fails seven of the twelve acceptance groups. Of 255 first-gradient tensors, 248 are nonexact. The worst gradient tensor relative L2 is 1.71984% (layer 0 Q), although that small-gradient tensor passes the absolute elementwise tolerance. Adam mu fails only for the embedding; nu/count pass. The largest update/next-parameter L2 errors are also in the embedding. These are not accepted under the unchanged BF16 criteria.

The BF16 trajectory has maximum absolute difference .894164 at measured sample 8. Final measured losses are 7.94388533 / 7.85619164. The normal executable independently exhibits parameter/moment errors: all four pairwise state-error summaries exactly equal their diagnostic counterparts. Summary equality does not prove byte identity of each variant's diagnostic and normal state.

## Backward onset

BF16 head and final-norm gradients are exact. In decoder 27, down/gate/post-attention-norm/K/V gradients are exact. **MLP up, O, Q and input norm are already nonexact**, with respective maximum errors 4.76837e-7, 4.76837e-7, 4.76837e-7 and 9.53674e-7. All still pass strict checks. Decoder 26 has every gradient nonexact, and differences spread toward the input. The first strict failing decoder layer encountered walking backward is **layer 7 V**, maximum error 1.14441e-5.

Parameter-gradient summaries cannot reveal the earliest activation/cotangent mismatch or order parallel branches. MLP up being nonexact means an attention-only onset is not established. Both source MLPs implement `down(silu(gate(x))*up(x))`; both blocks have the same two residual additions. No missing term or mismatched source cast is demonstrated.

## Existing comparisons

Prior 1.53B reference-attention FP32 source `ad2bc20438` has the same exact initial parameter hash (`d79fb8c3…f453f9a`), batch hash (`d65acfa3…9caa4`), architecture, optimizer, precision and XLA flag. Its 20-loss L2 was .678130% and maximum error .238260; Splash gives .366196%/.0926728. Mean absolute loss difference is slightly larger with Splash (.0143156 versus .0137542), so this is not uniformly smaller error or a passed gate. Backend selection changes both independent trajectories.

The 107M Splash source `36e2a9a97f` BF16 result was bit-exact in every tensor of all twelve groups and all twenty losses. That result is workload-specific. The larger case changes depth 8→28, width 1024→2048, MLP 2816→5632, head size 64→128, vocabulary 8192→65536 and sequence 512→2048, including attention tiling. No single factor is isolated.

The completed reference-attention common-state diagnostic at steps 0/25 supports tiny near-zero-gradient differences amplified by first-step Adam at selected coordinates, with all strict comparisons passing at the common step-25 state. This does not prove the new Splash trajectory mechanism and supplies no BF16 evidence. Repeating that generic check is unnecessary.

## Evidence needed next

**No production fix is justified yet.** Inspect the captured BF16 optimized HLO around the final decoder SwiGLU up-branch VJP, shared residual cotangent accumulation and rematerialized fusion. A later common-operand block diagnostic must use the actual 1.53B shape and source `36e2a9` Splash, rather than launching the held 107M/source `0ac` reference script. It should compare decoder-27 full block, MLP tail and attention/residual prefix, with common FP32 masters cast to BF16 inside the VJP and explicit materialized-versus-fused controls. This analysis launches no run.

Raw unaccepted median timings, for provenance: FP32 legacy/native 1032.478/1035.345 ms; BF16 286.208/277.658 ms. No ratios are calculated. Longer training and convergence remain unvalidated.

Complete group/per-tensor errors, all twenty losses, layer records, source pins and report hashes are in `/private/tmp/native-jax-splash-15b-numerical-analysis.json`. Analysis generator: `/private/tmp/analyze_native_jax_splash_15b_numerics.py`.
