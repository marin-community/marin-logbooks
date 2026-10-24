# Embedding backward static audit

**Confirmed in the normal train-step HLO:** the candidate uses different BF16 embedding-gradient reduction grouping. This supports a localization hypothesis, not a proved numerical cause. Full-Trainer parity remains rejected; no speed ratio is accepted.

Candidate report pin: `8aa8674d10894728ed0b455ae14e14f46457cf9f`.

| Stage | Native | Legacy |
|---|---|---|
| Incoming cotangent | BF16 B4 × S2048 × D2048, while.108[10] | BF16 B4 × S2048 × D2048, while.60[1] |
| Token route | Local B4, vocabulary offset, sort 8192 IDs | Normalize, permute 1↔2, batch all-gather B8, vocabulary offset, sort 16384 IDs |
| Before scatter | Full D2048 cotangent; gather matching sorted permutation | BF16 all-to-all exchanges feature chunks; gather matching sorted permutation; full B8 × D1024 |
| Scatter | BF16 [8192,2048] into [32768,2048] | BF16 [16384,1024] into [32768,1024] |
| After scatter | BF16 sum groups {0,2}/{1,3}, slice features to D1024 | Direct to optimizer |
| Optimizer | Completed BF16 [32768,1024] → FP32 | Completed BF16 [32768,1024] → FP32 |

Both begin embedding transpose from batch-sharded/full-feature BF16 cotangents. Native first accumulates each batch shard into full-feature rows and sums those partial BF16 rows across the data axis before retaining its feature shard. Legacy exchanges cotangent feature chunks and gathers matching token indices so each shard accumulates all batch tokens for its feature shard before any FP32 conversion. Same final placement does not constrain this intermediate grouping.

The indices are paired with cotangents: each path sorts adjusted token IDs together with an iota, then uses that permutation to gather BF16 cotangent rows before scatter. The embedding scatter uses a BF16 add reducer. The unrelated CE scatter is FP32 one-dimensional `[67108864]` with `region_15.18`; it is not this dataflow.

Both reported canonical embedding gradients have global shape `[65536,2048]`, dtype FP32, spec `[model,data]`, and the same device order. Native axes are Explicit; legacy axes are Auto. Intermediate token ownership and summation grouping still differ.

Critical evidence boundary: the reported gradient comparison is from the hook-enabled diagnostic executable; the retained HLO is explicitly the no-hook executable. The normal embedding gradient feeds the correct embedding optimizer operand, but no diagnostic optimized HLO was retained to establish its exact same lowering. Raw arrays are absent; neither incoming cotangent equality nor numerical causality is proved.

Both surviving gradient tensor failures were reported by the diagnostic capture. Reported surviving embedding error: max abs `0.00048828125`, relative L2 `0.00020731330732814968`. Upstream differences remain possible, including block-0 o_proj.

Evidence (line numbers are one-based; JSON retains full-line hashes and exact source prefixes through metadata; only backend configuration is omitted from display):

- `native` 5085–5086: Actual entry token and embedding parameters; parameter storage shards both vocabulary and features.
- `native` 5056–5056: Vocabulary offsets used by native indices are {0,32768,0,32768}.
- `native` 770–787: Normalize negative token indices, subtract vocabulary-shard offset; tuple index 0 is the raw adjusted token index used by embedding backward.
- `native` 5131–5131: Entrypoint binds native token normalization to batch tokens and partition-derived vocabulary offset.
- `native` 5312–5313: Actual adjusted token indices are copied for embedding backward.
- `native` 5323–5335: Actual entry sort/index permutation, BF16 activation cotangent, scatter, BF16 sum/feature slice, optimizer ordering.
- `native` 4095–4143: Native sorted token permutation gathers matching cotangent updates; scatter is BF16 [4,2048,2048] into [32768,2048].
- `native` 4146–4162: Native BF16 add reducer, all-reduce across {0,2}/{1,3}, then feature slice; feature offsets {0,0,1024,1024}.
- `native` 4171–4174: Native optimizer converts the completed BF16 gradient to FP32 before Adam moment arithmetic.
- `legacy` 5487–5487: Actual entry legacy embedding parameter has the same [32768,1024] storage shard.
- `legacy` 5457–5457: Legacy vocabulary offset table is {0,1,0,1}, multiplied by 32768.
- `legacy` 5527–5540: Legacy tokens normalize, collective-permute {0->0,1->2,2->1,3->3}, all-gather batch to [8,2048,1], subtract local vocabulary offset.
- `legacy` 1299–1312: Legacy index tuple index 0 is adjusted gathered token indices.
- `legacy` 5704–5725: Actual entry BF16 cotangent copies/bitcasts/all-to-all before sorting/gather/scatter and direct optimizer use.
- `legacy` 4985–5033: Legacy sorted token permutation gathers matching BF16 cotangent updates; BF16 [8,2048,1024] scatter goes into [32768,1024].
- `legacy` 5037–5042: Legacy completed BF16 scatter result converts to FP32 in the optimizer.
- `benchmark` 142–146: Canonical gradient export dispatches native to_state_dict versus legacy flattened state dict.
- `benchmark` 428–438: Reported gradients come from a distinct hook-enabled diagnostic train step, then canonical export.
- `benchmark` 459–470: Retained HLO is lowered/compiled no-hook normal train-step executable.
- `native_source` 43–72: Source-defined token/residual/storage/compute partition specs for the proposed control.
- `benchmark` 134–139: Legacy storage and compute axis mappings for the proposed control.
- `haliax_source` 70–81: Public legacy Embedding.embed is NamedArray.take followed by active scale.
- `native_source` 386–390: Native public embedding lookup is resharded weight.at[tokens].get(out_sharding=residual).
- `native_source` 447–450: Native canonical embedding gradient export retains token_embeddings under model.embed_tokens.weight.
- `legacy_source` 496–507: Legacy public embedding lookup and canonical HF embedding prefix.
- `native` 1949–1963: Unrelated FP32 one-dimensional CE scatter uses region_15.18, not the BF16 embedding reducer.

File hashes:

- `native` `c397cd0f3362628a21044a49271e62a3c560da5108c8355c8ffebb52ba54f243` — `/private/tmp/native-jax-query-scale-15b-trainer-v2-results/bfloat16/native.optimized_hlo.txt`
- `legacy` `423b5aa424139f9c3010cc1058e6e38853f744ea07809d06b20f98c68ecec9cb` — `/private/tmp/native-jax-query-scale-15b-trainer-v2-results/bfloat16/legacy.optimized_hlo.txt`
- `compare` `46e076e2e6780c85caa78f11f430701dd7172f2d6961c295911747be3903e403` — `/private/tmp/native-jax-query-scale-15b-trainer-v2-results/bfloat16/compare.json`
- `benchmark` `beb87d3c4293c176e24809418e0c2769859daecc57d00ef6ff6efda98beb0a77` — `/private/tmp/marin-native-jax-query-scale-rounding/lib/levanter/scripts/bench/bench_llama_native.py`
- `native_source` `c3cda8d2072dcc5069cdb7ff81561092ee170847a4e0f1516dfd4fa51542c984` — `/private/tmp/marin-native-jax-query-scale-rounding/lib/levanter/src/levanter/models/llama_native.py`
- `legacy_source` `3095d586edf05309174b344b22418e9b78180651036f62e89fb7fb473b0aa64f` — `/private/tmp/marin-native-jax-query-scale-rounding/lib/levanter/src/levanter/models/llama.py`
- `haliax_source` `d929ac66c4590c43afc7aa8b2b1f398aac0586662029b557587e8ecc3c916dd8` — `/private/tmp/marin-native-jax-query-scale-rounding/lib/haliax/src/haliax/nn/embedding.py`

Proposed bounded common-operand public VJP control (design only):

1. First pin source-derived layout: storage P(model,data), native compute P(model,None), tokens P(data,None), residual/cotangent P(data,None,None); legacy parameter/compute mappings and native sharding definitions are cited below. Then use one same-topology four-device v5p control with the frozen versions/XLA flags, global V65536 D2048 B8 S2048 and original storage P(model,data). Invoke public legacy Embedding.embed and native public JAX weight.at[tokens].get with the same residual output placement, preserving their Auto/Explicit axis modes. Differentiate with respect to FP32 weights through the production BF16 compute cast/reshard boundary.
2. Construct the FP32 embedding weights, int32 token IDs and exactly BF16 common cotangent once. Reuse identical global bytes for both VJPs and assert dtype/shape/values and canonical physical placement before comparing gradients. Use a frozen benchmark-token fixture if reconstructible, plus at most one explicit repeated-token/cancellation fixture; do not claim either reconstructs the absent full-Trainer cotangent.
3. Compare canonical FP32 embedding gradients under the unchanged BF16 atol=1e-5, rtol=1e-4, retaining max/mean abs, relative L2, mismatch count and selected coordinates. Include a common FP32-accumulation reference for the same tokens/cotangent to localize summation error without changing acceptance thresholds.
4. Retain both minimal public VJP optimized HLOs and verify whether the observed scatter/redistribution order recurs. If minimal graphs lower differently, record non-reproduction; add no speculative embedding fix. A passing control only bounds this isolated operation and does not clear full-Trainer parity or authorize a speed ratio.

No source edit, execution, compilation, test, job, remote query, download, egress, archive or PR occurred. Existing optimized-HLO parser definitions were reused with only the filename suffix adapted.
