# One isolated embedding feature-compute-view experiment

Question: can one ordinary public JAX lookup compute view make the native embedding transpose accumulate full-batch, feature-sharded BF16 updates as the legacy path does? This is a single frozen variant, not a production patch, custom VJP, accumulation-dtype change or fixture search. Stop after the retained results/HLO; if its transpose does not match the specified grouping, report non-reproduction.

The independent actual-TPU review at `/private/tmp/native-jax-embedding-vjp-tpu-independent-review.md` validated the original isolated control: forward exact, BF16 legacy/native mismatch at26,892 coordinates (max0.03125, relative L20.000257512987032736), with typed BF16 full-feature local-batch scatter then data sum in native versus full-batch feature-sharded scatter in legacy. It does not establish original full-model causality: original full-model cotangents/diagnostic HLO are absent, and hook/no-hook executables differ. Full-Trainer acceptance, adoption and performance remain unproved.

Reuse: `inherited_v2.py` is byte-exact immutable v2 runner SHA256 `fe7c320e0eb8c583dbce573eb4691cf402a4060e39ecb0eda8854701490cd51e`, imported as a helper without calling its main or modifying its globals. Existing source guards, exact three-statement original native fragment, production legacy `LlamaEmbedding.embed`, fixture, metrics, physical records, nonfinite receipt guard and JSON caps are reused unchanged. `inherited-config.json` is byte-exact SHA256 `9aa91221a40cda48a7154632f47908556de446c524a464a73ab208aa54508634`; the new config must preserve every inherited key/value and declares the one added experiment separately. The new runner and its entire feature-view function AST/statement count are frozen before observing results. No source8aa or primary worktree mutation occurs.

All four variants receive the same FP32 master table P(model,data), int32 token IDs P(data,None), and BF16 cotangent P(data,None,None). Outputs remain BF16 P(data,None,None); parameter gradients remain FP32 P(model,data). Legacy uses Auto axes; original native, feature view and FP32 diagnostic use Explicit axes. Actual shape/dtype/spec/mesh/device order/memory/global shard indices are checked against the declared input boundary before execution, and complete actual local layouts/addressable shards are preserved and compared with legacy for every variant. Matching input bytes are verified after device import. Outputs are physically compared for all three BF16 paths.

The only new feature-view function is ordinary `jax.vjp` through:

```python
weight = policy.cast_to_compute(w)  # actual frozen jmp BF16 cast, before view changes
weight = jax.sharding.reshard(weight, P("model", "data"))
tokens = jax.sharding.reshard(t, P(None, None))
hidden = weight.at[tokens].get(out_sharding=P(None, None, "data"))
return jax.sharding.reshard(hidden, P("data", None, None))
```

This is a deliberately changed isolated compute view using public `reshard/get`, not a claim that it already appears in production or that NativeLlamaModel has a public embed method. Original native continues to invoke its exact source fragment. This lookup-only scope excludes optional input-embedding normalization, which the original full-Trainer configuration explicitly disabled.

Hypothesized transpose: output reshard moves the common cotangent into full-batch/feature-sharded form before lookup transpose; BF16 scatter receives B8/S2048/D1024 updates into vocabulary-local V32768/D1024 rows, without a later cross-data sum. The gradient then returns to original FP32 storage. Only actual entrypoint HLO can establish this order. CPU F32 legalization with explicit BF16 round-trips must be described as CPU evidence, never typed TPU-HLO proof.

Fixture and runtime are unchanged: source8aa, exactly10 source/import hashes, explicit declared pin with honest no-Git metadata, fresh cache prefix and no bytecode writes, same versions/XLA flags/four ordered devices. Main V65536/D2048/B8/S2048; small CPU V64/D32/B4/S8; full dimensions are abstract-only locally. Benchmark seed42/generator seed44; one PCG20261023 unit-RMS-before-BF16 cotangent. No full-model cotangent, array artifact or additional seed/fixture exists. Pure lookup weight-gradient values are independent of synthetic table contents.

Retain original legacy/native controls and the inherited FP32 scatter higher-precision diagnostic, adding only feature_view. The primary acceptance pair is legacy versus feature_view under unchanged BF16 atol1e-5/rtol1e-4 and exact forward agreement at those gates. Original legacy/native rejection remains separately visible and does not invalidate the experiment's ability to test its new primary pair. Target parity is an isolated observation only; full_trainer_accepted stays false and speed_ratio null. FP32 scatter does not choose an exact mathematical oracle or replace BF16 acceptance.

All nonfinite output failures use the inherited explicit guard: a bounded partial report with source/config/runtime/common fixture and current variant provenance plus NaN/+Inf/-Inf counts precedes exit1. Known physical/byte/dtype/HLO-cap verification failures also retain structured partial reports. Finite numerical mismatches retain full reports and HLO; TPU target failure exits1, while CPU mechanics is judged separately. Mechanical injected nonfinite/placement failures are distinct probes outside the scientific fixture. No generic exception recovery or production monkeypatching is used.

Outputs are JSON plus complete HLO only: at most8 HLO files+1 JSON (hard count10), HLO2MiB/file and12MiB total, JSON256KiB, total13MiB, never truncation. No arrays/checkpoints/profiles/timing. Base guards reject unsupported source/runtime before scientific execution. Root and a fresh reviewer must review and archive before any TPU launch.

Shape-derived forward payload tradeoff on data2×model2, BF16=2bytes/int32=4bytes, ignoring algorithm overhead: original native feature-weight all-gather receives `(V/2)*(D/2)*2 = 67,108,864 bytes` (64MiB) per device. The feature view avoids it but replicates tokens, receiving `(B/2)*S*4 = 32,768 bytes` (32KiB) per device. Its forward activation reshard exchanges half of a local BF16 `(B,S,D/2)` buffer, receiving `B*S*(D/2)*2/2 = 16,777,216 bytes` (16MiB) per device. The model-axis reduction logical buffer remains `B*S*D*2/2 = 33,554,432 bytes` (32MiB) per device in either layout. These are logical payload estimates, not measured bytes, latency or a speed claim.
