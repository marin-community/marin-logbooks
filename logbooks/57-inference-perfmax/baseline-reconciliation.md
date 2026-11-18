# Hero baseline reconciliation

Final status, November 18: unbound BF16 TP1/DP32/EP32, retention zero and C128
is selected at 31.42858 output tokens/s/GPU. The matched NUMA hour demonstrates
no gain. Retention zero also improves the measured DP16/C32 and C64 points.
The [final handoff](handoff.md) gives controls, limits and rerun commands. FP8 at
retention zero or DP32 remains unmeasured. The audit below records the earlier
starting point before these experiments.

## Initial audit

The earlier 50.71015 output tokens/s/GPU recipe is useful prior evidence. It is
not a controlled regression baseline for the current 18.94936 result. It uses
another checkpoint/tokenizer and private compact KV support absent from the
reviewed runtime. At the initial audit, C64 was the highest-throughput qualified
step400407 point tested. C32 gave up 5.34% throughput while nearly
halving p90 request latency. Neither is established as the global best.

The audit reads the [earlier selection](https://github.com/marin-community/marin-logbooks/blob/d6f5660563606d7877e7f40b8a70a59703a35f52/logbooks/19-hero-agentx-throughput/owner-integration/hero-selected-recipe.json),
its executed manifest and readbacks, the current frozen C64 result, original
deployment report, model config and pinned source. The
[machine-readable comparison](analysis/hero-fp8-cache-validation/baseline-reconciliation.json)
retains identities and hashes.

| Control | Earlier confirmed recipe | Current measured BF16 reference |
| --- | --- | --- |
| Model | Hero-535B step144000 | Finalized Hero/Mochi SFT step400407 |
| Tokenizer | marin-tokenizer at a5ca45f2 | marin-tokenizer-tools at 3f6879ac |
| Actual runtime | Private a0be4d10f1 overlay | Reviewed 0ac9dcb157 wheel |
| Global KV storage | Six distinct heads | Six logical heads repeated to twelve stored heads |
| Reported token capacity/rank | 2,095,445 | 1,102,533 on the retained leader host |
| Hardware | 16 GB200, TP1/PP1/DP16/EP16 | Same type/count/parallelism; different nodes and date |
| Live session trees | 128 | 64; lower neighbor 32 |
| Prefix caching | Enabled | Enabled |
| Prefix-state retention | 0 | 128 |
| Scheduler budget | 2,048 tokens | 2,048 tokens |
| Maximum sequences/rank | 64 in the handoff | 128 in actual deployment |
| GPU memory fraction | 0.95 | 0.95 |
| NUMA binding | Enabled | Actual `numa_bind: false` |
| Compilation | Mode 3, graphs NONE; explicit deterministic compiler controls | Mode 3, PIECEWISE; combo kernels enabled |
| Router | v0.1.16, consistent hash, DP16, retries disabled | Same package and controls |
| Harness and corpus | AgentX 1.0.12/e4bb2f83; pinned 256K corpus, 393 sessions | Same identities, seed 42, context 262144 and native one-hour window |

Prepared-workload and preparation-code hashes differ. Sharing the raw corpus
does not make the tokenizer-dependent inputs identical.

Model/tokenizer and reviewed-runtime choices are deliberate constraints. The
compact-KV path is unavailable in the pinned runtime. [PR #110](https://github.com/marin-community/vllm/pull/110)
is still an open draft as of this audit. This goal does not implement, port or
qualify it. Its old model checks do not transfer to step400407.

The current preset's `grug_logical_kv_cache` key is inactive: no code in `0ac9dcb`
reads it. `GrugMoeAttention` passes twelve heads to `Attention` and expands the
six global heads before storage. Its flag therefore supplied no compact cache.
Contribution `4b32bbd15a` removes it from future presets. Frozen trial settings
remain intact. The skill now requires prior winning recipes and effective
settings to be checked before a maintained default is called the best baseline.

Omitted NUMA binding was a missed experimental lead; the inactive label supplied
no optimization. Retention
128 originally served the old test; the corrected checks now support zero. Its
performance remains to be measured on this checkpoint. Graph and compiler
choices also differ. These are testable leads, with no isolated gain assigned
to any one difference. Historical placement is not a current reservation.

Next experiments will test supported BF16 retention/NUMA/load choices and cheap
controls for the FP8 numerical failures. They will keep the checkpoint, goldens
and KV precision fixed. The healthy FP8 one-hour run finishes under its frozen
settings. No earlier-checkpoint reproduction or compact-KV campaign is required.
