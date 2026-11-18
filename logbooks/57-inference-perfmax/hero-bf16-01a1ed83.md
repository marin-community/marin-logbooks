# Hero BF16 serving choices

Owner: native Codex session `01a1ed83-c2a5-74b1-b98d-b48055085778`.
Worktree: `/home/romain/work/marin-community/marin/hero-bf16-01a1ed83`.
Campaign: `s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-bf16-01a1ed83/`.

## 2026-11-18 07:22 UTC — Start from the reconciled current checkpoint

The delegated objective is to improve BF16 serving through the overnight
campaign, with selected measurements finishing near 15:00 UTC and all owned
allocations released before 17:00 UTC. The share is at most 32 concurrent
GB200s and zero H100s, including retries. Request interactive priority and
accept normal Batch fallback. No questions or external notifications to Romain
are permitted overnight. The separate FP8 child owns its numerical work.

Start at published research revision `49a1460df3cadee3e6b262527f96b732ea364800`.
The [baseline audit](baseline-reconciliation.md) and its JSON distinguish the
old 50.71015 output tokens/s/GPU result from the applicable current references.
The old winner used step144000, another tokenizer, private compact KV,
retention zero, native NUMA and another compilation configuration. Runtime
`0ac9dcb157ee86ccb447bc717259159c41ad456a` expands six global KV heads to twelve.
Its `grug_logical_kv_cache` key is inactive. Do not port or qualify PR #110.

The current fixed step400407 references use sixteen GB200s at TP1/PP1/DP16/EP16:

| Offered session trees | Output tokens/s/GPU | P90 TTFT | P90 completion |
| --- | ---: | ---: | ---: |
| 32 | 17.93673 | 166.45803 s | 171.59327 s |
| 64 | 18.94936 | 312.16459 s | 324.18918 s |

C64 has the highest qualified throughput tested on this checkpoint. C32 has
lower latency at a 5.34% throughput cost. Neither establishes the global best.
The earlier 5% similarity band was an agent heuristic, not a user threshold.
Keep neighboring tradeoffs visible. One normal window per point leaves
repeat variability unknown.

Read the old selected recipe, current C32/C64 results and exact cache-v2 checks.
The C32 serving configuration passes all 19 exact cache comparisons at zero
difference. The corrected suite also supports retention zero in prior endpoint
evidence. A new serving candidate still runs the full shared suite before
timing. The fixed export, tokenizer, goldens, BF16 numerical criteria and BF16
KV precision remain unchanged.

First hypothesis: retention zero can reduce retained recurrent-state pressure
and free cache capacity, with a possible recomputation tradeoff in this long-session workload. Measure C32 and C64 with
the current PIECEWISE graph/compiler configuration and maximum sequences 128.
Only retention and the removal of the inactive key differ from C32's serving
settings. Use 32 CPUs and 256 GiB RAM; measured old peaks are about 111 GiB.
The C64 reference requested 768 GiB, which remains a comparison limitation.
Run normal one-hour windows because a supported gain at either load could
change the recommendation. Do not spend another warmup on a very short replay.

The pinned native NUMA path requires `numactl`; the current image omits it.
Inspect dependency provisioning and native CPU/memory-policy behavior on CPU
before deciding whether another GPU comparison is useful. Compilation and
CUDA graphs stay enabled. Source-only checks reject graph modes that the
current architecture cannot safely support.

The initial [inventory](analysis/hero-bf16-01a1ed83/inventory-initial.json) records
247 Iris-available GB200s out of 824 and 205 ready, schedulable GPU nodes.
This snapshot is not a reservation. Recheck admission and actual allocation.
Prepared workload key `a451c72f990cb5e6cc6650dfd302532ce1799cee6543716fec2124afdfab5ffb`
is already available in the shared cache. Any miss is handled by the runner's
CPU coordinator. Do not prepare the corpus on devbox.

## 2026-11-18 07:26 UTC — Resolve the current fixture path before retrying

Both initial coordinators stopped before GPU submission. Historical copied
profiles used `fixture: hero-sft-step400407`. Revision `9b93fb2d3f` had moved
the fixture and removed shortcut resolution. The maintained preset already
uses `experiments/evaluation/inference_correctness/fixtures/hero-sft-step400407.json`.
Update the two owned profiles to that path. The fixed fixture has identical
bytes, SHA256 `0ad43045025f31097472694724631adebe63b7c04120af36feb0ef38ede52380`.
Run the actual CPU reference lookup before retrying. No numerical criterion,
golden, serving flag or runtime changed.

The failed source was `c1743e44b55cf6c74f6713c7c971b782762a8aff`. Coordinators
`/romain/knotmeter-20261118t072255z-4698a9d73b3e` and
`/romain/knotmeter-20261118t072255z-b78aac7ff367` retain original submitted
plans, coordinator errors and result receipts in this leaf's analysis directory.
Both results report complete cost of zero GPU seconds and no serving attempts.
This demonstrates a profile-migration trap when copying prior evidence. Inspect
the current preset's check command as well as the historical serving settings.

## 2026-11-18 07:32 UTC — Shared preparation passes and native NUMA setup is reusable

The CPU reference lookup passes all 37 histories and 15,855 scored positions.
The C32 retry's coordinator finds the existing prepared workload with its exact
checksum and submits sixteen GPUs. Its four host allocations are at Batch,
following the requested interactive priority. The C64 retry uses the same clean
source `3e7c6ca44a15413dcb5491aad903ef89ba23f116` and controls. The new identities
are `/romain/knotmeter-20261118t072838z-3cca138ac87b/serve` and
`/romain/knotmeter-20261118t073018z-33071107ec92/serve`. Both original plans
are retained beside this notebook. No benchmark result is available yet.

The earlier winning recipe retains a 37,936-byte ARM64 `numactl` package at a
checksum-pinned shared URI. Its binary hash is also pinned. Reuse those bytes
instead of installing packages or changing the task image. The
[CPU probe](analysis/hero-bf16-01a1ed83/numa_probe.py) extracts it under `/tmp`
inside the owned C32 CPU coordinator. That coordinator is already on a GB200
ARM host and requests zero GPUs. Test native CPU and memory binding on each
CPU-bearing NUMA node without creating a CUDA context or loading the model.
The probe's exact source is published before execution.
