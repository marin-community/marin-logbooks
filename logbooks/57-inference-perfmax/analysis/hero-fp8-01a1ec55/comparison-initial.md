# Initial Hero BF16/FP8 comparison

Retain BF16 for serving. C32 is the balanced initial choice; C64 remains the throughput neighbor. Selective FP8 improves measured throughput but fails required numerical checks. Overnight improvement work continues.

| Serving configuration | Output tokens/s/GPU | P90 TTFT (s) | P90 completion (s) | Client cache reuse | Profile/warmup errors | GPU hours | Numerical outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [BF16 C32](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-fp8-01a1ec55/trials/20261118T045512Z-6bd49b5d193a/result.json) | 17.93673 | 166.458 | 171.593 | 85.24% | 0/2 | 27.041111 | Full v2 pass |
| [BF16 C64](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-fp8-01a1ec55/trials/20261118T030036Z-46000535d09e/result.json) | 18.94936 | 312.165 | 324.189 | 57.40% | 0/1 | 28.847778 | v1 pass plus matching v2 supplement |
| [Selective FP8 C64](s3://marin-us-east-02a/marin/knotmeter/campaigns/perfmax-hero-fp8-01a1ec55/trials/20261118T053943Z-0eded6360d5c/result.json) | 22.62167 | 263.604 | 278.605 | 73.93% | 0/1 | 29.472222 | Reference/cache fail; diagnostic performance |

All three use sixteen GB200 GPUs, TP1/DP16/EP16, the fixed step400407 checkpoint/tokenizer/goldens, AgentX 1.0.12, the same pinned corpus and prepared inputs, and native 3600-second windows. Comparison-control objects match exactly. All native protocol, admission, duration and gang controls pass. One window per point leaves repeat variability unmeasured.

C32 gives up 5.34% throughput for roughly 47% lower P90 latency. The earlier 5% similarity figure was an informal guide. It is not a user-approved cutoff. FP8 C64 gives 19.38% more throughput than BF16 C64 and 15.56% lower P90 TTFT. These measurements compare serving configurations: BF16 C64 requested 768 GiB RAM per host; C32 and FP8 requested 256 GiB. All requested 32 CPUs.

The candidate uses native per-tensor E4M3FN routed-expert weights, BF16 dense/shared projections and activations, FP32 routers, and BF16 KV/state views. All sixteen loaded ranks passed the storage check. Native Marlin stores FP8 bytes in INT32 words; actual expert storage is one byte per logical weight. The candidate wheel is source 244247767c and retains every native binary payload from reviewed 0ac9dcb. It is an explicit experimental runtime override.

At sixteen GPUs the candidate model occupies 56.07 GiB/GPU and offers 1,514,130 KV tokens/rank; BF16 occupies 86.42 GiB and offers 1,102,533. The inactive logical-KV flag supplied no compact KV storage. The [baseline reconciliation](../../baseline-reconciliation.md) separates this campaign from the older 50.71015 result.

FP8 still fails its unchanged provisional short-reference limits: maximum 0.719764 and P99 0.125869 versus 0.65/0.125. Long reference, batch and decode pass. The exact branch/replay probability comparison reaches 0.0346905 versus required zero. These top-50 probability-error bounds are serving checks; they do not establish RL learning qualification. No numerical budget was relaxed.

Each final export retains empty-content warmup errors. The FP8 progress logger reported zero, while its final JSONL records one. All three had zero profiling errors. Server counter-decrease warnings remain unresolved; this comparison uses the independent client exports.

Use an isolated checkout of each listed Marin source for replay. Historical profiles retain their original fixture paths; the updated contribution documents the equivalent explicit fixture paths. The native coordinator prepares or restores the fixed workload on CPU before requesting GPUs.

```sh
# Marin 1c9f39daf310aa369e17864744e486089095d0fb: BF16 C64
marin-env uv run --no-sync python -m experiments.evaluation.knotmeter.cli run \
  --profile logbooks/57-inference-perfmax/analysis/hero-fp8-01a1ec55/bf16-c64.yaml \
  --campaign perfmax-hero-fp8-01a1ec55 --priority interactive

# Marin 33d780669d: BF16 C32
marin-env uv run --no-sync python -m experiments.evaluation.knotmeter.cli run \
  --profile logbooks/57-inference-perfmax/analysis/hero-fp8-01a1ec55/bf16-c32.yaml \
  --campaign perfmax-hero-fp8-01a1ec55

# Marin bd82567837: selective FP8 C64; failures remain recommendation blockers
marin-env uv run --no-sync python -m experiments.evaluation.knotmeter.cli run \
  --profile logbooks/57-inference-perfmax/analysis/hero-fp8-01a1ec55/selective-fp8-marlin-c64.yaml \
  --campaign perfmax-hero-fp8-01a1ec55 --measure-failed-checks
```

The [machine-readable comparison](comparison-initial.json), [BF16 v2 identity/reuse proof](bf16-c64-v2-evidence-reuse.json), [loaded storage proof](selective-fp8-c64-a1-weight-storage-summary.json), [FP8 numerical summary](selective-fp8-c64-a1-check-summary.json), [FP8 export summary](selective-fp8-c64-a1-window-summary.json) and [complete cost ledger](cost-ledger.json) retain sources, controls, raw archive URIs and hashes. Every failed screen, setup and preemption remains charged.
