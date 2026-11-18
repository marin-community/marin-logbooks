# Hero BF16/FP8 campaign handoff

Retain BF16. The completed FP8 configuration is faster than the initial BF16 configuration at the tested retention-128, DP16/C64 settings, but it fails required numerical checks. Targeted native runtime fixes make all recorded cache persistence controls pass in later screens. The fixed-reference quality screen still blocks FP8 acceptance. No numerical bound was relaxed, and no hybrid one-hour window was launched.

The strongest BF16 point passing the provisional serving policy is the independently measured unbound TP1/DP32/EP32, retention 0, C128 configuration: **31.42858 output tokens/s/GPU**, P90 TTFT **268.89855 s**, P90 completion **281.34879 s**. Its [selection and controls](../hero-bf16-01a1ed83/final-selection.json) are published in research `531198bb028f5dcb4bef1b33c3427ed6c4135748`. A matching NUMA-bound hour passes all numerical and native controls but measures 31.11883, 276.22557, 289.07288 respectively, with two profiling HTTP500s and four empty warmup responses. This single-window comparison supports keeping the completed unbound point; it does not establish a general NUMA effect.

## Completed one-hour comparison

| Configuration | GPUs | Retention / concurrency | Output tokens/s/GPU | P90 TTFT / completion (s) | Numerical outcome |
| --- | ---: | --- | ---: | --- | --- |
| Initial BF16 |16|128 /32|17.93673|166.458 /171.593|Full shared v2 pass|
| Initial BF16 |16|128 /64|18.94936|312.165 /324.189|v1 pass plus matching v2 supplement|
| Initial selective FP8 |16|128 /64|22.62167|263.604 /278.605|Short reference and strict cache fail|
| Later BF16 |16|0 /32|19.29162|145.369 /156.444|Full shared v2 pass|
| Later BF16 |16|0 /64|23.60952|276.382 /282.529|Full shared v2 pass|
| Selected later BF16 |32|0 /128|31.42858|268.899 /281.349|Full shared v2 pass|
| NUMA-bound BF16 |32|0 /128|31.11883|276.226 /289.073|Full shared v2 pass|

The [initial comparison](comparison-initial.md) retains exact commands, raw client exports, native gang/admission validity, fixed input identities, costs and cleanup. Initial FP8 gains 19.38% throughput and reduces P90 TTFT 15.56% against initial BF16 C64. That measured gain remains diagnostic because quality fails. Initial BF16 C32 trades 5.34% throughput for about 47% lower latency against C64; the earlier 5% similarity heuristic was never a user-approved cutoff. Later retention 0 and 32-rank points provide stronger qualified neighbors and provide higher measured BF16 rates. FP8 at retention zero or DP32 is unmeasured, so its relative speed there is unknown.

All normal windows use native 3600-second profiling with fixed step 400407 weights, tokenizer, 37 reference histories, AgentX 1.0.12 and the same prepared corpus archive. The initial precision pair differs in requested RAM: BF16 C64 requested 768 GiB/host, FP8 requested 256 GiB; both requested 32 CPUs. Actual BF16 peak was about 107 GiB and FP8's final peak about 112 GiB. The initial precision result is a comparison of serving configurations, not an isolated weight-only causal claim. One window per point leaves repeat variability unmeasured. Server counter-decrease warnings remain unresolved; independent retained client exports supply cache accounting. The old 50.71015 recipe uses different checkpoint/tokenizer/runtime/compiler and compact-KV settings; [baseline reconciliation](../../baseline-reconciliation.md) prevents treating it as this campaign's baseline.

## FP8 outcomes and targeted fixes

The [chronological notebook](../../hero-fp8-01a1ec55.md) retains every earlier failure. Full FP8, native linear/backend screens and the initial selective window all fail the unchanged provisional quality policy. The policy is a conservative engineering screen, not a user-approved RL drift budget. Metrics are top-50 probability-error bounds, not raw logit error, KL or learning results.

Native component controls found two concrete issues. Equivalent legal within-expert token orders change Marlin results in the tested geometry; canonicalizing whole expert regions removes that component sensitivity. Zero-weight padded route IDs can also change real-row outputs; making padded router logits deterministic removes that component sensitivity while preserving real logits/routes and caller inputs. Canonicalization alone still fails five strict full-model cache cases. The combined canonicalization and padding wheel passes all 19 strict cache controls at retention 128 on four and eight GPUs. This does not qualify retention zero or the selected DP32 topology. Throughput of these fixes is unmeasured. These observations separate the repaired cache behavior from the remaining fixed-reference quality failure.

| Later FP8 screen | Actual precision | Cache | Fixed-reference result | Other result |
| --- | --- | --- | --- | --- |
| Padding/canonical, 4 GPUs |48 of 48 FP8 Marlin W8A16|19 exact zero|Short max 0.672875 >0.65|Batch/decode pass|
| Selective Triton, 4 GPUs |48 of 48 FP8 W8A8|19 exact zero|Short max 0.762755 and P99 0.149631 fail|Decode max 0.250870 >0.25|
| Mixed Triton, 8 GPUs |24 of 48 FP8 W8A8, 24 BF16|19 exact zero|Short max 0.704834 >0.65|Batch/decode pass|
| Hybrid, 8 GPUs |24 of 48 FP8 Marlin W8A16, 24 BF16 Triton|19 exact zero|Short max 0.708935 >0.65|Batch/decode pass|
| Hybrid backup, 8 GPUs |Intended 16 of 48 FP8 W8A16|Not run|Not run|Fixture/setup error before storage gate|

The final [24-layer hybrid report](hybrid24-model-a1-summary.json) completes all 37 references. Short P99 0.106966 passes 0.125; long maximum 0.230577 passes 0.35. Changed-split maximum 0.0726005, batch maximum 0.00534361 and decode maximum 0.0995656 pass their original bounds. All eight actual rank records prove 24 BF16 expert layers plus 24 packed FP8 expert layers, BF16 activations/projections/KV and FP32 router storage. Each eight-rank worker stores 65,229,815,808 BF16 expert bytes and 32,614,907,904 FP8 expert bytes. The leader's cgroup peak is 111,857,893,376 bytes; follower peak was not collected by this leader-only correctness finalizer.

Hybrid24 ran from clean Marin `faeebbdb634f6841bb28e6cb3ab7e3e7256da74e`, tree `b816424c98`, using immutable runtime `746cafea89518eb33184211eef271d7bb659058f`. Wheel SHA256 is `1cea7d874971c1dd10bd2ce44619aa0d35707a4ebb0f29617b051604e73cf941`. Grug, Marlin and factory Python hashes are verified in the built wheel and cold installation; all 19 native payload hashes match pinned `0ac9dcb`. [Twenty native tests](hybrid-wheel-a2-components.xml) pass, with separate [eight-case loader/output/graph evidence](hybrid-wheel-a3-components.json). Test source `3f418eebea934e9c348a906eeeac939f18704322` differs only in validation lifecycle from the runtime wheel. These small-component results establish compatibility; full fixed-model checks decide acceptance.

Hybrid24 cost 10,493.132 GPU-seconds. The [16-layer attempt](hybrid16-model-a1-summary.json) cost 4,043.152 GPU-seconds, including its coscheduled sibling. Its native pytest tail records one fixture/setup error after 462.77 s. No loaded-storage, cache, numerical or memory report exists. Both native host archives are retained, but the initiating exception was lost from the bounded Iris terminal tail and a later Finelog read returned zero lines. This is a setup failure with an unresolved cause, not a numerical failure. The two jobs ended by 12:20:50 UTC. They were already terminal when the authoritative 21:24 UTC stop instruction arrived. No resources remained allocated during the later handoff.

## Source and contribution state

The [source-head record](final-source-heads.json) distinguishes every frozen runtime and current-main contribution. The final High review is complete. Three separate draft contributions are prepared; no runtime PR is open yet and no contribution is merged into main:

- Selective online FP8 support: `906495c0cdeb562a7a7d15532f7ce1b7af431116`, branch `codex/hero-fp8-selective-publication-01a1ec55`. It starts from frozen `0474521` with a precision-neutral test rename. It preserves the existing factory/backend path and matches the measured `2442477` runtime functions.
- Canonical Marlin ordering: `b0d1a26b6e365fb9fc5ee1f991ac064a477e855b`.
- Deterministic padded routing: `139b7521151c580f2e524691b53be00127992e64`.

All use recorded current-main base `47a8e415bff17b9376f02d9b92edcc67e962eaee`. The earlier selective contribution `0474521acd8093ea4cbf884dd435f59d5ecd7d5f` stays frozen. The broader hybrid backend proposal `35c9e15b36e5a83c58908d5ada4bfc12c8720ce4` also stays frozen and unopened. The High review prompted that separation so the measured selective feature can be reviewed without the experimental per-layer backend API. [Publication validation](../hero-fp8-cache-validation/selective-publication-validation.json) records the exact split and preserved patches. The [combined advisory](selective-hybrid-contribution-review.json) retains six originally started native reviews, four interrupted lanes, recovery of only the missing four lanes plus composer, all observed models and actions, and answers to every finding. Two test-label findings are fixed. The common loader/parity test is retained with rationale; a proposal to broaden global Marlin fallback is rejected to preserve the narrow default behavior. Required lint and source checks pass. These current-main branches have CPU/source/advisory validation only; they do not inherit whole-runtime GPU qualification from the 0ac-based wheel.

Reusable experimental-runner fixes include durable checksum-verified cloud-wheel installation, explicit precision reporting, failed-check diagnostic timing with recommendation rejection, and cgroup peak retention. Root has promoted the reusable pieces to the contribution. Fixture-path migration is byte-preserving and future-facing; frozen historical run sources/profiles remain unchanged.

## Cost, cleanup and missing evidence

The [complete owned cost ledger](cost-ledger.json) totals **377,163.668 GPU-seconds = 104.767686 GB200-hours**, zero H100. It includes every admitted failed setup, preemption, screen, component and one-hour window. Queue-only attempts are separately retained at zero allocation cost. CPU native builds requested eight CPUs for 98, 77, 99 and 78 allocated seconds, with zero GPUs. Final hybrid allocations use retained provider schedule timestamps through explicitly labeled Iris-observed terminal fallback. Numerical screens/failures retain unsplit allocation costs; normal windows retain setup/check/warmup/profile/drain/upload phase accounting.

[Cleanup proof](final-cleanup.json) closes all 35 owned board rows and checks all 41 exact native job identities plus actual provider state. No owned job, pod, queue entry or allocation remains. Root's cache-validation share and the independent BF16 agent's ledger are separate from this total.

Missing acceptance evidence is explicit: no FP8 configuration passes all fixed-reference criteria; the 16-layer hybrid has no full-model gate; no FP8 performance measurement at retention zero or DP32 exists; no throughput measurement of the canonical/padding fixes or hybrid exists; current-main contribution branches lack whole-runtime GPU qualification; no repeated performance window or RL learning qualification exists. The final High review is complete; publication and terminal CI remain pending. The overnight-search requirement is partly unmet: work stopped before the deadline and the handoff resumed late. No replacement GPU work was launched. Retain the qualified BF16 selection while preserving these fixes and failures for further authorized work.
