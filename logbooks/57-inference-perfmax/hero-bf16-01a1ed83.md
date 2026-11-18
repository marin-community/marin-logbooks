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

## 2026-11-18 07:40 UTC — Native dependency passes on the current image

The [NUMA dependency result](analysis/hero-bf16-01a1ed83/numa-dependency-a2.json)
comes from the owned CPU coordinator on `sbzvxs64`, an ARM64 host. The retained
package and extracted binary hashes match. `libnuma` is available, default
memory policy is zero and `get_mempolicy` succeeds. CPU-plus-memory binding and
CPU-only binding both return zero on nodes 0 and 1. The child CPU masks are
`0-71` and `72-143`. `Mems_allowed_list` describes the container's cpuset and
does not itself prove the selected memory policy. The next probe also retains
`numactl --show` from each bound child.

The first task-exec used the base `python`, which lacked `zstandard`. The retry
used `/app/.venv/bin/python`, the actual coordinator environment, and passed.
The [execution record](analysis/hero-bf16-01a1ed83/numa-dependency-execution.json)
keeps that distinction. These probes create no additional allocation, CUDA
context or model. Their additional GPU cost is zero.

The [NUMA launcher](analysis/hero-bf16-01a1ed83/numa_launcher.py) stages the
same private binary before server startup and extends only that vLLM
subprocess's `PATH`. The profile explicitly declares `--numa-bind`; the pinned
runtime performs GPU-to-NUMA detection and binding. The research launcher
substitutes the worker entrypoint and server factory, then calls the native
coordinator and runner. It preserves preparation, checks, cache reset, fresh
server, native warmup, timing, gang controls, exports and allocation accounting.
The runtime wheel is unchanged. Each native host archive will include the
dependency hash and policy probes. This is a prepared candidate; no NUMA GPU
trial has been launched.

Pinned source describes retention zero as keeping semantic replay and shared
prefix junctions, while positive intervals also keep periodic checkpoints.
It applies to sliding-window and Mamba cache groups. It does not compact global
attention KV heads. The retained C64 server summary reports peak cache use
0.99690 and client reuse 57.40%, against theoretical reuse about 95.3%.
C32 reports peak use 0.76537 and client reuse 85.24%. These are correlated
pressure signals, not an isolated cause. Server counter-decrease warnings
remain; use client exports for the comparison. The retention-zero runs test
whether sparse local retention improves that tradeoff on the fixed checkpoint.

The [effective-policy probe](analysis/hero-bf16-01a1ed83/numa-dependency-a3.json)
confirms `policy: bind` with `membind: 0` and `membind: 1` for the two
CPU-plus-memory cases. The CPU-only controls retain the default memory policy.
The native plan CLI and Iris callable serialization both pass in the
[CPU validation](analysis/hero-bf16-01a1ed83/numa-launcher-cpu-validation.json).
Both live retention-zero deployments have passed their deployment identity
checks and are still running the remaining numerical probes.

## 2026-11-18 07:54 UTC — Retention zero passes both full shared suites

Both actual sixteen-GPU deployments pass reference, batch, cache, decode and
deployment checks before native warmup. Each has 19 exact cache comparisons;
maximum, P99 and P90 differences are all zero. C32 finishes checks at
07:48:07.929 UTC and C64 at 07:50:00.088 UTC. The
[live gate snapshots](analysis/hero-bf16-01a1ed83/retention0-a2-before-warmup-checks.json)
retain that scope and the current leader memory peaks: 118,086,631,424 bytes
(109.98 GiB) and 119,039,918,080 bytes (110.86 GiB). These are live high-water
marks; final host receipts will retain every host's peak and memory limit.
The native `measuring` status includes warmup. No AgentX result is claimed yet.

The first two CPU preparation failures returned CLI exit zero after printing
their failed result rows. Their result receipts correctly show
`job_succeeded: false`, no GPU attempts and zero GPU cost. Treat the retained
job/result state and validity checks as the outcome, rather than inferring
measurement success from the local CLI exit alone. The parent owns any shared
runner change prompted by this observation.

## 2026-11-18 08:47 UTC — Require actual worker affinity for the NUMA candidate

The parent reviewed the private dependency approach and required actual worker
affinity on every host before labeling a result NUMA active. The candidate now
loads a read-only [native worker extension](analysis/hero-bf16-01a1ed83/hero_numa_metadata.py).
Before numerical checks or warmup, its serving check calls the native collective
RPC path. Each actual worker records its PID, native RPC thread, local GPU and
DP identities, CPU mask, memory policy mode and node mask. It requires CPU
execution within the GPU-local NUMA node and strict `MPOL_BIND` to that node.
All thread CPU-mask counts are retained separately. These reads change no
model, tensor or cache state.

The native DP utility broadcasts to every engine and propagates any worker
failure, but returns only the first engine result. Each rank therefore also
writes its full metadata under `point/numa-affinity` in its own host archive.
The [check report](analysis/hero-bf16-01a1ed83/inspect_numa.py) states that
coverage limit explicitly. Inspect all sixteen rank records across the four
hosts before claiming NUMA active. A setup flag or capability probe alone is
insufficient. The current two retention-zero windows remain frozen.

The new ctypes memory-policy reader itself passes a
[native CPU component check](analysis/hero-bf16-01a1ed83/numa-memory-helper-cpu-validation.json)
under real `numactl` bindings to nodes 0 and 1. It returns mode 2 and the
matching one-node mask in both cases. This verifies the reader and syscall
before another GPU launch. Actual worker proof is still pending. Memory
policy describes allocation rules for that thread; it does not measure the
placement of resident pages.

## 2026-11-18 09:18 UTC — Retention zero improves both tested loads

Both normal one-hour windows pass native admission, duration coverage, unchanged
gang and the full shared numerical suite. The
[comparison](analysis/hero-bf16-01a1ed83/retention0-round1-comparison.json)
verifies equal checkpoint, tokenizer, corpus, prepared workload, protocol and
numerical criteria against each positive-retention reference.

| Configuration on sixteen GB200s | Output tokens/s/GPU | P90 TTFT | P90 completion | Client cache reuse | Profiling / warmup errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retention 128, C32 | 17.93673 | 166.45803 s | 171.59327 s | 85.2380% | 0 / 2 |
| Retention 0, C32 | 19.29162 | 145.36914 s | 156.44444 s | 94.1941% | 0 / 2 |
| Retention 128, C64 | 18.94936 | 312.16459 s | 324.18918 s | 57.40% | 0 / 1 |
| Retention 0, C64 | 23.60952 | 276.38157 s | 282.52929 s | 81.6861% | 1 / 3 |

Retention-zero C32 improves observed throughput by 7.55% and P90 TTFT by
12.67% against C32. C64 improves throughput by 24.59% and P90 TTFT by 11.46%
against C64. C64 now has 22.38% more throughput than the lower-load zero-retention
neighbor, with about 90% higher P90 TTFT. Both neighbors remain useful. These
are single windows on different nodes; repeat variability is unmeasured. The
positive-retention C64 reference requested 768 GiB per host; all other rows use
256 GiB. All request 32 CPUs and use the same actual serving runtime and graph
configuration. The inactive logical-KV key was removed only from the new rows.

Original client exports and raw checks remain in the checksum-verified native
archives. The leader archives are 334,618,697 bytes for C32, SHA256
`706355e1bc63c1cd76c610863f3139de10732216cfa9f3546548e9ba1b120003`, and
339,761,559 bytes for C64, SHA256
`e39902c1858b8e57c5d3e5faed4fde7c29202b3863d95244398466b4df84916a`.
Each was downloaded once. The comparison retains member hashes, all host
memory peaks and native result links. Both have 19 exact cache comparisons at
zero difference. These remain provisional serving checks and do not qualify
RL learning.

The [C32 request errors](analysis/hero-bf16-01a1ed83/retention0-c32-a2-request-errors.json)
and [C64 request errors](analysis/hero-bf16-01a1ed83/retention0-c64-a2-request-errors.json)
retain all dropped warmup records and the single failed profiling request.
C64's request `7e1937cf-dcd5-4bfc-bf47-e5736a7bbfde` fails with HTTP 500 at
08:24:03.598 UTC. The router reports a failed send to the local vLLM URL.
The [bounded log inspection](analysis/hero-bf16-01a1ed83/retention0-c64-a2-http500-analysis.json)
and [router excerpt](analysis/hero-bf16-01a1ed83/retention0-c64-a2-http500-router.txt)
do not establish a cause. An adjacent out-of-order stats warning concerns
engine 8; it is not evidence of a restart. The failed request has no timing or
token metrics. Native validity allows the observed 1/2109 profiling error ratio.

Complete allocated cost is 84,600 GPU seconds for C32 and 85,996 for C64,
170,596 total, or 47.387778 GB200 hours. This includes every setup, check,
warmup, profiling, drain, export and shutdown allocation. The two initial CPU
preparation failures add zero GPU cost. All GPU pods were gone before the two
remaining Succeeded CPU pods were deleted. The
[cleanup receipt](analysis/hero-bf16-01a1ed83/retention0-a2-cleanup.json) confirms
zero remaining owned pods. Resource row 2241 is closed.

The next selected test is TP1/DP32/EP32 with C128 on thirty-two GB200s. C64 still
has a 13.76-point client cache-reuse gap against its theoretical 95.4451%, and
reported peak KV use reaches 99.904%. More EP ranks reduce expert weight bytes
per rank and may free cache space; a larger communicator may also cost more.
This is a configuration comparison: allocation and total concurrency change
together, while offered session trees per GPU stay at four. Keep retention zero,
the exact runtime, BF16 KV, PIECEWISE graphs, 2,048 batched tokens, maximum
sequences 128 and 32 CPU/256 GiB per host. Shared checks run before native
warmup and the selected one-hour window. Do not continue on failed checks.

The [profile](analysis/hero-bf16-01a1ed83/retention0-dp32-c128.yaml) resolves on
CPU with the fixed runtime and current fixture. The
[inventory](analysis/hero-bf16-01a1ed83/dp32-inventory.json) sees 137 available
GPUs and ready spare domains with nine and eight whole GPU nodes. This is not
a reservation. Native Kueue supplies eight-host NVLink colocation. Row 2250
announces the 32-GPU share at requested interactive priority with normal Batch
fallback. NUMA remains prepared; its likely value will be reconsidered after
this capacity/communication result.

## 2026-11-18 09:25 UTC — Eight-host DP/EP32 candidate is admitted

The new trial is `/romain/knotmeter-20261118t091911z-31c846881afd/serve`, from
clean published source `88e31071f56412f5f6e8702a0644f738d0427543`. The original
[submitted plan](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-submitted-plan.json)
and [resolved plan](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-resolved-plan.json)
pin every input. The
[admission](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-admission.json)
shows all eight GPU hosts Running at Batch in `DH1-126-US-EAST-08A`, totaling
32 GB200s. Shared prepared-workload cache is a hit. The final prelaunch Iris
read saw 166 available GPUs. This uses the entire owned 32-GPU share, with no
other owned allocation or H100 work.

The [summary script](analysis/hero-bf16-01a1ed83/summarize_trial.py) reads the
unchanged native exports and reuses the earlier campaign's phase accounting.
Run it from the repository root as
`python -m logbooks.57-inference-perfmax.analysis.hero-bf16-01a1ed83.summarize_trial retention0-c32-a2 retention0-c64-a2`.
It checks original report hashes, keeps each selected statistics map intact and
verifies conservation of allocated GPU seconds across phases. The retained
native summaries include both full client-export values, all host memory peaks,
allocation times and separate setup, checks, warmup, profiling, drain/export and
shutdown costs. Large raw requests and numerical reports remain in the native
archives. The two R0 summaries conserve 84,600 and 85,996 GPU seconds exactly.

## 2026-11-18 09:48 UTC — DP/EP32 passes and enters the native hour

The full shared reference, batch, cache, decode and deployment checks pass.
The live [gate snapshot](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-before-warmup-checks.json)
reports actual TP1/DP32 across eight hosts with expert parallelism enabled.
Its reported KV capacity is 1,456,262 tokens per rank, compared with 1,102,533
in the retained sixteen-GPU deployment report, an increase of 32.08%. These
are deployment-report capacity values, not a measurement of aggregate reuse.
The sampled leader cgroup peak is 155,808,366,592 bytes (145.1 GiB), below its
256 GiB limit. Final receipts will cover all eight hosts.

Checks finish at 09:38:48.684 UTC. Native warmup then completes 1,406 requests
with zero errors in 440.4 seconds. Profiling starts at 09:46:16.250 UTC, with
the native 3,600-second admission boundary near 10:46:16 UTC. The window stays
frozen through export and cleanup. No throughput conclusion is available yet.

## 2026-11-18 11:15 UTC — DP/EP32 improves the measured tradeoff

The completed native hour passes all shared checks and native protocol,
coverage, submission and gang rules. The coordinator and all eight GPU tasks
succeed with no preemption. The [result](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-result.json),
[original export summary](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-native-summary.json),
[check summary](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-check-summary.json)
and [comparison](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-comparison.json)
retain the measurement and its limits.

| Retention-zero configuration | Output tokens/s/GPU | p90 TTFT, s | p90 completion, s | Profiling successes/errors |
| --- | ---: | ---: | ---: | ---: |
| 16 GB200, DP16/EP16, C32 | 19.29162 | 145.36914 | 156.44444 | 1586/0 |
| 16 GB200, DP16/EP16, C64 | 23.60952 | 276.38157 | 282.52929 | 2108/1 |
| 32 GB200, DP32/EP32, C128 | 31.42858 | 268.89855 | 281.34879 | 5245/0 |

The larger configuration gains 33.12% throughput per GPU with 2.71% lower
p90 TTFT against DP16/C64. Allocation and total offered concurrency change
together; four offered session trees per GPU stay fixed. Client reuse rises
from 81.6861% to 87.9519%, against a theoretical 94.0220% for this completed
mix. Reported peak KV use is 86.865%. Mean completed input length also falls
from 83,634 to 67,705 tokens. These observations support the configuration
choice but do not isolate its cause. Each configuration has one window;
repeat variability and a global optimum remain unmeasured. C32 remains the
lower-latency neighbor.

Correction to the 09:48 entry: the live warmup phase notice reported zero
errors, but the final export and [original request records](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-request-errors.json)
show three dropped warmup requests with empty response content. The 5,245
profiling successes and zero profiling errors are unchanged. Preserve both
the early log-derived claim and this correction.

The [leader receipt](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-leader-receipt.json)
pins the original 380,769,889-byte archive with SHA-256
`dc1d54ea6426a866be20f1cca9d51438622068b6dbe8fd6710027f116daa878f`.
Its client export SHA-256 is
`6878d0cf0599c6958347e8e47e6f75468691d90f0532764e004dc6ee31fbb7d3`.
The normal export and collection finished successfully; there was no
coordinator failure or recovery. Allocated cost is 181,460 GPU seconds,
50.405556 GB200 hours, including every setup, check, warmup, native hour,
drain, export and shutdown phase. Owned completed cost is now 352,056 GPU
seconds, or 97.793333 GB200 hours. Host memory peaks are 145.11, 110.15,
111.88, 110.69, 109.91, 112.15, 109.80 and 113.20 GiB, all under 256 GiB.
The [cleanup receipt](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-a1-cleanup.json)
confirms all GPU tasks released and no remaining owned pods. Row 2250 is closed.

The next selected comparison adds native NUMA binding to this same DP32/C128
configuration. The [profile](analysis/hero-bf16-01a1ed83/retention0-dp32-c128-numa.yaml)
uses the already prepared private numactl dependency, unchanged vLLM runtime,
and a required read-only check of actual worker CPU and memory policy. CPU
plan resolution passes. Retain all 32 rank records across eight hosts before
timing, rather than inferring binding from a flag. The check measures worker
thread allocation policy; it does not claim resident page placement. The
[inventory](analysis/hero-bf16-01a1ed83/numa-dp32-c128-inventory.json) sees 127
available GB200s, without reserving them. Row 2261 announces 32 GB200s and
zero H100s under this session's actual binding. Requested interactive priority
and ordinary Batch fallback remain unchanged.

## 2026-11-18 11:23 UTC — NUMA setup path failure is retained

NUMA trial `/romain/knotmeter-20261118t111414z-596d12bff98c/serve` uses source
`a71af3acb6ac190e6481602c6059e8f5bda0505d`. It stops before vLLM startup:
the serialized `__main__` factory retains the devbox `__file__`, and its
source-hash lookup cannot open that path inside `/app`. The
[submitted plan](analysis/hero-bf16-01a1ed83/numa-dp32-c128-a1-submitted-plan.json),
[resolved plan](analysis/hero-bf16-01a1ed83/numa-dp32-c128-a1-resolved-plan.json),
[failed result](analysis/hero-bf16-01a1ed83/numa-dp32-c128-a1-result.json) and
[leader receipt](analysis/hero-bf16-01a1ed83/numa-dp32-c128-a1-leader-receipt.json)
retain every attempt and traceback. The archive SHA-256 is
`89f735ed23e162f29f5a2e8ad164da5ddc40c7cb343d3111717356d29d1708a7`.
No serving check, warmup or measurement ran. Complete failed cost is
1,000.671997 GPU seconds, or 0.277964 GB200 hours. The
[cleanup receipt](analysis/hero-bf16-01a1ed83/numa-dp32-c128-a1-cleanup.json)
confirms no remaining owned pods or GPUs before retry.

The task-local factory now finds its source under the imported native
runner's repository root. The [local regression](analysis/hero-bf16-01a1ed83/numa-serialized-path-cpu-validation.json)
passes with a missing serialized `__file__`; dependency execution and server
creation there are explicitly CPU stubs. That does not cover the remote
boundary. Before another GPU allocation, run the published
[zero-GPU Iris setup probe](analysis/hero-bf16-01a1ed83/probe_numa_setup.py)
through the same by-value serialized factory, task image and CPU environment.
It performs real ARM dependency staging, local policy probes, source hashing
and native launcher construction without entering the server context.
