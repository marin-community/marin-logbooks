---
title: "Hero AgentX throughput on sixteen GB200s"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/19"
related_issues: ["https://github.com/marin-community/marin/issues/10388"]
research_branch: "codex/19-hero-agentx-throughput"
marin_base_revision: "ce69fe2538f26525a75d8d96e1820dce68cc568d"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: active
---

# Hero AgentX throughput on sixteen GB200s

## Question and decision criteria

Find the causes of Hero's low AgentX throughput and validate a faster BF16 serving recipe on sixteen GB200s. Keep the step-144000 export, tokenizer, all 393 pinned sessions, AgentX 1.0.12, 262,144-token window, seed 42 and preset replay controls fixed. Use short diagnostics to select candidates, then full-hour AgentX runs at matched loads and enough traffic to expose a throughput knee. A configuration gain and a gain from higher offered load are separate results.

Use one interactive allocation of four four-GPU nodes in a currently spare NVLink domain. Verify placement before allocation and after admission or replacement. Count setup, idle time and failures in GPU-hours. Keep large raw artifacts in object storage. Preserve baseline and failed trials. Publish retained code as focused draft PRs, leave every PR unmerged and hand the recipe to the benchmark owner.

## Baseline

The completed [C8](https://github.com/marin-community/marin/actions/runs/40705471793), [C16](https://github.com/marin-community/marin/actions/runs/40724334189) and [C32](https://github.com/marin-community/marin/actions/runs/40740879931) points share Marin source `ce69fe2538f26525a75d8d96e1820dce68cc568d` and [the frozen recipe](https://github.com/marin-community/marin/blob/ce69fe2538f26525a75d8d96e1820dce68cc568d/experiments/evaluation/agentx/hero.yaml). C8 means eight concurrent live agent session trees; subagent requests can raise instantaneous request concurrency above eight. The three points delivered 2.23178 / 3.27650 / 6.51112 output tokens/s/GPU with 262 / 418 / 841 completed measured requests and zero measured errors. C64 has no completed profiling window.

The recipe uses the permanent BF16 split export with manifest `b20fdd303eb6ff8993ae5b5d7add14d65787e70981c056fea6de9dea62acfae1`, tokenizer `marin-community/marin-tokenizer@a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`, released vLLM `3a054c458ca8a32609ba2bc578c4d9b93eabb395`, Torch 2.14.1+cu132 and fixed dependency constraints. Native logs confirm TP1/DP16/EP16/PP1, BF16/native KV, FlashInfer attention, Triton experts, linear expert placement, all-gather/reduce-scatter, 64 maximum sequences/rank, a 512-token scheduler budget and disabled prefix caching. ShortConv forces CUDA graphs to NONE; compilation remains enabled.

The earlier 571 output tokens/s/GPU result used 3,072 clients that immediately started a new request when one finished, roughly 149 input / 767 output tokens and 256 sequences/rank. Its token counters cover the interval while requests were still being supplied; final draining is excluded. It used another runtime. It is a reference to reproduce under matched conditions, not an AgentX target or a measured regression ratio.

## Entries

### 2026-10-28 21:36 UTC — audit before new GPU work

The C32 content-addressed Iris source bundle is `s3://marin-us-east-02a/iris/cw-us-east-08a/state/bundles/79cfd3095a80383a75dc9d6b8b0bb258edb9b9493661044ed4e042273b0693a0`. Downloaded bytes match that SHA256 (16,762,802 bytes). All four native host records name it. Comparing all 4,738 bundle members with `ce69fe2538f26525a75d8d96e1820dce68cc568d` found no untracked file and one different tracked member, `lib/levanter/infra/babysit-tpu-vm`, outside serving. The serving and benchmark source bytes match the commit. This research branch starts at that exact commit; inherited workflows are removed only for archive publication.

The baseline's separately built split-loader plugin was an abandoned candidate. It was stashed and unused. The completed slow runs used the released wheel's native split loader. Per-host `MARIN_VLLM_WHEEL_VERIFIED` records identify the same ARM wheel hash `e8520d4493e5a60f9f6aa96ea3bc80c42e45258e4d2c45882869fc5921df3e15`, SM100 and the compiled `_C_stable_libtorch` module inside the installed distribution. These records do not contain a checksum of every installed module. Reproduction will capture source files, extension hashes and a complete installed-package inventory on every host, in addition to launch provenance.

[The per-host extraction](baseline-inventory.json) covers all twelve measured hosts. Their exact constraints all hash to `b0e4e774c46cd6d6b77f7ed9c97cbcb84066c4bdc8b7361cdc3ad80316aa4771`. The separate four-host startup qualification retained complete package inventories, all with hash `8648cb29763f2f8e97981ad33b5d25381ff2ebcb40ccc46ea65221eacd9f2d27`; these are startup inventories, not retrospective freezes of the later measured processes. C8 and C16's two other [source bundles](baseline-bundles.json) have the same 4,738 member names and identical member bytes as C32. Only ZIP container bytes differ. No serving overlay was shipped in those bundles.

| Existing change | Baseline status | Decision |
| --- | --- | --- |
| [Split expert loading #77](https://github.com/marin-community/vllm/pull/77) and [bounded reload #78](https://github.com/marin-community/vllm/pull/78) | Native loader/reload implementation is in the fork's [vLLM 0.30 refresh](https://github.com/marin-community/vllm/pull/92). The running full-model load and verified released wheel are recorded. | Reuse; do not restore the abandoned plugin. Reload is not in the static AgentX hot path. |
| [ShortConv profiling metadata #83](https://github.com/marin-community/vllm/pull/83) | The released Grug module uses `attn_metadata_raw.get(self.prefix)` and the lag-zero path for missing Mamba metadata. The port contains `4c6a2cc228`. | Already present despite the older PR remaining open. |
| [Streaming reasoning/tool parser #94](https://github.com/marin-community/vllm/pull/94) | Missing from `3a054c`; released in `ac3b5ea5e851`. | Candidate correctness change; no measured throughput effect. |
| [Hero router precision / YaRN #97](https://github.com/marin-community/vllm/pull/97) | Missing from both released GPU wheels; merged as `b316582ba80ed7cbccb908f3a957578ae2ed2899`. The pinned export has default RoPE, not YaRN. Router rounding applies to Hero. | Pin separately and reuse existing code if selected. Requires focused numerical checks; do not silently change the baseline. |
| [Row-invariant gates #96](https://github.com/marin-community/vllm/pull/96) | Open, opt-in, disabled in the baseline; its full-engine evidence is Snowball/H100. | Not required by this throughput task. It does not establish Hero graph support. |
| [Compatibility cleanup #95](https://github.com/marin-community/vllm/pull/95) | Open; retains the Hero graph guard. Its documented tiny-GB200 removal probe produced NaN logprobs after batch sizes changed. | Do not remove the guard. Keep dependencies frozen; broad refresh is outside the current experiment. |
| [Marin explicit-EP benchmark #9661](https://github.com/marin-community/marin/pull/9661) | The AgentX command already enables EP and native logs prove EP16. | Reuse benchmark ideas where useful; a missing EP flag does not explain this baseline. |
| [Marin release adoption #10022](https://github.com/marin-community/marin/pull/10022) | The baseline selects `3a054c` explicitly in its isolated environment. | Shared-default adoption cannot improve these already pinned runs. |

Current vLLM main is `3c4f7d95edc9f38b2027f6d40fafc9df527b35dd`. GrugMoE is the fork's serving implementation for Hero and Snowball. Since `3a054c`, its only model implementation change on main is #97. Release/build credential and test-bundling changes do not change the serving hot path. Relevant open branches and their PRs were checked on 2026-10-28. No ready, qualified Hero graph fix was found.

Marin main was fetched at `623134de90` for the audit. Later auxiliary-request queue and broker recovery changes operate outside AgentX's direct native vLLM endpoint. Training-side routing and NCCL fixes are outside this static serving process. Baseline source `ce69fe2538` already carries the explicit runtime-pin and multi-host DP adapter needed by its recorded command; those remain pending in the benchmark delivery owner's [runner contribution #10645](https://github.com/marin-community/marin/pull/10645). Its separate [dashboard contribution #10646](https://github.com/marin-community/marin/pull/10646) is outside this program.

C32's client used 0.36 cores at the median and 0.47 at the observed maximum; cgroup throttled time did not grow. The roughly 70K-token average input, low requests per rank, disabled prefix reuse and graph-free ShortConv path motivate the first experiments. These observations do not yet apportion GPU time.

Next: finish per-host dependency and source audit, preserve the exact prepared workload, then acquire one held allocation. First run the pinned runtime's small native tests and controlled cold/warm branched requests. Compare baseline scheduling with a larger prefill budget and native prefix caching separately. Profile a diagnostic run before selecting any kernel or collective patch. Freeze and publish each experimental source/configuration before executing it.

### 2026-10-28 21:42 UTC — CPU preparation and first GPU checks

The unchanged native preparation command found the existing regional workload cache and passed its offline check: 9,602 native conversations, 68,266 turns, native key `34fa3e05468c1d3d80e509922bc96b24`, metadata SHA256 `91644e1a1759069c2687f4c697796925fe32f6028b1f80c2e5dfdfc763bfd8b6`. The source corpus has 393 recorded trace trees, called pinned sessions in the task; native conversation and turn counts describe their prepared representation. They do not count live replay sessions or completed benchmark requests. [The receipt](workload.json) retains all corpus, tokenizer and archive identities. Preparation used no GPUs.

```sh
marin-env env MARIN_TEMP_PREFIX=s3://marin-us-east-08a \
  uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python -m experiments.evaluation.agentx.workload \
  --series experiments/evaluation/agentx/hero.yaml --output PREPARATION_DIRECTORY
```

The first GPU operation will capture the actual installed dependencies, critical Python sources and compiled vLLM extension hashes on all four held hosts, using [capture_runtime.py](capture_runtime.py) in the same constrained isolated environment as serving. Then [native_shortconv_probe.py](native_shortconv_probe.py) will run the unchanged native request-history/isolation regression on one GPU. Its external test source is `tests/models/test_grugmoe.py` at vLLM `3a054c458ca8a32609ba2bc578c4d9b93eabb395`, SHA256 `e94366312fc412bbd17cb91cc5f10c25825b142b622ca4a36b0ed0e63459d670`. This small test does not establish full-engine prefix-cache or graph correctness. It retains the native reference calculation and numerical tolerances.

### 2026-10-28 21:56 UTC — held runtime and native kernel check

Held job `/romain/dev-gpu-hero-agentx-01a1846a` acquired four four-GPU nodes at interactive priority. [The allocation record](allocation.json) retains Pod UIDs, start times, resources and actual image digests. Fresh Kubernetes inventory before allocation and after admission confirmed one NVLink domain with no Hero pretraining. The four admitted hosts are distinct and match the allowed-node receipt. The task image is the baseline digest `b66c48a1caca212d2fa970f9fb4d1e0685ab27be9014f335687b1031244f0467`.

The isolated runtime launch prefix was copied from the recorded C32 command, with its exact constraints and `UV_NO_CONFIG=1`. The captures on [host 0](runtime-host-0.json), [host 1](runtime-host-1.json), [host 2](runtime-host-2.json) and [host 3](runtime-host-3.json) agree on all 205 installed packages, nine critical Python-source hashes and eighteen compiled-extension hashes. Normalized package names and versions exactly match the retained startup inventory: no changed, missing or additional package. Torch is 2.14.1+cu132, CUDA runtime 13.2 and NCCL 2.30.7. Sixteen unique physical GPU UUIDs are recorded. The installed Grug module matches the retained released-source snapshot byte for byte.

The [native ShortConv probe](shortconv-probe.json) passed on one GPU with the release's unchanged reference calculation and tolerances. It checks profiling without state mutation, separate request prefill histories and subsequent decode isolation. It does not qualify full-engine prefix caching or CUDA graphs.

Next run: [fresh C32 baseline](experiments/baseline-c32.json), unchanged one-hour AgentX recipe, all 393 traces, normal preset warmup/replay and the same seed. Use the held Pods with the existing `experiments.evaluation.agentx.run.run` entry point. The recipe fingerprint remains `55ff17c3c321ecfdd54cb5308ab0e39ed1e29833a3dbf2134dd04351ee50366a`. No intrusive profiler, Marin metrics collector or NCCL collector is enabled. All allocation time, including the runtime capture and setup, is charged to this program.

### 2026-10-28 22:10 UTC — full Hero ready; next diagnostics prepared

The fresh C32 server became ready at 22:10:05 UTC and AgentX began its native warmup. All sixteen DP/EP worker ranks logged their expected expert ownership and native kernel/communication paths. The four holder attempt UIDs still match admission. The native record calls the whole client phase `measuring`; that includes warmup and is not evidence that the one-hour profiling window has started. The run's actual profiling timestamps and validity will come from AgentX's export.

[diagnostic_server.py](diagnostic_server.py) prepares the next controlled experiments inside the same held Pods. It reuses the benchmark's server and Iris adapter, records the actual command and stops on a local file or a bounded deadline. It can configure the native Torch profiler for five iterations, with two delayed iterations, one warmup iteration, CPU stacks and tensor shapes, and no memory tracing. It has not run yet. Profiling will use a separate server state from AgentX confirmation. The adapter can also express TP1/DP4/EP4/PP4 on the same sixteen GPUs; its actual rank placement must be checked before interpreting a topology result.

One graph hypothesis is worth testing after profiling. The [native compilation setting](https://github.com/marin-community/vllm/blob/3a054c458ca8a32609ba2bc578c4d9b93eabb395/vllm/config/compilation.py) excludes listed `splitting_ops` from piecewise CUDA graphs. The baseline's effective list includes `vllm::short_conv`, while [Hero calls `vllm::grug_moe_short_conv`](https://github.com/marin-community/vllm/blob/3a054c458ca8a32609ba2bc578c4d9b93eabb395/vllm/model_executor/models/grugmoe.py). The Hero name is absent from the list. Keeping ShortConv outside graphs may permit graphs around other compiled pieces. This is an untested hypothesis, not the established cause of the earlier NaNs or a reason to remove the full-graph guard. It needs changing-batch, cold/warm and branched-state numerical tests before a full-model throughput trial.

The earlier 571-token short-input producer is `marin-community/MarinSkyRL@dfdc80be6643f49af8d22ab9adda5319ae34befa`, file `hero_serving_capacity.py`. It used batch-invariant mode, a Ray-managed engine client, captured routed experts, disabled FlashInfer autotuning, a 4K cap and 256 sequences/rank. Its fixed bank has SHA256 `b15aeac7c3eb714fe45fc6b8e274342ad26f00970d84f2b7985b42c422b4e905`. The supplied window covers native counter deltas between 20% and 80% of requests completing, before the final drain. These differences must be recorded in a current-runtime reproduction; the old output rate cannot by itself diagnose AgentX.

### 2026-10-28 22:38 UTC — actual baseline window; diagnostic controls

Native warmup completed at 22:38:09.752 UTC: 351 completed requests, no cancellations and zero errors. The measured profiling phase began at 22:38:09.843 UTC with a 3,600-second target. Kueue's admission and quota-reservation transitions both occurred at 21:50:38 UTC; GPU accounting starts there, including setup and warmup. The held allocation still has its original four attempt UIDs. No throughput conclusion is drawn from the unfinished window.

[cache_probe.py](cache_probe.py) prepares full-engine comparisons using the pinned tokenizer, an 8K-token shared prefix and three distinct continuations. Independent salted requests supply cold references and a cold repeat. A shared salt then exercises different branches and API batch sizes 1/2/3/1, always using native `X-data-parallel-rank: 0`. It retains all inputs, requests, responses, cache usage and logprob differences. Sixteen forced output tokens make this a small diagnostic; those output controls do not apply to AgentX. The comparison reports numerical differences without assigning a new tolerance. First run it with caching disabled as a negative control, then on a separate cache-enabled server.

[The diagnostic cases](experiments/native-diagnostics.json) use the released native `vllm bench serve` client and existing CPU-load sampler. One-token 4K and 64K input banks compare representative long-prefill shapes with the historical probes; random input tokens and current runtime are explicit differences. Separate five-step traces cover long prefill and short-input decoding. Short-input load points can show whether increasing concurrent decoding fills the experts. Execute them adaptively after the cheapest cases; these are neither AgentX points nor a direct reproduction of the historical Ray client. Keep intrusive profiles separate from the throughput cases.

The diagnostic launcher can now replace only prefix caching, token budget or sequence limit, and records those overrides next to its actual command. PP4 is a separate topology choice. No native serving implementation, dependency pin or baseline recipe has changed.

### 2026-10-28 23:13 UTC — exact historical inputs recovered

The retained full historical GB200 report is still available locally. Its 207,112,626 bytes hash to the issue's `04b728f72b5dd8a67b19efb4b4c884d5c525da531685e3bcc2f4cd47012c289a`. Extracting only its first workload bank recovered all 1,024 rows. Canonical JSON hashes to the recorded `b15aeac7c3eb714fe45fc6b8e274342ad26f00970d84f2b7985b42c422b4e905`. [The receipt](historical-workload.json) retains the original source and a permanent copy in this program's S3 prefix.

[historical_bank.py](historical_bank.py) supplies those original token IDs and output lengths to the released native benchmark's `SampleRequest` interface. Native HTTP requests, concurrency, measurement and detailed exports remain unchanged. This avoids a text/token round trip or a Ray environment refresh. It permits matching the historical input bank on the current runtime; API, runtime, batch-invariant mode, sequence/context limits and warmup differences still need to be stated. Compare whole-bank rates before making a supplied-window comparison.

The profiling launcher also records native NCCL initialization and graph selection with INFO logs in the separate diagnostic server. [summarize_profile.py](summarize_profile.py) reports per-device interval unions, NCCL overlap, CUDA runtime/driver calls and native operators. A disposable known-overlap gzip-trace check passed. Kernel sums do not establish critical-path shares, and busy NCCL intervals do not distinguish transfer from waiting.

[loaded_runtime.py](loaded_runtime.py) will take a read-only process-map and FlashInfer cache inventory during the baseline. It uses no CUDA/NVML calls and defers file hashing until after the measured run. This supplements the pre-start wheel inventory with dynamic libraries and downloaded cubins; cached membership alone does not prove execution.

### 2026-10-29 00:02 UTC — valid C32 repeat and diagnostic controls

The fresh C32 repeat passed native validation: 826 successful measured requests, zero errors, no replacement attempts and 100% TTFT/inter-token metric coverage against the required 95%. Its unchanged pinned processor reports **6.11275 output tokens/s/GPU** (97.80394 global) over a 3,629.83317-second successful-request window. Mean input/output lengths are 69,629 / 429.8 tokens. Mean TTFT is 55.44 seconds and inter-token latency is 0.27121 seconds. This reproduces the low rate, about 6% below the earlier C32 point, and is not an optimization gain. All 393 source traces remained eligible; the native window started 35 session trees and completed 10, so eligibility is not a claim that every trace finished.

The 3,600-second sending phase ended at 23:38:09 UTC. The preset's drain/cancellation path completed at 23:38:49 UTC; 38 unfinished credits were cancelled normally. These are distinct from the processor's successful-request time window. All four benchmark workers exited zero and validated their original attempt UIDs. GPU process lists were empty before the next server began. The [leader archive](s3://marin-us-east-08a/marin/users/romain/hero-agentx-perf-01a1846a/baseline-c32/host-0/outputs-5e4b8290ffd212088198899dd6070932ca51c693612ea1e20555293aa74c52c3.tar.zst) retains raw exports, the processed result and runtime audit; its SHA256 is the filename digest. The prepared workload also has a [permanent verified copy](workload-retention.json).

The runtime snapshot covers twenty library/cubin files per host and retains the ten selected FlashInfer files (24.9 MB/host). One GPU worker per host held a deleted FlashInfer `fmha_gen.so` mapping. That mapping's old file bytes could not be read through `/proc/.../map_files` under the container's permissions. It remains marked as a missing byte hash; the current cached library is not proof of the deleted mapping's identity. The native log identifies BF16 FlashInfer `trtllm-gen` on SM100 and the downloaded cubin family. No unsupported graph mode was enabled.

Correction to the diagnostic preparation entry: `run_logged` does **not** sample client CPU. The earlier client's 0.36/0.47-core observations came from an external sampler attached by the benchmark owner. The fresh C32 run has no `client-load.jsonl`, so those numbers do not describe this repeat. [client_load.py](client_load.py) reuses that sampler's process/thread accounting for subsequent native diagnostics; only its selector and output/deadline interface change. This preserves separate evidence for client headroom.

The separate diagnostic server started at 23:48 UTC from published source `cd751ff02ec655f626d5404334437dc80eee4158`. It retains native TP1/DP16/EP16, 512 tokens/step, disabled prefix caching and graph mode NONE, with the bounded Torch profiler configured but inactive until requested. The first 4K-input/one-output diagnostic completed all sixteen requests in 5.67 seconds. Its native client inherited server sampling defaults; the native client now warns that temperature is not implicitly zero. The queued cases now set temperature zero and a per-request seed of 42 explicitly, in addition to the dataset seed. Preserve that first diagnostic under its original command, and do not use it as a matched historical reproduction.

The disabled-cache branch control returned finite logprobs throughout. Single-request cold and final branch-A repeats matched exactly. Changing API batch sizes shifted some logprobs by up to 0.4039, and branch A's greedy continuation changed in the three-request batch despite caching being disabled. Later cache/graph comparisons must account for this existing batch sensitivity and compare corresponding cold/warm batch shapes; no tolerance was relaxed.

### 2026-10-29 00:58 UTC — native profile and focused next candidates

[The full-hour aggregate and four-host validation/retention receipts](results/baseline-c32.json) are now exported. [The separate diagnostic profile](results/diagnostic-baseline-profiles.json) captured five actual 512-token prefill steps on all sixteen ranks. Rank 0's GPU timeline spans 2,572.6 ms: 1,581.4 ms with only NCCL active, 442.3 ms with compute/memory active and 548.9 ms empty. This intrusive capture includes CPU stacks and shapes. It inflates rank skew and does not assign those fractions to the unprofiled baseline or distinguish collective transfer from waiting. Native transport logs explicitly select P2P/MNNVL across the sixteen-rank NVLink clique; NET/IB availability is not evidence of an IB fallback.

Five steps contain 720 ShortConv calls and roughly 16,370 CUDA launch calls on rank 0. The native ShortConv path allocates a decode result, stacks the decode/prefill pieces and copies the stack to its caller's buffer. A focused [source candidate](https://github.com/marin-community/vllm/commit/7b161cb460) writes decode directly into that buffer and copies prefill once. It adds a mixed decode/resumed-prefill check to the existing independent history/isolation regression, with its tolerances unchanged. GPU validation and performance measurement are pending; it is not a measured gain. The fork branch retains current main's already merged router fix, so any baseline-wheel overlay must isolate only this ShortConv delta.

The second native profile request returned success but emitted no new trace. Its schedule was exhausted after the first capture and the worker retains that profiler instance. Preserve this failed collector. The decode-only workload nevertheless completed all 32 requests, with mean inter-token latency 294.09 ms. The next fresh diagnostic server uses the native profiler with zero scheduled warmup and a five-iteration limit; check that the files exist before claiming a capture.

The historical C1024 case is still draining its original 4,096 requests. Its client monitor has no CPU throttling and uses substantially less than one core. Do not start another GPU diagnostic concurrently. A whole-bank C32 run would take roughly two hours at the observed decode rate; the queued lower-load cases are optional and will not be launched as a blind sweep.

The next focused full-engine case enables native prefix caching on a fresh released-wheel server, with all other limits fixed. [Its launch](experiments/diagnostic-prefix-cache.sh) retains the original graph guard. The branch probe now includes independent cold references at each API batch shape, because the disabled-cache control already showed batch-dependent logits. Requested settings are not proof of cache hits: retain effective Mamba mode, cache counters, cached-token usage and matching-shape branch comparisons before an AgentX trial.

## Final decision

Pending throughput experiments. No optimization gain or final serving recipe is established.
