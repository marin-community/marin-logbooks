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

## Final decision

Pending experiments. No new accelerator work has run in this program.
