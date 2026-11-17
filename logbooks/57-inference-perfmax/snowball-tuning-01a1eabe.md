# Snowball serving campaign

The decision is whether a smaller prefill token budget improves completed output
throughput or latency at 128 resident AgentX sessions. Retaining the checked-in
baseline is an acceptable result. This campaign uses the scope delegated for
[issue #57](https://github.com/marin-community/marin-logbooks/issues/57).

The initial source is `c07967597646fa1e0474e4982b193e36a373b647`. The execution
owner is native session `01a1eabe-cfa1-7f52-bd95-a69658df86e3`. Each execution plan
will retain the actual clean, published source revision.

## Fixed inputs

| Input | Value |
| --- | --- |
| Model and tokenizer | `open-athena/Marin-Snowball-67B-A2B-v0.1.1` at `59852d6f07261722bbec1edb6ab148317156b58e` |
| Hardware | One host, eight H100s, `cw-rno2a`, interactive priority |
| Runtime | `marin-vllm-gpu-20261113-0ac9dcb157ee`; source `0ac9dcb157ee86ccb447bc717259159c41ad456a` |
| Precision | BF16 weights and default BF16 cache; compilation and CUDA graphs enabled |
| Workload | `semianalysis_cc_traces_weka_062126_256k`; 393 sessions at dataset revision `8fecd2fc56694469f758f0afbbb6335ad3043740` |
| Corpus SHA256 | `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef` |
| Harness | AgentX 1.0.12, source `e4bb2f835ffedbbe6b76252ea5c93caec68c5faf` |
| Processor | `80dff120d27fd6092210f4546b8faa5e6c9723ac` |
| Protocol | Checked-in recipe: one-hour profiling, native warmup, replay, cache busting and validity |
| Checks | Complete `snowball-v0.1.1` fixture before AgentX, fixed Levanter scores and provisional policy |
| Golden manifest SHA256 | `f296e39eb6f4fae9f1964c92698667d5985bc9d2bfa9dcea16d22b2e0d8e8cd2` |
| Cache retention | Every 128 tokens, preserved for numerical/state coverage |
| Resource share | At most eight H100s at once, plus modest CPU preparation |

There is no qualified normal AgentX baseline for this runtime in the retained
campaign. Earlier short diagnostics have pending checks and cannot support a
recommendation. The [older calibration](analysis/snowball-historical-calibration.csv)
used a different runtime. Its three C128 repeats reached 134.55–135.33 completed
output tokens/s/GPU, with p90 TTFT 7.69–7.95 seconds. C256 reached 109.69–111.77
tokens/s/GPU with p90 TTFT 196.75–206.70 seconds. That supports starting at C128;
these values are historical evidence, not this campaign's baseline.

## Initial experiment ladder

1. Run the unchanged `snowball-h100` preset at C128 for a normal one-hour window.
   Its 8,192-token prefill budget, 128 sequences per engine and memory fraction
   0.94 define the baseline. Every required check must pass before profiling.
2. Test a fresh server at C128 with only `max_num_batched_tokens=4096` changed.
   Start with a 600-second diagnostic. It remains explicitly unqualified.
3. Confirm that candidate with a normal one-hour run if its throughput is close
   to or above baseline and its latency makes the tradeoff useful. Repeat a
   normal point if noise could change the decision. Stop a clearly worse point.

Completed output tokens/s/GPU is the primary measure. A throughput difference
within 5% is treated as similar for deciding whether a lower-latency candidate
deserves confirmation. Compare p90 TTFT and p90 completion separately. A short
diagnostic can reject a candidate or motivate confirmation; it cannot establish
a speedup. No universal latency cap or numerical-policy change is introduced.

The smaller prefill budget is a controlled serving hypothesis. vLLM's
[tuning guidance](https://docs.vllm.ai/en/latest/configuration/optimization/#performance-tuning-with-chunked-prefill)
predicts lower inter-token latency with smaller batches and better prefill
efficiency with larger batches. The pinned
[scheduler configuration](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/vllm/config/scheduler.py)
has chunked prefill enabled and accepts both values. Older Snowball
[serving observations](https://github.com/marin-community/marin/issues/10388#issuecomment-6299980093)
used 4,096 batched tokens but a synthetic load, so they establish plausibility
only. The recent upstream
[adaptive-prefill discussion](https://github.com/vllm-project/vllm/pull/67811)
also exposes this tradeoff; its experimental code is outside the fixed runtime
and is not used here. SGLang's
[tuning guide](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
similarly motivates observing batch load and memory pressure. It does not supply
a transferable setting for this vLLM runtime.

Preparation uses the runner's checksum-addressed CPU cache. Every attempt starts
a fresh server. Setup, compilation, warmup, drain, failures and retries count
toward allocated GPU-hours. Initial expectation is about 12 H100-hours for a
normal point and 5 for a diagnostic, with actual cost retained by the runner.
No fork refresh, quantization, speculative decoding, LMCache, trainer-alignment
investigation, scheduler change or dashboard work is in scope.

Campaign artifacts:
`s3://marin-us-east-08a/marin/knotmeter/campaigns/perfmax-snowball-tuning-01a1eabe`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fperfmax-snowball-tuning-01a1eabe)).
The [commands](analysis/snowball-tuning-01a1eabe.sh) preserve the initial ladder.

## Observations

2026-11-17: The assigned worktree was clean. GitHub authenticated as `yonromai`.
The resource board shows the parent's separate eight-H100 correctness run;
provider inventory also shows active production jobs. Those jobs remain under
their owners. This campaign will announce its separate one-host share before
submission and verify current admission and cleanup.

2026-11-17, baseline in progress: Published source
`548b9fd14d79db11148c7d637d32d4e9c8685f16` submitted root
`/romain/knotmeter-20261117t182804z-6603a552927f` and its `/serve` child. The GPU
native attempt is `fced1e4b46787c9a` on `g5ba762`. The prepared workload was a
verified cache hit: key
`2a67e147dfbe10a76d1f901c5a3de2539298aaf1a8fdc8e8c746aa8d14aa7b59`,
archive SHA256 `ce86367e7a15a98e373d6483a5e72cae4f422f4b5eb3107f820201226c88f3ce`.
The actual deployment passed the complete shared checks before native warmup.
The check report SHA256 is
`325d7b21f1f51b10262ed03f2e0c9296b9bdc7b84f44e8534d9e217adb193477`; the fixed
provisional policy SHA256 is
`9568a18b79e1510d6ffb405c03a640e2683cb868ed363f68612153d980938375`.

Interactive priority was requested and retained in both Iris job configurations.
Iris assigned both tasks to batch. The GPU Workload had priority 1 and label
`iris-accelerator-batch`; the CPU Workload had priority 0 and label
`iris-cpu-batch`. Continuing under Iris's own priority fallback was authorized.
No scheduling setting or shared service was changed.

### Next point: reduce resident sessions

The next point will be C64 with the baseline's 8,192-token budget and all other
serving settings unchanged. Run a normal one-hour window after the baseline
releases its GPUs. Defer the initial 4,096-token diagnostic.

This choice uses a partial-window observation to choose a hypothesis. At
19:20:14 UTC, 28:41 into profiling, AgentX had 2,507 successful requests and one
`ServerDisconnectedError`. Its live telemetry showed 93.6% KV use, 9.1% actual
prefix hits versus 95.2% theoretical replay reuse, and p95 TTFT 139.881 seconds.
These are diagnostic observations during the baseline; they are not its final
throughput or latency result.

Lower load may preserve more useful cached histories and reduce the latency
tail while keeping useful completed throughput. That mechanism is a hypothesis;
this comparison will measure the load tradeoff, not prove a cause. A normal C64
window gives a direct comparison and avoids repeating startup and native warmup
for an additional short test. The earlier calibration supports C64 as a
plausible lower-load point but uses a different runtime. Keep both final normal
windows, failures, checks and full allocation costs regardless of which wins.

### Completed C128 baseline

The [baseline receipt](analysis/snowball-tuning-01a1eabe-baseline.json) points to
the frozen plan, native archive, checks, processor aggregate and allocation
timestamps. The normal AgentX result passed native submission validity, worker
metrics, both duration-coverage checks and unchanged-gang checks. All five
shared-check groups passed. The checker source remains the original
`548b9fd14d79db11148c7d637d32d4e9c8685f16`.

| Sessions | Output tokens/s/GPU | p90 TTFT | p90 completion | Profiling successes/errors | H100-hours |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 124.39607 | 103.33113 s | 297.61639 s | 5,165 / 1 | 11.368889 |

The upstream measurement span was 3,629.77401 seconds for the normal one-hour
recipe, including its drain convention. Profiling retained one
`ServerDisconnectedError` and 59 native deadline cancellations. Final warmup
exports contain 989 nonempty responses and 417 `InvalidInferenceResultError`
records out of 1,406 requests. Those errors report empty content. Earlier
progress counters reported zero warmup errors; the final exported records are
the evidence used here. The upstream window remains valid under its pinned
protocol. These warmup errors are separate from the profiling error ratio.

The GPU pod finished all containers at 19:54:57 UTC. Cost uses its actual
scheduling time through that finish: 40,928 GPU-seconds, with no unknown
allocation time. The CPU coordinator finished at 20:06:28 UTC. It published the
final result naturally after the processor had written its local aggregate.
The cause of that publication delay was not established. Thread profiling was
attempted after the coordinator had already completed; no recovery, GPU retry
or shared-service change occurred.

The local Bash wrapper returned 127 after the CLI had printed its successful
report. Its source had been edited while the command was active, so Bash later
read shifted file-tail bytes as a command. The helper now uses `exec` for the
final command. The frozen GPU and coordinator jobs both succeeded. Keep that
wrapper failure separate from their results.
