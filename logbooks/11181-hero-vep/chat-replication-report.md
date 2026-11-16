# Four-variant raw versus chat-prefill replication

2026-11-16. Small formatting and mapping diagnostic; no full-panel rerun or replacement of the v3 benchmark.

## Scope and protocol

Select the first two groups in the frozen pilot ordering (893 and 732), and the first control and pathogenic variant in each: 13:107866379:T:C (control), 13:113118831:T:G (pathogenic), 11:112025527:G:C (control), 11:533875:G:T (pathogenic). Forward strand only, no chrX. Selection is defined before inference, not based on the model scores.

Each variant is evaluated in raw and chat-prefill format, with zero or two inline demonstrations, under both mappings: primary A=pathogenic/B=benign and reversed A=benign/B=pathogenic. The demonstrations' answers follow the chosen mapping. This yields 32 primary calls; the first variant's eight cases are repeated at the end for stability, giving 40 completions. This diagnostic explicitly uses both mappings and does not replace the single seeded mapping in the benchmark.

Use the previously frozen templates, candidate tokens ` A`=362 / ` B`=426, no thinking override, no tools, one serving rank, sequential calls and one-second gaps. Native chat without prefill is excluded from this replication. Each call requests two greedy tokens with ignore_eos=true and skip_special_tokens=false. End-token output does not establish natural stopping behavior. The same advertised final SFT checkpoint artifact is verified at discovery; weights are not independently checksummed.

## Answer-format probability

Each row below averages eight primary requests (four variants × two mappings). Repeats are excluded.

| Method | Format | Mean P(A or B) | Range |
|---|---|---:|---:|
| Zero-shot | Raw | 55.39% | 49.81–58.70% |
| Zero-shot | Chat prefill | 94.47% | 93.44–95.01% |
| Two-shot | Raw | 9.80% | 8.01–13.44% |
| Two-shot | Chat prefill | 98.27% | 97.79–98.61% |

Raw zero-shot generated space then end-of-text in five cases and A then two newlines in three. Raw two-shot generated space then end-of-text in all eight. Chat-prefill zero-shot generated B then end-of-turn in all eight; two-shot generated B then end-of-turn in five, A then end-of-turn in three.

## Mapping dependence

Zero-shot chat-prefill prefers literal B on all four variants under both mappings. Consequently all four predicted semantic classes flip when the label mapping is swapped. High probability on valid answer tokens is not evidence of biological discrimination.

Two-shot chat-prefill has three of four semantic predictions stable across mappings. The remaining control (13:107866379:T:C) switches from benign under the primary mapping to pathogenic under the reversed mapping; its pathogenic-minus-benign logprob margins are -0.01171875 and +0.08203125. The other stable calls include one correctly preferred pathogenic variant, one correctly preferred control, and one pathogenic variant called benign. This tiny set is diagnostic, not an accuracy estimate.

The raw baseline is also mapping-sensitive: one zero-shot variant changes sign; three two-shot variants change sign, and the fourth changes from a positive margin to an exact tie. Treat the tie as unresolved, not as evidence of an unambiguous benign prediction. Per-variant margins are saved in `chat-replication-summary.json`.

## Stability and execution

All eight end-of-run repeated cases matched their original A/B logprobs exactly (maximum absolute candidate or margin difference 0.0), with identical generated token sequences. The serving fingerprint remained constant during the run. This is bounded repeatability evidence, not a guarantee under other shared-service loads.

All 96 live prefix/full-candidate tokenization checks passed. Total 137 successful HTTP requests: one model discovery, 96 tokenizations, 40 completions. 80 generated tokens, 74.93 seconds. Source frozen before execution: `81367cfc023cde914ad280a3dfb9ffac8f850812`, `replicate_chat_probe.py` and `prepare_chat_probe.py`. Inputs SHA256: `c9cf3f0c7ee759c4ef8903d24dcce9847c1c953f4e951abdf89cf560b5d1c733`.

Offline preparation ran 20:14:35–20:14:36 UTC, exit 0, 0.83 seconds, peak RSS 222,048 KiB; resource thresholds, nonblocking heavy-work lock and single-thread/priority settings applied. Live budget: 137 HTTP requests, 600 seconds, 45-second request timeout. Private launcher and raw wire evidence retain authentication/routing context outside the public repository. Sanitized inputs and results are published alongside this report.

## Decision

Chat-prefill consistently improves answer-format compliance, but zero-shot B preference and residual few-shot mapping sensitivity remain. Do not treat this as established VEP improvement or automatically scale to a full benchmark. Preserve the existing v3 results. A useful next diagnostic would isolate label/order biases and compare task-relevant versus uninformative sequence inputs under a predeclared protocol. No further run, deployment, PR, or merge was performed; all task-owned inference processes exited.
