# Easy non-DNA diagnostic: completed November 16, 2026

## Decision

The final SFT checkpoint does **not** universally prefer B, and uncertainty does **not** universally produce B preference. In this small diagnostic the existing `Answer:` prefill correctly handles all easy questions under both label assignments. The cause of the VEP-specific bias remains unresolved; do not report a general broken B classifier or a demonstrated need for additional training.

## Results

| Condition | Contexts | Correct generated answers | Candidate B preferred | First-listed option preferred |
|---|---:|---:|---:|---:|
| Easy, chat + `Answer:` prefill | 24 | 24/24 | 12/24 | 12/24 |
| Easy, native short-answer chat | 24 | 21/24 | 9/24 | 9/24 (one candidate tie) |
| Unanswerable, chat + `Answer:` prefill | 8 | Not graded | 3/8 | 7/8 |

Candidate preference compares next-token A/B logprobs; generated correctness parses the complete short answer. Native candidate ranking has 21 correct, two incorrect, one exact tie; greedy decoding chooses A in the tied case, incorrectly. All 56 outputs parse as a single letter. The unknown-condition `candidate_correct` and `parsed_correct` fields in the generic JSON aggregator are zero because no answer key exists; they are **not accuracy measures** and must not be interpreted as 0% accuracy.

Every easy question is tested twice, with content assigned to opposite letters. Prefill is correct under both assignments for 12/12 questions; native is correct under both assignments for 9/12. These are 12 authored questions, not 24 independent questions or a representative benchmark.

Native errors all have B listed first, A second, with the correct content assigned to B:

| Question | Displayed options | Generated answer | Conditional P(B given A/B) |
|---|---|---|---:|
| Which planet do humans live on? | B. Earth; A. Mars | A | 48.73% |
| Which animal is a mammal? | B. Dog; A. Trout | A | 50.00% (candidate tie) |
| Which word appears in this exact text: red blue? | B. blue; A. green | A | 38.86% |

Prefill answers these same cases correctly. Native gets 12/12 A-then-B cases and 9/12 B-then-A cases correct. This suggests format/order sensitivity, but display order is balanced across questions, not independently varied within each question; the tiny sample does not establish a universal order effect.

For unknown outcomes (private fair coin/die/ball/bit), B is preferred 3/8 times, first-listed 7/8. The coin and ball controls prefer A under both content assignments; the die prefers B under both; the bit changes letters. This supports prompt-dependent fallback behavior, not a universal B fallback. These inputs explicitly state that evidence is missing, whereas VEP uncertainty could be implicit and qualitatively different. Forced choice cannot assess whether the model would appropriately abstain in ordinary conversation.

Mean first-token A/B probability mass is 96.01% for easy prefill, 85.53% for easy native and 78.55% for unknown prefill. Conditional A/B probabilities are not calibrated task confidence. No free-reasoning or tool-enabled condition was run.

## Existing VEP score analysis

- Zero-shot conditional P(B given A/B): median 62.09%, range 54.92–77.01%, mean 64.01%. B wins every context, but not with near-certain probability. Raw B-minus-A margin: mean 0.5834, SD 0.2278, range 0.1973–1.2090 logprob units.
- Two-shot conditional P(B given A/B): median 51.37%, range 36.70–69.62%. Mean raw margin 0.05875, SD 0.25584.
- Median absolute forward/reverse strand margin change: 0.1030 zero-shot and 0.1904 two-shot. Two-shot changes strict preference in 38/100 variants (including transitions to/from the one tie); zero-shot never changes preference. Variation is not proof of useful biological sensitivity.
- Two-shot agrees with the last demonstration's letter in 100/200 contexts: 40/100 when that letter is A, 60/100 when B, with one tie overall. This does not support universal last-answer copying. Demonstration order, semantic class and mapping are confounded; no demonstration-order intervention was performed.

## Provenance and execution

Frozen source, protocol and exact prompts: `a2c0615abea65a5324a3525c5cfad4d1d24f04a6`. Input SHA256 `50fe94d79c4508e79cb924fa0f3d6210b2e6c1c52e99599d214c6a5ad73ea320`. Pinned tokenizer and chat-template hashes are embedded in inputs. Same advertised checkpoint and same serving fingerprint as the 100-variant chat pilot; no independent weight checksum.

Live run: 2026-11-16 20:52:53–20:54:44 UTC, 110.85 seconds. 56 completions, 80 generated tokens, 168 matching tokenization checks, one model discovery: all 225 HTTP requests succeeded. Maximum completion latency 0.558 seconds. Fixed rank, sequential requests, one-second gaps; no latency or fingerprint guard fired. No new repeated-call stability test was included in this diagnostic; earlier pilot repeat results remain separate evidence. No active inference process remains.

Offline tokenizer preparation took 0.67 seconds, peak RSS 222,032 KiB, exit 0, with the shared lock, one-thread limits, nice and ionice. Existing-score and result summaries use bounded standard-library processing. No new dependencies, GPU deployment, VEP rerun or reserved validation use.

Public evidence: `easy_diagnostic.py`, `easy-diagnostic-protocol.md`, `easy-diagnostic-inputs.json`, `easy-existing-analysis.json`, `easy-diagnostic-scores.jsonl`, `easy-diagnostic-analysis.json`, `easy-diagnostic-summary.json`. Private wire responses and access metadata remain outside the repository.

Replay: `python easy_diagnostic.py analyze easy-diagnostic-inputs.json easy-diagnostic-scores.jsonl OUTPUT.json`; `python easy_diagnostic.py existing chat-pilot-scores.jsonl chat-pilot-inputs.json.gz OUTPUT.json`.

## Next decision

The current prefill is a viable easy-question classification interface. The VEP B preference should be described as specific to the tested prompts and task, with letter/position/semantic effects still entangled. A next small experiment could hold the VEP input fixed and independently vary option order, label symbols and DNA content; changing biology and labels together would not identify the cause. Such an expansion has not been run. Keep this experiment on the research branch, with no PR or merge into main.
