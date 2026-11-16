# Easy non-DNA answer-format diagnostic

Approved 2026-11-16. Experimental follow-up to the 100-variant chat pilot; research branch only, no PR or main merge. Question: does B preference occur even when the answer is obvious, or primarily under missing evidence? No VEP validation or new training.

## Frozen design

Twelve authored easy questions: four arithmetic, four facts, four text operations. Each uses both assignments of correct/incorrect content to A/B. Compare the pinned Marin chat template with assistant `Answer:` prefill against the same user message without prefill. Native chat is rendered explicitly through completions, without an additional server-side template; it asks for a single letter and allows up to 16 generated tokens. This is not an unrestricted reasoning test.

Four deliberately unanswerable private-random-outcome questions get both assignments, prefill only. These have no ground-truth answer and are not graded for accuracy. Forced choice here is an uncertainty control, not a test of appropriate abstention.

Within each question category, half the questions list A then B, half B then A. Order is fixed per question across assignments/formats. This counterbalances letter versus last-option preference across questions; it is not a fully crossed within-question order experiment.

Exact question strings, mappings, list order, rendered prompts and token IDs are in `easy-diagnostic-inputs.json`; source is `easy_diagnostic.py`. User template:

```text
Choose one of the two options. Reply with only the letter A or B.

{question}
{first_displayed_letter}. {its_option_text}
{second_displayed_letter}. {its_option_text}
```

Prefill candidates are canonical single tokens ` A`/` B` (362/426). Native first-token candidates are `A`/`B` (32/33); also retain the entire short generated answer and parse only a lone A/B after trimming whitespace and terminal markers. Candidate likelihood ranking and actual generated-answer correctness are separate metrics. First-token A/B mass may be low if the model starts with reasoning or whitespace.

## Budget and guards

56 completion requests: 48 easy (12 x 2 assignments x 2 formats), 8 unknown (4 x 2 assignments). 168 live tokenization checks plus one model discovery: maximum 225 HTTP requests, 416 generated tokens, 600 seconds. One request in flight, fixed rank, one-second completion gap; stop on error, changed serving fingerprint, token mismatch or completion latency over 10 seconds. Greedy decoding, no constrained vocabulary, explicit EOT/EOS stopping. Verify advertised checkpoint matches the previous pilot; preserve metadata privately. No automatic retry or expansion.

## Existing evidence analysis

Before inference, summarize raw B-minus-A logprob margins, conditional P(B|A or B), strand changes and last-demonstration answer agreement. Existing prompts keep demonstration order and option display order fixed; these observations cannot identify a causal demonstration/order effect. Conditional A/B probability is not calibrated biological confidence.

## Decision

Report paired accuracy under both mappings, native output compliance, A/B mass and letter/last-position preferences separately. Correct easy answers but biased unknown answers supports (without proving) uncertainty-related fallback. Easy failures isolated to prefill suggest format sensitivity; failures in both formats justify broader investigation. Keep all findings exploratory and distinguish this short-response probe from free-reasoning ability.
