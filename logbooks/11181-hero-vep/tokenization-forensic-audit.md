# Tokenization and end-token forensic audit

2026-11-13. Read-only audit of historical wire evidence and the serving-build sampler source. No new model calls, weights downloads, GPU deployment, benchmark rerun, or score change.

## What the saved responses establish

For the one matched v3 pilot query, the space-conditioned prompt is exactly the colon-ending token sequence plus token 220. The prompt has one BOS and no end-of-text (128001) or end-of-turn (128009) token. Server prompt-token usage equals the submitted integer-token count; `add_special_tokens=false`. The returned generated text is literally `<|end_of_text|>`, with returned token ID 128001. This rules out an accidental appended end marker in the client input or simply confusing the decoded identity of this output token.

| Context | P(space after colon) | P(end-of-text after space) | P(bare A/B after space) | P(A/B after newline) |
|---|---:|---:|---:|---:|
| Zero-shot VEP | 35.91% | 99.38% | 3.63e-7 | 63.81% |
| Two-shot VEP | 71.27% | 99.66% | 8.51e-8 | 32.70% |
| Simple alphabet control | 1.38% | 58.17% | 3.65e-7 | 61.52% |

The VEP colon prefixes match the frozen v3 query token-for-token, as recorded in `historical-space-continuation-audit.json`. The alphabet control asks for A or B without biological content. All three boundary variants within each context returned the same serving fingerprint. These are historical one-context diagnostics, not estimates for the full panel.

## Correction: end-token preference is not observed stopping

Previous descriptions of the space branch as leading to termination were too strong. The actual requests used `ignore_eos=true`, `max_tokens=1`, and greedy sampling (`temperature=0`). Every response reports `finish_reason=length`, not a stop. They prove an end-of-text token preference and actual one-token selection, but not that normal generation would stop on this token. The deployed generation-config stop IDs have not been recovered. No two-token decode from the original colon prefix was run. Stitching two separate prefill requests is not a substitute for that test.

## How logprobs were produced

The serving fingerprint identifies Marin vLLM source revision `3004cd8f163299116b96d1b324cfff418c387388`. Its `vllm/v1/sample/sampler.py` defaults to `raw_logprobs`: it computes a float32 log-softmax across the vocabulary before penalties, temperature, and sampling processors. `gather_specific_token_logprobs` then selects the requested token IDs and sampled token; it does not renormalize over those selected tokens. Source lines 75–118, 165–225, and 305–307 were inspected.

Source: https://github.com/marin-community/vllm/blob/3004cd8f163299116b96d1b324cfff418c387388/vllm/v1/sample/sampler.py

A deployment can override the logprob mode, so the default alone is not proof of live configuration. However, the historical greedy responses contain nondegenerate probabilities, and the matched zero-shot colon A/B values agree with the later temperature-one v3 records. The selected-token probabilities sum to less than one, consistent with full-vocabulary probabilities. No shortlist-renormalization bug was found. An engine/model implementation defect remains unexcluded without independent logits.

## Interpretation and next decisive checks

Canonical tokenization represents `Answer: A` with a leading-space letter token. Supplying a standalone space first instead forces a distinct token history. A plausible explanation is that this history is uncommon before letters in tokenized training data and associated with trailing whitespace or document endings. We have not measured the training corpus or inspected matched base/SFT logits, so this is a hypothesis, not a discovered training defect.

The space branch is not simply a hidden way to output the same answer: in the inspected contexts, bare-letter continuations are negligible. Newline plus bare-letter answers is a possible separately specified prompt experiment, not an automatic correction. Canonical v3 scoring remains unchanged.

A fresh registry snapshot still had no Hero/Mochi route. Once the same checkpoint is available, the bounded decisive comparison is: (1) greedy two-token generation from the colon prefix, (2) one-token generation after explicit space, (3) teacher-forced scoring of both token paths, and (4) independent full-vocabulary logits from a direct forward pass, with tokenizer/config and weights pinned. Inspect EOS/stop settings and report both token IDs and decoded text. Test the matched VEP query and alphabet control first. Any separate compute deployment requires its own resource decision; none was started here.

Supporting evidence: `tokenization-forensic-audit.json`. Private raw wire source SHA256: `ba4f2e335fe549bef05e2244ce2915f9af5f80577c57da17734d0d643d5fdf7f`. Endpoint details remain private.
