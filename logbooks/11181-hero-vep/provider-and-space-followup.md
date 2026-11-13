# GLM providers and continuation after standalone space

2026-11-13, exploratory diagnostics only. No benchmark scores changed.

## Mochi: matching historical evidence

A refreshed Iris registry contains no Hero/Mochi matching route. No new Mochi inference was attempted. Existing private `v2/eos-wire.jsonl` contains a directly relevant probe: for both zero-shot and two-shot, the colon-ending prompt's full token sequence exactly matches the v3 pilot forward query for variant 13:107866379:T:C, reversed mapping. The historical space-ending request is exactly that sequence plus token 220 (standalone space).

| Matched context | P(end-of-text 128001 given space) | P(bare A or B given space) |
|---|---:|---:|
| Zero-shot | 0.9937940944 | 3.63486e-7 |
| Two-shot | 0.9966006595 | 8.51350e-8 |

This token is end-of-text, distinct from the end-of-turn marker seen in other historical responses. The saved requests establish that, on this one matched variant, the standalone-space branch overwhelmingly favors termination rather than a bare-letter answer. It is not a fresh result or a claim about every query. Supporting token-identity audit: `historical-space-continuation-audit.json`. Raw wire evidence remains private. The existing canonical leading-space A/B scoring protocol is unchanged.

## GLM-5.3 through three additional OpenRouter providers

Source committed before execution: `76e359e3ea`, `probe_glm_providers.py`. Six sequential requests, no retries, 45-second timeout per request, at most 512 generated tokens per chat call and one per DNA echo call. Each pins its provider with fallbacks disabled and require_parameters=true. The model alias is z-ai/glm-5.3, not an immutable checkpoint pin. All three chat responses identify the requested provider and return visible answer A.

| Provider | Output-token logprobs | DNA prompt logprobs |
|---|---|---|
| Together | Returned, including final A and top-five alternatives | HTTP 400, missing prompt |
| Parasail FP8 | Null despite requested logprobs | HTTP 400, missing prompt |
| Novita FP8 | Null despite requested logprobs | HTTP 400, completions endpoint unsupported |

Together's response contains six logprob entries including trailing reasoning, a thinking delimiter, final A, and a special turn token. It is not aligned solely to the visible content string, nor is it a complete trace of all 67 generated tokens. Its final A logprob is -0.000038384653. This proves output-logprob availability in this control, not arbitrary candidate-token availability or comparable pre-reasoning classification scoring. B was absent from the final-answer top-five alternatives.

The Parasail validation error includes the upstream request shape: messages contains the submitted DNA while prompt is missing. This supports a routing/translation incompatibility for this raw completions request. Together reports a missing prompt too, but does not expose its upstream payload. Novita explicitly does not support the completions endpoint. None of these requests validates teacher-forced DNA likelihood scoring. No conclusion is drawn about untested providers or direct-provider APIs.

Successful generations used 67 + 85 + 148 = 300 tokens. Total reported cost: $0.00101672. Sanitized evidence is in `glm-provider-summary.json`; full requests and responses remain private. No deployment, full panel, PR, or merge to main.
