# Next-token competitors and OpenRouter compatibility diagnostic

2026-11-13. Exploratory follow-up to the protocol-v3 Mochi pilot; no panel rerun.

## Mochi: historical response audit

The attempted four-request top-20 replay stopped on its first request: the saved serving route returned HTTP 404 (no endpoint). A fresh Iris registry snapshot contained no Hero/Mochi match. No new Mochi inference succeeded. Endpoint details and raw wire responses remain private.

However, the original pilot wire responses preserved one sampled next token and its log probability for every classification request, in addition to the two requested A/B scores. Joining by `paired_request` recovers the following across 200 prompts per method (100 variants, both strands):

| Method | Sampled token | Count | Mean probability where sampled | Range where sampled |
|---|---|---:|---:|---:|
| Zero-shot | standalone space | 85 | 41.42% | 29.13–48.18% |
| Zero-shot | ` B` | 58 | 24.11% | 19.28–32.56% |
| Zero-shot | ` A` | 57 | 25.98% | 18.13–32.84% |
| Two-shot | standalone space | 155 | 66.60% | 55.26–74.81% |
| Two-shot | ` A` | 28 | 6.37% | 4.23–9.47% |
| Two-shot | `<|eot_id|>` | 17 | 17.17% | 12.96–23.04% |

These are conditional probability summaries, not full-distribution averages. The fixed sampling seed and selection of observed tokens mean sampling counts must not be read as estimates of unconditional probability mass. The saved top-logprob dictionaries contain only A/B and the sampled token; they do not identify all competitors in every prompt.

The two-shot examples coincide with strong probability on a standalone space, and sometimes an end-of-turn marker. This is evidence of a formatting/continuation competitor, not proof of why demonstrations induce it. In the 155 two-shot cases where space was sampled, its probability exceeded 50%, making it the unique most likely next token. The earlier tiny parity checks did not establish low end-of-turn probability over the full panel: these historical responses demonstrate material end-of-turn mass in at least 17 two-shot prompts.

Current scoring measures the canonical single-token paths ` A` and ` B`. It does not marginalize alternative paths such as standalone space followed by a letter. Adding space to the prompt changes conditioning and previously produced an end-of-turn problem; do not change the benchmark silently. Next live diagnostic should inspect top tokens, then conditional continuations after standalone space on the same paired prompts. No new score or post-hoc sign flip is introduced here.

## OpenRouter: bounded live capability checks

Pinned route: model `z-ai/glm-5.3`, provider `fireworks`, fallbacks disabled, `require_parameters=true`. Response model confirmed GLM-5.3 and provider Fireworks; no system fingerprint returned. This model alias is not an immutable checkpoint pin.

- Three initial requests rejected HTTP 400 because this route requires reasoning and cannot disable it. No model completion was returned.
- With that option removed, one chat request returned HTTP 200, one reasoning token, and `logprobs: null` despite `logprobs=true` and `top_logprobs=5`.
- Two raw `/completions` requests supplied a 20-base synthetic DNA prompt, `echo=true`, and `logprobs=5`, with/without `prompt_logprobs=5`. Both returned HTTP 400 from the provider reporting a missing `prompt` field. The recorded client payload includes `prompt`; the failure is somewhere along the routing/translation/provider path and cannot be localized from this response.
- One final chat control with a 128-token cap returned HTTP 200, all 128 tokens consumed by reasoning, no visible answer, and `logprobs: null`. It did not reach the answer-token boundary, so visible-answer logprob support remains inconclusive.

Seven OpenRouter requests total; two successful generations, 129 generated reasoning tokens, reported combined cost $0.00042868. No usable input-token likelihoods returned. This route is not validated for DNA scoring; these tests do not prove every OpenRouter provider lacks that capability.

## Reproduction and scope

Probe source was published before execution: initial `7f42f12f89`, reasoning correction `5b138abaa7`, final 128-token control configuration `b6f24863cd`. Source: `probe_next_tokens.py`; final control: `openrouter-output-control.json`. Private launchers inject approved credentials and route metadata at runtime. Requests use concurrency one and a 45-second per-request timeout. No GPU deployment, panel evaluation, PR, or main-branch merge.

Supporting `sampled-next-token-audit.json` summarizes the historical responses; `classification-answer-probabilities.json` gives full A/B summaries. Raw private responses, route snapshot, and request/error evidence are retained locally.
