# Matched raw versus chat formatting probe

2026-11-16. Six sequential completion requests, one autosomal variant, forward strand, zero-shot and two-shot. This is a diagnostic of output formatting, not a VEP accuracy estimate or a benchmark rerun.

## Inputs and controls

Variant `13:107866379:T:C`, frozen reversed mapping (A benign, B pathogenic), same query and demonstrations as the v3 pilot. The endpoint reports the same final SFT checkpoint export artifact as before; its serving fingerprint has changed. Checkpoint identity is based on advertised model metadata, not an independent weight checksum. All six arms were run on the current build, including fresh raw controls.

- **Raw:** original v3 text ending in `Answer:`; candidates ` A` and ` B`.
- **Chat prefill:** the same instructions, inline examples, and query are placed in one user message, removing its final newline plus `Answer:`. The pinned Marin template adds user/assistant role headers; assistant output is prefilled with `Answer:`. Candidates remain ` A` and ` B`.
- **Native chat:** the same user message and template, with no assistant prefill; candidates are bare `A` and `B` after the assistant-header newline.

No explicit thinking-mode override, tools, or additional system instructions. Few-shot demonstrations remain inline in the user message; they are not converted to separate chat turns. One BOS per input. The official template snapshot comes from Marin revision a922ffc740488a3aa82c8ac3f8cf8e9dd8d77ba7. Generation annotations, which emit no text, are removed by the offline renderer; no completed assistant messages are rendered in this probe. Exact rendered text and integer-token request payloads are saved in `chat-probe-inputs.json`.

Before each completion, the server tokenizer was checked on the prefix and each complete prefix-plus-answer: all 18 checks exactly matched the pinned tokenizer. Each answer adds exactly one token. Raw and chat-prefill candidate IDs are 362/426; native-chat IDs are 32/33.

## Results

Probabilities below are first-token full-vocabulary probabilities, not normalized over A/B.

| Shot count | Format | P(A) | P(B) | P(A or B) | Greedy two-token output |
|---|---|---:|---:|---:|---|
| Zero | Raw | 28.00% | 27.14% | 55.14% | space + `<\|end_of_text\|>` |
| Zero | Chat prefill | 27.30% | 67.70% | 95.01% | ` B` + `<\|eot_id\|>` |
| Zero | Native chat | 22.79% | 4.56% | 27.35% | `1.` |
| Two | Raw | 7.54% | 5.69% | 13.23% | space + `<\|end_of_text\|>` |
| Two | Chat prefill | 47.29% | 51.33% | 98.61% | ` B` + `<\|eot_id\|>` |
| Two | Native chat | 25.70% | 5.22% | 30.92% | `1.` |

Actual two-token decoding now confirms the raw space-to-end-of-text branch, rather than stitching separate prefill probes. On the second decode step, P(end-of-text given space) was 99.29% zero-shot and 99.69% two-shot. First-token standalone-space probability fell from 34.58% to 0.22% with zero-shot chat prefill, and from 66.12% to 0.079% with two-shot chat prefill.

All requests deliberately used ignore_eos=true, skip_special_tokens=false, temperature=0, max_tokens=2. Therefore finish_reason=length does not test natural stop behavior. Requested A/B/space/end-token IDs plus the sampled token are returned; this is not a complete top-token distribution. The native-chat output is only `1.`: it may be starting a numbered explanation, but two tokens cannot establish what would follow.

## Interpretation

Chat formatting with an explicit answer prefill substantially improves probability mass on the requested answer format for this one query. Simply adding chat headers without an answer prefill does not have the same effect. This supports a formatting-sensitive behavior in the served model and motivates a separately frozen chat-prefill classification condition. It does not prove better variant discrimination, an SFT training cause, or absence of serving defects. No post-hoc score flips, replacement of v3 results, or full-panel inference occurred.

The next useful check would replicate the raw versus chat-prefill comparison on a small balanced set of variants and both answer mappings, with repeats to assess current serving stability, before considering a panel rerun. That additional run has not been launched.

## Execution record

Source frozen and published before execution: `f3ed9b87a3280cbf65e7bfecba66627653dcf1ec`, `prepare_chat_probe.py`, plus `templates/marin-chat-pinned.jinja`. Private launcher injects existing credentials, endpoint and model at runtime; it pins the same serving rank as the original pilot and runs sequentially with one-second gaps. Endpoint/access details and raw responses remain private.

Offline preparation: 2026-11-16 20:08:04–20:08:05 UTC, exit 0, 0.72 seconds, peak RSS 222,176 KiB, shared heavy-work lock and thread/resource limits applied. Live work: 25 HTTP requests total (one model discovery, 18 tokenizations, six completions), all successful, 12.79 seconds, 12 generated tokens. Budget: 25 HTTP requests and 300 seconds, 45-second per-request timeout. Inputs SHA256 `ecbfb6dd002fb5cb4bbfe2a050753dc0f59a87e969d8d7a590f87493b99719ab`. Sanitized evidence: `chat-probe-summary.json`. No new deployment, PR, main merge, or active worker remains.
