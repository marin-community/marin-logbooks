# Pinned tokenizer reproduction and SFT formatting contract

2026-11-13. Offline diagnostic; no new model inference or deployment.

## Exact tokenizer reproduction

The final SFT export record in Marin #8089 recommends `marin-community/marin-tokenizer-tools` at revision `3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09`, with an explicit Marin chat template and EOS override. Downloaded only its tokenizer JSON (17,209,859 bytes) and small tokenizer configuration, no model weights. Tokenizer SHA256: `6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45`.

The offline tokenizer reproduced all nine saved `/tokenize` responses from the boundary diagnostic exactly, including BOS behavior. This establishes tokenization agreement on these inputs, not byte identity of the entire deployed tokenizer artifact.

| Text | Tokens, excluding automatic BOS |
|---|---|
| `Answer:` | 16533, 25 |
| `Answer: ` | 16533, 25, 220 |
| `Answer: A` | 16533, 25, 362 |
| `Answer: B` | 16533, 25, 426 |
| `Answer:\nA` | 16533, 512, 32 |

Both `[16533,25,362]` and `[16533,25,220,32]` decode to `Answer: A`. Re-encoding either decoded string yields `[16533,25,362]`. Thus `[space][A]` is a noncanonical tokenization of the completed string. The checkpoint's tokenizer is not mis-tokenizing the saved prompts; adding a standalone-space token really changes the conditional token history.

Code frozen before execution: `fea245f07d`, `audit_pinned_tokenizer.py`. Input: private historical EOS diagnostic wire file; output: `pinned-tokenizer-audit.json`. Local run 2026-11-13 21:32:04–21:32:05 UTC, exit 0, 0.66 seconds, peak RSS 217,284 KiB. Nonblocking shared heavy-work lock, single-thread environment, nice/ionice, resource thresholds checked. Download and resource use remained bounded. The private pinned tokenizer file is retained as reproduction evidence.

## Two distinct special-token contracts

The downloaded tokenizer configuration sets `eos_token` to `<|end_of_text|>` (128001) and contains no chat template. The export record explicitly instructs serving to supply `MARIN_CHAT_TEMPLATE` and set `eos_token_id=128009` (`<|eot_id|>`). It also states the weights export has neither tokenizer nor generation_config.json. This is an explicit configuration requirement, not evidence the previous deployment failed to follow it. Its effective stop configuration remains unverified.

This strengthens the earlier qualification: a high raw probability of token 128001 does not establish that a correctly configured chat deployment would stop on it. Historical probes used ignore_eos=true and ended at the token limit.

Source provenance: https://github.com/marin-community/marin/issues/8089
Tokenizer metadata: https://huggingface.co/marin-community/marin-tokenizer-tools/blob/3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09/tokenizer_config.json

## Formatting hypothesis, not a root-cause finding

The inspected Marin template emits role headers and opens assistant generation with an assistant header followed by newline. Assistant messages are trimmed and closed with `<|eot_id|>`. Our frozen VEP protocol supplies raw text ending in `Answer:` without these chat headers. That is consistent with a conventional raw multiple-choice likelihood benchmark, but is a distinct format from the recommended SFT chat interface.

The current SFT launcher also initializes the end-of-turn LM-head row from the end-of-text row at SFT start. This is a documented initialization; it does not imply their final logits remain equal and does not explain the space effect by itself.

Source inspected at checkout revision `a922ffc740488a3aa82c8ac3f8cf8e9dd8d77ba7`:
- `lib/marin/src/marin/datakit/chat_template.py`
- `experiments/grug/moe_hero_ep/launch_hero_sft.py`, special-token anchors and initialization.
These sources describe the current recipe/template; the exact historical SFT data rendering has not been independently replayed.

Training on canonical tokenizations and formatting sensitivity are plausible contributors. No corpus-frequency measurement or independent forward-pass result establishes the cause of the extreme end-of-text preference. A serving-model implementation defect remains possible.

## Next discriminating test

The refreshed serving registry still had zero Hero/Mochi matches. No new inference was attempted. Once the same checkpoint is available, compare raw colon, explicit standalone space, newline, and recommended chat-assistant prefix on the same VEP query and alphabet control. Capture actual stop settings; compare teacher-forced and two-token decode probabilities; compare independent logits if they disagree. Treat chat formatting as a separately declared experimental condition and preserve the existing v3 scores. No model weights were downloaded and no new GPU resources were started.
