# Native identity screen: v0.1 and A3

Reading generated think-boundary token IDs separates reasoning from answers in **52/52 v0.1 and 50/52 A3 core thinking-ON responses**. None of those post-think answers contains response planning. A3 still has two unclosed core responses, invented instructions in reasoning, and eleven canned identity nonanswers in its thinking-OFF third-person controls. These results do not establish release suitability.

v0.1 is the unmodified source model. A3 fine-tunes it for 36 updates with identity examples rendered thinking-OFF. All 264 responses were read with their role context, reasoning and answer. Each model receives the same 52 identity prompts and 14 third-person controls in each mode. Requests preserve the original template and decoding settings, adding only special-token visibility and returned token IDs. There is no forced think-start prefill. All 270 locally mirrored artifact hashes pass independent checks: 132 per-response files plus three metadata/consolidated files per model. Prompt token IDs from the local tokenizer, live server tokenization and completion response agree. The [row ledger](v01-A3-native-role-aware-ledger-marten-r1.json) links the raw outputs and derived views.

| Model / mode | Own Marin name /52 | Own MC creator /52 | Own OA creator /52 | Foreign self-claim /52 | Generated open+close /52 | Post-end planning / available finals | Dev14 rewrite / canned / other |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.1 OFF | 0 | 0 | 0 | 41 | — | — | 0 / 0 / 0 |
| v0.1 ON | 0 | 0 | 0 | 46 | 52 | 0 / 52 | 0 / 0 / 1 |
| A3 OFF | 30 | 34 | 0 | 0 | — | — | 0 / 11 / 0 |
| A3 ON | 31 | 44 | 0 | 0 | 50 | 0 / 50 | 0 / 0 / 0 |

Name credit accepts Marin's Latest MoE or Marin Snowball; all credited names here are the former. MC means Marin Community; OA means Open Athena. Creator credit requires development/training, excluding maintenance alone. Foreign counts definite own other-model/lab identity, excluding denials, third-person examples and hypothetical systems. A valid ON span has exactly one generated start-think ID (128002), followed by exactly one generated end-think ID (128003); canonical means start at generated position zero. ON identity credit uses only the answer after that end. Counts retain the full 52-prompt denominator, including rows without an available final. “Dev14” denotes the fourteen third-person controls. Its columns count false creator/product attribution, own-identity canned nonanswers, and other nonanswers; remaining controls answer substantively. Other factual errors are separate.

A3 ON identity prompts `id296-073/074` both stop after a generated start-think token without an end-think token. Their correct Marin name and creator remain in unclosed reasoning and receive no native-final credit. Every closed core span in both models starts canonically; there are no duplicate, reversed or extra header/control markers. All fourteen A3 ON controls close normally with no post-end planning. Thirteen v0.1 ON controls close; `ctl296-014` reaches its length limit in repeated reasoning without a final. A3 OFF `id296-041` also reaches its length limit in repetition. Every other response stops normally. The `id296-` and `ctl296-` prefixes distinguish identity and third-person prompts; their numeric suffixes below refer to these fixed IDs.

Reasoning still invents authority. Among the 52 ON identity prompts, literal system/policy/instruction/prompt references occur in v0.1 **41** and A3 **35** responses; literal policy references occur in **28/19**. Role-aware reading identifies **10/17 invented own-system attributions**, respectively: for example, A3 ON071 says the system supplies its Marin identity although the actual system says only “You are a helpful assistant.” Actual generic system quotations and hypothetical third-party prompts are excluded. A3 additionally invents a user identity instruction in ON042 and explicitly quotes nonexistent policy text in ON016/017/040. These categories can overlap; the quoted-policy count is narrower than generic or nonquoted policy invocations. No such own-system attribution or explicit fabricated policy quotation appears in either model's fourteen ON controls.

A3 ON calls itself “Marv” in004/005 and signs “Marvin” in038; these receive no accepted-name credit. Its ON control005 correctly attributes GPT-4 to OpenAI but inserts unsupported DeepSeek/Kimi/GLM training provenance. ON control014 imagines a friend's chatbot system saying ChatGPT was trained by Marin; this stays a hypothetical example, not a definite rewrite of the real ChatGPT service. Other unverified release dates, cutoffs and product details remain in the raw record. This is not a full factual audit.

## Separate legacy-derived proxy

Decoding the **whole** generated token sequence with special tokens hidden, while retaining the empty provider reasoning field, yields the following proxy for an unsplit consumer. Accepted names and creators stay the same; broad credit also permits affirmative own identity in planning. This leaves the earlier [A3 unchanged-parser table](../ablation-identity132-r1/reading-marten-r1.md) and [v0.1 table](../identity-v01-baseline-r1/read-v01-marten-r1.md) frozen; their answer fields contained reasoning without a token-based split.

| Model / ON | Broad own-name credit /52 | Broad creator credit /52 | Empty provider reasoning + visible planning /52 | Same planning in Dev14 |
|---|---:|---:|---:|---:|
| v0.1 | 0 | 0 | 52 | 14 |
| A3 | 40 | 46 | 42 | 14 |

The proxy permits unambiguous identity in planning, whereas native-final credit does not. It also exposes reasoning as visible answer text. The new special-token requests need not reproduce earlier completions bit-for-bit. One deterministic-decoding run per model supports these descriptive counts, without a training-effect or universal parser-causality claim.
