# A11 native identity screen

A11 gives the target name in 31/52 thinking-ON final answers and credits the Marin Community as developer or trainer in 37/52. Four ON outputs never emit an end-think token, so they receive no final-answer credit. All 48 closed core outputs have no response-planning text after that boundary. Identity errors and third-person attribution rewrites remain.

A11 continues training from the published **Marin-Snowball-67B-A2B-v0.1** checkpoint (`adc0078a`) for 72 updates with replay data and identity answers teaching the name **Marin's Latest MoE** and development/training by the **Marin Community**. Identity training examples have no reasoning. This screen contains 52 identity prompts and 14 third-person controls in each mode, with no injected identity system prompt. All 132 actual contexts and outputs were read; no second generation or forced thinking prefill was used.

| View | Mode | Target name /52 | MC developer/trainer /52 | OA developer/trainer /52 | Foreign self-claim /52 | Closed ON core /52 | Post-boundary planning |
|---|---|---:|---:|---:|---:|---:|---:|
| Native final | OFF | 27 | 39 | 0 | 1 | — | — |
| Native final | ON | 31 | 37 | 0 | 2 | 48 | 0/48 |

Native ON finals start after the first **generated** end-think token (ID 128003); prompt-owned markers cannot supply that boundary. MC means Marin Community; OA means Open Athena. Maintainers, contributors and model publishers alone receive no developer/trainer credit. ON name credit includes one contextual own-project mention (046); 034 credits the name but gives no creator-role credit for a publisher citation. Spanish, Chinese and Hindi role statements are read by meaning.

The unclosed core outputs are 023, 070, 073 and 074. Output 023 reaches the length limit while repeating identity text in reasoning; the other three stop after an identity recital without an end-think token. Correct names inside these spans receive no native-final credit. All 14 ON controls close. OFF outputs all stop normally; there are no OFF length truncations or execution errors.

The OFF foreign claim is 054, which says Microsoft Research trained the model. ON 035 identifies the current model as Open Athena's Latest MoE; ON 041 titles its own hypothetical README “ChatGPT.” The ON reasoning contains literal system/policy/instruction/prompt references in 34/52 core outputs, including unclosed 023; 18 mention policy. Sixteen attribute identity content to an absent or generic system message, including 004's invented contradictory system text. These reasoning errors are separate from post-boundary answer planning.

| Third-person controls /14 | OFF | ON |
|---|---:|---:|
| Attribution rewrites | 1 | 1 |
| Canned own-identity nonanswers | 7 | 0 |
| Other substantive answers | 6 | 13 |

OFF control 010 falsely says ChatGPT was developed by Open Athena/Stanford. ON control 005 describes GPT-4 as “trained by the Marin Community” with training data derived from DeepSeek, Kimi and GLM. Control 014's ON final discusses another chatbot's possible system prompt, which is relevant to its question; its reasoning invents an additional user request for a boxed answer. Own-name 034 also invents a release date, and 047 offers unverified contact details. The screen is not a general factuality certification.

Separately, decoding the **whole generated ID sequence** with special tokens hidden and passing it through the historical reasoning/answer parser yields ON name 44/52, creator 42/52 and visible response planning 43/52; ON controls show planning in 14/14. The parser cannot recover the hidden boundary markers, and provider reasoning fields are empty, so this view includes reasoning in the apparent answer. Its name and creator counts use the same identity and role criteria across that entire apparent answer, including reasoning-only statements. Visible planning means response-planning text appears there while the parsed reasoning field is empty. It does not overwrite the native-final table or any historical screen.

[All132 ledger](A11-native-role-aware-ledger-marten-r1.json) (`9246c688bf2e551bd8989d9b741aecef340639a9563e440a13a317d581dca052`), [native/legacy decoded views](../A11-native-and-legacy-derived-views-marten-r1.json), [actual launch/readiness provenance](actual-native-provenance-joins-marten-r1.json), and [all135 local artifact hashes](FULLGET135-native-output-index-margay-r1.json) retain the raw fields and IDs. CPU, live tokenization and returned prompt IDs agree on all 132 requests. This is one checkpoint and one deterministic diagnostic; BFCL and maths are separate. No release acceptance or isolated causal effect follows from these counts.
