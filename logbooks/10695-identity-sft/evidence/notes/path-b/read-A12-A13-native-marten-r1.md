# A12 and A13 native identity read

All 132 contexts, reasoning spans and final or unclosed outputs were read for each model. A12 uses the first 18 updates of A8a, recovered by export only. A13 duplicates each thinking-ON replay occurrence in the tool_first_call, tool_call_after_result and tool_reply families immediately after itself at weight 1, then repacks to 36 updates. The [recipe and analysis registration](bfcl-noise-r1/A12-A13-science-and-analysis-preregistration-marten-r1.json) and [realized doses](ablation-a12-a13-r1/data-counts.json) record the resulting cutoff and replay differences. Both screens retain the original native template, without forced thinking prefill. The canonical name is Marin’s Latest MoE; unambiguous translations receive name credit.

| Model / mode | Name in final /52 | Marin Community creator/trainer /52 | OpenAthena/Stanford creator /52 | Foreign self/model-lineage claim /52 |
|---|---:|---:|---:|---:|
| A12 OFF | 31 | 35 | 1 | 3 |
| A12 ON | 30 | 33 | 0 | 1 |
| A13 OFF | 29 | 30 | 1 | 1 |
| A13 ON | 35 | 32 | 0 | 0 |

The columns overlap. Maintainer-only and contribution-only answers do not receive creator credit. Name absence can reflect the prompt mix or an underanswer; these totals are not a requirement that every core prompt request both name and creator.

A12 has seven ON core rows without an end-think token: 016, 017, 047, 054, 067, 070 and 072. They receive no native-final credit, including correct identity recitals on 067/070. Unclosed row054 instead claims DeepSeek-V3 lineage; it is retained separately from the one foreign native-final claim, row039 signing itself ChatGPT. A12 control014 has two end-think tokens and a repeated name recital. Thus A12 has seven core boundary failures and one control boundary failure.

A13 has a first end-think token on all 52 ON core rows, but row063 contains a second end-think token inside a repeated answer. All semantic columns use the complete text after the first END, including the repeated text and extra marker: row063 receives name credit, no creator credit and no foreign-claim flag. It remains explicitly marked as malformed: 51/52 core rows have a canonical single span. No marker was deleted to obtain a clean result. Both models have zero post-END response-planning rows; all 132 responses per model stop normally, without reaching the length limit. Normal stop does not establish that thinking closed.

| Third-person controls | Canned self-identity nonanswers OFF /14 | Canned ON /14 | Final creator rewrites OFF / ON |
|---|---:|---:|---:|
| A12 | 11 | 7 | 0 / 0 |
| A13 | 11 | 1 | 0 / 0 |

Canned nonanswers substitute the model’s own identity recital for the requested third-person answer. Creator rewrites assign the wrong creator to the third-person subject; a self-recital alone does not count as that assignment. A13 reasoning invents supplied system text on core023 (its own name), 040 (a model-card sentence) and 042 (its training-data sources), although these requests contain no system role. Row043 drafts an Anthropic self-training claim, then corrects it; its final credits Marin Community. Control014 invents a step-by-step/boxed user instruction. Control005 transfers an unsupported training-data story onto GPT-4, without changing its stated creator to Marin Community. These are retained concerns despite the zero final creator-rewrite count. This read does not certify general factual accuracy.

For ON outputs, the separately labelled whole-generated-ID, skip-special-token proxy includes reasoning and unclosed text: A12 has 35/52 names and 42/52 Marin Community creators; A13 has 51/52 and 39/52. These count affirmative self-profile statements, including drafted answers inside reasoning; a denied identity or a mention about another model alone does not earn credit. A later correction does not erase the retained draft from this proxy. Visible response planning appears inside reasoning on one A12 core/six controls and seven A13 core/13 controls. Those proxy counts do not replace native-final counts or raw provider fields.

Evidence: [A12 all-132 ledger](native-boundary-A12-r1/A12-native-role-aware-ledger-marten-r1.json), [A13 all-132 ledger](native-boundary-A13-r1/A13-native-role-aware-ledger-marten-r1.json). Each ledger binds all 135 local artifact hashes, original context/request/native-prefix joins, and the separate decoded views. No extra generation, historical-parser workflow, score repair or raw-token edit was performed. One deterministic screen per checkpoint does not establish a causal training effect or a release decision.
