# M51 native identity read

All 132 contexts, reasoning spans and final answers were read. M51 is the linear merge with 49% A3 and 51% A8a weights. The canonical profile name is Marin’s Latest MoE; unambiguous translations receive name credit. This screen uses the original native template, without forced thinking prefill. All 66 thinking-ON outputs generate one leading start-think token and one end-think token; all 132 outputs stop normally, with no length limits reached or extra control tokens.

| Core identity rows | Canonical name in final /52 | Marin Community creator/trainer in final /52 | Foreign self/creator claim /52 | Post-end-think planning /52 |
|---|---:|---:|---:|---:|
| OFF | 30 | 30 | 0 | Not applicable |
| ON | 33 | 31 | 0 | 0 |

Each of the 52 ON finals states either the canonical name or the Marin Community creator/trainer. These columns overlap. Name absence can still underanswer a prompt: row035 asks for the model name on a form but receives only creator/trainer information. OFF rows040/046/047 name maintainers without stating a creator; row051 names only generic developers. OFF row054 makes an unsupported claim that xAI prepared its training data, without claiming to be an xAI model or to have xAI as its creator. Other unsupported factual details remain; this is not a general factual-QC pass.

| Third-person controls | Canned self-identity nonanswers /14 | Final third-person creator rewrites /14 |
|---|---:|---:|
| OFF | 11 | 0 |
| ON | 1 | 0 |

ON control006 answers “Who makes Claude?” with a self-identity recital. ON control014 puts a hypothetical “ChatGPT … trained by the Marin Community” system-message example in reasoning, but not in its final answer. That reasoning-only role contamination is separate from the zero final-rewrite count. The same reasoning invents a request for step-by-step boxed output.

The native split retains visible response planning in reasoning on seven core and 13 control rows. Three core reasoning spans contain literal system/policy/prompt references; row042 attributes a custom system prompt to itself and repeats an unsupported system-prompt claim in the final. These concerns are separate from post-end-think format failure.

For ON outputs, the separately labelled whole-generated-ID, skip-special-token proxy includes reasoning in its apparent answer: 48/52 core names and 42/52 creators, with visible response planning on seven core and 13 control rows. The proxy is a separate diagnostic; native-final counts and raw provider fields stay unchanged.

Evidence: [all-132 ledger](M51-native-role-aware-ledger-marten-r1.json), [native and legacy-derived views](M51-native-and-legacy-derived-views-marten-r1.json), [raw artifact index](M51/FULLGET135-native-output-and-owned-cost-index-margay-r1.json). One deterministic screen on one merged checkpoint does not establish a release decision or a causal training effect. No additional generation or parser workflow was run.
