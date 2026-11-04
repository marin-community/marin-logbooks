# Blind Panel-60 grading by genet

All 300 answer texts were read with their item prompt and Ben reference. The fuzzy rubric amendment overrides older Snowball/Open Athena reference wording: Marin-family names and Marin Community / Marin project / Open Athena developer, trainer or maintainer attributions are accepted. Missing names and creators have separate categories.

`clean` requires a responsive, substantively correct answer to the actual prompt. A correct identity recital does not answer a project/history/capability question. Explicit additional parts (spelling, nonprofit status, supplied year, comparison, documentation location) still require an answer. Generic introductions or human-versus-AI explanations may be clean with name/creator missing.

`name` and `creator` concern the assistant or the named target model in a comparison/citation question. An explicit affirmative name remains recorded even if the answer also contradicts it; contradiction is explained separately. A generic label such as Assistant is missing, not foreign. Adoption of a foreign identity through an explicit Yes to the prompt may set name=foreign even without a literal name in the answer; the literal field then remains null. `foreign` excludes denials, hypothetical examples and contextual contributor/data-source mentions.

`claimed_name`, `claimed_developer`, `claimed_trainer`, and `claimed_maintainer` retain attributed text/entities, with null for absent literal attribution. Created/built-by is recorded in claimed_developer; it is not a separate trainer claim. Contributions from an organization do not alone become a developer/trainer attribution.

`invented` flags concrete false/unsupported model, organization, history or biographical content, including contradictory factual self-descriptions; omission alone does not set it. `recital` identifies stock model-name/creator/trainer/maintainer statements, including recurring contributor and identity-denial variants. It is separate from correctness.

Every row contains a rationale; uncertain judgments have uncertain=true and an explicit reason. No arm-level aggregation or arm join was performed. Serialization/coverage/hash checks are automated; grading decisions were semantic judgments, not regex classification. The source file records manual row decisions and pre-seal consistency edits.

Before this pool, genet had seen part of Ben's source protocol/reference text but no generated Panel-60 answers, prior Panel-60 grades or arm key, and did not author the binding. Ben references were included in the cleared pool. The executor handoff additionally disclosed one item index with an arm-linked finish annotation, without the A-E letter or arm-letter mapping. This is a disclosed blinding limitation. It was not used in grading; only supplied answer content was assessed. No output directory, prior grade file or arm key was opened.

Grades are sealed before executor unblinding. The executor must record/verify the grades SHA256 before opening the withheld key.
