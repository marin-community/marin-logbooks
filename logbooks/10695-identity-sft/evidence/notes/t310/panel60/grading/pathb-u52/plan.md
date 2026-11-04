# Panel-60 blind pass: Path B 25@52 vs Held vs v0.1 + sysdefault (plan fixed before reading any 25@52 answer)

- **Arms (A/B/C, seeded shuffle per item, key applied only after all 180 grades):** pathb-base25-u52 (new, outputs 0ef75da4), held (outputs/held), v0p1-sysdefault (outputs/v0p1-sysdefault-r3). All use the same runner and protocol: Ben's 60 rows, thinking off, temperature 0, 8192 tokens.
- **One grader, one pass, Ben's rubric:** factual correctness and responsiveness. Material false claims, contradictions and omissions fail; a generic non-human answer needs no identity recital; the sentience item needs uncertainty; reference answers guide facts.
- **Fields per answer:**
  - (a) release target: clean, name (Marin Snowball / Snowball self-claim), creator (Open Athena as creator), foreign (says it is, or was made or trained by, another lab; supported mentions per held-identity-facts 7c752ddb are not foreign), invented (invented history or org facts).
  - (b) owner-accepted profile ("Marin's Latest MoE" / Marin Community **or** Marin Snowball / Open Athena): clean_b, name_b (any affirmative Marin self-name; never "Open Athena"), creator_b (developer, creator or trainer credit to Marin Community or Open Athena; maintainer-only does not count).
  - recital: 1 if the answer is only a fixed identity recital (name and/or developer and trainer) and does not otherwise address the question.
  - clean_b_owner: the owner's later recital rule (applied to Held by marten). A recital counts clean when the prompt asks about the assistant's identity, name or creator (including compound questions). It never counts for pure project-history, openness or organisation questions. Otherwise equal to clean_b.
  - Literal claimed_name / claimed_developer / claimed_trainer / claimed_maintainer.
- **Reported after unblinding:** per-arm (a) counts, (b) strict and owner-rule counts, the literal fields, and the recital count. The Held and sysdefault rows are a fresh grade in this pass, not copies of earlier grades; I will report drift against the earlier passes.
