# Panel-60 Held + v0.1 + sysdefault: grading plan (fixed before any Held answer exists)

- **Primary (unchanged):** Ben's ship rubric, one grader, one blind pass over 180 answers (Held, v0.1 r3, sysdefault r3). Per item, the three answers are shuffled to A/B/C with a seeded key, applied only after all 180 grades are written. The primary fields are the same as all earlier passes: clean, name (Marin Snowball / Snowball self-claim), creator (Open Athena as its creator), foreign self-claim, invented history or org facts.
- **Secondary fields, recorded per answer during the same blind pass but analysed only after unblinding:** `claimed_name`, `claimed_developer`, `claimed_trainer`, `claimed_maintainer`. Each is the literal text, or null if the answer gives none.
- **Foreign self-claim rule, applied to all arms alike:** it means saying it *is* another model, or that it was made or trained by another lab. Mentions supported by the Held training reference (`held-recipe-study/evidence/held-identity-facts.md` 7c752ddb) are **not** foreign self-claims. These are: maintainers Open Athena / Stanford / other institutions; contributions from NVIDIA / AI2; data from open-weight models such as DeepSeek / Kimi / GLM.
- **Reported separately:**
  - (a) release criteria: the primary Snowball / Open Athena counts;
  - (b) the Held model's own trained identity (Marin's Latest MoE, Marin Community), from the secondary fields.
- **Framing:** Held is a different model (pre-157k base plus 4,750 SFT updates at batch 256 over 216 sources), not a v0.1 fix.
- No prompt or decoding change.
