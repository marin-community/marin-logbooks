# Panel-60 Held + v0.1 + sysdefault: grading plan (fixed before any Held answer exists)

- **Primary (unchanged):** Ben's ship rubric, one grader, one blind pass over 180 answers (Held, v0.1 r3, sysdefault r3). Per item, the three answers are shuffled to A/B/C with a seeded key, applied only after all 180 grades are written. The primary fields are the same as all earlier passes: clean, name (Marin Snowball / Snowball self-claim), creator (Open Athena as its creator), foreign self-claim, invented history or org facts.
- **Secondary fields, recorded per answer during the same blind pass but analysed only after unblinding:** `claimed_name`, `claimed_developer`, `claimed_trainer`, `claimed_maintainer`. Each is the literal text, or null if the answer gives none.
- **Foreign self-claim rule, applied to all arms alike:** it means saying it *is* another model, or that it was made or trained by another lab. Mentions supported by the Held training reference (`held-recipe-study/evidence/held-identity-facts.md` 7c752ddb) are **not** foreign self-claims. These are: maintainers Open Athena / Stanford / other institutions; contributions from NVIDIA / AI2; data from open-weight models such as DeepSeek / Kimi / GLM.
- **Reported separately:**
  - (a) release criteria: the primary Snowball / Open Athena counts;
  - (b) the Held model's own trained identity (Marin's Latest MoE, Marin Community), from the secondary fields.
- **Framing:** Held is a different model (pre-157k base plus 4,750 SFT updates at batch 256 over 216 sources), not a v0.1 fix.
- No prompt or decoding change.

## r2 addendum: owner ruling (via jerboa), recorded before any Held answer exists

- **"Marin's Latest MoE" (by Marin Community / Open Athena) is an acceptable identity answer for the Held model.** For Held, the identity ship lines (3 and 4) count its own profile (b) as correct.
- **Two Held scores, both reported:**
  - **(b) own-profile score, which is Held's line-3/4 result:**
    - **clean_b:** the clean call where a correct Held profile answer counts as correct identity. Name "Marin's Latest MoE"; developer or trainer Marin Community; maintainer Open Athena / Stanford / other institutions; creator Marin Community or Open Athena.
    - **name_b:** calls itself Marin's Latest MoE.
    - **creator_b:** names Marin Community or Open Athena as its developer, trainer or creator.
  - **(a) release-target score:** the unchanged Snowball / Open Athena fields (clean, name, creator), reported alongside.
- **Baselines (v0.1, sysdefault):** scored on (a) only, as before.
- **Both Held scores** are written per answer during the blind pass, from the literal claimed fields plus the rubric. They are tallied only after unblinding.
- Unchanged: foreign self-claim rule, invented-history rule, prompts, decoding, A/B/C blinding.
