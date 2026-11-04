# Panel-60 on Path B candidates (A13, A8a, A3): what the binding review will check (shoveler, before any output)

Order: current-orders.md "BEN'S PANEL-60 on Path B candidates" (owner GO 2026-10-31 ~08:30Z). Vizcacha authors, shoveler reviews the binding, margay runs at production, cw-rno2a, one node per arm.

1. **Runner and protocol unchanged.** Ben's interactive runner and runtime pins as in the earlier panel bindings (t310/panel60/bindings.md); the canonical 60 rows byte-equal; thinking off, temperature 0, 8,192 output tokens. Arm changes are model name, location (the permanent S3 copies) and revision only. Each arm's tokenizer and chat template recorded; the three Path B exports should carry v0.1's plain template.
2. **References reused, not re-served.** The v0.1 and v0.1 + system-prompt panel responses are reused only after checking their archived configurations and full 60-item coverage, as the earlier binding did.
3. **Rubric amendment, written before grading.** The protocol's rubric was written for the Marin Snowball / Open Athena identity. The owner's ruling adds the Path B identity. The amendment must be stated in one place before any Path B answer is read, and applied to every arm:
   - name correct: "Marin Snowball" or "Marin's Latest MoE";
   - creator correct: Open Athena, or the Marin Community (including "developed and trained by the Marin Community, maintained by Open Athena");
   - history and project questions answered by pointing to marin.community or github.com/marin-community/marin count as correct;
   - foreign claims, refusals and invented facts are graded exactly as before.
   The amendment does not change the meaning of any existing label for the old identity.
4. **Reference grades under the amendment.** The earlier reference grades used the old rubric. Before comparing, identify every reference row whose label could change under the amendment (e.g. an answer naming the Marin Community), and regrade only those rows blind, recording both the old and the amended label. If none could change, say so.
5. **Blind grading.** All answers per item graded together under a fixed shuffled key, unblinded only after grades are saved; record which rows were graded by a model and which by hand, and disclose prior grader familiarity, as before.
6. **Report.** The pre-stated ship line 4 (clean >= 20, name >= 15, foreign claims <= 3, from the identity-v2 orders) for each arm, with both references; multilabel counts preserved; no extra comparison added.

## Update (owner via jerboa, before any output): fuzzy rubric, replacing item 3's identity lists
- Name correct: any Marin-family name (Marin's Latest MoE, Marin Snowball, Snowball, "a Marin model").
- Creator correct: the Marin Community, the Marin project, or Open Athena.
- Wrong: only another lab's identity (a foreign claim).
- A missing name is scored as its own category, not as wrong.
- Applies to every arm, references included; item 4 (regrade the reference rows whose label could change, keeping both labels) and items 1, 2, 5, 6 stand. The binding must state these rules verbatim before grading, and grade the old "name" and "foreign" labels so the ship-line counts (name >= 15, foreign <= 3) stay computable.

## Update 2: regrade all reference answers (agreeing with margay)
Instead of regrading only the reference rows whose label could change, regrade ALL v0.1 and v0.1 + system-prompt answers (existing outputs v0p1-r3 and v0p1-sysdefault-r3, no new serve) with the same fuzzy grader, blind and shuffled together with the three Path B arms, keeping the old grades beside the new ones. That makes every comparison like-for-like and gives the references a missing-name bucket. Strict-name counts may also be reported, labelled as the old rubric.
