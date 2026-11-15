---
title: "Atlas RL data catalog and curation campaign"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/48"
related_issues:
  - "https://github.com/marin-community/marin-logbooks/issues/49"
research_branch: "codex/48-atlas-campaign"
marin_base_revision: "6e09d9043eb71d7bd487b5e3d79e275b95c454fd"
archive_template_revision: "3744203cdaa19ab4b4943e9836e143e7a4c339c4"
status: active
---

# Atlas RL data catalog and curation campaign

## Question and decision criteria

- Goal and why it matters: Atlas is the RL data catalog / curation effort. This record keeps a chronological, evidence-backed account of the campaign: source ingestion and conversion, faithful-environment audit and repairs, oracle filtering + NoP negative controls, SAMPLE/FULL QC, overlap and duplicate folding, difficulty measurement, curriculum, the private preview app revisions (13–18, with 19 upcoming), and the user rulings of 2026-11-14/15.
- Baseline and alternatives: record-keeping task; no decision between arms.
- Evidence needed to choose: n/a (record). Every claim carries primary evidence (commit SHA, PR/job ID, receipt URI, landing record); anything unverified is labeled as a hypothesis.
- Constraints and unresolved questions: public record — no secrets, credentials, private user info, or private dataset contents. Production Atlas publication (v76) remains on HOLD by user directive throughout the period recorded here.

## Baseline

- Date: 2026-11-15 13:55 UTC (08:55 EST)
- Published source revision: branch `atlas-env-audit`, head `96fc0736fc95d8f21d11e1c03dfca108d568b220` (pushed 2026-11-15 13:51 UTC via atlas-push, landing record in ~/.local/share/atlas-push/landings.jsonl). Parent PR [marin#11277](https://github.com/marin-community/marin/pull/11277) is OPEN in DRAFT state, base `main`, title "Consolidate Atlas ingestion, qualification and evidence". Marin `main` untouched.
- Configuration, command, and external inputs: campaign work runs in experiment directories under /experiments/active/atlas-*; landings flow directly through `atlas-push` (receipt log ~/.local/share/atlas-push/landings.jsonl, shared registry lock `~/.local/share/atlas-push/registry.lock`).
- Environment and hardware when relevant: per-entry.
- Result and evidence: see entries.
- Limitations: this catch-up was compiled from experiment directories, the claim-audit reports, git/PR metadata and landing records; PR #11277 comment thread and per-reviewer QC detail were not exhaustively audited.

## Entries

> [!NOTE]
> Read the newest entries first: the 13:55 UTC catch-up entry below is superseded **in part** by the later revision entries at the end of this file — the current synopsis is in **revision 8 (17:07 UTC, after cold-read round 8)**, which consolidates the corrected gate matrix (revisions 5), the current facts/evidence index (revision 4), and the evidence-ledger state (revisions 6–7 and the `evidence/` files as updated 17:05 UTC). Historical synopses: revision 3 (14:44 UTC), revision 4 (14:54 UTC), revisions 6–7 (15:12/15:45 UTC). Corrections are appended, never rewritten.

### 2026-11-15 13:55 UTC - Campaign catch-up record (2026-11-12 through 2026-11-15)

- Question or hypothesis and reason for trying it: build the chronological catch-up record of the Atlas campaign so far, per the research-secretary brief.
- Published source revision: `atlas-env-audit` head `96fc0736fc95d8f21d11e1c03dfca108d568b220` (2026-11-15 13:51 UTC); marin_base_revision of this archive `6e09d9043eb71d7bd487b5e3d79e275b95c454fd`.
- Command or material configuration: sources read: /experiments/active/atlas-env-audit/{BRIEF.md,REPORT.md}, /experiments/active/atlas-rulings/* (10 files), /experiments/active/claim-audits/fullqc-pass3c.report.md, /experiments/active/HANDOFF-COMMON.md, per-campaign STATUS/BRIEF files under /experiments/active/atlas-*, git log of origin/atlas-env-audit, `gh pr view 11277`, ~/.local/share/atlas-push/landings.jsonl.
- External inputs and artifact references: listed inline below.
- Environment and hardware when relevant: inline below.
- Result, including failure or restart: chronology below (newest first within the day; times UTC with EST where useful).

**Campaign origin and goals (2026-11-12)**

- Campaign start. `atlas-env-audit/BRIEF.md`: audit every runnable agentic source for environment fidelity — a task must run in exactly its own specified environment; shared-image substitution (iris-task) and missing-environment → host-image resolution are forbidden; close the `TaskTroveConverter` gap (`environment_unused()` returning None when no Dockerfile exists); deprecate legacy `tasktrove-*` sources with admitted faithful replacements; repair in order of training value (SWE/agentic coding first); publish private TaskCompendium releases on open-athena; open a draft PR into `swe-repair-atlas`.
- User addendum (2026-11-12, in BRIEF.md): "repaired" = the published spec completely describes runtime behavior, with (a)/(b)/(c) classification (spec-accurate + QA / spec-accurate, QA missing / spec-vs-runtime mismatch); the 28 grandfathered MarinSkyRL blends are explicitly out of scope.
- Draft PR [marin#11277](https://github.com/marin-community/marin/pull/11277) opened 2026-11-12 21:11 UTC, titled "Consolidate Atlas ingestion, qualification and evidence", head branch `atlas-env-audit` (evidence: PR metadata).

**Codex handoff and expanded policy (2026-11-14 ~09:11 UTC)**

- The predecessor Codex worker exhausted quota mid-task; OpenCode workers took over (HANDOFF-COMMON.md). Standing user policy restated there: faithful environments; temperature 0.7 everywhere with judges via the production host judge service; 3-day timeouts; explicit `--target-cluster` co-located with data for every CPU job; never delete artifacts (bump versions); never restart clusters or controllers; QC never rejects (Result GOOD ≥90% clean / BAD ≥50% issues); compute pre-approved but GPU/CPU-hours reported; **no production Atlas publish (v76 HOLD)**; land code as child PRs into `atlas-env-audit`.

**User rulings (2026-11-14; all files in /experiments/active/atlas-rulings/)**

- `nonagentic-exempt-agentic-checks-20261114.md` (ruling SHA `b02c3b35…`): non-agentic tasks are exempt from ALL agentic-only checks (resources/locks/images/tools/backend/network/nop controls); supersedes `nonagentic-no-resource-spec-20261114.md` (SHA `14a52b16…`).
- `no-oracle-executability-policy-20261114.md`: per-task executability proof requirement dropped for no-oracle sources (campaign not launched); all other gates remain.
- `negative-control-retry-policy-20261114.md`: standing global policy — one identical-budget re-drive for NoP controls killed by -9/-6/-11 or producer interruption; verifier-deadline/source errors are annotations; admission = oracle passes AND nop fails.
- Legacy qualification adoptions (all offline, evidence-only, no replay/None-inference): **311 passes** (ruling SHA `2bee03ef…`, partition `95a71b3e…`, audit `d660ad1f…`); **249 small-role passes** (125 Unitsyn + 125 PyM; 1 genuine scored failure excluded); **18 producer-1485 passes** (separate, not reusing 311/249). Files: `legacy-311-role-qualification-adoption-20261114.{md,json}`, `legacy-249-…`, `legacy-18-producer-1485-adoption-20261114.md`.

**Merger role retired (2026-11-14 16:13 UTC)**

- Coordinator directive (`3b008184`): the merger session is retired; authors land directly through `atlas-push` and report to the coordinator. Final merger-window landings include CrossPy `245c7e672a…` (500 tasks stay EVAL/Excluded), actorless applicability `2b8590aa60…`, docs sync `ad9ec7731c…`+`d02bb41362…`, count projection `11fff26215…`+`21fbf617b2…`, positive-release binder `7bbaf33070…`+`346525a3…` (installs zero pins). Evidence: landings.jsonl, atlas-env-audit REPORT.md.
- Same day 12:29–13:38 UTC: a FULL-QC worker pushed two unexpected commits during a handoff (misread "ask for a slot" as a grant); the stale push was rejected and the branch fast-forwarded cleanly to `483`-prefix tip; no force/reset used.

**Preview app revisions (2026-11-14 → 2026-11-15)**

- Revision 13 (v9.3 catalog) published 2026-11-14 14:04 UTC; revision 14 published 15:51 UTC from `d02bb41362…`; preview13 verified 2026-11-14 14:52 UTC: 310 source IDs, 4,117,631 retained tasks (split 3,226,076 no-oracle TRAIN + 350,205 no-oracle EVAL + 533,587 oracle TRAIN + 7,763 oracle EVAL); **publication 14 held** pending corrected count projection.
- Revisions 15/16/17 published 2026-11-14 16:17/18:30/21:36 UTC. The contrast audit's frozen baseline measured the revision-17 package: 2,758,701 violation observations, later fixed to zero.
- **Revision 18 published and hosted-verified 2026-11-15 ~02:00 UTC** at applets.marina.oa.dev/a/c9742a15-…/v/18/ (package `af8d4d6b…`, 59 files): 86 API tests, 62 browser journeys, 804 real-data views pass; zero authored contrast findings; headers: TASKS 4,206,876 total / 3,251,230 shown; 3,322,365 admitted (TRAIN 2,972,160 · EVAL 350,205), 541,350 pending oracle filtration, 343,161 excluded; evidence-only cards for five filtered private Code releases (9,133 tasks) pending qualification. Production stays on **v76 HOLD**.
- **Revision 19 in preparation, NOT yet published** (no deployment-v19 artifact as of 2026-11-15 ~14:00 UTC): latest v19-* artifacts from ~13:41 UTC Nov 15 include catalog snapshot `atlas-preview-20261115-r19-v1` (310 sources / 28 assessments), maintained-admissions staging incl. nl2bash-oracle, pending-card journeys, 64/64 catalog tests, advisory 0 findings on `9bfcd64bf9…`.

**FULL-QC openreasoning pass 3c failure — verified cause (2026-11-15 03:44–12:31 UTC)**

- Report: /experiments/active/claim-audits/fullqc-pass3c.report.md; job `/benfeuer/atlas-full-qc-repair3-openreasoning-v3pass-c-v1`, source rev `87f63bd2…`. Outcome: 43/118 REVIEWED, 75 INVALID, no final artifact.
- **Verified cause: a deterministic tuple-unpacking crash in output assembly** (runner lines 143–144, `AttributeError: 'tuple' object has no attribute 'repaired_task_ids'` after "repaired 104 of 104 touched batches", reproduced twice; Iris auto-retry, failures=2).
- Refuted claims: the "32,768-token retry" claim (actual limits 8192/16384; 92.1% length-finish; 8.85M of 8.97M completion tokens are reasoning) and any preemption story: GPU telemetry shows 8 GPUs (TP4×PP2, nodes s41sxs64/shbrs64), 100% utilization gauges but ~33% SM / ~1.5% tensor active, ~656 tok/s, mean 239 s/request, **zero vLLM preemptions** — slowness is repeated reasoning-only generations plus sparse barrier-scheduled batches, with a fresh cache (0 cache hits across 710 requests).

**Puzzles QA repair completed (2026-11-15 ~02:10–07:45 UTC)**

- /experiments/active/atlas-puzzles-qa-repair/STATUS.md: private pin `open-athena/tasktrove-puzzles-taskcompendium-20261114` @ `adfcbd4a8d…` (6,921 tasks, digest `8542395b…`); oracle filter 6,921/6,921 admitted, empty controls 0.0; lexical contamination 0/6,921; **SAMPLE: 100 sampled, 97 reviewed, 3 unavailable, 22 issues (22.68%, SOME_ISSUES)**.
- Two prior SAMPLE attempts failed on infrastructure: 4×GB200 serving gang SchedulingGated through two 4-hour startup timeouts; attempt 3 cancelled by the coordinator and the frozen queue (sha `02ceafdb…`) handed to FULL QC, which ran it on 8×GB200 (serving job `…/inference-fa576089…`). Receipt bound under the shared `registry.lock`, commit `4bdbc7eeed…`.

**SAMPLE 5%-missing tolerance ruling (2026-11-15)**

- `atlas-rulings/sample-missing-tolerance-20261115.md`: SAMPLE qualifies with ≤5% missing rows (≥95 of 100 reviewed), panel fixed, missing rows recorded explicitly; triggered by the TACO-built Code SAMPLE at 99/100. Implemented in code and docs: commits `7c7b023e3e…`, `7ee5b79095…`, `6ab38e7a5f…` ("Keep SAMPLE tolerance guidance in Getting Started").

**2026-11-15 admissions (07:20–13:51 UTC)**

- QA/oracle lane: raw Stack admitted (110,600 tasks, `c707bfbd5e…`); five QA pins (110,152 tasks, `635e7b907f…`); final three QA pins → **nine QA admissions, 431,389 tasks complete** (`8ea14c762f…`); PyMethods-small and CrossCode-Python EVAL admitted (`2d8202d007…`, `c9f0235d8e…`); eight mechanically-excluded empty Code releases registered (`7765940987…`).
- Code lane completes **14 Code admissions**: rebuilt Unitsyn SAMPLE registered (`fd670c41d1…`), TACO-laion registered to complete the fourteen (`0eaf9b7fb4…`), admitted-Examples counts projected (`6ea661f26a…`). Branch tip lands as `96fc0736fc95d8f21d11e1c03dfca108d568b220` ("Acknowledge overlaps after TaskTrove code filtration", 13:51 UTC).

**Difficulty roster (2026-11-13, user decision)**

- Solver roster for difficulty v9.2/v3: Small = Qwen3-Coder-30B-A3B, Large = Qwen3.5-122B-A10B, Hosted = GLM-5.3, plus NEW Snowball v0.1; 100-task samples over admitted populations across 310 live sources; protocol `atlas-difficulty-v3` (/experiments/active/atlas-difficulty/BRIEF.md). Runs on a separate immutable record path.

**Sub-campaign one-liners (with evidence in their dirs)**

- Duplicate folding: containment-only stars at 0.95 threshold; handed to the Codex session 01a1db34 with an uncommitted diff.
- NN-overlap repair: never pair by task_id; 741→837-pair survey; aime24/e2egit user findings being fixed in atlas-overlap-audit.
- Tau repair: regenerated missing requirements.lock, new private pins, 2×20,035-task censuses bound.
- Derived-tags grader: GraderType enum + derived-contract census built; receipts pending first publication.
- Contrast audit: zero violations across 3,542 captures; landed `79d93172…`, `87f63bd2…`, `10f77329…` (+ `9741cf2529…`, `20b1d0b1a3…`, `2a334460e4…` 2026-11-15 00:08–02:15).
- Curriculum audits (atlas-curriculum-audit, atlas-curriculum-coverage): analysis lanes feeding the catalog.
- ~40+ child PRs merged into `atlas-env-audit` (e.g. marin#11439, #11444, #11421, #11436, #11429), each with lint/Pyrefly + affected tests + independent readbacks.

**Current state**

- Branch `atlas-env-audit` head `96fc0736fc95d8f21d11e1c03dfca108d568b220` (2026-11-15 13:51 UTC); PR #11277 OPEN/DRAFT, base `main`; merger retired; landings via atlas-push.
- Private preview revision 18 published; revision 19 in preparation; production Atlas v76 and marin#11120 on HOLD.
- Admissions 2026-11-15: 14 Code + 9 QA (431,389 tasks) + raw Stack (110,600) + 2 EVAL admissions + 8 typed-excluded empty Code releases; SAMPLE 5% tolerance in code and docs.
- SAMPLE panels archived: 245 panels — 168 GOOD / 72 SOME_ISSUES / 5 BAD (per PR #11277 body).

**Open items**

1. Publish and verify preview revision 19 (Nov 15 admissions, 5% SAMPLE tolerance display, overlap acknowledgements, backend census).
2. PR #11277 remains draft; full CI/validation pending; v76/#11120 HOLD until the user lifts it.
3. The pass-3c output-assembly bug (source rev `87f63bd2…`) is unfixed; 75 INVALID openreasoning rows need a repaired pass; reviewer strategy for reasoning-only truncations unresolved.
4. Openreview/legacy adoptions (311/249/18): whole-source collection, global NegativeControlPolicy, and content QA still pending before admission; no source pins installed by those rulings.
5. Puzzles QA: SOME_ISSUES (22 issues) recorded; remaining admission blockers are grading census + decontamination publication receipt; FULL QC owns the whole-source review.
6. Full QC: remaining sources (Stack, Open Reasoning, OpenSWE repairs; ≤1% unreviewed waiver standing) per atlas-full-qc/HANDOFF.md; difficulty snowball continuation across the remaining sources.
- Evidence: per-item references above; compiled by the research secretary from primary files, 2026-11-15 13:55 UTC (08:55 EST).
- Interpretation and limitations: counts and SHAs are as recorded in the cited primary artifacts; the catch-up spans only 2026-11-12 onward (campaign origin).
- Decision and next action: continue hourly catch-up entries on this branch.

### 2026-11-15 14:25 UTC - Revision 1 after cold-read round 1: corrections, glossary, policy, evidence classes

- Question or hypothesis and reason for trying it: an independent cold-read review of the 13:55 entry returned NOTES (38 items). Reason: the record must be understandable and assessable by a public newcomer without private context. This entry appends corrections and context; the 13:55 entry is preserved unedited (workflow rule: corrections are new entries).
- Published source revision: unchanged since the 13:55 entry — `atlas-env-audit` head `96fc0736fc95d8f21d11e1c03dfca108d568b220`; https://github.com/marin-community/marin/pull/11277 (open draft).
- Command or material configuration: review process: a context-free reviewer read only public GitHub material (this notebook at `f02de6077a46d8f8e9d21afd40de1df70195d3b2`, PR #50, issues #48/#49, public marin PRs) and returned a local-only notes report [R]. No private material was read by that reviewer; no secrets were found in the reviewed public text.
- External inputs and artifact references: https://github.com/marin-community/marin/pull/11277, https://github.com/marin-community/marin/pull/11421, https://github.com/marin-community/marin/issues/11490, https://github.com/marin-community/marin/pull/11120, https://github.com/marin-community/marin/blob/96fc0736fc95d8f21d11e1c03dfca108d568b220/docs/references/rl-data-atlas.md (public Atlas reference; read it first — it defines most terms below).
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: the following supplements/corrects the 13:55 entry.

**Evidence classes used in this record**

- [P] = publicly verifiable (link to marin/marin-logbooks/HF/public PR or commit).
- [M] = requires authenticated access (open-athena registry/HF private releases, Marin login); a hash authenticates available bytes but the bytes are not publicly readable.
- [R] = reported from local/private records (experiment dirs, audit reports, landing logs); not publicly inspectable; treat as reported, not independently verified.
- The 13:55 entry's statement "every claim carries primary evidence" is qualified accordingly: claims carry evidence as recorded in the cited primary artifacts, but most artifacts are [M] or [R]; where possible the underlying commits are [P] via marin PRs/commits.

**Corrections superseding the 13:55 entry**

1. PR #11277 was created 2026-11-12 **21:09:15Z** (GitHub `createdAt`), not 21:11 UTC. The base was later consolidated to target `main` (the original plan targeted `swe-repair-atlas`); the consolidation happened during the 2026-11-14 handoff period [R]; the current base is `main` [P].
2. Preview revision 14 (2026-11-14): the package was **built** (15:51 UTC) and **deployed/verified on the private preview app**, but its **publication approval was held** pending the corrected count projection. These are distinct states (build/deploy/verify/approve); "publication held" referred to the approval step. Preview13 remained the last user-approved revision until revision 17+.
3. Admission-related timestamps in the 13:55 entry are **commit times**; landings via `atlas-push` may occur later (landing receipts live in the local atlas-push log [R]).
4. Alias map: **CrossPy = CrossCode-Python** (same source); **PyM = PyMethods**; Unitsyn, TACO, Tau, raw Stack, Openreview are canonical source families. The 2026-11-14 CrossPy note ("500 tasks stay EVAL/Excluded") described that source's oracle-filter disposition at that time; the 2026-11-15 "CrossCode-Python EVAL admitted" (`c9f0235d8e…`) is the same source after the filter change. Both states are bound in registry revisions [M].
5. Difficulty sampling (2026-11-13 decision, per https://github.com/marin-community/marin/pull/11421): **up to 100 tasks per source** (not exactly 100); 42 oracle-pending sources carry pre-filter/resampling flags, so difficulty cohorts are measured pre-admission-filter for those sources; per-model selected task counts total 30,530 in that PR's description.
6. `9bfcd64bf9…` (13:55 entry, rev-19 line) is an **advisory-review artifact identifier**, not a marin commit; it does not resolve as a commit by design.
7. The 2026-11-15 FULL-QC worker reference "(session ses_e264443f…)" is an internal session identifier, unnecessary for the public record; worker/owner identity should be referenced by campaign lane only. Internal GPU node labels (s41sxs64/shbrs64) are operational metadata retained for the audit trail; future entries will avoid internal session/host identifiers.
8. The Puzzles headline is stage-specific: **oracle filter passed 6,921/6,921; SAMPLE completed (97 reviewed / 22 with issues, 22.68% = 22/97 tasks affected, 3 unavailable)** — the *source admission* remains pending the whole-source FULL review, the grading census, and the decontamination publication receipt (unchanged open items).
9. The "245 SAMPLE panels — 168 GOOD / 72 SOME_ISSUES / 5 BAD" tally is the **archived panel** set as recorded in the PR #11277 body at a dated snapshot; it is not 245 current independent source results, and archived grades should not be read as current catalog-wide quality.
10. The "as of ~14:00 UTC" statement in the 13:55 entry's rev-19 bullet refers to the compilation observation time; the entry's cutoff is 13:55 UTC.

**Glossary (compact; the public Atlas reference is normative)**

- **Atlas**: the RL data catalog/curation campaign. **TaskTrove**: the source corpus/pipeline where upstream datasets are converted to runnable tasks. **TaskCompendium**: published curated task releases on open-athena (Hugging Face org). **TaskSpec**: the per-task specification (description, environment, oracle, grading). **Harbor**: the agent-harness runtime used to execute tasks. **Iris / iris-task**: the job scheduler and its task-execution adapter. **open-athena**: the Hugging Face org hosting models/datasets (private releases need auth). **MarinSkyRL**: the RL training stack consuming these tasks; its 28 pre-existing blended sources are grandfathered out of the fidelity audit.
- **pin**: an immutable registry binding to a dataset revision (SHA/digest). **receipt**: a hash-bound record of an operation (admission, census, seal). **NoP / nop**: no-op negative control — an empty/degenerate execution run alongside real oracle runs; admission requires oracle passes AND nop failures (a nop that "succeeds" indicates a broken oracle). **SAMPLE**: a per-source review of a fixed ≤100-task panel; **FULL**: whole-source review; **QUICK**: lightweight check tier. Result thresholds: GOOD ≥90% clean, BAD ≥50% issues; QC never rejects — results annotate.
- **SWE**: software-engineering (agentic coding) sources. **NN overlap**: nearest-neighbor similarity survey across sources (pair-level duplicate risk). **actorless**: task variant without an agent role. **role / small-role / producer-1485 / 311 / 249 / 18**: legacy offline qualification cohorts (see 13:55 entry rulings). **positive-release binder**: bundling of positively-qualified releases (installs zero pins). **derived-contract census / grading census**: static metadata surveys of grader contracts.
- **QA** in this record means the question-answer source category (not quality assurance); quality assurance is called QC.

**Campaign question and completion criteria (record-keeping context)**

- Technical question: can upstream agentic/non-agentic task sources be ingested and converted so each task runs in exactly its own declared environment, filtered by oracle validity with NoP controls, quality-reviewed (SAMPLE/FULL), deduplicated/overlapped, difficulty-measured, and published as a curated catalog — without production publication until the user lifts the v76 hold?
- Completion criteria (campaign-level, as directed by the user): all runnable sources classified (a)/(b)/(c) on spec-vs-runtime fidelity with repairs landed; admission pipeline (oracle + NoP + QC + content gates) executed per source; preview app reflecting the catalog; production publication awaiting explicit user approval. Downstream RL benefit is explicitly **not** a measured criterion of this campaign.

**Population and counts (reconciliation)**

- 310 = live source IDs at the 2026-11-15 cuts (rev-13 and rev-18/19 snapshots); 326 archived = historical source ID population in the PR body (16 retired); the 28 MarinSkyRL blends are excluded from fidelity scope and from these counts.
- Revision-13 (2026-11-14 14:52 UTC): 4,117,631 retained tasks = 3,226,076 no-oracle TRAIN + 350,205 no-oracle EVAL + 533,587 oracle TRAIN + 7,763 oracle EVAL.
- Revision-18 (2026-11-15 ~02:00 UTC): 4,206,876 total instances; 3,251,230 shown; 3,322,365 admitted (TRAIN 2,972,160 · EVAL 350,205); 541,350 pending oracle filtration; 343,161 excluded. Units are task instances as counted per source contract (a task may appear in multiple source releases); "shown" is the preview view's display filter (folding hides overlap groups and filtered/pending cards); "admitted EVAL" is catalog EVAL admission (not an eval measurement). The 89,245 growth 13→18 and the 955,646 not-shown reflect the 2026-11-15 admissions plus view filtering; the per-snapshot reconciliation ledger lives in the registry [M], with the count-projection commits in marin (`11fff26215…`, `21fbf617b2…`) [P].
- 2026-11-15 admissions per release ledger: QA lane — five pins (110,152 tasks, `635e7b907f…`), final three pins completing **nine QA releases totaling 431,389 tasks** (`8ea14c762f…`) [M]; raw Stack 110,600 tasks (`c707bfbd5e…`) [M] — raw Stack remains in FULL QC; Code lane — 14 Code admissions including rebuilt Unitsyn SAMPLE (`fd670c41d1…`) and TACO-laion (`0eaf9b7fb4…`), plus PyMethods-small EVAL (`2d8202d007…`) and CrossCode-Python EVAL (`c9f0235d8e…`); eight empty Code releases typed-excluded (`7765940987…`). Five filtered private Code releases (9,133 tasks) remain evidence-only pending qualification. These categories are disjoint release sets [M].

**Admission policy matrix (effective 2026-11-15)**

- Agentic sources: faithful-environment check; oracle filtering with NoP negative controls (admission = oracle passes AND nop fails, after the 2026-11-14 negative-control retry policy: one identical-budget re-drive for controls killed by -9/-6/-11 or producer interruption); content Result computed (GOOD/SOME_ISSUES/BAD; QC never rejects);SAMPLE panel + FULL review gates; executability proofs required.
- Non-agentic sources: exempt from ALL agentic-only checks (ruling `nonagentic-exempt-agentic-checks-20261114` [R]); answer-graded oracle filtering applies where applicable.
- No-oracle sources: per-task executability proof requirement dropped (ruling `no-oracle-executability-policy-20261114` [R]); other gates remain.
- SAMPLE: fixed panel ≤100, ≥95% reviewed required (≤5% missing tolerance, ruling `sample-missing-tolerance-20261115`, retroactively applied to the TACO 99/100 panel); missing rows recorded; unknown/uncertain rows count against the reviewed denominator.
- FULL: whole-source review; ≤1% unreviewed waiver standing for remaining sources. Legacy qualifications (311/249/18) are **adopted historical evidence**, offline, no replay/None-inference; they install no admitted pins by themselves.

**Pass-3c failure: units and qualified interpretation**

- Job `/benfeuer/atlas-full-qc-repair3-openreasoning-v3pass-c-v1` [M]; source rev `87f63bd2…` [P] binds the source snapshot (the app-layout change in that commit is coincident work, not the runner fix). Units: 118 = source rows in the repair population; 43 REVIEWED / 75 INVALID are row outcomes; 104 = touched repair batches; 710 = inference requests across the attempt. No final artifact was persisted; partial row states were observed in the attempt logs [R].
- Telemetry: 8 GPUs (TP4×PP2), ~33% SM / ~1.5% tensor active (aggregate gauges), ~656 tok/s aggregate, mean 239 s/request, limits 8192/16384 completion tokens by batch tier, 92.1% length-finish (denominator: finished requests), 8.85M of 8.97M completion tokens reasoning-only. Zero vLLM engine preemption events in the queried logs; this establishes the engine-level fact only — scheduler/host-level events are excluded by the audit's job-lifecycle check [R], and the causal attribution (reasoning-only generations + sparse barrier-scheduled batches) is the auditor's inference, not a controlled comparison.
- **Unfixed**: the tuple-unpacking bug at runner lines 143–144 remains open; the runner file path was not published here because the fix is pending — it will be linked when the repair commit lands.

**Puzzles attempt table (SAMPLE lane)**

| Attempt | Window (UTC) | Hardware | Outcome |
|---|---|---|---|
| 1 | 2026-11-14 | 4×GB200 serving gang | SchedulingGated → 4 h startup timeout, no verdicts |
| 2 | 2026-11-14 | 4×GB200 serving gang | Same infrastructure failure |
| 3 | 2026-11-14 | — | Cancelled by coordinator; frozen queue (sha `02ceafdb…`) handed to FULL QC |
| 4 (FULL QC) | 2026-11-15 ~02:10–07:45 | 8×GB200 | Completed: 97/100 reviewed, 22 tasks with issues (22.68%), 3 unavailable |

- The completed unit is the **SAMPLE review**, not the whole-source review (which remains with FULL QC). Empty controls 0.0 = 0 failing empty-control tasks / 6,921 oracle-filter population. Lexical contamination 0/6,921 = zero matches under the lexical decontamination test against the campaign's reference corpus; this does not establish absence of semantic contamination.
- Reviewer: campaign-standard LLM reviewer per the QC configuration [M]; panel selection is a fixed seeded panel [M]; the 3 missing reviews are recorded as unavailable (missingness not random).

**Limitations (added)**

- Most evidence is [M]/[R]: an independent public reader cannot re-derive the counts or hashes from this record alone; sanitized aggregate receipts are published only as described here.
- SAMPLE panels are small (≤100) with non-random missingness; LLM reviewers have unquantified error rates; archived panel grades may be stale.
- Lexical-only decontamination; waived executability checks for no-oracle sources; legacy adoptions inherit historical evidence quality.
- No downstream RL training benefit has been measured for this catalog; catalog/UI counts do not imply training utility.
- This record was compiled by one secretary from primary artifacts; the PR #11277 comment thread and per-reviewer QC detail were not exhaustively audited.
- GPU/CPU-hour accounting is incomplete in this record: per-run costs are not consistently reported by the campaign lanes; where absent, mark unknown. Known items: pass-3c 8×H100 (see units above); puzzles SAMPLE attempt 4 on 8×GB200; two failed 4×GB200 serving attempts. Other runs' costs are unrecorded here.

**Open items (owners/criteria added)**

1. Preview revision 19 publish + verify — owner: preview lane; criteria: deployment-v19 artifact, hosted verification, counts match the 13:51 registry cut.
2. marin#11277 merge + v76/#11120 hold lift — owner: user decision; criteria: explicit user approval.
3. Pass-3c repair — owner: FULL-QC lane; criteria: fixed runner, repaired pass completes, final artifact bound.
4. Legacy adoptions (311/249/18) — owner: qualification lane; criteria: whole-source collection, global NegativeControlPolicy, content QA.
5. Puzzles — owner: FULL QC; criteria: whole-source review + grading census + decontamination receipt.
6. Full-QC remainder (Stack, Open Reasoning, OpenSWE) — owner: FULL-QC lane; criteria: sources reviewed to the ≤1% waiver or completion.
- Interpretation and limitations: as above.
- Decision and next action: hourly catch-up entries continue; this revision resolves the round-1 cold-read notes to the extent the public record can; the next fresh cold read will re-review the updated branch.

### 2026-11-15 14:35 UTC - Revision 2 after cold-read round 2: admission ledger corrections, gate policy, control polarity, public quality outcomes

- Question or hypothesis and reason for trying it: a fresh cold-read round 2 returned NOTES (36 items), including counting errors in revision 1. This entry corrects them with public commit evidence.
- Published source revision: correction binds the recorded 13:51 source cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`; the live branch head subsequently moved (`279a2655b3eef774ce9ab583d46bc9cc80946147`, observed 14:18:12 UTC) — moving-branch claims are timestamped as observations; this record pins the 13:51 cut.
- Command or material configuration: public evidence: https://github.com/marin-community/marin/commit/c707bfbd5e872db9439f5f39dc8af1405a86e85b (raw Stack), https://github.com/marin-community/marin/commit/635e7b907f95ea7a25d4be93e60dbecc6faedb0c (five QA), https://github.com/marin-community/marin/commit/8ea14c762f158e9e9a6cc5bab53b3cb328e7f23e (final three), https://github.com/marin-community/marin/commit/0eaf9b7fb447a4eaf1f528ac47c82ae7b2695407 (Code completion), https://github.com/marin-community/marin/commit/2d8202d0071a3ac9bd57efc4e92043c73bb04df5 (PyMethods-small), https://github.com/marin-community/marin/commit/c9f0235d8e8d6bf9e739caa907d213f9952ffc35 (CrossCode-Python), https://github.com/marin-community/marin/blob/96fc0736fc95d8f21d11e1c03dfca108d568b220/docs/references/task-curation.md (normative admission policy).
- External inputs and artifact references: as above plus https://github.com/marin-community/marin/issues/11490 (curriculum), https://github.com/marin-community/marin/commit/4bdbc7eeed5a861e0bb22d25a434dd5a98744484 (Puzzles SAMPLE binding).
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: corrections and supplements below.

**CORRECTIONS superseding the 13:55 entry and revision 1**

1. **Raw Stack double-count fixed.** Raw Stack (110,600 tasks) is a **member of the nine QA releases**, not a separate disjoint category. Nine releases = raw Stack 110,600 + five pins 110,152 + final three 210,637 = **431,389 tasks total** [P].
2. **Code/EVAL accounting fixed.** The fourteen Code releases total **42,406 tasks (41,906 TRAIN + 500 EVAL)** [P]; CrossCode-Python contributes the 500 EVAL; **PyMethods-small is one of the fourteen** (its converter declares TRAIN; 496/500 tasks retained at admission). The revision-1 "two EVAL admissions" as a disjoint category is withdrawn — those releases are members of the fourteen.
3. **Empty-control polarity clarified.** NoP controls must FAIL for admission (a control that passes indicates a broken oracle). Puzzles is non-agentic and answer-graded, so workspace NoP is N/A per the non-agentic exemption; the recorded "empty controls 0.0" is the **zero pass rate (mean reward 0.0) of degenerate/empty-input controls** [R], i.e., controls failed as required. "No failures" was never the criterion; the 13:55 phrasing was ambiguous.
4. **Review gates are per-release oversight levels, not universal.** Per the normative task-curation policy [P]: SAMPLE-level review sufficed for the 2026-11-15 Stack/Code admissions, with FULL review outstanding as follow-up work; Puzzles requires FULL before its admission per its lane; completion of review ≠ favorable Result (BAD is permitted; QC never rejects); "admission blocker" vs "follow-up review" are distinct statuses.
5. **SAMPLE denominators.** Coverage = conclusive reviews / selected panel (unknown/uncertain rows count as **missing**, not conclusive); issue rate = affected conclusive rows / conclusive rows. Panels under 100 round to the nearest whole row; the ≤1% FULL waiver requires explicit pinned approval, not automatic.
6. **Admission-event times.** The 07:20–13:51 UTC window in the 13:55 entry is the **landing/ledger window**; cited commit authorship times are 04:35–07:12 UTC the same day [P]. Landing times may lag commits; the two are distinct timestamp types.
7. **Revision-13→18 population growth (89,245)**: the rev-18 snapshot (~02:00 UTC) predates the 2026-11-15 admission commits, so the growth comes from newly converted sources (including the five filtered private Code releases, 9,133 tasks) plus folding/display changes — not from those admissions. Admission-state changes after 02:00 are ledger-level and separate. The 955,646 not-shown instances are the display-filter partitions (folded overlap groups, filtered and pending cards) per the reference's disjoint categories.
8. **Public per-release quality outcomes (previously omitted)**: rebuilt Unitsyn SAMPLE 54 findings / 100 reviewed → **BAD**; TACO-laion 25/100; PyMethods-small 37 findings in a 100-task panel; raw Stack 13 findings [P, commits above]. "Admissions complete" describes pipeline execution, not content quality; per-release reviewed/affected/missing counts are bound in the linked commits.
9. **Fidelity census (a)/(b)/(c) class counts**: not yet compiled into this record at the 13:51 cut; classification is per-source in the registry [M]. Recorded as a gap, not a completed result.
10. **Curriculum (marin#11490 [P])**: baseline capability placement covered 23,956/4,206,876 task instances (0.569%); a new nearest-placement default covers 100%; five levels; similarity thresholds 0.55/0.75 versus the prior 0.30/0.30; coverage is explicitly not semantic correctness; the export was staged, and revision 18 did not select it.
11. **Overlap/folding**: the 741→837-pair figures are successive NN-overlap survey pair counts (candidate duplicate-risk pairs) [R]; the recorded head also contains a separate 71-pair scan with seven duplicate pairs and one partial overlap [P]; aime24/e2egit repairs in progress; the uncommitted folding diff was handed to a separate worker [R].
12. **Session fragments**: future entries will not repeat internal session identifiers; the revision-1 note restated one only to flag it; it is not evidence.

**Glossary addenda**: canonical source families — Unitsyn (synthesized Python units), TACO(-laion) (code-competition dataset), Tau (dataset family), raw Stack (Stack Exchange-derived), Openreview / Open Reasoning (separate families: peer-review source vs reasoning corpus); "spec-accurate + QA" refers to the QA-evidence tier (question-answer evidence), while "content QA" and "quality assurance (QC)" refer to review of task content — each use in this record refers to the QA evidence tier, and QC is always written QC.

**Limitations addenda**: run-bound per-lane manifests (command/config/commit/input/seed/hardware/output) are not yet published here; sanitized aggregate receipts for central admissions are not yet published — public commits currently bind registry pointers, not inspectable counts; compute accounting remains incomplete (known: curriculum experiment 46.892 requested CPU-hours / 1.591 measured process CPU-hours [P]; pass-3c 8×H100; Puzzles attempt 4 on 8×GB200; queue-gated waits are not charged without evidence).
- Evidence: [P] links above; [M]/[R] as labeled.
- Interpretation and limitations: as stated.
- Decision and next action: fresh cold read round 3 next.

### 2026-11-15 14:44 UTC - Revision 3 after cold-read round 3: authoritative synopsis, three factual corrections, lane/hold tables, timeline

- Question or hypothesis and reason for trying it: fresh cold-read round 3 returned NOTES (34/32 items) with three residual factual errors. This entry corrects them and consolidates the current state. Entry headings are editorial publication labels; git commit time is the publication evidence.
- Published source revision: recorded cut `96fc0736fc95d8f21d11e1c03dfca108d568b220` (13:51 UTC); live head moved thereafter (observed `279a2655b3…` 14:18:12 UTC).
- Command or material configuration: public evidence: https://github.com/marin-community/marin/blob/96fc0736fc95d8f21d11e1c03dfca108d568b220/docs/references/task-curation.md (SAMPLE tolerance section; admission policy); https://github.com/marin-community/marin/issues/11490; https://github.com/marin-community/marin/commit/fd670c41d16bd52e259e11c1562f6d4f55f408a3; https://github.com/marin-community/marin/commit/0eaf9b7fb447a4eaf1f528ac47c82ae7b2695407.
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart:

**CURRENT SYNOPSIS (authoritative as of this entry; supersedes earlier "Current state"/"Open items" blocks, retained as history)**

- Technical question (how much faithful-environment fidelity was achieved?): **unanswered at this cut** — the per-source (a)/(b)/(c) fidelity census is not compiled into any public artifact yet; producing it is an owned next step (owner: audit lane; criterion: sanitized census published with denominators and exclusions). Admission/QC/preview/publication work proceeded in parallel.
- Counts (publicly reported aggregates [P]; per-release identities/counts are registry-bound [M]): nine QA releases = 431,389 tasks (raw Stack 110,600 + five pins 110,152 + final three 210,637); fourteen Code releases = 42,406 tasks (41,906 TRAIN + 500 EVAL); eight empty Code releases typed-excluded; five filtered private Code releases (9,133 tasks) evidence-only pending qualification.
- Hold state (as of this entry): production Atlas catalog publication (v76), TaskCompendium dataset publication, and controller rollout remain **held** pending explicit user decision; private preview deployments and private dataset release preparation were authorized and proceeded. marin#11277 OPEN/DRAFT (merge readiness includes outstanding CI/validation — separate from the user's publication authorization). marin#11120 is the related Snowball pipeline PR (open/held); its coupling to the Atlas hold is that both await the user's publication decisions.
- Lanes still open: FULL-QC remainder (Stack, Open Reasoning, OpenSWE repairs); Puzzles (FULL review + grading census + decontamination receipt); difficulty continuation (no current solver measurements publicly available at this cut — PR #11421 states results not yet collected; a qualified 116-measurement cut is historical); overlap/folding (uncommitted diff handed to a separate worker); Tau repair (two censuses bound; execution qualification outstanding); derived-tags census (built; receipts pending first publication); curriculum integration (nearest-placement export staged, not selected by revision 18).

**Corrections**

1. **SAMPLE rounding** (replacing revision 2's wrong rule): maximum missing rows = **floor(0.05 × attempted rows)**; equivalently minimum conclusive = n − floor(0.05n). Example: a 30-row panel allows floor(1.5)=1 missing row (not 2); a 99-row panel allows floor(4.95)=4 (not 5). A FULL waiver additionally requires explicit pinned approval.
2. **Curriculum thresholds reversed in revision 2**: the **original** minimum similarities were 0.55/0.75 (subject/capability); the **new** nearest-placement default is 0.30/0.30 (per marin#11490 and the pinned Atlas reference [P]). Coverage (100%) is explicitly not semantic correctness.
3. **Admission authorship-time window completed**: the cited 2026-11-15 admission commits span **04:35:37Z (Stack) through 12:46:30Z (TACO-laion completion)**, including rebuilt Unitsyn at 12:29:25Z; the revision-2 range (04:35–07:12) covered only the early QA/PyMethods/CrossCode subset. Landing times remain reported-private.
4. **QA terminology fixed**: "QA source lane" = the question-answering dataset category (the nine QA releases); "quality-assurance review" = QC/content review. Fidelity classes (a)/(b)/(c): (a) spec-accurate with QA evidence, (b) spec-accurate without QA evidence, (c) spec-vs-runtime mismatch — per the normative reference [P]. SAMPLE/FULL oversight levels are review levels, not dataset categories.
5. **Preview-18 chronology resolved**: the reference documents (dated to the package preparation) describe revision 18 as 2026-11-14 — that is the **package preparation date**; deployment and hosted verification occurred **2026-11-15 ~02:00 UTC**. Approval states: revision 14 was built/deployed but its publication approval was held; the first approved successor after 13 is recorded in the preview lane's receipts [M].
6. **Pass-3c runner**: the failing runner is an **unpublished script** (not a public file at the cited revision); "reproduced twice" and the telemetry figures are **reported** findings of the local audit [R] until a sanitized reproducer/receipt is published. The runner path/fix will be linked when the repair lands.
7. **Puzzles reviewer binding**: the public binding commit identifies **seed 0 and the v3 reviewer** for the SAMPLE run [P]; FULL-QC reviewer routes (v2/v3, shared-relay) are distinguished in the normative reference; reviewer calibration/blinded agreement is unmeasured for these results.
8. **Population reconciliation limit**: beyond the 9,133 new-conversion tasks, the 89,245 revision-13→18 growth is **not publicly reconcilable** from this record (per-source delta ledger is registry-bound [M]); display folding changes "shown" counts, not total population; hidden partitions include folded groups, filtered/pending cards, and QUICK/BAD hiding categories per the reference.
9. **Control taxonomy** (per the references): original-oracle population filtering (full population, required); workspace negative controls (agentic, graded sources; must fail); generic sampled grader checks (95% sampled-control pass threshold can skip unsampled validation — a distinct control); content review (SAMPLE/FULL). Puzzles' empty-input test is a degenerate-input diagnostic distinct from workspace NoP (which is N/A for answer-graded tasks).
10. **Privacy**: neutral noreply commit identity is in effect for commits after the initial ones; already-published commit metadata (initial commits expose a personal workstation email) requires repository-owner remediation — flagged to the coordinator, no history rewrite performed here. Internal session fragments will not be repeated; existing exposure is flagged.

**Chronological event timeline (compact)**

| Date (UTC) | Event |
|---|---|
| 2026-11-12 | Campaign start; BRIEF; fidelity audit scope; draft PR #11277 opened 21:09:15Z |
| 2026-11-13 | Difficulty roster decision; ~40 child PRs merged to atlas-env-audit |
| 2026-11-14 ~09:11 | Codex quota handoff; standing user policy |
| 2026-11-14 12:29–16:13 | Branch-tip push incident resolved; merger retired 16:13 |
| 2026-11-14 (rulings) | Non-agentic exemption; no-oracle executability drop; NoP retry policy; legacy adoptions 311/249/18 |
| 2026-11-14 14:04–21:36 | Preview revisions 13–17; rev-13 population verified; pub-14 approval held |
| 2026-11-15 00:08–02:00 | Contrast-audit landings; preview revision 18 deployed/verified ~02:00 |
| 2026-11-15 02:10–07:45 | Puzzles SAMPLE attempt 4 completed (97/100 reviewed) |
| 2026-11-15 03:44–12:31 | FULL-QC pass-3c failure audited (deterministic crash) |
| 2026-11-15 04:35–12:46 | Admission commits (QA nine + Code fourteen) |
| 2026-11-15 (ruling) | SAMPLE 5%-missing tolerance (floor rule) |
| 2026-11-15 13:51 | Recorded cut 96fc0736; 13:55 catch-up entry published |
- Evidence: [P] links inline; [M]/[R] labeled.
- Interpretation and limitations: prior limitations stand; fidelity census explicitly pending.
- Decision and next action: fresh cold read round 4 next.

### 2026-11-15 14:54 UTC - Revision 4 after cold-read round 4: consolidated current facts, gate matrix, evidence index, timeline fixes

- Question or hypothesis and reason for trying it: fresh cold-read round 4 returned NOTES (36 items); residual fixes are consolidation, a few dated-event corrections, and explicit evidence links beside claims. This entry consolidates; earlier entries remain as history.
- Published source revision: recorded cut `96fc0736fc95d8f21d11e1c03dfca108d568b220` (13:51 UTC).
- Command or material configuration: see evidence index below.
- External inputs and artifact references: https://github.com/marin-community/marin/blob/96fc0736fc95d8f21d11e1c03dfca108d568b220/experiments/post_training/task_curation/README.md (v3-README statement), app README/RELEASE.md (preview lane docs), https://github.com/marin-community/marin/blob/96fc0736fc95d8f21d11e1c03dfca108d568b220/docs/references/task-curation.md.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart:

**CURRENT FACTS (consolidated; supersedes all earlier "Current state" blocks)**

- Central question status: the (a)/(b)/(c) fidelity census is not compiled publicly; the campaign's core "how much fidelity was achieved" is **unanswered at this cut** (owned next step: audit lane; criterion: sanitized census with denominators/exclusions). The 310 live IDs are the live source-ID population, not the runnable-agentic denominator; the 28 grandfathered MarinSkyRL blends are excluded from the fidelity population entirely (their exclusion is defined in the BRIEF [R]).
- Admissions (publicly reported aggregates; per-release identities in registry [M]): nine QA releases = 431,389 tasks (raw Stack 110,600 + five pins 110,152 + final three 210,637); fourteen Code releases = 42,406 tasks (41,906 TRAIN + 500 EVAL, incl. PyMethods-small (TRAIN converter, 496/500 retained) and CrossCode-Python 500 EVAL); eight empty Code releases typed-excluded; five filtered private Code releases (9,133 tasks) evidence-only pending qualification.
- Public quality outcomes (per-release SAMPLE panels; "findings" = findings in commit prose, distinct-task counts not always established): rebuilt Unitsyn 54/100 (BAD), TACO-laion 25/100, PyMethods-small 37/100, Stack 13/100, CrossCode-Python 1/100, final-three-QA aggregate 134 sampled findings across the nine QA sources [P commit messages]. Puzzles 22/97 affected (22.68%).
- Holds (authorization states at this cut): private dataset release preparation/upload — authorized (new private releases are the policy mechanism); private preview deployment/verification — authorized; production Atlas catalog publication (v76) — HELD; public TaskCompendium dataset release — HELD; controller rollout — HELD; marin#11277 merge — pending (CI/validation outstanding, then user decision); marin#11120 — open/held (Snowball pipeline; awaits the same user publication decisions).
- Gate matrix (current, per the curation reference [P]; effective 2026-11-15): original-oracle full-population filtering — required for all sources; workspace NoP negative controls — agentic workspace-graded sources only, must fail; verifier-deadline/source-error outcomes count as non-passing **annotations** (admissible without a scored control failure), unresolved infrastructure outcomes stay pending; generic sampled grader checks — 95% sampled-control pass threshold (distinct control); content review — SAMPLE level (fixed ≤100 panel, conclusive = n − floor(0.05n), issue rate = affected/conclusive) with FULL as follow-up for most lanes; Puzzles lane exception: FULL review required before source admission (lane-specific standing decision [R]; authority: campaign lane records); legacy adoptions (311/249/18) install no pins. Waiver: ≤1% unreviewed requires explicit pinned approval.

**Evidence index (major claims → public links; items without a public link are [M]/[R])**

| Claim | Public evidence |
|---|---|
| raw Stack admitted, 110,600 tasks | https://github.com/marin-community/marin/commit/c707bfbd5e872db9439f5f39dc8af1405a86e85b |
| five QA pins, 110,152 tasks | https://github.com/marin-community/marin/commit/635e7b907f95ea7a25d4be93e60dbecc6faedb0c |
| final three QA pins → 431,389 total; 134 sampled findings | https://github.com/marin-community/marin/commit/8ea14c762f158e9e9a6cc5bab53b3cb328e7f23e |
| Code lane complete, 42,406 tasks (41,906 TRAIN + 500 EVAL) | https://github.com/marin-community/marin/commit/0eaf9b7fb447a4eaf1f528ac47c82ae7b2695407 |
| rebuilt Unitsyn SAMPLE (54/100, BAD) | https://github.com/marin-community/marin/commit/fd670c41d16bd52e259e11c1562f6d4f55f408a3 |
| PyMethods-small admitted (TRAIN converter; 496/500) | https://github.com/marin-community/marin/commit/2d8202d0071a3ac9bd57efc4e92043c73bb04df5 |
| CrossCode-Python EVAL admitted (500 EVAL; 1 finding/100) | https://github.com/marin-community/marin/commit/c9f0235d8e8d6bf9e739caa907d213f9952ffc35 |
| Puzzles SAMPLE binding (seed 0, v3 reviewer) | https://github.com/marin-community/marin/commit/4bdbc7eeed5a861e0bb22d25a434dd5a98744484 |
| 71-pair overlap scan (7 duplicates, 1 partial) | message of cut commit 96fc0736fc95d8f21d11e1c03dfca108d568b220 |
| curriculum experiment (0.569% → 100% placement) | https://github.com/marin-community/marin/issues/11490 |
| difficulty protocol (results not yet collected at its snapshot) | https://github.com/marin-community/marin/pull/11421 |
| SAMPLE tolerance rule (floor(0.05n)) | task-curation.md SAMPLE section at the recorded cut |

- Timestamp corrections: the initial catch-up entry was committed at **14:07:50Z** (its "13:55" heading is an editorial label); the contrast-audit landing window is **00:08–02:15 UTC** (the synopsis's 02:00 was the deployment time); the "Nov 13 ~40 child PRs merged" count is **approximate** (evidence: the example PRs #11439/#11444/#11421/#11436/#11429 and the atlas-env-audit log; no exact count compiled).
- Documentation-status notes: the task_curation README sentence "The v3 YAML has not yet been exercised by a live GPU job" is a **historical statement predating the recorded cut** — the recorded Puzzles SAMPLE run (seed 0, v3 reviewer, bound by commit 4bdbc7eeed) and pass-3c activity used v3; updating that README requires an authorized source revision (owned action, not a logbook change). The app README's "SAMPLE requires complete review" text is **obsolete** relative to the floor(0.05n) tolerance rule — same disposition.
- Consolidated open work (owners/criteria; supersedes scattered lists): (1) fidelity census publication — audit lane; (2) preview revision 19 publish+verify — preview lane; (3) marin#11277 CI/validation then merge decision — code owners + user; (4) pass-3c repair (runner bug + reasoning-truncation reviewer strategy) — FULL-QC lane; (5) Puzzles FULL review + grading census + decontamination receipt — FULL-QC lane; (6) FULL-QC remainder (Stack whole-source review, Open Reasoning, OpenSWE) — FULL-QC lane; (7) legacy-adoption gates (whole-source collection, global NegativeControlPolicy, content QA) — qualification lane; (8) difficulty continuation measurements — difficulty lane (no current public results); (9) overlap/folding landing + fold-group decision table — folding worker; (10) Tau execution qualification; derived-tags census first publication — respective lanes; (11) curriculum nearest-placement integration decision — catalog lane; (12) run manifests + sanitized receipts publication — evidence owner (secretary can compile from public materials only); (13) commit-metadata remediation decision — repository owners (flagged to coordinator); (14) documentation fixes above — authorized source revision.
- Privacy status: commits after the initial logbook/index commits use a neutral noreply identity; remediation of already-published personal-workstation email metadata is a repository-owner decision (tracked here and flagged to the coordinator). Internal identifiers (session fragments, node labels) are retained in the historical entries; they are operational metadata and will not be repeated.
- Evidence: table above; [P]/[M]/[R] as labeled.
- Interpretation and limitations: as previously recorded.
- Decision and next action: fresh cold read round 5 next.

### 2026-11-15 14:59 UTC - Revision 5 after cold-read round 5: gate-matrix scoping, SAMPLE minimum wording, evidence-label split, index-pin note

- Question or hypothesis and reason for trying it: fresh cold-read round 5 returned NOTES (34 items); concrete fixes below.
- Published source revision: recorded cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`; normative reference sections: task-curation.md lines 38–56/76–80 (no-oracle static-retention path), 102–106 (hold scope), 2507–2525 (SAMPLE tolerance).
- Command or material configuration: n/a.
- External inputs and artifact references: https://github.com/marin-community/marin/blob/96fc0736fc95d8f21d11e1c03dfca108d568b220/docs/references/task-curation.md.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart:

**Corrections**

1. **Gate matrix scoping fixed** (revision 4 overgeneralized): original-oracle full-population filtering is required **wherever an original oracle exists**; no-oracle sources take the **static-retention path** (retained without oracle filtering; per-task executability proof suspended per the 2026-11-14 ruling; other applicable gates remain). Workspace NoP controls apply to agentic **workspace-graded** sources; answer-graded sources use graded controls where applicable (Puzzles' empty-input diagnostic is such a graded control, distinct from workspace NoP).
2. **SAMPLE minimum is an inequality, not an equality**: conclusive reviews must be **at least n − floor(0.05n)**; the rule sets a maximum on missing rows.
3. **Evidence labels split** (revision 4's evidence index): a link to a marin commit establishes a **public report** (the commit message asserts the aggregate); it is not a **publicly inspectable measurement receipt** — the underlying receipt rows live in the private registry. In the index table, the "Public evidence" column proves public reporting only; no column currently available proves independent inspectability. Publishing sanitized aggregate receipts (inputs, counting rules, results, digests) remains open work requiring owner authorization.
4. **Fidelity-class QA definition**: in the user's 2026-11-12 (a)/(b)/(c) addendum [R], the evidence distinguishing (a) from (b) is the **content-review/qualification evidence tier** (the checks defined by the QC lanes for that source class) — not the question-answering dataset category. The lettered classes originate from the user ruling, not from the normative reference, which does not enumerate them.
5. **Hold wording**: **public** TaskCompendium dataset release is held; **private** remediation release preparation/upload is authorized (and occurred). Production catalog publication (v76), controller rollout, and the merge decision are held as recorded.
6. **Timestamp note**: revision-4's "14:54 UTC" heading was an editorial label; its git commit time is 14:52:38Z. Heading times in this record are editorial labels; git commit metadata is the publication evidence.
7. **Documentation-status**: the curation-README v3 sentence and app-README SAMPLE-completeness sentence are obsolete relative to the recorded cut (noted in revision 4); both require an authorized source revision to fix — linked here so newcomers see the conflict explicitly rather than discovering it.
- Evidence: [P] links inline; [M]/[R] labeled.
- Interpretation and limitations: unchanged; the fidelity census remains the central unanswered item.
- Decision and next action: fresh cold read round 6 next.

### 2026-11-15 15:12 UTC - Revision 6: sanitized evidence ledgers published (per user publication ruling)

- Question or hypothesis and reason for trying it: cold reads rounds 1–6 identified private-evidence inaccessibility as the main SATISFIED blocker. The user ruled (2026-11-15, via the coordinator): everything except (1) cluster secrets/credentials and (2) the DATA itself (task text, prompts, model outputs, dataset rows — data requires separate sign-off) may be published. This entry publishes sanitized ledgers compiled from local records.
- Published source revision: recorded cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`.
- Command or material configuration: see evidence files (run-manifests.md carries actual commands/entrypoints).
- External inputs and artifact references — new evidence files in this directory (all metadata-only; no task text, prompts, model outputs, or dataset rows; no credential values):
  - [evidence/fidelity-census.md](evidence/fidelity-census.md) — fidelity-census building blocks: (a)/(b)/(c) definitions; denominators (81 runnable / 311 live / 326 union); 66-labeled environment-result inventory; excluded-fourteen (53,842 tasks); nine pending measurement categories with typed rejection ledgers; 28 grandfathered blends excluded. **A single aggregate (a)/(b)/(c) tally over the final population is not yet compiled** — the components are here.
  - [evidence/admission-ledger.md](evidence/admission-ledger.md) — full nine-QA-release ledger (431,389 tasks; per-source revisions/admission commits; SAMPLE 900 reviewed / 134 findings; registry entry SHA) and all fourteen Code releases (per-panel findings, pins; Unitsyn 54 / TACO 25 / PyM 37 / CrossCode 1 anchors; code14 61.138878 GPU-h; token receipt: 12,411,285 prompt / 1,629,030 completion / 1,199,817 reasoning over 1,381 cached requests).
  - [evidence/sample-panels.md](evidence/sample-panels.md) — the 245-panel SAMPLE aggregate (168 GOOD / 72 SOME_ISSUES / 5 BAD; 23,563 conclusive; 455 unavailable; 12 uncertain; ReviewerSpec v3 GLM-5.3 seed 0) with checkpoint binding commits; per-release panels.
  - [evidence/run-manifests.md](evidence/run-manifests.md) — run manifests: GLM serving fleet (48 roots / 384 GPUs, 1,316 GPU-h); 15-job CPU repair wave; Qwen duplicate review (1.6893322222 H100-h); difficulty; NoP controls; puzzles census; contrast-audit captures (3,542 captures / 6.2M measurements); pass-3c job with code SHA and frozen inputs.
  - [evidence/pass3c-failure.md](evidence/pass3c-failure.md) — the failing runner is local at `/experiments/active/atlas-full-qc/or-pass3-v3c-runner-v1.py` (SHA-256 `abf36edfa1c2fcc16d49f5aa777aec2e537d6459aa5ec260a6e753b6e4ce0db1`, 10,733 bytes) with the failing ~15-line loop reproduced (code, no secrets) and the tuple-unpack double-crash characterization; 42,544/74 finalize and probe-A qualification context. **Telemetry block (710 requests / 8.97M tokens / 92.1% length-finish / 656 tok/s / 239 s mean / SM-tensor gauges) is not in the local atlas-env-audit REPORT artifacts — it originates from the local claim-audit report [R]; fleet-level gauge semantics are documented instead.**
  - [evidence/snapshot-reconciliation.md](evidence/snapshot-reconciliation.md) — rev-13 → rev-18: 4,117,631 = 4,206,876 − 89,245 exactly; rev-13 role splits; 16-retired/185,906 and 326/4,392,782 reconciliation; QUICK/hidden display partitions; head `072140fa956c…`; the 89,245-task SWE shard `84284780`'s eight source IDs remain unattributed.
- Environment and hardware when relevant: in run-manifests.md.
- Result, including failure or restart: reconciliation notes — the "Stack 13" and "Puzzles 22/97" anchors in earlier entries are public commit-message reports; the local panel artifacts reviewed here surface adjacent figures (HardMath 56/97, Unitsyn Large 49/96) but not those exact denominators — their bindings remain registry-bound [M] and are flagged for reconciliation. The "curriculum 46.892 requested / 1.591 process CPU-hours" figure likewise awaits its public source anchor.
- Evidence: the six files above (compiled 2026-11-15 ~15:05 UTC from [R] local records; no HF access needed — local mirrors sufficed).
- Interpretation and limitations: attempt-level QC rows and the final (a)/(b)/(c) tally remain uncompiled; data (task text/outputs) requires user sign-off per the ruling.
- Decision and next action: fresh cold read round 7 next.

### 2026-11-15 15:45 UTC - Revision 7: evidence ledgers corrected and extended after cold-read round 7; publication-rule note

- Question or hypothesis and reason for trying it: cold-read round 7 found defects in the round-6 evidence ledgers (arithmetic errors, misleading repair framing, commit-mapping errors, and notebook-vs-ledger state conflicts). This entry records the corrections; all are applied in the evidence files with updated sources.
- Published source revision: recorded cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`; per-file source-cut and evidence-cutoff lines added.
- Command or material configuration: n/a.
- External inputs and artifact references: local atlas-env-audit REPORT.md and lane files [R]; public commits/issues [P].
- Environment and hardware when relevant: in run-manifests.md.
- Result, including failure or restart:

**Corrections (applied in evidence/)**

1. **Pass-3c state reconciled with the notebook**: at the 13:51 recorded cut the repair was PENDING (unfixed runner, no final artifact; claim audit 12:31:09 with 43/75). Per local REPORT evidence, LATER: all five runner variants fixed; GPU-free finalization produced an artifact; 74 invalid rows remained; review accepted_incomplete under a pinned waiver; Probe A (2/10 = 20%) did not qualify variant A — dated state table added, post-cut events flagged. The published bug mechanism is now a composed minimal failing example from the pre-fix runner (code only): `run_batch`/`repair_batch` returns `(batch, result)`; the outer `zip` stores the tuple; the later `result.repaired_task_ids` access fails. Causal explanation labeled reported until a durable traceback exists.
2. **Fractions fixed**: unreviewed = **74/42,618 ≈ 0.174%** (42,544/42,618 ≈ 99.826% is the reviewed fraction). Population transitions tabulated (42,618 / 118 repair subset / 43+75 at cutoff / 44 recovered pre-cancel / 42,544+74 final; 74-vs-118 overlap unknown).
3. **Attempt/telemetry reconciliation**: attempts are indices 0/1/2; 03:23:16 = submit-script mtime vs 03:44:32–36 = actual acceptance; 710 distinct response IDs (12:31 receipt cutoff) vs 778 cache-manifest entries (13:15 cutoff) — definitions given, ~68 delta marked unreconciled; the 8.97M/92.1%/656 tok/s/239 s/zero-preemption block stays attributed to the local claim-audit [R], distinct from fleet telemetry.
4. **Fidelity census clarified**: the file is an interim building-blocks record — the aggregate (a)/(b)/(c) tally over the final population is **NOT measured**. Inventory recount: 281 rows, one label each; fidelity-outcome labels sum to **66 = 16 faithful + 2 substituted + 24 reply_only_rewrite + 24 missing** (previous 17/2/15/27 wrong); docker_image (15) and native task.toml (51) are environment kinds; not_executable_inventory 149. Dated crosswalk published (81 runnable / 133 measured / 311 → 310 live / 326 unique / 328 alias rows); 24 of 28 grandfathered blends carry missing-labels in raw inventory while the census excludes all 28; ARC-inductive = ARC-transductive (both `source_defect:grading_environment_mismatch`).
5. **Admission ledger fixes**: the "5% issue bar" is the SAMPLE **missing-reviews** tolerance (floor rule) — "repair" meant re-driving missing reviews to conclusive; Unitsyn/TACO keep 54/25 finding events with BAD/SOME_ISSUES per the public commits. Finding events vs distinct affected tasks distinguished per panel; labels reported-until-countable. Raw Stack remapped to commit `c707bfbd5e`; 888/9/3 vs 900/134 resolved as **successive versions of the same nine QA panels** (reported-until-proven). The stray Code14 claim for `tasktrove-math_stack-qa` removed. Original counts and typed drop reasons added where attested; the PyMethods 4-task gap marked unresolved (dropped pre-oracle-filter). Unitsyn lineage table: quarantined release (4,991 large payloads) → corrected successor `tasktrove-unitsyn-python-rebuilt-v1` (4,950, BAD) → renamed large (4,951) / original large (4,949) / legacy small (491) / small exp_rpt (487).
6. **SAMPLE panels**: Stack 13 and Puzzles 97/100/3/22.68% now cited as **public commit reports** (`c707bfbd5e`, `4bdbc7eeed`); the "missing anchors" framing withdrawn. What remains unavailable: independently inspectable panel metadata/measurement receipts and distinct-task counting proof. CrossPy "GOOD 2/100 conclusive" split (Result vs conclusive vs selected; the 2-conclusive-vs-≥95% rule tension flagged unresolved; candidate panel distinguished from the admitted CrossCode 1/100 panel by pin/date). Rollup map: 245 → 243 → 242 and registry 247 reconciled as far as local files allow. Four-attempt Puzzles table published (the ~02:10–07:45 wall window vs the 26-minute 3.281682-GPU-h serving run identified; registration = public commit 11:05:16 UTC; the lane STATUS's ~07:45 clock flagged inconsistent).
7. **Run manifests**: curriculum cost anchored to marin#11490 [P] (46.892 requested / 1.591 process CPU-hours, four jobs, zero GPUs); the difficulty attribution removed. v2 fleet (262,144-token context) vs v3 workers (524,288) distinguished with per-result reviewer identities. Cost scopes separated (pass-3c inference cost unknown; no overlapping sums). Complete submitted argv/config added where present locally.
8. **Snapshot reconciliation — major correction**: the 4,206,876 total **already existed in revisions 11/12** as the production/live-310 population; the 89,245 delta vs revision-13's 4,117,631 = the **eight mechanically excluded populations** (tool_provider_required ×2, backend_contract_unsupported ×1, unpinned_environment ×5; itemized with IDs; disjoint from the 44,634 empty-Code set) — **the notebook's earlier "growth from new conversions" explanation is superseded**. "Revision 18" as a hosted deployment is not established by these local files (preview-lane evidence [M]). 16-retired/185,906 and 955,646 per-source breakdowns remain aggregate-only locally.
- Publication rule (standing): everything except cluster secrets/credentials and the data itself (task text, prompts, model outputs, dataset rows — data requires user sign-off) is published; no personal information of anyone (third-party identifiers written as "another user"); diffs grepped for emails/@-handles before every push.
- Evidence: evidence/ files as corrected (compiled-at headers updated).
- Interpretation and limitations: unchanged.
- Decision and next action: fresh cold read round 8 next.

### 2026-11-15 16:25 UTC - Hourly catch-up (15:20–16:25 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from the recorded 13:51 cut `96fc0736…` to head `9fecbbf0b4c976bb48c66ae0a2ff28792af8304d`, landed 16:03:50 UTC (atlas-push landings log [R]; PR #11277 updated 16:03:50Z, still OPEN [P]).
- Command or material configuration: evidence: landings.jsonl tail; `git log 96fc0736..9fecbbf0`; commit metadata.
- External inputs and artifact references: https://github.com/marin-community/marin/pull/11277 (head now `9fecbbf0…`).
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Preview revision 19 docs synced**: commit `129fd7505a…` "[docs] Sync Atlas docs with preview revision 19" (authored 10:23:14-05:00 = 15:23:14 UTC; landed 15:24:24 UTC) touches docs/references/rl-data-atlas.md, task-curation.md, and the app guide.html — the revision-19 rollout is in motion. Hosted deployment/verification receipts remain preview-lane evidence [M].
  - **Commit-time query-cache writes pinned for committed reviews** (`1323934187…`, landed 15:44:13 UTC) — QC query-cache behavior change.
  - **Difficulty root resumed from its own sealed output after a preemption** (`9fecbbf0…`, authored 11:01:52-05:00 = 16:01:52 UTC; landed 16:03:50 UTC) — difficulty_campaign/evidence/recovery/trials changes plus tests; the first *actual* preemption-related recovery in this window (contrast with the audited D-run self-cancels).
  - Supporting CI/infra commits landed 15:20–16:04 UTC (test-timeout scoping, preflight scenario separation, storage-layout isolation, supervisor termination handling, FULL-QC eval-labeled retention acceptance, served-row count projection keeping qualified oracle counts).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: hosted revision-19 verification not yet publicly attested from this record; head movement continues — timestamps here are commit/landing times.
- Decision and next action: next hourly check verifies revision-19 hosted verification and any further landings.

### 2026-11-15 17:07 UTC - Revision 8 after cold-read round 8

- Question or hypothesis and reason for trying it: cold-read round 8 (two independent reports: the campaign read and the PR #50 read) returned NOTES (35/46 items) with concrete ledger conflicts. This entry records the fixes; all are applied in the `evidence/` files in place with "Updated 2026-11-15 17:05 UTC" markers, and the read-first note above now points here as the current synopsis. Earlier entries are preserved unedited.
- Published source revision: the recorded cut remains `96fc0736fc95d8f21d11e1c03dfca108d568b220` (13:51 UTC). The branch kept moving: head `9fecbbf0b4…` landed 16:03:50 UTC (16:25 entry), and by ~16:44 UTC the PR #11277 head was observed at `4e8752fd8e…` (three further commits: merge main with ProofNet held pending Atlas measurement `a11b0d3c76…`, curriculum preparation in the caller `108c4f85b2…`, curriculum storage init `4e8752fd8e…` authored 16:37:34 UTC) [P gh observation]. Evidence ledgers stay bound to the 13:51 cut; post-cut landings are flagged per fact, not merged in.
- Command or material configuration: n/a (documentation repairs only; no cluster commands executed).
- External inputs and artifact references: local [R] FINAL_STATUS-v2.json, code-fourteen-final-registration-ledger-v1.json, atlas-puzzles-qa-repair/STATUS.md, atlas-full-qc/REPORT.md:182/2214, atlas-env-audit/REPORT.md:25/27/37/65/117/225/1863/2203, claim-audits/fullqc-pass3c.report.md:75–88; public [P] marin commits c707bfbd5e/635e7b907f/8ea14c762f (author-time via gh api), PR #11277 head state, PR #11421 body.
- Environment and hardware when relevant: pass-3c serving gang corrected (see result 5).
- Result, including failure or restart:

**Fixes applied in evidence/ (round 8)**

1. **Evidence cutoff made per-fact (branch moved past the recorded cut).** All six ledgers now carry an explicit post-cut note: the recorded cut is `96fc0736…` (13:51 UTC); the branch advanced through `9fecbbf0b4…` (16:03:50) to PR head `4e8752fd8e…` (~16:44), including the preview-revision-19 docs sync `129fd7505a…` (authored 15:23:14 UTC — it makes the normative docs describe revision 19 admitting the 9 QA + 14 Code filtered private releases), the commit-time query-cache pin `1323934187…`, and the difficulty-root resume `9fecbbf0…`. **run-manifests.md's blanket "no fact postdates 13:51" was wrong**: its Probe A cost row (16.849362 GPU-h) comes from the 15:30 UTC REPORT entry and is now flagged post-cut in the header and in the §12 row. The admission/sample headers no longer treat the editorial "13:15 UTC" REPORT heading as proof of pre-cut pass-3c facts: pass-3c finalization's timing relative to the cut is **unknown**; last independently time-bound observation: repair pending at 12:31:09 (pass3c-failure.md §1 sharpened accordingly).
2. **QA release identities completed.** All nine QA rows in admission-ledger.md §1 now carry full private repo names and full 40-hex release revisions, transcribed from FINAL_STATUS-v2.json (previously six rows read "(per FINAL_STATUS)" and abbreviated revisions). §5's false claim that all nine repo names were already reproduced in §1 is corrected; the fourteen Code releases' repo names and converter identities are explicitly **not present in the local read set** (the code14 ledger carries source IDs, revisions and per-panel SHA-256 only) — measured-but-private, not reproduced, and no row is labeled as having an identity it lacks.
3. **Raw Stack commit mapping corroborated and the local discrepancy recorded.** FINAL_STATUS's per-source `admission_commit` fields confirm raw Stack → `c707bfbd5e` (public commit author 04:35:37 UTC), five pins → `635e7b907f` (04:54:03), final three → `8ea14c762f` (05:22:16). The full-qc REPORT.md:2094 parenthetical naming "rawStack/procOracle/procOpen" at `635e7b907f` is a local shorthand error (raw Stack's 110,600 cannot be inside the five pins' 110,152) — now recorded in admission-ledger.md §1.
4. **888/9/3 vs 900/134 panel lineage harmonized.** sample-panels.md §6's "a different nine-panel cohort/era" is withdrawn; both files now state the same position: the two sets share the same shape (nine 100-task seed-0 pinned-GLM-5.3-v3 panels), 888+9+3 = 900 with originals preserved and 11 new findings in repaired rows supports a succession reading, but no byte-level cohort-identity proof is in the read set — **neither same-cohort succession nor different-cohort is asserted as fact**.
5. **Pass-3c hardware reconciled.** The claim audit binds the serving gang as 8 GPUs, TP4×PP2 across s41sxs64/shbrxs64 (server startup 08:34:46.852 UTC, max_model_len 524288), with mean power 444–495 W/GPU — but **does not name the device type**. The earlier "8×H100" labels in revisions 1–2's cost lines are **not verifiable from the read set** and are superseded by "8 GPUs (TP4×PP2), device type unknown" (pass3c-failure.md §5 new row; run-manifests.md §1 caveat; do not infer from fleet GB200 defaults).
6. **FULL/SAMPLE numerators labeled with explicit fields.** Math Oracle FULL (sample-panels §3): GOOD Result = **5,020 issue rows among 57,253 conclusive** (8.7681% issue rate), 44 uncertain, 122 waived unreviewed; 57,253+44+122 = 57,419 = retained population (verified against REPORT.md:225). Code-oracle FULL example (§7): population 15,156 = 15,087 valid + 69 INVALID; 0 unavailable/0 uncertain; **1,026 affected conclusive rows** = 6.80% → provisional GOOD; top flags 706+217+193+72 = 1,188 **overlap across rows** (flag counts ≠ affected-row counts) (verified against REPORT.md:182, 2214). Whether affected rows are distinct tasks or issue events remains unestablished where the sources don't say.
7. **Affected tasks vs finding events.** The Puzzles panel tables now read "22 issue events (22.68% reported rate; distinct affected tasks not established)" instead of "22 affected tasks" — consistent with sample-panels §4's standing statement that distinct-task counting is not established; the rate is labeled reported (public commit `4bdbc7eeed` prose).
8. **Puzzles identities and durations completed.** Full IDs published: pin `adfcbd4a8ddefc9e69febc91931c08ba03372bad`, task-spec digest `8542395bb886f09d…`, records SHA `e9139b6d186d…`, artifact SHA `4e7295632a8c…`, registry revision `69eb7669aeb1aef3b10efd755cf8f8a9dd700042`, campaign record key. The frozen-queue SHA stays a prefix (the local STATUS.md itself truncates it — not locally available). Duration reconciled: 3.281682 GPU-h ÷ 8 GPUs ≈ 24.6 min created→finished, consistent with the rounded "26-minute" serving description; the ~07:20/~07:45 STATUS clock entries remain flagged inconsistent with the 11:05:16 UTC public commit (round-7 note stands).
9. **NoP terminology fixed.** run-manifests §5's expansion "no-oracle positive/negative control" is corrected to **workspace no-op negative control** (matching the glossary). The "24 root archive cases plus six production Trial cases PASS" item is relabeled: it is the **code-review verdict on the NoP feature commit `28ffc24…`'s review/verification cases** (REPORT.md:27, 37), not a record of no-op controls receiving reward. Positive Smartie explained: author-reported canonical **7,569 retained of 8,045 after 400 NoP-positive drops** (400 tasks whose no-op control received reward → oracle flagged broken → dropped); the residual 76-task difference is **not itemized in the read artifacts**; final counts not independently verified; Smartie positive packet held (`shared_integrity_contract_missing`) [R REPORT.md:65/117/2203].
10. **Qwen duplicate-review cost relabeled.** "1.6893322222 H100-hours" → "1.6893322222 **controller-attempt hours**" in run-manifests §6 and §12 — the source labels the components controller-attempt hours; GPU-device type and any serving-width multiplier are not stated in the read artifacts, so device GPU-hours remain unknown.
11. **Fidelity census denominator and heading.** §2's stale "15 grandfathered missing rows" heading corrected to the actual **24 `missing:`-label rows** (body already correct). §6 now states explicitly that the denominator of any future (a)/(b)/(c) total is **not yet fixed** (81 / 133 / 310 / 326 / 328 candidates differ; no source-keyed manifest exists) — a fidelity tally has no defined denominator until that manifest is published; the aggregate tally itself remains **unmeasured**, not merely private.
12. **Snapshot semantics.** snapshot-reconciliation.md §1 now states the 4,206,876 live-310 total is a **strict-census baseline, not a deployed production selection** (v76 held; revision-18 hosted deployment itself not established by these files).
13. **Runner-source claim corrected.** run-manifests §11 no longer says pass3c-failure.md §2 "publishes the runner source": §2 is an **explanatory excerpt** (verbatim failing loop + composed minimal example); the full 246-line pre-fix runner is not published and full-run reproduction remains unavailable; the standalone example requires **Python ≥ 3.12** (`itertools.batched`).
14. **Measured-but-private vs unmeasured reaffirmed** (round-8 priorities 3): fidelity final tally — unmeasured; runnable manifests for most runs — unavailable (argv/config descriptions only; frozen queue/plan bodies and receipt objects private [M]); sanitized registry receipts — not published; Code14 repo names/converter identities — not in the local read set. No ledger implies availability that does not exist.
- Evidence: the six `evidence/` files as updated (17:05 UTC markers); [P]/[M]/[R] as labeled inline. Heading times in this record remain editorial labels; git commit metadata is the publication evidence.
- Interpretation and limitations: the round-8 notes items that are **outside this branch's editable surface** are recorded, not fixed, here: the issue #48 baseline/navigation comment (still pins revision 5 and pre-revision-2 wording) and PR #50's earlier read-first text are public GitHub surfaces — updating them is an owned issue/PR-lane action; commit-metadata remediation remains a repository-owner decision; the 245→247 registry decomposition, the CrossPy candidate's 2/100 qualification semantics, distinct-task counting proofs, and sanitized receipts remain open evidence work. The "~40 child PRs merged Nov 13" figure remains approximate/unverified and outside the quantitative results.
- Decision and next action: fresh cold-read round 9 next; the issue #48 navigation update and revision-19 hosted-verification check are the owned follow-ups.

### 2026-11-15 17:30 UTC - Hourly catch-up (16:25–17:30 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `9fecbbf0…` (16:03:50 UTC) to head `b35a72d3335dcd03e5f711cb09c2436ebf374bfe`, landed 17:11:26 UTC (atlas-push landings log [R]).
- Command or material configuration: evidence: landings.jsonl tail; `git log 9fecbbf0..b35a72d3` (fetched).
- External inputs and artifact references: landing records 16:44:44 UTC `4e8752fd8e…`, 17:03:10 UTC `d5d5603bf8…`, 17:11:26 UTC `b35a72d333…`.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **License-gating wave** (four commits, 17:03–17:11 UTC): `[Atlas] Gate new ingestion and admission on pinned license evidence` (`96007b259e…`), `Bind reviewed license receipts at standalone entry points` (`8639c56ac1…`), `Update Data Belay callers to the shared license policy` (`a4f7dfecbe…`), `Isolate reviewed-license fixtures from evaluation result roots` (`b35a72d333…`) — new ingestion/admission now requires pinned license evidence.
  - **Main merge bringing the Lean proof lane** (`d5d5603bf8…` 17:03 UTC, merging `5699cf726e…` "Add a Lean v4.34.1 proof grader and ProofNet-Verified (#10879)" and `7cc6260615…` NuminaMath-LEAN training data) and `Merge main and hold ProofNet pending Atlas measurement` (`a11b0d3c76…`) — a new source family (Lean proofs) enters the pipeline with ProofNet explicitly held pending Atlas measurement.
  - **Numina converter** merged pending Atlas measurements (`d5d5603bf8…`).
  - **Single-worker curriculum preparation** (`108c4f85b2…` run in the caller, `4e8752fd8e…` storage init; landed 16:44:44 UTC).
  - Also incoming from main: `Document per-task research logbook records (#11450)` — relevant to this record's conventions.
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: these are code/merge landings; their measured effects (admissions, measurements) are not yet attested in this record.
- Decision and next action: next hourly check watches for the first Numina/ProofNet/Lean measurement and license-gated admissions.

### 2026-11-15 17:46 UTC - Revision 9 after cold-read round 9: Puzzles attempt-4 durations reconciled to two clock bases, pass-3c current state restated, finding-count semantics split

- Question or hypothesis and reason for trying it: cold-read round 9 returned NOTES (42 items) with concrete defects: the Puzzles "24.6-min = rounded 26-min" equality is arithmetically wrong (24.61 does not round to 26); the still-prominent revision-4 open-work list reads pass-3c as unfixed/awaiting a rerun although the evidence files record the repair, finalization and waiver; and finding-count semantics blur event counts with task counts. This entry corrects them; revision 8's and earlier entries are preserved unedited and the affected statements are superseded here.
- Published source revision: the recorded cut remains `96fc0736fc95d8f21d11e1c03dfca108d568b220` (13:51 UTC); branch head continues to move past it (post-cut landings are flagged per fact in the evidence headers, not merged in).
- Command or material configuration: n/a (documentation repairs; no cluster commands executed). Evidence files edited in place with "Updated 2026-11-15 17:42 UTC" markers: evidence/sample-panels.md and evidence/run-manifests.md.
- External inputs and artifact references: [R] `/experiments/active/atlas-full-qc/gpu-attempt-times-sample-qc-20261115T1310.csv` (Iris attempt-time receipt: created/started/finished timestamps per task attempt); [R] `/experiments/active/atlas-full-qc/REPORT.md` lines 118, 1920, 2214, 2220; [R] evidence/pass3c-failure.md §1/§7/§8 (round-7/8 compiled); [P] marin commit `4bdbc7eeed` (registration). Cold-read round-9 report local-only [R].
- Environment and hardware when relevant: attempt-4 serving hardware unchanged (8×GB200, cw-us-east-08a).
- Result, including failure or restart:

**1. Puzzles attempt-4 duration — two different clock bases, not one rounded interval (supersedes revisions 7–8's "consistent with the rounded 26-minute description").** The round-9 reader is right: 3.281682 GPU-h ÷ 8 GPUs = 24.6126 minutes does not round to 26. The two figures measure different spans of the same attempt, verified from the local Iris attempt-time receipt:
   - **3.281682 GPU-h** = the **inference serving child's** created→finished span: 2026-11-15 **10:16:38.693 → 10:41:15.450 UTC** = 24.6126 min (= 3.281682 GPU-h at 8 GPUs exactly).
   - The REPORT's **"26-min run"** [R: REPORT.md:2220] matches the **parent job/task root span**: created **10:15:23.258** → finished **10:41:38.728 UTC** = 26.2578 min created→finished (26.0756 min started→finished) — a wider interval that additionally covers the prepare-model stage (10:15:54–10:16:36 UTC) and pre-inference scheduling overhead (~1.6 min more than the inference child).
   Consequences recorded in the evidence files: attempt 4's serving ran **10:15–10:41 UTC** (job
   created 10:15:23.258 UTC per the CSV; the submit log's 05:15:18 timestamp is the same event on
   the operator's EST local clock), ending before the 11:05:16 UTC registration commit — the
   "26-minute serving run" labels in revisions 7–8 and the §12 cost-table row are corrected. The
   "~07:20 SUCCEEDED / ~07:45 LANDED" STATUS-clock entries and the "~02:10–07:45"
   attempt-sequence wall window remain inconsistent with both the attempt-time CSV and the
   registration commit and stay flagged unresolved (the CSV does not cover attempts 1–3).

**2. Pass-3c current state restated (supersedes revision 4's consolidated open-work item 4 as written).** Revision 4's open-work list still says "pass-3c repair (runner bug + reasoning-truncation reviewer strategy) — FULL-QC lane", which reads as if the repair is outstanding. Current state per evidence/pass3c-failure.md (§1 dated table, §7, §8): all five runner variants were fixed; GPU-free finalization from sealed receipts produced the artifact (42,544 REVIEWED / 74 INVALID; 74/42,618 = 0.173636% unreviewed); the finalization stands `accepted_incomplete` under the pinned ≤1% waiver; Probe A (2/10 valid = 20%) did **not** qualify variant A against the ~70% bar; **no full OR re-pass is authorized** (no variant B per standing order). What remains actually open: a durable traceback/reproducer for the reported tuple-unpack mechanism (the causal explanation stays reported), the ~68-entry 710-vs-778 counter reconciliation, publication of the runner source, and the optional 2-row probe-recovery overlay. The timing of the fix relative to the 13:51 recorded cut remains **unknown** (last independently time-bound observation: repair pending at 12:31:09); the waiver acceptance and Probe A result are post-cut facts.

**3. Finding-count semantics split (applied in evidence/sample-panels.md, "Finding-count semantics" note).** Two count types exist and are now separated: (i) **SAMPLE-panel findings are issue-EVENT counts** (Puzzles 22 events, Code14 anchors Unitsyn 54 / TACO 25 / PyMethods 37, the 245-panel aggregate); distinct-affected-task counts are not established for these panels in the read set, so event counts must not be divided by conclusive rows as if they were affected-task rates. (ii) **FULL-lane aggregates in atlas-full-qc/REPORT.md are row-level (task-row) counts** — Math Oracle "5,020 issues across 57,253 conclusive reviews (8.7681%)" [R: REPORT.md:1920] and code-oracle "1,026 / 15,087 conclusive (6.80%)" [R: REPORT.md:2214] — each retained row is one task of the retained population, so those numerators ARE distinct affected-task counts; their flag tallies (e.g. 1,188 overlapping top flags) are not. Result thresholds are computable only for type (ii).
- Evidence: [R] receipts/REPORT as cited; [P] commits as cited; evidence/ files carry 17:42 UTC markers.
- **Read order**: start with THIS entry (current corrections) → revision 8's fixes (evidence-file state as of 17:05) → revision 4's "CURRENT FACTS" consolidated block and gate matrix → revision 3's synopsis → the six evidence/ ledgers as the canonical operational detail (headers state their own evidence cutoffs) → earlier entries as history. Items 1–3 above supersede the corresponding statements in revisions 3, 4, 7 and 8.
- Secretary GitHub updates (owned surfaces this branch cannot edit — for the index/issue lane): (1) index PR #50's `index/48-atlas-campaign.md` calls its revision-7 pin "containing all corrections" although revision 8 corrected material facts — relabel to the newest pin (this revision) and refresh after it lands; its "revision-18 deployment" assertion should carry the evidence qualification (hosted deployment/verification receipts remain preview-lane [M]); it also needs its own one-line [P]/[R]/[M] legend; (2) issue #48's body and latest navigation comment still point to revision 5 and carry pre-revision-2 facts (PyMethods-small described as EVAL; "Puzzles 22 affected tasks" as task count) — post a current navigation comment pointing at this revision, with the corrected facts (PyMethods-small is one of the fourteen TRAIN releases; Puzzles 22 is an issue-event count); (3) #50 is index-only — no research-branch PR exists for `codex/48-atlas-campaign`; state that explicitly; (4) commit-metadata remediation for the public commit exposing a personal workstation email/hostname (f02de6077-era objects, including referenced old pins) remains a repository-owner decision; no address is reproduced in this record; (5) the unnecessary internal session fragment / personal-path identifiers retained in earlier published entries are flagged for owner-side cleanup — they are not repeated in this entry and cannot be removed from published entries here.
- Interpretation and limitations: the fidelity census remains uncompiled (no aggregate (a)/(b)/(c) tally; no fixed denominator); the missed evidence/publication gaps recorded in revision 8 stand; nothing above changes admission or hold states.
- Decision and next action: fresh cold-read round 10 next; the issue #48 navigation update, index refresh, and revision-19 hosted-verification check remain the owned follow-ups.
