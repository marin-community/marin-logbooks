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
> Read the newest entries first: the 13:55 UTC catch-up entry below is superseded **in part** by the later revision entries (14:25/14:35 UTC) at the end of this file, which append glossary, policy, corrections, and evidence classes. Corrections are appended, never rewritten.

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
