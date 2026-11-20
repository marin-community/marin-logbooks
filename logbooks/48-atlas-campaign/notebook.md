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
> Read the newest entries first: the 13:55 UTC catch-up entry below is superseded **in part** by the later revision entries at the end of this file — the **Known limits** section (bottom) lists what cannot be fixed by policy or was never measured. The **current synopsis** is revision 10 (18:45 UTC) as corrected in place by revisions 11–13 (19:00–20:10 UTC) and the re-review fixes (20:30–20:55 UTC); the **live gate/status** is the latest hourly entry plus the issue body (rewritten 20:10 UTC). Revision 4's consolidated facts/gate matrix/evidence index and its open-work table are the base documents, superseded where revision 5's gate-matrix corrections, revision 9's pass-3c state override (fixed/finalized/waiver post-cut), revisions 11–13 and the hourly entries correct them; the evidence/ ledgers are corrected in place (see each file's "Updated" markers). Historical synopses: revisions 3, 4, 6–7 remain as the correction trail. Corrections are appended, never rewritten.

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

### 2026-11-15 18:30 UTC - Hourly catch-up (17:30–18:30 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `b35a72d333…` (17:11:26 UTC) to head `b5b4267ba2dd4e2f94fa1d831d773c10084c55a3`, landed 17:49:45 UTC (atlas-push landings log [R]); no new inbox items beyond the round-10 cold-read batch.
- Command or material configuration: evidence: landings.jsonl tail; `git log b35a72d3..b5b4267b` (fetched).
- External inputs and artifact references: landing records 17:40:46 UTC `87e0f8c49f…`, 17:49:45 UTC `b5b4267ba2…`.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **License-lift wave continues** (`b5b4267ba2…` "Record dataset-level license lifts and confirmed ASDiv restriction"; `188cdd52db…` "Enforce recorded license bans across Atlas admission"): dataset-level license decisions are now recorded and enforced at admission — ASDiv carries a confirmed restriction (ban recorded).
  - `87e0f8c49f…` (17:40:46 UTC): synthetic catalog fixtures bound to their license evidence.
  - `3a30d5c641…` "Bound representative controls and preserve full audit cases" and `8224ac66d0…` "Check curriculum invariants before a representative browser pass" — control-bounding and curriculum-invariant checks precede representative browser passes.
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only; measured effects not yet attested in this record.
- Decision and next action: next hourly check watches for the license-lift ledger content in the registry and any ASDiv exclusion surfacing in counts.

### 2026-11-15 18:45 UTC - Revision 10 after cold-read round 10: code-oracle FULL snapshots reconciled with the public finalized binding, Stack FULL reconciled, finding-count semantics aligned

- Question or hypothesis and reason for trying it: cold-read round 10 (two reports: the campaign read and the PR-#50 read) returned NOTES. The concrete defect corrected here: the ledgers' code-oracle FULL example (1,026/15,087, provisional GOOD) is an **undated frozen-assembly stage**, while the public pinned FULL binding reports the later finalized figures (1,033/15,151, GOOD, accepted_incomplete) — readers could not reconcile the two, and the same files carried contradictory statements about whether FULL numerators are distinct-task counts. Revision 9's entries are preserved unedited; this entry and the evidence-file updates are the corrections.
- Published source revision: the recorded cut remains `96fc0736fc95d8f21d11e1c03dfca108d568b220` (13:51 UTC); branch head continues to move past it (post-cut landings are flagged per fact in the evidence headers, not merged in).
- Command or material configuration: n/a (documentation repairs; no cluster commands executed). Evidence file edited in place with an "Updated 2026-11-15 18:45 UTC" marker: evidence/sample-panels.md.
- External inputs and artifact references: [R] `/experiments/active/atlas-full-qc/REPORT.md` lines 182, 2214, 2216 (frozen-assembly snapshot and repair lane); [R] `/experiments/active/atlas-full-qc/code-oracle-content-final-v1.json` (finalization as-of 2026-11-14 12:34:19 UTC) and `code-oracle-content-repair-audit-v1.json` (repair transition: 64 re-reviewed / 5 invalid; `primary_valid_records_byte_preserved: 15087`); [P] public `experiments/post_training/task_curation/datasets/full_content_results.json` at the recorded cut (code-oracle: reviewed 15,151 / issues 1,033 / unavailable 5 / uncertain 0 / 15,156 total / issue_rate 0.06818031813081644 / GOOD / `accepted_incomplete`, input pin `f2b8131f…`; Stack: reviewed 110,168 / issues 21,412 / unavailable 394 / uncertain 38 / 110,600 total / 19.4358% / SOME_ISSUES / `accepted_incomplete`, pin `41a27ac6…`; Math Oracle: reviewed 57,253 / issues 5,020 / 8.7681% / GOOD / `accepted_incomplete`, pin `43d45617…`). Cold-read round-10 reports local-only [R].
- Environment and hardware when relevant: unchanged.
- Result, including failure or restart:

**Fixes applied in evidence/sample-panels.md (round 10)**

1. **Code-oracle FULL snapshots reconciled (supersedes the §7 undated example).** The ledger's example is now presented as the **historical frozen-assembly stage** (pre-repair; [R] REPORT.md:182/2214): population 15,156 retained = 15,087 valid + 69 INVALID; 0 unavailable/0 uncertain; 1,026 affected conclusive rows = 6.80% → provisional GOOD. The **later finalized FULL binding is public** [P: `full_content_results.json` at the recorded cut]: 15,151 reviewed / 1,033 issues / 5 unavailable / 0 uncertain / 15,156 total; issue rate 0.06818031813081644 (1,033/15,151 = 6.8180%); verdict GOOD; `accepted_incomplete` (5 unreviewed = 0.033% under the standing ≤1% waiver); input pin `f2b8131f…`. Both snapshots are dated; the finalized sidecar's SHAs match the public binding.
2. **Delta explained (7 rows / 64 tasks)**: the CPU repair overlay re-reviewed the 69 INVALID rows — 64 became valid reviews, 5 remained invalid (token-limit/schema-failure reasons itemized in the sidecar); the original 15,087 valid observations were preserved byte-for-byte [R: repair audit `primary_valid_records_byte_preserved: 15087`, `repair_status_counts: reviewed 64 / invalid 5`]. Hence 15,087 → 15,151 reviewed (+64) and 1,026 → 1,033 affected rows (+7 issues among the newly reviewed 64; per-flag deltas 706→710, 217→218, 193→195 = +7). The 1,026/15,087 figure predates the repair pass; the 1,033/15,151 figure is the later public finalized result — neither is wrong; they are different stages of the same retained 15,156-row population.
3. **Later Stack FULL reconciled**: the public transport's processed-Stack FULL result (110,168/21,412/110,600, 394 unavailable, 38 uncertain, SOME_ISSUES, `accepted_incomplete`) is bound at the retained-census pin `41a27ac6…` — a **historical** result distinct from the admitted processed-Stack release pin `5c6818d7…` (admission commit `8ea14c762f`, 13-findings SAMPLE panel) recorded in admission-ledger.md §1. "Stack remains in FULL QC" refers to review at the current pin; historical FULL aggregates must not be quoted as current-pin results without their pins.
4. **Finding-count semantics aligned (supersedes the contradictory clauses flagged in rounds 9–10)**: the §3 Math Oracle row and §7 no longer say "distinct-task counting is not established" for FULL aggregates — per the round-9 Finding-count-semantics note (now cross-referenced), FULL-lane numerators/denominators are **row-level (task-row) counts** (each retained row is one task), and the public transport now carries the Math Oracle aggregate (5,020/57,253, 8.7681%, pin `43d45617…`). The residual uncertainty is about privately held per-row observations, not about what the aggregates count.
- Evidence: [R] receipts/REPORT as cited; [P] the public `full_content_results.json` transport at the recorded cut; evidence/sample-panels.md carries the 18:45 UTC update marker.
- **Read order**: start with THIS entry (current corrections) → revision 9's fixes (Puzzles durations, pass-3c current state, finding-count semantics) → revision 8's fixes (evidence-file state as of 17:05) → revision 4's "CURRENT FACTS" consolidated block and gate matrix → revision 3's synopsis → the six evidence/ ledgers as the canonical operational detail (headers state their own evidence cutoffs) → earlier entries as history. Items 1–4 above supersede the corresponding statements in revisions 3, 4, 8 and 9.
- Secretary GitHub updates (owned surfaces this branch cannot edit — for the index/issue lane): (1) index PR #50's `index/48-atlas-campaign.md` still calls its pinned link the "revision-7 commit containing all corrections" although the URL pins revision 9 `fd0da058…` — relabel to the actual pin and refresh after this revision lands; its "revision-18 deployment" assertion should carry the evidence qualification (hosted deployment/verification receipts remain preview-lane [M]); add the one-line [P]/[M]/[R] legend; (2) issue #48's body and latest navigation comment still point to revision 5 and carry pre-revision-2 facts — post a current navigation comment pointing at this revision with the corrected facts (PyMethods-small is one of the fourteen TRAIN releases; Puzzles 22 is an issue-event count); (3) #50 is index-only — no research-branch PR exists for `codex/48-atlas-campaign`; state that explicitly; (4) the FULL-QC open-work line "Stack whole-source review" should distinguish the historical public Stack FULL result (pin `41a27ac6…`, SOME_ISSUES 19.4358%, `accepted_incomplete`) from the current-pin review still owed (admitted release pin `5c6818d7…`), and may cite the public `full_content_results.json` transport as primary public aggregate evidence; (5) commit-metadata remediation for the public commits exposing a personal workstation email/hostname (f02de6077-era objects, including referenced old pins) remains a repository-owner decision; no address is reproduced in this record.
- Interpretation and limitations: the fidelity census remains uncompiled (no aggregate (a)/(b)/(c) tally; no fixed denominator); the evidence/publication gaps recorded in revision 8 stand; nothing above changes admission or hold states; the repair-transition membership claim rests on the local repair-audit receipt [R] — its byte-level row mapping is not published.
- Decision and next action: next hourly check resumes; the issue #48 navigation update, index refresh, and revision-19 hosted-verification check remain the owned follow-ups.

### 2026-11-15 19:00 UTC - Revision 11 after cold-read round 11: issue_count semantics corrected from public code

- Question or hypothesis and reason for trying it: cold-read round 11 showed the record's "issue events / distinct-task counting not established" framing contradicts the public runtime code. This entry corrects the semantics; evidence/sample-panels.md is updated in place.
- Published source revision: recorded cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`; branch has since advanced to `b5b4267ba2…` (17:49:45 UTC landing) with the license-lift wave — recorded separately in the hourly entries.
- Command or material configuration: verification: `git show atlas-env-audit:experiments/post_training/task_curation/full_qc_review.py` lines 78–103 at the recorded cut — `quality_result()` increments `issue_count` **once per reviewed task whose verdict has issues** (`issue_count += 1` inside the per-review loop; `issues` entries are keyed by `review.task_id`), and unavailable/uncertain reviews are excluded from `reviewed` before the issues check. Exactly-once panel membership is enforced downstream in receipt generation (round-11 reader's verification; consistent with the full_content_receipts.py waiver binding).
- External inputs and artifact references: permalink https://github.com/marin-community/marin/blob/96fc0736fc95d8f21d11e1c03dfca108d568b220/experiments/post_training/task_curation/full_qc_review.py#L78-L103 [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: correction — **every `issue_count` figure in this record is a distinct-affected-reviewed-task count** (one count per affected task, exactly-once panel membership). Consequences: the Puzzles SAMPLE result is 22 affected tasks / 97 conclusive (22.68% = affected/conclusive), directly comparable with the GOOD/SOME_ISSUES/BAD thresholds; the Math Oracle FULL row's 5,020/57,253 is distinct affected rows/conclusive rows; the earlier "22 issue events, distinct affected tasks not established" and "whether findings are issue events or tasks" qualifications are **withdrawn**. The units caveat that remains is different: SAMPLE panels' per-release counts are public commit-message reports whose underlying receipts stay registry-bound [M] — the semantics are now known, the inspectability limitation unchanged.
- Evidence: code permalink above; sample-panels.md updated (Updated 2026-11-15 ~19:00 UTC).
- Interpretation and limitations: reviewer calibration/blinded agreement remains unmeasured; the fidelity census and per-release inspectable receipts remain the main open evidence items.
- Decision and next action: fresh cold read round 12 next.

### 2026-11-15 19:40 UTC - Hourly catch-up (18:30–19:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `b5b4267ba2…` (17:49:45 UTC) to head `549a05d9512e56a4f39d09906d1e88c637d8c654`, landed 19:30:09 UTC (atlas-push landings log [R]).
- Command or material configuration: evidence: landings.jsonl tail; `git log b5b4267b..549a05d9` (fetched).
- External inputs and artifact references: landing records 18:48:31 UTC `38baf5b41c…`, 18:52:35 UTC `e1c4351074…`, 19:19:50 UTC `38c85626de…`, 19:30:09 UTC `549a05d951…`.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **New source intake pipelines prepared** (`3d40ef88e5…` pending verified-SWE and terminal-pivot intake pipelines; `0965e10512…` intake validation dependencies + regional storage setup; `7e61a5f523…` terminal-pivot grader environment for artifact builds).
  - **BrowseComp-Plus offline search tasks prepared** (`acfed05dd5…`), with credential-free offline retrieval initialization enforced (`38baf5b41c…`, landed 18:48:31 UTC).
  - **IFBench converted** (`0c250a6102…` released IFBench training split with its original scorer) but held pending intake measurements (`6fe3b4595d…` "Keep IFBench pending until intake measurements are bound").
  - **Cascade terminal graders prepared** and unsupported runtimes accounted (`38c85626de…`, landed 19:19:50 UTC).
  - **Actor-env repair SAMPLE receipts registered** (`549a05d951…`, 19:30:09 UTC — the head; actorless-lane SAMPLE receipts now bound).
  - Infra: `[iris] Repair CPU topology drift and validate backend regions` (`e1c4351074…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings and receipt registration only; the new sources (verified SWE, terminal-pivot, BrowseComp-Plus, IFBench, Cascade) have not yet produced measured admissions in this record.
- Decision and next action: next hourly check watches for IFBench/terminal-pivot intake measurements and the first licensed admissions of the new sources.

### 2026-11-15 19:55 UTC - Revision 12 after cold-read round 13: affected-task semantics propagated to admission-ledger

- Question or hypothesis and reason for trying it: cold-read rounds 11–13 repeatedly flagged that the canonical admission ledger still used "issue events" semantics. Corrected in evidence/admission-ledger.md (Updated 2026-11-15 ~19:55 UTC): its Finding-semantics note now derives the counting rule from the public runtime code — `issue_count` increments once per reviewed task whose verdict has issues (permalink `full_qc_review.py#L78–L103` at the recorded cut), so all panel counts in that ledger are distinct affected reviewed tasks. Labels remain reported pending per-task row publication.
- Published source revision: recorded cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`; branch head has advanced to `549a05d951…` (19:30:09 UTC landing; hourly entry above).
- Command or material configuration: as above.
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: the notebook's revision-10 "whether affected rows are distinct tasks or issue events remains unestablished" wording is superseded by this entry and by revision 11; the read-first note and index point here.
- Evidence: permalink above; ledger update markers.
- Interpretation and limitations: reviewer calibration unmeasured; fidelity census and per-release inspectable receipts remain the main open evidence items.
- Decision and next action: fresh cold read round 14 next.

### 2026-11-15 20:10 UTC - Revision 13 after cold-read round 14: sample-panels semantics note fully corrected

- Question or hypothesis and reason for trying it: cold-read round 14 found the round-11 correction had not reached sample-panels.md's own Finding-count-semantics header note (which still asserted issue-event semantics). Corrected in place.
- Published source revision: unchanged cut; branch head `549a05d951…` per the 19:40 hourly entry.
- Command or material configuration: as revision 11.
- External inputs and artifact references: `full_qc_review.py#L78–L103` permalink at the recorded cut.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: sample-panels.md's Finding-count-semantics note (and the §-level rows flagged by round 14) now state uniformly: every count in that file is a distinct-affected-reviewed-task count (SAMPLE and FULL alike); per-row flag tallies are separate diagnostics, not issue_count. All "issue-event" assertions are withdrawn with the note marked "round-14 correction".
- Evidence: sample-panels.md (Updated 2026-11-15 ~20:10 UTC).
- Interpretation and limitations: unchanged.
- Decision and next action: fresh cold read round 15 next.

## Known limits

This section lists what cannot be fixed in this public record by policy or by absence of measurement. Cold readers should not re-raise these as notes.

1. **Private data and per-source contents**: task text, prompts, model outputs, and dataset rows are never published (user data sign-off required); the per-task finding rows behind every panel count live in the private registry `open-athena/marin-rl-data-atlas-registry` [M]; counts are commit-message reports or local-ledger transcriptions.
2. **Unpublishable artifacts**: the per-release admission receipts, SAMPLE/FULL panel receipts, ruling documents, frozen queue digests, the preview app (private deployment; hosted-verification receipts [M]), and per-run private configs/inputs. Pointers/hashes are published; the artifacts themselves are not.
3. **Reproduction impossible by policy or inputs**: full end-to-end reproduction requires private registry inputs and per-run consumed configs not retained publicly; the failure reproducer for pass-3c is a local unpublished script (excerpt + SHA published); production publication (v76) and PR merges are user decisions, not reproducible actions.
4. **Measurements never taken**: the aggregate (a)/(b)/(c) fidelity census tally over the final population (denominator not yet fixed); independent reviewer calibration/blinded agreement; GPU/CPU-hour accounting for most runs (only curriculum CPU-hours and some serving runs recorded); hosted preview revision-19 verification; landed-time (vs commit-time) records for most admissions.

### 2026-11-15 21:20 UTC - Hourly catch-up (19:40–21:20 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `549a05d951…` (19:30:09 UTC) to head `036cceab31e06844720488fd29e504d479777f05`, landed 21:12:36 UTC — a large batch (~24 commits, landings 21:04:51 / 21:09:57 / 21:12:36 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 549a05d9..036cceab` (fetched).
- External inputs and artifact references: landing records as above; PR #50 (index/48) merged with index head 546eb090f7 per the coordinator.
- Environment and hardware when relevant: iris change admits 08a gVisor CPU work through a hostname-only TAS flavor (`7c86252b6c…`).
- Result, including failure or restart: material changes in the window:
  - **License-gate wave continues and hardens**: dataset-level license checks made deterministic (`1dea2e839f…`); license grants recognized in README headings (`cd8c1efcb5…`); checked intake receipts bound and source evidence compacted (`9d9527fd5e…`); recorded wrappers normalized before license checks (`59cd4ad75a…`); existing user license rulings bound to exact Atlas source pins (`b6c3d2443d…`); checked source licenses preserved in private dataset publications (`3f3faa031c…`); publication preflight applied to declared source use (`c79ceecbdd…`); license evidence kept OUTSIDE the retained publication schema (`3103384d66…`); Atlas docs now point at the stage-0 license gate and the DF inventory recipe (`d4640207df…`).
  - **Intake census and measurements**: complete intake census registered and lexical measurements registered (`c5477d3a98…`).
  - **New source capture lanes**: original Multi-SWE grading archives captured from pinned images (`0fcca0ddcd…`); regional Iris staging required for full private-image capture (`a4bf6703b0…`); validated RST packages declared for native Harbor intake (`4092cd6362…`); capture placement verified from controller backend records and sealed by the current attempt (`c422e5b7db…`, `0dec582a8b…`).
  - **OpenResearcher indexing lane (new)**: regional asset receipts with stable import/content identities (`d74b6ebcf1…`); pinned assets staged to regional storage (`9938f7db3a…`); OpenResearcher index built and verified in regional Iris jobs (`42a8472079…`); unique research questions extracted and indexing runtime isolated (`ebebe0ea77…`); research questions screened against complete corpus evidence (`234b49bec3…`); regional roots checked with storage path containment (`036cceab31…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings and receipt registrations; the measured effects (admissions, license-lift outcomes, indexer quality) are not yet attested in this record.
- Decision and next action: next hourly check watches for the first license-gated admission outcomes and OpenResearcher index results. Note: this entry is appended after the branch's SATISFIED read (b319468a3aa) — routine catch-ups continue on the branch; the index pin (PR #50, merged at 546eb090f7) remains the reviewed snapshot.

### 2026-11-15 22:35 UTC - Hourly catch-up (21:20–22:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `036cceab31…` (21:12:36 UTC) to head `8dc5766a2f1c23f883cf650c4f45b00425f56176`, landed 22:28:06 UTC (landings 21:51:31 / 22:07:10 / 22:28:06 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 036cceab..8dc5766a` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: research-judge children kept on their regional controller (`ade3515972…`).
- Result, including failure or restart: material changes in the window:
  - **OpenResearcher lane continues**: regional search images prepared with verified index layers (`0db5846e5f…`).
  - **Retained-release lineage hardened**: exact raw source archives preserved in retained release lineage (`25516b242a…`); measured pending sources bound to retained release receipts (`4ba3fb99db…`); image capture artifact identities kept stable across CLI entry points (`8dc5766a2f…`, head).
  - Incoming from main: `[verifyit] Stop Atlas retries after repeated math deadlines (#11518)` (`210239de55…`) — retry policy change for Atlas math-deadline loops.
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only; measured effects not yet attested in this record.
- Decision and next action: next hourly check watches for retained-release license/publication outcomes and any OpenResearcher search results.

### 2026-11-15 23:45 UTC - Hourly catch-up (22:35–23:45 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `8dc5766a2f…` (22:28:06 UTC) to head `738638f20bed9bb0b6d706c01944833aa3d42ea3`, landed 23:32:09 UTC (landings 23:17:24 / 23:30:48 / 23:32:09 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 8dc5766a..738638f2` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: research grounding judge fitted on one H100 (`3de4e095d0…`); measured grounding host resources pinned before staging (`738638f20b…`, head).
- Result, including failure or restart: material changes in the window:
  - **License lane**: user-decided TaskTrove container licenses bound (`a4bf955d1f…`); approved research licenses bound to the retained population (`f469c6848d…`); research task conversion bound to complete grounding receipts (`0767c2fa7b…`).
  - **Grounding/indexing lane continues**: HTTP transport recorded for the pinned grounding tokenizer (`ae964efff1…`); research grounding judge fitted on one H100; measured grounding host resources pinned before staging (head).
  - **Archive hygiene**: truncated raw archive names kept valid on Hugging Face (`26a18f992a…`); complete raw repository acquisition clarified (`20d8363f69…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only; measured effects not yet attested in this record.
- Decision and next action: next hourly check watches for the grounding-judge run results and the first license-bound retained-population publications.

### 2026-11-16 00:35 UTC - Hourly catch-up (23:45–00:35 UTC window, now 2026-11-16)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `738638f20b…` (23:32:09 UTC Nov 15) to head `47c99fc4229a76ee61a8c91c8f97119e478fb721`, landed 00:23:08 UTC Nov 16 (landings 00:00:52 / 00:18:11 / 00:23:08 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 738638f2..47c99fc4` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **First license lift**: the `Evalchemy:ifbench` license ban was lifted on its pinned ODC-BY grant (`943750a4bd…`) — the first recorded ban→grant transition under the new license-gate regime.
  - **IFBench and Cascade intake measurements recorded** (`6a5cd0bb5d…`); IFBench train checked against the full held-out benchmark (`112fa38131…`); original IFBench exclusions verified before sealing reports (`49695efb7a…`).
  - **Grading-boundary hardening**: shared-source grading boundaries fail closed (`47c99fc422…`, head); SHARED source-scorer graders protected for the Prime pytest families (`75e67c8dc4…`).
  - **Terminal-Pivot admission decision documented** with grader limits (`c6d3ec51b9…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings and measurement records; the admission outcomes flowing from the IFBench measurements and license lift are not yet attested in this record.
- Decision and next action: next hourly check watches for the IFBench/Cascade admission decisions and any further license lifts.

### 2026-11-16 01:35 UTC - Hourly catch-up (00:35–01:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `47c99fc422…` (00:23:08 UTC) to head `d1fd107b7784f76591cce6fa943d4aaec5297f17`, landed 00:55:28 UTC.
- Command or material configuration: evidence: landings.jsonl tail; `git log 47c99fc4..d1fd107b` (fetched).
- External inputs and artifact references: landing record as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: one landing in the window — `[Atlas] Bind static retention evidence to selected private releases` (`d1fd107b77…`): the static-retention path (the no-oracle sources' retention route per the 2026-11-14 ruling) now has its evidence bound to the selected private releases. No long-loop changes (STATE.md unchanged).
- Evidence: commit above; landing timestamp in the local atlas-push log [R].
- Interpretation and limitations: single code landing; measured effects not yet attested.
- Decision and next action: next hourly check watches for further license lifts/admissions and any retained-publication outcomes.

### 2026-11-16 02:35 UTC - Hourly catch-up (01:35–02:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `d1fd107b77…` (00:55:28 UTC) to head `91816039f97035ec1e2606c3f2fbff063608bcac`, landed 02:09:40 UTC (landings 01:42:24 / 02:05:18 / 02:09:40 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log d1fd107b..91816039` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **IFBench decontamination recorded** against the complete test split (`91816039f9…`, head) — follows the license lift and intake measurements from the previous window.
  - **Grounding retries within sealed budget**: truncated grounding responses retried within the sealed budget (`dc288fd79e…`).
  - **SAMPLE retry documentation**: fixed-panel SAMPLE retries documented with a fresh response cache (`1d231c30c1…`); SAMPLE retry evidence and missing-review accounting clarified (`d6b5f17f3b…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings and measurement records; the decontamination outcome and admission decisions are not yet attested in this record.
- Decision and next action: next hourly check watches for the IFBench admission decision following the decontamination record.

### 2026-11-16 03:35 UTC - Hourly catch-up (02:35–03:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `91816039f9…` (02:09:40 UTC) to head `01242464815072dbe017828693d9d66e29d703a9`, landed 03:31:01 UTC (landings 03:06:15 / 03:08:43 / 03:31:01 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 91816039..01242464` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window — **first admissions under the license-gate regime**:
  - **Terminal-Pivot ADMITTED** with its original grader and BAD content quality (`19a5f21787…`) — the first admission carrying an explicit BAD content Result (consistent with QC-never-rejects); its measured source-overlap results retained (`6c8e8901e2…`).
  - **Retained Cascade multi-domain and MOPD tasks admitted** (`0124246481…`, head).
  - Failed grounding HTTP responses preserved for diagnosis (`d3ca57ba4e…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: admission commits report the outcomes; the underlying receipts stay registry-private [M]. These are the first post-license-gate admissions observed in this record.
- Decision and next action: next hourly check watches for further license-lift admissions (IFBench following its decontamination record) and Cascade/OpenResearcher results.

### 2026-11-16 04:35 UTC - Hourly catch-up (03:35–04:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `0124246481…` (03:31:01 UTC) to head `f87a9f8fde921c544eaeb8ead6aca2d137ba5328`, landed 04:28:24 UTC (landings 04:02:54 / 04:11:42 / 04:28:24 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 01242464..f87a9f8f` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **License-receipt sweep completes**: swept checker-PASS receipts bound for **Available** catalog sources (`161f80bed4…`) and for **Not selectable / Excluded** rows (`fcd2c8a021…`) — the license sweep now covers every catalog disposition; the canonical license receipt registry is sharded (`9a167fd13c…`); licensed unavailable sources are preserved in catalog publication (`f87a9f8fde…`, head).
  - **Adapter QA driver**: protected shared-source adapter QA driver added (`6678ea4cc9…`).
  - **Grounding budget**: grounding prompt tokens counted before enforcing the context budget (`cac61710ad…`).
  - **Capture identity**: prepared-image identity restored across skopeo OCI config rewrites (`23455512ac…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the license sweep's outcome (which rows are Available/Not-selectable/Excluded) is bound in the registry [M], not enumerated in this record.
- Decision and next action: next hourly check watches for the post-sweep catalog publication and any further admissions.

### 2026-11-16 05:35 UTC - Hourly catch-up (04:35–05:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `f87a9f8fde…` (04:28:24 UTC) to head `0fc733f087a0202183077fea2e5a1eda718806fe`, landed 05:25:23 UTC (landings 04:34:31 / 04:40:25 / 05:25:23 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log f87a9f8f..0fc733f0` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Retained IFBench training population ADMITTED** (`0fc733f087…`, head) — the second license-lift admission (following the `Evalchemy:ifbench` ban lift and the decontamination record from the 02:35 window).
  - **Grounding/validation hardening**: grounding requirement budgets checked before nested validation (`5b85372021…`); prepared-image identity compared on runtime-relevant config only (`dd0b08ec1c…`); protected-adapter QA evidence named by run (`6da0915223…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: admission commit reports the outcome; the underlying receipts stay registry-private [M]. IFBench is now the second source admitted through the license-gate regime (after Terminal-Pivot and the Cascade/MOPD batch).
- Decision and next action: next hourly check watches for further license-lift admissions and post-sweep catalog publication.

### 2026-11-16 06:35 UTC - Hourly catch-up (05:35–06:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `0fc733f087…` (05:25:23 UTC) to head `9ca238b157ab45cb83cd095132d89e9757adabaf`, landed 06:23:19 UTC (landings 06:02:44 / 06:16:24 / 06:23:19 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 0fc733f0..9ca238b1` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: shared-source controller kept loadable on 3.9 task images (`6b240fe9d5…`) with the interpreter resolved to 3.9-or-newer (`590ec31ab8…`) — a compatibility constraint for task-image runners.
- Result, including failure or restart: material changes in the window:
  - **OpenResearcher exclusions bounded**: incomplete OpenResearcher grounding exclusions bound/recorded (`9ca238b157…`, head) — the indexer lane's grounding coverage limits are now explicit.
  - **Hidden verifier delivery**: root-actor hidden verifiers delivered at grading time after quiesce (`6156a2ab12…` / `fb51ae2b7c…` duplicate landing).
  - **Shared-source controller compatibility**: kept loadable on 3.9 task images; interpreter resolution to 3.9-or-newer.
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the exclusion ledger contents stay registry-private [M].
- Decision and next action: next hourly check watches for OpenResearcher index measurement results and further admissions.

### 2026-11-16 07:35 UTC - Hourly catch-up (06:35–07:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `9ca238b157…` (06:23:19 UTC) to head `c1b5d37d18b5494a46ca689c16b3d3ac54837856`, landed 07:18:57 UTC (landings 06:39:24 / 07:18:57 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 9ca238b1..c1b5d37d` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: two landings in the window — protected SWE converters bound to complete source image maps (`c1b5d37d18…`, head: the faithful-environment pairing now covers the SWE converters' image maps); shared controllers kept loadable on 3.9 task interpreters (`dd0cce7306…`, extending the compatibility fix from the previous window to all shared controllers). No long-loop changes (STATE.md unchanged).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for OpenResearcher results and further license-gate admissions.

### 2026-11-16 08:35 UTC - Hourly catch-up (07:35–08:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `c1b5d37d18…` (07:18:57 UTC) to head `ed848d11fec0e8ef9d3f9cb6defe00fe256a0d21`, landed 08:24:58 UTC (landings 08:10:43 / 08:24:58 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log c1b5d37d..ed848d11` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: two landings in the window — native declared setup now recorded in **physical NoP controls** (`c7054cf453…`: the negative controls capture the physical setup, strengthening the faithful-environment pairing for the NoP lane); large OpenResearcher retrieval responses preserved (`ed848d11fe…`, head: the indexer lane's retrieval outputs retained, with the known privacy boundary — retrieval responses are metadata/index material, not task data). No long-loop changes (STATE.md unchanged).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only; the retained responses' access class follows the standing publication rule (no task data).
- Decision and next action: next hourly check watches for OpenResearcher index results and further retained-release publications.

### 2026-11-16 09:35 UTC - Hourly catch-up (08:35–09:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `ed848d11fe…` (08:24:58 UTC) to head `d5d7571ba28efda6f11981294140d388cdedaaa6`, landed 09:24:26 UTC (landings 08:44:34 / 09:08:36 / 09:24:26 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log ed848d11..d5d7571b` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: explicit declarative Daytona provisioning ported (`de34dfccea…` — the canonical Harbor/Daytona declarative provisioning used by exp2's eval, now in the Atlas lane).
- Result, including failure or restart: material changes in the window:
  - **Root-delivery contract typing**: the root-delivery SHARED contract typed for original-root sources (`ce0eea3f1b…`) and bound to ScriptGrader consumers with unbounded deadlines (`d5d7571ba2…`, head — the grading-time hidden-verifier delivery lane from the 06:35 window gets its contract formalized).
  - **Protected-adapter QA**: the source testbed Python recorded in the protected-adapter QA (`1d1fe38188…`).
  - **Daytona provisioning**: explicit declarative Daytona provisioning ported (`de34dfccea…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for root-delivery consumer adoption in admissions and further retained-release publications.

### 2026-11-16 10:35 UTC - Hourly catch-up (09:35–10:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `d5d7571ba2…` (09:24:26 UTC) to head `fada02f02b0ed4c12da050da751b4723bbcb2153`, landed 10:25:48 UTC (landings 09:59:23 / 10:10:13 / 10:25:48 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log d5d7571b..fada02f0` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Root-delivery contracts adopted in the exporter**: TaskSpec Harbor exporter now accepts root-delivery contracts (`41a8cf743b…`) — completing the 09:35 window's contract formalization.
  - **Protected SWE population measurements bound** (`2801347c3a…`).
  - **DECLARATIVE DockerBuildExecution exporter opt-in added** (`fbb6400fdfe6…`) — extends the declarative-provisioning port.
  - **OpenCode prompts repaired from pinned raw questions** (`fada02f02b…`, head; by a different landing session `01a1dfb7`) — OpenCode-prompt repair work is part of this campaign's prompt-repair lane.
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for root-delivery consumer adoption in new admissions and further license lifts.

### 2026-11-16 11:35 UTC - Hourly catch-up (10:35–11:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `fada02f02b…` (10:25:48 UTC) to head `f7c0bbc30162586cd43626b87c66a86577206895`, landed 11:33:35 UTC (~10 landings 11:09–11:33 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log fada02f0..f7c0bbc3` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Two more license bans lifted (user-approved)**: smartie swesmith-admitted and calibforge-shared-v2 origin bans lifted (`f7c0bbc301…`, head).
  - **Protected SWE retention selected**: protected SWE retained private releases selected (`6f130d22fa…`) — a retention/selection step toward admission.
  - **Delivered-verifier lane matured**: native reward-file result mode added to delivered verifiers (`e86420612c…`); delivered-verifier streams redirected into a declared artifact path (`0b5f0049c4…`); RST source verifiers bound to original-root TaskSpecs (`c96c06158f…`).
  - **Declarative Daytona adoption**: original TaskSpec builds run through declarative Daytona (`a5a64a697a…`); process sessions preserved through TaskSpec machine wrappers (`d8ffd6dc4d…`).
  - **Pin moves**: user-approved card-only TaskTrove pin moves adopted (`ecb748e990…`).
  - **Docs/availability**: pinned OpenCode question repair phases documented (`067f9e221b…`); unavailable metadata preserved without computed results (`36d98e2374…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the lifted bans' admission outcomes are not yet attested in this record.
- Decision and next action: next hourly check watches for smartie/calibforge admissions following the ban lifts.

### 2026-11-16 12:40 UTC - Hourly catch-up (11:35–12:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `f7c0bbc301…` (11:33:35 UTC) to head `f484da62b76c518dc3726deb0864c91b707f1fc1`, landed 12:30:25 UTC (landings 12:18:20 / 12:25:45 / 12:30:25 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log f7c0bbc3..f484da62` (fetched).
- External inputs and artifact references: landing records as above. Cross-campaign: exp3 production launched (logged in the #49 notebook this hour).
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **IF/Cascade re-pinning**: re-pinned IF and Cascade sources fingerprinted and stale retained pins refreshed (`f484da62b7…`, head).
  - **Assessment/Result gates**: Atlas empty assessment pages checked without masking missing evidence (`b8ce30bf3a…`); the computed-Result gate kept for campaign-reviewed unavailable sources (`7126925e9d…`).
  - **Delivery contracts adopted**: Multi-SWE home delivery and the typed delivery contract added (`e226cb6c8f…`); declarative Docker contexts exposed in TaskSpec trial preparation (`ae0e65907d…`).
  - **OpenResearcher**: complete documents transferred (`912e04c03c…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the re-pinned IF/Cascade admission outcomes are not yet attested in this record.
- Decision and next action: next hourly check watches for IF/Cascade re-pinned admissions and exp3 progression.

### 2026-11-16 13:40 UTC - Hourly catch-up (12:40–13:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `f484da62b7…` (12:30:25 UTC) to head `062882844315cdb0c9d1491bd81a771d8bd7d522`, landed 13:25:54 UTC (landings 13:04:38 / 13:07:11 — by the coordinator session itself — / 13:25:54 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log f484da62..06288284` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Preview revision 20 synced**: curation docs and Getting Started synced to preview revision 20 (`84bba26de4…`), with worker-provider and cluster-credential detail kept out of curation docs (`900ef88733…`) — revision 20 is now in the docs; hosted verification remains preview-lane evidence [M].
  - **Publication hardening**: prior catalog snapshot frozen before testing publication (`7d41c56f60…`); unavailable publication verified to preserve archived assessments (`f26743f112…`); frozen OpenResearcher intake registered as pending (`7451834ea9…`); campaign quality regression kept after the shared gate fix (`0628828443…`, head).
  - **Multi-SWE delivery**: images bound to faithful delivery and host report parsing (`dbaba96d34…`); home restore reproduces source member validation and destination removal (`dd5c7ba361…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the rev-20 hosted deployment is not attested from this record.
- Decision and next action: next hourly check watches for rev-20 hosted verification and frozen-intake progression.

### 2026-11-16 14:40 UTC - Hourly catch-up (13:40–14:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `0628828443…` (13:25:54 UTC) to head `455379a66821a9e0f0a6ae11e20c6a18fc5fb6b7`, landed 14:23:11 UTC (~21 landings 14:20–14:23 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 06288284..455379a6` (fetched).
- External inputs and artifact references: landing records as above; new repair-candidate lane dirs /experiments/active/atlas-repair-candidate-lane/ and new-rl-data-ingest/.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **New repair-candidate lane built**: match and comparative review built (`48dff0e55e…`), lint addressed (`f924bcd5de…`), tracked parquet inputs streamed to local files before decode (`44c352ec8e…`), match diagnostics with similarity samples and previews reported (`87706ba9c7…`), drop advisories suppressed for candidates with no tracked overlap (`79a351f350…`), tracked task archives lowered through the same preview conversion (`17e8821a1a…`), prepared content and archive pins supported in the tracked loader (`9c40911e70…`); lane stage documented in the task-curation reference (`c212765e9c…`); lane STATUS updated with dry-run findings (`4f674d012a…`).
  - **Sharded license receipt override documented** (`455379a668…`, head — following the 04:35 registry sharding).
  - **Status taxonomy / Atlas UI**: source status taxonomy with typed exclusion claims (`df04188b86…`), source-status page + pending queue rows + metadata exclusions (`b46c19a822…`), admission procedure documented (`cdc8c002b2…`), typed QC findings and manual audit served in the Atlas (`09d6d89d42…`), unreachable source judges treated like unspecified ones for judge-tier choice (`94c6dc2429…`), Excluded sources hidden from the default view (`3f3349ae5a…`), status-page journey/filter fixes.
  - **User rulings recorded**: needs-ruling and OpenGCUA rows (`3bd135eb94…`); RL source status rules for experiment agents documented (`30871a72b7…`); zero-drop retention plans published through the standard CLI (`d6b4a80b92…`); candidate shell checks and interrupted dedup recovery documented (`cf0dd53eaf…`); original shell interface enabled in converted workspace tasks (`92905eb368…`); pending intake queue refreshed (`eed759613c…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the repair-candidate lane's dry-run findings are recorded in its STATUS, not yet attested here in detail.
- Decision and next action: next hourly check watches for repair-candidate lane results and further user-ruling adoptions.

### 2026-11-16 15:50 UTC - Hourly catch-up (14:40–15:50 UTC window; resumed after a model-switch gap)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief. The secretary's session was switched to a new model after the previous provider hit its monthly limit; this entry covers 14:40→15:48 UTC.
- Published source revision: branch `atlas-env-audit` advanced from `455379a668…` (14:23:11 UTC) to head `e3826a043ee7afd5cdd7aaebf810764ec11f3ce4`, landed 15:42:30 UTC (two landings).
- Command or material configuration: evidence: landings.jsonl tail; `git log 455379a6..e3826a04` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: two landings in the window — protected source graders exposed in versioned review evidence (`c8bb502552…`); historical review pins and original verifier access clarified (`e3826a043e…`, head — ties the review-evidence pins to original verifier access, the discipline the cold reads requested).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for repair-candidate lane results and license-gate admissions.

### 2026-11-16 16:40 UTC - Hourly catch-up (15:50–16:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `e3826a043e…` (15:42:30 UTC) to head `2215bf34c06b43ad5c1f260d32d45aaa50dc597c`, landed 16:22:31 UTC (~7 landings 16:17–16:22 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log e3826a04..2215bf34` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **License-filtered source admission decision typed**: the typed user-decision receipt bound for the license-filtered Skill2Env source (`2215bf34c0…`, head) — a user decision recorded as a receipt, the pattern the cold reads asked for.
  - **First QUICK-oversight admission**: QUICK oversight admitted, lexical remediation queued, unsupported backends excluded (`64c560adcd…`); difficulty-gap exclusion reasons added with the shared mapping (`0335753c16…`); normalization approved and running per the repair-candidate lane STATUS (`81ea7a3d85…`).
  - **Repair-candidate matching hardened**: stripped tracked index rebuilt from pinned URIs (`a6b6127e9e…`); shared leading setup sections stripped before repair matching — an approved policy (`216e508097…`); the strongest pair's stripped text sampled in match diagnostics (`1148009186…`).
  - **Root tests importable; Harbor preflight timeout guard lifted** (`050cfe96f2…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the Skill2Env user-decision and QUICK-admission outcomes are committed reports, registry-private underneath [M].
- Decision and next action: next hourly check watches for the lexical remediation outcome and further typed user-decision receipts.

### 2026-11-16 17:40 UTC - Hourly catch-up (16:40–17:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `2215bf34c0…` (16:22:31 UTC) to head `d2f5b0ccbb1e6dfe9cdb6731d85050f9d559b121`, landed 17:29:35 UTC (~12 landings 17:22–17:29 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 2215bf34..d2f5b0cc` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **More backend-gate decisions**: absent-role sources exempt from the backend gate; the four unbuildable cohorts excluded (`761122b0f5…`); empty backend maps kept blocked by the admission gate (`275fba1d1d…`); full admission required only for Available rows on catalog load (`89c17d9b85…`); judge tiers served inside the federated campaign root (`d2f5b0ccbb1…`, head).
  - **OpenResearcher pre-admission fingerprint bound** (`743f03880f…`); audit count projections appended after the retained cohort view columns (`af5cfc8ca4…`); Daytona difficulty sources prepared through the declarative-capable backends (`354490699c…`).
  - **Status extras rebuilt on the rebased head** (`bc16010003…`, `e056cd914a…`); served BackendSurvey identity cited in the math_tir claims (`75fe18cf9c…`); status page composed from unique source IDs (`46754aef91…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the four excluded unbuildable cohorts' identities are registry-bound [M].
- Decision and next action: next hourly check watches for smartie/calibforge admissions and the status-page rollout.

### 2026-11-16 18:40 UTC - Hourly catch-up (17:40–18:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `d2f5b0ccbb1…` (17:29:35 UTC) to head `baa792383d421ae22fa9707d2d01545f2655d58e`, landed 18:33:14 UTC (~22 landings 18:30–18:33 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log d2f5b0cc..baa79238` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Lexical remediation batches 1–3 and swesmith executed with receipts** (`662251fc80…`, `32be1e0c05…` incl. the ARC-AGI transductive tag-tuple fix, `95febca725…`) and their fingerprints refreshed (`13d9f9aead…`, `eeedad377a…`, `aecfd25a82…` pointing at the corrected/merged registry revisions `e2af6731`/`023b6c47`); status extras refreshed after the adoptions (`fa24a72528…`).
  - **Ten verified FULL content Result bindings recorded** (`35b317af85…`); archived original-pin content receipts validated through zero-drop retained lineage (`01128dd427…`); mutual-best made the standing repair-lane matching rule (`305d0fbc74…`).
  - **Catalog/infrastructure**: license CLI records parsed from stdout (`baa792383d…`, head); machine-less TaskSpec grading counted against declared network policy (`6e33840dd9…`); status-table columns bounded, reason-column nesting stopped (`4187a8d37e…`); status page readable in both themes with a permanent sweep (`e63cc6fb2c…`); root catalog fixtures given measured N/A backend rows (`ea4a2510fb…`); Harbor Daytona test provider aligned with blocking cleanup (`c2caaaf7aa…`, `4f72be9b70…`); large committed data guarded behind file/directory limits with a shrink-only allowlist (`799177e655…`); packaged admission module freed from taskcompendium with label-based backend matching (`afea5c7c03…`, `98b0d05741…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the remediation receipts' underlying panels stay registry-private [M].
- Decision and next action: next hourly check watches for post-remediation admissions and the FULL-binding outcomes.

### 2026-11-16 19:40 UTC - Hourly catch-up (18:40–19:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `baa792383d…` (18:33:14 UTC) to head `d9056675328448388417a18f1457439ad6f64fd3`, landed 19:19:24 UTC (~9 landings 19:05–19:19 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log baa79238..d9056675` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **License receipt registry moved to a referenced private Hugging Face dataset** (`b89222bb67…`), with the referenced registry documented (`d905667532…`, head); lowercase package-grant identifiers for HF card validation (`2f9005de05…`).
  - **User-decided retained-pin licenses bound for the MiMo and CalibForge remediation releases** (`fb298c9336…`).
  - **Docs synced to preview revision 21** (`5c7e57a9ff…` — rev 21 follows rev 20's sync from the 13:40 window); status-table Reasons column clamped (`cab7a906c7…`); import probe pointed at the repository root (`271b1e7515…`).
  - **Zero-drop retained lineage**: archived original-pin content evidence bound through it (`37c764979c…`, duplicate `0241beec87…` — two landing records for the same change).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the HF-hosted license registry is private [M].
- Decision and next action: next hourly check watches for rev-21 hosted verification and post-remediation admissions.

### 2026-11-16 20:40 UTC - Hourly catch-up (19:40–20:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `d905667532…` (19:19:24 UTC) to head `ad2d01588a4661c328fabf993d7e35aa33732142`, landed 20:30:42 UTC (~13 landings 20:26–20:30 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log d9056675..ad2d0158` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Re-pinned Numina retention completed**: census, dedup, lexical and retained receipts bound in the registry (`d4baae8153…`); retained-release license receipt bound (`d7b3b618fc…`); complete retained receipt collection restored with the re-pinned Numina receipt (`4b52e74098…`); fingerprint pinned onto the landed index (`a6cc725a12…`, `907af187a4…`).
  - **License remediation batches 6–9 + reasoning gym + adopted-hold receipts recorded** (`504fb461d1…`) and bound onto the restored registry (`230cbfd182…`); staged receipts file accepted by the license reader (`476ab4323d…`).
  - **Consumed NoP identity preserved for scored oracle failures** (`204d3d6af1…`); repair lane's serving placement wait extended to 24 hours (`068d23bb06…`, `ad2d01588a…` head).
  - **Source-duplicate acknowledgements moved to the referenced private HF dataset** (`285bc25b66…` — following the license registry's move last window).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the Numina retained-population admission outcome is not yet attested in this record.
- Decision and next action: next hourly check watches for the Numina admission and further remediation-batch adoptions.

### 2026-11-16 22:40 UTC - Hourly catch-up (21:40–22:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `63f9bc5eef…` (20:45:34 UTC) to head `4155e3cb032136b25d55b4e9a3db819f31860e12`, landed 22:20:54 UTC (~21 landings 22:19–22:20 UTC incl. **two main merges**).
- Command or material configuration: evidence: landings.jsonl tail; `git log 63f9bc5e..4155e3cb` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Main merges into atlas-env-audit** (`203aa08a5f…`, `4155e3cb03…` head): "restore the pull_request merge ref and CI" — brings #10661 (faster RL startup), #11554 (importable root tests / preflight timeout), #11551 (Iris supervisor termination handling), #11550 (taskforge image unpacking / registry manifest readback), #11549 (Iris test logging/E2E timeouts) from main.
  - **Grading-channel clarifications**: conclusive controls kept for ungraded oracle outcomes (`74f355acfa…`); original vs NoP grading-channel comparison clarified (`2ae2828790…`); proved upstream verifier exceptions recorded as source exclusions (`f7cefc33d6…`).
  - **Repair-lane splice**: scripted-filter turn edits spliced into the original rendered text (`27a4eee30b…` — the V6 serializer correction for exp3's data arm); surviving turn order kept across dropped turns (`903b9bca13…`); concurrent Bloom cache fills isolated (`1985d3b598…`); numbered receipt shards only for next-part picking (`835e186860…`); raw-open verified FULL content Result binding referenced (`0b3a9fe477…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the CI restoration effects are not yet attested in this record.
- Decision and next action: next hourly check watches for CI state on the merged head and further repair-lane/V6 progress.

### 2026-11-16 23:40 UTC - Hourly catch-up (22:40–23:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `4155e3cb03…` (22:20:54 UTC) to head `db12047ab89cbb207edefaaf8e00a6385be8e9f3`, landed 23:26:40 UTC (~10 landings 23:06–23:26 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 4155e3cb..db12047a` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **math_gym-qa re-pin chain completed**: math-v5 deprecation in favor of the repaired math_gym-qa pin (`33326cb6b1…`), repointed retained-population fingerprint pinned (`8ef84b5c67…`), measured containment in its math-v5 origin acknowledged (`c91ba5a977…`), stale whole-source Result binding superseded on the current derived head (`febcf7fcad…`).
  - **POLARIS intake progressed**: complete intake measurements recorded (`81ca07ba59…`); status moved from Intake-queued to Measurements-pending (`db12047ab8…`, head).
  - **Infrastructure NoP re-drives retained** when diagnosed (`b87db43ae3…`); fresh SAMPLE receipts bound for the remediation versions (`137e57df8d…`).
  - **Infra/docs**: data allowlist compared against origin/main and fetched in lint CI (`99eba82474…`); judge serving probe stops advertising tools (`3fe9559d3c…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the math_gym-qa retained-population admission outcome is not yet attested in this record.
- Decision and next action: next hourly check watches for POLARIS measurements and the math_gym-qa admission.

### 2026-11-17 00:40 UTC - Hourly catch-up (23:40–00:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `db12047ab8…` (23:26:40 UTC) to head `abf170f40e4d`, landed ~00:20 UTC (~8 landings 00:11–00:19 UTC incl. a main merge).
- Command or material configuration: evidence: landings.jsonl tail; `git log db12047a..abf170f4` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Main merge into atlas-env-audit**: takes the Minerva scorer revision and taskforge test packaging for the PR merge ref (`abf170f40e…`, head).
  - **Shared-adapter fixes**: bootstrap failures observed in the full SWE run fixed (`b290895221…`); Harbor lock refreshed for tokenizer test dependencies (`d82647917b…`); judge-probe tools drop completed in the tokenizer check (`b2c280ff4d…`); rolloutengine test group given its tool-free test dependencies (`6e4af090c2…`); tool-free completion events narrowed so the role check type-checks (`82c6f3d8e8…`).
  - **Preservation fixes**: PDBThink training tasks and their native verifier preserved (`301a3128e6…`); approved NoP grades preserved after teardown failures (`8bd001230a…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for shared-adapter re-run outcomes and POLARIS progression.

### 2026-11-17 01:40 UTC - Hourly catch-up (00:40–01:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `abf170f40e…` (~00:20 UTC) to head `86775b6632fe62aebdeab70635d188cc869a6185`, landed 01:34:31 UTC (~10 landings 01:15–01:34 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log abf170f4..86775b66` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **License decisions continue**: FWeindel dataset-level ban replaced with per-folder decisions under the repackager ruling (`86775b6632…`, head); FWeindel/validated-tasks banned on the missing dataset-level license (`cf80b02e71…`).
  - **POLARIS pre-admission fingerprint bound** (`f9e15a4369…`).
  - **Registry publish guard**: publishes that would blank rows held by unstaged sibling shards refused (`bf2ba49620…`).
  - **NoP retries**: approved NoP retries allowed after verifier archive failures (`5dfd0022f3…`).
  - **Evalchemy identity carry-forward**: fingerprints rebound to the carried-forward identity (`219a3615be…`); receipts re-measured and carried across dedup, census and sample panels to the post-Minerva identity (`346c35df85…`).
  - **Difficulty coverage computed in the Atlas rebuild as code** (`2612179fe5…`); tool-free model upath import sourced from universal-pathlib (`37004a63fe…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the FWeindel per-folder decisions' details are registry-bound [M].
- Decision and next action: next hourly check watches for POLARIS progression and further license decisions.

### 2026-11-17 01:45 UTC - Correction to the 01:40 hourly entry (FWeindel license decisions)

- Question or hypothesis and reason for trying it: coordinator correction relayed from the user's repackager ruling — the 01:40 entry's phrase "FWeindel/validated-tasks banned on the missing dataset-level license" is wrong.
- Published source revision: unchanged `86775b6632…`.
- Command or material configuration: n/a.
- External inputs and artifact references: coordinator correction (rhizome message, 2026-11-17); commit `86775b6632…`.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **Corrected reading** — per the repackager ruling, the FWeindel dataset-level ban was **removed** by `86775b6632…`, not applied. The per-folder decisions: **27 FWeindel folders PASS on their original upstreams and are lifted**; **6 folders (BugsInPy, TermiGen) carry per-folder bans**; **4 folders (Endless v1–v3, TMax v1.1) await the user**. The 01:40 hourly entry's banned-on-missing-license phrasing is superseded by this correction.
- Evidence: commit `86775b6632…` [P]; coordinator correction [R].
- Interpretation and limitations: the per-folder license detail lives in the registry [M].
- Decision and next action: next hourly entries use the corrected reading.

### 2026-11-17 02:40 UTC - Hourly catch-up (01:40–02:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `86775b6632…` (01:34:31 UTC) to head `05ccef12575b1a936d28616f9505928401705650`, landed 02:32:57 UTC (~10 landings 02:22–02:32 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 86775b66..05ccef12` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **POLARIS ADMITTED**: its 53,167 retained training tasks admitted (`90ebae40a5…`) — following the pre-admission fingerprint binding from the previous window.
  - **PDBThink checked against its complete native held-out splits** (`05ccef1257…`, head).
  - **Frozen-packet QA driver hardened** after the v1 run abort (`6258459da4…`).
  - **Retention/dedup hardening**: conversion shards checked before deduplication repartitions them (`70d0f1ea9c…`); memory bounded during task retention and deduplication (`171005b6ae…`).
  - **Infra/docs**: campaign continuation's serving digest documented (`363951727f…`); campaign queue fetched through the Atlas query channel (`5adf6f573b…`); evaluation-corpus publication delegated to the shared publisher with advisory cleanups (`886de3e565…`, `2311738682…`); tool-free completion request payload typed explicitly (`a67780fac5…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: admission commit reports the outcome; receipts registry-private [M].
- Decision and next action: next hourly check watches for PDBThink outcomes and further license decisions.

### 2026-11-17 03:40 UTC - Hourly catch-up (02:40–03:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `05ccef1257…` (02:32:57 UTC) to head `6089da60a9a1144e34d00f06f58eb3eab9dd59f4`, landed 03:31:03 UTC (~10 landings 03:23–03:31 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 05ccef12..6089da60` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: guest controller runtime kept compatible with Python 3.9 (`df3c389ed0…`, `1581b5cab5…`).
- Result, including failure or restart: material changes in the window:
  - **PDBThink chain progressed**: train population measurements recorded (`032c16a1e0…`); repair serialization kept aligned with its census (`54bd638fcd…`); native verifier limits preserved in private repairs (`af6694751e…`).
  - **puzzles-qa prepared locator rebuilt** from the retained-release pinned inputs (`6089da60a9…`, head); frozen-packet QA driver rebound to the v2026.11.16.3 freeze (`4e3b71287f…`).
  - **Journey oracle census bound** at the exact declaration identity (`1f09d31d8a…`); review summaries decoupled from execution artifact imports (`3142fa462a…`); directory symlinks kept outside the large-data payload classifier (`e3c5071fc8…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; PDBThink measurement contents are registry-bound [M].
- Decision and next action: next hourly check watches for PDBThink admission progression and puzzles-qa locator effects.

### 2026-11-17 09:40 UTC - Hourly catch-up (03:40–09:40 UTC window; resumed after a session gap)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief. The session was idle from ~04:40 UTC; this entry covers the gap through the 09:34 readback.
- Published source revision: branch `atlas-env-audit` advanced from `6089da60a9…` (03:31:03 UTC) to head `f1fc42c6a4ab58c77f522dfed7f2fb5b7c5bba2a`, landed 08:56:31 UTC (~12 landings 07:28–08:56 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 6089da60..f1fc42c6` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Cross-check budget gating**: cross-check resolved role allocations keyed against the map's calibrated budgets (`54f118c8ed…`); deadlines read from the executor-resolved shape (`cb91c30288…`); informational baseline-spec field kept out of cross-check gating (`83a5eda6f8…`).
  - **Codeforces restoration**: configured Codeforces dedup binding and raw measurements restored (`e4fe5a842e…`, `d0d3180052…`); regenerated SWE v3 measurements bound (`cd57e6f5e1…`).
  - **Repaired-source publication**: repaired sources published from staged lexical receipts (`a21ce68a6827…`).
  - **RST/Multi QA + capture**: archive-verbatim packet consumption for the RST/Multi QA (`4028397450…`); QA artifact downloads keyed by row identity, not URI basename (`5526cb5d39…`); RST images quiesced without requiring guest Python (`f1fc42c6a4…`, head); Dockerfile heredocs preserved during Daytona preflight with spaced-delimiter recognition (`258abe317e…`, `2434954905…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for repaired-source admission outcomes and RST capture progression.

### 2026-11-17 10:40 UTC - Hourly catch-up (09:40–10:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `f1fc42c6a4…` (08:56:31 UTC) to head `2ff07684cf2c9a68f10ebc3b6981c3ac8e13ea8e`, landed 10:33:16 UTC (~8 landings 10:26–10:33 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log f1fc42c6..2ff07684` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: harmony-served judge probes rendered through the openai_harmony encoder (`c82170ac1a…`).
- Result, including failure or restart: material changes in the window:
  - **Repaired Codeforces release ADMITTED** (`4754e71408…`) — following the 09:40 window's dedup/raw restoration.
  - **Qualified passing subsets admitted with pending excluded tasks** (`0fe8259846…`); passing subsets published alongside pending infrastructure outcomes (`5ac8f6ad03…`) — the infrastructure-failure carve-out from the standing retry policy is now visible in admissions.
  - **CalibForge lineage ruled**: the calibforge rebuilt-vs-shared overlap acknowledged as ruled lineage (`2ff07684cf…`, head); CF-adoption advisory cleanups applied across the shared lanes (`2f583394d2…`); reviewed overlap pairs from run 44012930875 acknowledged (`26354678cc…`).
  - **Runtime**: identity-affecting #11547 review fixes synced onto the shared runtime (`2997d803ca…`); harmony encoder for judge probes.
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: admission commits report outcomes; receipts registry-private [M].
- Decision and next action: next hourly check watches for further passing-subset admissions and judge-probe qualification.

### 2026-11-17 11:40 UTC - Hourly catch-up (10:40–11:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `2ff07684cf…` (10:33:16 UTC) to head `abe5f511d461a526a2a9f20b2e77dda4887ee3bf`, landed 11:28:32 UTC (~9 landings 11:19–11:28 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 2ff07684..abe5f511` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Standing policies pinned with tests**: duplicate outcomes, artifact retention, faithful environments, and region movement documented as standing curation policies (`8a137bf9ac…`); the FULL finalization waiver bound of 1% unreviewed now enforced (`9eb2f6fb90…`); approved archived sources bound to frozen retained records (`0841333ec4…`).
  - **Multi-SWE verification visible**: captured Multi-SWE verifiers exposed in review payload 6 (`b929737cd9…`); the revision-6 SAMPLE reviewer hash registered (`459d253da6…`); a VERIFY support level added for demands the offline survey cannot observe (`dec7e2c26c…`); declared file metadata restored in the guest after single-file uploads (`a3a26fd179…`).
  - **Backend/infra**: docker memory limits enforced swapless so declared caps cannot be exceeded through swap (`abe5f511d4…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for policy-pinned admission outcomes.

### 2026-11-17 12:40 UTC - Hourly catch-up (11:40–12:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `abe5f511d4…` (11:28:32 UTC) to head `2b7ee67d967623ba78b3f7e6aee083e186d22dca`, landed 12:33:26 UTC (~10 landings 12:26–12:33 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log abe5f511..2b7ee67d` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Atlas preview staged with prebuilt assets and no build command** (`3867eed37c…`) — the preview deployment path changes shape; deployment class carried through survey merges (`2b7ee67d96…`, head).
  - **FACET artifact lane**: each pinned FACET image preserved as a sealed artifact (`a63eb0f3bd…`); image receipts loadable after CLI execution (`6b8c4d6afe…`); FACET artifact model formatted (`9ad96332fe…`).
  - **production_checkers module import guards added with a rename test** (`4ef1967b12…` and two duplicate landing records).
  - **Rounds/watchdog layer**: advisory findings resolved on the rounds layer and watchdog (`309e521db8…`); a warning added when a rounds launch ships a worktree behind atlas-env-audit (`0cfd8595b8…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the prebuilt-asset preview deployment outcome.

### 2026-11-17 13:40 UTC - Hourly catch-up (12:40–13:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `2b7ee67d96…` (12:33:26 UTC) to head `908b57e410c54161043c42e9a00283e1af9c706f`, landed 13:30:21 UTC (~10 landings 13:11–13:30 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 2b7ee67d..908b57e4` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **New VERIFY support level operationalized**: new VERIFY admissions materialized as Pending with zero-admitted sources excluded (`77dcb3543a…` + duplicate `77dcb3543a`-pair); VERIFY admissions gated — admitted sources keep status with a pending marker (`361761fbf1…`); the fixture waiver bound to its real denominator (`4361c02419…`); one production VERIFY cell is enough — N/A cannot negate pending qualification (`3e3b482ac7…`); the backend verification pending marker rendered on the backends page (`c8452e6585…`).
  - **Re-admission**: the competitive-coding rebuilt cohort readmitted (`48421f3366…`); re-pinned retained sources bound to their retained grading evidence (`908b57e410…`, head).
  - **TaskSpec hygiene**: absent execution requirements kept out of serialized TaskSpecs (`d9e801196d…`); full-review fixture bump matched to the endpoint's text ordering (`dfd6a80231…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the pending-marker semantics follow the VERIFY level defined in the previous window.
- Decision and next action: next hourly check watches for VERIFY-pending progression to full admission.

### 2026-11-17 14:40 UTC - Hourly catch-up (13:40–14:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `908b57e410…` (13:30:21 UTC) to head `7c1ebdb0823b5412a82e32aae2422fe617208791`, landed 14:30:17 UTC (~13 landings 14:11–14:30 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 908b57e4..7c1ebdb0` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Difficulty measurement instrumentation**: bulk difficulty and declaration reads paginated under the transport bound (`7ae58fe23a…` + duplicate landing); measurement objects stored with bounded readbacks and a checkbox accent (`ceca95f3ca…`); difficulty measurement inserts/readbacks batched (`b523cef69a…`); loader difficulty reads projected to the server's summary shape (`7c1ebdb082…`, head); journey geometry derived from the served inventory with the fixture backend renamed (`e13262f168…`).
  - **VERIFY/admission semantics refined**: only Available sources counted as admitted, and the VERIFY downgrade gated on runnable support (`0a3493cea2…`); grader network-profile conflicts shipped as typed Pending (`9112004bb6…`); a pending source's Examples cell kept unmeasured (`f625e3af62…`).
  - **FULL cohort summaries honor the review policy** (`7a7f90b1fa…` + duplicate landing); TaskSpec identity now distinguishes absent execution requirements from explicit nulls (`baf6327d53…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the difficulty-instrumentation rollout and further VERIFY semantics.

### 2026-11-17 15:40 UTC - Hourly catch-up (14:40–15:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `7c1ebdb082…` (14:30:17 UTC) to head `50d2fdd6b6df0ae2322be4d77ee99664d5cab122`, landed 15:30:25 UTC (~12 landings 15:23–15:30 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 7c1ebdb0..50d2fdd6` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Lexical clearance now gates new Available admissions** (`7e53f42d8e…`) — implementing the standing lexical-clearance policy; armed for new Available admissions (`dd86422b4e…`); aime eval-benchmark overlaps acknowledged with the batched readback mock updated (`2e8736cded…`); lexical gate test imports hoisted (`41b5d01fbe…`).
  - **RST v3 private release ADMITTED** (`f873cccf6b…`) — the archived RST v3 private release admitted.
  - **License-gate fixture qualified as an admitted source** (`aa3f0e8a82…`).
  - **TaskSpec serializer hardening**: caller include/exclude arguments honored (`191cef19b8…`); str-typed late-joining fields serialized without assuming models (`0f569bbde3…`).
  - **Infra**: oversized campaign evidence archived atomically (`45d90022df…`); /sources cohort summaries fetched per pipeline under the gateway bound (`a5069372cc…`) with the item grouping key indexed (`4c4a4ae71e…`); internal GHCR visibility allowed for FACET images (`50d2fdd6b6…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: admission commits report outcomes; receipts registry-private [M].
- Decision and next action: next hourly check watches for lexical-clearance-gated admissions and RST v3 downstream outcomes.

### 2026-11-17 16:40 UTC - Hourly catch-up (15:40–16:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `50d2fdd6b6…` (15:30:25 UTC) to head `81a120a9130c4056caa5b70546d536a24a1591d6`, landed 16:26:11 UTC (~10 landings 16:14–16:26 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 50d2fdd6..81a120a9` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **SWE-rebench family added**: added to the protected SHARED source adapter (`d288466f87…`) with the SWE-rebench flow proved and two machinery interplays fixed (`772444d259…`); the contract workspace created in the shared-base image layer (`81a120a913…`, head).
  - **Guide/refs synced with current curation rulings** (`9cc97d3fad…`).
  - **Daytona sandbox identities**: owned sandbox IDs recorded through finalization (`8223f5070c…`); Harbor provider fakes given sandbox identities (`51d3654bd2…`).
  - **Recovery/calibration machinery**: recovery fixture execution kept inside the attempt loop (`0040c907ac…`) with the current trial CLI (`532086c7af…`); calibration configs read from their named CLI option (`1971727cbc…`).
  - **Atlas UI**: loaded review summaries reused and navigation shared (`9f35e5ac2e…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for SWE-rebench family progression and further delivered-verifier outcomes.

### 2026-11-17 17:40 UTC - Hourly catch-up (16:40–17:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `81a120a913…` (16:26:11 UTC) to head `7de61d0b06a5e792ade2a8182579e98844cb48c9`, landed 17:29:45 UTC (~10 landings 17:00–17:29 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 81a120a9..7de61d0b` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **AIME decontamination chain completed and ADMITTED**: historical AIME routed through training decontamination (`7995b24e45…`); the decontaminated 918-task AIME training release adopted (`4b6c745449…`); AIME's training decontamination receipt selected in the Atlas (`24c08194cc…`); the AIME training fingerprint view selected in the Atlas (`7de61d0b06…`, head).
  - **Delivery contracts honored**: the backend survey honors matched delivery contracts (`7e066abb4f…`).
  - **e2egit-v2 corrections**: census pair restored and stale math_tir rulings withdrawn (`64a739cb79…`); census/lexical references pointed at the re-measured receipts (`a2345c9f18…`).
  - **Identical evaluation-ID groups reused during causal attribution** (`d21bf96d84…`).
  - **Status UI keyboard traversal**: scroll containers included (`35126d475d…`); keyboard targets kept visible on narrow screens (`af414c144a…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: admission commits report outcomes; receipts registry-private [M].
- Decision and next action: next hourly check watches for further decontamination-chain admissions.

### 2026-11-17 18:45 UTC - Hourly catch-up (17:40–18:45 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `7de61d0b06…` (17:29:45 UTC) to head `c80683567df3d1d250ac238d00c0d3b6b1a5dfb8`, landed 18:32:50 UTC (~12 landings 18:23–18:32 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 7de61d0b..c8068356` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **aime acknowledgement superseded**: the aime acknowledgements moved to the TRAIN decontamination receipt (`f9963f6215…`) — aligning the acknowledgement with the decontamination chain from the 17:40 window.
  - **Partial backend semantics**: Partial backend support preserved with capability evidence (`30f8b3446d…`); Partial eligibility and role counts clarified (`1ac354a1bb…`).
  - **Rounds/driver machinery**: coverage-driven rounds launched only for the frozen sample selection (`c80683567d…`, head); operator-stopped rounds carried through the rounds driver (`79ae17cbbe…`); script-grader submissions staged before setup commands run (`1ecadd3fdb…`).
  - **Lexical/data hygiene**: missing document-frequency filters rejected before lexical marking (`f66ca128ca…`); corpus-common filter storage keys preserved on load (`228eafb169…`); native runner inputs restored in curation tests (`46493940fc…`); only the declaration fields the coverage classifier reads are fetched (`ad863bc70e…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for coverage-driven rounds results and further aime supersessions.

### 2026-11-17 19:45 UTC - Hourly catch-up (18:45–19:45 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `c80683567d…` (18:32:50 UTC) to head `b2661b15516eecd476148357310958125e0ed421`, landed 19:24:25 UTC (5 landings 18:53–19:24 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log c8068356..b2661b15` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **New-admission rule tightened**: the SAMPLE panel is now required on the **current pin** for new admissions (`b2661b1551…`, head) — closing the earlier gap where a panel on a prior pin could support a new admission.
  - **FACET filesystem binding**: FACET tasks bound to the preserved SIF image filesystems (`9e3f1963e5…`).
  - **Fingerprint summaries projected** before transferring source rows (`36f5b9a351…`).
  - **Docs**: the PARTIAL backend state and matched delivery contracts documented (`acf864bbab…`).
  - **Rounds double-submit guard** now ignores terminal prior attempts (`1fd4352377…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for current-pin SAMPLE enforcement effects on new admissions.

### 2026-11-17 20:40 UTC - Hourly catch-up (19:45–20:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `b2661b1551…` (19:24:25 UTC) to head `69841e38a40015bfc88e6c009eedd7b81477aac4`, landed 20:00:58 UTC (1 landing).
- Command or material configuration: evidence: landings.jsonl tail; `git log b2661b15..69841e38` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: one landing in the window — Atlas docs updated for the new license-receipt registry reference (`69841e38a4…`, following the 19:40 window's registry move to the referenced private HF dataset).
- Evidence: commit above; landing timestamp in the local atlas-push log [R].
- Interpretation and limitations: docs landing only.
- Decision and next action: next hourly check watches for license-receipt documentation follow-through and further admissions.

### 2026-11-17 21:40 UTC - Hourly catch-up (20:40–21:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `69841e38a4…` (20:00:58 UTC) to head `d49ca9e53a35a3022c306cca9e4306cbd5676510`, landed 20:52:54 UTC (3 landings 20:41–20:52 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 69841e38..d49ca9e5` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window: task packet buffering bounded during oracle calibration (`034d7e83b1…`); source-table responses trimmed to rendered evidence (`66040ec809…`); the source display projection contract stated (`d49ca9e53a…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for continued license-receipt/prelight follow-through.

### 2026-11-17 22:40 UTC - Hourly catch-up (21:40–22:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `d49ca9e53a…` (20:52:54 UTC) to head `bb0fa0e90215bb9082569dbf50ab97ccc2f84e62`, landed 21:35:08 UTC (1 landing).
- Command or material configuration: evidence: landings.jsonl tail; `git log d49ca9e5..bb0fa0e9` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: one landing in the window — source inventories reused **only while their inputs are unchanged** (`bb0fa0e902…`, head: guarding inventory reuse against stale inputs).
- Evidence: commit above; landing timestamp in the local atlas-push log [R].
- Interpretation and limitations: single code landing.
- Decision and next action: next hourly check watches for continued license/prelight follow-through.

### 2026-11-17 23:40 UTC - Hourly catch-up (22:40–23:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `bb0fa0e902…` (21:35:08 UTC) to head `9cbe20a66a189888feca7d63e91bc1497008a38e`, landed 23:30:35 UTC (~10 landings 22:51–23:30 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log bb0fa0e9..9cbe20a6` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **FACET census and lexical measurements recorded** (`9cbe20a66a…`, head).
  - **Restriction-tier hardening**: restriction tier carried on staged plans and raw snapshots (`ba5c6ecf7a…`); erased restriction tags fail closed at publish, with overrides only strengthening the tier (`124ec82ace…`); pre-tier archived release plans tolerated while the missing tag is refused at publish (`311232abbf…`).
  - **Private-declaration projection tightened**: content-bearing evidence fields dropped from the projection (`57d517c9dc…`); free-text QC method dropped (`3c7cf35615…`); the reviewer rubric redacted from private assessment reports (`2028e0e38e…`) with the redaction documented in the private data tier docs (`3d0fc6946e…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the FACET census/lexical measurement contents are registry-bound [M].
- Decision and next action: next hourly check watches for FACET admission progression and the private-tier publication outcomes.

### 2026-11-18 00:35 UTC - Hourly catch-up (23:40–00:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` — no new landings since `9cbe20a66a…` (23:30:35 UTC Nov 17) at this readback.
- Command or material configuration: evidence: landings.jsonl tail; `git ls-remote` (no new head).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new Atlas landings in the window; the branch state is unchanged from the 23:40 entry. Cross-campaign note: the #11619 NCCL transport probes completed on all three clusters (default env passes over InfiniBand/NVLink everywhere; no cluster needs the socket override) and exp3 `.51` is being restarted from scratch on IB — recorded in the #49 notebook.
- Evidence: `git ls-remote` [R]; #11619 comment 6964135906 [P].
- Interpretation and limitations: quiet window.
- Decision and next action: next hourly check resumes normal Atlas coverage.

### 2026-11-18 01:40 UTC - Hourly catch-up (00:35–01:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `9cbe20a66a…` (23:30:35 UTC Nov 17) to head `b9c7f2101a2f16ff7386f4cf1ec18ed9ece46456`, landed 00:06:13 UTC (4 landings 23:48–00:06 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 9cbe20a6..b9c7f210` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Rebuilt-cohort remediation completed end to end**: retained releases bound (`298a6c9b00…`); the four re-measured Evalchemy evals completed end to end (`5628db75e7…`); the serving cluster recorded in the repair review report (`9296df6d1c…`).
  - **Census-receipt validation tightened**: census receipts refused when their own evidence contradicts them (`b9c7f2101a…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the re-measured eval outcomes are registry-bound [M].
- Decision and next action: next hourly check watches for rebuilt-cohort admission outcomes.

### 2026-11-18 02:40 UTC - Hourly catch-up (01:40–02:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `b9c7f2101a…` (00:06:13 UTC) to head `d90e8163f9bf31e72e21fdf685549eb8e412d40b`, landed 01:30:57 UTC (~10 landings 01:12–01:30 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log b9c7f210..d90e8163` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: an internal image-pull probe added to the backend survey's live qualification (`ab98ca94d9…`) — extending the #11619-style transport verification work into the survey machinery.
- Result, including failure or restart: material changes in the window:
  - **License chain hardening**: FWeindel folder receipts bound to their checked upstream evidence (`e0b05b19ee…`); FACET receipts bound to their checked upstream evidence (`348638b904…`); the checker-verified license stated on published cards (`b25ee06adf…`); license identifiers published in the Hub card spelling (`d90e8163f9…`, head).
  - **Census/review integrity**: census evidence audit scoped to the changed sources (`82d194e044…`); imported review rows verified with an exact keyed database join (`5726512449…`); reviewer batch storage bounded with identity-based row comparison (`7d27be7308…`); FACET lexical and grading receipt references restored (`e17ea85201…`).
  - **Status extras refreshed** from the current queue (`c4c32731be…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for license-chain follow-through and status-rollout effects.

### 2026-11-18 02:40 UTC - Hourly catch-up (01:40–02:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `d90e8163f9…` (01:30:57 UTC) to head `7198849d687c3c471cdb4a542c6b0157900a0832`, landed 02:26:40 UTC (~7 landings 02:05–02:26 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log d90e8163..7198849d` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: the iris pull probe kept off the unenforceable storage cell (`8e8dd8b9bc…` — the runtime-probe work touching this campaign's transport verification).
- Result, including failure or restart: material changes in the window:
  - **FWeindel receipts completed**: the 27 PASS per-folder receipts rebound with resolved upstream pins (`7198849d68…`, head) — completing the FWeindel license-decision chain from the repackager ruling (27 lifted / 6 banned / 4 pending user).
  - **Cohort/QC hardening**: canonical cohort summaries materialized through bounded projections (`2183ba6e48…`); deadline-bound exact review readbacks split without masking failures (`ba857afdaa…`); QC inventory coverage queries bounded with batch counts kept separate from content (`132741cbac…`); unused QA payloads excluded from cohort aggregate group keys (`76b2422ae5…`).
  - **Transport probe**: the iris pull probe kept off the unenforceable storage cell (`8e8dd8b9bc…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the re-bound receipts' detail is registry-bound [M].
- Decision and next action: next hourly check watches for FWeindel follow-through and the pending-user folders' decision.

### 2026-11-18 03:40 UTC - Hourly catch-up (02:40–03:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `7198849d68…` (02:26:40 UTC) to head `d5d7b4ee28ff5616624fdc0b305ba7280d91f772`, landed 03:20:28 UTC (~8 landings 03:15–03:20 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 7198849d..d5d7b4ee` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **FineEnvs conversion lane added**: conversion for grouped input images (`a0082d5559…`); allocation-aware FineEnvs image grouping (`d5d7b4ee28…`, head).
  - **Grading/verifier preservation**: nested native verifier files preserved during grading (`dda3158dde…`).
  - **Inventory timestamp transport fix**: inventory metadata timestamps compared by instant across transports (`9113cfeeae…`).
  - **Repair-candidate gating**: repair candidates gated on bound per-folder lineage receipts (`d4a5acbacd…`); oversized candidate SQL reads paged with the actual source projection checked (`b18ccd3621…`).
  - **math_gym-qa finding recorded on the source**: LaTeX expected-answer finding recorded (`77f2eeea3a…`).
  - **Infra**: the Iris lifecycle used when a watched job has no recent log line (`eff6fa5e75…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for FineEnvs conversion outcomes and repair-candidate gating effects.

### 2026-11-18 04:40 UTC - Hourly catch-up (03:40–04:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `d5d7b4ee28…` (03:20:28 UTC) to head `7ff75501935a4e480ade9c8a22d83c23cb6eada5`, landed 04:32:43 UTC (~8 landings 03:43–04:32 UTC incl. two coordinator-session landings).
- Command or material configuration: evidence: landings.jsonl tail; `git log d5d7b4ee..7ff75501` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Preview revisions 28 and 29**: Atlas docs synced to preview revision 28 (`5571716aab…`), then revision 29's admission display policy recorded (`7ff7550193…`, head) — the preview cadence accelerated to two revisions in one window.
  - **RST trial path**: `trial_attempt_timeout_sec` consumed in the TaskSpec Daytona path (`3e2f590278…`).
  - **Curriculum/status**: curriculum sample reads kept within the SQL gateway byte budget (`f4cd29b178…`); candidate Status responses exercised during preview preflight (`1aeb5fae81…`); absent privacy tags preserved in the lightweight Status projection (`16ae44e8c0…`).
  - **Policy note**: limit deviations need a per-source user ruling (`69d8aad755…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; rev-28/29 hosted verification remains preview-lane evidence [M].
- Decision and next action: next hourly check watches for rev-29 hosted verification and the admission-display policy rollout.

### 2026-11-18 05:40 UTC - Hourly catch-up (04:40–05:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `7ff7550193…` (04:32:43 UTC) to head `df6fe0dc3a9f084e5ea8472bfa8d02ce4132c763`, landed 05:20:16 UTC (3 landings 05:13–05:20 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 7ff75501..df6fe0dc` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **nl2bash repair-lane r4 review report pinned** (`df6fe0dc3a…`, head) — the sealed repair-lane record now includes the r4 review report.
  - **Repair-lane UX**: short pair aliases echoed and per-pair comparative decisions persisted (`a854e4fa93…`).
  - **Trial host capabilities resolved from the worker environment** (`bb10af1189…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the r4 report's contents are registry-bound [M].
- Decision and next action: next hourly check watches for the nl2bash repair-lane outcomes.

### 2026-11-18 06:40 UTC - Hourly catch-up (05:40–06:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `df6fe0dc3a…` (05:20:16 UTC) to head `f2b39492853403151fb00f0d8d45acc4d074a577`, landed 05:39:22 UTC (1 landing).
- Command or material configuration: evidence: landings.jsonl tail; `git log df6fe0dc..f2b39492` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: one landing in the window — the re-measured census, lexical and dedup receipts bound for the **13 non-oracle-mediated sources** (`f2b3949285…`, head) — extending the repair-lane's receipt chain to the non-oracle-mediated set.
- Evidence: commit above; landing timestamp in the local atlas-push log [R].
- Interpretation and limitations: single code landing; the receipt contents are registry-bound [M].
- Decision and next action: next hourly check watches for the non-oracle-mediated sources' admission progression.

### 2026-11-18 07:40 UTC - Hourly catch-up (06:40–07:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `f2b3949285…` (05:39:22 UTC) to head `ede08d135a26e078122a65d1cf7bc67ba68acf22`, landed 07:09:02 UTC (1 landing).
- Command or material configuration: evidence: landings.jsonl tail; `git log f2b39492..ede08d13` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: one landing in the window — the sealed **nl2bash repair-lane r5 review report pinned** (`ede08d135a…`, head) — the repair-lane record now carries r5 following the r4 report pinned in the 05:40 window.
- Evidence: commit above; landing timestamp in the local atlas-push log [R].
- Interpretation and limitations: single landing; the r5 report's contents are registry-bound [M].
- Decision and next action: next hourly check watches for the nl2bash repair-lane admission progression.

### 2026-11-18 08:40 UTC - Hourly catch-up (07:40–08:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `ede08d135a…` (07:09:02 UTC) to head `e7328675bdeb0eb6e6e73f88cb8fb437895e8e69`, landed 08:25:04 UTC (2 landings 07:44 / 08:25 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log ede08d13..e7328675` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **pymethods repair-lane r6 review report pinned** (`30fc5134fa…`) — the repair-lane record now carries r6, following nl2bash r5.
  - **Winner-cohort conversion stage added to the repair lane** (`e7328675bd…`, head) — the winning repair candidates' conversion path is now a stage of its own.
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the r6 report's contents are registry-bound [M].
- Decision and next action: next hourly check watches for the winner-cohort conversion outcomes.

### 2026-11-18 09:40 UTC - Hourly catch-up (08:40–09:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `e7328675bd…` (08:25:04 UTC) to head `6f92f4483426f4762ed52b93f3b9a35d65ec2f30`, landed 09:27:20 UTC (~6 landings 08:43–09:27 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log e7328675..6f92f448` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Repair-lane cohort winners lowered through the production source recipe** (`e9f953dfc7…`); tracked runtime bindings carried into repair cohorts with **native Harbor task export** (`6f92f44834…`, head) — the winner-cohort conversion stage from the previous window is now exporting runnable native tasks.
  - **Registry hygiene**: stale receipt-shard writes rejected before upload (`f6f6c267f6…`).
  - **OpenMDW 1.1 publications**: structure-closure releases recorded under OpenMDW 1.1 (`68cd5b3aa4…`); first-party publications resolved from their own card (`a5a063c1d0…`); recorded lineage receipts honored in the publication check (`ec4edb8b23…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the exported repair-cohort tasks' intake outcomes.

### 2026-11-18 10:40 UTC - Hourly catch-up (09:40–10:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `6f92f44834…` (09:27:20 UTC) to head `9a3c9e6bf398d787f84ceeb9cd8e7f011c2bdf84`, landed 10:32:49 UTC (~6 landings 10:07–10:32 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 6f92f448..9a3c9e6b` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Repair cohorts scoped by conversion identity with full TaskSpec serialization** (`9a3c9e6bf3…`, head) — completing the winner-cohort conversion lane.
  - **FWeindel wrapper normalization**: wrapper solve.sh normalized in the pymethods cohort with prepare-args fixed (`c9c0555f62…`); candidate args bound to the modes that consume them (`5c11f7056f…`).
  - **Publication integrity**: partial publication runs resumable, foreign content pins refused (`83cb09a5cf…`).
  - **Grading**: math verifier deadlines tagged with their grading phase (`6f86b4eff3…`).
  - **Docs/tests**: recovery re-read comment rewritten without the issue reference (`6226c25acf…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the serialized repair-cohort TaskSpecs' intake outcomes.

### 2026-11-18 11:40 UTC - Hourly catch-up (10:40–11:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `9a3c9e6bf3…` (10:32:49 UTC) to head `5e2b95d754f7ba8445abc34806233eb3e93e12a1`, landed 11:27:39 UTC (~5 landings 11:07–11:27 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 9a3c9e6b..5e2b95d7` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window — **repair-cohort runtime joins completed**: runtime bindings joined by tracked task ID (`3809b282c2…`), through the two-vintage tracked chain (`8175c8c99f…`) with strict zipping (`d339c72ba3…`) and a two-vintage runtime join (prepared id → instruction hash → runtime; `9539213d48…`); repair winners whose spec shards were lost mid-attempt reconverted (`5e2b95d754…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the rejoined repair cohorts' export outcomes.

### 2026-11-18 12:40 UTC - Hourly catch-up (11:40–12:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `5e2b95d754…` (11:27:39 UTC) to head `eb3e33733b02966c645bbaa49b511574e422299d`, landed 12:28:12 UTC (~8 landings 12:16–12:28 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 5e2b95d7..eb3e3373` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **HARDMath RHS remediation variant registered** (`58b5f199ba…` + duplicate landing; `f92a8fdf86…` defers the HARDMath rhs variant registration until its registry receipts land).
  - **Held-source re-publications adopted**: the 2026.11.18.1-r1 retained re-publications adopted for the 3 held sources (`e064e06434…`).
  - **Export hygiene**: empty directories preserved in the Harbor export upload (`eb3e33733b…`, head); content-identity lint fixed (`0f35ee180b…`).
  - **Oracle calibration**: cleanup grace exposed for oracle calibration (`c251c27a03…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the held-source re-publication and HARDMath variant outcomes.

### 2026-11-18 13:40 UTC - Hourly catch-up (12:40–13:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `eb3e33733b…` (12:28:12 UTC) to head `6c960dae97a70400c20126958e9289c3ccf84bae`, landed 13:26:12 UTC (1 landing).
- Command or material configuration: evidence: landings.jsonl tail; `git log eb3e3373..6c960dae` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: one landing in the window — a **single-workflow oracle campaign for the repair lane** (`6c960dae97…`, head): the repair lane's oracle filtering is consolidated into one workflow.
- Evidence: commit above; landing timestamp in the local atlas-push log [R].
- Interpretation and limitations: single code landing.
- Decision and next action: next hourly check watches for the single-workflow oracle campaign's outcomes.

### 2026-11-18 14:40 UTC - Hourly catch-up (13:40–14:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `6c960dae97…` (13:26:12 UTC) to head `887056f8ef64628259a5806b952ae227c095db27`, landed 14:33:12 UTC (~10 landings 14:16–14:33 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 6c960dae..887056f8` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Repair-lane cohort conversion completed**: the tracked production grader environment bound in cohort conversion (`2bed41a6bd…`); CoreWeave S3 credentials exported for the Harbor trial subprocess (`887056f8ef…`, head) — the winner-cohort tasks are now runnable end to end.
  - **Receipt-audit**: every Atlas receipt family audited to bind the canonical source declarations (`08e05bdc02…`); declaration audit formatted (`8e9460e179…`).
  - **Daytona/quarantine machinery**: shared bounded quarantine capacity across Daytona control roots (`d02ca9ca46…`); ambiguous cleanup quarantined between control trials (`a0fb43e037…`); single-shot sandbox deletion alignment (`efaa82513f…`); campaign workflow collects from the calibration root (`236b1348bf…`); the `--campaign-receipt` dispatch fall-through to the plain cohort branch fixed (`7fd50fb713…`).
  - **Quantine contract**: the shared quarantine reserve contract clarified (`1f4b7b016a…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the runnable repair-cohort trials' outcomes.

### 2026-11-18 15:40 UTC - Hourly catch-up (14:40–15:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `887056f8ef…` (14:33:12 UTC) to head `7db60c9c0fcd927b4b629cbda4a27e1aaaa71f96`, landed 15:12:23 UTC (~5 landings 14:45–15:12 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 887056f8..7db60c9c` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Single-workflow oracle campaign RUN on the tracked LOCAL route with collect artifacts published** (`12c8933d9d…`); the LOCAL route's delegated cgroup resolved under both worker layouts (`7db60c9c0f…`, head) — the single-workflow oracle campaign from the 13:40 window is now executing and publishing.
  - **Competitive-coding SAMPLE receipts bound** at the 2026.11.18.1-r1 retained pin (`61efb4dd79…`).
  - **HARDMath rhs release bound** to its checked upstream MIT grant (`746499a202…`).
  - **Quarantine design**: the single-slot Daytona quarantine design restored (`231ee30346…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the oracle-campaign collect artifacts' contents are registry-bound [M].
- Decision and next action: next hourly check watches for the single-workflow oracle campaign's outcomes and the competitive-coding SAMPLE decisions.

### 2026-11-18 16:55 UTC - Hourly catch-up (15:40–16:55 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `7db60c9c0f…` (15:12:23 UTC) to head `de499faa8e94b35d5a10c0662bcf303bddd3b7ce`, landed 16:48:09 UTC (~9 landings 16:35–16:48 UTC incl. one coordinator landing).
- Command or material configuration: evidence: landings.jsonl tail; `git log 7db60c9c..de499faa` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Preview revision 30**: Atlas docs synced to preview revision 30 (`42e5398817…`).
  - **Difficulty self-serve report landed**: difficulty self-serve report, continuation preflight, and frozen-contract vintages (`447800a249…`); report wording: aggregated-by-run with own-status breakdown (`de499faa8e…`, head).
  - **HARDMath rhs receipt rebound** to the retained-schema revision (`3158253ef4…`) with the retained tasks.parquet schema pinned in the publisher test (`f2218395bb…`).
  - **LOCAL worker hardening**: fail fast on unprivileged LOCAL workers with full disk tooling installed (`3455edec52…`); campaign worker prep creates its root before writing the evidence file (`deb19ccd21…`); apt configured to run as root inside capability-less local image machines (`e2535d31c7…`).
  - **Status/source evidence**: source evidence kept responsive with difficulty rollouts linked (`98cd489361…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; rev-30 hosted verification remains preview-lane evidence [M].
- Decision and next action: next hourly check watches for rev-30 hosted verification and the difficulty self-serve report's first outputs.

### 2026-11-18 17:40 UTC - Hourly catch-up (16:55–17:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `de499faa8e…` (16:48:09 UTC) to head `b21c95aa3dbb4213e05c41d3ca2c6303c40500df`, landed 17:30:01 UTC (~4 landings 17:23–17:30 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log de499faa..b21c95aa` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window: the repair cohort's prepared actor image bound at conversion (`b21c95aa3d…`, head — completing the winner-cohort lane's image chain); marina preview Postgres reclaimed after child failures (`de43ac2f24…`) with signal cleanup clarified for main-thread owners (`a2fb435524…`); apt state trees made writable for capability-less local machines (`48930bed22…`, extending the 16:55-window LOCAL hardening).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the repair-cohort image chain's outcomes.

### 2026-11-18 18:40 UTC - Hourly catch-up (17:40–18:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `b21c95aa3d…` (17:30:01 UTC) to head `781e5ffb8339c60e1b9693b4c00e4967d4f3fc8f`, landed 18:19:28 UTC (~10 landings 18:00–18:19 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log b21c95aa..781e5ffb` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **HARDMath rhs variant fully registered**: the variant registered on its retained release (`f0fbb6ef9d…`) with its registry receipts pinned (`9e837e3ede…`) — completing the chain from the 12:40 window.
  - **Hosted relay/solver lane**: the hosted relay probed before hosted campaign submissions (`6b834a3ee7…`); the hosted solver and shared judge route served from Together (`33f0edefe9…`); the hosted endpoint probed with its served model name (`26fdac0ea3…`); the Together switch imports organized with the unused relay resolution dropped (`aa0933062d…`).
  - **Docs/repair**: the nl2bash prepared actor image recipe recorded (`781e5ffb83…`); Atlas Quality levels explained using recorded FULL review inputs (`d48e270044…`).
  - **Task-image machinery**: the pinned kaniko executor extracted when the task image lacks one (`76f2f2ff87…`); failing Atlas audit requests reported with HTTP and body context (`43c1c2cdff…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the hosted-relay qualification outcomes and HARDMath variant results.

### 2026-11-18 19:40 UTC - Hourly catch-up (18:40–19:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `781e5ffb83…` (18:19:28 UTC) to head `a09fdebbe561e315f46b9c0768c2950aa41f8d23`, landed 19:23:59 UTC (~9 landings 19:20–19:23 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 781e5ffb..a09fdebb` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Preview revision 31**: Atlas docs synced to preview revision 31 (`03bae9c771…`).
  - **Re-pin panel carry-forward**: byte-identical re-pin panels carried through the selected-source router (`87c983f7e5…`) and bound at the current pin (`e70ec80a21…`); QC reviews carried across re-pins on exact task-contract identity (`a5375a617c…`) — the re-pinned Codeforces chain from the 10:40 window is now QC-complete.
  - **Repaired Codeforces receipts pinned**: the repaired Codeforces release's registry receipts pinned (`eda8d327c1…`).
  - **Difficulty report**: prints per-run added sources and per-solver active runs (`a09fdebbe5…`, head).
  - **Typing/preservation**: typed absent-versus-null normalization with the recovered SAMPLE outcome (`1f2d457784…`); the campaign work root created before the registry authfile (`01a38cd0f0…`); a recoverable Atlas Source error shown when required scripts fail (`f205d5443c…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; rev-31 hosted verification remains preview-lane evidence [M].
- Decision and next action: next hourly check watches for rev-31 hosted verification and the re-pinned Codeforces admission outcomes.

### 2026-11-18 20:40 UTC - Hourly catch-up (19:40–20:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `a09fdebbe5…` (19:23:59 UTC) to head `12868b01a348ee22f1a9cc5bfdd3d424e998ad23`, landed 20:17:28 UTC (3 landings 19:45–20:17 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log a09fdebb..12868b01` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Recovery survey completed**: every SAMPLE source's carried prior reviews recovered at its current pin (`e18164622e…`); blank registry shard placeholders skipped in the recovery survey (`f9bb8d4d99…`); the recovery survey flattened over the registry's per-source receipt maps (`12868b01a3…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the recovered reviews' contents are registry-bound [M].
- Decision and next action: next hourly check watches for the recovered-review publication outcomes.

### 2026-11-18 21:40 UTC - Hourly catch-up (20:40–21:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `12868b01a3…` (20:17:28 UTC) to head `f93cc722a6b4c21e4b68dac626e531e4559d66bb`, landed 20:58:49 UTC (2 landings 20:46 / 20:58 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 12868b01..f93cc722` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Recovery queue hardened**: the recovery queue's receipt references pinned to the registry part bytes (`c0cb85d488…`).
  - **NoP verdicts preserved**: nop verdicts recorded before post-grade machine cleanup failures are bound (`f93cc722a6…`, head) — controls graded before a cleanup failure no longer count as missing.
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the recovery queue's publication outcomes.

### 2026-11-18 22:40 UTC - Hourly catch-up (21:40–22:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `f93cc722a6…` (20:58:49 UTC) to head `0b20ffc5b4bc7045b8c8530616a7c4c5a684756d`, landed 22:28:09 UTC (2 landings 22:11 / 22:28 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log f93cc722..0b20ffc5` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Quarantine capacity counting fixed**: only unresolved sandboxes count against quarantine capacity (`48b57ce326…`).
  - **Recovered coverage counting fixed**: carried reviews counted in the recovered coverage even when the panel observed the task (`0b20ffc5b4…`, head) — completing the recovery-queue counting chain from the 21:40 window.
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the recovery queue's publication outcomes.

### 2026-11-19 00:35 UTC - Hourly catch-up (22:40–00:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` head unchanged at `0b20ffc5b4…` (22:28:09 UTC Nov 18) — no new landings this window.
- Command or material configuration: evidence: `git ls-remote` (unchanged head).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new `atlas-env-audit` landings in the window (the fetched main-side commits — marina applet hosts, vLLM adoption #11635, Harbor campaign trial attempt bound #10963, haliax GPU-family helper #11543 — are on main, not this branch).
- Evidence: `git ls-remote` [R].
- Interpretation and limitations: quiet window on the branch.
- Decision and next action: next hourly check resumes normal Atlas coverage.

### 2026-11-19 01:35 UTC - Hourly catch-up (00:35–01:35 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` head unchanged at `0b20ffc5b4…` — no new landings this window (second quiet window in a row).
- Command or material configuration: evidence: `git ls-remote` (unchanged head).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new `atlas-env-audit` landings in the window.
- Evidence: `git ls-remote` [R].
- Interpretation and limitations: quiet window on the branch.
- Decision and next action: next hourly check resumes normal Atlas coverage.

### 2026-11-19 01:40 UTC - Correction to the 00:35 entry: one landing DID land (branch head moved)

- Question or hypothesis and reason for trying it: the 00:35 entry claimed the branch head was unchanged at `0b20ffc5b4…`; the fetch in the same cycle had in fact brought `87056e29e64c9e…` ("Bind prior panels to their population content and record refused priors", landed ~00:33 UTC Nov 18 by session ses_e105) — I misread ls-remote vs the fetched ref. The current head at this cycle's check remains `87056e29e64c9e…` (no further landings).
- Published source revision: branch head `87056e29e64c9e…`.
- Command or material configuration: evidence: `git ls-remote` + fetched log.
- External inputs and artifact references: landing record in the local atlas-push log [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **Correction** — one landing landed in the 00:35 window (`87056e29e64c9e…`: prior panels bound to their population content, refused priors recorded); the "quiet window" claim is withdrawn. Lesson: cross-check `git ls-remote` against the fetched ref before declaring a window quiet.
- Evidence: landing record [R].
- Interpretation and limitations: corrected entry.
- Decision and next action: apply the cross-check going forward.

### 2026-11-19 02:40 UTC - Hourly catch-up (01:40–02:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `87056e29e6…` (~00:33 UTC Nov 18) to head `fdd6471490b820574045150037e2f71803eb5ab2`, landed 00:38:32 UTC (1 landing).
- Command or material configuration: evidence: landings.jsonl tail; `git log 87056e29..fdd64714` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: one landing in the window — **exact top-up reviews reported that complete each source's panel coverage** (`fdd6471490…`, head) — following the recovery-queue counting fixes from the 21:40/22:40 windows; the top-up reviews needed to complete each panel's coverage are now itemized.
- Evidence: commit above; landing timestamp in the local atlas-push log [R].
- Interpretation and limitations: single landing.
- Decision and next action: next hourly check watches for the top-up reviews' execution outcomes.

### 2026-11-19 03:40 UTC - Hourly catch-up (02:40–03:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `fdd6471490…` (00:38:32 UTC) to head `548fb252ee94fceb13e46cbfb192a4fcc0358753`, landed 01:45:08 UTC (1 landing).
- Command or material configuration: evidence: landings.jsonl tail; `git log fdd64714..548fb252` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: one landing in the window — **panel coverage completed with typed fresh reviews of the top-up tasks** (`548fb252ee…`, head) — the itemized top-up reviews from the 02:40 window have been executed as typed fresh reviews, completing panel coverage.
- Evidence: commit above; landing timestamp in the local atlas-push log [R].
- Interpretation and limitations: single landing.
- Decision and next action: next hourly check watches for the completed panels' publication outcomes.

### 2026-11-19 03:40 UTC - Hourly catch-up (02:40–03:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` head unchanged at `548fb252ee…` (01:45:08 UTC) — no new landings this window (cross-checked against the fetched ref).
- Command or material configuration: evidence: `git ls-remote` + fetched ref (cross-checked).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new `atlas-env-audit` landings in the window.
- Evidence: `git ls-remote` [R], cross-checked.
- Interpretation and limitations: quiet window on the branch.
- Decision and next action: next hourly check resumes normal Atlas coverage.

### 2026-11-19 04:40 UTC - Hourly catch-up (03:40–04:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` head unchanged at `548fb252ee…` (01:45:08 UTC) — no new landings this window (cross-checked against the fetched ref).
- Command or material configuration: evidence: `git ls-remote` + fetched ref (cross-checked).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new `atlas-env-audit` landings in the window.
- Evidence: `git ls-remote` [R], cross-checked.
- Interpretation and limitations: quiet window on the branch.
- Decision and next action: next hourly check resumes normal Atlas coverage.

### 2026-11-19 05:40 UTC - Hourly catch-up (04:40–05:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` head unchanged at `548fb252ee…` (01:45:08 UTC) — no new landings this window (cross-checked against the fetched ref).
- Command or material configuration: evidence: `git ls-remote` + fetched ref (cross-checked).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new `atlas-env-audit` landings in the window.
- Evidence: `git ls-remote` [R], cross-checked.
- Interpretation and limitations: quiet window on the branch.
- Decision and next action: next hourly check resumes normal Atlas coverage.

### 2026-11-19 06:40 UTC - Hourly catch-up (05:40–06:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `548fb252ee…` (01:45:08 UTC) to head `a7a5e9debad37bba9fee11ed9db8aa044463cdb8`, landed 05:56:10 UTC (2 landings 05:38 / 05:56 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 548fb252..a7a5e9de` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: the actor image is built inside an ephemeral kaniko sandbox (`a7a5e9deba…`) — extending the kaniko machinery from the 18:40 window.
- Result, including failure or restart: material changes in the window:
  - **Actor image built inside an ephemeral kaniko sandbox** (`a7a5e9deba…`, head).
  - **Workspace image inheritance**: graders that name no machine inherit the workspace image (`db80a48b18…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the kaniko-built actor image outcomes.

### 2026-11-19 07:40 UTC - Hourly catch-up (06:40–07:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` head unchanged at `a7a5e9deba…` (05:56:10 UTC) — no new landings this window (cross-checked against the fetched ref).
- Command or material configuration: evidence: `git ls-remote` + fetched ref (cross-checked).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new `atlas-env-audit` landings in the window.
- Evidence: `git ls-remote` [R], cross-checked.
- Interpretation and limitations: quiet window on the branch.
- Decision and next action: next hourly check resumes normal Atlas coverage.

### 2026-11-18 09:40 UTC - Hourly catch-up (07:40–08:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `a7a5e9deba…` (05:56:10 UTC) to head `561e2c8c091190f47535a54d8bdba70d0bc4db64`, landed 08:02:00 UTC (1 landing).
- Command or material configuration: evidence: landings.jsonl tail; `git log a7a5e9de..561e2c8c` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: one landing in the window — the kaniko sandbox kept writable for apt and the build's own certificates (`561e2c8c0911…`, head), extending the 05:40-window ephemeral-kaniko actor-image work.
- Evidence: commit above; landing timestamp in the local atlas-push log [R].
- Interpretation and limitations: single code landing.
- Decision and next action: next hourly check watches for the kaniko-built image outcomes.

### 2026-11-19 09:40 UTC - Hourly catch-up (08:40–09:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `561e2c8c09…` (08:02:00 UTC) to head `900cb18689180a5b41747594a8de534518855ea7`, landed 09:28:51 UTC (2 landings 08:40 / 09:28 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 561e2c8c..900cb186` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window: the dead tar push after kaniko's own registry push removed (`f8fd6798ea…`); DCAgent python-test installs pinned with hashed requirement locks (`900cb18689…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the kaniko-push outcomes.

### 2026-11-19 10:40 UTC - Hourly catch-up (09:40–10:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `900cb18689…` (09:28:51 UTC) to head `91ebdc65fe2f369d9f1109164038ad2cece7ed70`, landed 10:12:28 UTC (3 landings 09:53–10:12 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 900cb186..91ebdc65` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window: reviews carried across **grading-environment-only re-conversions** (`01b7bda1c0…`); verifyit bracket-scanning fixes (a sequence's brackets scanned once for its enclosing layer and members, `91ebdc65fe…`; a list's outer brackets stripped only when they enclose the whole text, `fd11caddf0…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the re-conversion carry-forward outcomes.

### 2026-11-19 11:40 UTC - Hourly catch-up (10:40–11:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `91ebdc65fe…` (10:12:28 UTC) to head `179314c6d12a2fd9cd280d96f11c7e332ca75033`, landed 11:14:30 UTC (2 landings 10:58 / 11:14 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 91ebdc65..179314c6` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **V4 pinned as the default SAMPLE reviewer profile for new campaigns** (`179314c6d1…`, head) — formalizing the reviewer-profile decision for new campaigns.
  - **Equivalence allowances typed** for calibration, actor setup pinning, and lock resources (`6df109b5a5…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the V4-default rollout outcomes.

### 2026-11-19 12:40 UTC - Hourly catch-up (11:40–12:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `179314c6d1…` (11:14:30 UTC) to head `af480c32e5fa`, landed 12:33 UTC (~10 landings 12:16–12:33 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 179314c6..af480c32` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window — **retained-copy/staging resumability hardening**: the staged catalog load made resumable (`5d70349494…`) with idempotent resume and retained copy (`0598322457…`), staged-prefix counting in resumed insert totals (`87f351bb60…`), upsert of the load row on resume (`7bf42e1ecd…`), resume by skipping each table's staged prefix (`af480c32e5…`, head); retained-copy buckets raised to 128 (`b33881f13b…`); per-task staging batches capped at five rows (`6f52eabb96…`); backend import verified against the database's staged rows (`3cab636545…`); byte-identity branch order modeled in the retained-panel tests (`8532a5562a…`); status-extras output directories created when absent (`074441963d…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the resumability hardening's rollout effects.

### 2026-11-19 14:40 UTC - Hourly catch-up (12:40–14:40 UTC window; covers two revive slots)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` head unchanged at `af480c32e5…` (12:33 UTC) — no new landings this window (cross-checked against the fetched ref).
- Command or material configuration: evidence: `git ls-remote` + fetched ref (cross-checked).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new `atlas-env-audit` landings in the window.
- Evidence: `git ls-remote` [R], cross-checked.
- Interpretation and limitations: quiet window on the branch.
- Decision and next action: next hourly check resumes normal Atlas coverage.

### 2026-11-19 15:45 UTC - Hourly catch-up (14:40–15:45 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` head unchanged at `af480c32e5…` (12:33 UTC) — no new landings this window (cross-checked against the fetched ref).
- Command or material configuration: evidence: `git ls-remote` + fetched ref (cross-checked).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new `atlas-env-audit` landings in the window.
- Evidence: `git ls-remote` [R], cross-checked.
- Interpretation and limitations: quiet window on the branch.
- Decision and next action: next hourly check resumes normal Atlas coverage.

### 2026-11-19 16:40 UTC - Hourly catch-up (15:45–16:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `af480c32e5…` (12:33 UTC) to head `75a0797a6df736f7607eac1c43b9841a7a5899bf`, landed 16:20:24 UTC (7 landings 16:09–16:20 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log af480c32..75a0797a` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Task replacement acceptance gates stated** (`75a0797a6d…`, head) and **source workflows/evidence reuse defined** (`f0a5d22ccf…`) — the repair-lane's replacement semantics formalized.
  - **Reviewer-version cache separation**: separate cache entries verified for all qualified reviewer versions (`8d66de1f45…`).
  - **Declaration actor binding typed** (`ac9908e558…`); scoped projection waivers allowed until the preview refresh seals (`15646ec7de…`); the codeforces contract census bound to the selected release (`d060a6a4a4…`); serving projections checked against the active catalog (`620773ccb4…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the task-replacement gate outcomes.

### 2026-11-19 17:40 UTC - Hourly catch-up (16:40–17:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `75a0797a6d…` (16:20:24 UTC) to head `2dcc46a22367d88bc9e93f9e49287888f5406dc3`, landed 17:23:42 UTC (~10 landings 16:44–17:23 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 75a0797a..2dcc46a2` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Contract-correction lane hardened**: contract correction inputs frozen with shared receipt serialization (`2dcc46a223…`, head); seven corrected derived-contract receipts selected explicitly (`c468ea49a6…`).
  - **Bounded-window hardening**: staged catalog rows counted in bounded key windows (`49a8b816ef…`); cohort summaries aggregated in bounded task windows (`487107883a…`); count window sizes asserted from query results (`56cd72bb9a…`).
  - **Projection waivers evaluated against the gated audit findings** (`2474d1a4f7…`); retained-view checks scoped to the test schema (`8356b69bef…`); backend counts preserved while error messages redacted (`8f73a4da1c…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the corrected derived-contract receipts' outcomes.

### 2026-11-19 18:40 UTC - Hourly catch-up (17:40–18:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `2dcc46a223…` (17:23:42 UTC) to head `d7cc5109b85b4b4e36a87496fdf4b0a73cbb3ebd`, landed 18:22:00 UTC (~6 landings 18:03–18:22 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 2dcc46a2..d7cc5109` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window: root execution respected in delivery-contract backend fits (`d7cc5109b8…`, head); curriculum imports exercised through the bounded query interface (`b3cc38c9c9…`); curriculum privacy checks kept within the query response limit (`f92f3b48ec…`); archive selection and its historical-input regression clarified (`ef80b356db…`); archived receipt corrections applied before audit parsing (`739aec91d1…`); curriculum scores preserved when license evidence changes (`5ee6baa4a3…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the license-evidence-change outcomes.

### 2026-11-19 19:40 UTC - Hourly catch-up (18:40–19:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `d7cc5109b8…` (18:22:00 UTC) to head `6dfad51a3f9869e3bb94ac3e2758a35e9441d395`, landed 19:29:58 UTC (~11 landings 19:04–19:29 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log d7cc5109..6dfad51a` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Ruling-62 landing chain**: the ruling-62-era atlas landings (waiver seal state, corrected derived-contract receipts, catalog publication hardening) merged into the r31 maintenance merge (`73f5093501…`); ruling-60 projection waivers extended for the ruling-62 landing and the current preview load (`cda2d4c8b6…`); the fingerprint index rebound for the ruling-62 re-measured populations (`024c381d27…`); the ruling-62 re-measured census, lexical, grading and dedup receipts bound for the 26 drifted sources (`b5568f300e…`); withheld-content refusal expectations updated to the unified staging guards (`6dfad51a3f…`, head).
  - **Pristine image graph shared** across source families (`4fd02b192c…`); the pinned EndlessTerminals runtime images preserved (`eacbc9fa99…`).
  - **Audit-seal discipline**: observed refresh seal state required in audit selections (`6e729696b9…`); preview projection waivers expire on the strict refresh seal event (`40b7e6589d…`); the duplicate-review Wilson interval pointed at the moved evalrecords module (`50fa4c0d57…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the ruling-62 receipts' contents are registry-bound [M].
- Decision and next action: next hourly check watches for the ruling-62 receipt outcomes.

### 2026-11-19 20:40 UTC - Hourly catch-up (19:40–20:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `6dfad51a3f…` (19:29:58 UTC) to head `bc7ea126771aa8fcd3742d370e2d73fc8f657d6b`, landed 20:28:59 UTC (~11 landings 20:28 UTC incl. one landing by session ses_e1bc3… = the exp3/exp4 runtime agent).
- Command or material configuration: evidence: landings.jsonl tail; `git log 6dfad51a..bc7ea126` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **atlas-push merge-preserving mode** for merge-carrying branches (`37d9e17bd7…`) — the landing tooling now preserves merges on this branch.
  - **Curriculum projection through canonical IDs** (`6d2090b8c2…`); the live alias projection regression clarified (`761ca7a2ce…`).
  - **Attribution/attribution-journeys**: public-display grants recorded and upstream attribution shown (`13b8806c7b…`); publication attribution and guide targets kept consistent (`b93c76f36a…`); the complete source inventory opened in attribution journeys (`260c57de9c…`).
  - **Curriculum releases read qualified source contracts** (`8337814030…`); qualified fixtures named with a shared converter pin (`821b4d9b08…`); actorless fixture counts recorded explicitly (`3d1f99a410…`); measured browser fixtures parsed with the qualified contract (`bc7ea12677…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the curriculum-release contract outcomes.

### 2026-11-19 21:40 UTC - Hourly catch-up (20:40–21:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `bc7ea12677…` (20:28:59 UTC) to head `b5e169718e5211f1a3cf79716cd1441a481a03d6`, landed 21:10:35 UTC (3 landings 21:10 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log bc7ea126..b5e16971` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Binary curriculum text preserved losslessly** through catalog storage encoding (`ebd770732b…`) and kept when resuming catalog loads (`d2e168fc1f…`) — the catalog now stores and resumes binary text without loss.
  - **Text preflights validated before curriculum selection** (`b5e169718e…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the lossless-storage rollout outcomes.

### 2026-11-19 22:40 UTC - Hourly catch-up (21:40–22:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `b5e169718e…` (21:10:35 UTC) to head `235296be114c8cc5f9b571115f935b04555578b8`, landed 21:56:24 UTC (~10 landings 21:47–21:56 UTC incl. an r33 main merge).
- Command or material configuration: evidence: landings.jsonl tail; `git log b5e16971..235296be` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Fail-closed InfiniBand gate for multi-node RL launches merged from main** (`84f4b534ac…`, #11619) — the #11619 probe summary's recommended gate check is now implemented on this branch via the main merge.
  - **r32 maintenance merge** of the concurrent atlas landings (`13825bdb8b…`); ruling-62 (r32) re-measured census/lexical/grading/dedup receipts bound for **22 drifted sources** (`29e073b5e1…`); the atlas-difficulty-v2-65k16k legacy protocol registered as read-only (`c830c8dc94…`).
  - **r33 main merge** (`235296be11…`, head) bringing #11689 (Daytona log-stream boundary workaround), #11745 (Daytona quota/usage CLI for sizing concurrency), #11686 (Micro210 policy frozen at the model-binding boundary), #11666 (Daytona process bridge on Python 3.8 guests), #11288 (expert RNG sharded at initialization).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: the fail-closed gate applies to future multi-node RL launches; the exp3 `.55` restart predates it.
- Decision and next action: next hourly check watches for the fail-closed guard's first gate interactions and further maintenance merges.

### 2026-11-19 23:40 UTC - Hourly catch-up (22:40–23:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `235296be11…` (21:56:24 UTC) to head `22fe092c6f657ccd39b5daaf58e07f64be55e1ac`, landed 23:02:27 UTC (~7 landings 22:59–23:02 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 235296be..22fe092c` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **Curriculum lineage preserved across verified task identity changes** (`d14d4d4246…`); verified curriculum task correspondence documented (`8edcbe3a4e…`); correspondence checks separated with typed mapping references preserved (`6ec2479953…`); qualified declarations preserved in frozen curriculum inputs (`c5910e2e34…`); qualified curriculum input preservation tested (`055853c447…`).
  - **Legacy retained grading**: the read-only legacy retained grading shape added and the audit's applet reads bounded (`6b99e42ca9…`); concurrent curriculum landings merged into the legacy retained grading fix (`22fe092c6f…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the legacy-retained-grading fix's rollout outcomes.

### 2026-11-20 00:40 UTC - Hourly catch-up (23:40–00:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `22fe092c6f…` (23:02:27 UTC) to head `e0b918dde6eda18d5e450912c7655270ad906a72`, landed 00:32:24 UTC (~10 landings 23:35–00:32 UTC incl. one by the exp3/exp4 runtime agent).
- Command or material configuration: evidence: landings.jsonl tail; `git log 22fe092c..e0b918dd` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **SWE-repair lane landed**: the SWE-rebench V2.1 repair module, fidelity gate driver, wave runs, and the grader-unscorable fix (`eae68562cc…`); vendored harness trees exempted from repo lint (`3ec42a57ff…`, `d43e6bb912…`); the grade template renamed to .py.tmpl (`db1289d6b5…`); lint-clean repair module (`ad12cd7c68…`); the source registered only once its within-source dedup receipt exists (`0618163b36…`); the registration gate flattened (`3a23dd2bf7…`); the repaired source registered as **pending intake measurement** (`e0b918dde6…`, head).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings; the registered source awaits intake measurement.
- Decision and next action: next hourly check watches for the swe-repair intake measurement outcomes.

### 2026-11-20 01:40 UTC - Hourly catch-up (00:40–01:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `e0b918dde6…` (00:32:24 UTC) to head `fc979f62f647e541608fa7ea207dca8cffa72db2`, landed 01:27:29 UTC (~10 landings 01:13–01:27 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log e0b918dd..fc979f62` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **swe-repair intake chain completed**: the intake conversion writes its artifacts under the intake prefix (`c6aa01befe…`); conversion inputs pinned at the landed revision with their current digests (`d2a37a45bd…`) and fetched over HTTP (`defa01cae2…`); intake input overrides require local execution with `--cache-dir` (`ee4060cc88…`); end-of-file newlines on the receipts/intake inputs (`9c871f80ac…`, `be8aa40d33…`); generated Atlas intake metadata refreshed (`d2103ebc0a…`); the gap-exclusion test ported to the typed gap table (`fc979f62f6…`, head).
  - **Curriculum gateway tests**: the gateway limit reused in the curriculum regression test (`cb3867586f…`); curriculum metadata reads bounded for the query gateway (`2e873500be…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the swe-repair intake registration outcomes.

### 2026-11-20 02:40 UTC - Hourly catch-up (01:40–02:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `fc979f62f6…` (01:27:29 UTC) to head `677a1900052a74e6bca9a319741c828c48ba7bfd`, landed 02:32:48 UTC (~8 landings 02:05–02:32 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log fc979f62..677a1900` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window:
  - **swe-repair intake bootstrap hardened**: fetches the landed script over HTTP (`677a190005…`, head) with redirect-following, archive verification, and tree confirmation (`9b51fdb7db…`); an exact-argv launcher for the intake conversion with a loud-failure trap (`baf3c1f10a…`); the intake launcher linted (`f3bd2f220a…`); tomli declared for source-fingerprint baseline comparisons (`5b96873d46…`).
  - **Served-catalog drift handling**: drifted retained bindings recorded as waivable served-catalog gaps instead of crashing (`51dc449f73…`); staging clarified on withheld content (`79831643c2…`); content withholding honored in Atlas browse checks (`35ef316483…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the swe-repair intake bootstrap outcomes.

### 2026-11-20 03:40 UTC - Hourly catch-up (02:40–03:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` advanced from `677a190005…` (02:32:48 UTC) to head `d707f3252056661a2831be7792dcd467227d4759`, landed 03:23:46 UTC (3 landings 03:01 / 03:23 UTC).
- Command or material configuration: evidence: landings.jsonl tail; `git log 677a1900..d707f325` (fetched).
- External inputs and artifact references: landing records as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: material changes in the window: compact curriculum sample records typed (`d707f32520…`, head); curriculum task bodies kept out of the membership shuffle (`f05d1fde58…`); an unbound-census sample binding kept pending instead of crashing with the curriculum converter versions verified (`c69a7035a2…`).
- Evidence: commits above; landing timestamps in the local atlas-push log [R].
- Interpretation and limitations: code/landings only.
- Decision and next action: next hourly check watches for the curriculum-shuffle outcomes.

### 2026-11-20 04:40 UTC - Hourly catch-up (03:40–04:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: branch `atlas-env-audit` head unchanged at `d707f32520…` (03:23:46 UTC) — no new landings this window (cross-checked against the fetched ref).
- Command or material configuration: evidence: `git ls-remote` + fetched ref (cross-checked).
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no new `atlas-env-audit` landings in the window.
- Evidence: `git ls-remote` [R], cross-checked.
- Interpretation and limitations: quiet window on the branch.
- Decision and next action: next hourly check resumes normal Atlas coverage.
