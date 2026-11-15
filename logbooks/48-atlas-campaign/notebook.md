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
