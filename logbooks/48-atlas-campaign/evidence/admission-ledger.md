# Evidence: Atlas admission ledger — oracle-filtered releases (QA and Code lanes)

- **What this is**: per-release ledger for the nine QA (math) oracle-filtered releases and the
  fourteen Code oracle-filtered releases, plus the eight zero-output (typed-exclusion) Code
  releases and the nine pending measurement categories, with the Unitsyn lineage. Counts, pins,
  digests and commits only.
- **Compiled at**: 2026-11-15 (UTC); corrected by cold-read round 7.
  Updated 2026-11-15 17:05 UTC (cold-read round 8: §1 QA table completed with all nine private
  repo names and full release revisions transcribed from FINAL_STATUS-v2.json; §5 corrected
  accordingly; commit-mapping note records the local REPORT.md:2094 discrepancy; the branch
  moved past this cut after compilation — post-cut landings are NOT incorporated here).
- **Source cut**: recorded 13:51 UTC Nov 15 cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`.
  **Evidence cutoff**: all facts in this file are as of the cut above unless a line explicitly marks a later observation; any post-cut fact is flagged at its line (round-13 wording fix: no blanket claim). The latest cited observations
  are the 13:15 UTC full-qc REPORT entry (Code14 completion, priority repairs, pass-3c
  finalization) and the 05:33 UTC QA FINAL_STATUS. **Cutoff qualification**: REPORT heading
  times are editorial labels and entry write times are not independently pinned — the Code14
  facts above are treated as pre-cut, while the pass-3c finalization fact carried by the same
  heading has timing **relative to the cut unknown** (last independently time-bound
  observation: repair pending at 12:31:09; see pass3c-failure.md §1). Post-cut branch
  landings (head `9fecbbf0b4…` 16:03:50 UTC and later) are NOT reflected in this file.
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/active/atlas-oracle-filter-qa/FINAL_STATUS-v2.json`
  (completed 2026-11-15T05:33:13Z), [R] `.../atlas-oracle-filter-code/code-fourteen-final-registration-ledger-v1.json`,
  [R] `.../atlas-oracle-filter-code/code-mechanical-empty-bindings-v2/manifest.json` + `measurement.json`,
  [R] `.../atlas-oracle-filter-code/mechanical-original-filter-v3/summary.json`,
  [R] `.../atlas-oracle-filter-code/code-final-atlas-count-projection-after-v1.json`,
  [R] `.../atlas-oracle-filter-code/empty-registry-stage-v1-verification.json`,
  [R] `.../atlas-oracle-filter-code/CODE14-reviewer-token-receipt-from-fullqc-20261115.md`,
  [R] `.../complete/atlas-env-audit/excluded-fourteen.json` + `pending-nine-measurements.json` + REPORT.md:1781,
  [R] `.../active/atlas-preview-rebuild/identities-v19-v2.json` (overlap ledger),
  [P] marin commits c707bfbd5e872db9439f5f39dc8af1405a86e85b, fd670c41d16bd52e259e11c1562f6d4f55f408a3,
  0eaf9b7fb447a4eaf1f528ac47c82ae7b2695407 (commit-message reports),
  [P] private registry `open-athena/marin-rl-data-atlas-registry` (repo name public; contents private).
- **Access class**: metadata/counts/digests only. No task text, prompts or outputs.
- **Finding semantics (applies to every panel row below; round-13 correction per the public runtime code)**: `issue_count` increments **once per reviewed task whose verdict has issues** (`full_qc_review.py#L78–L103` at the recorded cut — permalink https://github.com/marin-community/marin/blob/96fc0736fc95d8f21d11e1c03dfca108d568b220/experiments/post_training/task_curation/full_qc_review.py#L78-L103), so "findings" counts here are **distinct affected reviewed tasks**, not multi-flag event counts. The earlier "issue events; not distinct-affected-task counts" wording is withdrawn. The BAD / SOME_ISSUES quality labels remain **reported** (public commit messages and local ledgers) pending publication of the per-task finding rows for independent recomputation.

## 1. Nine QA (math) releases — oracle filtration + SAMPLE admission (FINAL_STATUS-v2.json)

Lane scope: "Nine QA/math sources with original oracles" — 431,389 tasks, all original-oracle
checks passed (430,489 primary + 900 maintained exact-parent retries; 0 rejected / 0 pending /
0 provider holds). Coverage SHA `62a49f0ae99ba4cb4bf14f1f73f7a4067f17cede7c5f0b3d2dc880e1ce991f84`.
Private publication version `2026.11.14.2-final-private`; fresh duplicate drops 0; lexical drops 0.

**Commit mapping (corrected, and now corroborated by the lane's own final artifact)**: raw
Stack (`laion__nemotron-gym-math-stack-overflow-v3`, 110,600) is admitted by **c707bfbd5e**
(author 2026-11-15 04:35:37 UTC), NOT by 635e7b907f; `635e7b907f` (author 2026-11-15 04:54:03
UTC) carries five pins; `8ea14c762f` (author 05:22:16 UTC) carries the final three. The
per-source `admission_commit` fields in FINAL_STATUS-v2.json bind exactly this mapping. The
processed Stack twin `tasktrove-math_stack-qa` (same 110,600 population) is admitted at
`8ea14c762f`, distinct from raw Stack. Both carry 13-findings panels in the per-source table
below. Any statement assigning raw Stack to 635e7b907f is superseded: the local full-qc
REPORT entry at REPORT.md:2094 names "rawStack/procOracle/procOpen" as landed at
`635e7b907f…`, but raw Stack cannot be among the five pins (110,600 ≠ the five-pin total
110,152) and both the public commit message of c707bfbd5e ("Admit the raw Stack
oracle-filtered pin") and the FINAL_STATUS `admission_commit` field bind raw Stack to
c707bfbd5e — the REPORT parenthetical is a local shorthand error (see sample-panels.md §4
for the corresponding fix).

| Release (source_id) | Retained | Private repo (public name; contents [P]/[M]) | Private release revision | Admission commit | Per-panel reviewed / findings / Result |
| --- | ---: | --- | --- | --- | --- |
| TaskTrove:tasktrove-math_prism-qa | 2,221 | open-athena/campaign-release-tasktrove-math-prism-qa-9433b411-oracle-filtered-20261114-v2 | `12332f2b420313ff5cd635d0bd2ecf688b7db6ae` | `635e7b907f` | 100 / 15 / some_issues |
| TaskTrove:laion__nemo-prism-math-v3 | 2,221 | open-athena/task-trove-laion-nemo-prism-math-v3-a3d51da3-oracle-filtered-20261114-v2 | `5d593e5ca5584f81377a3654abc9a22f87494f96` | `635e7b907f` | 100 / 15 / some_issues |
| TaskTrove:laion__nemotron-gym-instruction-following-calendar-v3 | 5,673 | open-athena/task-trove-laion-nemotron-gym-instruction-follow-aafde1e3-oracle-filtered-20261114-v2 | `5c2d64e5881bbe9c3fc94c6fdfb4a9c8db175e17` | `635e7b907f` | 100 / 1 / good |
| TaskTrove:tasktrove-math_oracle-qa | 57,419 | open-athena/campaign-release-tasktrove-math-oracle-qa-7f7f7b29-oracle-filtered-20261114-v2 | `6656fc7bfaa524198f8a52ae7db3165c69ab5301` | `635e7b907f` | 100 / 11 / some_issues |
| TaskTrove:SankalpKJ__nemotron-math-oracle-filtered-v2 | 57,419 | open-athena/task-trove-sankalpkj-nemotron-math-oracle-filter-eaf403d5-oracle-filtered-20261114-v2 | `ca4f2be9881621bff9f29c2acfc02df02ac60a85` | `8ea14c762f` | 100 / 5 / good |
| TaskTrove:laion__nemotron-gym-math-openmathreasoning-v2 | 42,618 | open-athena/task-trove-laion-nemotron-gym-math-openmathreaso-c006510f-oracle-filtered-20261114-v2 | `48079c75912680dd3d1381e5289e4a47556eb4dc` | `8ea14c762f` | 100 / 30 / some_issues |
| TaskTrove:tasktrove-math_stack-qa | 110,600 | open-athena/campaign-release-tasktrove-math-stack-qa-dd13c545-oracle-filtered-20261114-v2 | `5c6818d70670ff2da80ff73fe062bba8c0937be2` | `8ea14c762f` | 100 / 13 / some_issues |
| TaskTrove:tasktrove-math_openreasoning-qa | 42,618 | open-athena/campaign-release-tasktrove-math-openreasoning-qa-45247a4a-oracle-filtered-20261114-v2 | `5528a398ba2372d53f803ce3e93c6cca2f4b4aae` | `635e7b907f` | 100 / 31 / some_issues |
| TaskTrove:laion__nemotron-gym-math-stack-overflow-v3 (**raw Stack**) | 110,600 | open-athena/task-trove-laion-nemotron-gym-math-stack-overflo-410a4406-oracle-filtered-20261114-v2 | `dbec3ca6aacfb8fc0574633646b22cd5706d8971` | **`c707bfbd5e`** | 100 / 13 / some_issues |

SAMPLE aggregate for the nine: 900 reviewed / 0 unavailable / 0 uncertain; **134 finding
affected reviewed tasks** (123 original preserved + 11 new in repaired rows; one count per affected task per the runtime code); 0 mechanical content drops.
Reviewer: SHA `a3e17ee0607a1bf59bdbb8e0b9bffda2e8788390aaf41c7ab73c20b68325c9ea`, review
runtime revision `87f63bd298ce9b312f0d5571495f683f8f5f2a91`. Admission registry entry:
`part-000.json` @ registry revision `b8920640e68a6a3d0046da9fa1a853a5d83fea00`, entry SHA-256
`fa84d6af281a105f…`; SAMPLE registry entries total 247 with all prior peer entries preserved.
Blockers 0. Intended use: train; oversight: sample. Admission commits list has 3 entries;
current head at completion `8ea14c762f`.

**888/9/3 vs 900/134 — successive versions of the same cohort (reported-until-proven)**: the
earlier readback [R] `atlas-sample-results/pr-pins-body-v7.md` (file dated 2026-11-13) binds a
nine-panel wave with **888 conclusive / 9 unavailable / 3 uncertain** observations (6 GOOD /
3 SOME_ISSUES; five Results provisional; archive tree `9a4ca93bc4a2…`). The 2026-11-15
FINAL_STATUS records the same-shape nine-panel set after repair: **900 reviewed / 0 unavailable
/ 0 uncertain / 134 affected reviewed tasks**, with its own fields recording that the original
observations/findings are preserved (`all_original_findings_preserved = 123`) and repaired rows
carry the artifacts under `2026.11.14.2-final-repair-v2` (the full-qc REPORT.md:2094 records
"all 9 panels 100/100 REVIEWED (33 rows repaired round 1)"). The arithmetic 888+9+3 = 900, the
preserved-findings field, and the repair provenance support reading these as **successive
versions of the same nine QA panels** (pre-repair readback → post-repair final status), with
the Result split legitimately recomputed after repair (6/3 → 2/7). A byte-level cohort-identity
proof is not in the read set — reported until proven; until that proof exists, neither
same-cohort succession nor a different-cohort reading is established as fact (the wording in
sample-panels.md §6 is harmonized to this state).

## 2. Fourteen Code releases — final registration ledger (code-fourteen-final-registration-ledger-v1.json)

Per-source: pinned private revision (full 40-hex IDs below), retained count, seed-0 100-task
SAMPLE panel, findings, TRAIN/EVAL role, Daytona machine slots. All `selection_status:
Available`, blockers []. All counts post-repair where a repair occurred (see below).

| Release (source_id) | Retained | Revision | Reviewed | Unavail | Findings (affected tasks) | Use | Slots (Daytona machine slots reserved for the panel's serving/grading run; a scheduler quantity, not a task count) |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| TaskTrove:DCAgent__exp_rpt_pymethods2test-v3 | 496 | `abf6368f9988a3a97745684fb6c3aa488f2ffa40` | 100 | 0 | 37 | train | 496 |
| TaskTrove:laion__exp_rpt_crosscodeeval-python-v2 | 500 | `5c1270d7d2ec851e65d360e8efe752c556fe14ea` | 100 | 0 | 1 | eval | 500 |
| TaskTrove:laion__nemotron-gym-agent-calendar-v2 | 2,699 | `d885672abbb489b64ee04e2cb1ac60c0901b56f1` | 99 | 1 | 2 | train | 0 |
| TaskTrove:DCAgent__exp_rpt_unitsyn-python-v4 | 487 | `8ecfda4d934131068163ba45e7a59f15d2f722bc` | 98 | 2 | 38 | train | 487 |
| TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2 | 4,951 | `9bb2b58153faa81e9ef156425be4054cffefe0e4` | 96 | 4 | 44 | train | 4,951 |
| TaskTrove:DCAgent__exp_rpt_unitsyn-python-large-v2 | 4,949 | `2e229e8c80f561969c1cab87e4ccbc7fb049ce87` | 98 | 2 | 43 | train | 4,949 |
| TaskTrove:tasktrove-pymethods2test-rebuilt-v1 | 496 | `d066a8fbb4d7cef9d8bd34cc508125dc242645bd` | 95 | 5 | 41 | train | 496 |
| TaskTrove:tasktrove-nl2bash-rebuilt-v1 | 1,426 | `f9a1e4dd911b845932627221aaa2817b30f614d1` | 98 | 2 | 22 | train | 1,426 |
| TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2 | 1,422 | `780889d966a2913b2e70b0a35a32b11740f31c01` | 99 | 1 | 15 | train | 1,422 |
| TaskTrove:tasktrove-pymethods2test-large-rebuilt-v1 | 4,951 | `96d040e14c48eca5060fcd65311b5a3ace909bf4` | 98 | 2 | 39 | train | 4,951 |
| TaskTrove:tasktrove-unitsyn-python-large-rebuilt-v1 | 4,951 | `e1cc81dfc0d125107caf2f9cde1d495028883696` | 99 | 1 | 46 | train | 4,951 |
| TaskTrove:tasktrove-taco-built-image | 5,065 | `45be6203b9c21b90ec99e5eb27a9a2c72ae6a017` | 99 | 1 | 24 | train | 5,065 |
| TaskTrove:tasktrove-unitsyn-python-rebuilt-v1 | 4,950 | `be15b2fd3ad2cbc60dfb55cbe95c5b4fe5cb983b` | 100 | 0 | 54 | train | 4,950 |
| TaskTrove:laion__exp_rpt_taco-v2 | 5,063 | `77909f8c8049f881c0b6d33f536347eec88bbc4b` | 100 | 0 | 25 | train | 5,063 |

**Post-SAMPLE repair — corrected characterization**: the earlier sentence "the two sources
above the coordinator's 5% issue bar were repaired to 100/100" was misleading. The coordinator's
5% bar is the **SAMPLE missing-reviews tolerance** (SAMPLE qualifies with ≥95 conclusive of a
100-row panel; max missing = floor(0.05 × attempted rows) — ruling `sample-missing-tolerance-20261115`),
**not a content-issue bar**. "Repair" here = **re-driving the missing (unavailable) reviews to
conclusive** with a priority worker; it does not address content findings. The two panels that
had exceeded the missing-reviews tolerance were `tasktrove-unitsyn-python-rebuilt-v1` and
`laion__exp_rpt-taco-v2`; both are recorded 100 reviewed / 0 unavailable in the final
registration ledger (artifact SHAs `b9c27181…` and `6d6ebf3c…` per the 13:15 UTC REPORT entry).
Exact pre-repair per-panel reviewed/missing counts are **not itemized in the local artifacts
read** (the only lane-level aggregate is the token receipt's caveat that the **45** pre-repair
unavailable rows' failed/truncated requests were uncached, of which the final ledger retains
21 across the 14 panels). Ordering: the priority repair preceded final registration — the Code
lane registered 12/14 directly (commit `ab0f5e19aa`), with the final TACO transaction staged
under registry.lock. **Final quality labels unchanged by the repair**: Unitsyn-rebuilt still
carries its 54 affected reviewed tasks and is a **BAD** Result, and TACO-laion still carries 25 affected
tasks — both per the public commits [P] `fd670c41d16bd52e259e11c1562f6d4f55f408a3`
("54 findings in 100 reviewed tasks, a BAD content Result") and `0eaf9b7fb447a4eaf1f528ac47c82ae7b2695407`
("25 findings and no unavailable or uncertain rows"). The BAD/SOME_ISSUES labels are reported
(commit) until per-task counting inputs are available.

Code14 SAMPLE token receipt (relayed from the retired Code lane, [R]
`.../atlas-oracle-filter-code/CODE14-reviewer-token-receipt-from-fullqc-20261115.md`): 1,381
unique cached requests (1,366 + 15 priority); 12,411,285 prompt / 1,629,030 completion
(1,199,817 reasoning, 73.6%) tokens; finish_reason=length among cached responses 0; failed
attempts are uncached so failed-attempt spend is not receipted. GPU basis: 61.138878 GPU-h
(created→finished).

## 3. Eight zero-output Code releases — typed exclusions (mechanical-original-filter-v3 + empty-bindings)

All eight were admitted 0 of their original tasks under the mechanical-originals policy.
Input→output accounting from `mechanical-original-filter-v3/summary.json` [R]: original
population counts below are the cohort `tasks` counts; **typed drop reason is
`unpinned_environment` for 100% of every population**; oracle passes 0, scored oracle failures
0, source commands executed 0 (mechanical receipts only, `MechanicalOracleTaskReceipt`); all
eight share backend revision `c32a278a1e7ce7053dd20d14671960aae1f9c843`.

| Release (source_id) | Original | Excluded (all `unpinned_environment`) | Filtered release revision (prefix) |
| --- | ---: | ---: | --- |
| laion__exp_rpt_crosscodeeval-csharp-v4 | 1,768 | 1,768 | f3ecbc0ba1 |
| laion__exp_rpt_crosscodeeval-java-v3 | 2,139 | 2,139 | 542329b074 |
| laion__exp_rpt_codenet-python-v4 | 6,975 | 6,975 | 0320e76df0 |
| laion__exp_rpt_methods2test-large-v4 | 1,194 | 1,194 | 403a8fc1c6 |
| laion__exp_rpt_crosscodeeval-typescript-v2 | 3,356 | 3,356 | 037e2acbb1 |
| laion__toolscale-v4 | 4,048 | 4,048 | 3e51233d84 |
| laion__tulu3-sft-personas-math-sandboxes-verified-v3 | 9,998 | 9,998 | 23ece71989 |
| SankalpKJ__nemotron-code-oracle-filtered | 15,156 | 15,156 | 16efa9039d |
| **Total** | **44,634** | **44,634** | |

Registry stage proof: `empty-registry-stage-v1-verification.json` [R] — registry parent
`ffe23de2a6…`, integration revision `bdab76fa32…`; oracle_release_receipts 8→16, oracle_admitted
11 (unchanged), sample_qc 247 (unchanged).
Related but distinct: the SWE lane's reproduction shard `84284780` records 89,245 original
tasks across **8 different** sources, 0 admitted — mechanical exclusions, not source execution
(atlas-env-audit REPORT.md:65; the eight source IDs are itemized in snapshot-reconciliation.md
§3; the two eight-source sets are disjoint by source identity).

## 4. Nine pending measurement categories (pending-nine-measurements.json)

Nine pending categories covering ten source rows (ARC has two variants), measured 2026-11-12
with zero provider calls; per-row admission state `blocked_pending_decontamination_resource_qualification_and_QA`.
Puzzles 6,926→6,926; calendar-agent 2,699; if_calendar 5,673; structured 9,437→9,008 (rejections:
invalid_schema 218, null_grader 15, unsatisfiable_schema 196); ifeval 46,391→46,242
(exclusive_language_positional_word 149); knowledge_mcqa 616,888→613,002; multichallenge 1,068;
reasoning_gym 14,259; arc-inductive 10,000 (all excluded `source_defect:grading_environment_mismatch`,
Daytona unsafe `host_image_required`); arc-transductive 10,000 (same).

## 5. Input→output accounting for the fourteen Code releases (where locally attested)

- The registration ledger and `code-final-atlas-count-projection-after-v1.json` record the
  **current retained = admitted population** per release (e.g. pymethods2test-v3: 496/496,
  excluded_count 0) — the fourteen Code releases' filtration dropped 0 tasks at the oracle
  filter; lexical drops 0 (`lexical_status: clean`, `lexical_dropped_tasks: 0` for all 14).
- **Original (pre-conversion) population counts per Code release are not carried** in the local
  registration artifacts (the mechanical-original-filter-v3 summary covers only the eight empty
  releases). Legacy pinned counts exist for the corresponding *old* cohorts in
  `excluded-fourteen.json` (e.g. pymethods2test 500, pymethods2test-large 4,990, unitsyn-python
  4,991, unitsyn-python-large 4,991, nl2bash 1,498) — those are the superseded cohorts, not
  these releases.
- **PyMethods-small 4-task gap — explicitly unresolved**: `DCAgent__exp_rpt_pymethods2test-v3`
  retains 496 against the legacy 500-row cohort. The typed exclusion ledger for the 4 dropped
  rows was **not found** in the local artifacts read (the mechanical-original-filter-v3 summary
  contains no pymethods entry; the projection records 496→496 with excluded_count 0, so the 4
  rows were dropped before the oracle filter, at conversion/assembly). TODO.
- **Full private release revision IDs**: the 40-hex revisions are published in the §2 table.
  The full private **repo names and release revisions for all nine QA releases** were
  transcribed from FINAL_STATUS-v2.json into §1 in the round-8 update (previously six rows
  read "(per FINAL_STATUS)"). The full private **repo names for the fourteen Code releases**
  are not carried in the local registration ledger (the code14 JSON carries source IDs,
  revisions and per-panel SHA-256 only; prefix-level identifiers otherwise) — **not present
  in the local read set** [M]; the same applies to the fourteen Code releases' converter
  identities (not itemized locally — §6). No QA or Code row should be read as having a
  published repo identity that §1/§2 does not actually contain.

## 6. Unitsyn lineage (with fidelity-census.md §4)

| Lineage step | Identity | Count | Status | Source |
| --- | --- | ---: | --- | --- |
| Legacy small source (retained) | `TaskTrove:tasktrove-unitsyn` | 491 | faithful; deliberately NOT deprecated (its purported replacement had the wrong population) | [R] REPORT.md:1500, 1443-1501 |
| Purported small rebuilt release (**quarantined**) | Campaign Release `tasktrove-unitsyn-python-rebuilt-v1` (name `unitsyn-python-v1-rebuilt`); dataset `open-athena/tasktrove-unitsyn-python-v1-taskcompendium-20261112` @ `3ed40d26a51c8ca0cf26c15cc84316409534a937`; pinned upstream declared `DCAgent__exp_rpt_unitsyn-python-v4 @ 02923004846e…` | 4,991 | **Excluded/quarantined**: payloads identify `DCAgent__exp_rpt_unitsyn-python-large-v2@1` — the LARGE population, not the 491-row small cohort; publishes only README+tasks, no qualification/liveness receipts | [R] REPORT.md:1781; baseline-declarations.json |
| Corrected successor (admitted) | `TaskTrove:tasktrove-unitsyn-python-rebuilt-v1`, private revision `be15b2fd3ad2cbc60dfb55cbe95c5b4fe5cb983b` | 4,950 | SAMPLE 100/100, 54 affected reviewed tasks, **BAD** [P fd670c41d1]; its population is the unitsyn-python LARGE lineage — exact content match (rank_score 1.0) with `DCAgent__exp_rpt_unitsyn-python-large-v2` and `tasktrove-unitsyn-python-large-rebuilt-v1` in the overlap ledger | [R] code14 ledger; [R] identities-v19-v2.json |
| Renamed large population | `TaskTrove:tasktrove-unitsyn-python-large-rebuilt-v1`, revision `e1cc81dfc0d125107caf2f9cde1d495028883696` | 4,951 | SAMPLE 99/1, 46 affected reviewed tasks | [R] code14 ledger |
| Original large release | `TaskTrove:DCAgent__exp_rpt_unitsyn-python-large-v2`, revision `2e229e8c80f561969c1cab87e4ccbc7fb049ce87` | 4,949 | SAMPLE 98/2, 43 affected reviewed tasks | [R] code14 ledger |
| Legacy large source | `TaskTrove:tasktrove-unitsyn_large` | 4,991 | faithful; deprecated | [R] REPORT.md:1501 |
| Small exp_rpt source | `TaskTrove:DCAgent__exp_rpt_unitsyn-python-v4`, revision `8ecfda4d934131068163ba45e7a59f15d2f722bc` | 487 | SAMPLE 98/2, 38 affected reviewed tasks | [R] code14 ledger |

**Converter identities**: converter versions for these releases are not itemized in the local
read set (the code14 ledger carries release revisions only) — unknown; the quarantined
release's declared pinned upstream revision is recorded above. The three large-lineage
releases' near-identical counts (4,949/4,950/4,951) reflect per-release retained filtering over
one shared population, per the overlap ledger's exact-content matches.

## 7. Explicit gaps

- Per-release TRAIN/EVAL for QA releases uniformly `train` in the ledger; no EVAL Code release
  except crosscodeeval-python-v2 (`eval`).
- Original→retained deltas for the 14 Code releases beyond the PyMethods-small 4-task gap were
  not itemized in the read artifacts; TODO.
- The PyMethods 4-task gap (§5) and the 89,245-task 8-source SWE set's per-source exclusion
  ledger (itemized by population in snapshot-reconciliation.md §3) remain the open accounting
  items.
