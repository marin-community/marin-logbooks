# Evidence: Atlas admission ledger — oracle-filtered releases (QA and Code lanes)

- **What this is**: per-release ledger for the nine QA (math) oracle-filtered releases and the
  fourteen Code oracle-filtered releases, plus the eight zero-output (typed-exclusion) Code
  releases and the nine pending measurement categories. Counts, pins, digests and commits only.
- **Compiled at**: 2026-11-15 (UTC; from local file timestamps)
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/active/atlas-oracle-filter-qa/FINAL_STATUS-v2.json`
  (completed 2026-11-15T05:33Z), [R] `.../atlas-oracle-filter-code/code-fourteen-final-registration-ledger-v1.json`,
  [R] `.../atlas-oracle-filter-code/code-mechanical-empty-bindings-v2/manifest.json` + `measurement.json`,
  [R] `.../atlas-oracle-filter-code/mechanical-original-filter-v3/summary.json`,
  [R] `.../atlas-oracle-filter-code/code-final-atlas-count-projection-after-v1.json`,
  [R] `.../atlas-oracle-filter-code/empty-registry-stage-v1-verification.json`,
  [R] `.../complete/atlas-env-audit/excluded-fourteen.json` + `pending-nine-measurements.json`,
  [P] private registry `open-athena/marin-rl-data-atlas-registry` (repo name public; contents private).
- **Access class**: metadata/counts/digests only. No task text, prompts or outputs.

## 1. Nine QA (math) releases — oracle filtration + SAMPLE admission (FINAL_STATUS-v2.json)

Lane scope: "Nine QA/math sources with original oracles" — 431,389 tasks, all original-oracle
checks passed (430,489 primary + 900 maintained exact-parent retries; 0 rejected / 0 pending /
0 provider holds). Coverage SHA `62a49f0ae99ba4cb4bf14f1f73f7a4067f17cede7c5f0b3d2dc880e1ce991f84`.
Private publication version `2026.11.14.2-final-private`; fresh duplicate drops 0; lexical drops 0.

| Release (source_id) | Retained | Private release revision (prefix) | Admission commit (prefix) | TRAIN/EVAL |
| --- | ---: | --- | --- | --- |
| TaskTrove:tasktrove-math_prism-qa | 2,221 | 12332f2b42 | 635e7b907f | train |
| TaskTrove:laion__nemo-prism-math-v3 | 2,221 | 5d593e5ca5 | 635e7b907f | train |
| TaskTrove:laion__nemotron-gym-instruction-following-calendar-v3 | 5,673 | 5c2d64e588 | 635e7b907f | train |
| TaskTrove:tasktrove-math_oracle-qa | 57,419 | 6656fc7bfa | 635e7b907f | train |
| TaskTrove:SankalpKJ__nemotron-math-oracle-filtered-v2 | 57,419 | ca4f2be988 | 8ea14c762f | train |
| TaskTrove:laion__nemotron-gym-math-openmathreasoning-v2 | 42,618 | 48079c7591 | 8ea14c762f | train |
| TaskTrove:tasktrove-math_stack-qa | 110,600 | 5c6818d706 | 8ea14c762f | train |
| TaskTrove:tasktrove-math_openreasoning-qa | 42,618 | 5528a398ba | 635e7b907f | train |
| TaskTrove:laion__nemotron-gym-math-stack-overflow-v3 | 110,600 | dbec3ca6aa | c707bfbd5e | train |

SAMPLE gate for all nine: 900 reviewed / 0 unavailable / 0 uncertain; 134 findings (123 original
preserved + 11 new in repaired rows); 0 mechanical content drops. Admission registry entry:
`part-000.json` @ registry revision `b8920640e68a…`, entry SHA-256 `fa84d6af281a105f…`;
SAMPLE registry entries total 247 with all prior peer entries preserved. Blockers 0.
Intended use: train; oversight: sample. (Attribution caution: "findings" are review issue-events,
not necessarily distinct tasks.) Current head commit at completion `8ea14c762f…`; commits list has 3 entries.

## 2. Fourteen Code releases — final registration ledger (code-fourteen-final-registration-ledger-v1.json)

Per-source: pinned private revision, retained count, seed-0 100-task SAMPLE panel results,
findings count, TRAIN/EVAL role, Daytona machine slots. All `selection_status: Available`, blockers [].

| Release (source_id) | Retained | Revision (prefix) | Reviewed | Unavail | Findings | Use | Slots |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| TaskTrove:DCAgent__exp_rpt_pymethods2test-v3 | 496 | abf6368f99 | 100 | 0 | 37 | train | 496 |
| TaskTrove:laion__exp_rpt_crosscodeeval-python-v2 | 500 | 5c1270d7d2 | 100 | 0 | 1 | eval | 500 |
| TaskTrove:laion__nemotron-gym-agent-calendar-v2 | 2,699 | d885672abb | 99 | 1 | 2 | train | 0 |
| TaskTrove:DCAgent__exp_rpt_unitsyn-python-v4 | 487 | 8ecfda4d93 | 98 | 2 | 38 | train | 487 |
| TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2 | 4,951 | 9bb2b58153 | 96 | 4 | 44 | train | 4,951 |
| TaskTrove:DCAgent__exp_rpt_unitsyn-python-large-v2 | 4,949 | 2e229e8c80 | 98 | 2 | 43 | train | 4,949 |
| TaskTrove:tasktrove-pymethods2test-rebuilt-v1 | 496 | d066a8fbb4 | 95 | 5 | 41 | train | 496 |
| TaskTrove:tasktrove-nl2bash-rebuilt-v1 | 1,426 | f9a1e4dd91 | 98 | 2 | 22 | train | 1,426 |
| TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2 | 1,422 | 780889d966 | 99 | 1 | 15 | train | 1,422 |
| TaskTrove:tasktrove-pymethods2test-large-rebuilt-v1 | 4,951 | 96d040e14c | 98 | 2 | 39 | train | 4,951 |
| TaskTrove:tasktrove-unitsyn-python-large-rebuilt-v1 | 4,951 | e1cc81dfc0 | 99 | 1 | 46 | train | 4,951 |
| TaskTrove:tasktrove-taco-built-image | 5,065 | 45be6203b9 | 99 | 1 | 24 | train | 5,065 |
| TaskTrove:tasktrove-unitsyn-python-rebuilt-v1 | 4,950 | be15b2fd3a | 100 | 0 | 54 | train | 4,950 |
| TaskTrove:laion__exp_rpt_taco-v2 | 5,063 | 77909f8c80 | 100 | 0 | 25 | train | 5,063 |

Post-SAMPLE repair: the two sources above the coordinator's 5% issue bar were repaired to
100/100 by a priority worker — unitsyn-rebuilt (artifact `b9c27181…`) and taco-laion
(`6d6ebf3c…`). Source: [R] `.../atlas-full-qc/REPORT.md:2220` (13:15 UTC Nov 15). The Code lane
registered 12/14 directly (commit `ab0f5e19aa`), with the final TACO transaction staged under
registry.lock.

PyMethods-small 496/500: the final ledger records 496 retained (both `DCAgent__exp_rpt_pymethods2test-v3`
and `tasktrove-pymethods2test-rebuilt-v1`). The typed exclusion ledger for the 4 dropped rows
was **not found** in the local artifacts read — TODO (candidates: lexical drops vs source-defect
rejections; the mechanical-original-filter-v3 summary contains no pymethods entry).

Code14 SAMPLE token receipt (relayed from the retired Code lane, [R] `.../atlas-oracle-filter-code/CODE14-reviewer-token-receipt-from-fullqc-20261115.md`):
1,381 unique cached requests (1,366 + 15 priority); 12,411,285 prompt / 1,629,030 completion
(1,199,817 reasoning, 73.6%) tokens; finish_reason=length among cached responses 0; failed
attempts are uncached so failed-attempt spend is not receipted. GPU basis: 61.138878 GPU-h
(created→finished).

## 3. Eight zero-output Code releases — typed exclusions (code-mechanical-empty-bindings-v2)

All eight were admitted 0 of their original tasks under the mechanical-originals policy,
exclusion reason `unpinned_environment` for 100% of each population. Input/output pins are
SHA-bound in `manifest.json` [R]; original cohort parquet and oracle-filter JSON live under
`s3://marin-us-east-02a/marin/rl-data/atlas/oracle-filter-code/2026.11.14.79-mechanical-originals-policy-v2/…`
(storage URIs fine to publish; no data).

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
11 (unchanged), sample_qc 247 (unchanged). Zero oracle passes, zero scored oracle failures,
zero source commands executed (mechanical receipts only, `MechanicalOracleTaskReceipt`).
Related but distinct: the SWE lane's reproduction shard `84284780` records 89,245 original
tasks across 8 sources, 0 admitted, full ID/hash/private-output/installed-receipt rechecks true —
mechanical exclusions, not source execution (REPORT.md of atlas-env-audit, 14:52 UTC entry).
**TODO**: map those 8 SWE source IDs explicitly (not found in the read set).

## 4. Nine pending measurement categories (pending-nine-measurements.json)

Nine pending categories covering ten source rows (ARC has two variants), measured 2026-11-12
with zero provider calls; per-row admission state `blocked_pending_decontamination_resource_qualification_and_QA`.
Puzzles 6,926→6,926; calendar-agent 2,699; if_calendar 5,673; structured 9,437→9,008 (rejections:
invalid_schema 218, null_grader 15, unsatisfiable_schema 196); ifeval 46,391→46,242
(exclusive_language_positional_word 149); knowledge_mcqa 616,888→613,002; multichallenge 1,068;
reasoning_gym 14,259; arc-inductive 10,000 (all excluded `source_defect:grading_environment_mismatch`,
Daytona unsafe `host_image_required`); arc-transductive 10,000 (same).

## 5. Explicit gaps

- Per-release TRAIN/EVAL for QA releases uniformly `train` in the ledger; no EVAL Code release
  except crosscodeeval-python-v2 (`eval`).
- Original→retained deltas for the 14 Code releases beyond the PyMethods-small 4-task gap were
  not itemized in the read artifacts (no per-release exclusion manifests read); TODO.
- The "496/500" question above and the 89,245-task 8-source ID list remain unresolved.
