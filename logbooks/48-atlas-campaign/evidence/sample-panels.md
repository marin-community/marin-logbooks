# Evidence: SAMPLE panel ledger — Atlas content QA

- **What this is**: per-panel SAMPLE oversight ledger: panel size, reviewed, unavailable,
  uncertain, findings, Result aggregates, reviewer spec, binding commits, and the archived-245
  aggregate. Counts and digests only.
- **Compiled at**: 2026-11-15 (UTC)
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/active/atlas-sample-results/`
  (checkpoint*-pr-body-v*.md, remaining132-pr-body-v1.md, priority-104-pr-body-v2.md),
  [R] `.../atlas-oracle-filter-qa/FINAL_STATUS-v2.json`, [R] `.../atlas-oracle-filter-code/code-fourteen-final-registration-ledger-v1.json`,
  [R] `.../atlas-full-qc/REPORT.md` (Code14, puzzles SAMPLE entries), [P] private registry
  `open-athena/marin-rl-data-atlas-registry` (panels and observations are private; only counts
  and hashes are reproduced).
- **Access class**: aggregate panel metadata only; no per-task findings text.

## 1. Reviewer spec and panel construction (checkpoint245-pr-body-v3.md [R])

- ReviewerSpec **v3** pins GLM-5.3 at temperature 0.7; panels use **seed 0** and select up to
  **100 tasks per source**. (Earlier waves used pinned ReviewerSpec v2; v3 is the current standard.)
- Labels use conclusive reviews as the denominator, excluding unavailable and uncertain reviews.

## 2. Archived 245 panels (checkpoint245-pr-body-v3.md [R])

| Metric | Value |
| --- | ---: |
| Panels archived | 245 |
| GOOD | 168 |
| SOME_ISSUES | 72 |
| BAD | 5 |
| Selected tasks | 24,030 |
| Conclusive reviews | 23,563 |
| Unavailable observations | 455 |
| Uncertain observations | 12 |
| Matching Knowledge/Safety FULL precedence → active SAMPLE Results | 243 (167 GOOD / 71 SOME / 5 BAD) |
| One repaired IFEval transport timeout | REVIEWED; other 99 observations in its panel preserved |

Earlier checkpoint aggregates ([R] PR bodies in atlas-sample-results/): 132 sources = 76 GOOD /
56 SOME (12,918 conclusive, 202 unavailable, 10 uncertain); 104 sources = 61 GOOD / 43 SOME;
100 panels = 59 GOOD / 41 SOME. The 245 = 168/72/5 split is the closing aggregate.

## 3. Current-era panels (post-archive, per lane)

### Code14 wave (code-fourteen-final-registration-ledger-v1.json [R]; 14 panels × ~100)

All panels 100-size (reviewed 95–100 due to unavailable); findings are issue-events from the
reviewer, not necessarily distinct affected tasks. See admission-ledger.md for the full table.
Anchor panels: Unitsyn-rebuilt 54 findings (tasktrove-unitsyn-python-rebuilt-v1), TACO-laion 25
(laion__exp_rpt_taco-v2), PyMethods-small 37 (pymethods2test-v3), CrossCode 1
(crosscodeeval-python-v2, EVAL), plus nl2bash 22/15, pym-large 44/39, unitsyn-large 43/46,
unitsyn-v4 38, calendar-agent 2. Two >5% sources repaired to 100/100 by the priority worker
(unitsyn-rebuilt `b9c27181…`, taco-laion `6d6ebf3c…`). Binding commit `ab0f5e19aa` (12/14
registered directly; TACO staged under registry.lock). [R] atlas-full-qc/REPORT.md:2220.

A **Stack panel with 13 findings** was not found in the artifacts read — TODO (the Stack name in
the Code14 ledger is `tasktrove-math_stack-qa`, which is the FULL-QC stack source, 110,600 tasks,
not a 13-finding SAMPLE panel).

### Puzzles SAMPLE (26-minute run; registered `4bdbc7eeed` by its lane)

[R] atlas-full-qc/REPORT.md:2220. The "22/97" per-panel issue count in the task prompt was not
found in local artifacts — TODO. Related measured facts: checkpoint149 body reports HardMath
"issues in 56 of 97 conclusive reviews" and rebuilt Unitsyn Large "49 of 96" [R]
(checkpoint149-pr-body-v1.md) — the 97-denominator numbers belong to those panels, not Puzzles.

### QA-lane nine panels (FINAL_STATUS-v2.json [R])

900 reviewed / 0 unavailable / 0 uncertain across nine 100-task panels; **134 findings**
(123 original preserved + 11 new in repaired rows); "final-three" interpretation: the
134-finding aggregate covers the final three admission commits (635e7b907f / 8ea14c762f /
c707bfbd5e). Result labels for these panels: the three canonical first admissions
(rawStack/procOracle/procOpen at 635e7b907f) were deep-verified by the QA lane [R]
atlas-full-qc/REPORT.md:2094. Panel digest example: processed-prism panel `ddafe587…` verified
against the QA lane's quoted SHA [R] REPORT.md:2117.

### CrossPy encoding-repair panel

[R] atlas-full-qc/REPORT.md:2141: panel `open-athena/crosscode-python-b3fb68e4-encoding-repair-20261114-v2 @80dff000`,
converter `2026.11.14.crosspy-encoding-repair-v2`, task_specs `b7291756…`, retained 500, seed-0
sample 100, pinned-V2 reviewer identity, **GOOD 2/100 conclusive** (this is the repair-candidate
panel; 2 conclusive GOOD is the GOOD count, not an issue count).

## 4. Binding commits and receipts (where recorded)

- SAMPLE checkpoint bindings: #11364 author-frozen `19efd88c132e44c196ac69cbcad3c02bedf8e4aa`;
  #11368 normal-refresh `52498aef9bfbd38f9ae8bc56931e95a28446f042` [R] atlas-env-audit REPORT.md:594.
- Expanded-100: #11400 merged 22:10:47 UTC at `e998bf387c` (59 GOOD / 41 SOME) [R] REPORT.md:514.
- Expanded-132: #11414 merged 00:18:48 UTC at `c497` (132 = 76 GOOD / 56 SOME) [R] REPORT.md:466.
- Checkpoint-245: #11451 merged 05:23:32 UTC as `8cab748694` (raw 245 = 168/72/5) [R] REPORT.md:288.
- Private archive readback (nine-panel wave): registry tree `9a4ca93bc4a2…`, 888 conclusive
  reviews, nine unavailable, three uncertain; five Results carry a provisional flag for
  unavailable/uncertain observations [R] pr-pins-body-v7.md.

## 5. Result semantics and caveats

- Result thresholds (inclusive boundary ruling): exactly 10% issues = GOOD, exactly 50% = BAD [R]
  atlas-env-audit REPORT.md (17:11 UTC entry).
- Findings may be issue-events: one task can generate multiple flags (e.g. FULL-QC code-oracle
  reports top overlapping flags 706 rubric-mismatch + 217 missing-context + 193 wrong-reference +
  72 ambiguity against 1,026 GOOD of 15,087 conclusive) [R] atlas-full-qc REPORT.md:2214.
- GOOD/SOME/BAD rollups cited at multiple counts (209, 181, 149, 132, 104, 100) are snapshots of
  the same growing archive, not independent panels.

## 6. Explicit gaps

- Per-panel binding commit + panel digest for all 245 archived panels: digests live in the
  private registry receipts (part-*.json); not locally enumerable in the read set — TODO.
- "Puzzles 22/97" and "Stack 13" anchors unresolved (see above).
- v2/v3 reviewer-era boundary per panel not itemized; v2-era panels are inside the archived 245.
