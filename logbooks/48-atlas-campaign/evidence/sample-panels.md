# Evidence: SAMPLE panel ledger — Atlas content QA

- **What this is**: per-panel SAMPLE oversight ledger: panel size, reviewed, unavailable,
  uncertain, findings, Result aggregates, reviewer spec, binding commits, and the archived-245
  aggregate, plus the panel-rollup snapshot map and the Puzzles four-attempt table. Counts and
  digests only.
- **Compiled at**: 2026-11-15 (UTC); corrected by cold-read round 7.
  Updated 2026-11-15 17:05 UTC (cold-read round 8: §3 Math-Oracle and §7 code-oracle numerators
  labeled with explicit fields; Puzzles "22 affected tasks" relabeled as finding events;
  §5 full pin/receipt IDs and the 3.281682-GPU-h ÷ 8-GPU duration reconciliation; §6 nine-panel
  lineage wording harmonized with admission-ledger.md §1; the branch moved past this cut after
  compilation — post-cut landings are NOT incorporated here).
- **Source cut**: recorded 13:51 UTC Nov 15 cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`.
  **Evidence cutoff**: no fact in this file postdates that cut. The latest cited events are the
  Puzzles SAMPLE registration (public commit 4bdbc7eeed, author 2026-11-15 11:05:16 UTC) and
  the 13:15 UTC full-qc REPORT entry (Code14 panels, repairs). **Cutoff qualification**: REPORT
  heading times are editorial labels; the Code14 panel facts are treated as pre-cut, and any
  pass-3c state carried by the same heading has timing relative to the cut unknown
  (pass3c-failure.md §1). Post-cut branch landings (head `9fecbbf0b4…` 16:03:50 UTC and later)
  are NOT reflected in this file.
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/active/atlas-sample-results/`
  (checkpoint*-pr-body-v*.md, remaining132-pr-body-v1.md, priority-104-pr-body-v2.md,
  pr-pins-body-v7.md), [R] `.../atlas-oracle-filter-qa/FINAL_STATUS-v2.json`,
  [R] `.../atlas-oracle-filter-code/code-fourteen-final-registration-ledger-v1.json`,
  [R] `.../atlas-full-qc/REPORT.md`, [R] `.../active/atlas-puzzles-qa-repair/STATUS.md`,
  [P] marin commits c707bfbd5e872db9439f5f39dc8af1405a86e85b and
  4bdbc7eeed5a861e0bb22d25a434dd5a98744484 (public commit-message reports),
  [P] private registry `open-athena/marin-rl-data-atlas-registry` (panels and observations are
  private; only counts and hashes are reproduced).
- **Access class**: aggregate panel metadata only; no per-task findings text. "Findings" are
  reviewer issue-events, not necessarily distinct affected tasks, except where a source says
  otherwise.

## 1. Reviewer spec and panel construction (checkpoint245-pr-body-v3.md [R])

- ReviewerSpec **v3** pins GLM-5.3 at temperature 0.7; panels use **seed 0** and select up to
  **100 tasks per source**. (Earlier waves used pinned ReviewerSpec v2; v3 is the current
  standard. The v2/v3 boundary per archived panel is not itemized locally — see §5.)
- SAMPLE qualification rule: conclusive reviews ≥ n − floor(0.05 × n) of the fixed selected
  panel; unavailable/uncertain rows count as missing, not conclusive (ruling
  `sample-missing-tolerance-20261115`, mirrored in `docs/references/task-curation.md` [P]).

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
| After TWO FULL precedences (Knowledge, Safety) | 243 active (167 GOOD / 71 SOME / 5 BAD) |
| After THREE FULL precedences (+ Math Oracle) | 242 active (167 GOOD / 70 SOME / 5 BAD) [R atlas-env-audit REPORT.md:225] |
| One repaired IFEval transport timeout | REVIEWED; other 99 observations in its panel preserved |

Earlier checkpoint aggregates ([R] PR bodies in atlas-sample-results/): 132 sources = 76 GOOD /
56 SOME (12,918 conclusive, 202 unavailable, 10 uncertain); 104 sources = 61 GOOD / 43 SOME;
100 panels = 59 GOOD / 41 SOME. The 245 = 168/72/5 split is the closing aggregate.

## 3. Panel rollup map — reconciling 245 / 243 / 242 / 247

- **245** = raw archived SAMPLE panels at the closing checkpoint aggregate (binding #11451
  merged 05:23:32 UTC as `8cab748694` [R] atlas-env-audit REPORT.md:288).
- **243** = active SAMPLE Results after TWO matching Knowledge/Safety FULL precedences
  (167/71/5) [R].
- **242** = after the THIRD FULL precedence (Math Oracle replaces its SOME_ISSUES SAMPLE;
  167 GOOD / 70 SOME / 5 BAD) [R] REPORT.md:225.
- **247** = SAMPLE registry entries after the QA lane's nine fresh exact-pin panels landed
  (registry revision `b8920640e68a…`, FINAL_STATUS 2026-11-15T05:33Z) and as re-quoted at the
  Puzzles SAMPLE registration (registry revision `69eb7669aeb1…`, "247 sample receipts"
  [R] puzzles STATUS.md). The two-entry increment between the 245 archived aggregate and the
  247 registry entries is **not decomposed in the read set** — unresolved (candidates: two
  net-new entries vs retained superseded panels; no local artifact decides).

Snapshot table (source-keyed; as far as local files allow — per-panel digests for all 245 live
in the private registry receipts and are not locally enumerable):

| Snapshot / panel set | Date (UTC) | Reviewer era | Result counts | Status |
| --- | --- | --- | --- | --- |
| First 100 panels | ≤2026-11-13 | v2-era waves | 59 GOOD / 41 SOME | archived, inside the 245 |
| 104 panels | ≤2026-11-13 | mixed v2/v3 | 61 GOOD / 43 SOME | archived |
| 132 panels (#11414, merged 00:18:48, `c497`) | 2026-11-14 | mixed | 76 GOOD / 56 SOME | archived |
| Nine-panel computed-Results wave (pr-pins-body-v7) | 2026-11-13 | pinned GLM-5.3 v3 | 6 GOOD / 3 SOME_ISSUES; 888 conclusive / 9 unavailable / 3 uncertain; 5 provisional Results | bound to archive tree `9a4ca93bc4a2…` |
| 245 archived aggregate (#11451) | 2026-11-14→15, merged 05:23:32 | mixed (v2/v3 boundary not itemized) | 168 GOOD / 72 SOME / 5 BAD | archived |
| Knowledge FULL precedence | ≤2026-11-14 | — | replaces its SAMPLE | FULL-superseded |
| Safety FULL precedence | ≤2026-11-14 | — | replaces its SAMPLE | FULL-superseded |
| Math Oracle FULL (binding `a90ec881…`) | 2026-11-14 07:16 | pinned V2 | **GOOD Result; 5,020 issue rows among 57,253 conclusive reviews (8.7681% issue rate); 44 uncertain; 122 waived unreviewed — 57,253 + 44 + 122 = 57,419 = the retained population; accepted-incomplete final while strict-coverage-false/execution-pending stays explicit.** Whether the 5,020 are distinct affected tasks or issue events is not stated by the source — reported [R REPORT.md:225] | FULL-superseded its SAMPLE |
| CrossCode Python FULL (#11459, merged 06:57:27) | 2026-11-14 | — | GOOD, 8/500 issues, complete coverage, zero unavailable/uncertain, no waiver | 6th current FULL Result |
| QA-lane nine exact-pin panels | 2026-11-15 (FINAL_STATUS 05:33) | reviewer SHA `a3e17ee0…`, runtime `87f63bd2…` | 900 reviewed / 0 unavailable / 0 uncertain / 134 finding events (123 preserved + 11 new in repaired rows) | new-era panels, registry revision `b8920640e68a…` |
| Code14 fourteen panels | 2026-11-15 (completed 13:15 entry) | pinned V3 (524,288 context) | per-release in admission-ledger.md §2 | new-era panels |
| Puzzles SAMPLE panel | 2026-11-15 (registered 4bdbc7eeed) | pinned V3 | 97 reviewed / 3 unavailable / 22 issue events (22.68% reported rate; distinct affected tasks not established — §4) SOME_ISSUES | new-era panel |

## 4. Anchor panels with PUBLIC reports (corrected — the earlier "not found" TODOs are removed)

- **Raw Stack, 13 findings — public report [P]**: marin commit
  `c707bfbd5e872db9439f5f39dc8af1405a86e85b` (author 2026-11-15 04:35:37 UTC), "Admit the raw
  Stack oracle-filtered pin with current SAMPLE evidence": "its exact-pin 100-task content
  review completed. Store its 13 content findings in the current quality result and derive
  SAMPLE oversight from that review's receipt." The Stack name in the Code14/QA ledgers is
  `tasktrove-math_stack-qa` (processed twin, same 110,600 population, admitted at `8ea14c762f`,
  also 13 finding events) — the raw Stack release is
  `laion__nemotron-gym-math-stack-overflow-v3` (see admission-ledger.md §1 for the mapping).
- **Puzzles, 97/100 — public report [P]**: marin commit
  `4bdbc7eeed5a861e0bb22d25a434dd5a98744484` (author 2026-11-15 11:05:16 UTC), "Bind the
  puzzles QA SAMPLE content review receipt": "the seed-0 100-task panel through the pinned V3
  reviewer, 97/100 reviewed with 3 unavailable and 22 recorded source-content issues (22.68%
  issue rate, SOME_ISSUES), bound to the exact current pin population."
- **What remains unavailable for both anchors**: the panels' independently inspectable metadata
  and measurement receipts (panel object contents, per-observation rows) live in the private
  registry [M]; and a distinct-task counting proof (that the 13 / 22 finding events map to 13 /
  22 distinct tasks) is not established by the public commit text — the commit-message figures
  are reported aggregates. Per the round-5 evidence-label rule: the commits prove public
  *reporting*, not independent inspectability.

## 5. Current-era panels (post-archive, per lane)

### Code14 wave (code-fourteen-final-registration-ledger-v1.json [R]; 14 panels × ~100)

All panels 100-size (reviewed 95–100 due to unavailable); findings are issue-events from the
reviewer, not necessarily distinct affected tasks. Full table with full revision IDs:
admission-ledger.md §2. Anchor panels: Unitsyn-rebuilt 54 finding events (BAD [P fd670c41d1]),
TACO-laion 25, PyMethods-small 37, CrossCode 1 (crosscodeeval-python-v2, EVAL), plus nl2bash
22/15, pym-large 44/39, unitsyn-large 43/46, unitsyn-v4 38, calendar-agent 2. The two panels
that had exceeded the 5%-missing-reviews tolerance were re-driven to 100/100 by the priority
worker (Unitsyn-rebuilt `b9c27181…`, TACO-laion `6d6ebf3c…`) — a missing-reviews repair, not a
content fix; final quality labels unchanged (see admission-ledger.md §2 for the corrected
characterization). Binding commit `ab0f5e19aa` (12/14 registered directly; TACO staged under
registry.lock) [R] atlas-full-qc/REPORT.md:2220.

### Puzzles SAMPLE — four-attempt table (attempt timing reconciled)

The notebook's "~02:10–07:45" window is the **wall window of the attempt sequence + receipt
binding** per the puzzles lane's STATUS.md clock; the ledger's **26-minute 3.281682 GPU-h run**
is attempt 4's **serving run** (created→finished). These are different durations of different
things; neither is the other. Note: the STATUS.md entries "~07:20 SUCCEEDED / ~07:45 LANDED"
use the lane's own clock and are **not consistent with the public commit timestamp**
(`4bdbc7eeed` author 2026-11-15 11:05:16 UTC); the public commit timestamp is the publication
evidence and the STATUS.md clock entries are reported [R].

| Attempt | Job ID | Window (UTC) | Hardware / allocation | Outcome |
| --- | --- | --- | --- | --- |
| 1 | `/benfeuer/puzzles-qa-sample-qc-v1` | SchedulingGated from ~2026-11-14 21:05; failed ~2026-11-15 02:10 (14,400 s startup timeout) | 4×GB200 serving gang; never placed, ~0 GPU-h | no verdicts |
| 2 | `/benfeuer/puzzles-qa-sample-qc-v2` | launched ~02:10; failed ~06:15 identically (4 h gate timeout) | 4×GB200; ~0 GPU-h | no verdicts |
| 3 | `/benfeuer/puzzles-qa-sample-qc-v3` | launched ~06:15; CANCELLED by coordinator before placement | — (~0 GPU-h) | no placement |
| 4 (FULL-QC worker) | `/benfeuer/atlas-full-qc-sample-v3-puzzles-v1` | handoff accepted ~06:50; serving run **26 min (rounded)**, ending before the 11:05:16 UTC registration | 8×GB200 (cw-us-east-08a), serving job `…/inference-fa576089876f4c23bcdf8333b783cc86`; **3.281682 GPU-h** (created→finished) | **100 sampled / 97 reviewed / 3 unavailable / 22 issue events (22.68% reported rate; distinct affected tasks not established) / SOME_ISSUES** |

Shared attempt metadata: source pin `open-athena/tasktrove-puzzles-taskcompendium-20261114` @
`adfcbd4a8ddefc9e69febc91931c08ba03372bad` (6,921 rows, task-spec digest
`8542395bb886f09d1d075edeecc6e8f6963c9da5dc04fb8336f991bb4957c6a3`); frozen queue sha
`02ceafdb…` (attempts 1–3, handed to attempt 4 unchanged; the local STATUS.md records this
hash only as a prefix — the full queue SHA is not in the local read set); reviewer pinned V3
(glm-5.3 @ `aca966e4`, temp 0.7, `PINNED_REVIEWER_SHA256 a3e17ee0…` — the same reviewer-spec
SHA as the QA lane's), seed 0; receipts: records SHA
`e9139b6d186db5f6174216ef14f34a706d9668ffed1b0b77f8a2f8e3d0ecd199`, artifact SHA
`4e7295632a8cbb74c2b9f32105bb52eb531f57561aab40015eefd88d3116a161`, campaign record
`s3://…/sample-qc/2026.11.14.1/campaigns/4a84cf4dcca4419aaa101fbb3be9f648/ac2a4f24b9a765b71fbc.json`; registration commit
`4bdbc7eeed` under the shared registry.lock, registry revision
`69eb7669aeb1aef3b10efd755cf8f8a9dd700042` (247 sample receipts). Reviewer identity is bound
by the public commit [P] ("seed-0 … pinned V3 reviewer").
**Attempt-4 duration reconciliation**: 3.281682 GPU-h ÷ 8 GPUs = 0.410210 h ≈ **24.6 min**
created→finished — consistent with the rounded "26-minute" serving description; the two
figures describe the same interval at different rounding, not different events.

**Dedup census ↔ release connection**: the release's population is the deduplicated census —
raw **6,926 → unique 6,921** (5 within-source duplicates), recorded at the conversion census
(input_rows 6,926 / converted_rows 6,921 / rejections {}), the pin-population dedup census
(6,921/6,921/0), and the lexical re-run (same counts) [R] puzzles STATUS.md milestones 1–7;
the SAMPLE panel is drawn from the 6,921-row pinned population.

### QA-lane nine panels (FINAL_STATUS-v2.json [R])

900 reviewed / 0 unavailable / 0 uncertain across nine 100-task panels; **134 finding events**
(123 original preserved + 11 new in repaired rows). Final-three admission commits:
`635e7b907f` / `8ea14c762f` / `c707bfbd5e` — **raw Stack is on `c707bfbd5e`** (corrected; the
earlier claim that raw Stack was admitted at `635e7b907f` is wrong — see admission-ledger.md
§1). Panel digest example: processed-prism panel `ddafe587…` verified against the QA lane's
quoted SHA [R] REPORT.md:2117. See admission-ledger.md §1 for the 888/9/3 → 900/134 succession.

### CrossPy encoding-repair candidate panel (distinct from the admitted CrossCode panel)

[R] atlas-full-qc/REPORT.md:2125, 2141: panel
`open-athena/crosscode-python-b3fb68e4-encoding-repair-20261114-v2 @80dff000`, converter
`2026.11.14.crosspy-encoding-repair-v2`, task_specs `b7291756…`, retained 500, seed-0 sample
100, pinned-**V2** reviewer identity, recorded as "**GOOD 2/100 conclusive**" (bind_sample_result
PASSES; receipt landed with the promotion, private-registry revision `1a1a400e…`).

Count separation and caveat: the receipt's recorded aggregate is a Result (GOOD) computed on
**2 conclusive** observations of a 100-row selected panel. The remaining split (clean/affected
rows among the 2 conclusive, and unavailable/uncertain counts) is **not** in the local artifacts
— unresolved. If "2 conclusive" is the panel's conclusive-review count, the panel would **fail
the SAMPLE qualification rule** (needs ≥95 conclusive of 100) — the local sources do not show
how the receipt reconciles with that rule (e.g. whether the receipt covers a partial panel or
the 2/100 figure has different semantics); **mark unresolved**. This is the **repair-candidate
panel** at pin `b3fb68e4 @80dff000` (Nov 14, converter crosspy-encoding-repair-v2) — distinct
from the **admitted** CrossCode 1/100 panel at pin `5c1270d7d2…` (EVAL, code14 ledger,
admission commit `c9f0235d8e`, Nov 15): different pin, different converter, different date.

## 6. Binding commits and receipts (where recorded)

- SAMPLE checkpoint bindings: #11364 author-frozen `19efd88c132e44c196ac69cbcad3c02bedf8e4aa`;
  #11368 normal-refresh `52498aef9bfbd38f9ae8bc56931e95a28446f042` [R] atlas-env-audit REPORT.md:594.
- Expanded-100: #11400 merged 22:10:47 UTC at `e998bf387c` (59 GOOD / 41 SOME) [R] REPORT.md:514.
- Expanded-132: #11414 merged 00:18:48 UTC at `c497` (132 = 76 GOOD / 56 SOME) [R] REPORT.md:466.
- Checkpoint-245: #11451 merged 05:23:32 UTC as `8cab748694` (raw 245 = 168/72/5) [R] REPORT.md:288.
- Nine-panel archive readback (Nov-13 wave): registry tree `9a4ca93bc4a2…`, 888 conclusive
  reviews, nine unavailable, three uncertain; five Results carry a provisional flag [R]
  pr-pins-body-v7.md (dated 2026-11-13). **Relationship to the QA lane's Nov-15 exact-pin
  panels (900/134): reported as a succession, not established** — the two sets share the same
  shape (nine 100-task seed-0 pinned-GLM-5.3-v3 panels), 888+9+3 = 900, and the FINAL_STATUS
  records its originals preserved with repaired rows, which supports reading the Nov-15 set
  as the post-repair state of the Nov-13 wave; but a byte-level cohort-identity proof (same
  source pins and selected task hashes) is not in the read set. Until then neither
  same-cohort succession nor "a different nine-panel cohort/era" is asserted as fact here —
  see admission-ledger.md §1 for the harmonized reading.

## 7. Result semantics and caveats

- Result thresholds (inclusive boundary ruling): exactly 10% issues = GOOD, exactly 50% = BAD
  [R] atlas-env-audit REPORT.md (17:11 UTC entry).
- Findings may be issue-events: one task can generate multiple flags. Full-code-oracle
  example with explicit fields [R] atlas-full-qc REPORT.md:182, 2214
  (`SankalpKJ__nemotron-code-oracle-filtered`): population 15,156 retained = **15,087 valid**
  (conclusive-review denominator) + **69 INVALID**; 0 unavailable; 0 uncertain; **1,026
  affected conclusive rows** = 6.80% issue rate → **provisional GOOD Result** (≤10% boundary);
  top **overlapping** issue flags 706 rubric-mismatch + 217 missing-context + 193
  wrong-reference + 72 ambiguity (sum 1,188 > 1,026 — flags overlap across rows, so flag
  counts are not affected-row counts). The 1,026 is an affected-row count, not a clean count
  and not a findings-vs-tasks claim; distinct-task counting is not established.
- GOOD/SOME/BAD rollups cited at multiple counts (245, 209, 181, 149, 132, 104, 100) are
  snapshots of the same growing archive, not independent panels.

## 8. Explicit gaps

- Per-panel binding commit + panel digest for all 245 archived panels: digests live in the
  private registry receipts (part-*.json); not locally enumerable in the read set — TODO.
- v2/v3 reviewer-era boundary per archived panel not itemized; v2-era panels are inside the
  archived 245 — do not imply all archived panels used the current (v3) reviewer.
- CrossPy candidate-panel 2/100 semantics vs the SAMPLE ≥95%-conclusive rule — unresolved (§5).
- The 245 → 247 registry-entry increment — unresolved (§3).
