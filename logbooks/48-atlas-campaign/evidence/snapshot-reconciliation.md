# Evidence: Snapshot reconciliation — revision-13 vs production/live-310 population deltas

- **What this is**: population reconciliation between the private preview revision-13 inventory
  (4,117,631 tasks) and the 4,206,876-task live/production population, with whatever delta
  components and hidden-partition counts the local ledgers record. Only what is locally
  attested is published; the rest is marked unmeasured.
- **Compiled at**: 2026-11-15 (UTC)
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/complete/atlas-env-audit/REPORT.md`
  (entries cited inline), [R] `.../atlas-env-audit/qa-census-current101-20261114-0844.json` (referenced),
  [R] `.../atlas-env-audit/backend-v93-v7-independent-input-proof-20261114-1444.json`,
  [R] `.../atlas-oracle-filter-code/mechanical-original-filter-v3/summary.json`,
  [R] `.../atlas-oracle-filter-qa/FINAL_STATUS-v2.json`, [P] hosted preview revisions at
  `applets.marina.oa.dev/a/c9742a15-fb46-4d56-bc88-aa916351ebd3/v/{10,11,12,13}/` (public URLs, private app).
- **Access class**: counts, digests, source IDs only.

## 1. The two totals

| Snapshot | Total tasks | Sources | Where attested |
| --- | ---: | ---: | --- |
| Production / live-310 (strict census) | **4,206,876** | 310 live | [R] atlas-env-audit REPORT.md:700 (strict-census proof `strict-310-independent-proof-20261113-1718.json`: "310 live populations … 4,206,876 tasks"); also hosted previews v9.2/revision-12 "live310/4,206,876" (REPORT.md:173) |
| Private preview v9.3 / **revision-13** | **4,117,631** | 310 (302 measured + 8 explicit zero/unassessed) | [R] REPORT.md:45 (`backend-v93-v7-independent-input-proof-20261114-1444.json`; published from exact `072140fa95…`) |
| **Delta** | **89,245** | 8 zero-Excluded TRAIN sources | [R] REPORT.md:51 + 65 |

Revision-13's identity is attested: hosted revision 13 at
`https://applets.marina.oa.dev/a/c9742a15-fb46-4d56-bc88-aa916351ebd3/v/13/` [P], published from
exact head `072140fa956c8dd8715dae343fdb19ca9459dff2`; current v7 map file `f2ae1654…`, receipt
`1ec393c4…`, actual declarations `6b73b089…` SHA-verified.

**Caveat on "revision-18"**: the task prompt frames the comparison as revision-13 vs
revision-18; a hosted "revision 18" was **not found** in the local artifacts read. The
4,206,876 total is attested as the production/live-310 population (previews 9.1/9.2 = revisions
11/12, and production v76). **TODO**: locate the revision-18 publication record if one exists.

## 2. Revision-13 internal split (inventory counts, not admission claims) [R] REPORT.md:51

| Category | Sources | TRAIN | EVAL |
| --- | ---: | ---: | ---: |
| no-oracle | 268 | 166 | 102 |
| oracle-bearing | 34 | 30 | 4 |
| zero (Excluded) | 8 | 8 | 0 |

Tasks by role: 3,226,076 no-oracle TRAIN + 350,205 no-oracle EVAL + 533,587 oracle TRAIN +
7,763 oracle EVAL = **4,117,631** (inventory counts; "inventory alone admits nothing").

## 3. Delta components — the 89,245

- The delta is the original task mass of the 8 zero-Excluded TRAIN sources that revision-13's
  receipt-based projection excluded. [R] REPORT.md:65: root SHA-verified "the SWE reproduction
  `a0dfb8d9` and shard `84284780`: **89,245 original tasks across 8 sources, 0 admitted**, full
  ID/hash/private-output/installed-receipt rechecks true; these are mechanical exclusions, not
  source execution."
- 4,206,876 − 4,117,631 = **89,245** exactly.
- The 8 source IDs: the SWE-mechanical-exclusion set behind shard `84284780` was **not itemized**
  in the artifacts read — TODO. (The Code lane's eight mechanical-empty releases total 44,634
  tasks with `unpinned_environment` exclusions — a different, overlapping-but-not-equal set; see
  admission-ledger.md §3.)
- Per-source exclusion ledgers for these 8: **unmeasured locally**; the ledger lives in the
  SWE reproduction receipts (shard `84284780`, reproduction `a0dfb8d9`).

## 4. Hidden-partition / category counts recorded

- **Retired partition**: exactly **16** fully-ungraded retired populations totaling **185,906**
  tasks, excluded from live-310 [R] REPORT.md:700, 412.
- **Archived census**: 326 populations / **4,392,782** retained tasks (4,206,876 + 185,906
  reconciles exactly) [R] REPORT.md:865, 955.
- **Train/Eval split (live-310)**: 204 TRAIN = 3,848,908 tasks + 106 EVAL = 357,968 tasks
  [R] REPORT.md:412, 388.
- **Folded (duplicate-folding) counts**: within the archived 326, original deduplication removed
  **1,829,846** copies and TRAIN lexical remediation removed **4,769** further contract groups
  [R] REPORT.md:969. The archive pre-dedup raw total was not reconciled line-by-line in the read
  set — partial only.
- **QUICK/QUICK-BAD display partitions** (v9.2/revision-12, hosted): 229 displayed SAMPLE /
  20 FULL (13 small whole-population panels + 7 explicit) / **61 QUICK**; stored Results 175 GOOD
  / 74 SOME / 5 BAD / 56 null; 245 raw SAMPLE panels → 242 active after 3 FULL precedence;
  five legacy QUICK Results (4 StackExchange GOOD + WizardLM SOME) explain remaining computed
  values [R] REPORT.md:187. Revision-13 UI: 227 SAMPLE / 33 FULL / 50 QUICK; stored 179 GOOD /
  76 SOME / 5 BAD / 50 null [R] REPORT.md:45.
- Difficulty overlay (v9.2): only the qualified 4cff cut — 1,240 records: 116 measured /
  1 pending / 1,123 N0 [R] REPORT.md:173.

## 5. What reconciles exactly vs partially

Exactly attested: 4,206,876 = 4,117,631 + 89,245; 4,206,876 + 185,906 = 4,392,782; live-310
TRAIN+EVAL = 3,848,908 + 357,968 = 4,206,876.

Partial / unmeasured:
- Per-source identity of the 89,245-task 8-source set (shard `84284780`) — TODO.
- A per-source ledger of the 16 retired populations' 185,906 tasks (aggregate only in the
  strict-census proof) — TODO.
- Line-by-line raw→dedup→lexical→retained waterfall per source — not compiled in the read set.
- Revision-14+ projections (TRAIN admitted / EVAL measured-only / oracle-pending) were HELD
  pending the revised counts requirement [R] REPORT.md:51, 25: publication held because the
  intermediate "302 pending" label did not implement the no-original-oracle exception; the
  old 2,647,618-count package was never published and superseded.

## 6. Pointers for the notebook

- Registry snapshot JSONs to re-read for a fuller reconciliation:
  `.../atlas-env-audit/232-id-reference-inventory-current.json`,
  `actual-pipelines-after-id-migration.json`, `consumer-graphs-after-232-id-migration.json`,
  `registry-before-232-id-migration.json` (all [R]).
- The count-projection code that revision-14/18-era projections were to use: builder's frozen
  a2ab/890ea count projection (16 files, reviewed) [R] REPORT.md:35.
