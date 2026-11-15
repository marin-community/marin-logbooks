# Evidence: Snapshot reconciliation — revision-13 vs the production/live-310 population

- **What this is**: population reconciliation between the private preview revision-13 inventory
  (4,117,631 tasks) and the 4,206,876-task production/live-310 population — with the corrected
  delta explanation (the eight mechanically excluded populations), the dated snapshot table, and
  the hidden-partition counts the local ledgers record. Only what is locally attested is
  published; the rest is marked unmeasured.
- **Compiled at**: 2026-11-15 (UTC); major correction by cold-read round 7.
- **Source cut**: recorded 13:51 UTC Nov 15 cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`.
  **Evidence cutoff**: no fact in this file postdates that cut. Latest cited artifacts: the
  v9.3-era count projection (created 2026-11-14T13:20:31Z) and the 2026-11-14 preview readbacks.
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/complete/atlas-env-audit/REPORT.md`
  (entries cited inline), [R] `.../atlas-env-audit/strict-310-independent-proof-20261113-1718.json`,
  [R] `.../atlas-env-audit/backend-v93-v7-independent-input-proof-20261114-1444.json`,
  [R] `.../atlas-env-audit/preview-v92-integration-readback-20261114-0858.json` (referenced),
  [R] `.../active/atlas-preview-rebuild/v14-count-source-projection-v2.json` (rev-9.3-era
  projection, 310 sources, created 2026-11-14T13:20:31Z),
  [R] `.../atlas-oracle-filter-code/mechanical-original-filter-v3/summary.json`,
  [R] `.../atlas-oracle-filter-qa/FINAL_STATUS-v2.json`,
  [P] hosted preview revision URLs `applets.marina.oa.dev/a/c9742a15-fb46-4d56-bc88-aa916351ebd3/v/{10,11,12,13}/`
  (public URLs, private app; contents [M]).
- **Access class**: counts, digests, source IDs only.

## 1. The two totals — and the corrected delta explanation

| Snapshot | Total tasks | Sources | Where attested |
| --- | ---: | ---: | --- |
| Production / live-310 (strict census) | **4,206,876** | 310 live | [R] strict-310-independent-proof-20261113-1718.json ("310 live populations … 4,206,876 tasks"); **already the totals of preview revisions 11/12** — the v9.2/revision-12 readback records "live310/4,206,876 tasks" [R] preview-v92-integration-readback-20261114-0858.json (REPORT.md:173) |
| Private preview v9.3 / **revision-13** | **4,117,631** | 310 (302 measured + 8 explicit zero/unassessed) | [R] REPORT.md:45 (backend-v93-v7-independent-input-proof-20261114-1444.json; published from exact `072140fa95…`) |
| **Delta** | **89,245** | 8 zero-Excluded TRAIN sources | [R] REPORT.md:51, 65 — see §3 |

**Major correction (supersedes the "new conversions" growth explanation)**: the 4,206,876 total
**already existed in earlier preview revisions (11/12) as the production/live-310 population** —
it is not a revision-18-era growth product. The 89,245 delta against revision-13's 4,117,631
corresponds to **eight mechanically excluded populations**: revision-13's receipt-based count
projection carried these eight as zero-Excluded TRAIN rows (the "eight zero replacements" in the
builder's pinset), so their original task mass is absent from 4,117,631 but present in the
live-310/production totals. The earlier explanation that revision-13→18 growth came from newly
converted sources (including the five filtered private Code releases, 9,133 tasks) plus
folding/display changes is **superseded**.

## 2. Revision-13 internal split (inventory counts, not admission claims) [R] REPORT.md:51

| Category | Sources | TRAIN | EVAL |
| --- | ---: | ---: | ---: |
| no-oracle | 268 | 166 | 102 |
| oracle-bearing | 34 | 30 | 4 |
| zero (Excluded) | 8 | 8 | 0 |

Tasks by role: 3,226,076 no-oracle TRAIN + 350,205 no-oracle EVAL + 533,587 oracle TRAIN +
7,763 oracle EVAL = **4,117,631** (inventory counts; "inventory alone admits nothing").

## 3. Delta components — the eight mechanically excluded populations (89,245, itemized)

The eight zero-Excluded TRAIN sources and their original task masses, with typed exclusion
reasons, from the rev-9.3-era projection [R] `v14-count-source-projection-v2.json`
(status `Excluded`, task_count 0, exclusion ledger cited per row):

| Source ID | Original tasks | Typed exclusion reason |
| --- | ---: | --- |
| TaskTrove:DCAgent__swe_rebench_v2_patched_oracle-v2 | 13,562 | `tool_provider_required` (13,562) |
| TaskTrove:GAIR__OpenSWE__openswe_oss | 36,884 | `backend_contract_unsupported` (36,884) |
| TaskTrove:laion__openswe-tasks-patched-v7-oracle-success | 11,730 | `unpinned_environment` (11,730) |
| TaskTrove:laion__r2egym-patched-full-oracle-v3 | 2,574 | `unpinned_environment` (2,574) |
| TaskTrove:laion__swegym-tasks-patched-validated-v5 | 2,428 | `unpinned_environment` (2,428) |
| TaskTrove:laion__swesmith-oracle-filtered-v2 | 12,720 | `tool_provider_required` (12,720) |
| TaskTrove:R2E-Gym__R2E-Gym-V1 | 6,909 | `unpinned_environment` (6,909) |
| TaskTrove:SWE-Gym__SWE-Gym | 2,438 | `unpinned_environment` (2,438) |
| **Total** | **89,245** | |

Sum check: 13,562 + 36,884 + 11,730 + 2,574 + 2,428 + 12,720 + 6,909 + 2,438 = **89,245** =
4,206,876 − 4,117,631. Corroborating record: the SWE reproduction `a0dfb8d9` / shard `84284780`
"89,245 original tasks across 8 sources, 0 admitted, full ID/hash/private-output/installed-
receipt rechecks true; these are mechanical exclusions, not source execution" [R] REPORT.md:65.

**Overlap with the 44,634 empty-Code set**: **disjoint** — none of the eight source IDs above
appears in the Code lane's eight mechanical-empty releases (crosscodeeval-{csharp,java,
typescript}, codenet-python, methods2test-large, toolscale, tulu3-sft-personas-math-sandboxes,
nemotron-code-oracle-filtered; 44,634 tasks, all `unpinned_environment` — admission-ledger.md
§3). Two separately counted eight-source mechanical-exclusion sets; no shared population.

## 4. Dated snapshot table

| App revision | Date (UTC) | Selected catalog / registry / input hash | Total tasks | Count semantics | Publication state |
| --- | --- | --- | ---: | --- | --- |
| Revision 12 (v9.2 preview) | published 2026-11-14 from immutable 101-child `ee1e7702f2…` | NN projection SHA `bb490e2dc0bd…`, completion `54c39b1116…` | 4,206,876 (live310) | task instances per source contract; a task may appear in multiple source releases | hosted & read-back verified (`preview-v92-integration-readback-20261114-0858.json`); hosted URL `…/v/12/` [P URL, private app] |
| Revision 13 (v9.3) | published 2026-11-14 14:04; verified 14:52 | head `072140fa956c8dd8715dae343fdb19ca9459dff2`; v7 map `f2ae1654…`, receipt `1ec393c4…`, actual declarations `6b73b089…` (SHA-verified) | 4,117,631 (eight zero-Excluded rows projected to 0) | same instance semantics; "inventory alone admits nothing" | hosted & verified at `…/v/13/` [P URL]; revision-14's publication APPROVAL later held pending corrected count projection |
| Revision 18 | headers per the 13:55 notebook catch-up: deployed/verified ~02:00 UTC 2026-11-15; package `af8d4d6b…` (59 files) | 310 sources / 4,206,876 total; 3,251,230 shown; 3,322,365 admitted (TRAIN 2,972,160 · EVAL 350,205); 541,350 pending oracle filtration; 343,161 excluded | 4,206,876 | same instance semantics; "shown" is the display filter (folding hides overlap groups and filtered/pending cards) | **"revision 18" as a hosted deployment is NOT established by these local files** — the hosted-deployment/verification evidence lives in the preview lane [M]; the notebook's catch-up record of it is reported [R] |

Production catalog (public v76) unchanged throughout — publication on HOLD by user directive.

## 5. Hidden-partition / category counts recorded

- **Retired partition**: exactly **16** fully-ungraded retired populations totaling **185,906**
  tasks, excluded from live-310 [R] REPORT.md:700, 412; the strict-census proof records the
  count only. **Per-source identities of the 16 are not itemized in the read set** — unknown.
  (Distinct from the separate retirement of 66 pipeline-less declarations, 232,965 counted
  tasks, in `retired-inventory-only.json` — fidelity-census.md §2.)
- **Archived census**: 326 populations / **4,392,782** retained tasks (4,206,876 + 185,906
  reconciles exactly) [R] REPORT.md:865, 955.
- **Train/Eval split (live-310)**: 204 TRAIN = 3,848,908 tasks + 106 EVAL = 357,968 tasks [R]
  REPORT.md:412, 388.
- **Folded (duplicate-folding) counts**: within the archived 326, original deduplication removed
  **1,829,846** copies and TRAIN lexical remediation removed **4,769** further contract groups
  [R] REPORT.md:969. The archive pre-dedup raw total was not reconciled line-by-line in the read
  set — partial only.
- **955,646 not-shown instances (revision-18 headers)**: 4,206,876 − 3,251,230 shown. These are
  the display-filter partitions (folded overlap groups, filtered/pending cards, QUICK/BAD hiding
  categories per the reference [P]). **A per-source breakdown of the 955,646 is not present in
  the local artifacts read** — unknown; the per-snapshot reconciliation ledger lives in the
  registry [M], with the count-projection commits in marin (`11fff26215…`, `21fbf617b2…`) [P].
- **QUICK/QUICK-BAD display partitions** (v9.2/revision-12, hosted): 229 displayed SAMPLE /
  20 FULL (13 small whole-population panels + 7 explicit) / **61 QUICK**; stored Results 175 GOOD
  / 74 SOME / 5 BAD / 56 null; 245 raw SAMPLE panels → 242 active after 3 FULL precedence; five
  legacy QUICK Results (4 StackExchange GOOD + WizardLM SOME) explain remaining computed values
  [R] REPORT.md:187. Revision-13 UI: 227 SAMPLE / 33 FULL / 50 QUICK; stored 179 GOOD / 76 SOME
  / 5 BAD / 50 null [R] REPORT.md:45.
- Difficulty overlay (v9.2): only the qualified 4cff cut — 1,240 records: 116 measured /
  1 pending / 1,123 N0 [R] REPORT.md:173.

## 6. What reconciles exactly vs partially

Exactly attested: 4,206,876 = 4,117,631 + 89,245; the eight-source itemization sums to 89,245;
4,206,876 + 185,906 = 4,392,782; live-310 TRAIN+EVAL = 3,848,908 + 357,968 = 4,206,876.

Partial / unmeasured:
- Per-source ledger of the 16 retired populations' 185,906 tasks (aggregate only) — TODO.
- Per-source breakdown of the 955,646 not-shown instances — TODO (registry [M]).
- Line-by-line raw→dedup→lexical→retained waterfall per source — not compiled in the read set.
- Revision-14+ projections (TRAIN admitted / EVAL measured-only / oracle-pending) were HELD
  pending the revised counts requirement [R] REPORT.md:51, 25: publication held because the
  intermediate "302 pending" label did not implement the no-original-oracle exception; the old
  2,647,618-count package was never published and superseded.

## 7. Pointers for the notebook

- Registry snapshot JSONs to re-read for a fuller reconciliation:
  `.../atlas-env-audit/232-id-reference-inventory-current.json`,
  `actual-pipelines-after-id-migration.json`, `consumer-graphs-after-232-id-migration.json`,
  `registry-before-232-id-migration.json` (all [R]).
- The count-projection code that revision-14/18-era projections were to use: builder's frozen
  a2ab/890ea count projection (16 files, reviewed) [R] REPORT.md:35.
- The eight-source exclusion ledger cited per row lives in the SWE reproduction receipts (shard
  `84284780`, reproduction `a0dfb8d9`) [M].
