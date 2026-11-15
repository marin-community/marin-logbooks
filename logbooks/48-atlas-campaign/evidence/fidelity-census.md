# Evidence: Atlas environment-fidelity classification census (a)/(b)/(c) — INTERIM BUILDING BLOCKS

> **Status: this is an interim building-blocks file, NOT a completed census.** The aggregate
> "(a) N / (b) M / (c) K" tally over the final population is **NOT measured anywhere in the
> local artifacts read**. What exists here is the classification definition, the population
> crosswalk, the measured environment-result inventory, and the disposition ledgers. Any
> statement of a completed (a)/(b)/(c) census over the 310–326 sources is unsupported.

- **What this is**: compiled record of the environment-fidelity classification *inputs* for the
  Atlas logbook campaign. Definitions come from the user's BRIEF addendum; the per-source
  measured inventory and disposition are compiled from the audit report and its JSON artifacts.
- **Compiled at**: 2026-11-15 (UTC); corrected and re-tallied by cold-read round 7.
- **Source cut**: recorded 13:51 UTC Nov 15 cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`.
  **Evidence cutoff**: no fact in this file postdates that cut; every cited artifact was
  created before it (latest: the Code-lane admission ledger, 2026-11-15 05:33–13:15 UTC).
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/complete/atlas-env-audit/BRIEF.md`,
  [R] `.../atlas-env-audit/REPORT.md` (497 KB, 2,203 lines), [R] `.../atlas-env-audit/inventory.json`
  + `measurements/`, [R] `.../atlas-env-audit/pending-nine-measurements.json`,
  [R] `.../atlas-env-audit/excluded-fourteen.json`, [R] `.../atlas-env-audit/retired-inventory-only.json`,
  [R] `.../active/atlas-preview-rebuild/v14-count-source-projection-v2.json`,
  [R] `.../active/atlas-preview-rebuild/strict-310-independent-proof-20261113-1718.json`,
  [P] definitions mirrored in marin docs `docs/references/task-curation.md` (branch `atlas-env-audit`).
- **Access class**: metadata/counts only — task text, prompts and model outputs are never
  reproduced here.

## 1. Classification definitions (user ruling, BRIEF.md:54-67)

- **(a)** spec fully describes runtime behavior AND QA evidence exists → fine; record evidence.
- **(b)** spec fully describes behavior but QA is missing → needs SAMPLE or FULL QA before it
  stays runnable.
- **(c)** runtime behavior differs from spec (substituted image, missing environment resolved
  to host, or a declared environment ignored) → needs repair.
- Ingestion-time rewrites (lowering, reply-only rewrites) are acceptable as **(a)** when the
  published spec is the whole truth and robust QA passed.
- **28 grandfathered MarinSkyRL blends** (GRANDFATHERED_MARINSKYRL_SOURCES) are explicitly out
  of scope (Addendum 2): not audited, not repaired; left exactly as-is. Source: [R] BRIEF.md:65-67.

## 2. Dated population crosswalk (published as of this correction)

| Population | Count | Date / observation cut | Source | Notes |
| --- | ---: | --- | --- | --- |
| Runnable agentic sources at audit start | 81 | 2026-11-12 (BRIEF baseline) | [R] BRIEF.md:29 | the audit's denominator, not the catalog |
| Completed runtime measurements, first inventory pass | 133 | 2026-11-11 snapshot | [R] REPORT.md:1421 | `pending`/`not executable` explicitly unknown, not zero |
| Live Atlas sources (SQL snapshot) | 311 | 2026-11-11 04:12 UTC, snapshot `8674227db4ca…` | [R] REPORT.md:1421 | — |
| Live-310 strict-census population | 310 | 2026-11-13 17:18 UTC proof | [R] strict-310-independent-proof-20261113-1718.json | 4,206,876 tasks |
| Archived census populations | 326 | 2026-11-13 | [R] REPORT.md:865, 955 | 4,392,782 tasks incl. 16 retired |
| Registry unique generated IDs | 326 | 2026-11-13 (232-alias migration) | [R] REPORT.md:2101, 2113 | 232 approved aliases cover 229 current legacy registrations + 3 historical retired QA IDs |
| Union-inventory rows (inventory.json baseline) | 328 | 2026-11-11 | [R] REPORT.md:1421, 2113 | 326 unique IDs + later alias rows |

**311 → 310 reconciliation**: the 2026-11-11 SQL snapshot counted 311 live sources; the
2026-11-13 strict census proves 310 live + exactly 16 fully-ungraded retired populations. The
identity of the source that moved out of the live set between the two cuts is **not itemized**
in the read artifacts (the strict proof records counts only) — unknown. The 16 retired
populations total **185,906** tasks (aggregate only; per-source identities of the 16 are also
not itemized in the read set — unknown; note the *separate* 66 pipeline-less entries retired
from code declarations in `0f385622b8`, tracked in `retired-inventory-only.json` with 232,965
counted tasks, are a different retirement).

**15 grandfathered missing rows vs 28 excluded blends**: the fidelity census counts
(a)/(b)/(c) over **non-grandfathered** sources only. The raw inventory table (§3) DOES include
grandfathered blend rows — 24 rows carry `missing:` environment-result labels in the published
table (the `nemotron_ultra_*` mopd/rlvr blends). The 28-blend grandfathered set is larger than
those 24 rows; the identities of the remaining blends not appearing with a `missing:` label in
the published table are **not itemized in the read set** (unknown). So: blends are excluded
from census counts, and the raw inventory includes (some of) them as rows — both stated.

**ARC-inductive vs ARC-transductive exclusion status**: both variants are typed-excluded with
the same reason: 10,000 rows each, `source_defect:grading_environment_mismatch`, Daytona unsafe
`host_image_required`; both carry the `substituted:` environment-result label in the inventory
(10,000 each). Neither is repaired; neither is in the requalification queue.

## 3. Environment-result inventory (recounted; REPORT.md table, lines 1423-1704)

The published inventory table has **281 rows**, each carrying exactly one label (labels do not
overlap). Counting basis: every row of the table at REPORT.md:1423–1704 tallied by its
`Environment result` field. **The earlier draft's tally (66 rows = 17+2+15+27) was wrong**; the
correct label tallies are:

| Label | Kind | Rows | Notes / examples (counts per source) |
| --- | --- | ---: | --- |
| `faithful:` | fidelity outcome | 16 | tasktrove-code_contests 8,728→8,222; taco 10,000→5,116; stack_pytest 500→492; unitsyn 491; curriculum_easy 509→505; swe_rebench 18,319→13,562; … |
| `substituted:` | fidelity outcome | 2 | tasktrove-arc_inductive 10,000; tasktrove-arc_transductive 10,000 |
| `reply_only_rewrite:` | fidelity outcome | 24 | ifeval 46,391→46,242; knowledge_mcqa 616,888→613,002; science-openqa 150,644; safety 89,066; puzzles 6,926; … |
| `missing:` | fidelity outcome | 24 | the `nemotron_ultra_*` mopd/rlvr blend rows in the table (grandfathered, out of scope) |
| `docker_image:` | environment kind (rebuilt/private-qualified cohorts) | 15 | calibforge 1,357; codeforces 9,745; competitive-coding 13,973; unitsyn-python-v1-rebuilt (excluded, §4) … |
| `native task.toml` | environment kind | 51 | swesmith-verifier-repaired-v5 7,016 (5 snapshots, FULL oversight); DCAgent excluded cohorts … |
| `not_executable_inventory` | inventory status | 149 | evalchemy-*, GAIR OpenSWE, SWE-Gym, R2E-Gym, XiaomiMiMo pools, benchmarks (aime24, apps, …) |
| **Total** | | **281** | 66 fidelity-outcome + 15 docker_image + 51 native task.toml + 149 not_executable |

The four fidelity-outcome labels (faithful/substituted/reply_only_rewrite/missing) sum to
**66 rows** — the "66" figure of the earlier draft was the fidelity-outcome subtotal, but its
component counts (17/2/15/27) were wrong; `docker_image:` and `native task.toml` are
environment-kind labels, not fidelity outcomes, and `not_executable_inventory` is an inventory
status. Per-source rows are the table itself (REPORT.md:1423-1704); the tally above is the
stated counting basis.

## 4. Dispositions produced by the campaign (REPORT.md 1813-1901 and cross-references)

- 14 excluded cohorts requalified: `excluded-fourteen.json` [R] lists them with pinned row
  counts totaling 53,842 (calibforge 1,357, nl2bash 1,498, codeforces 9,745, competitive-coding
  13,973, curriculum-easy 505, curriculum-medium 488, e2egit 477, e2egit-large 4,993, multifile
  4,842, pymethods2test 500, pymethods2test-large 4,990, stack-pytest 492, unitsyn-python 4,991,
  unitsyn-python-large 4,991). CodeContests (8,222) and TACO (5,116) requalified separately via
  #11237; total requalification queue 16 cohorts / 67,180 pinned rows (REPORT.md:1813, 1838).
- Nine pending measurement categories (ten source rows; ARC has two variants): all measured
  without provider calls; per-source pins, converter versions and exclusion ledgers in
  `pending-nine-measurements.json` [R] (e.g. puzzles 6,926 input→6,926 converted, 0 rejections;
  knowledge_mcqa 616,888→613,002 with typed rejections invalid_reference 3,824, reviewed_defect
  1, unsupported_answer_contract 61; ARC-inductive/transductive as in §2).
- Repaired/unrepaired: repaired = sources whose (c) findings were fixed via true-environment
  builds or exact lowerings, then qualified with the full-population oracle filter and admitted
  (see admission-ledger.md). Unrepaired/out-of-scope = the 28 grandfathered blends and the two
  ARC variants (100% typed exclusion, source defect).
- Deprecated legacy sources with admitted faithful replacements: 14 deprecated
  (tasktrove-code_contests, -codeforces, -competitive_coding, -curriculum_easy,
  -curriculum_medium, -e2egit, -e2egit_large, -multifile, -nl2bash, -pymethods,
  -pymethods_large, -stack_pytest, -taco, -unitsyn_large); the small raw Unitsyn source
  (tasktrove-unitsyn) was deliberately NOT deprecated because its purported replacement had the
  wrong population (REPORT.md:1443-1501).
- Quarantine: the mislabeled small Unitsyn rebuilt release was quarantined — its 4,991 payloads
  identify `DCAgent__exp_rpt_unitsyn-python-large-v2@1` (population mismatch vs the 491-row
  true small source) (REPORT.md:1781). Lineage and successor identities: admission-ledger.md §6.

## 5. What is a census and what is not

- A complete, machine-readable (a)/(b)/(c) count over all 310–326 sources was **not** located
  as a single artifact in the audit directory; the census is distributed across `inventory.json`,
  `measurements/`, `pending-nine-measurements.json`, the report table, and later per-source
  review verdicts (adapter/backend/curriculum review JSONs such as
  `backend-68a-independent-review-20261114-1138.json`).
- Per-source review verdicts that do exist as discrete artifacts include: adapter-67 (v1/v2/v3),
  adapter-92 and adapter-e932 targeted reviews, backend-4a/68a/producer-04ed, curriculum-22a,
  crosspy-cc130, archive-census-b650, capacity-a93, builder-fd7e, calendar-8378, compiler-5b3a —
  all [R] under `.../atlas-env-audit/` with timestamps in their filenames.
- Unmeasured at the time of the inventory: 133 of ~326 sources had completed runtime
  measurements; `pending` and `not executable` rows were explicitly unknown (REPORT.md:1421).
  Later rounds (grading census #11355, admission strict census #11360, preview v9.3 inventory)
  bring the live 310-source population to full metadata coverage, but the explicit (a)/(b)/(c)
  label tally over the final population remains **unmeasured in a single place** — TODO for the
  logbook notebook to derive from `inventory.json` + the current registry declarations.

## 6. Explicit gaps

- No single total of "(a) sources: N, (b): M, (c): K" exists in the audited artifacts found.
- Snapshot/oversight columns in the inventory are per-source; a campaign-wide snapshot-count
  aggregate was not compiled in the sources read.
- The (b)-class QA completion status after the SAMPLE waves is tracked per source in the SAMPLE
  checkpoint PR bodies (see sample-panels evidence file), not folded back into the fidelity census.
- Per-source identities: the 16 retired populations (185,906 tasks), the 311→310 transition
  source, and the blends outside the 24 `missing:`-labeled rows — all unknown in the read set.
