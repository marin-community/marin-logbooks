# Evidence: Atlas environment-fidelity classification census (a)/(b)/(c)

- **What this is**: compiled record of the environment-fidelity classification census for the
  Atlas logbook campaign. Definitions come from the user's BRIEF addendum; the per-source
  measured inventory and disposition are compiled from the audit report and its JSON artifacts.
- **Compiled at**: 2026-11-15 (UTC, from file timestamps; exact compile time is this build)
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/complete/atlas-env-audit/BRIEF.md`
  (note: the brief lives under `experiments/complete/`, not `active/` as first referenced),
  [R] `.../atlas-env-audit/REPORT.md` (497 KB, 2,203 lines),
  [R] `.../atlas-env-audit/inventory.json` + `measurements/` (referenced by report; not exhaustively re-parsed here),
  [R] `.../atlas-env-audit/pending-nine-measurements.json`,
  [R] `.../atlas-env-audit/excluded-fourteen.json`,
  [P] definitions mirrored in marin docs `docs/references/task-curation.md` (branch `atlas-env-audit`).
- **Access class**: metadata/counts only — task text, prompts and model outputs are never reproduced here.

## 1. Classification definitions (user ruling, BRIEF.md:54-63)

- **(a)** spec fully describes runtime behavior AND SAMPLE/FULL QA evidence exists → fine; record evidence.
- **(b)** spec fully describes behavior but QA is missing → needs SAMPLE or FULL QA before it stays runnable.
- **(c)** runtime behavior differs from spec (substituted image, missing environment resolved to host, or a declared environment ignored) → needs repair.
- Ingestion-time rewrites (lowering, reply-only rewrites) are acceptable as **(a)** when the published spec is the whole truth and robust QA passed.
- **28 grandfathered MarinSkyRL blends** (GRANDFATHERED_MARINSKYRL_SOURCES) are explicitly out of scope
  (Addendum 2, 10:20 UTC): not audited, not repaired; left exactly as-is. Source: [R] BRIEF.md:65-67.

## 2. Denominators actually measured (REPORT.md:1421, inventory table at REPORT.md:1419-1702)

| Population | Count | Source |
| --- | ---: | --- |
| Runnable agentic sources at audit start (`runnable_sources()`) | 81 | BRIEF.md:29 |
| Live Atlas sources | 311 | REPORT.md:1421 (SQL snapshot `8674227db4ca…`, loaded 2026-11-11 04:12 UTC) |
| Union inventory (branch ∪ live, unique IDs) | 326–328 (326 unique IDs; 328 rows with later aliases) | REPORT.md:1421, 1506 |
| Completed runtime measurements in the first inventory pass | 133 | REPORT.md:1421 |
| `pending` and `not executable` rows | explicitly unknown, not zero | REPORT.md:1421 |

## 3. Environment result classes in the measured inventory (REPORT.md table, lines 1425-1702)

Per-source environment results recorded in the published table (each row carries its measured
row count, environment result, snapshot count and oversight level). Class-label tallies from the
66 rows carrying a `faithful:/substituted:/missing:/reply_only_rewrite:` result label:

| Environment result label | Rows in table | Examples (counts per source) |
| --- | ---: | --- |
| `faithful:` (own spec environment; subset also faithful after drops) | 17 | tasktrove-code_contests 8,728→8,222; taco 10,000→5,116; stack_pytest 500→492; unitsyn 491; curriculum_easy 509→505 |
| `substituted:` | 2 | tasktrove-arc_inductive 10,000; tasktrove-arc_transductive 10,000 |
| `reply_only_rewrite:` (spec-accurate after rewrite, QA pending at that time) | 27 | ifeval 46,391→46,242; knowledge_mcqa 616,888→613,002; science-openqa 150,644; safety 89,066 |
| `missing:` (grandfathered mopd blends, out of scope) | 15 | all `nemotron_ultra_*` rows, unknown snapshots |
| `docker_image:` (rebuilt/private-qualified cohorts) | 14 | calibforge 1,357; codeforces 9,745; competitive-coding 13,973 |
| `native task.toml` | 7 | swesmith-verifier-repaired-v5 7,016 (5 snapshots, FULL oversight); DCAgent excluded cohorts |
| `not_executable_inventory` | many | evalchemy-*, GAIR OpenSWE, SWE-Gym, XiaomiMiMo pools, benchmarks (aime24, apps, …) |

## 4. Dispositions produced by the campaign (REPORT.md 1830-1901 and cross-references)

- 14 excluded cohorts requalified: `excluded-fourteen.json` [R] lists them with pinned row counts
  totaling 53,842 (calibforge 1,357, nl2bash 1,498, codeforces 9,745, competitive-coding 13,973,
  curriculum-easy 505, curriculum-medium 488, e2egit 477, e2egit-large 4,993, multifile 4,842,
  pymethods2test 500, pymethods2test-large 4,990, stack-pytest 492, unitsyn-python 4,991,
  unitsyn-python-large 4,991). CodeContests (8,222) and TACO (5,116) requalified separately via
  #11237; total requalification queue 16 cohorts / 67,180 pinned rows (REPORT.md:1813, 1838).
- Nine pending measurement categories (ten source rows; ARC has two variants): all measured
  without provider calls; per-source pins, converter versions and exclusion ledgers in
  `pending-nine-measurements.json` [R] (e.g. puzzles 6,926 input→6,926 converted, 0 rejections;
  knowledge_mcqa 616,888→613,002 with typed rejections invalid_reference 3,824,
  reviewed_defect 1, unsupported_answer_contract 61; ARC-inductive 10,000 excluded entirely as
  `source_defect:grading_environment_mismatch`, Daytona unsafe reason `host_image_required`).
- Repaired/unrepaired: repaired = sources whose (c) findings were fixed via true-environment
  builds or exact lowerings, then qualified with the full-population oracle filter and admitted
  (see admission-ledger evidence file). Unrepaired/out-of-scope = the 28 grandfathered blends and
  the ARC-inductive source (100% typed exclusion, source defect).
- Deprecated legacy sources with admitted faithful replacements: 14 deprecated
  (tasktrove-code_contests, -codeforces, -competitive_coding, -curriculum_easy, -curriculum_medium,
  -e2egit, -e2egit_large, -multifile, -nl2bash, -pymethods, -pymethods_large, -stack_pytest,
  -taco, -unitsyn_large); the small raw Unitsyn source (tasktrove-unitsyn) was deliberately NOT
  deprecated because its purported replacement had the wrong population (REPORT.md:1443-1501,
  REPORT.md deprecation note).
- Quarantine: the mislabeled small Unitsyn rebuilt release was quarantined — its 4,991 payloads
  identify `DCAgent__exp_rpt_unitsyn-python-large-v2@1` (population mismatch vs the 491-row true
  small source) (REPORT.md "Quarantined the mislabeled Unitsyn small release").

## 5. What is a census and what is not

- A complete, machine-readable (a)/(b)/(c) count over all 310–326 sources was **not** located as a
  single artifact in the audit directory; the census is distributed across `inventory.json`,
  `measurements/`, `pending-nine-measurements.json`, the report table, and later per-source
  review verdicts (adapter/backend/curriculum review JSONs such as `backend-68a-independent-review-20261114-1138.json`).
- Per-source review verdicts that do exist as discrete artifacts include: adapter-67 (v1/v2/v3),
  adapter-92 and adapter-e932 targeted reviews, backend-4a/68a/producer-04ed, curriculum-22a,
  crosspy-cc130, archive-census-b650, capacity-a93, builder-fd7e, calendar-8378, compiler-5b3a —
  all [R] under `.../atlas-env-audit/` with timestamps in their filenames.
- Unmeasured at the time of the inventory: 133 of ~326 sources had completed runtime measurements;
  `pending` and `not executable` rows were explicitly unknown (REPORT.md:1421). Later rounds
  (grading census #11355, admission strict census #11360, preview v9.3 inventory) bring the live
  310-source population to full metadata coverage, but the explicit (a)/(b)/(c) label tally over
  the final population remains **unmeasured in a single place** — TODO for the logbook notebook
  to derive from `inventory.json` + the current registry declarations.

## 6. Explicit gaps

- No single total of "(a) sources: N, (b): M, (c): K" exists in the audited artifacts found.
- Snapshot/oversight columns in the inventory are per-source; a campaign-wide snapshot-count
  aggregate was not compiled in the sources read.
- The (b)-class QA completion status after the SAMPLE waves is tracked per source in the SAMPLE
  checkpoint PR bodies (see sample-panels evidence file), not folded back into the fidelity census.
