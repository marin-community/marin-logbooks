# Evidence: Run manifests — Atlas campaign major runs and attempts

- **What this is**: one row/section per major run or attempt: lane, Iris job ID, command/entrypoint
  class, code pin, hardware, timing, outcome, cost if recorded, receipts — plus, new in this
  revision, the complete submitted argv/config where present locally (§11) and a scope-separated
  cost table (§12). Nothing redacted except secrets (none are present in these sources). Storage
  URIs and job IDs are metadata.
- **Compiled at**: 2026-11-15 (UTC); corrected by cold-read round 7.
- **Source cut**: recorded 13:51 UTC Nov 15 cut `96fc0736fc95d8f21d11e1c03dfca108d568b220`.
  **Evidence cutoff**: no fact in this file postdates that cut; the latest cited REPORT entry is
  the 13:15 UTC "Code14 complete" entry. (The file's REPORT read includes later entries, e.g.
  15:30/15:45 UTC; none of their facts are used here.)
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/active/atlas-full-qc/REPORT.md`,
  [R] `.../atlas-full-qc/*.csv|*.json` (gpu-attempt-times, repair-cache-progress, submissions),
  [R] `.../atlas-full-qc/*-submit.log`, [R] `.../atlas-oracle-filter-qa/FINAL_STATUS-v2.json`,
  [R] `.../atlas-oracle-filter-code/`, [R] `.../complete/atlas-env-audit/REPORT.md`,
  [R] `.../complete/atlas-contrast-audit/STATUS.md`, [R] `.../active/atlas-difficulty/`,
  [R] `.../active/atlas-puzzles-qa-repair/STATUS.md`,
  [P] marin issue https://github.com/marin-community/marin/issues/11490 (curriculum cost anchor).
- **Access class**: run metadata only; no task content, no secrets. Any secret-bearing argument
  would be omitted; none were found in the cited entries.

## 1. Reviewer serving fleet (FULL-QC content QC) — v2 and v3 distinguished

- Reviewer model: public immutable weights `zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42`
  (141 safetensor shards, 755,632,050,320 stored bytes; native FP8 E4M3, BF16 unquantized
  tensors) [R] REPORT.md:5.
- **Preserved v2 fleet**: serving image `vllm/vllm-openai@sha256:61fc8a896b0a…`; arm64 manifest
  `sha256:2a7cde230b59…`; placement TP4 × PP2 on two GB200 nodes (8 GPUs), **262,144-token
  serving context**; compilation + CUDA graphs on [R] REPORT.md:9, 1960 ("the unchanged
  262,144-token model configuration"), 1616 ("all existing FULL models use 262,144-token V2
  runtime").
- **v3 workers**: **524,288-token serving context**, 8,192 initial / 16,384 retry generation
  tokens, three attempts, unchanged GLM checkpoint/FP8/temperature 0.7/low reasoning
  effort/task-quality-v2 rubric/payload revision 4 [R] REPORT.md:1125, 2111; corroborated by the
  pass-3c serving metadata (max_model_len=524288) [R] claim-audit fullqc-pass3c.report.md.
- KV-block fix: vLLM 0.28 chose 32-token blocks on one host → `ValueError: No common block size
  for 32` during CUDA-graph profiling; PR #11300 pinned `--block-size 64`; merged 02:31:39 UTC
  2026-11-14 as `ee1d530741` [R] REPORT.md:468-482.
- Campaign roots: 48 CPU shard roots + 384 serving GPUs (v2r2); allocation estimate 612.780
  GPU-h at 08:05 UTC (2026-11-14); combined v1+v2 = 1,316.353 GPU-h at that estimate [R]
  REPORT.md:755, 834-842. (A separately scoped "conservative allocation" of 9,482.214080 GPU-h —
  sealed v1 703.573623 + v2 8,778.640457 — is recorded at REPORT.md:1862 with a different
  summation basis; the two are **not** summed here.)
- v1 root `/benfeuer/atlas-full-qc-content-q05-v1`: 8.656291 GPU-h, cancelled per coordinator
  [R] REPORT.md:54, 782.
- v2 content roots (examples): `/benfeuer/atlas-full-qc-content-q00-v3` 1,084.710922 GPU-h [R]
  REPORT.md:834; shard-022/027/030/034/036 inference attempts ended with natural primary
  completion (3.6–176.2 GPU-h each) [R] REPORT.md:66-103.
- Fleet telemetry (Finelog `telemetry_v1.vllm`, origin cw-us-east-08a, 2026-11-14
  16:52:01–17:09:34 UTC, 4,240 metric rows): mean 3.63 running requests/model; median generation
  rate 196.87 tok/s; summed model rates 9,637.89 tok/s; waiting mean 0.559; queue zero 95.50%
  of time. Gauges are interval-integrated without extrapolation [R] REPORT.md:1298-1300.
- **Reviewer identity per result, where locally bound**: QA nine panels — reviewer SHA
  `a3e17ee0…`, runtime `87f63bd2…` [R] FINAL_STATUS-v2.json; Code14 — per-release `SAMPLE_sha256`
  in the registration ledger, pinned V3 (524,288 context) [R]; Puzzles — pinned V3 (seed 0)
  [P] 4bdbc7eeed; CrossPy encoding-repair candidate panel — pinned **V2** identity [R]
  REPORT.md:2141. **Do not imply all archived panels used the current reviewer**: the archived
  245 include v2-era waves; the v2/v3 boundary per panel is not itemized locally.

## 2. CPU repair wave (incomplete-contract repairs, zero added GPUs)

- 15 active repair jobs at 12:20 UTC 2026-11-14: twelve newly-assembled TRAIN sources +
  code-oracle + Stack + Open Reasoning; 16 batch workers × 4 requests, 4 CPUs / 16 GiB /
  40 GiB, BATCH band 3 / 259,200,000 ms, cw-us-east-08a, frozen code
  `0e70c55a771d2b1404094f4253281ad4975bda89` [R] REPORT.md:2089.
- Total pending repair targets 476 across 12 sources (per-source table at REPORT.md:2080s).
- Repair runner fix: SAMPLE repair worker v1 crashed on deterministic
  StoragePath-in-deployment bug; fixed as `sample-repair-runner-v2.py` + regression test;
  relaunched `/benfeuer/atlas-full-qc-sample-v3-repair-v2` [R] REPORT.md:2094.
- Typed-binding factories: 13 zero-GPU CW02 verification jobs
  `atlas-full-qc-<name>-binding-direct-v1` succeeded (12:11 UTC submit 2026-11-14) [R] REPORT.md:2193.

## 3. OR (Open Reasoning) pass-3 campaign — see pass3c-failure.md for the failure record

- pass-3 runner family: `or-pass3-v3-runner-v1.py`, `or-pass3-v3b-runner-v1.py`,
  `or-pass3-v3c-runner-v1.py` (pre-fix SHA-256
  `abf36edfa1c2fcc16d49f5aa777aec2e537d6459aa5ec260a6e753b6e4ce0db1`), submitted via
  `submit-or-pass3-v3{,b,c}-v1.py` (script mtimes 02:44:08 / 03:07:07 / 03:23:16 UTC
  2026-11-15); job `/benfeuer/atlas-full-qc-repair3-openreasoning-v3pass-c-v1` submitted and
  accepted 03:44:32–03:44:36 UTC 2026-11-15 (4 CPU / 16 GiB / 40 GiB, batch band, controller
  127.0.0.1:10002) [R] submit log.
- Finalization was GPU-free from sealed receipts (`or-pass3-finalize-from-receipts-v1.py`,
  SHA-256 `2a2978ad2c2e32ae3d266290f0398eafc9089623bc2d302ec8204564a6978ea6`).

## 4. SAMPLE QC runs

- Nine-source QA SAMPLE worker: `/benfeuer/atlas-full-qc-sample-v3-repair-v2` — all 9 panels
  100/100 REVIEWED (33 rows repaired round 1) [R] REPORT.md:2094.
- Code14 worker: `/benfeuer/atlas-full-qc-sample-v3-code14-v1`, submitted 05:15:59 UTC
  2026-11-15 (submission JSON in §11), 14 sources / 1,400 tasks, queue-v14 SHA `49c66897…`,
  own 8×GB200 V3 serving; peak concurrent 72; total 61.138878 GPU-h (initial 38.123440 +
  cancelled general repair 16.688584 + priority 6.326853); completed per the 13:15 UTC entry
  [R] REPORT.md:2096, 2220.
- Puzzles SAMPLE attempt 4: FULL-QC worker `/benfeuer/atlas-full-qc-sample-v3-puzzles-v1`,
  **26-minute serving run**, 3.281682 GPU-h (8×GB200, created→finished); registered by public
  commit `4bdbc7eeed` [P]; the four-attempt table (wall window vs serving run) is in
  sample-panels.md §5.
- Five-source SAMPLE V3 worker: one five-source worker covering five 100-row seed-0 panels
  (500 tasks), queue `sample-qc/2026.11.14.2-final/queue-v1.json` SHA `1961d523…` [R]
  REPORT.md:2117.
- SAMPLE repair v1 job `/benfeuer/atlas-full-qc-sample-v3-repair-v1` (attempt CSV
  `gpu-attempt-times-sample-qc-20261115-final.csv`).

## 5. Oracle-filter and NoP control drives

- QA lane original-oracle runs: launch plans
  `2026.11.14.*-original-oracles-launch-plan-…json` with multi-source SHA-chained partitions;
  continuations `2026.11.14.{4..12}-original-oracles-continuation-plan-*.json` [R]
  atlas-oracle-filter-qa/. Final usage (scope-separated in §12): original_filter CPU
  28,833.046968 core-hours (allocation basis = requested × retained attempt wall time),
  SAMPLE_QC GPU 40.54752 GPU-h, SAMPLE_QC CPU parent+prepare 20.844402 core-h, judge tokens 0,
  Daytona 0 [R] FINAL_STATUS-v2.json.
- NoP (no-oracle positive/negative control) drives: NoP `28ffc24` reviewed with 24 root archive
  cases plus six production Trial cases PASS; manifest `06bbc4` SHA independently verified [R]
  atlas-env-audit REPORT.md:25. NoP controls commit `c8833deb68` (mandatory oracle log
  retrieval) [R] REPORT.md 15:10 entry. Positive Smartie NoP: new canonical `7569/8045` with
  400 NoP-positive drops, not independently adopted [R] REPORT.md:65.
- Code-lane v13 SWE reproduction shard `84284780`: 89,245 original tasks across 8 sources, 0
  admitted, full rechecks true; mechanical exclusions only [R] atlas-env-audit REPORT.md:65
  (the eight source IDs are itemized in snapshot-reconciliation.md §3).

## 6. Duplicate (NN overlap) review runs — Qwen3 pair survey

- v2: `/benfeuer/atlas-final326-duplicate-review-20261113-v2` failed after 46.5 s CPU —
  regional controller rejected the driver's added `cw-rno2a` self-target; no GPU use [R]
  REPORT.md:1049,1057.
- v3: accepted 04:36:48 UTC (2026-11-14) after 3 transient DNS retries; pinned Qwen revision,
  temperature 0.7, 65,536-token serving context, BATCH priority; final total
  **1.6893322222 H100-hours** (v3 0.7589955556 + v4 0.1651580556 + v5 0.7651786111
  controller-attempt hours) [R] REPORT.md:973,1017,1041.
- Outer-client deadline mismatch during batch preemption: v3 job hit an API read timeout after
  47 min 40.62 s root execution (fixed 180 s client timeout vs broker 1,920 s lease / 1,980 s
  proxy deadline / 2,400 s readiness window) [R] REPORT.md:1015.

## 7. Difficulty runs

- Hosted difficulty submit logs `atlas-difficulty-hosted-20261113-v1-submit{,-v2}.log`, health
  checks `…-v2-health-20261114T045716.log`/`T052106.log`, Linux CI diagnostic
  `atlas-difficulty-ci-linux-diagnostic-20261113-v1-submit.log` [R] `.../active/atlas-difficulty/`.
- Qualified difficulty cut: 1,240 records — 116 measured / 1 pending / 1,123 N0 [R]
  atlas-env-audit REPORT.md (v9.2 entry).

## 8. Curriculum experiment — public cost anchor (corrected; previously a TODO)

- [P] marin issue #11490 ("[Atlas] Compare selectable curriculum confidence levels") publicly
  reports the curriculum export run's cost: "**The four successful regional CPU jobs used
  46.892 requested CPU-hours (CPU request × task runtime, excluding queue time) and 1.591
  measured producer-process CPU-hours. No GPUs, new embeddings, or ranking computations were
  used.**" The five-level sample Parquet is 88.28 MB; its pre-aggregated confidence sidecar is
  13.52 MB. These figures belong to the **curriculum** experiment, not to the difficulty runs
  (an earlier draft placed them under difficulty with a TODO — removed).
- Curriculum archive: 326 sources / 4,392,782 retained; live-310 cut 34,461 bounded samples,
  204 TRAIN (3,848,908) + 106 EVAL (357,968); outcomes 23,956 assigned + 2,744,320 subject-only
  + 1,438,600 out-of-scope, zero insufficient/skipped [R] atlas-env-audit REPORT.md:388,412.

## 9. Contrast-audit captures (Atlas UI accessibility)

- Authored UI: 0 remaining violations across **3,542 state captures / 6,205,486 measurements**
  at 1440/390/320 px, light+dark. Baseline frozen revision-17 package + captured live API
  responses: **2,758,701 violation observations in 3,594 captures** (not unique defects).
  Coverage reconciles to 1,638 functional state categories (3,594 − 88 fewer probes + 36
  revalidation captures = 3,542) [R] `.../complete/atlas-contrast-audit/STATUS.md:3,15,51,64`.
- Public contrast: 1.17:1 dark → 13.77:1 light / 12.05:1 dark after fixes.

## 10. Puzzles serving

- Puzzles QA-repair lane: census milestones recorded in [R]
  `.../active/atlas-puzzles-qa-repair/STATUS.md` (6,921/6,921 literal extraction; 3,871 exact +
  3,050 math grader modes; dedup 6,926 raw → 6,921 unique, 5 within-source duplicates; converter
  landed `e3569506b8`, labels fix `44528e65ee`).

## 11. Manifest descriptions vs actual reproducible manifests

What is published here are **manifest descriptions** (submitted argv/config recorded in local
logs). The **actual reproducible manifests** — the full frozen queue/plan JSONs and their
immutable receipt objects — live at the storage URIs below and require private access [M].

- **OR pass-3c job (submitted argv/config, 2026-11-15 03:44 UTC)** [R]
  `repair3-openreasoning-v3pass-c-submit.log`: `bash -lc uv run --no-sync --package marin-core
  --extra dedup python -c '<246-line runner source>'`; Resources `cpu=4, memory=16G, disk=40G`;
  priority band `batch`; controller `http://127.0.0.1:10002`; runner constants (storage URIs +
  SHA-256 digests, no secrets): primary/pass-2a/pass-2b artifact SHAs, output root
  `…/repair-overlay/2026.11.14.1/96fc19f9b4b2e033c9b3-pass3v3c`, request cache
  `…/repair-request-cache/2026.11.14.1-v3c`, BATCH_WORKERS=16, serving root
  `/benfeuer/atlas-full-qc-recovery-serving-v3-01`. The runner source itself is published in
  pass3c-failure.md §2 (pre-fix; code only).
- **Code14 worker (complete submission config, 2026-11-15 05:15:59 UTC)** [R]
  `sample-v3-code14-submission-v1.json`: job `/benfeuer/atlas-full-qc-sample-v3-code14-v1`;
  queue `s3://marin-us-east-02a/marin/rl-data/atlas/sample-qc/2026.11.15.1-code-final/queue-v14-code.json`
  SHA `49c66897e23f92f3127e70901ca688c871c7d6df95e38d47907442ca29a06ae4`, 14 sources / 1,400
  sampled tasks; output root `s3://…/sample-qc/2026.11.15.1-code-final`; review cache
  `s3://…/sample-qc/request-cache-v3-2026.11.15.1-code`; reviewer "pinned V3 profile (524288
  context), own serving via the reviewed literal endpoint"; frozen head
  `87f63bd298ce9b312f0d5571495f683f8f5f2a91`; target cluster `cw-us-east-08a`; priority
  `batch`; timeout 259,200 s.
- **Puzzles attempt 4 worker invocation (described, 2026-11-15)** [R] puzzles STATUS.md: FULL-QC
  worker built with "exactly my invocation" — puzzles output root + shared review-cache, pinned
  V3 reviewer, own 8-GB200 serving; frozen queue sha `02ceafdb…`. The literal argv is not
  separately archived locally beyond this description.
- **CPU repair wave**: per-source submission receipts `repair-<name>-submission-v1.json`, freeze
  logs `freeze-<name>-v1.log`, submit logs `submit-<name>-v1.log` (argv + resource line + job ID
  + dashboard URL) [R] REPORT.md:2090; frozen repair plans (examples): openreasoning-v1.json
  SHA `d5facde6…`, openswe-v1.json SHA `268079ec…`, knowledge-v1.json SHA `f2c82024…`,
  code-oracle-v1.json SHA `38c011d6…` under `s3://…/full-qc/repair-plans/2026.11.13.1/` [M].
- **What requires private inputs** [M]: the frozen queue/plan JSON bodies, the S3 receipt
  objects (`s3://marin-us-east-02a/…`), and the private registry/HF release objects. The
  argv/config *descriptions* above are complete as recorded locally and contain no secrets; no
  sanitization was needed beyond that nothing secret-bearing was present.

## 12. Cost table — scopes separated; no overlapping sums

| Scope | Figures (units as recorded) | Basis / caveats | Source |
| --- | --- | --- | --- |
| FULL-QC content fleet (v2-era estimate) | 612.780 GPU-h allocation estimate at 08:05 UTC 2026-11-14; "combined v1+v2 = 1,316.353 GPU-h" | point-in-time allocation estimate; a separately scoped "conservative allocation" of 9,482.214080 GPU-h exists (REPORT.md:1862) with a different basis — the two are NOT summed | [R] REPORT.md:755, 834-842, 1862 |
| FULL-QC v1 root | 8.656291 GPU-h (cancelled) | single root | [R] REPORT.md:54, 782 |
| FULL-QC v2 example roots | q00-v3 1,084.710922 GPU-h; shards 022/027/030/034/036 3.6–176.2 GPU-h each | per-root | [R] REPORT.md:66-103, 834 |
| OR pass-3c attempts (inference) | **unknown** — the pass-3c serving root's GPU-hours are not attributed to the job in the read set (job-scoped telemetry exists in the claim audit; no GPU-h figure) | two failed attempts + attempt 2 cancelled | [R] claim audit |
| Probe A (OR truncation probe) | 16.849362 GPU-h | inference-only, created→finished | [R] REPORT.md:2252 |
| Code14 SAMPLE worker | 61.138878 GPU-h total = initial 38.123440 + cancelled general repair 16.688584 + priority 6.326853 (components sum to the total) | created→finished | [R] REPORT.md:2220 |
| QA-lane original-oracle filter | 28,833.046968 **CPU** core-hours | requested × retained attempt wall time | [R] FINAL_STATUS-v2.json |
| QA-lane SAMPLE QC | 40.54752 GPU-h; CPU parent+prepare 20.844402 core-h; judge tokens 0; Daytona 0 | — | [R] FINAL_STATUS-v2.json |
| Puzzles SAMPLE attempt 4 | 3.281682 GPU-h | 8×GB200, 26-min serving run | [R] REPORT.md:2220 |
| Duplicate review (Qwen pair survey) | 1.6893322222 **H100-hours** total = v3 0.7589955556 + v4 0.1651580556 + v5 0.7651786111 | the components are labeled "controller-attempt hours" in the source; whether each attempt hour maps 1:1 to GPU-device-hours or requires a serving-width multiplier (e.g. ×N GPUs) is **not stated in the read artifacts** — the sum is presented in the source's own unit with no multiplier applied | [R] REPORT.md:973, 1017, 1041 |
| Curriculum export | 46.892 requested CPU-hours + 1.591 measured producer-process CPU-hours; four successful regional CPU jobs; zero GPUs | requested = CPU request × task runtime, excluding queue time [P #11490] | [P] marin issue #11490 |
| Difficulty runs | per-run costs not recorded in the difficulty artifacts read; the 46.892/1.591 figures are NOT difficulty costs (see curriculum row) | — | [R] |

## 13. Receipts convention

Most jobs leave local evidence: `*-submit.log` (argv, resource line, job ID, dashboard URL),
`repair-*-submission-v1.json`, `gpu-attempt-times-*.csv` (attempt timestamps in ms epoch),
`gpu-hours-*.json`, and S3 receipts under `s3://marin-us-east-02a/marin/rl-data/atlas/...`
(all publishable as metadata; object contents require private access [M]).
