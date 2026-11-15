---
title: "Snowball IsoFLOP ablations (archived campaign)"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/54"
related_issues:
  - "https://github.com/marin-community/marin-logbooks/issues/49"
research_branch: "codex/54-snowball-isoflop-ablations"
marin_base_revision: "6e09d9043eb71d7bd487b5e3d79e275b95c454fd"
archive_template_revision: "3744203cdaa19ab4b4943e9836e143e7a4c339c4"
status: archived
---

# Snowball IsoFLOP ablations (archived campaign)

## Question and decision criteria

- Goal and why it matters: small-scale IsoFLOP/ablation cells on a separate 64-GPU interactive allocation, run to answer long-loop/SFT-mix questions (optimizer choice, layout selection, loss masks) cheaply before touching the 67B campaign. This record is the campaign's **archive**: what ran, the results, the audit-corrected stall/guard history, the known misconfigurations, why it was stopped, and what may NOT be concluded from it.
- Baseline and alternatives: archived campaign, stopped early by the user on 2026-11-15 ("misconfigured at the start and aren't telling us much") — 866.32 H100-hours spent, nothing deleted, endpoints preserved.
- Evidence needed to choose: n/a (archive). Every claim carries evidence (job IDs, commit/branch SHAs, receipt URIs/SHAs, file paths [R]); unverified items labeled.
- Constraints and unresolved questions: public record; no secrets; no data (task text/prompts/outputs/rows); no personal information (third parties written as "another user").

## Baseline

- Date: 2026-11-15 (archive compiled)
- Published source revision: experiment files on branch `snowball-isoflop-ablations` (marin repo); FINAL.md under /experiments/complete/snowball-isoflop-ablations/ [R] (dir moved from active/ to complete/ — older paths in FINAL.md/RECOMMENDATIONS cite active/).
- Configuration, command, and external inputs: see entry.
- Environment and hardware when relevant: separate 64-GPU interactive allocation (outside the long-loop ≤256-H100 ceiling); cluster cw-rno2a.
- Result and evidence: see entry.
- Limitations: campaign misconfigured per the user; the "not valid to conclude" list below.

## Entries

### 2026-11-15 16:16 UTC - Campaign archive record

- Question or hypothesis: Do assistant-only loss masks, per-size batch/parallel-layout (sonic vs ring_gather_combine) geometry, and optimizer/LR choice (Muon-H vs adam) transfer from small pretrained Snowflake bases (d512/d768/d1024, 262,144-token SFT) to inform the 67B long-loop SFT/RL system?
- Published source revision: Configuration `2026.11.13.10`; final commit on branch `snowball-isoflop-ablations` = `94d180c765`. Landing-branch commits pushed during the campaign (no PRs, per directive): `f3f580abc9`, `abbb79bdeb`, `3e3fe29440`, `1d5bd7b971`, `3461eb0e97`, `59ffe6f503`, `e7ef25b469`, `d8387467ec`, `f103997ccd`, `f9a12ef08c`, `94d180c765`. Stop-order code: `fbc123c0bd` (cross-attempt trial reuse), `e7ef25b469` (attempt-scoped trial identities).
- Command or material configuration: Manually submitted Iris roots (no auto-launcher; every root submitted by the agent) on cluster cw-rno2a, one H100x8 node per child, interactive priority, 259,200 s timeouts. Phase A: `continuation.py` / `run.py --phase A`; B/C: `run_probes.py --phase B / C-throughput / C-loss`; D: `optimizers.py --phase D` (3 sizes × 2 optimizers × 3 LR candidates, equal 3e18 reference-FLOP tuning budgets). Budget guard `guard.py`: `charged_h100_hours + active_gpu_reservation/60 >= ceiling_h100_hours` (one-minute allowance = 0.4 H100-h at 24 GPUs). Watchdog: 1800 s "no progress after process_started" in D children; stall classification from terminal error text per `f9a12ef08c` (regression test `test_failure_classification_reads_aggregate_status_without_task_detail`); D-only stall-retry policy (`-retry1`/`-retry2`, exclusion records, D result persistence) per `94d180c765`. Q5 (`optimizers.py --phase Q5`) never launched.
- External inputs and artifact references: S3 prefix `S/` = `s3://marin-us-east-08a/marin/users/benfeuer/snowball-isoflop/2026.11.13.10/` (Filer-browsable; reads need Marin credentials). Audits: `/experiments/active/claim-audits/{snowball-d-kills,isoflop-recs,isoflop-recs-v2}.report.md`. Public stall issue: marin#11440. Recommendations rev 3 SHA-256 `9fb091eb5829f08faa4c414aa6accc58abf9fe66ee47a50c91c26272ecb1857e`. Receipts/guard snapshots: `/experiments/active/snowball-isoflop-ablations/artifacts/pretrain-baseline-revision2/native-cpu-v10/resume-20261114/`.
- Environment and hardware when relevant: cw-rno2a 8×H100 nodes; BF16 compute/BF16 output/FP32 optimizer master, `gpu_fa4_cute` attention, sonic EP1 baseline dispatch, optimizer offload on, jax/jaxlib/jax-cuda13-plugin 0.11.1 (per marin#11440).
- Result, including failure or restart: Campaign terminated early on 2026-11-15 by user decision — quoted in FINAL.md: "stop the Snowball isoflop ablations EARLY. They were misconfigured at the start and aren't telling us much." The user-cited misconfiguration is the campaign scope itself (FINAL.md §4.1): three small from-scratch models at 3e18–3e19 budgets with no measured connection to the 67B long-loop SFT/RL target. Total observed spend **866.32 H100-h** (guard `S/campaign-budget-guard-allocated/latest.json`): A 544.89 + B 107.47 + C 63.32 + D 150.64 + Q5 0. All artifacts retained; nothing deleted. Chronology below.
- Evidence: cited per item below (job IDs, commits, S3 URIs, file paths).
- Interpretation and limitations: see "NOT valid to conclude" and Limitations.
- Decision and next action: Archive the campaign record; apply rev-3 recommendations only as scoped/hypothesis-labeled; do not reuse numbers for the 67B system without new measurements there.

## Chronological archive

**Phase A — full grid, 3 sizes × 2 masks × 3 seeds: 18/18 cells SUCCEEDED with permanent endpoints.** Roots `A-d{512,768,d1024}-...-r2`…`r14` (15 roots across resizes; per-size lineage: d512/d768 roots r4, r7, r10, r13; d1024 roots r10, r13, r14). Endpoints (all `is_temporary=false`, verified by credentialed re-audit; zero-based step indices):

| Cell | Permanent endpoint |
|---|---|
| d512 full/assistant seed0 (3e19) | `S/A-host512-r4-d512-{full,assistant}-seed0-n3766/checkpoints/step-3765/` |
| d512 full seed1, seed2 (1e19) | `S/A-host512-r7-d512-full-seed{1,2}-n1255/checkpoints/step-1254/` |
| d512 assistant seed1, seed2 (1e19) | `S/A-host512-r13-d512-assistant-seed{1,2}-n1255/checkpoints/step-1254/` |
| d768 full/assistant seed0 (3e19) | `S/A-host512-r4-d768-{full,assistant}-seed0-n2426/checkpoints/step-2425/` |
| d768 full seed1 (1e19) | `S/A-host512-r7-d768-full-seed1-n808/checkpoints/step-807/` |
| d768 full seed2 (1e19) | `S/A-host512-r10-d768-full-seed2-n808/checkpoints/step-807/` |
| d768 assistant seed1, seed2 (1e19) | `S/A-host512-r13-d768-assistant-seed{1,2}-n808/checkpoints/step-807/` |
| d1024 full seed0 (3e19-class, n1241) | `S/A-host768-r10-d1024-full-seed0-n1241/checkpoints/step-1240/` |
| d1024 full seed1, seed2 (1e19) | `S/A-host768-r{10,13}-d1024-full-seed{1,2}-n413/checkpoints/step-412/` |
| d1024 assistant seed0 (3e19-class, n1241) | `S/A-host768-r14-d1024-assistant-seed0-n1241/checkpoints/step-1240/` |
| d1024 assistant seed1, seed2 (1e19) | `S/A-host768-r14-d1024-assistant-seed{1,2}-n413/checkpoints/step-412/` |

Extra retained d1024 seed0 full endpoints: 3e18 at `S/A-host768-r3-d1024-full-seed0-n1241/checkpoints/step-123/`, 1e19 at `.../step-412/`. Training completion ≠ quality comparison: no score phase ever ran.

**Phase B — batch saturation search: selections complete.** Roots `B-20261113-v10` (killed), `B-cp8-host512-r2/r3/r4`, `B-small`, `B-ladder-r5..r10`, `B-d1024-ring-host768-r3`, `B-d1024-r11..r14`. Result files `S/manager-B/completed-{512,768,1024}.json` (reconciled, additive to the killed root's ledger; credentialed re-audit verified). Selected layouts (context 8, data 1, offload on):

| Size | Layout | Microbatch × accum | Positions/s/GPU | Min headroom |
|---|---|---|---|---|
| 512 | sonic | 1 × 128 | 104,017.5 | 0.652 |
| 768 | sonic | 8 × 8 | 75,952.9 | 0.338 |
| 1024 | ring_gather_combine | 4 × 8 | 40,090.9 | 0.328 |

Known B-phase defects: B-r4 runner omitted explicit layouts so probes defaulted to CP4/data2 geometry, which stalls for these sizes (agent error; both mb1 anchors burned stall cycles). Stall map (all watchdog exit 124): d512 sonic mb2 `92fd78ebb829` (B-r3); d1024 ring mb1/acc16 `d051dd905174` (B-r5) and mb2/acc16 `9247ac23e8ba` (B-r7); CP4/data2 mb1/acc8 d512 `4fe9b1b7d741` (KILLED) and d768 `015e1d929baf` (FAILED/124); d768 sonic mb16 recorded negative headroom −0.0257 (`B-host512-r10-...-mb16-acc8-...-d768-...-n8/probe-result.json`, memory-limit class).

**Phase C — throughput + loss at selected batch: complete.** Roots `C-throughput-r4` (killed), `C-throughput-r5` (killed mid-offload0), `C-loss-r1`. Files `S/manager-C-throughput/completed-{512,768,1024}.json`, `S/manager-C-loss/completed-{512,768,1024}.json` (re-audit verified; zero dispatch drops everywhere). Matched-batch throughput (positions/s/GPU): d512 sonic 98,262.8 vs ring 94,133.6; d768 sonic 75,885.8 vs ring 76,990.4; d1024 ring baseline 39,903.7. Equal-3e18 final `train/loss`: d512 sonic 1.747133 vs ring 1.747180; d768 sonic 1.430427 vs ring 1.430483; d1024 ring 0.953114. Selections (lower final loss): sonic for d512/d768, ring for d1024 (`manager-C-loss/selection.json`). Offload-0 finalists were killed by the agent's own tree cancellation and excluded under the degenerate-finalist rule — the layout comparison is incomplete by design.

**Phase D — optimizer/LR sweep, 18 trials: 17 of 18 completed; no d1024 winner.** Roots `D-r1`…`D-r7` (assistant-only). Result files `S/manager-D/completed-512.json` and `completed-768.json` complete with winners; `completed-1024.json` never written (stopped at 5 of 6 d1024 trials).

- d512 winners (no exclusions, `degenerate_lr_tuning` absent): MuonH lr 2.536339912212702e-04 → final `eval/loss` 1.413755; adam lr 5.853092105106236e-05 → 1.300438. **adam wins by 0.113317.**
- d768 winners (no exclusions): MuonH lr 1.5640155002547376e-04 → 1.216624; adam lr 3.609266539049395e-05 → 1.166501. **adam wins by 0.05012369** (full precision).
- d1024 (5/6 trials): MuonH center/3 1.66667e-05, center 5e-05, upper 1.04957e-04; adam center/3 3.84615e-06, center 1.15385e-05 (adam center completed on `-retry1` after a stall in D-r7 — the retry policy worked once, live). The adam upper trial (lr 2.42209e-05, child `59c010b63b88`) was killed by the stop order. Per-trial losses live in `S/D-r7-*` and earlier `S/D-*` probe-result.json files; W&B rounded summaries corroborate (d1024 muonh upper r4: 1.0135).

**Audit-corrected stall/guard history (marin#11440-era + D-phase).** Primary audit: `claim-audits/snowball-d-kills.report.md` (23 children across D-r2..r5; all attempts attempt 0; zero PREEMPTED attempts).

1. The "preemption wave" narrative was WRONG. D-r2 (root death 08:38:25.483 UTC), D-r3 (10:07:49.618) and D-r5 (12:23:20.749) self-cancelled via the `budgeted_map` handler after a d1024 child hit the 1800 s watchdog (exit 124 on all ranks, no OOM; failed children `7e51038ed5b9`, `ae246dc12b8d`, `85da4fc5abad`). D-r4 was cancelled by the budget guard at the provisional 96 H100-h ceiling (11:20:20.352 UTC; guard log "Stopped registered root /benfeuer/snowball-isoflop-D-r4-v10 at phase D limit" at 11:20:21.325). Re-audit recovered the pre-cancel snapshot: observed 95.77756444507175 + 24/60 = 96.1776 ≥ 96.0 ceiling; terminal reconstruction 95.802171 (D-r1 17.594962 + D-r2 28.392260 + D-r3 29.115416 + D-r4 20.699533). `preemptions=1` on job status is a counter artifact (any KILLED started attempt counts; counter code path in the controller), and "Terminated by user" is the generic TerminateJob reason. Real Kueue evictions in the same window belonged to other users' jobs, not this tree.
2. d1024 startup stalls are real and d1024-specific but not deterministic: in D-r2..r5, 3 FAILED/124, 3 SUCCEEDED, 1 KILLED across seven attempts (3/6 success-failure outcomes; 3/7 with the killed attempt). Log evidence (r3 child `ae246dc12b8d`): eval gate passed, data loaded, HLO lowering 28–31 s, then 17 data-loader warnings (09:46:44.879–09:47:28.476 UTC) until the watchdog at 10:06:33.950; the successful r4 sibling covered lowering→first step in 46.583 s. Hang sits after HLO lowering and before the first update. Cause unverified (hypothesis: collective first-step setup). marin#11440 documents the first-update stall controls: command buffers off + `CUDA_LAUNCH_BLOCKING=1` (+ async-collectives disable for some geometries) completed 20-update calibrations as scoped workarounds, but a 64-H100 67B reproduction (comment by another contributor) did not qualify the workaround at full-model geometry.
3. Code fixes during the campaign (each with tests): cross-attempt trial reuse `fbc123c0bd`; attempt-scoped trial identities `e7ef25b469`; stall classification from job status `f103997ccd`, corrected to terminal-error-text matching `f9a12ef08c` (GetJobStatus carries no task/attempt detail — empty `tasks[*].attempts[*].exit_code` caused D-r5 to self-cancel on a stall the code was meant to absorb); stall-retry policy + D result persistence `94d180c765`.
4. Budget-guard ceiling raises, each reported with observed spend: D 96 → 143.802171 (at observed 95.802171) → 157.419767 (at observed 133.419767) → 172.458971 (at observed 148.458971). Final live guard (config LastModified 2026-11-15 14:26:49 UTC): A 2200 (reserved 172), B 119.135304, C 131.685816, D 157.419767, Q5 1152.

**Known misconfigurations and defects** (FINAL.md §4): (1) campaign scope as above; (2) B-r4 default-geometry regression; (3) in-cluster log blindness — the regional log server serves nothing for terminal child jobs, so server-side classification saw empty input (fixed by `f9a12ef08c`/`f103997ccd`); (4) aggregate GetJobStatus carries no task/attempt detail; (5) D results not persisted before `94d180c765` (a tree loss would have discarded winners); (6) provisional phase ceilings set by the agent caused the one D-r4 guard stop; (7) manual reconciliation defects caught before damage (B reconciliation files missing the `["result"]` wrapper crashed C-r4 at admission; a d1024 endpoint initially cited under the wrong root phase, corrected to `A-host768-r10` after an existence check).

## Current state

No Q5 launch happened; no auto-launcher exists; after the final cancel no Snowball root is running or pending on the controller. Campaign directory moved to `/experiments/complete/snowball-isoflop-ablations/` (FINAL.md and RECOMMENDATIONS still cite the former `active/` paths — those paths are stale). All S3 artifacts retained.

## Open items (from RECOMMENDATIONS-next-long-loop.md rev 3)

- Keep both optimizers in any future sweep for d512/d1024; d768 adam winner is verified but subject to a Q5 check. PROVISIONAL.
- Batch/layout geometry (B table above) validated only for the tested optimizer/mask/data settings; any new setup needs a fresh throughput check. PROVISIONAL.
- Keep the three sizes and the 1e19/3e19 endpoint ladder; score phase (`scoring.py` over endpoint manifests) still needed for size/mask comparisons. PROVISIONAL.
- Apply the D-only stall-retry policy and terminal-error-text stall classification to future D sweeps; success rate unmeasured (one live success). PROVISIONAL.
- Keep the budget guard for every phase; set ceilings from reconstructed terminal ledger spend before each phase; never raise a ceiling without stating observed spend. PROVISIONAL as practice.
- Do-not-do list (each PROVISIONAL, scoped to exact tested configs): d512 sonic mb2/acc8; d1024 ring mb1/acc16 (batch 16) and mb2/acc16 (batch 32); d768 sonic mb16/acc8 without a new memory check; CP4/data2 mb1/acc8 for d512/d768 without a new stall check; never discard a geometry after one stall (3 successes coexist with 3 stalls in the same ring mb4/acc8 family).
- Q5 paired optimizer repeats at 1e19 and the score phase are the two missing campaign components; both would be needed for any size/mask/optimizer conclusion.

## NOT valid to conclude

1. No d1024 optimizer or LR winner exists (5/6 trials; adam upper never measured; `completed-1024.json` absent).
2. No mask comparison — both masks trained at every size, but no score-phase artifact compares them.
3. No cross-size loss comparison — score phase never ran; 1e19/3e19 ladder points are design choices, not measured optima.
4. No transfer to the 67B long-loop system — all measurements are small from-scratch models; every transfer statement is a hypothesis. marin#11440's 64-H100 comment independently shows the 8-GPU workaround did not qualify at full-model geometry.
5. The d1024 stall cause is unverified — hang location evidenced (post-lowering, pre-first-update), mechanism not; "not a code bug / not infrastructure" was never established.
6. The retry policy's success rate is not measured (n=1 live success).
7. W&B per-rank loss values are rounded corroborations only; exact values live in the credentialed-only S3 result files.

## Limitations

- This record summarizes audits' verified claims; the audits could not execute tests, verify strict historical absence of deleted objects, or prove universal artifact absence outside the inspected namespace.
- The 3e18/3e19 budget labels are target ceilings (`at_budget` floors to whole updates), not exact realized FLOPs; "equal budget" comparisons mean equal target.
- Checkpoint metadata carries step/timestamp/permanence only — no tensor-level integrity audit was performed.
- The budget-guard numbers describe this campaign's reservations; do not reuse the ceilings for a different reservation width.
- IsoFLOP trends here are bounded by two size points each; no curve projection outside d512–d1024 is supported.
- marin#11440 is still OPEN and predates the D-phase history in this record; its comments cover the calibration/B-era stalls only.

### 2026-11-15 16:55 UTC - Revision 1 after cold-read round 1: definitions, initialization correction, phase map, trial tables, budget history, privacy redactions

- Question or hypothesis and reason for trying it: cold-read round 1 returned NOTES (32 items). This entry corrects and completes the archive record from local sources (FINAL.md, RECOMMENDATIONS-next-long-loop.md rev 3, PLAN.md, the three claim-audit reports) and the pinned public code on branch `snowball-isoflop-ablations`. The 16:16 archive entry above is preserved unedited except for privacy redactions (below); every correction is stated here.
- Published source revision: experiment files pinned at https://github.com/marin-community/marin/tree/94d180c765ee503a8b6d947a2a945c6055bbeffc (branch `snowball-isoflop-ablations`); training identities unchanged.
- Command or material configuration: definitions below verified against the pinned code (config.py, probes.py, measurements.py, run_probes.py, optimizers.py at the pin above); numbers cross-checked against the audits.
- External inputs and artifact references: FINAL.md [R]; RECOMMENDATIONS-next-long-loop.md rev 3 [R]; claim-audits/{snowball-d-kills,isoflop-recs,isoflop-recs-v2}.report.md [R]; marin#11440 (https://github.com/marin-community/marin/issues/11440, incl. the [64-H100 comment](https://github.com/marin-community/marin/issues/11440#issuecomment-6873648985)) [P]; recipe `experiments/post_training/snowball_isoflop/inputs/exp1-recipe.json.gz` on the pinned branch [P].
- Environment and hardware when relevant: unchanged.
- Result, including failure or restart:

**Privacy redactions applied in place (16:16 entry and Current state)**

1. Absolute home paths (`/Users/...`) replaced with relative locators (`/experiments/...`) in the artifact-references line and Current state. The S3 URIs (which contain the operator storage prefix) are retained intentionally as operational provenance.
2. The phrase "another user's academic identity and another username" replaced with "other users' jobs"; no third-party identifiers are published in this record.

**Evidence legend and navigation**

- `[P]` = publicly inspectable (marin public code/commits/issues, HF cards); `[R]` = reported from local reports or credentialed audits — a public reader cannot inspect these bytes here; `[M]` = requires Marin/CoreWeave credentials (S3 objects, W&B). "Verified" throughout means **verified by a credentialed audit or local audit [R]**, not independently inspectable publicly; public code establishes procedures, not that particular runs produced the reported outcomes. The index for this record is a separate PR (marin-logbooks PR #55); this notebook is pinned on research branch `codex/54-snowball-isoflop-ablations`.

**Initialization — "pretrained" vs "from-scratch" (corrects the 16:16 entry's mixed wording)**

3. Every cell **initialized from a pinned pretrained Snowflake base checkpoint**, not from random initialization and not from scratch: d512/d768/d1024 = `open-athena/Snowflake-d{512,768,1024}-Base-262K-2026.10.19` at revisions `557d1776…` / `3b68348a…` / `dd3c5e3c…` respectively [P: input receipts `inputs/d{size}-hf-config.json` on the pinned branch; SRC config.py builds `HFModel(model_ref=<repo>@<revision>)`]. What was fresh: the **SFT trajectories** — new optimizer state, new data schedule, 262,144-token assistant/SFT objective; these are fresh fine-tuning runs of pretrained bases, and the word "from-scratch" in the user's early-stop quote and FINAL.md refers to the campaign-scope judgment (small models trained only at 3e18–3e19 budgets, no measured connection to the 67B system), not to random initialization. No pretraining optimizer state was resumed.
4. Token-initialization semantics (material setup, held constant across arms): seven special-token output columns (`SPECIAL_IDS` = 128006, 128007, 128002, 128003, 128005, 128011, 128009) are reset to fixed anchor columns (`ANCHOR_IDS` = 3560, 1984, 1781, 4320, 5507, 842, 128001) with a special-token learning-rate multiplier √32 (`SnowballSftConfig(SPECIAL_IDS, ANCHOR_IDS, math.sqrt(32), True)`) [P: SRC config.py]; every run passes two **initial-loss gates** — the raw checkpoint's initial eval and a second eval after the reset must both match independently computed losses within tolerance (per marin#11440's reproduction recipe). The gates' scalar baseline values are in each control directory's receipts [M] and are not published here.
5. Mask semantics: `assistant` rewrites the chat format of SFT-tagged leaves to `serialized_chat` (message_start 128006, assistant prefix 128006/78191/128007/198) so supervision covers assistant tokens; `full` keeps declared formats (replay text carries loss); **pretrain replay is a separate source family and never receives the SFT chat format** [P: SRC config.py `data_config`].

**Model identities (compact table; the 16:16 entry omitted these)**

| Size d = hidden | Layers | Total params (BF16) | MoE topology | Attention heads / KV heads | Context | HF base @ revision |
|---|---|---|---|---|---|---|
| 512 | 6 | 746,607,104 | 256 routed experts, top-4 + 1 shared; moe_intermediate 256 | 4 / 1 | 262,144 | Snowflake-d512-Base-262K-2026.10.19 @ `557d1776…` |
| 768 | 8 | 2,039,270,912 | 256 routed, top-4 + 1 shared; moe_intermediate 384 | 6 / 1 | 262,144 | Snowflake-d768-Base-262K-2026.10.19 @ `3b68348a…` |
| 1024 | 11 | 4,764,584,704 | 256 routed, top-4 + 1 shared; moe_intermediate 512 | 8 / 2 | 262,144 | Snowflake-d1024-Base-262K-2026.10.19 @ `dd3c5e3c…` |

Vocab 128,256; sliding window 2048; full-attention layers are every 4th layer plus the last. "67B" = the Snowball production target system (67B-total/~2B-active MoE; distinct from the small Snowflake ladder — the base cards note the small ladder omits a pretraining stage used by Snowball). "Snowflake" = the small base ladder; "Snowball" = the 67B model family. Active parameter counts per size are not published here (derivable from the pinned HF configs).

**Phase/design map (the selection rules actually used; the 16:16 entry described termination only)**

| Phase | Scientific question | Varied | Fixed controls | Budget | Selection rule | Repetitions | Status |
|---|---|---|---|---|---|---|---|
| A | Do mask/size cells train stably to permanent endpoints? | size (512/768/1024), mask (full/assistant), seed (0/1/2) | global batch 8, loss-curve runtime, assistant/SFT recipe, pinned bases | seed0 3e19; seeds 1/2 1e19 reference targets | n/a (no selection; endpoint training) | 3 seeds per size/mask (seed0 at 3e19, seeds 1/2 at 1e19 — not a matched three-seed ladder at every budget) | 18/18 SUCCEEDED; no score phase |
| B | Saturate batch per size | microbatch × accumulation at CP8/data1/EP1 | context 8, data 1, offload on, MuonH lr 5e-05, seed 0, 8 updates | bounded ≤7 probes/size | saturation search (below) | 1 | selections complete; result files [M] |
| C | Compare layouts at the selected batch | dispatch (sonic vs ring), offload | B-selected batch per size, 3e18 target | equal 3e18 target per finalist | lower final `train/loss` (loss-first; ties untested) | 1 per finalist | surviving subset complete; offload-0 finalists killed → degenerate-finalist exclusion; d1024 had one feasible finalist (ring) |
| D | Optimizer/LR at equal compute | optimizer (MuonH vs adam), LR (center/3, center, upper) | C-selected layout per size, assistant mask, seed 0 | 3e18 target per trial | lower final `eval/loss` per optimizer; winner = min across optimizers | seed 0 only | 17/18 trials; d1024 has no winner |
| Q5 | Paired optimizer repeats at 1e19 | optimizer, seed (0/1/2) | D winners | 1e19 per repeat | planned: paired comparison | 3 seeds × 2 optimizers per size | never launched |

**Glossary (project terms not derivable from general ML background)**

- **IsoFLOP**: equal-compute scaling ablation (equal *target* FLOPs across compared cells). **long-loop**: the #49 SFT→RL→eval production loop this campaign was meant to inform. **SFT/RL**: supervised fine-tuning / reinforcement learning. **Muon-H (MuonH)**: the Muon-family optimizer used as `grug_moe_muonh_v1`; **adam** the comparison optimizer. **Sonic** and **ring_gather_combine**: the two MoE **token-dispatch** implementations compared (both are dispatch backends, not model architectures). **degenerate finalist / degenerate_lr_tuning**: a finalist invalidated by external cancellation (e.g., killed mid-run by a tree cancellation or preemption wave) and excluded under the pre-registered rule; **an operational bookkeeping label, not a scientific quality verdict**. **CP/data/EP (context/data/expert parallel)**: mesh axes; **mb/accum**: per-shard microbatch × accumulation. **Iris root/child**: a submitted root job and its child tasks. **Filer**: the authenticated storage browser for S3 objects. **HLO lowering**: the XLA compilation stage observed before the stall window. **Q5**: the planned-but-never-launched paired optimizer repeats at 1e19. **dispatch-count instrumentation**: sender/receiver drop counters enabled for B/C diagnostics (an attribution caveat vs the main calibrations).

**Compute and batch units (corrects the 16:16 entry's "3e18 reference FLOPs" ambiguity and the task issue's "position budgets")**

6. The 3e18/1e19/3e19 numbers are **analytic reference training-FLOP targets**, not position budgets and not measured FLOPs: `budget_steps`/`at_budget` divide the target by `training_flops_per_position(size) × batch × seq_len` and **floor to whole updates**, so a target is a ceiling, not a guarantee of exact realized compute [P: SRC config.py `training_flops_per_position`/`budget_steps`; probes.py `at_budget`]. The analytic formula is 3× (forward+backward) per-token model FLOPs including attention (sliding window 2048) and the active MoE routing (top-4 + 1 shared expert); it **excludes** optimizer, evaluation, and I/O overhead. The same per-position reference cost re-applied to realized steps gives, e.g., ≈2.9994e19 (d768 A seed0, 2426 updates), ≈2.9996e19 (d512 seed0, 3766 updates), ≈9.98e18 (d1024, 413 updates) — derived analytic values, not hardware counters [R: isoflop-recs-v2].
7. **Positions** = packed 262,144-token sequence positions. Realized per-trial examples [R: isoflop-recs-v2]: each d768 D trial consumed 503,316,480 positions (64 sequences × 262,144 × 30 updates) with stored reference_flops 2.96723416281514e18; the d512 MuonH-upper trial stopped at 22 updates (2.9310945899996774e18); the d1024 MuonH-upper trial ran 30 updates (2.9967750165965046e18). **Equal target ≠ equal realized compute ≠ equal supervised-token exposure** across sizes.
8. **Batch geometry**: global sequences per update = microbatch × accumulation × data × expert × replica; context parallelism shards each 262,144-position sequence over `context` devices, and every probe uses exactly one 8-GPU node [P: SRC probes.py `Layout.batch_shards`/`Probe.batch_size`]. The B-selected CP8/data1/EP1 geometries therefore imply **128 / 64 / 32 sequences per update** for d512/d768/d1024. **This supersedes the #49 record's "D cells … at batch 256" glossary line** (that record's revision 8 withdraws it): the D trials ran the C-selected geometries above, not a global batch of 256. Phase A cells trained at global batch 8 with per-size meshes recorded in their consumed-config receipts [M]. B and C/A/D also differ in **runtime**: B/C ran the SOURCE runtime; A/D ran the loss-curve runtime = the synchronous-collectives workaround (`--xla_gpu_enable_command_buffer=` off, `CUDA_LAUNCH_BLOCKING=1`, `--xla_gpu_disable_async_collectives=ALLCOLLECTIVES`, plus the always-on `--xla_gpu_shard_autotuning=false` and `--xla_gpu_nccl_termination_timeout_seconds=600`) [P: SRC config.py `RuntimePolicy`/`LOSS_CURVE_RUNTIME`/`build_spec`]. Throughput figures measured under the SOURCE runtime (B/C) therefore do not automatically transfer to A/D-phase conditions or to the 67B system; A/D cells consumed the workaround runtime, which #11440 shows is a scoped workaround whose timings are excluded from source-runtime comparisons.
9. **Training data** (material to every conclusion; summarized here for the first time): the consumed recipe is the pinned `inputs/exp1-recipe.json.gz` on the campaign branch [P: https://github.com/marin-community/marin/blob/94d180c765ee503a8b6d947a2a945c6055bbeffc/experiments/post_training/snowball_isoflop/inputs/exp1-recipe.json.gz] — 377 declared components with a 19-entry phase weight schedule; **only phase 0 is reached at these budgets** and the code keeps exactly phase-0 weights [P: SRC config.py `data_config`]. Phase-0 leaf weights aggregate to **79.53% SFT chat sources / 20.47% pretrain replay** (computed from the recipe's phase-0 weights) — this is **not** a pure assistant-chat mixture. Sampling unit: packed 262,144-position sequences; tokenizer pinned by the recipe (`e1["data"]["tokenizer"]`); four pretrain sources are reserved for validation at 2 sequences each (`pretrain/c01q2`, `c02q4`, `c20q2`, `c34q3`); packing/shuffling settings are in the recipe. Per-cell realized loss-token exposure is not published; dataset rows are not published (data rule).

**Metric definitions (loss, throughput, headroom, B selection)**

10. **`eval/loss` (D selection metric)** = the **final assistant-token micro cross-entropy at the last update** (global_step = steps−1) of the assistant-target validation; the code states "the primary metric is assistant-token micro cross entropy" and preserves macro/source curves alongside [P: SRC optimizers.py `final_loss`]. Lower is better. `train/loss` (C-loss comparisons) is the training-loss channel at the final update. Loss type/units: cross-entropy as implemented (per-token normalization as implemented); the exact token-count denominators, evaluated source weights and held-out population sizes per trial are in the result files' loss-curve rows (e.g., 30 rows per d768 trial) [M] and are **not** reconstructed here; no pre-SFT baseline loss is published (the two initial-loss gates are separate diagnostics).
11. **B throughput/headroom**: `measurements.py` distinguishes **learner** timing (`throughput/duration`) from **inclusive** timing (sum of component timings) and the matching positions/s/GPU variants; a warm measurement **excludes the first three updates** and requires **at least five complete warm update records**, reporting **medians** over warm updates [P: SRC measurements.py `warm_measurement`]. The B table's positions/s/GPU and "Min headroom" come from each result file's stored measurement summary [M]; the saturation rule itself compares **inclusive** positions/s/GPU. "Min headroom" = `minimum_memory_headroom_fraction` over warm updates with all 8 devices measured — a **fraction of allocator headroom under the JAX allocator cap (~75% of physical GPU memory per marin#11440)**; 0.652/0.338/0.328 are fractions, and the d768 sonic mb16 value −0.0257 is a **failed memory margin against that cap** (max peak 60.91 GiB), not negative physical free memory.
12. **B selection rule** [P: SRC run_probes.py `saturation_search`]: at most 7 bounded probes per size; every valid probe must record zero sender/receiver dispatch drops and present memory statistics with headroom ≥ 0.1; the search stops with a recorded `reason` per size — `two_consecutive_gains_below_5_percent` (selecting the third-from-last valid probe), `stall_limit_before_bracketed_saturation`, `memory_limit_before_bracketed_saturation`, or `search_limit_before_bracketed_saturation` — with an accumulation-growth fallback at the last safe microbatch. The per-size `reason` fields live in `S/manager-B/completed-{512,768,1024}.json` [M]. **Treat the B selections as "best feasible observed within the bounded search"; full saturation bracketing was not established for every size.**

**Phase D trial-level results (the 16:16 entry published only winners)**

13. **d768 — all six trials, exact stored values** [R: isoflop-recs-v2 re-audit of `S/manager-D/completed-768.json`; all status=complete, no exclusions, `degenerate_lr_tuning` absent; each trial 30 loss-curve rows, positions 503,316,480, reference_flops 2.96723416281514e18; seed 0]:

| Optimizer | Candidate | LR | Final eval/loss (step 29) |
|---|---|---|---|
| MuonH | center/3 | 1.6666666666666667e-05 | 1.2880408763885498 |
| MuonH | center | 5e-05 | 1.2603148221969604 |
| MuonH | upper | 0.00015640155002547376 | **1.2166242599487305** (winner) |
| adam | center/3 | 3.846153846153847e-06 | 1.217353343963623 |
| adam | center | 1.153846153846154e-05 | 1.1831244230270386 |
| adam | upper | 3.609266539049395e-05 | **1.1665005683898926** (winner) |

14. **Precision correction (d768 gap)**: the full-precision gap is 1.2166242599487305 − 1.1665005683898926 = **0.05012369155883789** (rounds to 0.050124 at six decimals). "0.05012369" (used in FINAL.md and the 16:16 entry) is that full-precision value at eight decimals and **cannot** be recovered by subtracting the six-decimal displayed losses 1.216624 − 1.166501 (= 0.050123); the earlier "0.050123" was an artifact of subtracting rounded inputs.
15. **Optimizer-conclusion scope (consistent across sizes)**: d512 and d768 are both **descriptive seed-0 tuning winners within the searched grid** — all four winning LRs are the **upper search boundary**, so higher LRs were never tested; these are not identified LR optima. The d768 stored winner values are additionally **re-audit-verified** [R: isoflop-recs-v2]; the d512 winner file (`completed-512.json`) postdates that audit (the re-audit found it absent at ~14:49 UTC; FINAL.md later records it complete) and has **not** been independently re-verified — attestation status: recorded in FINAL.md [R]. No optimizer conclusion (as opposed to these tuning observations) is possible without Q5-style paired repeats.
16. **d512 trials** — winners [R: FINAL.md §2]: MuonH upper 2.536339912212702e-04 → eval/loss 1.413755; adam upper 5.853092105106236e-05 → 1.300438 (no exclusions). Re-audit-verified upper-candidate probe [R: isoflop-recs-v2]: MuonH upper → 1.4137552976608276 at 22 updates (reference_flops 2.9310945899996774e18); rounded W&B corroboration 1.41376. The other d512 trial losses live in `S/manager-D/completed-512.json` [M] and are not transcribed here.
17. **d1024 trials (5 of 6; no winner)** — MuonH center/3 1.66667e-05, center 5e-05, upper 1.04957e-04; adam center/3 3.84615e-06, center 1.15385e-05; adam upper 2.42209e-05 **killed by the stop order** (child `59c010b63b88`). Re-audit-verified [R: isoflop-recs-v2]: MuonH upper → **1.0135018825531006** at 30 updates (reference_flops 2.9967750165965046e18; rounded W&B 1.0135). The adam center trial completed on its first retry (`-retry1`, child `0d58a494741d`) after the D-r7 attempt `966fbb8c4da6` FAILED/124 — one live success of the retry policy. Exact per-trial losses for the other four trials live in the `S/D-*` probe-result.json files [M].

**Phase A endpoint manifest (explicit 18 rows; replaces the 16:16 entry's shorthand with `…` and ambiguous r/seed pairings)**

All endpoints permanent (`is_temporary=false`), zero-based step indices; prefix `S/` = `s3://marin-us-east-08a/marin/users/benfeuer/snowball-isoflop/2026.11.13.10/` [R: FINAL.md §2; steps/permanence re-audit-verified for the eight starred objects; timestamps from that re-audit]:

| # | Cell | Endpoint (under `S/`) |
|---|---|---|
| 1 | d512 full seed0 (3e19) | `A-host512-r4-d512-full-seed0-n3766/checkpoints/step-3765/` ★ 2026-11-14T12:51:36Z |
| 2 | d512 assistant seed0 (3e19) | `A-host512-r4-d512-assistant-seed0-n3766/checkpoints/step-3765/` ★ 2026-11-14T12:33:37Z |
| 3 | d512 full seed1 (1e19) | `A-host512-r7-d512-full-seed1-n1255/checkpoints/step-1254/` |
| 4 | d512 full seed2 (1e19) | `A-host512-r7-d512-full-seed2-n1255/checkpoints/step-1254/` |
| 5 | d512 assistant seed1 (1e19) | `A-host512-r13-d512-assistant-seed1-n1255/checkpoints/step-1254/` |
| 6 | d512 assistant seed2 (1e19) | `A-host512-r13-d512-assistant-seed2-n1255/checkpoints/step-1254/` |
| 7 | d768 full seed0 (3e19) | `A-host512-r4-d768-full-seed0-n2426/checkpoints/step-2425/` ★ 2026-11-14T13:47:15Z |
| 8 | d768 assistant seed0 (3e19) | `A-host512-r4-d768-assistant-seed0-n2426/checkpoints/step-2425/` ★ 2026-11-14T12:45:11Z |
| 9 | d768 full seed1 (1e19) | `A-host512-r7-d768-full-seed1-n808/checkpoints/step-807/` |
| 10 | d768 full seed2 (1e19) | `A-host512-r10-d768-full-seed2-n808/checkpoints/step-807/` |
| 11 | d768 assistant seed1 (1e19) | `A-host512-r13-d768-assistant-seed1-n808/checkpoints/step-807/` |
| 12 | d768 assistant seed2 (1e19) | `A-host512-r13-d768-assistant-seed2-n808/checkpoints/step-807/` |
| 13 | d1024 full seed0 (3e19-class) | `A-host768-r10-d1024-full-seed0-n1241/checkpoints/step-1240/` ★ 2026-11-14T18:54:09Z |
| 14 | d1024 full seed1 (1e19) | `A-host768-r10-d1024-full-seed1-n413/checkpoints/step-412/` |
| 15 | d1024 full seed2 (1e19) | `A-host768-r13-d1024-full-seed2-n413/checkpoints/step-412/` ★ 2026-11-15T02:08:48Z |
| 16 | d1024 assistant seed0 (3e19-class) | `A-host768-r14-d1024-assistant-seed0-n1241/checkpoints/step-1240/` ★ 2026-11-15T07:36:07Z |
| 17 | d1024 assistant seed1 (1e19) | `A-host768-r14-d1024-assistant-seed1-n413/checkpoints/step-412/` ★ 2026-11-15T06:42:20Z |
| 18 | d1024 assistant seed2 (1e19) | `A-host768-r14-d1024-assistant-seed2-n413/checkpoints/step-412/` |

Seed↔root mapping is explicit above (e.g., d512 seeds 1/2 under r7 full and r13 assistant; d1024 seed0 under r10 full / r14 assistant; d1024 full seed1 under r10, seed2 under r13). Extra retained d1024 seed0 full endpoints: 3e18 at `A-host768-r3-d1024-full-seed0-n1241/checkpoints/step-123/` and 1e19 at `.../step-412/`. **Seed 0 reaches 3e19 while seeds 1/2 end at 1e19 — this is not a three-seed replication at every budget.** Training completion ≠ quality comparison: no score phase ever ran, so canonical scoring remains unrun for all 18 endpoints.

**Budget/ceiling history (timestamped; approved maxima vs operational ceilings)**

18. **Approved phase maxima** (initial guard config; PLAN): A 2200, B 144, C 384, **D 384**, Q5 1152 [R: `config-initial.json` re-audit read 2026-11-15 14:40–14:49 UTC; PLAN]. The task issue's "D 384 was raised twice" conflates the approved maximum with the operational ceilings below.
19. **Operational D-ceiling history** (each raise reported with observed spend at the time; sources [R]):

| D ceiling | Set/observed | Source and attestation |
|---|---|---|
| 96 (provisional) | initial operational cap; covered D-r1..r4 | guard-config-after-d-r2/r3/r4 snapshots [R: d-kills audit] |
| 143.802171 | raised after the D-r4 guard stop at observed 95.802171 (+48 requested) [R: PLAN] | guard-config-after-d-r5.json snapshot [R] |
| 157.419767 | raised at stated observed 133.419767; live config.json LastModified 2026-11-15 14:26:49 UTC; latest.json (14:42:57) still showed 157.419767 | local raise2 snapshot + re-audit reads [R]; **the stated 133.419767 observed spend is not independently verified** (re-audit marked it unresolvable from retained records) |
| 172.458971 | raised at stated observed 148.458971, after the 14:42:57 latest.json snapshot | FINAL.md §2 [R]; **not independently re-verified by any audit — attested only by FINAL.md** |

20. **Stop semantics at the D-r4 guard stop** (recovered by the re-audit [R]): decision snapshot observed 95.77756444507175 + 24/60 = 96.1776 ≥ 96 ceiling (`limit-reached.json`, 11:20:21 UTC); terminal reconstruction 95.802171 (D-r1 17.594962 + D-r2 28.392260 + D-r3 29.115416 + D-r4 20.699533). B/C operational ceilings also moved during the campaign (B 119.135304 and C 131.685816 at the 14:26:49 snapshot; FINAL.md records final B 144 / C 384 observed-under). **Total observed spend 866.32 H100-h** (A 544.89 + B 107.47 + C 63.32 + D 150.64 + Q5 0, guard latest.json) is the guard's **charged-allocation account** (includes startup, failed attempts and reservations under its accounting; excludes nothing it charged); it is not a provider billing invoice, and the guard object is mutable — the terminal ledger snapshots cited here are the record.
21. **Dates added to the chronology**: all D-phase root-death/stop times in the 16:16 entry are **2026-11-15 UTC** (D-r4 guard stop 11:20:20.352; D-r5 12:23:20.749); guard config snapshot 14:26:49 UTC; audits observed 14:29–14:37 UTC (v1) and 14:40–14:49 UTC (v2 re-audit); archive entry drafted 16:16 UTC.

**Stall claims scoped (corrects overbroad "d1024-specific" and unlabelled ratios)**

22. **The first-update stall diagnosis applies to specific runs, not a universal property**: marin#11440 [P] documents first-update stalls for **d512 and d768** under several geometries/runtime settings, and this campaign's own B section recorded d512/d768 stalls (sonic mb2; CP4/data2 probes). The audited D-phase stall record covers the **d1024 ring mb4/acc8 (batch 32) assistant-mask D-r2..r5 cells** under this campaign's 1800 s watchdog only. Sonic-vs-ring was **measured per size** (C): d512 sonic faster (98,262.8 vs 94,133.6 pos/s/GPU), d768 ring marginally faster in throughput (76,990.4 vs 75,885.8) but sonic selected on final loss (stored train/loss gap 1.7471798658370972 − 1.7471333742141724 = 0.0000465), d1024 ring the only feasible finalist — no cross-size dispatch conclusion.
23. **d1024 D-r2..r5 seven-attempt table** [R: d-kills audit controller timeline; all attempt 0; geometry ring mb4/acc8, batch 32, assistant mask]:

| Root | Child | Role | Outcome (2026-11-15 UTC) |
|---|---|---|---|
| D-r2 | 54cb57eea796 | muonh center/3 | SUCCEEDED/0 07:54:08.902 |
| D-r2 | 7e51038ed5b9 | muonh center | FAILED/124 (watchdog) 08:38:07.163 |
| D-r3 | a1306637ee29 | muonh center | SUCCEEDED/0 09:21:43.014 |
| D-r3 | ae246dc12b8d | muonh upper | FAILED/124 (watchdog) 10:07:28.191 |
| D-r4 | 6ba9c1ead0d6 | muonh upper | SUCCEEDED/0 10:57:06.157 |
| D-r4 | 320641588fc6 | adam center/3 | KILLED (parent cascade at the guard stop) 11:20:20.352 |
| D-r5 | 85da4fc5abad | adam center/3 | FAILED/124 (watchdog) 12:22:49.004 |

Failure denominators, labeled: **3/6 among success/failure outcomes; 3/7 including the censored killed attempt.** Same-LR pairs failed and succeeded across roots (muonh center failed r2/succeeded r3; upper failed r3/succeeded r4) — one stall does not establish deterministic failure at a geometry.

**Cancellation attribution and fix mapping**

24. The cancellation verdicts (watchdog exit-124 self-cancels via `budgeted_map`; D-r4 phase-budget guard; counter artifacts) remain **findings of the local audit [R]**: the 23-child timeline, query predicates/time bounds and decisive sanitized event rows live in the local report and are not published here; pinned public code supports the *possible* mechanisms only. marin#11440 predates D and is **not** evidence for the D cancellation history.
25. **Fix descriptions mapped to commits** (the 16:16 entry's "stall classification from job status `f103997ccd`, corrected to `f9a12ef08c`" compressed two changes): job-status-based probe-failure classification is **`59ffe6f503`** ("Classify probe failures from job status, not logs"); `f103997ccd` changed D-phase handling; `f9a12ef08c` added terminal-error-text recognition with the regression test `test_failure_classification_reads_aggregate_status_without_task_detail`; `fbc123c0bd` cross-attempt trial reuse; `e7ef25b469` attempt-scoped trial identities; `94d180c765` stall-retry policy + D result persistence. **"Each with tests" means tests are present; the audits did not execute them** — test presence, not verified passing. The retry policy has **one live success** (d1024 adam center via `-retry1`); its success rate is unmeasured, and treating the underlying stall as "infrastructure failure" reflects the retry policy's classification, not a proven cause.

**Retention and next-step state (corrects stronger wording in the 16:16 entry)**

26. **Retention**: the stop order requested preservation, and every inspected endpoint existed at its audit observation time (timestamps above; re-audit 14:40–14:49 UTC). "Nothing deleted" is accurate **as of the audits and within the inspected namespace**; strict historical absence of deleted objects was not verified. Checkpoint `is_temporary=false` is a metadata flag — it does not establish indefinite storage retention or resumability, and no tensor-level integrity audit was performed.
27. **Execution state**: the campaign is **archived and stopped; no further work is authorized or planned by this record.** The RECOMMENDATIONS document (rev 3) is an **untested proposal** [R]; its items (keep three sizes, run the score phase, Q5 repeats) are conditional suggestions, not committed continuations. If ever revived: canonical scoring (`scoring.py` over endpoint manifests) would address **size/mask** comparisons; paired seed repeats (Q5-style) would address **optimizer** uncertainty; 67B transfer requires new full-model measurements (the #11440 64-H100 reproduction did not qualify the 8-GPU workaround at full-model geometry [P]). These are three separate questions; "both needed" phrasing in the 16:16 entry conflated them. No owner/tracking link exists for any revival item (open work item).

**Privacy finding (flagged, not reproduced)**

28. The public marin commit `94d180c765…` (the campaign's final source head, cited throughout this record) exposes a personal workstation email/hostname in its public author metadata. No address is reproduced anywhere in this record; commits are referenced by SHA only. Flagged for repository-owner remediation; no history rewrite performed here.
- Evidence: [P] links inline; [R] local reports/audits as cited; [M] credentialed-only objects as cited.
- Interpretation and limitations: the 16:16 entry's sections superseded by this entry: the "from-scratch" phrasing (item 3), the missing model/phase/glossary detail (items 4–9), the unlabeled denominators and overbroad stall wording (items 22–23), the ceiling history (items 18–20), the retention and next-step wording (items 26–27), and the missing trial-level tables (items 13–17). Items NOT addressed (owner actions or unavailable evidence): publishing sanitized result-file/ledger extracts ([M] objects need a publication decision); the exact observed spend stated at the 157.419767 raise (not in any retained record); tensor-level integrity checks; and the marin commit-metadata remediation (owner decision).
- Decision and next action: fresh cold read round 2 next.
