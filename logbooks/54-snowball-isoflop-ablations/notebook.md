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
- External inputs and artifact references: S3 prefix `S/` = `s3://marin-us-east-08a/marin/users/benfeuer/snowball-isoflop/2026.11.13.10/` (Filer-browsable; reads need Marin credentials). Audits: `/Users/benfeuer/Documents/experiments/active/claim-audits/{snowball-d-kills,isoflop-recs,isoflop-recs-v2}.report.md`. Public stall issue: marin#11440. Recommendations rev 3 SHA-256 `9fb091eb5829f08faa4c414aa6accc58abf9fe66ee47a50c91c26272ecb1857e`. Receipts/guard snapshots: `/Users/benfeuer/Documents/experiments/active/snowball-isoflop-ablations/artifacts/pretrain-baseline-revision2/native-cpu-v10/resume-20261114/`.
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

1. The "preemption wave" narrative was WRONG. D-r2 (root death 08:38:25.483 UTC), D-r3 (10:07:49.618) and D-r5 (12:23:20.749) self-cancelled via the `budgeted_map` handler after a d1024 child hit the 1800 s watchdog (exit 124 on all ranks, no OOM; failed children `7e51038ed5b9`, `ae246dc12b8d`, `85da4fc5abad`). D-r4 was cancelled by the budget guard at the provisional 96 H100-h ceiling (11:20:20.352 UTC; guard log "Stopped registered root /benfeuer/snowball-isoflop-D-r4-v10 at phase D limit" at 11:20:21.325). Re-audit recovered the pre-cancel snapshot: observed 95.77756444507175 + 24/60 = 96.1776 ≥ 96.0 ceiling; terminal reconstruction 95.802171 (D-r1 17.594962 + D-r2 28.392260 + D-r3 29.115416 + D-r4 20.699533). `preemptions=1` on job status is a counter artifact (any KILLED started attempt counts; counter code path in the controller), and "Terminated by user" is the generic TerminateJob reason. Real Kueue evictions in the same window belonged to other users (another user's academic identity and another username), not this tree.
2. d1024 startup stalls are real and d1024-specific but not deterministic: in D-r2..r5, 3 FAILED/124, 3 SUCCEEDED, 1 KILLED across seven attempts (3/6 success-failure outcomes; 3/7 with the killed attempt). Log evidence (r3 child `ae246dc12b8d`): eval gate passed, data loaded, HLO lowering 28–31 s, then 17 data-loader warnings (09:46:44.879–09:47:28.476 UTC) until the watchdog at 10:06:33.950; the successful r4 sibling covered lowering→first step in 46.583 s. Hang sits after HLO lowering and before the first update. Cause unverified (hypothesis: collective first-step setup). marin#11440 documents the first-update stall controls: command buffers off + `CUDA_LAUNCH_BLOCKING=1` (+ async-collectives disable for some geometries) completed 20-update calibrations as scoped workarounds, but a 64-H100 67B reproduction (comment by another contributor) did not qualify the workaround at full-model geometry.
3. Code fixes during the campaign (each with tests): cross-attempt trial reuse `fbc123c0bd`; attempt-scoped trial identities `e7ef25b469`; stall classification from job status `f103997ccd`, corrected to terminal-error-text matching `f9a12ef08c` (GetJobStatus carries no task/attempt detail — empty `tasks[*].attempts[*].exit_code` caused D-r5 to self-cancel on a stall the code was meant to absorb); stall-retry policy + D result persistence `94d180c765`.
4. Budget-guard ceiling raises, each reported with observed spend: D 96 → 143.802171 (at observed 95.802171) → 157.419767 (at observed 133.419767) → 172.458971 (at observed 148.458971). Final live guard (config LastModified 2026-11-15 14:26:49 UTC): A 2200 (reserved 172), B 119.135304, C 131.685816, D 157.419767, Q5 1152.

**Known misconfigurations and defects** (FINAL.md §4): (1) campaign scope as above; (2) B-r4 default-geometry regression; (3) in-cluster log blindness — the regional log server serves nothing for terminal child jobs, so server-side classification saw empty input (fixed by `f9a12ef08c`/`f103997ccd`); (4) aggregate GetJobStatus carries no task/attempt detail; (5) D results not persisted before `94d180c765` (a tree loss would have discarded winners); (6) provisional phase ceilings set by the agent caused the one D-r4 guard stop; (7) manual reconciliation defects caught before damage (B reconciliation files missing the `["result"]` wrapper crashed C-r4 at admission; a d1024 endpoint initially cited under the wrong root phase, corrected to `A-host768-r10` after an existence check).

## Current state

No Q5 launch happened; no auto-launcher exists; after the final cancel no Snowball root is running or pending on the controller. Campaign directory moved to `/Users/benfeuer/Documents/experiments/complete/snowball-isoflop-ablations/` (FINAL.md and RECOMMENDATIONS still cite the former `active/` paths — those paths are stale). All S3 artifacts retained.

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
