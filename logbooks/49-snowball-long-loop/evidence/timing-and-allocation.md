# Timing and allocation ledger — Snowball long-loop exp1 and exp2

- **What it is:** stage boundaries (SFT start/end, handoff, RL start/end, eval start/end, seal), cancelled/failed attempts with H100-hour costs, per-stage GPU allocations, RL budget consumption and semantics, and end-to-end bounds.
- **Compiled at:** 2026-11-15T15:14Z; round-7 cold-read corrections applied 2026-11-15.
- **Sources:** [R] `/Users/benfeuer/Documents/experiments/active/sft-new-mix-candidate/STATE.md`; [R] `/Users/benfeuer/Documents/experiments/complete/long-loop-prototype/STATUS.md`; [R] exp2 artifacts under `.../sft-new-mix-candidate/artifacts/experiment2/` (as cited per row); [P] marin#11337 round-1 body (published exp1 clock anchors) and HF model pages; [M] W&B/EvalDash links noted inline (require authentication/Marina login).
- **Access class:** timestamps, durations, job names, allocation counts and H100-hour aggregates only. "H100-hours" figures are reconstructed allocation/duration accounts from attempt records, **not provider billing** unless stated.

## Experiment 1 (round 1) stage ledger [R: STATUS.md]

| Stage | Start (UTC) | End (UTC) | Notes / cost |
|---|---|---|---|
| SFT attempt `.11` | submitted 2026-11-09 20:44 (`/benfeuer/long-loop-full-sft-2026.11.09.11`, pin `d7795c7709…`, 746 steps / 50.063B positions / 256 H100) | FAILED 2026-11-09 21:07 after first optimizer step | experiment-code bug in `checkpoint_samples`; no downstream handoff admitted |
| SFT `.12` | submitted 2026-11-09 (after `.11`; source `9662fdba02…`); 32/32 running 2026-11-09 22:45 | COMPLETED 2026-11-10 18:52:33 (drain end) | step746 verified, 50,063,212,544 scheduled positions; reused-SFT allocation later reconstructed as **5,393.887062 H100-h** |
| Handoff/export (`.10.1` graph) | 2026-11-10 19:07 (RL admission submitted) | adoption 2026-11-10 19:36:57 (28m24s), handoff export 20:09:40 (32m39s), endpoint 20:14:29 (4m44s) | `.10.1` itself failed at runtime `validate_megatron_cfg` (CP4 vs packing) — no RL updates |
| RL admission attempts `.10.1–.10.9` | 2026-11-10 19:07 → 2026-11-11 | all failed before/at first update | closed costs: `.10.10` 228.7249 H100-h (created-to-terminal envelope 231.0532); `.10.9` 167.790245 (RL 167.425480 + preparation 0.364765) |
| RL attempts `.11.1–.11.4`, `.12.1–.12.4` | 2026-11-11 → 2026-11-12 | stopped by user/verifier repair | closed costs: `.11.1` 256.4028356, `.11.2` 682.536213 (2 updates), `.11.3` 427.083804 (0 steps), `.11.4` 279.683556 (0 steps), `.12.1` 136.723062 (0 updates), `.12.2` 81.5514, `.12.4` 186.7655 |
| RL training run `.12.5` | submitted 2026-11-12 (engine-ready 07:36:46) | RL FAILED 2026-11-12 20:55:18.825 (147.045 s after the nominal 12 h boundary 20:52:51.780) | closed RL allocation **3,447.110733 H100-h**; native totals 45,576.78 s (see decomposition note below); RL budget receipt after step17: 41,558.937 s of 43,200 s, 1,800 s drain margin |
| Best16 policy-only export | 2026-11-12 21:32 (job on 80 H100, source `b29f6d045d`) | succeeded 2026-11-12/13 (per finalization ledger) | export 33.9652 H100-h |
| Final Micro210 eval | eval child submitted 2026-11-12 22:55:38.954; eval child SUCCEEDED | recovery eval SUCCEEDED 2026-11-13 ~02:38 | original eval 298.3989 + judge 20.6328 + recovery eval 81.8863 + recovery judge 6.9793 = within finalization ledger **441.8624533 H100-h** |
| Report/archive/seal | post6–post11 CPU chain 2026-11-13 | **archive sealed 2026-11-13** (coordinator completion 06:19:50.069 [P]; finalization report 06:40) | seal SHA `5a67530dacefa5af0523a2010671c86097e3d7f666582616b339adeaa98bd585`, 2926 objects / 1,351,769,694 bytes; public HF best16 at revision `475ba62d4e9498b075d52ec7168449b19a4ab683` [P] |

**Exp1 `.12.5` native-duration decomposition is NON-EXHAUSTIVE:** the listed native components (admission 8,748.87 + training preparation 1,354.17 + policy training 4,033.21 + sync 336.53 + regular checkpoints 4,366.14 + evaluation inclusive 26,557.22 — native eval drains 16,906.81 s contained within evaluation) sum to **45,396.14 s against the stated total 45,576.78 s, leaving a 180.64 s residual**. The component list does not close the total; unlisted phases account for the residual. This is a partial decomposition, not a closed accounting [R: STATUS.md L902].

**End-to-end anchors (published, use these as the clock anchors):** the round-1 recipe entry publishes initial loop submission **2026-11-09 21:45:42.016 UTC → archive coordinator completion 2026-11-13 06:19:50.069 UTC = 80.568904 h**, with the audited scope of **12,624.162622 requested H100-hours across 659 closed attempts** (full-loop/best16 prefixes; differently named probes/plumbing and CPU work excluded) [P: marin#11337 round-1 body, "Evidence and limits"; R: STATUS.md L1052, L1082]. The measured 80.568904 h figure matches these anchors; overrun **44.568904 h** over the 36 h target. Final timing supplement published under east08 `evidence/20261113-best16-final-timing` (report SHA `086b14cd856626bd3689198cd0f13c3b383792fcea3549adf1652bf7008384ea`, bundle SHA `1ee76f2e6b8ba0b12ca71de0c1e442dd12068bbc8a418ed9df8328b72a9616b4`).
Per-stage GPU allocations: SFT 256 H100 (32×8); RL 256 H100 peak (80 learner + 176 rollout); best16 export 80 H100; final eval 128 candidate + 8 judge = 136 peak requested.

## Experiment 2 (round 2) stage ledger [R: STATE.md + artifacts/experiment2]

| Stage | Start (UTC) | End (UTC) | Notes / cost |
|---|---|---|---|
| SFT smoke/attempt full9 | submitted 2026-11-13 19:07:40 | wrapper failure 2026-11-13 19:11:13 | **0 H100h** (no allocation) |
| SFT attempt full10 (batch 8) | 2026-11-13 19:17:26 (32/32 running) | cancelled 2026-11-13 19:40:02 (user: batch 256) | **66.3749 H100-h** |
| SFT full11 (batch 256, source `442bf9d878`) | submitted 2026-11-13 19:59:09; child running 20:00:57 | COMPLETED 2026-11-14 07:49:22 (204/204 updates) | allocation **2,994.3104 H100-h** (32 tasks × 11h41m47.49s × 8); final train loss 0.690054, validation 0.676990 (context in exp2-data-accounting.md) |
| Handoff | automatic export running from 2026-11-14 ~07:48 | **COMPLETED 2026-11-14 08:14:20** | step203 + consumed-config SHA `1286fb00…` bound by handoff receipt SHA `855aaeb7d96abc10a2f574acfa5cec6cdba9446196c8d308bf50684e9d6950e1` |
| RL attempt `.13.11` | submitted 2026-11-14 08:09:48; admitted 14:40:59 after **6h31m11s** queue wait | startup failed 2026-11-14 (vLLM engine init) | RL occupied **~110.252 H100-h** across 32 eight-GPU tasks |
| RL recovery `.14.12` | submitted 2026-11-14 15:48:41; admitted 16:26:57 | cancel **acknowledged 2026-11-14 17:06:33** (iris CLI "Cancelled jobs"); cancellation summary recorded **17:09:22.876** after verifying zero remaining pods | **167.3094 H100-h** excluding queue. Timestamp reconciliation: STATE.md L77's "cancelled at 17:06:33" is the **cancel-requested/acknowledged** time (`cancel_acknowledged_utc` in `recovery12-harness-failure-summary.json`, matching the local iris cancel-command log); 17:09:22.875881 is the summary's `recorded_utc`, i.e. the **terminal/verified-zero-pods** time. Both are correct; they bound different events. |
| RL recovery `.14.13` (runtime `038ad7f188`) | submitted 2026-11-14 17:15:26; admitted 17:18:52; first optimizer update 18:16:58 | RL/step20 compute complete 2026-11-15 ~04:20 (checkpoint20 committed 04:24:01) | 256 H100 (80 learner + 176 inference); W&B `o40fk9b6` FINISHED at 20 [M] |
| Checkpoint export + HF publish | conversion on **80 H100** (executed allocation: export request receipt records 10 nodes × 8 GPUs, first attempt, SUCCEEDED); files published 2026-11-15 05:33 | export/preparation receipts 2026-11-15 ~05:37 | model prep 28.94 s; 39 shards / 134,157,778,944 tensor bytes. The as-run export allocation is **not among the enumerated configured stage allocations** (SFT 256 / RL 256 / eval 136 in `full11-launch.json`), so it is recorded here as an **executed deviation from the published stage-allocation list**; **ceiling (≤256 H100) compliance remains configured/reported, not independently verified**. |
| Final evaluation | 128-H100 candidate entered eval session 2026-11-15 05:59:42; judge 8 H100 (136 peak) | **eval job SUCCEEDED by 2026-11-15 08:26**; **scoring coverage completed after verifier recovery ~08:35** (recovered view/report committed 437/440 + exact TB2 infra-unscored UIDs) | 72-hour deadline; 440 attempts (see final-eval-attempts.md). These are distinct events: the job terminal at 08:26 did not yet include the recovered 437/440 scored view. |
| Postprocessing `.15.17→.19` | .17 submitted 2026-11-15 ~08:26; deliberately cancelled (accounting too slow); .18 cancelled; .19 submitted 09:18 | CPU accounting/archive/publication children succeeded 2026-11-15 | CPU-only: 32 CPU / 64 GB / 500 GB scratch; 1,800/5,400 s deadlines |
| **Seal** | — | **completed 2026-11-15 10:49:53.982839 UTC** | seal tag `experiments/exp2-rl20-2026.11.15` at postprocessing commit `4913edc488`; completion receipt SHA `839a0308fca1dd2a48740b944de661affa249eb31c078529dbf8be8638a8b42d` |

**End-to-end (computed values, assumptions stated):**

- **Endpoint elapsed = 38 h 50 min 44.982839 s**, computed from the ledger's initial submission timestamp **2026-11-13 19:59:09 UTC** [R: `full11-launch.json` `submitted_utc_observed` 19:59:09.704789, used at second resolution] to the public completion **2026-11-15 10:49:53.982839 UTC** [R: `completion19-receipt.json` `completed_utc`; P: #11337 round-2 comment].
- **Running-to-completion = 38 h 48 min 56.982839 s (≈ 38 h 49 min)**, computed from the first observed SFT child RUNNING **2026-11-13 20:00:57.063583 UTC** to the same completion. This is the previously published "≥ 38 h 49 min" lower bound, now given as an exact computed span.
- **Assumption:** both figures treat the submission/RUNNING observation timestamps as the true stage-start anchors. The submission timestamp is an observer observation (`.704789` fractional part), so the endpoint figure carries ~1 s second-level uncertainty; no queue-wait phase attribution is required for either subtraction. **Phase attribution (how much of the span was queue/admission vs compute vs postprocessing) remains unmeasured.**
- Exp1's measured end-to-end was 80.568904 h for round 1 (published anchors above). Exp2 does not publish a single measured end-to-end figure beyond these computed spans.

## RL budget semantics and consumption (exp2)

- Configured budget: **36,000 s (10 h)** wall clock, counted from **first optimizer completion** (18:16:58 UTC 2026-11-14), including eval/save work, with a recorded **1,800 s drain margin**; verified stop occurs at a completed-step/log boundary (threshold ≈ 04:16:58 UTC 2026-11-15) [R: STATE.md L63, `recovery13-second-optimizer-milestone.json`].
- Actual consumption: the on_log budget stop fired after interval callbacks reporting **38,575.20 s elapsed against the configured 36,000 s budget**, with zero additional updates after the stop [R: `recovery13-rl-compute-complete-best20-milestone-20261115.json`].
- Exp1's RL budget for contrast: **43,200 s (12 h)** nominal with 1,800 s drain margin [R: STATUS.md L886].

## Unavailable / unknown (explicit)

- **Provider billing figures:** none exist in the compiled sources; every H100-hour figure is a reconstructed allocation/duration account.
- **Exp2 cancelled/failed attempt costs beyond those listed** (full9 0; full10 66.3749; `.13.11` ~110.252; `.14.12` 167.3094): any additional sub-attempt costs were not broken out in STATE.md.
- **Exp2 phase attribution within the end-to-end spans:** queue/admission vs compute vs postprocessing shares are not measured; the endpoint subtractions above do not require them.
- **Exp2 queue-wait costs before .14.13 admission** (capacity gating periods for full11 RL and .14.12) are recorded as wall-clock waits (6h31m11s; ~38m16s) but are excluded from the H100-hour figures cited ("excluding queue").
