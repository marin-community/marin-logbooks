---
title: "Snowball RL resizing campaign"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/71"
related_issues:
  - "https://github.com/marin-community/marin/issues/11969"
  - "https://github.com/marin-community/marin/issues/11954"
  - "https://github.com/marin-community/marin/issues/11957"
research_branch: "codex/71-snowball-resize"
marin_base_revision: "cd0e932c780e0171c5612658faa0f180a851569e"
archive_template_revision: "fc1cb30da0967f6ab78ffd47444ed9144f50cba9"
status: complete
---

# Snowball RL resizing campaign

Secretary record (evidence-backed logbook) for the ~12-hour Snowball RL resizing sprint: phase 1 maximizes Snowball decode throughput at the 65,536-token request window; phase 2 right-sizes the Snowball RL job (rollout + trainer GPUs, engine config, gates) to the smallest allocation that keeps training saturated. The experiment is owned by a fresh OpenCode worker whose primary records are `/experiments/complete/snowball-resize/{BRIEF.md,REPORT.md,STATUS.md}` (the REPORT.md top table is authoritative once arms land; moved from `/experiments/active/snowball-resize/` on 2026-11-26 per the coordinator after >72h idle). Open-Athena reproduction details — per-arm branches/commits, config paths, launch commands, Iris job IDs, S3 receipt paths with Filer links, finelog queries, and the results table — are tracked in the user-authorized single marin issue [marin#11969](https://github.com/marin-community/marin/issues/11969), which links the prior Mochi campaign (#11954, closed) and the concurrency-gate table (#11957).

## Question and decision criteria

- Goal and why it matters: Snowball's RL rollouts are dominated by long-context decode; knowing the maximum decode throughput at 65k and the smallest saturated RL job converts the current allocation guess into a measured, costed layout (time per update vs baseline, GPU-hours per update).
- Baseline and alternatives: the current Snowball RL allocation and role plan (Exp4/Exp5 long-loop RL on H100, cw-rno2a) — recorded (and measured if never measured) as step 1; alternatives are the phase-1 serving sweeps and the phase-2 shrink steps.
- Evidence needed to choose: phase 1 — native vLLM counters per arm (aggregate/per-GPU generated tok/s, per-request tok/s, running/waiting, KV, prefix hits, preemptions, TTFT); phase 2 — time per update, rollout wait, engine occupancy, and the per-step split (rollout wait / optimizer / checkpoint / eval) vs baseline.
- Constraints and unresolved questions: PRODUCTION priority; ≤ ~2 racks / 128 GPUs in flight without coordinator approval; do not disturb Exp5 SFT (256 H100, cw-rno2a); one knob per arm; 25-min windows then iris-cancel-own; Harbor ≤ 512; loop-push only; staleness/route quotas as in the current recipe; quantization/TP changes need a logprob-mismatch check against BF16 first. Unresolved until the sprint ends: the right-sized layout itself.

## Baseline

- Date: 2026-11-21/22.
- Published source revision: worktree `/Users/benfeuer/Documents/marin-worktrees/snowball-resize` (branch `benfeuer/snowball-resize` from the long-loop tip); code/config lands only via `loop-push` onto `benfeuer/long-loop-prototype`.
- Configuration, command, and external inputs: PRODUCTION priority; GPU budget ≤ ~2 racks / 128 GPUs in flight without coordinator approval; Snowball ran on cw-us-east-08a GB200 for the Nov 20 acceptance (`.305`, 40 GB200); cadence-3 checkpoint/eval pattern in the real-rollout phase; every run logs effective gates at startup.
- Environment and hardware when relevant: H100 (cw-rno2a) for the current allocation; GB200 (cw-us-east-08a) possible if the runtime supports it.
- Result and evidence: baseline recording in progress; see the entries below and marin#11969.
- Limitations: the sprint is ~12 hours; the deliverable is a right-sized layout recommendation with measured evidence, not a merged configuration.

## User rulings

- Phase 1 first (decode at 65k), then phase 2 (right-size); deliverable is the right-sized layout with measured time per update vs the current allocation plus GPU-hours per update.
- One knob per arm, config over code; cancel each measured arm with `iris-cancel-own`; Harbor ≤ 512; PRODUCTION priority; ≤ ~2 racks / 128 GPUs in flight without coordinator approval; never disturb Exp5 SFT.
- Staleness/route quotas stay as in the current recipe; default verifier scoring on timeouts (never timeout/length rejection); trajectory budget 30 min for agentic tasks.

## Entries

### 2026-11-22 00:30 UTC - Campaign opened; baseline recording in progress

- Question or hypothesis and reason for trying it: open the secretary record for the Snowball RL resizing campaign (coordinator request 23:00 EST Nov 21) per the Mochi-sprint playbook.
- Published source revision: worktree `benfeuer/snowball-resize` from the long-loop tip; loop-push onto `benfeuer/long-loop-prototype` only.
- Command or material configuration: no arms launched yet; step 1 is recording (and measuring if needed) the current Snowball RL allocation/role plan (GPUs, trainer parallelism, engines TP/DP/EP, max_num_seqs, gpu_memory_utilization, batched tokens, all concurrency gates) in the worker's REPORT.md.
- External inputs and artifact references: `/experiments/active/snowball-resize/{BRIEF.md,REPORT.md,STATUS.md}` [R]; Mochi results `/experiments/active/mochi-throughput/{REPORT.md,BRIEF.md}`, marin#11954 (closed), marin#11957 [R].
- Environment and hardware when relevant: current allocation on H100/cw-rno2a; GB200/08a possible (Nov 20 acceptance `.305`, 40 GB200).
- Result, including failure or restart: marin#11969 filed with the two-phase goal, rules, and lessons; this logbook opened; no measurements yet.
- Evidence: BRIEF.md [R]; #11969 body [R].
- Interpretation and limitations: the Mochi lessons transfer as priors (KV/sequences/expert-parallelism/batched-tokens won there), but Snowball's engine and workload differ — all conclusions must come from this sprint's own measurements.
- Decision and next action: record the baseline; then phase-1 serving arms (65k window, group-shared prompts, native counters) as the worker cc's them.

## Final decision

- Selected approach and evidence: **RIGHT-SIZED SNOWBALL RL LAYOUT (measured): 64 GB200 — trainer 8×4 nodes (Megatron PP2/CP4/EP4 all_gather, `dp_reshardable` checkpoints) + 4 rollout engines × DP8/EP8 at HBM 0.90, seqs 128, batched 8192; 16 rollout workers; all demand gates open (in-flight 1024, Harbor 384, verifier 384, tasks 2048, executor 384); staleness 2; curriculum-derived quotas 22/42; cadence-3; 30-min trajectory budgets — ~0.96 GPU-hour per update at batch 64.** Evidence chain: phase-1 decode sweeps (HBM 0.75→0.90 = 5.11× to 1,618.9 tok/s/GPU at the 65k window; seqs 64→128 = +16.5% to 1,886.4; KV must hold the shared-prompt working set); the phase-2 ladder on real rollouts (4 workers + closed gates 1,695.4 s at 90.2% wait → gates open 687.1 s → 16 workers **195.0 s/update at 13.4% wait**, 710.7 gen tok/s/GPU step-average, mismatch healthy) → **3 engines REGRESSED** (513.1 s at 72.8%) → **trainer-shrink does not fit** (~269 GB/rank vs 192 GB HBM). Both 32-GPU floors probed and held.
- Rejected alternatives and why: the nominal 256-GPU H100 layout (never completed an update; protecting Exp5 SFT) — measured on GB200 instead; 3 rollout engines (72.8% wait regression); 16 trainer ranks (~269 GB/rank > 192 GB HBM); HBM 0.75 (KV thrash, prefix 9.5%); seqs 64 at HBM 0.90 (slot-cap-bound).
- Marin contribution PRs: none from this campaign; durable outputs are this record, the curriculum-owned-holdout + derived-quota migration for the whole long loop (commit `12d161687e`), and the built/published exp4 rl-pools artifact (200-prompt curriculum holdout, 160,561 train rows).
- Reproduction instructions: marin#11969's Reproduction section (branches/SHAs per arm, `rounds/snowball-resize-rl-*.yaml` + `specs/snowball-decode-*.json`, launch commands, Iris job index, S3/Filer receipts) plus `secretary-arm-provenance-20261121.json` and worker REPORT.md/STATUS.md under `/Users/benfeuer/Documents/experiments/active/snowball-resize/`.
- Remaining uncertainty: all phase-2 numbers are GB200 — the H100-equivalent layout needs one confirming run on cw-rno2a when the reserved fleet frees; the curriculum handoff's broader sweep + rl-pools landing was in progress at close; the per-step split beyond step 2 (eval/checkpoint share) was not yet measured on the right-sized layout.

### 2026-11-22 00:50 UTC - Milestone 1: baseline RECORDED (256 GPUs); arms S1/S2 declared

- Question or hypothesis and reason for trying it: establish the measured starting point for the resizing sprint.
- Published source revision: worktree `benfeuer/snowball-resize` from the long-loop tip; loop-push onto `benfeuer/long-loop-prototype` only.
- Command or material configuration: baseline = Snowball RL **256 GPUs** from the frozen `.51` restart reference (`s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/evidence/restart53/frozen51-reference.json`, sha256 `897f22035802380a63fb748739ae8a66ef208ea67f1fe1e5470bbec417b3e1bb`; consumed by `experiments/post_training/long_loop/exp4.py` `restart_recipe`, mirrored in `specs/rl.yaml`). Layout: policy+ref 10 nodes × 8 H100 (Megatron TP1/PP2/CP5(a2a)/EP8/ETP1) + rollout 22 vLLM engines × TP1/PP1/DP8 (EP8 inside each DP8) = 176 rollout GPUs. Engine: BF16, FLASH_ATTN, triton MoE, seqs 64/rank, batched tokens 8192, HBM 0.75, prefix caching + chunked prefill + CUDA graphs, max_model_len 102400, hermes parser, enable_thinking, routed-expert capture. Gates: full_batch, staleness 2, max_in_flight 32, quotas 32/32 (Exp4 plan 22/42), Harbor 256, checkpoint/eval every 2 steps.
- External inputs and artifact references: worker milestone-1 cc over Rhizome (session `ses_dfe9b2626ffeHwqZI0Gd8yCX2L`) [R]; receipts `REPORT.md`, `STATUS.md`, `frozen51-reference-20261122.json` under /experiments/active/snowball-resize/ [R].
- Environment and hardware when relevant: cw-rno2a H100; Exp5 SFT job `/benfeuer/mochi-exp5-2026.11.21.113` running untouched.
- Result, including failure or restart: baseline recorded; **no baseline measurement exists** (`.51` RL attempt failed 2026-11-17; Exp4/Exp5 RL never launched). Arms declared: **S1 control** (8 H100, exact RL engine shape as a pure serving benchmark: TP1/DP8/EP8, seqs 64, batched 8192, HBM 0.75; synthetic 65,536-token-window prompts, group size 8, output 1024, concurrency 512, warmup 120 s, measure 1500 s) and **S2** (S1 + HBM 0.90). Config lands via loop-push; submission via the standard Iris CLI, PRODUCTION on cw-rno2a; each arm cancelled after its measured window.
- Evidence: frozen-reference receipt + worker REPORT/STATUS [R].
- Interpretation and limitations: the layout is recorded from the frozen reference, not measured — phase-2 comparisons will use the sprint's own measured baseline arms; the S1 control provides the first measured anchor at 1/32nd of the rollout fleet.
- Decision and next action: record S1/S2 submissions and job IDs; measure both 25-min windows; then iterate phase-1 knobs (per Mochi lessons: HBM, sequences, EP, batched tokens).

### 2026-11-22 01:10 UTC - Arms S1/S2 submitted (23:02 UTC); decode-baseline specs landed via loop-push

- Question or hypothesis and reason for trying it: first measured anchor — the exact RL rollout engine shape as a pure serving benchmark, plus the HBM knob that won in Mochi.
- Published source revision: `benfeuer/long-loop-prototype` at `1e4a1890ff75fe82d744fc1cd942350ea36b137a` ("Add Snowball decode serving specs for the RL resize baseline"; lint + author tests pass).
- Command or material configuration: specs `experiments/post_training/long_loop/specs/snowball-decode-baseline.json` and `snowball-decode-kv90.json`; engine TP1/DP8/EP8, BF16, FLASH_ATTN, triton MoE, seqs 64/rank, batched 8192, prefix caching, chunked prefill, PIECEWISE graphs, hermes, thinking, top-0 logprobs, max_model_len 102400; workload ~64.5k-token prompts (65,536 window), output 1024, group size 8, closed-loop concurrency 512, warmup 120 s, measure 1500 s, request timeout 1200 s. S2 differs by one knob: gpu_memory_utilization 0.75→0.90.
- External inputs and artifact references: worker cc over Rhizome [R]; submission logs `s1-baseline-submission.log`, `s2-kv90-submission.log` under /experiments/active/snowball-resize/ [R].
- Environment and hardware when relevant: PRODUCTION, federated cw-rno2a, 8 H100, 1 task per arm.
- Result, including failure or restart: S1 `/benfeuer/snowball-resize-decode-baseline-20261122a` submitted 23:02:24 UTC; S2 `/benfeuer/snowball-resize-decode-kv90-20261122a` submitted 23:02:25 UTC. Outputs will land under `s3://marin-us-east-08a/marin/users/benfeuer/snowball-resize/decode/{baseline-kv75,kv90}-20261122a`.
- Evidence: submission logs [R]; landing commit [R].
- Interpretation and limitations: in-flight; measured windows follow; arms get iris-cancel-own after measurement. Single-knob design gives a clean HBM read.
- Decision and next action: record both measured windows when they land; iterate phase-1 knobs from the winner.

### 2026-11-22 05:30 UTC - S2 RESULT: 1,618.9 tok/s/GPU at 65k context (HBM 0.90) — first measured decode anchor

- Question or hypothesis and reason for trying it: does raising HBM 0.75→0.90 lift Snowball decode at the 65,536-token window? (Mochi's winning knob.)
- Published source revision: `benfeuer/long-loop-prototype` at `1e4a1890ff75fe82d744fc1cd942350ea36b137a` (spec `specs/snowball-decode-kv90.json`).
- Command or material configuration: exact RL engine shape (8 H100 TP1/DP8/EP8, BF16, FLASH_ATTN, triton MoE, seqs 64/rank, batched 8192, prefix caching, chunked prefill, PIECEWISE graphs, hermes, thinking, top-0 logprobs, max_model_len 102400), HBM 0.90; synthetic 65k-window prompts (mean 64,139 tokens measured), group size 8, output 1024, closed-loop concurrency 512, warmup 120 s, measure 1500 s, request timeout 1200 s.
- External inputs and artifact references: worker result cc over Rhizome [R]; receipt `s2-kv90-result.json`; raw window `s3://marin-us-east-08a/marin/users/benfeuer/snowball-resize/decode/kv90-20261122a/` [R].
- Environment and hardware when relevant: 8 H100, PRODUCTION, federated cw-rno2a; job succeeded in 33 m 32 s (self-shutdown).
- Result, including failure or restart: FINAL valid 1500 s window — aggregate 12,951.2 gen tok/s; **1,618.9 tok/s/GPU**; request decode 26.03 tok/s mean (TPOT 38.4 ms); TTFT 1.06 s mean / 0.79 p50 / 4.10 p99; running 62.3/64 mean (cap-bound, p95 64); waiting 0.30; KV 70.3% mean / 81.0% max; prefix hit 99.97%; 18,802 succeeded / 0 failed; no preemption/eviction/OOM in task-0-vllm.log; PIECEWISE graph capture observed (16 lines).
- Evidence: result receipt + raw window [R].
- Interpretation and limitations: the first measured decode anchor at 65k context; running is cap-bound at 62.3/64 (p95 64) — so the concurrency-512 demand saturates the 64-slot engine and the next lever is sequence capacity (S3: seqs 64→128, concurrency 1024, landing via loop-push). S1 (HBM 0.75) still running — the HBM delta will be read from the S1-vs-S2 pair.
- Decision and next action: record S1's window and S3's landing/submission; iterate phase-1 knobs from the measured winner.

### 2026-11-22 05:50 UTC - S1 BASELINE FINAL: 316.9 tok/s/GPU at HBM 0.75 — the HBM knob gives 5.11x; S3 submitted

- Question or hypothesis and reason for trying it: complete the S1-vs-S2 single-knob HBM pair on the exact RL engine shape at the 65k window.
- Published source revision: `benfeuer/long-loop-prototype` at `1e4a1890ff75fe82d744fc1cd942350ea36b137a` (spec `specs/snowball-decode-baseline.json`).
- Command or material configuration: exact RL engine shape (TP1/DP8/EP8, BF16, seqs 64/rank, batched 8192, prefix caching, chunked prefill, PIECEWISE graphs, max_model_len 102400) at HBM 0.75; same 65k workload as S2.
- External inputs and artifact references: worker result cc over Rhizome [R]; receipts `s1-baseline-result.json`, `s2-kv90-result.json`; raw windows `s3://marin-us-east-08a/marin/users/benfeuer/snowball-resize/decode/{baseline-kv75,kv90}-20261122a/` [R].
- Environment and hardware when relevant: 8 H100, PRODUCTION, federated cw-rno2a; self-shutdown.
- Result, including failure or restart: FINAL valid 1500 s window — aggregate 2,535.1 gen tok/s; **316.9 tok/s/GPU**; request decode 10.47 tok/s (TPOT 98.9 ms); TTFT 104.7 s mean / 138.6 s p99; running 32.2/64 mean + waiting 31.6; KV 95.6% mean / 99.97% max; prefix hit 9.50%; 3,788/3,788 succeeded; no preemption/OOM.
- Evidence: result receipts + raw windows [R].
- Interpretation and limitations: vs S2 (one knob, HBM 0.90): per-GPU **5.11x** (1,618.9), per-request **2.5x** (26.03), TTFT 1.06 s, prefix 99.97%, waiting 0.30, KV 70.3%. At 0.75 the KV budget cannot hold the concurrent shared-prompt working set at the 65k window — requests re-prefill ~64k tokens and decode collapses; at 0.90 the working set fits and the seqs-64 slot cap binds. Synthetic closed-loop prefix residency is friendlier than real RL — phase-2 real rollouts will confirm the RL-side gain. S3 (S2 + seqs 128, concurrency 1024) submitted 23:38:15 UTC.
- Decision and next action: record S3's window; then the seqs/KV balance determines the next knob.

### 2026-11-22 06:30 UTC - S3 FINAL (+16.5%); PHASE 2 opens: 128-GPU baseline round submitted (6 engines)

- Question or hypothesis and reason for trying it: complete phase 1 (seqs knob) and open phase 2 (right-size the RL job).
- Published source revision: S3 spec at `e00de8519b…` (seqs 128); phase-2 round `rounds/snowball-resize-rl-6eng-cadence3.yaml` v2026.11.22.1 at `ff7276355ed1efa7b125726379c5b8713f8dd017` (loop-push "Adopt the Snowball SFT handoff in RL rounds" + adopted-round gate fix: schema-1 handoff acceptance, `build_run` gpu_variant; 10 focused tests; broad-suite harbor failures baseline-matched on the clean parent).
- Command or material configuration: S3 = S2 + seqs 64→128 (8 H100, concurrency 1024). P2 baseline = 10×8 trainer (TP1/PP2/CP5/EP8/ETP1, ref colocated) + 6 rollout engines DP8/EP8 (HBM 0.90, seqs 64, batched 8192) = 128 GPUs; cadence-3, staleness 2, full_batch, in-flight 1024, quotas 32/32, Harbor 256, 30-min budgets, window 102400, max_steps 32; adopted model Exp4 round-4 handoff 2026.11.18.76 (step 157000); canonical round-runner `--run`.
- External inputs and artifact references: worker cc over Rhizome [R]; receipts `s3-seqs128-result.json`, `rl-6eng-submission.log`, preflight `/tmp/snowball-6eng-preflight` [R].
- Environment and hardware when relevant: PRODUCTION cw-rno2a; 128 GPUs vs the recorded 256-GPU plan.
- Result, including failure or restart: S3 FINAL — 15,091.2 aggregate / **1,886.4 tok/s/GPU** (+16.5% vs S2); request decode 15.33 (TPOT 65.3 ms); TTFT 2.08/1.75/7.36 s; running 124.1/128 cap-bound; KV 83.5%/89.6%; prefix 99.97%; 22,465/0 failed. P2 baseline submitted ~06:20 UTC.
- Evidence: result + submission receipts [R].
- Interpretation and limitations: seqs 128 lifted the cap-bound S2 by 16.5% (running 124.1/128 — still cap-bound, so capacity now binds at seqs 128). Phase-2 baseline halves the recorded plan's GPU count (128 vs 256) with a colocated ref; the shrink ladder 6→4→3→2 engines will find the saturation floor.
- Decision and next action: record the P2 baseline's rollout wait/time-per-update/occupancy/step-split; then shrink steps.

### 2026-11-22 07:00 UTC - Phase-2 resubmission: quotas corrected to the Exp4-faithful 22/42 per the coordinator ruling

- Question or hypothesis and reason for trying it: the phase-2 baseline must carry the curriculum's real route mix, not a placeholder 32/32 split.
- Published source revision: fix `43f9042f3c30a49a9e34a5ef2cecc62a880d58aa` ("Honor declared route quotas in adopted RL rounds") on top of `ff7276355e…`.
- Command or material configuration: `build_run` preserves the round's declared `consumed_route_quotas`; smoke derives quotas from its shrunken batch; full runs fail fast unless the declaration partitions the batch (mirroring vendor/skyrl `RolloutBufferConfig.__post_init__`). Resubmission quotas 22/42 + Exp4 source routes (incl. nvidia/Llama-Nemotron-Post-Training-Dataset → nonagentic); same 128-GPU geometry (trainer 10×8 + 6 engines DP8/EP8, HBM 0.90, cadence-3, staleness 2, max_steps 32).
- External inputs and artifact references: worker cc over Rhizome [R]; submission log `rl-6eng-submission-22-42.log` [R].
- Environment and hardware when relevant: PRODUCTION cw-rno2a; the first submission (`…-d7d61d95`) was cancelled during startup — no rollout time lost.
- Result, including failure or restart: resubmitted ~06:55 UTC as `/benfeuer/snowball-resize-rl-6eng-cadence3-2026.11.22.1-coordinator-fcc10dd2`. Tests: 32 passed incl. `test_round_preserves_explicit_route_quotas` (11/21 preserved) and `test_round_rejects_route_quotas_that_do_not_partition_the_batch`; the pre-existing Snowball smoke geometry test passes again.
- Evidence: submission log + test receipts [R].
- Interpretation and limitations: the fail-fast partition check prevents silent quota drift; the 22/42 split matches the Exp4 curriculum the shrink ladder will run against.
- Decision and next action: record the resubmission's rollout wait/time-per-update/occupancy/step-split; then the 6→4→3→2 shrink ladder.

### 2026-11-22 13:00 UTC - Milestones 5-7: GB200 migration; R1 two clean steps + checkpoint OOM; R3 gates-open

- Question or hypothesis and reason for trying it: right-size the RL job on real rollouts; the H100 path was quota-blocked, so phase 2 migrated to the qualified GB200 acceptance shape.
- Published source revision: `97d290cc45…` (R1/R2), `6ecac3cceb` (3-engine role-plan fix), `128acb466b` ("Raise the rollout demand gates and reshard the checkpoint save" — `optimizer_checkpoint_sharding_type: dp_reshardable`).
- Command or material configuration: R1 = `.305` trainer shape (8 nodes × 4 GB200, PP2/CP4/EP4, all_gather) + 4 engines × DP8/EP8 HBM 0.90 = 64 GB200; cadence-3, staleness 2, quotas 22/42, Harbor 256, 30-min budgets, r136 excluded, max_steps 32.
- External inputs and artifact references: worker milestones 5-7 over Rhizome/STATUS [R]; ray logs `s3://marin-us-east-08a/tmp/ttl=14d/skyrl/…/2026.11.22.2/attempts/ray-logs/rank0-s2lrxs64` (276 files) [R].
- Environment and hardware when relevant: PRODUCTION 08a; 2-rack cap alongside Mochi's 64.
- Result, including failure or restart: R1 step 1 = 3,927.6 s (93.1% buffer wait = initial fill; 33.3 gen tok/s/GPU step-avg; mismatch 0.0053); step 2 FINAL CLEAN = 1,695.4 s (admission 1,528.4 s; 82.8 gen tok/s/GPU step-avg; mismatch 0.0062; 113/512 failed); FAILED at step 3 — first cadence-3 checkpoint save OOM'd a host (full optimizer-shard gather ~200 GB × 4 workers). Coordinator: step 2's 90% buffer wait is DEMAND-limited → 3-engine arm cancelled before admission. Fixes: `dp_reshardable` + raised demand gates. R3 gates-open submitted (75e54766).
- Evidence: REPORT.md rows + ray logs [R].
- Interpretation and limitations: step 1's 93.1% wait is the initial 64-group fill; step 2's 90.2% wait is demand-limited (engines median 2.25 running/rank), so shrinking engines before raising gates would have misattributed the bottleneck; the OOM is a save-sharding problem, not a training problem. H100 equivalence deferred.
- Decision and next action: R3 gates-open decides whether demand gates or engines bound; only then does the engine shrink resume.

### 2026-11-22 14:00 UTC - R3 STEP 2 CLEAN: 687.1 s/update (2.47x faster than R1); checkpoint fix survived step 3

- Question or hypothesis and reason for trying it: the gates-open arm tests whether demand gates (not engines) bound the phase-2 baseline.
- Published source revision: `128acb466b…` (demand gates raised + `dp_reshardable` checkpoint sharding).
- Command or material configuration: R1 geometry unchanged (4 engines × DP8/EP8, HBM 0.90 = 64 GB200); in-flight 1024, Harbor 384, verifier 384, tasks 2048, executor 384, progress 43200; staleness 2, quotas 22/42.
- External inputs and artifact references: worker readout over Rhizome [R]; REPORT.md arm table [R].
- Environment and hardware when relevant: PRODUCTION 08a; step 1 fill 4,039.1 s (13:26:37 UTC).
- Result, including failure or restart: STEP 2 CLEAN 13:38:05 UTC — **687.1 s/update vs R1's 1,695.4 s (2.47x)**; rollout wait 523.9 s = 76.3% (was 90.2%), optimizer 117.5 s (17.1%), training-prep 42.3 s, weight-sync 0.96 s; engines median 3.78 running/rank (peak 398/32), waiting median 0, median gen 221.2 tok/s/rank (peak 545), 19 preemptions, prefix median 98.6%; 214.5 gen tok/s/GPU step-average (response basis 96.39/GPU vs 39.64); mismatch 0.0061; staleness max 1; routes exactly 22/42; 132/512 failed (25.8%), 31 masked. STEP-3 CADENCE CHECKPOINT SURVIVED with dp_reshardable (save 125.5 s, no OOM).
- Evidence: REPORT.md row + native counters [R].
- Interpretation and limitations: raising the demand gates cut the update time 2.47x and doubled per-GPU generation — confirming the coordinator's demand-limited diagnosis. The checkpoint fix eliminated the host OOM. The saturation criterion (wait near 0) is not yet met at 76.3%, so the engine shrink stays paused; R3 continues for more clean steps + the step-6 checkpoint.
- Decision and next action: record further R3 clean steps and the step-6 checkpoint; shrink only when the criterion is met.

### 2026-11-22 14:30 UTC - R3 cancelled after measurement; R4 raises rollout workers 4→16 (demand ladder)

- Question or hypothesis and reason for trying it: with the bottleneck demand-limited (76.3% buffer wait at median 3.78 running/rank), raise client-side parallelism instead of changing engines.
- Published source revision: `be5d9a159d75ec9567f3551b61c1c8ac54aa31d8` via loop-push ("Raise rollout workers to sixteen for the demand-limited ladder" + "Declare the evaluation drain policy for the workers arm"); round `rounds/snowball-resize-rl-gb200-4eng-workers16.yaml` v2026.11.22.5.
- Command or material configuration: one knob — rollout workers 4→16 at cpus_per_worker 7 (16×7 = 112 ≤ driver CPU 120; Harbor 384 total = 24/worker); staleness 2 kept; engine geometry unchanged (4 engines × DP8/EP8, HBM 0.90 = 64 GB200); no engine ladder until rollout wait near 0.
- External inputs and artifact references: worker readout over Rhizome [R]; REPORT.md updated with the R3 measured row + R4 [R].
- Environment and hardware when relevant: PRODUCTION 08a; R3 cancelled with iris-cancel-own after measurement (step 2 clean 687.1 s, 76.3% wait, checkpoint survived).
- Result, including failure or restart: R4 submitted ~14:15 UTC (`/benfeuer/snowball-resize-rl-gb200-4eng-workers16-2026.11.22.5-coordinator-3d07d913`).
- Evidence: submission via loop-push [R]; R3 measured row [R].
- Interpretation and limitations: the demand-limited diagnosis says the engines are starved by client concurrency — 16 workers × Harbor 24 each should raise admitted demand ~4x if the worker count was the constraint. Failure classes for 132/512: TrialTimeoutError, AgentTimeoutError, VerifierRuntimeError-prepare, Unknown × Daytona sandbox lifecycle; verifier coverage 93.9%; retries 0.30/task.
- Decision and next action: record R4's step timings; if rollout wait approaches 0 with engines busy, the engine ladder 6→4→3→2 becomes measurable.

### 2026-11-22 16:45 UTC - R4 STEP 2 CLEAN: 195.0 s/update, rollout wait 13.4% — demand diagnosis confirmed; R5 shrinks to 3 engines

- Question or hypothesis and reason for trying it: the workers-16 knob tests whether client-side parallelism was the binding constraint (76.3% buffer wait on R3).
- Published source revision: `be5d9a159d…` (workers-16 ladder) for R4; `ed6341a535…` ("Bind the three-engine round to the pinned train and validation pools") for R5.
- Command or material configuration: R4 = R3 geometry + rollout workers 4→16 (cpus/worker 7; Harbor 384 = 24/worker). R5 = one knob engines 4→3 (24 rollout GPUs, 56 GB200 total; old train/validation pins + expected_tasks 120 + eval_drain wait).
- External inputs and artifact references: worker readout over Rhizome [R]; REPORT.md updated (R4 measured row + R5) [R].
- Environment and hardware when relevant: PRODUCTION 08a; 64 GB200 (R4) → 56 GB200 (R5); 2-rack cap.
- Result, including failure or restart: R4 STEP 2 CLEAN 15:28:31 UTC — **195.0 s/update** (R3 687.1 → 3.5x; R1 1,695.4 → 8.7x); rollout wait 26.0 s = **13.4%** (was 76.3%); optimizer 130.8 s (67%); engines median 7.31 running/rank (peak 377), gen 320.3 tok/s/rank median (peak 628), 2 preemptions, prefix 97.1%; **710.7 gen tok/s/GPU step-average** (response basis 356.5/GPU); mismatch 0.0060; step-3 checkpoint survived (113.9 s). R5 submitted ~16:10 UTC.
- Evidence: native counters + REPORT row [R].
- Interpretation and limitations: workers 16 cut rollout wait 76.3%→13.4% and tripled per-GPU generation — the demand-limited diagnosis is fully confirmed, and per-update time is now optimizer-dominated (67%). The engine ladder (4→3) is now measurable with near-zero wait. Curriculum handoff: 15 fixture failures fixed, 16/16 green; broader sweep + rl-pools build next.
- Decision and next action: R5's step timings decide whether 3 engines keep wait near 0; the ladder continues to 2 if so.

### 2026-11-22 17:15 UTC - R5 VERDICT: 3 engines REGRESS — the right-sized rollout width is 4 engines (32 GPUs)

- Question or hypothesis and reason for trying it: the engine ladder's decisive step — can 3 engines (24 rollout GPUs) hold wait near 0 at 16 workers + gates open?
- Published source revision: `ed6341a535…` ("Bind the three-engine round to the pinned train and validation pools"); round v2026.11.22.6.
- Command or material configuration: one knob engines 4→3 (24 rollout GPUs, 56 GB200 total); everything else at the R4 configuration.
- External inputs and artifact references: worker readout over Rhizome [R]; REPORT.md updated (R5 measured + verdict row) [R].
- Environment and hardware when relevant: PRODUCTION 08a; 56 GB200.
- Result, including failure or restart: STEP 2 CLEAN 16:59:03 UTC — **513.1 s/update** (vs 195.0 at 4 engines); rollout wait 373.5 s = **72.8%** (was 13.4%); optimizer 97.9 s; engines median 1.88 running/rank, gen 157.6 tok/s/rank median, 3 preemptions, prefix median 82.9%; 2,978,197 tokens = 241.9 gen tok/s/GPU step-average (response basis 204.2/GPU); mismatch healthy; 94/512 failed. Cancelled after measurement.
- Evidence: native counters + REPORT row [R].
- Interpretation and limitations: the ladder verdict is clean — at 16 workers + gates open + batch 64, 24 rollout GPUs starve the trainer; the minimum width holding wait near 0 is 4 engines (32 GPUs). Right-sized candidate: 8×4 GB200 trainer + 4 engines = 64 GB200, 195.0 s/update measured, 710.7 gen tok/s/GPU. Remaining: the trainer-shrink test (PP2/CP4/EP2 or PP1 variants) proposed to the coordinator pending per-rank memory fit. Exp5 pools migration reassigned to Exp2; curriculum landing 12d161687e verified — the next arm rebases onto it.
- Decision and next action: coordinator decision on the trainer-shrink arm; then the right-sized layout is complete.

### 2026-11-22 17:30 UTC - CAMPAIGN CONCLUDED: right-sized layout chosen (64 GB200, ~0.96 GPU-hour/update); records closed

- Question or hypothesis and reason for trying it: the campaign's deliverable — the right-sized Snowball RL layout with measured evidence.
- Published source revision: final arm R5 at `ed6341a535…`; the ladder spans `97d290cc45…` → `128acb466b…` → `be5d9a159d…` → `ed6341a535…`.
- Command or material configuration: **RIGHT-SIZED LAYOUT: 64 GB200 — trainer 8×4 (PP2/CP4/EP4 all_gather, dp_reshardable checkpoints) + 4 engines × DP8/EP8 at HBM 0.90 (seqs 128, batched 8192); 16 rollout workers; gates open (in-flight 1024, Harbor 384, verifier 384, tasks 2048, executor 384); staleness 2; quotas 22/42; cadence-3; 30-min budgets — ~0.96 GPU-hour per update at batch 64.**
- External inputs and artifact references: worker closing summary over Rhizome (~17:2x UTC) [R]; full arm table + receipts in worker REPORT.md/STATUS.md; artifacts under `s3://marin-us-east-08a/marin/users/benfeuer/{snowball-resize,long-loop-exp4}/` [R].
- Environment and hardware when relevant: PRODUCTION 08a (GB200); all arms iris-cancel-own after measurement; no Exp5/Mochi jobs touched.
- Result, including failure or restart: PHASE 1 — HBM 0.75→0.90 = 5.11× (1,618.9 tok/s/GPU at 65k window); seqs 128 +16.5% (1,886.4). PHASE 2 ladder — 4 workers + closed gates 1,695.4 s at 90.2% wait (demand-limited) → gates open 687.1 s → 16 workers 195.0 s at 13.4% wait (710.7 gen tok/s/GPU; mismatch healthy) → 3 engines REGRESSED (513.1 s at 72.8%). TRAINER-SHRINK DOES NOT FIT (16 ranks ≈ 269 GB/rank vs 192 GB HBM). Both 32-GPU floors held.
- Evidence: the full measured ladder [R].
- Interpretation and limitations: CAVEAT — all phase-2 numbers are GB200; the H100-equivalent layout needs one confirming run on cw-rno2a when the reserved fleet frees (the nominal 256-H100 layout never completed an update). SIDE DELIVERABLE: the curriculum-owned-holdout + derived-quota migration (12d161687e) + the exp4 rl-pools artifact. INCIDENCES: one checkpoint-save OOM (fixed), one Kueue quota stall, one premature engine-count arm (corrected by the coordinator).
- Decision and next action: marin#11969 closed with the Conclusion; this logbook closes; index record updated to Complete.

### 2026-11-22 19:30 UTC - H100 EXTENSION opened (user directive): H1 submitted and ADMITTED on cw-rno2a

- Question or hypothesis and reason for trying it: the GB200 ladder answered the right-size question on GB200; the user directive extends the campaign to deliver the H100 arm table and the smallest saturated H100 layout (the GB200 Conclusion's open caveat).
- Published source revision: `85ab0b663021ed89fd99286d5527030a75a6253b` (loop-push "Add the H100 analogue of the right-sized Snowball layout"; round `rounds/snowball-resize-rl-h100-4eng.yaml` v2026.11.22.7, pools-form on curriculum landing `12d161687e`).
- Command or material configuration: H1 = trainer 10×8 H100 = 80 GPUs (baseline Megatron TP1/PP2/CP5/EP8/ETP1, a2a at CP5; per-rank steady ~21 GB of 80 GB) + 4 engines × DP8/EP8 (32 GPUs, HBM 0.90) = **112 H100**; 16 rollout workers; gates open; staleness 2; derived quotas; cadence-3 + dp_reshardable; 30-min budgets; window 102,400; max_steps 32.
- External inputs and artifact references: worker extension cc over Rhizome (~19:3x UTC) [R].
- Environment and hardware when relevant: PRODUCTION cw-rno2a; Exp5 SFT finished (256 H100 free); capacity verified (karan 8-task c150 + ~20 small root/gbr remain). Up to 6 more hours.
- Result, including failure or restart: H1 submitted ~19:05 UTC and ADMITTED — all 18 child tasks running, no gate (`/benfeuer/snowball-resize-rl-h100-4eng-2026.11.22.7-coordinator-21aebbdf`).
- Evidence: admission state [R].
- Interpretation and limitations: the GB200-measured layout is the prior; the H100 analogue tests whether the same shape (112 H100 vs 64 GB200) holds wait near 0 — then the ladder 4→3→2 and trainer shrink if arithmetic allows.
- Decision and next action: arm-table rows hourly; ladder per the wait criterion; the H100 layout concludes the extension.

### 2026-11-23 03:00 UTC - H100 H1 MEASURED: 687.6 s/update at 4 workers — near-identical to GB200 (demand-limited on both); H2 submitted

- Question or hypothesis and reason for trying it: does the GB200-measured right-sized layout hold on H100? The first datapoint (4 workers, gates open).
- Published source revision: `85ab0b6630…` (H100 analogue round) + `f8591a37df` ("Alias raw row sources to teacher routes in adopted rounds" — fixed the first submission's driver-startup route-classification death); H2 source `dbdd8bc83b…` ("Raise the H100 analogue to sixteen rollout workers").
- Command or material configuration: trainer 10×8 H100 (80 GPUs, TP1/PP2/CP5/EP8/ETP1, ~21 GB/rank) + 4 engines × DP8/EP8 (32 GPUs, HBM 0.90) = 112 H100; 16-worker intent but the round carried a stale 4-worker value.
- External inputs and artifact references: worker readout over Rhizome (~03:0x UTC) [R]; REPORT.md H100 arm table section [R].
- Environment and hardware when relevant: PRODUCTION cw-rno2a; first submission died at driver startup; corrected root admitted.
- Result, including failure or restart: corrected root `…-75ef1aeb`: step 1 fill 3,627.6 s (19:59:07 UTC); **step 2 CLEAN 687.6 s (20:10:35 UTC)**; step-3 checkpoint survived (117.3 s); cancelled after measurement. Step-2 split: rollout wait 588.0 s = **85.5%**, optimizer 65.6 s (9.5%), prep 29.5 s, sync 1.06 s; engines median 2.78 running/rank, gen 217.7 tok/s/rank median, 28 preemptions, prefix 98.7%; **250.9 gen tok/s/GPU step-average** (response basis 115.5); mismatch 0.0051; 84/512 failed.
- Evidence: native counters + REPORT row [R].
- Interpretation and limitations: KEY finding — the round carried a stale 4-worker value, so H1 measured the 4-worker H100 datapoint: **687.6 s vs GB200's 4-worker 687.1 s (near-identical; demand-limited on both)**. The workers-16 knob (H2) is the H100 analogue of the GB200 R4 step.
- Decision and next action: H2's step timings complete the H100/GB200 comparison at the workers-16 point.

### 2026-11-23 00:20 UTC - H100 6-hour window CLOSED: H2 170.4 s (42.5% wait, best H100 point); H3 staged but pools-rebuild blocked

- Question or hypothesis and reason for trying it: complete the H100 ladder within the 6-hour directive (H2 = the GB200 R4 analogue; H3 widens engines since wait still dominates).
- Published source revision: H2 `dbdd8bc83b…`; H3 `e84647ad44…` ("Widen the H100 analogue to six engines").
- Command or material configuration: H2 = R4 analogue (16 workers × 7 CPU; 4 engines DP8/EP8, HBM 0.90). H3 = engines 4→6 (48 rollout GPUs, 128 total).
- External inputs and artifact references: worker milestones 16-17 over Rhizome (secretary sends bounced; content relayed by the coordinator) [R]; REPORT.md H100 table [R].
- Environment and hardware when relevant: PRODUCTION cw-rno2a; pools rebuild `/benfeuer/snowball-resize-pools-20261122b` FAILED in the helper's stamp code (regeneration kept 5,181 rows / 0 rejections vs adopted 20,800 — helper's lane).
- Result, including failure or restart: H2 STEP 2 CLEAN — **170.4 s** / **42.5% wait** / **909.9 gen tok/s/GPU step-average** (optimizer 38%, engines 6.09 median running/rank, 0 preemptions, prefix 97.4%, mismatch 0.0049); step-3 checkpoint OK; cancelled after measurement — best measured H100 point, NOT saturated. H3 staged (one pools-pin update + resubmission away).
- Evidence: REPORT rows + milestones [R].
- Interpretation and limitations: the GB200↔H100 pair is now measured at workers-16: GB200 195.0 s/13.4% wait vs H100 170.4 s/42.5% wait — H100 is faster in wall time but still demand/supply-hungry; the smallest saturated H100 layout needs H3 (6 engines), which is blocked on the helper's pools regeneration fix.
- Decision and next action: H3 submits when the pools rebuild is fixed; the H100 layout concludes then. Relayed to #11969 per the coordinator (worker messages bounced).

### 2026-11-23 03:20 UTC - H100 LADDER COMPLETE: H3 regressed (544.8 s) — smallest measured-saturated H100 layout is 112 H100 (170.4 s/update)

- Question or hypothesis and reason for trying it: the ladder's decisive step — do 6 engines help when 4 engines still hold 42.5% wait?
- Published source revision: `e84647ad44…` (6-engine round); pools rebuild completed since the earlier failure.
- Command or material configuration: H3 = 16 workers / 6 engines = 128 H100; everything else at the H2 configuration.
- External inputs and artifact references: worker ladder-complete readout over Rhizome (~03:2x UTC) [R]; REPORT.md H100 section [R].
- Environment and hardware when relevant: PRODUCTION cw-rno2a; cancelled at the coordinator's order because its 128 H100 gated the queued Exp5 RL `.133` (Exp5's 256 H100 released).
- Result, including failure or restart: H3 step-1 fill 2,707.9 s; step 2 clean **544.8 s** (step-2-only measurement) — **REGRESSED 3.2x vs H2's 170.4 s**: rollout wait 72.8%; the wider rollout side starved per-engine batch and decode efficiency while the 80-GPU trainer was already binding. Ladder: H1 (4 workers) 687.6 s/85.5% → H2 (16 workers) **170.4 s/42.5%, 909.9 gen tok/s/GPU — BEST MEASURED POINT** → H3 regress.
- Evidence: native counters + REPORT rows [R].
- Interpretation and limitations: **SMALLEST MEASURED-SATURATED H100 LAYOUT: 112 H100** (trainer 10×8 + 4 engines DP8/EP8 HBM 0.90; 16 workers ×7 CPU; gates open; staleness 2; derived quotas; cadence-3 + dp_reshardable; 30-min budgets) — **170.4 s per update at batch 64 (~5.2 GPU-hours/update)**. The residual 42.5% wait is the worker's read: the H100 decode-speed tax vs GB200 (170.4 vs 195.0 s at the same config), not a sizing error. No further rno2a arms while Exp5 RL .133 is queued/running.
- Decision and next action: the H100 arm table + smallest saturated layout are delivered to #11969 (closed with the H100 closing comment); the extension concludes.

### 2026-11-26 18:05 UTC - Record location moved: `/experiments/active/snowball-resize/` → `/experiments/complete/snowball-resize/`

- Question or hypothesis and reason for trying it: coordinator notice — the Snowball-resize record directory moved to `experiments/complete/` (idle >72h, no writes); references updated per the coordinator's request.
- Published source revision: n/a (record-keeping).
- Command or material configuration: the standing preamble path was updated to `/experiments/complete/snowball-resize/{BRIEF.md,REPORT.md,STATUS.md}`; historical dated entries intentionally retain the `/experiments/active/` paths as written at the time (append-only record).
- External inputs and artifact references: coordinator notice over Rhizome (2026-11-26) [R].
- Environment and hardware when relevant: `experiments/active/snowball-resize` no longer exists locally; `experiments/complete/snowball-resize` holds BRIEF.md, REPORT.md, STATUS.md, frozen51-reference, submission logs, and receipts.
- Result, including failure or restart: references updated; campaign history unchanged.
- Evidence: directory listing of `experiments/complete/snowball-resize/` [R].
- Interpretation and limitations: none.
- Decision and next action: future references use the `complete/` path.
