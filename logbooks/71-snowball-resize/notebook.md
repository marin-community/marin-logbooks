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
status: active
---

# Snowball RL resizing campaign

Secretary record (evidence-backed logbook) for the ~12-hour Snowball RL resizing sprint: phase 1 maximizes Snowball decode throughput at the 65,536-token request window; phase 2 right-sizes the Snowball RL job (rollout + trainer GPUs, engine config, gates) to the smallest allocation that keeps training saturated. The experiment is owned by a fresh OpenCode worker whose primary records are `/experiments/active/snowball-resize/{BRIEF.md,REPORT.md,STATUS.md}` (the REPORT.md top table is authoritative once arms land). Open-Athena reproduction details — per-arm branches/commits, config paths, launch commands, Iris job IDs, S3 receipt paths with Filer links, finelog queries, and the results table — are tracked in the user-authorized single marin issue [marin#11969](https://github.com/marin-community/marin/issues/11969), which links the prior Mochi campaign (#11954, closed) and the concurrency-gate table (#11957).

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

- Selected approach and evidence:
- Rejected alternatives and why:
- Marin contribution PRs:
- Reproduction instructions:
- Remaining uncertainty:

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
