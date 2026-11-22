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
