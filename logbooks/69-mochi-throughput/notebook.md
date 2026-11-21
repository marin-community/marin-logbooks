---
title: "Mochi RL throughput campaign"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/69"
related_issues:
  - "https://github.com/marin-community/marin/issues/11954"
research_branch: "codex/69-mochi-throughput"
marin_base_revision: "cb991a5a5c8af4d670d56757eeae3e59edd4a778"
archive_template_revision: "fc1cb30da0967f6ab78ffd47444ed9144f50cba9"
status: active
---

# Mochi RL throughput campaign

Secretary record (evidence-backed logbook) for the 6-hour Mochi decode/rollout throughput sprint that selects the configuration for the first real Mochi long-loop RL training run. The experiment is owned by a Codex worker (session `01a1fc35-78a2-7110-9b2f-eb37a6567faf`) whose primary records are `/experiments/active/mochi-throughput/{BRIEF.md,REPORT.md,STATUS.md}` (the REPORT.md LIVE-SPRINT table is the authoritative arm table). Open-Athena reproduction details — per-arm branches/commits, round configs, launch commands, Iris job IDs, S3 receipt paths with Filer links, finelog counter method, and the full results table — are tracked in the user-authorized single marin issue [marin#11954](https://github.com/marin-community/marin/issues/11954).

## Question and decision criteria

- Goal and why it matters: choose the engine and admission configuration for the first real Mochi (535B-A23B) long-loop RL training run. The long-loop plan needs ~50B SFT tokens in a 20 h stage plus RL rounds; Mochi currently decodes ~10 tok/s per request and ~25 tok/s per GB200 at eval concurrency, and SFT trains ~143–177k tok/s per rack (marin#8089).
- Baseline and alternatives: the v103 default RL configuration versus measured sweeps of engine knobs (max_num_seqs, HBM/KV utilization, expert parallelism, shared-prefix layout, batched tokens) and admission knobs (inflight workers, max_staleness_steps, Harbor slots), on two lanes — real-RL rollout arms (48 trainer + 12 rollout GB200) and pure-decode capacity arms (no trainer).
- Evidence needed to choose: per-arm config diff, measurement window, aggregate and per-GPU decode tok/s, per-request tok/s, running/engine, KV mean/max; native Finelog counter windows where available; a code trace of rollout concurrency; the user's rulings.
- Constraints and unresolved questions: config changes only, no engine code changes; the Daytona RL org is shared (one job ~1/4–1/3 of it); A-arm capacity ≤ 2 racks; rapid measure-then-cancel iteration. Unresolved until the sprint ends: whether cache-through expert-parallel layouts (EP16) transfer to real training, and the final training-config selection.

## Baseline

- Date: 2026-11-21.
- Published source revision: branch `benfeuer/long-loop-prototype` (round configs under `experiments/post_training/long_loop/rounds/mochi-*.yaml`, specs under `specs/mochi-decode-*.json`) and its sprint branch `benfeuer/mochi-throughput`; per-arm revisions `b0a9727622d54a75101c455f3e304d6ae3003565` (v103), `46063946b49d9d677f5aa0c94a26f27715550764` (`.104` seqs32), `3a859712f51d3ddff46748e850208e8315dc53da` (`.105`–`.107` + first EP16 pair), `e0bf304558d4fdbf3b41a9b68285918eb19ca1ca` (`.108` Harbor1024), `c6515fbdc7bc2b51f0f43649dac08ee54fa3397d` (`.109` Harbor384), `3f4a4c68ce4fad43d1a639d1d390d1ae3e3649b1` (latest B pair), `e9a16e37ce65754192208551df44e972e8c9de7b` (pure-decode implementation).
- Configuration, command, and external inputs: cluster `cw-us-east-08a`, PRODUCTION priority, GB200, r136 hard-excluded, one-NVLink-gang placement, BF16, FLASH_ATTN + triton MoE, prefix caching, compilation + PIECEWISE CUDA graphs. Real-RL arms launch via the round runner (`uv run --no-sync python -m experiments.post_training.long_loop.round_runner --round <yaml> --source-revision <sha> --validation-output <path> --through rl --run`); pure-decode arms via `uv run --no-sync python -m experiments.post_training.long_loop.throughput` with a pinned spec JSON. Counters come from the Finelog sink of SkyRL's `inference_stats` callback (`vendor/skyrl/skyrl-train/skyrl_train/callbacks/builtin.py:1253`, `:1108`; `inference_observability.py:130`), read as reset-safe cumulative deltas over fixed windows.
- Environment and hardware when relevant: GB200 racks; A arms 48 trainer + 12 rollout GPUs (A3/A4: 60-GPU jobs), B decode arms 12–16 GPUs; Nemotron reference lane up to 64 GB200.
- Result and evidence: see the sprint-start entry below and marin#11954 for the full table.
- Limitations: A1–A3 rates are console-sample estimates (the recipes omitted the `inference_stats` callback, so native `generation_tokens_total` counters were absent); fixed-pool serving rates are cache-reuse upper bounds, and only real-rollout (A) arms establish the training gain.

## Rollout concurrency (code trace)

- `TaskRolloutWorker` holds `task_slots` and `harbor_task_slots` semaphores (`vendor/skyrl/skyrl-train/skyrl_train/rollouts/task_worker.py:390-391`), acquired per task (`:548-552`).
- Harbor `n_concurrent_trials` is consumed at `vendor/skyrl/skyrl-train/skyrl_train/inline_harbor_client.py:119`.
- Documented rule: slots = Harbor `n_concurrent_trials` (16 when unset) // num workers (`lib/skyrlrecipe/src/skyrlrecipe/sections.py:458`); config key `terminal_bench.harbor.n_concurrent_trials` (`experiments/post_training/long_loop/stage_config.py:38`).
- Consumed route quotas require `batch_policy: full_batch` (`vendor/skyrl/skyrl-train/skyrl_train/rollouts/buffer.py:49,72`; `lib/skyrlrecipe/src/skyrlrecipe/sections.py:440,444`).

## User rulings

- Config changes only — no engine code changes ("config over code").
- The Daytona RL org is shared; one job consumes roughly 1/4–1/3 of it; keep A-arm capacity ≤ 2 racks.
- Rapid iteration: measure a 25-min window (or one valid counter window) per arm, then cancel; do not wait for update or checkpoint completion.

## Entries

### 2026-11-21 17:20 UTC - Campaign start: sprint state and measured arms at record open

- Question or hypothesis and reason for trying it: open the secretary record for the Mochi RL throughput campaign (BRIEF addition ~12:05 PM EST) and record every arm measured so far, the rollout-concurrency trace, and the user's rulings.
- Published source revision: sprint sources as listed in Baseline; latest landings `c6515fbdc7bc2b51f0f43649dac08ee54fa3397d` (`.109` Harbor384, loop-push) and `3f4a4c68ce4fad43d1a639d1d390d1ae3e3649b1` (B pair).
- Command or material configuration: arms as in the table below; all jobs PRODUCTION on `cw-us-east-08a`, r136 excluded, graphs on.
- External inputs and artifact references: worker REPORT.md/STATUS.md [R]; worker readouts over Rhizome (14:0x–17:1x UTC) [R]; marin#11954 (repro detail).
- Environment and hardware when relevant: A arms 60-GPU jobs (12 rollout GPUs counted for per-GPU rates); B decode arms 12–16 GPUs; ~92 GPUs steady at 17:15 UTC including the in-flight B pair.
- Result, including failure or restart (rates marked * are console-sample estimates; native Finelog counter windows otherwise; "running" = running requests per engine; KV mean/max):

| Arm (change) | Source | GPUs | Window | Agg tok/s | tok/s/GPU | Req tok/s | Running/engine | KV mean/max | Outcome |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| `.103` v103 control | `b0a9727622` | 12 rollout | first-25min | 670.82* | 55.90* | n/a | 11.69/16 | 72.11%/– | cancelled |
| A1 inflight128 (`.105`) | `3a859712f5` | 12 rollout | first-25min | 1361.78* | 113.48* | n/a | 13.71/16 | 66.57%/– | cancelled |
| A1 older late window | `3a859712f5` | 12 rollout | late-25min | 402.54* | 33.54* | n/a | 7.14/16 | 72.64%/– | different run age — not an aligned A/B denominator |
| A2 staleness11 (`.106`) | `3a859712f5` | 12 rollout | first-25min | 1454.57* | 121.21* | n/a | 15.67/16 | 61.73%/– | cancelled; latest-25min 1164.60*/97.05*, 13.43/16, 79.07% |
| A3 engine32/KV85 (`.107`) | `3a859712f5` | 12 rollout | first-25min | 1526.61* | 127.22* | n/a | 30.45/32 | 72.47%/– | measured, cancelled; current training candidate |
| A4 Harbor1024 (`.108`) | `e0bf304558` | — | — | — | — | — | — | — | cancelled before measurement (shared-quota rule) |
| A4 Harbor384 (`.109`) | `c6515fbdc7` | 60 | pending | — | — | — | — | — | in flight; native counters on |
| B control seqs16 | `e9a16e37ce` | 12 | final | 2487.54 | 207.30 | 13.45 | 15.69/16 | 88.35%/98.46% | cancelled after valid window |
| B seqs32/HBM.75 (`.104` change) | `46063946b4` | 12 | final | 2153.76 | 179.48 | 10.84 | 16.73/32 | 94.35%/99.93% | cancelled after valid window |
| B KV85 | `e9a16e37ce` | 12 | final | 4004.60 | 333.72 | 11.09 | 30.79/32 | 81.20%/93.14% | succeeded |
| B batch16384 | `e9a16e37ce` | 12 | final | 2923.21 | 243.60 | 12.32 | 20.27/32 | 94.06%/100% | succeeded |
| B shared8/seqs32 | `e9a16e37ce` | 12 | final | 4734.29 | 394.52 | 12.95 | 30.69/32 | 64.01%/88.11% | succeeded |
| B shared8 short/seqs32 | `e9a16e37ce` | 12 | final | 4557.53 | 379.79 | 12.30 | 30.81/32 | 16.02%/30.32% | succeeded |
| B shared8/seqs64 | `e9a16e37ce` | 12 | final | 5394.30 | 449.53 | 10.09 | 44.36/64 | 94.71%/100% | succeeded; 13.94% over shared32 but KV 100%, prefix hits 52.89%→36.46%, TTFT 4.53→42.98 s |
| B shared8 short/seqs64 | `e9a16e37ce` | 12 | — | — | — | — | — | — | startup failed (EngineCore RuntimeError → NCCL store errors); no valid rate |
| B EP16/seqs64/shared8 | `3a859712f5` | 16 | long window | 8834.35 | 552.15 | 9.27 | 59.79/64 | 85.43%/– | succeeded |
| B seqs128/shared8 short | `3a859712f5` | 12 | short window | 15857.07 | 1321.42 | 10.92 | 122.31/128 | 65.12%/– | succeeded |
| B EP16/seqs128/shared8 | `3f4a4c68ce` | 16 | pending | — | — | — | — | — | in flight |
| B EP16/seqs64/batch4096/shared8 | `3f4a4c68ce` | 16 | pending | — | — | — | — | — | in flight |

- Evidence: receipts `a3-first25-analysis.json`, `pure-ep16-seqs64-result.json`, `pure-seqs128-short-result.json`, `decode-control-live-samples-1308.jsonl` + `analyze_decode_live.py` (control window 12:58:49–13:07:52 UTC: 1,341,242 generated-token increments over 542.47 s = 2,472.48 tok/s aggregate), and the per-arm Iris job IDs / S3 receipt paths catalogued in marin#11954.
- Interpretation and limitations: best real-rollout arm is A3 (`engine32-kv85`: inflight128, staleness11, seqs32, HBM 0.85) at 127.22 tok/s/GPU first-25min — 2.28x the v103 control — and it is the current training candidate. Best pure-decode arm is EP16/seqs64/shared8 at 552.15 tok/s/GPU; the seqs128 short window (1321.42 tok/s/GPU, 12 GPUs) is a short-window serving bound, not a training rate. Fixed-pool serving rates remain cache-reuse upper bounds; only real-rollout arms establish the training gain. A1–A3 native counters were absent (callback omission), so their rates are console-sample estimates; the `inference_stats` callback is on from `.109` onward.
- Decision and next action: training selection 19:30 UTC (14:30 EST); real Mochi RL training launch ~19:45 UTC on the best measured config; keep this record and marin#11954 updated as arms land; close both when the real run is verified past its first update or the user ends the campaign.

## Final decision

- Selected approach and evidence:
- Rejected alternatives and why:
- Marin contribution PRs:
- Reproduction instructions:
- Remaining uncertainty:

### 2026-11-21 17:25 UTC - Client-cap measurement on A3; Harbor384 in flight

- Question or hypothesis and reason for trying it: is the rollout client (coordinator/workers) the bottleneck at the measured admission level, and what admission headroom remains?
- Published source revision: A3 at `3a859712f51d3ddff46748e850208e8315dc53da`; `.109` Harbor384 at `c6515fbdc7bc2b51f0f43649dac08ee54fa3397d`.
- Command or material configuration: RayState inspection plus py-spy sampling of the coordinator and 32 workers; Harbor384 config `experiments/post_training/long_loop/rounds/mochi-throughput-harbor384.yaml` (Harbor 256→384 only, 32 workers).
- External inputs and artifact references: worker readout over Rhizome (~17:2x UTC) [R]; REPORT.md client-cap sections and profiles [R].
- Environment and hardware when relevant: A3 job 60 GB200 (12 rollout); 144-CPU hard-pinned head per A job.
- Result, including failure or restart: A3 RayState showed 128 RUNNING group calls across 32 workers, up to 1024 scheduled trajectories (exact executing count unavailable; queued/completed siblings included). First-25min occupancy 30.45/32 running + 5.94 waiting/engine; at 16:48 UTC, 430 pending model requests with 29.58 running + 5.75 waiting/engine. CPU: coordinator 0.49% of one core; 32 workers summed 0.77 cores in a 2 s sample; py-spy main threads in epoll wait; loop-lag mean 5.51 ms (max 874 ms at startup) — no CPU/GIL saturation evidence. Harbor1024 (`.108`) had been cancelled on the 11:55 shared-quota rule (375–500 vCPU/job); Harbor384 (`.109`) is submitted at `/benfeuer/long-loop-mochi-throughput-harbor384-2026.11.21.109-coordinator-16513921` with native `inference_stats` counters on.
- Evidence: RayState counts and py-spy profiles in REPORT.md [R]; worker readout [R].
- Interpretation and limitations: the client is not the bottleneck at this scale; the binding constraints are engine-side occupancy and the 128-group-call schedule (1024 scheduled trajectories). A 128-worker client arm needs 1 CPU/worker and a 136-CPU allocation because 128×2 exceeds the 144-CPU head, and two 60-GPU A arms plus the running B pair would exceed 128 GPUs, so `.109`'s window is retained before replacing it.
- Decision and next action: prepare the 128-worker client arm with Harbor384 fixed; no staleness/inflight changes or coordinator code fix are justified by current evidence.

### 2026-11-21 17:35 UTC - 128-worker client arm landed and validated (`.110`)

- Question or hypothesis and reason for trying it: test client admission headroom above the 32-worker level, per the 17:25 entry's plan.
- Published source revision: `7f245b01735ced953e3ce490db8bca9159bc0d1f` (`.110`), landed via loop-push; 5 adopted-round tests, lint and types pass.
- Command or material configuration: `experiments/post_training/long_loop/rounds/mochi-throughput-workers128.yaml` — Harbor384 fixed; workers 32→128; CPU/worker 2→1; allocation 120→136 (required by the 144-CPU hard-pinned head; a declared-resource confound).
- External inputs and artifact references: worker readout over Rhizome (~17:3x UTC) [R]; validation receipt `sprint-workers128-validation/*/receipt.json`.
- Environment and hardware when relevant: not launched yet — running it alongside the B pair would exceed the 128-GPU cap.
- Result, including failure or restart: preflight/validation passed; no measurement yet.
- Evidence: validation receipt [R]; worker readout [R].
- Interpretation and limitations: the declared-resource change (1 CPU/worker, 136-CPU allocation) is a confound to keep in mind when comparing against the 32-worker arm; the arm is ready to `--run` immediately after Harbor384's (`.109`) measurement/cancellation.
- Decision and next action: launch `.110` after `.109`'s window; no further staleness/inflight changes; keep #11954 and this record updated.

### 2026-11-21 17:40 UTC - Correction: A2-vs-A1 is not a staleness gain; candidate restores default staleness 3

- Question or hypothesis and reason for trying it: the 17:20 entry and the worker REPORT treated the A2-vs-A1 difference as a staleness effect; the worker accepted a correction that invalidates that causal reading.
- Published source revision: unchanged — all A measurement windows were still at step 1 with a non-binding staleness ceiling.
- Command or material configuration: no launch changes; `.109` (Harbor384) keeps its launch config; the unlaunched `.110` (workers128) recipe is being re-landed with the default staleness 3 and its native preflight refreshed.
- External inputs and artifact references: worker correction over Rhizome (~17:4x UTC) [R]; REPORT/STATUS in /experiments/active/mochi-throughput/ [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: the A2-vs-A1 causal claim is withdrawn; the raw rates in the 17:20 table are retained as noise/age observations (no table rows deleted). The current training candidate restores `max_staleness_steps` 3 (default) alongside inflight128, seqs32, HBM 0.85.
- Evidence: step-1 state of all A windows and the non-binding ceiling [R].
- Interpretation and limitations: no staleness benefit is claimed for any A arm, including `.109`; no further staleness arms will run — focus moves to client caps/GIL.
- Decision and next action: candidate = inflight128 + default staleness 3 + seqs32 + HBM 0.85; continue with the client-cap plan (`.110` after `.109`'s window); training selection 19:30 UTC.
