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

- `TaskRolloutWorker` holds `task_slots` and `harbor_task_slots` semaphores (`vendor/skyrl/skyrl-train/skyrl_train/rollouts/task_worker.py:390-391`), acquired per task (`:548-552`); the Harbor semaphore wraps the entire `engine.run_task` (verified 2026-11-21 ~18:20 UTC).
- Harbor `n_concurrent_trials` is consumed at `vendor/skyrl/skyrl-train/skyrl_train/inline_harbor_client.py:119`.
- Documented rule: slots = Harbor `n_concurrent_trials` (16 when unset) // num workers (`lib/skyrlrecipe/src/skyrlrecipe/sections.py:458`); config key `terminal_bench.harbor.n_concurrent_trials` (`experiments/post_training/long_loop/stage_config.py:38`). For the 256-slot config that is 8/worker; Harbor384 gives 12/worker.
- Precision (verified): the 8-slot semaphore is one group's worth, not an atomic group lock — partial groups can interleave; dispatch `min(_pending)` has no Harbor-slot awareness; the generic 1024/worker task slots are unbinding.
- Measurement limit (verified): no explicit Harbor-semaphore wait logs are emitted, and A5's bounded logs contain 0 direct slot-wait records, so actual Harbor occupancy is NOT measured; A3's 128 group calls / 430 pending model requests do not count Harbor occupancy.
- Staleness admission (verified): `FullBatch.lease_batch` uses `range(step, step + staleness + 1)`, so staleness can change admission during step 1 even without a policy update — this corrects an earlier blanket "never binds" statement; the A2 causal-gain claim remains withdrawn and default staleness 3 is kept per direction.
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

### 2026-11-21 17:50 UTC - Staleness correction complete in source; `.110` re-landed at default 3

- Question or hypothesis and reason for trying it: complete the 17:40 correction by landing the default-staleness configuration for the unlaunched client arm.
- Published source revision: `8aee017ea5cfcd9748c372fc8555be2c8483f757` (loop-push; 5 adopted-round tests, lint, types pass).
- Command or material configuration: `.110` `rounds/mochi-throughput-workers128.yaml` now carries default `max_staleness_steps` 3; exact native preflight passed with compiled 3 confirmed.
- External inputs and artifact references: worker readout over Rhizome (~17:5x UTC) [R]; receipt `sprint-workers128-default3-validation/*/receipt.json`; REPORT/STATUS in /experiments/active/mochi-throughput/ [R].
- Environment and hardware when relevant: still unlaunched; GPU-cap sequencing unchanged (client arm after the running A arm's measurement window).
- Result, including failure or restart: correction landed; REPORT withdraws the A2 causal-gain claim; A3's actual-11 provenance is kept without a staleness-benefit claim; training candidate is staleness 3.
- Evidence: native preflight receipt [R]; worker readout [R].
- Interpretation and limitations: the `.110` declared-resource confound (1 CPU/worker, 136-CPU allocation) remains to keep in mind at comparison time; the running `.109` keeps its launch config (staleness 11 in its actual config) with no benefit claimed.
- Decision and next action: launch `.110` after the running A arm's measurement; no staleness arms; training selection 19:30 UTC.

### 2026-11-21 17:58 UTC - A5/Harbor384 GPU child identified; B pair weights loaded; no new measurement yet

- Question or hypothesis and reason for trying it: track the running arms' startup state between measurement windows.
- Published source revision: `.109` running at `c6515fbdc7bc2b51f0f43649dac08ee54fa3397d`; next client arm `.110` ready at `8aee017ea5cfcd9748c372fc8555be2c8483f757` (default staleness 3).
- Command or material configuration: startup-receipt inspection of the `.109` root job.
- External inputs and artifact references: worker readout over Rhizome (~17:5x UTC) [R]; bounded Finelog receipts `sprint-current-startup-1709.jsonl`, `sprint-b-live-1709.jsonl`, `sprint-a5-live-1709.jsonl` in /experiments/active/mochi-throughput/ [R].
- Environment and hardware when relevant: `.109` GPU child `/benfeuer/long-loop-mochi-throughput-harbor384-2026.11.21.109-coordinator-16513921/long-loop-mochi-throughput-harbor384-2026.11.21.109-ade8d45b80bb` — 48 trainer + 12 rollout GB200.
- Result, including failure or restart: A5 engine init at 17:09 UTC with no generation counter window yet; both B EP16 arms (seqs128/shared8; seqs64/batch4096/shared8) loaded weights ~17:07–17:09 UTC. No measured new arm to cancel.
- Evidence: startup receipt + Finelog startup samples [R].
- Interpretation and limitations: startup state only — no throughput claim; the next client arm (`.110`) waits for the running A arm's window (GPU cap 128).
- Decision and next action: prepare the B HBM 0.90 arm against the completed EP16/seqs64 (0.85); record `.109`'s measured window when it lands.

### 2026-11-21 18:05 UTC - B HBM 0.90 arm submitted (EP16/seqs64/kv90/shared8)

- Question or hypothesis and reason for trying it: does raising HBM/KV utilization 0.85→0.90 lift the EP16/seqs64/shared8 decode rate (552.15 tok/s/GPU baseline) further?
- Published source revision: `530ab73b76a28a3e15a741d8f946a0dfaeea8818` (loop-push; 7 serving tests, lint pass).
- Command or material configuration: spec `experiments/post_training/long_loop/specs/mochi-decode-ep16-seqs64-kv90-shared8.json` — only change HBM 0.85→0.90 versus the completed EP16/seqs64/shared8 arm; 16 GB200, PRODUCTION 08a, r136 excluded, graphs on.
- External inputs and artifact references: worker readout over Rhizome (~18:0x UTC) [R]; submission receipt `sprint-ep16-kv90-submission.log`; output `s3://marin-us-east-08a/marin/users/benfeuer/mochi-throughput/decode/ep16-seqs64-kv90-shared8-20261121a`.
- Environment and hardware when relevant: capacity A60 + B16×3 = 108 ≤ 128 GPUs, so the arm launched in parallel rather than waiting for a slot.
- Result, including failure or restart: submitted; no measured window yet (the 17:09-start arms are still in startup; nothing to cancel).
- Evidence: submission receipt [R]; job `/benfeuer/mochi-throughput-decode-ep16-seqs64-kv90-shared8-20261121a`.
- Interpretation and limitations: single-knob delta against a completed baseline — clean attribution when the window lands; KV at 0.90 raises preemption risk if pressure appears.
- Decision and next action: record the measured window when it lands; `.110` still waits for the running A arm's window.

### 2026-11-21 18:10 UTC - User direction: workers128 retired unlaunched; next real arm is EP16/seqs64 (`.111` in preparation)

- Question or hypothesis and reason for trying it: the user directed (12:13 EST) that the workers128 client arm will not run; the worker accepted and is preparing the next real arm instead.
- Published source revision: `.110` retired unlaunched (final source `8aee017ea5cfcd9748c372fc8555be2c8483f757`, compiled staleness 3 confirmed); `.111` being prepared via loop-push (source SHA pending).
- Command or material configuration: next real arm = EP16/seqs64 in a full RL round: 16 rollout + 48 trainer = 64 GPU; HBM 0.85, batch 8192, Harbor384, staleness 3, inflight128 fixed.
- External inputs and artifact references: user direction [R]; worker readout over Rhizome (~18:1x UTC) [R].
- Environment and hardware when relevant: with three 16-GPU B arms steady (B EP16 pair + kv90), total 112 ≤ 128 GPUs.
- Result, including failure or restart: no job submitted for `.110` — A3 already queued model demand and showed a non-GIL CPU profile, answering the client-cap question without the arm (see the 17:25 entry).
- Evidence: A3 client-cap measurement (17:25 entry); worker readout [R].
- Interpretation and limitations: the real-RL EP16/seq64 arm tests whether the best pure-decode geometry (EP16 serving bound 552.15 tok/s/GPU) transfers to training; its measured rollout rate will be the decisive comparison against A3's 127.22 tok/s/GPU. The `.110` declared-resource confound no longer matters since the arm will not run.
- Decision and next action: land `.111`, launch after the Harbor384 window; candidate/gain report 19:30 UTC; real training launch 19:45 UTC.

### 2026-11-21 18:15 UTC - `.111` real EP16 arm landed and validated; workers128 recipe deleted

- Question or hypothesis and reason for trying it: prepare the decisive real-RL test of the EP16 geometry per the 18:10 plan change.
- Published source revision: `058da8dcc2332525d6245ba8f7bae6ea18308b4c` (loop-push; 5 adopted-round tests, lint, types pass).
- Command or material configuration: `experiments/post_training/long_loop/rounds/mochi-throughput-ep16-seqs64.yaml` (`.111`) — seq64/EP16/DP16/TP1, HBM 0.85, batch 8192, Harbor384, staleness 3, workers 32, inflight128; native preflight confirms 48 trainer + 16 rollout = 64 GPU.
- External inputs and artifact references: worker readout over Rhizome (~18:1x UTC) [R]; validation receipt `sprint-real-ep16-seqs64-validation/*/receipt.json`.
- Environment and hardware when relevant: launches as A7 immediately after the A5/Harbor384 (`.109`) measured/cancelled window; total 112 GPU with the B trio (pair + kv90, launched 17:12 UTC).
- Result, including failure or restart: landed and validated; not launched yet. The workers128 recipe was deleted without ever launching.
- Evidence: native preflight receipt [R]; worker readout [R].
- Interpretation and limitations: this is the first real-RL arm with the EP16 rollout geometry — its measured rollout rate tests whether the 552.15 tok/s/GPU serving bound transfers to training, against A3's 127.22 tok/s/GPU (staleness-neutral comparison at step-1 windows).
- Decision and next action: launch after the `.109` window; candidate/gain report 19:30 UTC; real training launch 19:45 UTC.

### 2026-11-21 18:20 UTC - Code-trace verification: Harbor occupancy unmeasured; staleness can affect step-1 admission

- Question or hypothesis and reason for trying it: verify the recorded rollout-concurrency trace against the source before it is relied on for interpretation.
- Published source revision: `.109` running at `c6515fbdc7bc2b51f0f43649dac08ee54fa3397d`; `.111` ready at `058da8dcc2332525d6245ba8f7bae6ea18308b4c`.
- Command or material configuration: source inspection of `task_worker.py` / `inline_harbor_client.py` and `FullBatch.lease_batch`; review of A5's bounded logs.
- External inputs and artifact references: worker readout over Rhizome (~18:2x UTC) [R].
- Environment and hardware when relevant: Harbor384 gives 12 slots/worker with native `inference_stats` counters on.
- Result, including failure or restart: the Harbor semaphore (256//32 = 8/worker) wraps the entire `engine.run_task`; dispatch `min(_pending)` has no Harbor-slot awareness; the 8-slot semaphore is one group's worth, not an atomic group lock (partial groups can interleave); the generic 1024/worker task slots are unbinding. No explicit slot-wait logs exist and A5's bounded logs contain 0 direct slot-wait records, so actual Harbor occupancy is NOT measured; A3's 128 group calls / 430 pending model requests are not Harbor occupancy. `FullBatch.lease_batch` uses `range(step, step + staleness + 1)`, so staleness CAN change admission during step 1 even without a policy update — this corrects the earlier blanket "never binds" statement, but does NOT re-establish the A2 causal gain; default staleness 3 is kept per direction.
- Evidence: source trace and A5 bounded logs [R]; the standing trace section above was updated in place with these verified points.
- Interpretation and limitations: Harbor occupancy remains unmeasured this sprint; later rulings (11:55 shared quota, 12:13 engine capacity) override the old 1024-slot proposal.
- Decision and next action: keep default staleness 3; treat Harbor occupancy as unmeasured; launch A7 (`.111`) after the Harbor384 window; workers128 never ran.

### 2026-11-21 18:25 UTC - PROVISIONAL native window on B batch4096: 780.18 tok/s/GPU early post-warmup

- Question or hypothesis and reason for trying it: first live counter evidence from the freshly launched B EP16 arms.
- Published source revision: B pair at `3f4a4c68ce4fad43d1a639d1d390d1ae3e3649b1`; job `/benfeuer/mochi-throughput-decode-ep16-seqs64-batch4096-shared8-20261121a`.
- Command or material configuration: native counter delta over 17:13:19–17:16:22 UTC (182.507 s), all 16 ranks, no resets.
- External inputs and artifact references: worker readout over Rhizome (~18:2x UTC) [R]; `b4096-live-analysis-1716.json`, `sprint-b-live-1716.jsonl` in /experiments/active/mochi-throughput/ [R].
- Environment and hardware when relevant: 16 GB200; all current arms kept (108 GPU with A5).
- Result, including failure or restart: 12,482.81 aggregate / 780.18 tok/s/GPU (also the DP-engine mean), 52.92/64 running, 0.53 waiting/engine, KV 47.02%, TTFT mean 1.623 s; per-request decode = 1/mean TPOT = 12.45 tok/s (not an arithmetic mean of request rates).
- Evidence: native counter analysis receipt [R].
- Interpretation and limitations: PROVISIONAL — early post-warmup, NOT a 25-min winner; the final 1500 s receipt is expected ~17:38 UTC plus drain. Well above the completed EP16/seqs64 (batch 8192) baseline's 552.15 final, but the comparison is not yet apples-to-apples (different batched-token budget and a much shorter window).
- Decision and next action: await the final receipts for the B trio and the matching A5 window; A7 (`.111`, `058da8dcc2`) launches after the matching A5 window/cancel.

### 2026-11-21 18:30 UTC - User direction: quota-off arm `.112` in preparation; B pauses after its windows

- Question or hypothesis and reason for trying it: the user directed (12:18 EST) a quota-off variant to isolate the effect of consumed route quotas on rollout admission.
- Published source revision: `.112` being prepared as `rounds/mochi-throughput-noquotas.yaml` (source SHA pending); `.111`/A7 remains the next arm after the A5 window.
- Command or material configuration: `.112` = `consumed_route_quotas{}` (quotas off) with `batch_policy: full_batch` kept, seq32/HBM 0.85, Harbor 256, workers 32, inflight128, staleness 3, native `inference_stats`.
- External inputs and artifact references: user direction [R]; worker readout over Rhizome (~18:3x UTC) [R].
- Environment and hardware when relevant: scheduling plan — after the B windows are measured, B pauses so A7 (EP16, 64 GPU) + quota-off (60 GPU) run in parallel: 124 ≤ 128 GPUs, around 17:45 UTC after the A5 window is retained.
- Result, including failure or restart: preparation in progress; no launch yet. B 4096 and seq128 25-min results will be preserved; kv90 gets at least a 20-min window if ready. No 128-worker launch.
- Evidence: user direction and worker plan [R].
- Interpretation and limitations: the quota-off arm is a single-knob delta against the A5/Harbor384 baseline family, but note it uses Harbor 256 (not 384) per the direction, so the comparison carries a second knob difference; agentic share will be reported from actual batch/source evidence, not an assumed 50/50 (native provenance coverage being checked).
- Decision and next action: record the `.112` landing SHA and its measured window when available; A7 launches after the A5 window; report 19:30 UTC; real training 19:45 UTC.

### 2026-11-21 18:35 UTC - Quota-free `.112` landed; provenance observation fix enables exact agentic-share reporting

- Question or hypothesis and reason for trying it: land the user-directed quota-off arm and the provenance fix that makes the agentic-share measurement exact.
- Published source revision: `.112` quota-free `aab5e5f57bfa1ef5a6cb3c599ba40302a8e92247`; 1-arg observation fix `3226044a8fcf1dacad4b022074f94784493f83b7` (loop-push; 5 tests/lint/types pass).
- Command or material configuration: `rounds/mochi-throughput-noquotas.yaml` — `consumed_route_quotas{}` (quotas off), `batch_policy: full_batch` kept, seq32/HBM 0.85, Harbor 256, workers 32, inflight128, staleness 3, native `inference_stats`.
- External inputs and artifact references: worker readout over Rhizome (~18:3x UTC) [R].
- Environment and hardware when relevant: quota-free and EP16 A7 native preflights refreshing on the fix source; planned parallel A7 (64 GPU) + quota-free (60) = 124 ≤ 128 after the A5/B windows (~17:45 UTC), B paused as explicitly allowed.
- Result, including failure or restart: both landings passed; no sampling/admission compute change from the fix. `record_group_disposition` now passes `outcome.uid` (the API accepted a UID but the context previously omitted it), enabling exact unique consumed-groups/step joined to existing `task_group_completed(uid, source)`.
- Evidence: landing receipts via loop-push [R]; worker readout [R].
- Interpretation and limitations: the agentic share will be reported per **consumed** batch; a completion-only mix or an unfinished batch will be labeled separately — prevents an assumed 50/50 split from entering the record.
- Decision and next action: record `.112`'s measured window when available; no extra quota-1024 or worker-128 launch; report 19:30 UTC; real training 19:45 UTC.

### 2026-11-21 18:40 UTC - Both next A arms preflight PASSED on the observation-fix source

- Question or hypothesis and reason for trying it: confirm the two prepared A arms compile correctly on the provenance-fix source before the parallel launch.
- Published source revision: `3226044a8fcf1dacad4b022074f94784493f83b7`.
- Command or material configuration: native preflights for `.112` quota-free (compiled `consumed_route_quotas{}` + full_batch + seq32/HBM 0.85/Harbor 256; 48+12 = 60 GPU) and `.111` EP16 seq64 (compiled staleness 3 + `inference_stats`; 48+16 = 64 GPU).
- External inputs and artifact references: worker readout over Rhizome (~18:4x UTC) [R]; receipts `sprint-noquotas-validation/*/receipt.json`, `sprint-real-ep16-seqs64-uid-validation/*/receipt.json` in /experiments/active/mochi-throughput/ [R].
- Environment and hardware when relevant: current A5 Harbor384 + B trio = 108 GPU; parallel A7 + quota-free would be 124 ≤ 128; none of the running arms has a 25-min window yet.
- Result, including failure or restart: both preflights passed; both arms ready.
- Evidence: native preflight receipts [R].
- Interpretation and limitations: no new measurement; the agentic-share join will use consumed UID/step + completed UID/source via the landed observation fix.
- Decision and next action: launch both in parallel (124 GPU) after the current A5/B windows are retained (~17:45 UTC), pausing B as authorized.

### 2026-11-21 18:45 UTC - Correction: quota-free compiled confirmation premature; quotas still 16/16 in compiled config

- Question or hypothesis and reason for trying it: the 18:40 entry recorded the quota-free compiled confirmation; the worker then caught a semantic mismatch and withdrew it.
- Published source revision: current `3226044a8fcf1dacad4b022074f94784493f83b7`; `.112` recipe source unchanged (`aab5e5f57bfa1ef5a6cb3c599ba40302a8e92247`).
- Command or material configuration: explicit compiled-config assert on the landed `.112` recipe.
- External inputs and artifact references: worker correction over Rhizome (~18:4x UTC) [R]; failing receipt `sprint-noquotas-validation/c76d9962e1486f7ba8087f545fe65a90061ecac611922c6323ee8839b1debd2e/compiled.yaml`.
- Environment and hardware when relevant: no quota-free job launched; current running set remains A5 Harbor384 + B trio (108 GPU).
- Result, including failure or restart: the native validator passed but the compiled config still had `consumed_route_quotas` 16/16 — the recursive recipe merge of `{}` preserves base entries. The 18:40 "both preflights passed" statement is corrected: only the EP16 `.111` preflight is valid/ready (compiled staleness 3 + `inference_stats`, 48+16 = 64 GPU).
- Evidence: compiled.yaml failing receipt [R]; worker correction [R].
- Interpretation and limitations: the long-loop recipe merge boundary is being fixed to really clear quotas, with a behavior regression test and a compiled assert gating the launch; the parallel 124-GPU plan now depends on that fix landing. Lesson for the record: check the compiled config — native preflight alone misses this class of semantic mismatch.
- Decision and next action: record the merge-boundary fix and the re-run compiled assert before any `.112` launch; A7 remains ready after the A5 window.
