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
- Quota setting path (verified 2026-11-21 ~18:50 UTC): `experiments/post_training/long_loop/nvl72.py:343` — `build_run` unconditionally overwrote `consumed_route_quotas` with groups//2 each (the root cause of the `.112` compiled mismatch); the fix preserves an explicit `{}` while keeping the 50/50 resizing for non-empty quotas, and a one-rack/three-rack round regression asserts launch quotas `{}` + full_batch.
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

### 2026-11-21 18:50 UTC - Root cause refined: nvl72.build_run unconditionally overwrote consumed_route_quotas

- Question or hypothesis and reason for trying it: pin down why the compiled `.112` config still carried 16/16 quotas despite the empty recipe value.
- Published source revision: fix being validated/landed via loop-push on top of `3226044a8fcf1dacad4b022074f94784493f83b7`; `.112` recipe source unchanged (`aab5e5f57bfa1ef5a6cb3c599ba40302a8e92247`).
- Command or material configuration: source trace of `experiments/post_training/long_loop/nvl72.py:343` — `build_run` unconditionally overwrote `consumed_route_quotas` with groups//2 each (the current checkout already shows the fixed conditional form, `... if recipe.trainer.rollout_buffer.consumed_route_quotas else {}`).
- External inputs and artifact references: worker readout over Rhizome (~18:5x UTC) [R]; source trace/receipts in the worker STATUS [R].
- Environment and hardware when relevant: no quota-free job launched; EP16 `.111` remains ready; running set unchanged (A5 Harbor384 + B trio, 108 GPU).
- Result, including failure or restart: the mismatch is NOT a mere empty-dict merge issue — `build_run` itself overwrote the quotas. The fix preserves an explicit `{}` while keeping the 50/50 resizing for non-empty quotas, and a real one-rack/three-rack round regression now asserts launch quotas `{}` + full_batch.
- Evidence: source trace at nvl72.py:343 [R]; failing compiled receipt from the 18:45 entry [R].
- Interpretation and limitations: the standing trace section above was updated in place with the verified quota-setting path; the compiled preflight must be refreshed after the fix lands before `.112` can launch.
- Decision and next action: record the refreshed preflight and the `.112` launch when they occur; A7 still launches after the A5 window.

### 2026-11-21 18:55 UTC - Quota-free fix confirmed; compiled assert passes; both parallel arms ready

- Question or hypothesis and reason for trying it: close the `.112` correction arc by landing the nvl72 quota fix and re-running the compiled assert.
- Published source revision: `30d6357c550d35617b49e9f0c630c3754d7c87f2` (loop-push; 8 adopted RL round tests including the `{}` one/three-rack regressions, lint/types pass). Verified in the worktree: nvl72.py:346 now carries the fixed `else {}` conditional.
- Command or material configuration: `.112` compiled config asserts `{}` + full_batch + seq32 + staleness 3 + `inference_stats`, 48+12 = 60 GPU; `.111` EP16 seq64 normal 16/16 preflight refreshed at the same HEAD, 48+16 = 64 GPU.
- External inputs and artifact references: worker readout over Rhizome (~18:5x UTC) [R]; receipts `sprint-noquotas-fixed-validation/82ccea7b0aeb3d1037a4742b282d3daa1ec397e2500d77b0dccf63c189ede10e/receipt.json` and `sprint-real-ep16-seqs64-final-validation/*` in /experiments/active/mochi-throughput/ [R].
- Environment and hardware when relevant: parallel launch = A7 (64 GPU) + quota-free (60 GPU) = 124 ≤ 128, after the current A5/B windows/cancels (~17:45 UTC); B pause authorized. The UID observability fix (`3226044a8f…`, an ancestor) provides the actual consumed-group mix join.
- Result, including failure or restart: fix confirmed; the old c76d9962 compiled quota-free receipt is INVALID and now clearly withdrawn.
- Evidence: compiled-assert receipts [R]; worker readout [R].
- Interpretation and limitations: the correction chain (18:40 premature → 18:45 withdrawn → 18:50 root cause → 18:55 confirmed) is complete; `.112` remains unlaunched until the scheduled parallel slot.
- Decision and next action: launch `.111` + `.112` in parallel after the current windows; record measured windows as they land; report 19:30 UTC; real training 19:45 UTC.

### 2026-11-21 19:00 UTC - Job-index correction: `.108` Harbor1024 root ID recorded; no Harbor512 job exists

- Question or hypothesis and reason for trying it: the coordinator's 11:53 Harbor512 correction prompted the worker to confirm the exact identity and cancellation receipt of the cancelled Harbor1024 arm.
- Published source revision: unchanged — quota-free A8 compiled `{}` confirmed at `30d6357c550d35617b49e9f0c630c3754d7c87f2`; EP16/384 `.111` unchanged per the 12:13 direction.
- Command or material configuration: cancellation receipt review.
- External inputs and artifact references: worker readout over Rhizome (~19:0x UTC) [R]; receipt `sprint-a4-quota-cancellation.log`.
- Environment and hardware when relevant: no 1024 running and no 512 job submitted; the running set remains A5 Harbor384 (`.109`) + the B trio (108 GPU).
- Result, including failure or restart: the `.108` Harbor1024 root was `/benfeuer/long-loop-mochi-throughput-harbor1024-2026.11.21.108-coordinator-07903cf1`, cancelled at 17:01 UTC before measurement — previously the job ID had not been recorded in this notebook or in marin#11954.
- Evidence: cancellation receipt [R].
- Interpretation and limitations: pure bookkeeping correction; no measurement impact.
- Decision and next action: job index updated in marin#11954; continue waiting for the A5/B measured windows.

### 2026-11-21 19:05 UTC - Daytona accounting limitation; 512-trial upper limit accepted; lifecycle telemetry landing for A7/A8

- Question or hypothesis and reason for trying it: quantify actual sandbox CPU usage attributable to the sprint's A arms, and bound future usage.
- Published source revision: quota-free fix `30d6357c550d35617b49e9f0c630c3754d7c87f2` compiled `{}` verified; lifecycle telemetry landing via loop-push (8 focused SDK lifecycle/creation tests pass; full machine-file tests running).
- Command or material configuration: SDK acquires sandbox name + `provider.cpu` and confirms delete/not-found release, job-attributed via native telemetry context.
- External inputs and artifact references: worker readout over Rhizome (~19:0x UTC) [R]; worker REPORT now includes a peak column.
- Environment and hardware when relevant: A7 (`.111`, Harbor 384) ceiling 1536 vCPU; A8 (`.112`, Harbor 256) ceiling 1024 vCPU; 512-trial upper limit ~2048 vCPU at 4 CPU each.
- Result, including failure or restart: historical actual vCPU peaks for `.103`/A1/A2/A3/current A5 are UNAVAILABLE (retained org inventory is unattributed); configured ceilings are not measured peaks; all pure-B arms use 0 Daytona vCPU.
- Evidence: worker readout and REPORT peak column [R].
- Interpretation and limitations: per-GPU throughput results are unaffected, but any CPU-side cost claims for past arms must cite ceilings, not peaks; future A7/A8 arms will carry actual lifecycle telemetry.
- Decision and next action: 512-trial upper limit recorded as a user ruling (no exceedance without approval); record actual peaks from A7/A8 telemetry when available.

### 2026-11-21 19:15 UTC - Partial native table 17:38 UTC: first A5 native window (134.13/GPU); B trio provisional; Daytona ownership metric landed

- Question or hypothesis and reason for trying it: capture the first native-counter windows from the running A5 and B trio before their final receipts.
- Published source revision: A5 at `c6515fbdc7bc2b51f0f43649dac08ee54fa3397d`; B pair `3f4a4c68ce4fad43d1a639d1d390d1ae3e3649b1`; kv90 `530ab73b76a28a3e15a741d8f946a0dfaeea8818`; Daytona ownership metric `8b1e1686a985691f9454ed2c01c757c79f3ec736`.
- Command or material configuration: native counter deltas with no resets; A5 window from 17:19:36 UTC (first 17.6 min; 25-min deadline 17:44:36 UTC).
- External inputs and artifact references: worker readout over Rhizome (~19:1x UTC) [R]; `sprint-live-analysis-1738.json` in /experiments/active/mochi-throughput/ [R].
- Environment and hardware when relevant: A5 60 GB200 (12 rollout counted); B trio 16 GB200 each; all current arms kept (108 GPU).
- Result, including failure or restart (ALL PROVISIONAL): A5/Harbor384 134.13 tok/s/GPU (1609.60 agg), 31.49/32 running + 12.67 waiting, KV 59.52%, request 4.21, TTFT 57.33 s — the first native A-arm window; the client is already queued in A5 as well. B batch4096 751.43/GPU (12,022.93 agg, ~23 min post-warmup, TTFT 1.52 s); B EP16/seqs128 484.57/GPU (7,753.19 agg, 68.60/128 running, KV 95.06%, TTFT 113.44 s); B kv90 844.22/GPU (13,507.55 agg, early, TTFT 1.66 s).
- Evidence: sprint-live-analysis-1738.json [R]; worker readout [R].
- Interpretation and limitations: all provisional — final receipts pending. A5's 134.13/GPU (first 17.6 min) already exceeds A3's first-25min 127.22 console estimate, but the windows differ in method (native vs console) and age; the matched-window comparison waits for the 25-min receipt. The seq128 arm's KV 95.06% and TTFT 113.44 s indicate heavy cache pressure at long windows.
- Decision and next action: update rows to final when receipts land; Daytona ownership metric recorded (deletion events = delete_requested, not freed; provider-inventory join will yield observed allocated-vCPU peak); both next-A preflights refreshing on `8b1e1686a9`.

### 2026-11-21 19:20 UTC - FINAL: B batch4096 751.05 tok/s/GPU (+36.02% over batch8192 baseline); A7/A8 preflights passed

- Question or hypothesis and reason for trying it: does doubling batched tokens (8192→4096 per-rank path → 4096) lift the EP16/seqs64/shared8 decode rate? Final window answers it.
- Published source revision: `3f4a4c68ce4fad43d1a639d1d390d1ae3e3649b1`; job `/benfeuer/mochi-throughput-decode-ep16-seqs64-batch4096-shared8-20261121a`.
- Command or material configuration: valid 1500.818 s window, zero failed requests; 18,553 overlapping requests valid; drain 123.564 s.
- External inputs and artifact references: worker readout over Rhizome (~19:2x UTC) [R]; receipt `pure-ep16-batch4096-result.json`; S3 final artifact `decode/ep16-seqs64-kv85-batch4096-shared8-20261121a/synthetic-context64-output1024-shared8-seed17-c1024.json`.
- Environment and hardware when relevant: 16 GB200.
- Result, including failure or restart: 12,016.76 aggregate / **751.0475 tok/s/GPU**; request arithmetic mean 12.2081 tok/s; 61.5895/64 running; KV 52.4058% mean / 65.5849% max; TTFT mean 3.1973 s; prefix hit 99.7617%. Versus the EP16 seq64 batch8192 baseline (552.1466/GPU): **+36.02%**; request 9.2675→12.2081; KV 85.43→52.41; prefix 48.86→99.76.
- Evidence: final result receipt [R].
- Interpretation and limitations: the gain comes from cache fit on a fixed 32-shared-prompt-pool — NOT yet a real-RL gain; real-rollout validation must follow. Best valid long-B is now 751.05; kv90's 844.22/GPU remains provisional.
- Decision and next action: sequencing per the worker — A7 (`.111`) runs its specified batch 8192 first; the best one-knob B candidate follows in real rollouts. Both A7/A8 preflights passed on `8b1e1686a985691f9454ed2c01c757c79f3ec736`; A5's 25-min window completes 17:44:36 UTC.

### 2026-11-21 19:30 UTC - A7 real EP16 arm submitted; measured A5 cancelled 17:45:29 UTC

- Question or hypothesis and reason for trying it: run the decisive real-RL test of the EP16 rollout geometry (the 18:10/18:15 plan).
- Published source revision: `8b1e1686a985691f9454ed2c01c757c79f3ec736` (includes the UID observability fix and Daytona ownership metric).
- Command or material configuration: `rounds/mochi-throughput-ep16-seqs64.yaml` (`.111`) — seq64/EP16/DP16/TP1, HBM 0.85, batch 8192, Harbor384, staleness 3, workers 32, inflight128, native `inference_stats`; 48 trainer + 16 rollout = 64 GB200, PRODUCTION 08a, r136 excluded, graphs on.
- External inputs and artifact references: worker readout over Rhizome (~19:3x UTC) [R]; submission receipt `sprint-real-ep16-submission.log`.
- Environment and hardware when relevant: job `/benfeuer/long-loop-mochi-throughput-ep16-seqs64-2026.11.21.111-coordinator-22f1e6e8`; quota-free A8 (`.112`) launches next as remaining B measurement/drain releases capacity (parallel A7+A8 = 124 GPU).
- Result, including failure or restart: A7 submitted; A5/Harbor384 measured and cancelled at 17:45:29 UTC (final receipt pending — its provisional first-17.6min rate was 134.13 tok/s/GPU).
- Evidence: submission receipt [R]; worker readout [R].
- Interpretation and limitations: A7's measured rollout rate will be compared against A3's 127.22/GPU (console, first-25min) and A5's native provisional 134.13/GPU (first-17.6min) — method and window-age differences noted in prior entries; native counters are on for a like-for-like A5/A7 comparison.
- Decision and next action: record A5's final receipt and A7's window when they land; A8 launch as capacity frees; report 19:30 UTC; real training 19:45 UTC.

### 2026-11-21 19:40 UTC - FINAL: A5/Harbor384 25-min 124.97/GPU — ~flat vs A3; Harbor capacity gave no early-window gain

- Question or hypothesis and reason for trying it: does raising Harbor capacity 256→384 improve real-rollout throughput versus A3 (Harbor 256)?
- Published source revision: `c6515fbdc7bc2b51f0f43649dac08ee54fa3397d`; job `/benfeuer/long-loop-mochi-throughput-harbor384-2026.11.21.109-coordinator-16513921`.
- Command or material configuration: native final window 17:19:36–17:44:36 UTC (25 min, all 12 ranks, 0 resets); native `inference_stats` counters on.
- External inputs and artifact references: worker readout over Rhizome (~19:4x UTC) [R]; receipts `a5-final25-analysis.json`, `a5-all-engine-raw-stats-1744.json`, `a5-native-final25-1744.jsonl` in /experiments/active/mochi-throughput/ [R].
- Environment and hardware when relevant: 60 GB200 (12 rollout counted); A7 (`.111`) running; B seq128/kv90 final windows and drain expected ~17:51 UTC before quota-free A8 launches (parallel 124 GPU).
- Result, including failure or restart: 1499.61 aggregate / **124.967 tok/s/GPU** native; 30.842/32 running, 11.258 waiting; KV 67.255%; request 1/mean TPOT 4.038; TTFT 53.039 s. Last 10 min 105.61/GPU, 29.94/32 running, KV 85.30%. Cancelled 17:45:29 UTC.
- Evidence: final 25-min analysis receipts [R].
- Interpretation and limitations: matched-method comparisons — A5 raw console first-20 ~133.01/GPU vs A3 raw 134.876 (−1.4%, no gain); A5 raw first-25 127.35 vs A3 console 127.22 (~flat, baseline driver-dedup). More Harbor capacity did not improve early-25min throughput; engines were already full/queued, so the Harbor-slot knob is saturated at this operating point. The provisional 134.13 first-17.6min figure overstated the final rate (early-window bias now quantified).
- Decision and next action: A7's EP16 window is the remaining decisive A comparison; record it when it lands; A8 launch ~17:51 UTC; report 19:30 UTC; real training 19:45 UTC.

### 2026-11-21 19:45 UTC - Linked: rollout concurrency-limits design issue marin#11957

- Question or hypothesis and reason for trying it: the coordinator directed (user request) that the new rollout concurrency-limits design issue be linked from this campaign's records.
- Published source revision: n/a (design issue, not a measured arm).
- Command or material configuration: n/a.
- External inputs and artifact references: [marin#11957](https://github.com/marin-community/marin/issues/11957) — "[skyrl] Unify the rollout concurrency limits into one explicit budget"; the table of every admission limit (Harbor slots, task slots, dispatch, route quotas, staleness lease range) and which one bound in each Mochi arm of this sprint.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: linked from marin#11954 (one 🤖 comment + the Related list) and recorded here.
- Evidence: issue exists (OPEN) — verified via the GitHub API [R].
- Interpretation and limitations: the design issue consumes the binding-behavior evidence this sprint produced (A5 Harbor saturation, batch4096 cache fit, staleness lease semantics, quota-off root cause); this notebook's code-trace section feeds it.
- Decision and next action: keep the per-arm "which limit bound" facts flowing to #11957 as arms land.

### 2026-11-21 19:55 UTC - B trio finals; engine-side context split measured; context-gap fix landed

- Question or hypothesis and reason for trying it: finalize the B EP16 trio and measure the prompt/decode context split on the same engine.
- Published source revision: context-gap fix `119ea9984b7e400e926ced96bc8a4a3f8df0a11b` (loop-push; existing timing events get numeric prompt/context tokens from pinned vLLM `state.prompt_len`, not prompt text; public-evidence unit test/lint pass); B pair `3f4a4c68ce…`; kv90 `530ab73b76…`.
- Command or material configuration: final 1500 s windows on seq128 and kv90; both cancelled-own 17:54:58–59 UTC; B paused thereafter.
- External inputs and artifact references: worker readout over Rhizome (~19:5x UTC) [R]; retained receipts (worker retains both finals' receipts).
- Environment and hardware when relevant: A7 running (64 GPU); quota-free A8 validating/submitting on the new source; parallel A7+A8 = 124 GPU. No 1024 or workers128 launch; 12:13 Harbor384 stands.
- Result, including failure or restart: B EP16/seqs128 FINAL 443.95/GPU (aggregate 7103.2 derived ×16), KV 94.94%, TTFT 121.13 s — KV-heavy, the seq128 serving bound collapses under cache pressure. B kv90 FINAL 831.73/GPU (13,307.68 agg), KV 52.96%, TTFT 2.925 s — below batch4096's 751.05. All zero errors. Context split: A5 native 25-min 1,663.40 prompt vs 124.97 decode tok/s/GPU; 66 KV preemptions; token-weighted prefix hit 19.575% (console 33% rank mean overstated); prefill mean 1.727 s vs decode 86.739 s (n=3551); 93.0% of tokens are prompt, NOT 93% of GPU time. A3 raw first-20: est. 1,218.73 prompt vs 134.88 decode; 8 retained driver request-rate samples median 3.40/mean 3.54 tok/s (dedup-biased); A3 running-context distribution and exact preemptions not retained (callback off).
- Evidence: worker receipts (retained) [R]; readout [R].
- Interpretation and limitations: best valid long-B is batch4096 at 751.05/GPU; kv90 (HBM 0.90 alone) does not beat it as a serving config. The context split reframes decode-rate interpretation — prompt tokens dominate token counts, and prefix-hit consoles overstate token-weighted reality; the prompt-length fix makes the next arm's context distribution exact. These facts feed marin#11957's binding-limits table.
- Decision and next action: record A8's job ID and A7's measured window when they land; report 19:30 UTC milestone processing complete; real training 19:45 UTC.

### 2026-11-21 20:05 UTC - A8 quota-free submitted; A7+A8 parallel pair complete (124 GPU)

- Question or hypothesis and reason for trying it: launch the user-directed quota-off arm alongside A7 to isolate the consumed-route-quota effect on real-rollout admission.
- Published source revision: `119ea9984b7e400e926ced96bc8a4a3f8df0a11b` (includes the context-gap fix, UID observability, and Daytona CPU observability).
- Command or material configuration: `rounds/mochi-throughput-noquotas.yaml` (`.112`) — `consumed_route_quotas{}` compiled-asserted, full_batch kept, seq32/HBM 0.85, batch 8192, Harbor 256, workers 32, staleness 3, inflight128, native `inference_stats`; 48 trainer + 12 rollout = 60 GB200, PRODUCTION 08a, r136 excluded, graphs on.
- External inputs and artifact references: worker readout over Rhizome (~20:0x UTC) [R]; receipt `sprint-noquotas-submission.log`; preflight `sprint-noquotas-context-validation/82ccea7b0aeb3d1037a4742b282d3daa1ec397e2500d77b0dccf63c189ede10e/receipt.json`.
- Environment and hardware when relevant: job `/benfeuer/long-loop-mochi-throughput-noquotas-2026.11.21.112-coordinator-8aa614a8`; A7 GPU child `…-a6f6316732f0` RUNNING (submitted 17:47:13 UTC); A7+A8 = 124 GPU; all B arms measured/cancelled/paused.
- Result, including failure or restart: A8 submitted with the compiled `{}` assert included at submission time (the 18:45-class mismatch cannot recur silently).
- Evidence: submission receipt + preflight [R].
- Interpretation and limitations: A8 is a single-knob delta versus A5 (Harbor 256 vs 384 differs too — second knob noted in the 18:30 entry); its numeric context observability makes the prompt/decode split exact for this arm, addressing the A3/A5 retention gap.
- Decision and next action: record A7's and A8's measured windows when they land; report 19:30 UTC milestones complete; real training 19:45 UTC.

### 2026-11-21 20:10 UTC - A8 GPU child admitted; A9 HBM90 one-knob real arm landed (`.113`)

- Question or hypothesis and reason for trying it: complete the A8 launch and prepare the HBM 0.90 real-rollout test (the real-RL counterpart of the kv90 serving arm).
- Published source revision: `.113` landed at `f26fc27bcd86dac16bcf867435f72ef9166cf463` (loop-push; 8 adopted RL round tests/lint/types pass; native preflight in progress); A8 root at `119ea9984b7…`.
- Command or material configuration: `rounds/mochi-throughput-ep16-seqs64-kv90.yaml` — ONLY HBM 0.85→0.90 vs A7; 48 trainer + 16 rollout = 64 GPU.
- External inputs and artifact references: worker readout over Rhizome (~20:1x UTC) [R].
- Environment and hardware when relevant: A8 GPU child `/benfeuer/long-loop-mochi-throughput-noquotas-2026.11.21.112-coordinator-8aa614a8/long-loop-mochi-throughput-noquotas-2026.11.21.112-aefb515be064` RUNNING (submitted 17:59:47.811 UTC, root 17:58:30); both A7/A8 GPU children admitted, total 124 GB200; B paused. A9 cannot launch concurrently with A7+A8 (124+64 > 128) — it replaces A7 after its 25-min window.
- Result, including failure or restart: landings passed; A9 not launched yet. A7 startup bounded logs are being read at the 15-min gate rather than waiting an hour.
- Evidence: loop-push landing receipts [R]; worker readout [R].
- Interpretation and limitations: A9 completes the one-knob HBM sweep on the real-rollout lane (0.85→0.90), mirroring the kv90 serving arm (which finished below batch4096 as a serving config); its rollout rate will show whether KV headroom helps real rollouts.
- Decision and next action: record A9's preflight and launch when they occur; A7/A8 windows are the current decisive evidence.

### 2026-11-21 20:15 UTC - Correction: kv90 831.73/GPU IS the best valid long-B (earlier "below batch4096" was wrong); A9 preflight confirmed; A7 startup healthy

- Question or hypothesis and reason for trying it: the worker's latest readout states the best valid long-B is kv90 at 831.73/GPU, contradicting this notebook's 19:55 interpretation; also confirms A9 readiness and A7 startup health.
- Published source revision: A9 `.113` at `f26fc27bcd86dac16bcf867435f72ef9166cf463` (preflight receipt `sprint-real-ep16-kv90-validation/*/receipt.json`); A7 at `8b1e1686a9…` (startup logs `a7-startup-1802.jsonl`).
- Command or material configuration: A9 = `rounds/mochi-throughput-ep16-seqs64-kv90.yaml`, ONLY HBM 0.85→0.90 vs A7, 64 GPU; native preflight confirmed 64 GPU/seq64/staleness 3/HBM 0.90.
- External inputs and artifact references: worker readout over Rhizome (~20:1x UTC) [R]; receipts as above [R].
- Environment and hardware when relevant: A7+A8 = 124 GB200 running; A9 replaces A7 after its 25-min window (not concurrent).
- Result, including failure or restart: CORRECTION — the 19:55 entry said kv90 (831.73/GPU) "does not beat" batch4096 (751.05/GPU); that was an arithmetic misreading: 831.73 > 751.05, so **kv90 is the best valid long-B**. A7 startup gate HEALTHY: graphs 17:59 UTC across 16 ranks, initial weight sync 18:02:04.827 UTC, zero Traceback/RuntimeError/OOM in bounded logs.
- Evidence: final receipts (kv90 831.73 vs batch4096 751.05) [R]; A7 startup bounded logs [R].
- Interpretation and limitations: the kv90 provisional 844.22 overstated its final 831.73, but the final still leads batch4096; the real-RL winner is still pending A7/A9 windows. A9 replaces A7 after its window per the GPU-cap sequencing.
- Decision and next action: kv90 is the leading one-knob B candidate for the real-rollout follow-up; record A7/A8 windows next.

### 2026-11-21 20:30 UTC - A7 early native window: 112.83/GPU at 5.28 min — below age-aligned A5; no gain claim

- Question or hypothesis and reason for trying it: first live comparison of the EP16 rollout geometry (A7) against A5 under native counters, age-aligned.
- Published source revision: A7 at `8b1e1686a985691f9454ed2c01c757c79f3ec736`.
- Command or material configuration: native window over the first 5.28 min (from 18:02:04.921 UTC start; 25-min deadline 18:27:04.921); age-aligned A5 comparison at the first same 316.9 s.
- External inputs and artifact references: worker readout over Rhizome (~20:2x UTC) [R]; receipts `a7-native-partial-1806-analysis.json`, `a7-vs-a5-matched-early-analysis.json` in /experiments/active/mochi-throughput/ [R].
- Environment and hardware when relevant: A7 64 GPU (16 rollout ranks); A8 initializing its 12 engines with 0 errors; A7+A8 = 124 GPU; B paused; A9 (`.113`, HBM 0.90) ready to replace A7 after its window.
- Result, including failure or restart: A7 early 1805.21 aggregate / 112.83 tok/s/GPU, 37.43/64 running + 0.43 waiting, KV 27.01%, prefix 75.51%, 0 preemptions, request 2.72, TTFT 4.39 s. Age-aligned A5: 1958.59 aggregate / 163.22/GPU, 31.03/32 running, 18.32 waiting, KV 29.82%. A7 early is −30.9% per-GPU / −7.8% aggregate.
- Evidence: native partial analyses [R].
- Interpretation and limitations: NO gain claim — await the 25-min window. Mechanistic read: demand (~600 requests aggregated in both) is spread by EP16/cap-64 with almost no queue, versus A5's 18-waiting queue at cap 32. Confound: A7 restores staleness 3 vs A5's actual 11 (lease-lookahead difference), so the engine-geometry effect is not isolated.
- Decision and next action: record A7's 25-min final and the A5-vs-A7 verdict when it lands (18:27:04.921 UTC); A9 replaces A7 after the window unless new steering arrives.

### 2026-11-21 20:40 UTC - Found: HTTP request-ID tracing bug (join-coverage gap); causality-preserving HBM85 control planned

- Question or hypothesis and reason for trying it: the new numeric context fields were not joining to request timings — root cause found.
- Published source revision: fix in progress (source SHA pending); no live-job patch.
- Command or material configuration: `model_clients` sends lowercase `x-request-id`; the pinned vLLM adapter's `_MinimalRequest.headers` is a case-sensitive plain dict; `BaseServing._base_request_id` looks up `X-Request-Id`, misses it, and generates a different ID. Fix: case-insensitive Starlette headers + a stable ID generated once; adapter moves to the CPU-light `inference_http_backend` with a behavior regression (case variants + timing join).
- External inputs and artifact references: worker readout over Rhizome (~20:4x UTC) [R].
- Environment and hardware when relevant: A7/A8 continue unchanged at 124 GPU; A8's startup 15-min gate bounded errors read in progress.
- Result, including failure or restart: join-coverage gap confirmed in A3's all-12 raw logs — 3,611 submitted vs only 24 first_token and 3,181 ended events joined. The bug may also affect ID-based group behavior; no gain claim.
- Evidence: A3 raw-log event counts [R].
- Interpretation and limitations: this explains the residual context-distribution gap despite the `119ea9984b…` numeric-fields landing; ID-based joins (including agentic-share grouping) must be re-validated after the fix.
- Decision and next action: causality plan — the next HBM 0.90 arm carries the header fix, and an HBM 0.85 same-fix control arm runs after the A8 window (slots 64+64 = 128), both 25-min windows finishing ~19:10–19:22 UTC before the 19:30 choice; record the fix SHA and control-arm launch when they land.

### 2026-11-21 20:50 UTC - FINAL: A7 25-min 107.38/GPU — per-GPU below A5; no EP real-rollout gain established; A9 submitted with header fix

- Question or hypothesis and reason for trying it: does the EP16 rollout geometry (A7) beat A5's EP12 at cap 32 in real rollouts over a full 25-min window?
- Published source revision: A7 at `8b1e1686a985691f9454ed2c01c757c79f3ec736`; A9 at `858ab5d15bf3118ae0358ff79569ec6b33320a68` (lands the HTTP request-ID header fix).
- Command or material configuration: A7 final 25-min native window (16 ranks); A9 = `rounds/mochi-throughput-ep16-seqs64-kv90.yaml`, ONLY HBM 0.85→0.90 vs A7, 64 GB200.
- External inputs and artifact references: worker readout over Rhizome (~20:5x UTC) [R]; receipts `a7-final25-analysis.json`, `sprint-a9-submission.log`, `sprint-a9-httpid-validation/a81b9f95…/receipt.json`.
- Environment and hardware when relevant: A9+A8 = 124 GPU running; A10 (same-source HBM 0.85) planned after A8's window to isolate the HBM knob.
- Result, including failure or restart: A7 FINAL 1718.01 aggregate / **107.376 tok/s/GPU**, 25.894/64 running + 0.092 waiting, KV 46.60% mean / 89.71% max, request 3.940, TTFT 2.252 s, prompt 3224.19/GPU, prefix 88.576%, 0 preemptions; retained and cancelled 18:31 UTC. A9 submitted.
- Evidence: A7 final analysis receipt [R]; A9 submission receipt [R].
- Interpretation and limitations: **per-GPU 107.38 < A5's 124.97 despite +14.6% aggregate — no EP real-rollout gain established.** The EP16 serving-bound gains (552–831 tok/s/GPU on the fixed-prompt-pool decode lane) do not transfer to real RL rollouts. Confounds already in the record (staleness 3-vs-11 across A5/A7; age alignment) do not rescue a gain. A9 (HBM 0.90) and A10 (HBM 0.85, same source) will isolate the HBM knob next.
- Decision and next action: record A8's and A9's windows; the 19:30 candidate/gain choice proceeds on A5-class configs unless A9/A10 overturn; real training launch 19:45 UTC.

### 2026-11-21 21:00 UTC - A9 GPU child admitted; A8 deadline 18:39:00.943 UTC; agentic share pending a consumed batch

- Question or hypothesis and reason for trying it: provenance refresh for the running parallel pair and the agentic-share capture plan.
- Published source revision: unchanged (A9 at `858ab5d15b…` with the header fix; A8 at `119ea9984b7…`).
- Command or material configuration: A9 GPU child `/benfeuer/long-loop-mochi-throughput-ep16-seqs64-kv90-2026.11.21.113-coordinator-63f22c5c/long-loop-mochi-throughput-ep16-seqs64-kv90-2026.11.21.113-ab3bbe76a0cb` (submitted 18:33:31 UTC).
- External inputs and artifact references: worker readout over Rhizome (~21:0x UTC) [R].
- Environment and hardware when relevant: A8's exact 25-min deadline 18:39:00.943 UTC; A9+A8 = 124 GPU; B paused.
- Result, including failure or restart: at 18:33 UTC, 83 task groups completed but 0 consumed (one fully_masked) — no consumed-batch agentic share yet; it will be captured at A8's final. A7's request/context population is incomplete on the old case-sensitive HTTP source; A9/A10 carry the fixed header join.
- Evidence: worker live table + group counters [R].
- Interpretation and limitations: no unmeasured-throughput claims from A7's context fields; the consumed-batch agentic share remains pending until a batch is consumed.
- Decision and next action: capture A8's final (deadline 18:39:00.943 UTC) and the agentic-share figure; training choice 19:30 UTC / launch 19:45 UTC unchanged.

### 2026-11-21 21:10 UTC - FINAL: A8 quota-free 208.63/GPU (3.73x `.103` raw) — top real-rollout rate, heavy mix caveats; A10 submitting

- Question or hypothesis and reason for trying it: does removing consumed route quotas (with staleness 3, Harbor 256, seq32) raise real-rollout throughput?
- Published source revision: A8 at `119ea9984b7e400e926ced96bc8a4a3f8df0a11b`; A10 submitting at `858ab5d15bf3118ae0358ff79569ec6b33320a68` (HBM 0.85 same-source control).
- Command or material configuration: native final 25-min window (12 ranks); quotas `{}` compiled-asserted.
- External inputs and artifact references: worker readout over Rhizome (~21:1x UTC) [R]; receipts `a8-final25-analysis.json`, `a8-batchmix-context-final25-analysis.json`, `a8-all-engine-raw-stats-context-1839.json`, `sprint-a8-measured-cancellation.log`.
- Environment and hardware when relevant: A9+A10 = 128 GPU after A10 lands; B paused.
- Result, including failure or restart: 2503.570 aggregate / **208.63 tok/s/rollout-GPU**; 21.9886/32 running + 9.1964 waiting; KV 50.0609% mean / 95.2839% max; request 9.02554; TTFT 90.359 s; 0 preemptions; prompt 935.353/GPU; prefix 86.649%. Last 10 min: 156.806/GPU, 14.755/32 running, 0 waiting, TTFT 0.797 s. Cancelled 18:39:45 UTC; all 12 raw logs/native retained.
- Evidence: final analysis receipts [R].
- Interpretation and limitations: the raw 3.73x over the `.103` control (55.90) is the top real-rollout rate so far, but the mix caveat dominates — 90 completed groups ALL nonagentic, 0 consumed batches (one fully_masked), so the consumed-agentic share is UNAVAILABLE (not 0%); do not label a training gain. Timing coverage 88/1867 request IDs (HTTP join bug); only 9 active valid context samples (10.6k–14.2k tokens from 69–72-token prompts) — NOT the real agentic context population.
- Decision and next action: A10 (HBM 0.85, same source as A9) will isolate the HBM knob; record A9/A10 finals; the 19:30 choice proceeds with the mix caveat attached to A8's number.

### 2026-11-21 21:20 UTC - A10 HBM .85 control submitted; A9-vs-A10 pair isolates only the HBM knob

- Question or hypothesis and reason for trying it: complete the HBM-isolation pair on the real-rollout lane with a shared HTTP identity fix.
- Published source revision: A10 at `858ab5d15bf3118ae0358ff79569ec6b33320a68` — SAME source as A9; recipe `rounds/mochi-throughput-ep16-httpid-control.yaml` (`.114`).
- Command or material configuration: A9 (HBM 0.90) vs A10 (HBM 0.85); everything else fixed — EP16/seq64/batch8192/Harbor384/staleness3/workers32/inflight128; 64 GB200 each.
- External inputs and artifact references: worker readout over Rhizome (~21:2x UTC) [R]; receipts `sprint-a10-submission.log`, `sprint-a10-httpid-validation/b0879291…/receipt.json`.
- Environment and hardware when relevant: A9+A10 = 128 GPU after A8's cancel 18:39:45 UTC; next verified one-shot 18:48 UTC for A9's startup 15-min gate/counters.
- Result, including failure or restart: A10 submitted; A9 in flight.
- Evidence: submission + preflight receipts [R].
- Interpretation and limitations: the pair shares the request-ID fix, so any A9-vs-A10 difference attributes to HBM alone; the A8 mix caveat (0 consumed batches, all-nonagentic completions) still applies to workload-mix-sensitive readings.
- Decision and next action: record both 25-min finals; training choice 19:30 UTC, real train 19:45 UTC unchanged.

### 2026-11-21 21:45 UTC - Reconciled final table; conservative training candidate stated; A9/A10 in flight

- Question or hypothesis and reason for trying it: reconcile the sprint's native first-25min table across the secretary and worker records before the 19:30 training choice.
- Published source revision: A9/A10 on the matched `858ab5d15bf3118ae0358ff79569ec6b33320a68` source at 128 GPU total.
- Command or material configuration: native real first-25min windows; request = reciprocal pooled mean TPOT (B rows use the arithmetic mean, flagged as a different definition).
- External inputs and artifact references: worker final readout table over Rhizome (~21:4x UTC) [R]; REPORT/STATUS reconciled.
- Environment and hardware when relevant: next one-shot 18:48 UTC (A9 startup gate); real training launch 19:45 UTC.
- Result, including failure or restart: A5 124.97/GPU (req 4.04, KV 67.25%, prompt 1663.40/GPU, prefix 19.58%, 66 preemptions); A7 107.38/GPU (prompt 3224.19/GPU, prefix 88.58%, 0 preemptions; last-10min 91.37/GPU at 19.75/64 with no queue — the larger engine shape was underfilled by then); A8 208.63/GPU (req 9.03, prompt 935.35/GPU, prefix 86.65%, 0 preemptions; workload-mix caveat, no 3.73x training-gain claim); B EP16/HBM90 serving final 831.73/GPU (prefix 99.78%).
- Evidence: reconciled table from the worker readout [R]; all underlying receipts already catalogued in marin#11954.
- Interpretation and limitations: the EP16 per-GPU shortfall is now mechanistically explained (underfilled engine at the tail of the window); the A8 figure remains mix-caveated. The conservative training candidate stands at seq32/HBM 0.85/batch 8192/EP12/workers 32/Harbor 256/inflight128/staleness 3, with the quota choice pending the 19:30 evidence.
- Decision and next action: record A9/A10 finals when they land; the 19:30 choice finalizes the quota decision; real training launch 19:45 UTC.

### 2026-11-21 21:55 UTC - Worker's consolidated authoritative arm table received (18:42:50 UTC); secretary table synced

- Question or hypothesis and reason for trying it: sync the secretary's #11954 table against the worker's consolidated authoritative table (which adds the Peak Daytona vCPU column).
- Published source revision: table as of 2026-11-21T18:42:50.698134+00:00; REPORT/STATUS reconciled.
- Command or material configuration: no change — sync only.
- External inputs and artifact references: worker consolidated table over Rhizome [R]; receipts rooted at /experiments/active/mochi-throughput (per-arm index follows separately from the worker).
- Environment and hardware when relevant: A9/A10 in flight at 128 GPU on the matched source.
- Result, including failure or restart: table synced — A5 KV max 99.84%, B EP16/seqs64 KV max 99.99%, B EP16/seqs128 final running 66.97/128 and request 6.44, kv90 KV max 66.93%, and an A6/workers128 row added ("retired before launch — client already fills engines", 0 Daytona vCPU). The worker's A3 concurrency-evidence block (128 RUNNING group calls, 1024 scheduled coroutines, workers' max_concurrent_tasks 1024, no GIL bottleneck) matches the standing trace section.
- Evidence: consolidated table [R].
- Interpretation and limitations: caveats restated by the worker — .103/A1/A2/A3 console estimates vs A5/A7/A8 native; A2 no causal staleness gain; A8 zero consumed batches/90 nonagentic completions (agentic share unavailable); serving fixed-prompt-pool 99.8% prefix is an upper bound, not a real-RL gain.
- Decision and next action: worker will cc every readout/submission/measurement/config decision/training launch; secretary keeps syncing.

### 2026-11-21 22:05 UTC - Provenance index received and synced: per-arm config|source|job|receipts for both lanes

- Question or hypothesis and reason for trying it: align every arm's Source attribution in marin#11954 with the worker's authoritative provenance index.
- Published source revision: machine-readable index `secretary-arm-provenance-20261121.json` in /experiments/active/mochi-throughput/.
- Command or material configuration: no change — attribution sync only.
- External inputs and artifact references: worker real-lane and serving-lane provenance tables over Rhizome [R].
- Environment and hardware when relevant: A9/A10 pending native 25-min on the matched `858ab5d15b…` source (only HBM differs).
- Result, including failure or restart: corrections applied in marin#11954 — A1 `.105` source is `476a95362f…` (was mislumped under `3a859712f5…`), A2 `.106` is `3bc5ae3fda…`; A4 Harbor1024 config-landed `e0bf304558…` but was actually submitted at `3f4a4c68ce…`; B seqs32 decode arm source is `e9a16e37ce…` (the `.104` real-RL job is a different arm); B KV85 `f91a4966b6…`; B batch16384 `476a95362f…`; B shared8 pair `d08bf17b3a…`; B seqs64 pair `3bc5ae3fda…`. Workload-shape caveat added: fixed-pool shared8 = 32 prompt bodies / 256 request identities, 6439/1024 (long) or 139/1024 (short); repeated cycles can overstate fresh-RL prefix reuse; serving runs 12 GPUs unless EP16 (16).
- Evidence: worker provenance tables [R]; resolved SHAs verified via `git rev-parse` in the checkout.
- Interpretation and limitations: per-arm sources were lumped in early sprint bookkeeping; the authoritative index now governs. A6/workers128 never launched; the `.104` round is not in the live arm table (provenance explained separately by the worker).
- Decision and next action: the provenance index governs future Source cells; A9/A10 finals next; choice 19:30 / real train 19:45.

### 2026-11-21 22:15 UTC - A7 transfer verdict quantified; provisional training candidate formalized

- Question or hypothesis and reason for trying it: the coordinator asked the worker for the requested REAL EP16 seq64 arm's outcome and the full table; the response quantifies the serving-to-rollout transfer failure and formalizes the provisional candidate.
- Published source revision: unchanged — A7 `8b1e1686a9…`; A9/A10 on matched `858ab5d15b…`.
- Command or material configuration: provisional training candidate = EP12, seq32, HBM 0.85, batch 8192, Harbor 256, workers 32, inflight128, staleness 3, graphs/prefix caching ON.
- External inputs and artifact references: worker response over Rhizome (~22:1x UTC) [R]; A7 receipts `a7-final25-analysis.json`, `a7-context-summary-final25.jsonl`, `sprint-a7-measured-cancellation.log`.
- Environment and hardware when relevant: A9/A10 running at 128 GPU; startup-gate one-shot 18:48 UTC armed (no duplicate timer).
- Result, including failure or restart: A7 — 1.92x the `.103` control raw (55.90), but **−14.1% per-GPU vs A5's Harbor384 124.97 while using 33% more rollout GPUs**; serving 552.15/GPU did NOT transfer as a real per-GPU gain.
- Evidence: A7 final receipts [R].
- Interpretation and limitations: the provisional candidate's supporting numbers are framed honestly — A3's 2.28x is a short-window/proxy observation (its staleness 11 is not retained for training), and A8's 3.73x raw is not a training gain (0 consumed batches, 90 nonagentic completions).
- Decision and next action: final choice 19:30 UTC after A9/A10; real training launch 19:45 UTC; secretary records the chosen config and the launch verification, then closes the records per the campaign rules.

### 2026-11-21 22:25 UTC - A10 GPU child admitted; A9/A10 engine init underway, no counter window yet

- Question or hypothesis and reason for trying it: scheduled-checkpoint reconciliation of the parallel pair's startup state.
- Published source revision: both arms on `858ab5d15bf3118ae0358ff79569ec6b33320a68`.
- Command or material configuration: A9 (`.113`, HBM 0.90) vs A10 (`.114`, HBM 0.85) — only HBM differs; Harbor 384 each (no >512), graphs on, r136 excluded; 128 GPU total.
- External inputs and artifact references: worker readout over Rhizome (~22:2x UTC) [R]; diagnostics `a9-iris-bounded-startup-1847.txt`, `a9-startup-task0-1847.txt`, `a9-native-start-marker-1846.jsonl`, `a10-startup-1846.jsonl`, `sprint-a10-initial-job-tree.txt`.
- Environment and hardware when relevant: A10 GPU child `…-114-coordinator-4fa04020/…-114-a8a50ac911fe` submitted 18:42:17 UTC; A9 engine init 18:45–18:46 UTC; A10 engine init 18:46:38 UTC.
- Result, including failure or restart: no GPU counter window started on either arm yet. A9's regional Finelog rows are empty while Iris retains progress — a stall cannot be inferred from empty regional rows; no fatal Traceback/OOM in the bounded latest 80 lines (a custom `model_type` parse warning followed by init progress is not terminal proof).
- Evidence: startup diagnostics receipts [R].
- Interpretation and limitations: startup-only state; no throughput claim. The empty-regional-Finelog vs Iris-retained-progress discrepancy is a monitoring gap, not engine evidence.
- Decision and next action: A10's 15-min qualification gate near 18:57 UTC; capture both native windows when they start; selection 19:30 UTC, training 19:45 UTC.

### 2026-11-21 22:35 UTC - A9 window defined (ends 19:13:18.607 UTC); HTTP join confirmed working; hub-vs-regional monitoring note

- Question or hypothesis and reason for trying it: pin A9's measurement window and confirm the request-ID fix works before the 19:30 selection.
- Published source revision: both arms on `858ab5d15bf3118ae0358ff79569ec6b33320a68`.
- Command or material configuration: A9 first native async config 18:48:18.509 UTC (rollout_wait 18:48:18.607); exact 25-min window ends 19:13:18.607 UTC.
- External inputs and artifact references: worker readout over Rhizome (~22:3x UTC) [R]; receipts `a9-native-first-rollout-1849.jsonl`, `a9-first-request-error-1850.txt`, `a10-first-request-startup-1850.txt`, `sprint-wake-fired-1848.json`.
- Environment and hardware when relevant: 128 GPU total (64 each); Harbor 384 per job; graphs on; r136 excluded. A10 still engine-initializing (bounded 70 lines clean).
- Result, including failure or restart: FlashAttention AuxData message in A9 was a UserWarning, not fatal; bounded next 120 lines show successful first_token + ended numeric prompt/context records, 0 Traceback/RuntimeError/OOM. Sample prompts 1711/1742/2367 tokens demonstrate the HTTP join working on actual turns; full coverage to be measured from all 16 raw actors (not claimed 100%).
- Evidence: native first-rollout receipt [R]; startup receipts [R].
- Interpretation and limitations: the native hub (marin) has fresh 16-engine rows while the 08a regional query was empty — A9 measurements must use the hub with full-series identity/cluster preserved; the <1-min ramp is not a stable-throughput claim.
- Decision and next action: A9's window ends 19:13:18.607 UTC — before the 19:30 selection, so its final can inform the choice; A10's window follows; record both finals.

### 2026-11-21 22:50 UTC - A9 join validated at 100%; real context distribution measured; training recipe `.115` prepared

- Question or hypothesis and reason for trying it: confirm the request-ID fix end-to-end and capture the real agentic-context population before the 19:30 choice.
- Published source revision: A9/A10 on `858ab5d15bf3118ae0358ff79569ec6b33320a68`; `.115` training recipe `rounds/mochi-throughput-train.yaml` prepared (loop-push pending).
- Command or material configuration: `.115` — REAL 32 steps; incidental max_inflight 1024 with effective 128 unchanged; task slots 1024/worker, executor 256, verifier 64, GenRM 64, progress 43200, Harbor 384, staleness 3, route quotas 16+16 kept; engine provisional until 19:30.
- External inputs and artifact references: worker readout over Rhizome (~22:5x UTC) [R]; receipts `a9-context-partial-1900-analysis.json`, `a9-all-engine-context-1900.json`.
- Environment and hardware when relevant: A9 window ends 19:13:18.607 UTC; A10 first rollout 18:57:58.540 UTC, final due 19:22:58.540 UTC; 128 GPU total; hub fresh/region empty (hub = measurement source).
- Result, including failure or restart: A9 HTTP fix validated on ALL 16 actors — 2834 submitted / 2834 first_token (100% with prompt/context lengths; A8 was 88/1867 = 4.7%). Context distribution (340 active-decoder observations, 60 s refresh): context median 3386 / p90 20040 / p95 27041 / max 48208; prompt median 1242 / p90 19844 / p95 26771 / max 47595. A9 first 9.4 min native: 1891.244 agg / 118.20 tok/s/GPU, 32.85/64 running + 0.41 waiting, KV 29.74%/58.08%, request 3.18, TTFT 7.34 s, prompt 2395.03/GPU, prefix 84.59%, 0 preemptions.
- Evidence: A9 context/native analyses [R].
- Interpretation and limitations: the real agentic-context population is now measured — median context ~3.4k with a heavy tail to ~48k, which reframes the KV-budget discussion versus the synthetic 6.4k-prompt serving arms. A9's early 118.20/GPU is below A5's final 124.97 but window-age differs; the 25-min finals decide.
- Decision and next action: BRIEF final-gate instruction implemented — no new worker/staleness/inflight arms; record both 25-min finals; the 19:30 choice selects the engine (quota decision included) and the 19:45 launch uses `.115`.

### 2026-11-21 23:05 UTC - Final gates landed (`cf7e04df26…`); `.115` training recipe drafted with gate rationale

- Question or hypothesis and reason for trying it: land the final-gate configuration and observability before the real training launch.
- Published source revision: `cf7e04df26c99f74e52d5c39e3cb7b6e593ea8e1` (loop-push; 17 CPU/runtime cases + 8 adopted RL round tests, lint/types pass); config verified present in the worktree (`rounds/mochi-throughput-train.yaml`).
- Command or material configuration: `.115` REAL 32-step recipe, 10 h wall budget — staleness 3 (128 groups × 8 = 1024 trajectories global; 32/worker), max_in_flight pure 1024 (effective 128 unchanged), task slots 1024/worker, executor 256, verifier 64 (+12 Harbor conservative 44), GenRM 64, progress 43200 (> 29,760 retry budget), Harbor 384/job = 12 enforced/worker (≤512), route quotas 16/16 kept, teacher disabled.
- External inputs and artifact references: worker readout over Rhizome (~23:0x UTC) [R]; proofs `final-gates-tests.log`, `final-gates-native-config-event.json`, `a9-symbolic-slot-profile-1902.txt`, `final-gates-landing.log`, `sprint-final-training-preflight.log`.
- Environment and hardware when relevant: running A9/A10 stay on the `858ab5d15b…` source — the new code is observability, not a live patch; NO-SUBMIT native preflight running; no training job submitted yet.
- Result, including failure or restart: gates landed and tested; startup logging now records all concurrency limits, which buffer gate blocked, and structured Harbor/task-slot blocked/resumed events; the symbolic-8 cap is numerically unchanged (py-spy found no math-grader frames — not proof of never-queue).
- Evidence: landing/test receipts [R].
- Interpretation and limitations: the gate values are derived from the measured semantics recorded in this notebook (staleness lease, Harbor slots, quota admission); the engine geometry remains provisional until the 19:30 choice.
- Decision and next action: record the NO-SUBMIT preflight result, A9/A10 finals (19:13:18.607 / 19:22:58.540 UTC), the 19:30 choice, and the 19:45 real `--run`.

### 2026-11-21 23:15 UTC - Matched 9.4-min comparison: A10 (HBM .85) leads A9 early; training-draft preflight PASSED

- Question or hypothesis and reason for trying it: the A9-vs-A10 pair finally isolates the HBM knob on identical code, config, and HTTP identity.
- Published source revision: both on `858ab5d15bf3118ae0358ff79569ec6b33320a68`.
- Command or material configuration: matched first-9.4-min windows (all 16 ranks, 0 resets); training draft compiled assert = 32 steps, 48 trainer + 12 rollout provisional, staleness 3, quotas 16/16, graphs PIECEWISE.
- External inputs and artifact references: worker readout over Rhizome (~23:1x UTC) [R]; receipts `a9-partial-1858-analysis.json`, `a10-matched9min-1907-analysis.json`, `sprint-final-training-validation/6bc29178…/receipt.json`, `final-training-compiled-assertions.json`.
- Environment and hardware when relevant: 128 GPU total; Harbor 384 each; A9 final due 19:13:18.607 UTC; A10 final due 19:22:58.540 UTC.
- Result, including failure or restart: A9 HBM .90 — 1891.24 agg / 118.20/GPU / req 3.185 / 32.85/64 / KV 29.74/58.08% / TTFT 7.339 s / prefix 84.59%. A10 HBM .85 — 2244.33 agg / **140.27/GPU** / req 4.725 / 26.73/64 / KV 27.46/54.98% / TTFT 3.771 s / prefix 80.82%. Prompt 2395.03 vs 1781.09/GPU; cache-query MISS 370.26 vs 342.45. Both 0 preemptions.
- Evidence: matched-window analyses [R]; preflight receipt [R].
- Interpretation and limitations: A10 leads early by 18.7% per-GPU, but the natural real-rollout workload differs between arms — not a precise causal HBM gain; the 25-min finals decide. The training draft is fully validated pending the 19:30 engine choice.
- Decision and next action: record both 25-min finals; the 19:30 choice finalizes the engine geometry and the quota decision; real `--run` 19:45 UTC.

### 2026-11-21 23:30 UTC - FINAL: A9 108.73/GPU — no proven HBM 0.90 gain over A7's 0.85; context tail quantified

- Question or hypothesis and reason for trying it: does HBM 0.90 beat HBM 0.85 on the identical real-rollout configuration (A9 vs A7, and the matched A10 control)?
- Published source revision: `858ab5d15bf3118ae0358ff79569ec6b33320a68`; recipe `rounds/mochi-throughput-ep16-seqs64-kv90.yaml`.
- Command or material configuration: A9 final 25-min native window (all 16 ranks, 0 resets); cancelled 19:14:24 UTC.
- External inputs and artifact references: worker readout over Rhizome (~23:3x UTC) [R]; receipts `a9-final25-analysis.json`, `a9-context-final25-analysis.json`, `a9-all-engine-final25-1913.json`, `a9-native-final25-1913.jsonl`, `a9-context-final25-1913.jsonl`, `sprint-a9-measured-cancellation.log`.
- Environment and hardware when relevant: only A10 runs now (64 GPU, Harbor 384); all old A5/A7/A8/B arms measured and released; workers128 never launched.
- Result, including failure or restart: 1739.60268 aggregate / **108.72517 tok/s/rollout-GPU**; 22.9463/64 running + 0.15436 waiting; KV 33.5594% mean / 62.036% max; request 4.12194; TTFT 4.60466 s; prompt 2424.379/GPU; prefix 89.94489%; 0 preemptions. Last 10 min: 93.38118/GPU, 15.397/64 running, 0 waiting, KV 35.627%, request 6.0437, TTFT 0.9928 s.
- Evidence: final 25-min analysis receipts [R].
- Interpretation and limitations: **HBM 0.90 has no proven real gain over A7's 0.85 (108.73 vs 107.38)** — the matched-source A10 control's full-25min window ends 19:22:58.540 UTC and decides the HBM question properly (its early 140.27/GPU is not final). A9's final context population: 99.935% coverage (4617/4620 IDs, 4393 ended, 224 active-decoder observations); active context median 9146.5 / p90 31251 / p95 37843 / max 48868; prompt median 7113 / p95 36178 / max 48489 — the context tail grows materially over the rollout versus the early-window snapshot (60 s refresh, not the instantaneous population).
- Decision and next action: A10's final at 19:22:58.540 UTC completes the HBM isolation; the 19:30 choice and 19:45 real 32-step training launch follow.

### 2026-11-21 23:45 UTC - Curriculum audit: GenRM cap inactive; symbolic-8 off the active path; raw agentic share 3.49%

- Question or hypothesis and reason for trying it: verify which final-gate caps actually bind on the canonical training curriculum before the 19:30 choice.
- Published source revision: audit run read-only on A10's in-region staged parquet (`.115` cf7 source pins the dataset URIs); no new job or data download.
- Command or material configuration: grader/verifier scan of the staged curriculum + 4-worker stack dump cross-check.
- External inputs and artifact references: worker readout over Rhizome (~23:4x UTC) [R]; receipts `curriculum-grader-counts-1913.txt`, `curriculum-legacy-grader-usage-1917.json`, `curriculum-script-entrypoints-1919.txt`, `a9-symbolic-slot-profile-1902.txt`.
- Environment and hardware when relevant: only A10 runs (64 GPU, Harbor 384, source 858); A9 measured/cancelled FINAL at 108.73/GPU.
- Result, including failure or restart: raw agentic 5,608 rows (all NoGrader/HarborSWE); nonagentic 154,953 rows (all ScriptGrader/Verifyit). GenRM requires legacy Nemotron NoGrader + harness — zero such nonagentic rows exist, so the GenRM cap is INACTIVE on this curriculum. All 25 unique inline Python/shell verifier resources scanned: no `_SYMBOLIC_SLOTS`/math_with_judge/genrm/max_concurrent_comparisons references; SkyRL utility imports only in NemoUltra/LlamaPost scripts (sample entrypoints calendar/answer-extraction and IFEval). The symbolic-8 cap is not observed on the active path (4-worker stack dump: 0 math frames). Raw-pool agentic share 3.49% — NOT the consumed-batch share; A8's 0-consumed remains UNAVAILABLE.
- Evidence: audit receipts [R].
- Interpretation and limitations: the GenRM-64 raise in the final config is non-binding/inactive; the symbolic-8 cap stays numerically unchanged; teacher disabled. The 3.49% raw share reinforces NOT claiming A8's 208.63/GPU as a like-for-like 50/50 training gain — the consumed-batch mix remains unmeasured.
- Decision and next action: A10's full-25min final at 19:22:58.540 UTC decides the HBM question; selection 19:30 UTC; real training 19:45 UTC.

### 2026-11-21 23:55 UTC - TRAINING SELECTION: A10 geometry (EP16/seq64, HBM 0.85, 64 GB200) chosen at 2.169x control

- Question or hypothesis and reason for trying it: the sprint's central decision — which real-rollout geometry trains first.
- Published source revision: selection measured on `858ab5d15bf3118ae0358ff79569ec6b33320a68` (A10 root `/benfeuer/long-loop-mochi-throughput-ep16-httpid-control-2026.11.21.114-coordinator-4fa04020`); final `.115` recipe on the cf7 gates.
- Command or material configuration: selected = EP16/DP16/TP1, seq64, HBM 0.85, batch 8192; 48 trainer + 16 rollout = 64 GB200; `.115` changes ONLY DP12→16, EP12→16, seq32→64 with all cf7 gates retained (Harbor 384, staleness 3, quotas 16+16).
- External inputs and artifact references: worker selection readout over Rhizome (~23:5x UTC) [R]; receipts `a10-final25-analysis.json`, `a10-native-final25-1923.jsonl`, `a10-context-final25-1923.jsonl`, `a10-all-engine-final25-1923.json`, `sprint-a10-measured-cancellation.log`, `selected-training-landing.log`.
- Environment and hardware when relevant: both A9/A10 finals retained and cancelled; lane at 0 GPU; post-rebase native preflight in progress.
- Result, including failure or restart: A10 FINAL 25-min — 1940.10955 aggregate / **121.25685 tok/s/rollout-GPU**; 20.4122/64 running + 0.04283 waiting; KV 39.3497% mean / 84.7560% max; request 5.71435; TTFT 2.13218 s; prefix 90.37887%; 0 preemptions; last 10 min 108.03/GPU with near-zero queue. Observed 2.169x per-GPU and 2.892x aggregate vs the `.103` control (55.90/670.82); short windows are NOT an update-wall-clock guarantee.
- Evidence: A10 final receipts [R].
- Interpretation and limitations: same-source HBM 0.85 is 11.53% above the 0.90 observationally (single-window traffic varies); A8's quota-off 208.6/GPU is not eligible like-for-like (0 consumed batches, mix unknown, 16/16 routes kept in current semantics); A3/A5's older staleness-11 is not retained for training. The selection preserves staleness 3 and 50/50 routes.
- Decision and next action: REAL `--run` of `.115` at 19:45 UTC (14:45 EST) on the authorized schedule; the secretary verifies the launch past its first update before closing the campaign records.

### 2026-11-22 00:10 UTC - FINAL TRAINING CONFIG LANDED (`5957d8bb…`) and preflight-passed; 19:45 `--run` armed

- Question or hypothesis and reason for trying it: the selected A10 geometry is now a landed, compiled-asserted training recipe ready for the real launch.
- Published source revision: **`5957d8bbcf0cd945243cdd544eb4433f7e36ac88`** (loop-push; verified in the worktree — `rounds/mochi-throughput-train.yaml` carries version 2026.11.21.115).
- Command or material configuration: REAL 32 steps / 10 h budget; 64 GB200 (48 trainer + 16 rollout); TP1/DP16/EP16; seq64 / HBM 0.85 / batch 8192; PIECEWISE graphs; r136 excluded; PRODUCTION 08a; workers 32 (CPU 2), Harbor 384 ≤ 512, staleness 3 / quotas 16+16; pure caps (max_inflight 1024 effective 128, task 1024, executor 256, verifier 64, GenRM 64, progress 43200); teacher disabled (legacy Math-8/GenRM paths unused by the canonical curriculum).
- External inputs and artifact references: worker readout over Rhizome (~00:0x UTC) [R]; exact preflight `sprint-final-training-selected-validation/c4e0e6f33f8e30c03e9375f737a6ecc74bef0e7a227eff37cc68be49cc230f66/receipt.json`; proofs `selected-training-compiled-assertions.json`, `selected-training-landing.log`, `sprint-launch-wake.json`.
- Environment and hardware when relevant: all probes cancelled — lane at 0 GPU; the launch one-shot is armed for 19:45 UTC (single; guard `sprint-training-submission.log`, round runner `--run` exactly once).
- Result, including failure or restart: EXACT final native post-rebase preflight PASSED; compiled assertions prove 64 GPU / 32 steps / gates / default staleness 3 / quotas / PIECEWISE graphs on the exact source; the prior cf7 engine-12 preflight is superseded.
- Evidence: preflight + compiled-assertion receipts [R]; worktree verification [R].
- Interpretation and limitations: the recipe is ready; the REAL run is not a 25-min cancel probe — measured A10 rates are first-25min observations, not an update-wall-clock promise.
- Decision and next action: capture the training root job ID at 19:45 UTC, verify the run past its first update, then write the Conclusion and close marin#11954 and this logbook per the campaign rules.

### 2026-11-22 00:30 UTC - Phase-2 retained-data analysis: serving-vs-real gap decomposition; KV exhaustion not a proven bottleneck

- Question or hypothesis and reason for trying it: explain the serving-to-real rollout gap from retained data before the launch (no config change).
- Published source revision: analysis over retained receipts; final training `5957d8bbcf…` / `rounds/mochi-throughput-train.yaml` `.115` unchanged.
- Command or material configuration: matched HBM .85 EP16/seq64/batch8192 comparison (serving `3a859712f5…`/`530ab73b76…` benchmarks vs real `858ab5d15b…`); bench-spec check.
- External inputs and artifact references: worker readout over Rhizome (~00:3x UTC) [R]; receipts `serving-vs-real-gap-decomposition.json`, `a10-request-speed-by-concurrency-context.json`, `a9-request-speed-by-context-final25.json`.
- Environment and hardware when relevant: all probes cancelled; 19:45 UTC one-shot verified loaded (no duplicate, submission log absent).
- Result, including failure or restart: serving 552.147/GPU at 59.79/64 running vs REAL A10 121.257/GPU at 20.41/64 running — approximate descriptive split of the 4.55x gap = 2.93x running-request count x 1.55x effective generated tok/s per running-request second (9.23 vs 5.94); NOT causal/TPOT (running includes prefill; rank-sample vs RL time-weighted). Best B HBM90 831.73/GPU (61.51 running, prefix 99.78%) is 6.86x real; HBM 0.90 did NOT transfer real (A9 108.73 vs A10 121.26). Zero real preemptions + underfilled late engines ⇒ KV exhaustion is not a proven bottleneck on the selected arm. A10 per-request >=30s decode-progress intervals by running bucket: 1–16 → 6.91 tok/s (n=1471), 17–32 → 6.51 (n=4321), 33–48 → 4.23 (n=1164); no 49–64 interval. A9 completed-request context bins NOT monotonic (<4k median 3.29 vs 32–64k 5.97 tok/s) — "long context alone causes all slowdown" unproven (completion censoring + time/concurrency confound). Bench spec CONFIRMED to include routed-expert capture + logprobs true/top-0.
- Evidence: retained-data analyses [R].
- Interpretation and limitations: descriptive, not causal; the residual difference needs a future engine profile, not unmeasured config changes before launch.
- Decision and next action: capture the 19:45 UTC training launch, verify past the first update, then close the records.

### 2026-11-22 00:45 UTC - Prelaunch guard verified; CUDA-graph receipt and sampling parity added to the record

- Question or hypothesis and reason for trying it: confirm the launch prerequisites before the 19:45 UTC `--run`.
- Published source revision: working tree clean at `5957d8bbcf0cd945243cdd544eb4433f7e36ac88`; no config change.
- Command or material configuration: guard checks — single loaded timer (main self-wake == launch JSON; `sprint-training-submission.log` absent), no duplicate timer, no old-arm resubmission.
- External inputs and artifact references: worker readout over Rhizome (~00:4x UTC) [R]; receipts `a10-cuda-graph-capture-receipt.txt` (18:54:36/18:55:16 "Graph capturing finished", 16 rank repeats), `sprint-launch-wake.json`.
- Environment and hardware when relevant: lane at 0 GPU; all A/B probes measured/cancelled; actual A-job Daytona peaks unavailable (B all 0).
- Result, including failure or restart: guard verified. Sampling parity: pure-B and RL both temperature 0.7 / top_p 1, logprobs true (top-0), routed-expert capture ON — not a greedy benchmark shortcut. Residual serving-vs-real workload differences (fixed 32-shared-prompt pool / 1024 ignore-EOS / post-120s warmup vs fresh variable-length RL contexts / EOS / first-25min window) mean the residual gap is not an isolated kernel cause.
- Evidence: guard receipts [R]; CUDA-graph receipt [R].
- Interpretation and limitations: prerequisites hold; the residual gap analysis remains descriptive.
- Decision and next action: capture the training root ID at 19:45 UTC, verify past the first update, then close the records.

### 2026-11-22 01:00 UTC - REAL TRAINING SUBMITTED: `.115` at 19:45 UTC (job …-train-2026.11.21.115-coordinator-49544525)

- Question or hypothesis and reason for trying it: launch the selected A10 geometry as real RL training — the campaign's objective.
- Published source revision: `5957d8bbcf0cd945243cdd544eb4433f7e36ac88` (recipe `rounds/mochi-throughput-train.yaml` v2026.11.21.115).
- Command or material configuration: standard round runner `--run` exactly once at 19:45 UTC (single one-shot; guard `sprint-training-submission.log`); 64 GB200 (48 trainer + 16 rollout), TP1/DP16/EP16, seq64/HBM 0.85/batch 8192, REAL 32 steps / 10 h budget, all cf7 gates, teacher disabled.
- External inputs and artifact references: worker submission notice over Rhizome (~01:0x UTC) [R].
- Environment and hardware when relevant: PRODUCTION 08a, r136 excluded, graphs on.
- Result, including failure or restart: **training submitted** — root `/benfeuer/long-loop-mochi-throughput-train-2026.11.21.115-coordinator-49544525`. This is real training, not a 25-min cancel probe.
- Evidence: worker submission notice [R]; the preflight/compiled-assertion chain from the prior entries.
- Interpretation and limitations: measured A10 rates are first-25min observations, not an update-wall-clock promise; the run must be verified past its first update before any training-throughput claim.
- Decision and next action: verify the run past its first update, then write the Conclusion (selected approach, rejected alternatives, reproduction, remaining uncertainty) and close marin#11954 and this logbook per the campaign rules.

### 2026-11-22 01:15 UTC - Training GPU child admitted; post-selection FP8/offload probe direction (A11/A12) recorded

- Question or hypothesis and reason for trying it: track the real training's startup and the new post-selection probe request without disturbing the training run.
- Published source revision: training on `5957d8bbcf0cd945243cdd544eb4433f7e36ac88`; A11/A12 probes to be prepared on the requested EP12 baseline.
- Command or material configuration: A11 (FP8) and A12 (native-offload) — fresh probes; ONE probe alongside training (124 GPU total).
- External inputs and artifact references: worker readout over Rhizome (~01:1x UTC) [R]; receipt `training-submission-receipt.json`.
- Environment and hardware when relevant: training GPU child `…-train-2026.11.21.115-a25d2a2abc62` admitted; startup one-shot 20:00 UTC loaded; no polling.
- Result, including failure or restart: training running; probes in preparation (not launched).
- Evidence: submission receipt [R]; worker readout [R].
- Interpretation and limitations: the historical "20% prefix" diagnosis is superseded (latest EP16 prefix 90.38%, 0 preemptions). The FP8/offload probes must never alter or cancel the real training.
- Decision and next action: record A11/A12 landings/launches when they occur; verify training past its first update; then close the records.

### 2026-11-22 01:30 UTC - A11 FP8-KV submitted on the EP12 baseline; A12 native-offload prepared but held

- Question or hypothesis and reason for trying it: the post-selection FP8/offload request — does `kv_cache_dtype=fp8` lift real-rollout throughput on the EP12 baseline?
- Published source revision: A11 at `ce13e39e8435a9149a1268d6b25db54c32f75411` (recipe `rounds/mochi-throughput-kv-fp8.yaml`, `.116`); A12 `.117` prepared/validated, not submitted.
- Command or material configuration: A11 = EP12/seq32/HBM 0.85/batch 8192 with ONE engine knob `kv_cache_dtype=fp8`; Harbor 384, default staleness 3, quotas 16+16; graphs on; PRODUCTION 08a; r136 excluded; 60 GB200.
- External inputs and artifact references: worker readout over Rhizome (~01:3x UTC) [R]; receipts `sprint-kv-fp8-submission.log`, `sprint-kv-fp8-validation/90c48929…/receipt.json`.
- Environment and hardware when relevant: training 64 + probe 60 = 124 ≤ 128; the real training is untouched. A12's 160 GiB/GPU draft capacity needs a physical layout check; the pinned connector requires `enable_cumem_allocator=true` with existing expandable_segments.
- Result, including failure or restart: A11 submitted; A12 held until A11 is measured/released.
- Evidence: submission + preflight receipts [R].
- Interpretation and limitations: single-knob delta on the EP12 baseline; the historical A5 staleness-11 comparison remains a caveat — no isolated-gain claim. Fresh labels A11/A12 avoid colliding with the used A9/A10.
- Decision and next action: record A11's measured window; launch A12 after A11 releases; verify training past its first update; then close the records.

### 2026-11-22 01:45 UTC - Bounded startup (20:00 UTC): training CUDA graphs captured; A11 NCCL rendezvous; A12 draft capacity caveats

- Question or hypothesis and reason for trying it: confirm both running jobs' startup health before any throughput claim.
- Published source revision: training `5957d8bbcf…`; A11 `ce13e39e84…`.
- Command or material configuration: bounded startup log capture at 20:00 UTC.
- External inputs and artifact references: worker readout over Rhizome (~01:4x UTC) [R]; receipts `training-startup-2000.jsonl`, `training-native-startup-2000.jsonl`, `kv-fp8-bounded-startup-2000.txt`.
- Environment and hardware when relevant: training 64 + A11 60 = 124 GPU.
- Result, including failure or restart: training GPU child a25d2a2abc62 — successful CUDA graph captures 19:58:02/19:58:52 (rank-repeated), policy workers start 19:59:17; no rollout/native gate event yet, so throughput/window not claimed; known custom-model config-parser ERROR/retry strings also occurred in successful prior arms; no Traceback/RuntimeError/OOM in the bounded capture. A11 GPU child a3902907d2a8 found, NCCL rendezvous 19:59:35, no decode measurement yet.
- Evidence: bounded startup receipts [R].
- Interpretation and limitations: startup-only state — no throughput claim for either job. A12's 160 GiB/GPU offload draft is logical-only (2.671875 GiB/64k prefix excludes physical padding/short conv; the average 32 Harbor conversations/engine is not worst-case coverage) — actual layout/session skew must be validated before launch.
- Decision and next action: bounded qualification/marker checkpoint 20:12:27 UTC; then A11's actual rollout + 25-min/cancel; A12 launches only after layout validation and A11's release.

### 2026-11-22 02:20 UTC - New direction: A13 `.118` planned (Harbor 512 is the real knob); A12 deprioritized

- Question or hypothesis and reason for trying it: the user's 13:36 EST direction — EP16 is a regime change, so probe the Harbor admission allowance next.
- Published source revision: A13 `.118` to be landed via loop-push/native preflight (source SHA to follow); training `.115` and A11 `.116` unchanged.
- Command or material configuration: A13 = EP16/seq64/HBM 0.85/batch 8192 BF16 + Harbor 384→512 + max_in_flight 128→192; route 16/16, staleness 3 retained; no staleness change; no third arm, nothing >512.
- External inputs and artifact references: worker readout over Rhizome (~02:2x UTC) [R].
- Environment and hardware when relevant: sequencing — A11 (60 GPU) measured then released before A13 (64 GPU) launches, 128 total; training untouched.
- Result, including failure or restart: plan only — nothing launched.
- Evidence: worker readout [R]; source check `buffer.max_untrained_groups` = (staleness 3 + 1) × batch 32 = 128, so max_in_flight 192 does not raise the effective lease lookahead — Harbor 512 is the actual changed admission allowance.
- Interpretation and limitations: demand-limit gain ceiling from Harbor alone is 1.33x if the model-time fraction is stable — not a prediction or proven gain; occupancy will be measured rather than claiming a 192-group effect. A13 takes priority over the draft A12 offload (the old 20%-prefix diagnosis no longer describes EP16).
- Decision and next action: record A13's landing/launch when they occur; A11's window remains the next measurement.
