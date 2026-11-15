# D-sweep run-cancellation audit — findings of a local audit [R]; job identities now published

- **What it is:** sanitized findings of a read-only local audit of the Snowball IsoFLOP phase-D run terminations on cluster `cw-rno2a` (2026-11-15). All times UTC; EST = UTC−05:00. Job IDs, task suffixes, exit codes, timestamps and budget figures are published below. This file publishes the audit's findings verbatim in aggregate form; **no OOM events were found in the queried logs**.
- **Compiled at:** 2026-11-15T15:14Z
- **Sources:** [R] `/Users/benfeuer/Documents/experiments/active/claim-audits/snowball-d-kills.report.md` (primary, full findings); [R] `.../snowball-d-kills.log` (supporting query transcript, 7,798 lines — consulted selectively, not read in full); [R] local audit-queried sources: Finelog hub `marin`/`log` rows and regional `iris.task_event`, controller DB attempts (queried read-only; regional workload-log queries returned no matching rows while the hub retained the attempts).
- **Access class:** job identities, states, timestamps, exit codes and budget aggregates are publishable per ruling. Audit scope was read-only; no jobs, branches, configurations, controllers or clusters were changed.

## Verdicts

1. **REFUTED — "preemption waves" killed D-r2/r3/r4.** D-r2 and D-r3 cancelled themselves after a child watchdog failure; D-r4 was cancelled by the experiment's separate phase-budget guard; D-r5 then followed the same self-cancellation path. Every audited task has exactly **attempt 0**, with **zero PREEMPTED attempts**. No evidence of a "10th wave".
2. **CONFIRMED — D-r3 d1024 muonh lr≈1.05e−4 (exact LR 0.000104957, child suffix `ae246dc12b8d`) hit the 1800-second watchdog; all eight ranks exited 124; no recorded OOM.** Qualification: the precise signal is "No progress after process_started", before any recorded training loss. This establishes a startup/first-update progress timeout; it does not establish the underlying cause or prove an optimizer kernel had begun.
3. **UNSUPPORTED — "D-r5 was running clean, zero stalls" around 12:20 UTC.** RUNNING for ~41 minutes with no scheduler preemption at that instant is supported; training health is not — its d1024 child hit the watchdog at 12:21:52 UTC and the root self-cancelled at 12:23:20.749 UTC.

## Per-root cancellation table

| Root | Root death UTC / EST | Initiating cause and issuer |
|---|---|---|
| D-r2 | 08:38:25.483 / 03:38:25.483 | Its d1024 muonh lr5e−5 child failed with watchdog exit 124; the root's own `budgeted_map` exception handler called `context.client.job(context.job_id).cancel()`. Two remaining children killed by parent cascade; four had succeeded. |
| D-r3 | 10:07:49.618 / 05:07:49.618 | Its d1024 muonh lr0.000104957 child failed with watchdog exit 124; same root self-cancellation. Two children killed by cascade; four had succeeded. |
| D-r4 | 11:20:20.352 / 06:20:20.352 | `/benfeuer/snowball-isoflop-budget-guard-allocated-v10` reached the phase-D ceiling and cancelled r4 ("Terminated by user" audit; guard log "Stopped registered root ... at phase D limit"). Three children killed by cascade; three had succeeded. The root's later exception handler issued a redundant cancellation. |
| D-r5 | 12:23:20.749 / 07:23:20.749 | Its d1024 **adam** lr3.84615e−6 child failed with watchdog exit 124; same root self-cancellation. d512 killed by cascade; d768 had succeeded. |

The primary evidence identifies **agent-authored code running inside Iris** as the cancellation source. Controller audit lines do not include a caller IP/principal, so they cannot independently exclude an additional redundant CLI request.

## Counter-defect code references (file:line + effect) [R: local files]

Public pinned-code links added round 7: the Iris controller references resolve at main commit [`e9ae26899c90512195e68f0edec808bb42b88bf3`](https://github.com/marin-community/marin/commit/e9ae26899c90512195e68f0edec808bb42b88bf3) — the controller version (`e9ae268 (main) (runner)`) reported by `iris cluster status` during the audit — and the experiment-code references resolve on the pushed branch `snowball-isoflop-ablations` at commit [`f103997ccd80c6d889f7677552efebaba432d329`](https://github.com/marin-community/marin/commit/f103997ccd80c6d889f7677552efebaba432d329) (the pre-fix snapshot; the watchdog-classification fix [`f9a12ef08c`](https://github.com/marin-community/marin/commit/f9a12ef08c10bb4118c2c65436f5e3d5825de090) landed after the audit). Line anchors below were re-verified against those commits; where the audit's cited line number differs slightly from the verified anchor, both are given.

- `lib/iris/src/iris/cluster/controller/attempt_counts.py` — [`#L42–L50`](https://github.com/marin-community/marin/blob/e9ae26899c90512195e68f0edec808bb42b88bf3/lib/iris/src/iris/cluster/controller/attempt_counts.py#L42-L50) (audit cite `:40`): includes `TASK_STATE_KILLED` ([#L49](https://github.com/marin-community/marin/blob/e9ae26899c90512195e68f0edec808bb42b88bf3/lib/iris/src/iris/cluster/controller/attempt_counts.py#L49)) in `PREEMPTION_ATTEMPT_STATES`; a non-null attempt start charges the counter, so a normal explicit cancellation of a started task yields `preemptions=1`. It is **not** a count of proven Kueue evictions.
- `lib/iris/src/iris/cluster/controller/jobs.py:1108` — [`def terminate_job(` at #L1108](https://github.com/marin-community/marin/blob/e9ae26899c90512195e68f0edec808bb42b88bf3/lib/iris/src/iris/cluster/controller/jobs.py#L1108): TerminateJob uses the generic reason `Terminated by user` whether called by CLI or application; root `failures=0` describes its own attempt; the DB records null exit codes for killed attempts, so a CLI `exit=0` presentation does not establish successful process completion.
- `experiments/post_training/snowball_isoflop/budget.py` (audit cite `:105`) — the completion/exception handler at [`#L105–L112`](https://github.com/marin-community/marin/blob/f103997ccd80c6d889f7677552efebaba432d329/experiments/post_training/snowball_isoflop/budget.py#L105-L112): logs an exception and cancels the root on **any** task or budget-monitor exception; the r2/r3/r5 tracebacks originate at `results[name] = future.result()` ([#L107](https://github.com/marin-community/marin/blob/f103997ccd80c6d889f7677552efebaba432d329/experiments/post_training/snowball_isoflop/budget.py#L107), handler `except`/log/cancel at #L109–L112) — the child-failure trigger, not the per-root budget exception.
- `experiments/post_training/snowball_isoflop/guard.py` — [`#L83`](https://github.com/marin-community/marin/blob/f103997ccd80c6d889f7677552efebaba432d329/experiments/post_training/snowball_isoflop/guard.py#L83) checks `charged_h100_hours + active_gpu_reservation / 60 >= ceiling_h100_hours`, then cancels registered roots at [`#L113–L118`](https://github.com/marin-community/marin/blob/f103997ccd80c6d889f7677552efebaba432d329/experiments/post_training/snowball_isoflop/guard.py#L113-L118).
- `experiments/post_training/snowball_isoflop/run_probes.py` (audit cite `:56`; pre-fix snapshot) — `_failure_signals` ([#L55](https://github.com/marin-community/marin/blob/f103997ccd80c6d889f7677552efebaba432d329/experiments/post_training/snowball_isoflop/run_probes.py#L55), exit-code scan #L59–L64) set `watchdog_exit` only by iterating `error.status.tasks[*].attempts[*].exit_code`; a live GetJobStatus for r5's failed child returned `state=JOB_STATE_FAILED, failure_count=1, preemption_count=0, task_count=1` **without tasks/attempt details**, so this path could leave `watchdog_exit=False` despite "exit code 124" in the error text, re-raise, and trigger whole-tree cancellation (r5 traceback confirms the escape). The subsequent fix [`f9a12ef08c`](https://github.com/marin-community/marin/commit/f9a12ef08c10bb4118c2c65436f5e3d5825de090) classifies watchdog stalls from the terminal error text.
- Local `PLAN.md:905–911` records the "eighth/ninth wave" explanation; those notes are contradicted by the primary logs above. No files were fixed by the audit.

## Query scope and OOM findings

- D-r3's failed child: bounded hub log query returned **19,280 rows, 8 rank-exit-124 lines, 0 train/loss lines, and 0 RESOURCE_EXHAUSTED / out-of-memory / OOMKilled lines**.
- D-r5's failed child: **19,284 / 8 / 0 / 0** respectively.
- Neither failed child has an OOMKilled task event. **State of the evidence: no OOM events in queried logs** ("No recorded OOM" is the supported conclusion; empty regional job/process logs cannot establish absence of errors because regional historical workload-log queries returned no matching rows).
- Failed-child suffixes queried: r2 `7e51038ed5b9`, r3 `ae246dc12b8d`, r5 `85da4fc5abad`.

## Budget figures

Guard ceilings [R: local snapshots under `experiments/active/snowball-isoflop-ablations/artifacts/pretrain-baseline-revision2/native-cpu-v10/resume-20261114/`]: **D ceiling 96** in `guard-config-after-d-r2.json`, `...r3.json`, `...r4.json`; `...r5.json` raises it to **143.802171**. The cap covers r1–r4 together, not r4 alone.

Reconstructed completed GPU attempt durations (controller attempt start/end pairs; terminal total, not the exact pre-cancel poll snapshot):

| Root | H100-hours |
|---|---:|
| D-r1 | 17.594962 |
| D-r2 | 28.392260 |
| D-r3 | 29.115416 |
| D-r4 | 20.699533 |
| **Total** | **95.802171** |

This verifies the ~**95.8 of 96** accounting at the end of r4. With three active eight-GPU children the guard's one-minute allowance is **0.4 H100-hours**; stopping below 96 is expected. Reading the saved object `.../campaign-budget-guard-allocated/limit-reached.json` via `fsutil cat --raw` was unavailable because local CoreWeave credentials were missing; no credentials were printed or changed.

## Task/attempt census (27 audited tasks; all attempt IDs 0)

- Database attempt states: **12 SUCCEEDED, 3 FAILED, 12 KILLED**.
- Regional task-event counts: 12 JobCompletedTaskSucceeded, 3 TaskTerminated, 4 JobFinalizedTaskKilled, 8 ParentJobTerminated, 3 container Error, 24 SchedulingGated, 18 PodInitializing, 8 Unschedulable. No preemption/eviction/disruption reason appears for this tree; "Preemption is not helpful for scheduling" scheduler messages are admission diagnostics, not preemption evidence.

Full per-task timeline (submitted / attempt started / terminal, UTC; outcome = DB attempt exit code) is a 27-row table in the primary report [R: `snowball-d-kills.report.md` lines 94–122]; highlights: the three FAILED exit-124 children are D-r2 `7e51038ed5b9` (08:38:07.163), D-r3 `ae246dc12b8d` (10:07:28.191), D-r5 `85da4fc5abad` (12:22:49.004); all other non-succeeded tasks were KILLED / null (parent cascade or guard).

## Other users and cluster health (context)

- 07:20–12:30 UTC shows real Kueue priority reclamation affecting another user's workloads (60 WorkloadEvictedDueToPreempted observations across 39 tasks; user identifier redacted from this public ledger) and a second user (1 event), with explicit preemptor/preemtee paths — distinct from the Snowball cancellations. In ±60-second windows around each of the four Snowball root deaths, other users have **zero** recorded preemption/eviction/disruption/cancellation or failed-task events; the interval has zero k8s/disruption events; cluster status at 12:30:49 UTC reported Running/Healthy. **No cluster-wide killing event or broken scheduler/controller is demonstrated.** Whether the progress timeouts arise from compilation, collectives, code or node behavior remains unresolved.

## Reproduction

Read-only reproduction commands (controller SQL, Finelog queries, ±60-second cluster comparison, budget reconstruction SQL) are in the primary report [R: `snowball-d-kills.report.md` lines 124–221]; they reference no credentials and no secret values.
