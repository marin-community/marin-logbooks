---
title: "Mochi action loops and loop guards"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/86"
related_issues: ["https://github.com/marin-community/marin/issues/12264", "https://github.com/marin-community/marin/issues/12240"]
research_branch: "codex/86-mochi-loop-guards"
marin_base_revision: "5fedd5a5fa1ecbd2c832a0863a05ef4c56a94395"
archive_template_revision: "3846d697465e8629294936c6a29e2061bb1c296d"
status: active
---

# Mochi action loops and loop guards

## Question and decision criteria

Measure repeated actions in Mochi SFT 400407, then compare a stall stop, repetition penalty 1.1 and paired antithrash RL. The predecessor Dr Doom experiment ended without a gain, so both RL arms initialize from SFT 400407.

The frozen panel has 823 trials: TB2 89×3, TBLite 100×2, SWE-bench Verified 100×1 and SWESmith held-out 128×2. The 128 held-out SWESmith rows are excluded from the training pool. The full protocol is in `cohorts/manifest.json`; its canonical SHA256 is `562bd617cd6d38369ae1c8d534b2df769888a8eec59d0017578bd001be6ef985`.

A target cohort has at least 10 confirmed action-loop trials in S0. The stall stop must halve repeat-turn share; repetition penalty must reduce its loop upper bound to at most half the baseline confirmed loop count. The preregistered RL measure is repeat-turn share against the concurrent antithrash-off RL control. Every target cohort must satisfy its loop criterion. Per-cohort passes must be at least control minus max(3, 5% of trials); pooled passes may lose at most 2% of panel trials. Loop gains cannot offset a correctness breach. Non-target cohorts remain measured and must not regress.

Repeat-turn share is repeated assistant turns inside detected action loops divided by all assessed assistant turns. Loop UB is confirmed looping trials plus unassessed trials. Infrastructure failures are excluded from scored outcomes and add unassessed trials to UB. Known agent timeouts retain their verifier outcomes. At least 90% of each cohort must be valid before sealing. Token-tail loops are reported separately and never substitute for action-loop counts.

The fixed clock begins 2026-11-25 17:01:36 UTC and ends 2026-11-27 17:01:36 UTC. No new RL update starts after 2026-11-27 01:01:36 UTC. Drop the RL comparison if it cannot start by 2026-11-26 13:01:36 UTC. Fix equal update count N from measured S0 timing before submitting either RL arm. The optional SFT-filter audit is blocked on the pinned 216-store manifest; no SFT training is planned.

## Baseline

- Initializer: `open-athena/Marin-Mochi-535B-A23B@sft-fullab-v2-step-400407`, bound through its approved regional copy. Export manifest SHA256: `1a7886b05df25338a12642b25f25525c0eb6c5551384934a2f089d7ec5b4a88d`.
- Sampling: temperature 0.7, top-p 1, seed 20261125, paired task/repetition/turn seeds across arms. Stop token IDs 128001/128009. Frozen task limits and image/resources remain unchanged.
- Hardware: cw-us-east-08a, production priority; 128 GB200, 32 hosts × 4 GPUs; eight DP16/EP16 engines, TP1/BF16, CUDA graphs enabled. Panel host memory 128 GiB based on measured peak plus headroom. All racks eligible by the user's subsequent directive.
- Native entrypoint: `uv run python -m experiments.post_training.mochi_loop_guards --preset s0-baseline --version 2026.11.25.9 --run --wait` with regional storage credentials and EVAL provider credentials supplied out of band. Inspect `__main__.py` for environment binding and `--plan` for the resolved artifact graph. Never print credentials.
- GPU source at initial archive creation: [595a1368f1354eac4ff13c54ce805e142bf7a4ec](https://github.com/marin-community/marin/commit/595a1368f1354eac4ff13c54ce805e142bf7a4ec). CPU audit source at archive creation: [5fedd5a5fa1ecbd2c832a0863a05ef4c56a94395](https://github.com/marin-community/marin/commit/5fedd5a5fa1ecbd2c832a0863a05ef4c56a94395).
- Result: corrected S0 is sealed on823 trials,796assessed/27infra, allfourcohortsvalid. FullstopFAIL/no-repeat; originalRPFAIL with one full-armrepeat triggered. See the dated measurements below.
- Execution identities and operational evidence: [tracking milestone](https://github.com/marin-community/marin/issues/12264#issuecomment-7175699574). Raw traces and diagnostics remain in their regional artifact stores, outside this public archive.

## Entries

### 2026-11-25 17:01 UTC — Start the fixed clock

Approved prep was published at `9d47ffc834`. Execution keeps its panel, detector and interventions; native binding and configuration repairs are recorded separately. Earlier pending requests were replaced only after correcting execution defects or applying explicit resource directives. No results from these attempts enter a scientific comparison. The initial rack exclusion was removed on direct user instruction; all layouts now permit every rack. The clock never resets on recovery.

### 2026-11-25 21:17 UTC — Repair missing native session TTL

S0 v5 placed but rollout worker construction raised `ConfigAttributeError: Missing key environment.task_sessions.daytona.ttl_minutes`, then the GPU gang hit its task-failure limit. The exact physical attempt was checked in hub Finelog with origin cluster and a bounded timestamp window. This was configuration failure before usable collection, not a model result.

[Source a31da3f1eabc2ae26a5925fe7a2058400e16d720](https://github.com/marin-community/marin/commit/a31da3f1eabc2ae26a5925fe7a2058400e16d720) explicitly binds base and evaluation TTL to 480 minutes, covering the frozen eight-hour attempt backstop. Task timeouts are unchanged. Native factory construction, real plan, lint and typecheck passed. [Recovery evidence](https://github.com/marin-community/marin/issues/12264#issuecomment-7174908851).

### 2026-11-25 21:52 UTC — Use declarative evaluation image builds

S0 v6 initialized engines and began TB2, then repeated `Snapshot quota exceeded. Maximum allowed: 45` during setup. The native operational guide prescribes declarative EVAL builds for this panel; persistent snapshot provisioning cannot cover its unique task images. Only the owned v6 root and child were retired; shared snapshots and other jobs were untouched. Partial v6 results are invalidated.

[Source 595a1368f1354eac4ff13c54ce805e142bf7a4ec](https://github.com/marin-community/marin/commit/595a1368f1354eac4ff13c54ce805e142bf7a4ec) sets base/eval image builds to declarative, retaining TTL 480, exact image digests, resources and task timeouts. Native factory enum/TTL checks, real plan, lint and typecheck passed. V7 placed all 32 hosts and captured CUDA graphs. TB2 began at 22:19:42 UTC. At 22:32:42 UTC, 94/267 TB2 trials had returned. Returned counts are neither passes nor a validity estimate; persisted records must be audited first. No snapshot-quota errors appeared in that window.

### 2026-11-25 22:18 UTC — Clarify CPU evidence identity and reporting

Counterfactual samples now preserve attempt ID, native UID and ordinal so repeated trials remain distinguishable. Markdown reports explicitly separate token-tail confirmed/UB counts from action-loop confirmed/UB counts; detector thresholds and JSON metrics are unchanged. Thirty-eight focused audit/counterfactual tests and required lint pass. These CPU-only changes do not restart or alter the active v7 GPU bundle.

Next: finish S0, audit every cohort, compute stall-stop counterfactuals and hand-check up to 30 fired trials. Seal target cohorts before S1. S1 runs stop and repetition-penalty arms concurrently at 64 GPUs each. Then use measured timing and checkpoint overhead to fix S2 N and retain a common completed checkpoint if the T+32 cutoff binds. Export both RL arms and run S3 against their paired control.

### 2026-11-25 22:47 UTC — Make the RL threshold explicit

The RL criterion above is quantitative: treatment repeat-turn share must be no greater than 0.5 times the concurrent RL control on every S0-defined target cohort, with correctness held. This measure was registered before baseline results in [the tracking issue](https://github.com/marin-community/marin/issues/12264#issuecomment-7171525899); this entry clarifies the notebook rather than changing the criterion. Target membership is inherited from S0, never reselected from the trained control.

Non-target cohorts do not need a 50% loop reduction. Report all registered loop measures and flag increases rather than hiding them in a pooled gain; the design does not specify a separate numerical loop-regression tolerance. The correctness tolerances and validity requirement above remain applicable. No baseline result is available yet.

### 2026-11-25 23:00 UTC — Spread CPU workers after a driver host OOM

S0 v7 failed at approximately 22:48 UTC, after 149/267 TB2 trials had returned. Task 0 exited 137 with OOMKilled; its 31 peers stopped with the gang. No completed baseline or intervention result is inferred from this partial collection.

Hub Finelog `iris.task`, exact task-0 attempt 0 and origin cw-us-east-08a, recorded 54 samples from 21:54:38 through 22:47:44 UTC. The last sample was 130,860 MiB (127.79 GiB) against the 128 GiB host limit. The other 31 hosts peaked at at most 65.2 GiB. Task 0 grew from 44.1 GiB before rollout workers to 83.8 GiB after startup, then to its limit. The native pool hard-pinned all 32 CPU rollout workers to the driver's node; individual workers were approximately 2.5–3.2 GiB late in collection. This identifies host-memory concentration, not a CUDA HBM error.

[Source a56109c38e2f331ef2f659a8e70e116780ed6ef7](https://github.com/marin-community/marin/commit/a56109c38e2f331ef2f659a8e70e116780ed6ef7) adds explicit worker placement and sets panels to spread across existing hosts. Training retains its default driver placement. GPU geometry, 128 GiB requests, production priority and all-rack eligibility remain unchanged. The materialized scientific SkyRL configuration was compared with v7 and is identical after removing worker placement. The real S0 plan, 52 focused tests, lint and typecheck pass. A real two-node Ray regression observed both workers on the driver under driver placement and on two nodes under spread placement. Canonical schema generation also now preserves the unchanged antithrash defaults.

S0 v8 was accepted at 23:00:26 UTC; first admission inspection is pending. The original clock and frozen protocol remain unchanged. S2 periodic checkpoints were separately validated on regional CPUs against the real pool, but N remains unfixed and neither RL arm has been submitted.

### 2026-11-25 23:06 UTC — Preserve the CPU audit environment boundary

[Source c1fa4d2f4ded70a6f7a6e1d8db8ccc38f322df8a](https://github.com/marin-community/marin/commit/c1fa4d2f4ded70a6f7a6e1d8db8ccc38f322df8a) runs the panel-binding adapter in Marin's environment and invokes the detector through the existing frozen SkyRL CPU command. The standalone SkyRL environment does not install Marin, so using it for the binding module would fail before auditing. Both import/CLI-help checks pass. This changes CPU postprocessing only; the v8 GPU source remains a56109c38e. Seven author-schema tests also pass, supplementing the 52 focused checks above. S0 v8 placed all 32 hosts at 23:02:30 UTC; actual worker distribution remains pending model startup.

### 2026-11-25 23:24 UTC — Apply the registered budget drop before baseline results

S0 remained unsealed after its planned T+6 window. Apply the design's drop order: omit the optional report-only panels, then omit SWE100 from S1 and S3 to protect time for both required guard arms and paired RL. This decision precedes inspection of baseline loop incidence, validity or correctness results. S0 still evaluates the full 823 trials. Follow-ups evaluate TB2 267, TBLite 200 and SWESmith held-out 256: 723 trials per arm. Task IDs, repetitions, seeds, protocol and detector on those cohorts are unchanged.

[Source 19e1f10b0593de7d085be4199cd2cd62f244f9b5](https://github.com/marin-community/marin/commit/19e1f10b0593de7d085be4199cd2cd62f244f9b5) records the follow-up subset in both guard presets and the S3 factory; the audit adapter records excluded cohorts explicitly. Both real native S1 plans and required lint/typecheck pass, with 64 GPUs per arm. The live S0 v8 bundle is unchanged. Neither guard arm has been submitted.

Keep target status inherited from S0 for every retained cohort. SWE100 will have a baseline result and an explicit budget-drop marker in follow-up tables; follow-up effect claims cover the 723 measured trials, not SWE100. N remains unfixed pending valid S0 timings. The original clock, T+20 start cutoff and T+32 update cutoff remain fixed. No threshold was relaxed and no detector was widened.

### 2026-11-25 23:35 UTC — Repair actor export after the placement change

S0 v8 loaded models and captured graphs, but failed before trials during Ray actor-class export: `PicklingError: Cannot pickle files that are not opened for reading: a`. An initialization diagnostic added to the worker captured the configured Loguru append-only file sink. W&B finish timeouts were secondary cleanup failures. The exact task-0 attempt was checked in hub Finelog; root error sequence 32045445029 was recorded at 23:28:18.480 UTC. The earlier two-node placement test used a lightweight actor, so it did not exercise serialization of the production class.

The failure reproduced locally by exporting the real modified RolloutWorker class with an appended runtime log sink. [Source 496dd1b5a876f5a4f9750115a79c4715e2897ee8](https://github.com/marin-community/marin/commit/496dd1b5a876f5a4f9750115a79c4715e2897ee8) uses stdlib logging for that diagnostic and adds the reproduced actor-class regression. Both the real-class serialization check and two-node placement test pass, as do the real native S0 plan and required lint/typecheck. Scientific SkyRL settings and allocation compare equal to v8; the full 823-trial S0 is retained.

S0 v9 was accepted at 23:35:32 UTC, with first admission inspection pending. V8 produced no trial collection and cannot validate the post-rollout memory behavior of spread placement. V9 must verify actual actor distribution and memory after initialization. Original T0 and cutoffs remain fixed; no baseline or intervention result is claimed.

### 2026-11-26 00:08 UTC — V9 reaches collection with distributed CPU workers

S0 v9 initialized all 32 rollout workers and began TB2 at approximately 2026-11-25 23:59 UTC. At 00:05:57 UTC, 14/267 TB2 trials had returned. These are returned counts, not scored, valid or passing counts. Current logs contain a task-level TmuxSessionEnded interruption and a native transient Daytona retry; their scoring disposition awaits persisted records. No PicklingError, OOM or snapshot-quota error appears in this window.

Root logs show worker processes on at least 11 remote hosts; log deduplication prevents an exact census from those messages. Hub Finelog `iris.task`, origin cw-us-east-08a, all 32 v9 task IDs at attempt 0 with query bounds 23:59–00:10 UTC and samples available through 00:07 UTC, reports a post-startup head maximum of 48.76 GiB, with all peers at most 44.88 GiB. The attempt's model-startup peak is 68.01 GiB. Requests remain 128 GiB. This verifies initial distribution and memory behavior; it is not a full-cohort stress result. Source remains 496dd1b5a876f5a4f9750115a79c4715e2897ee8.

Next: finish all S0 cohorts, audit validity and action loops, select targets from S0 and review offline stall fires before intervention claims. No baseline is sealed and no effect is claimed.

### 2026-11-26 00:14 UTC — Keep counterfactual outcomes absent when unassessed

Before inspecting S0 scientific results, [source fccf79e47ba14c5ea87bdfb330735672eec95902](https://github.com/marin-community/marin/commit/fccf79e47ba14c5ea87bdfb330735672eec95902) corrects the offline stopped-trial pass-rate denominator. Fired trials with unknown outcomes are counted explicitly and excluded from that rate, rather than lowering it. The shared #10197 policy also keeps infrastructure errors absent even if a raw verdict was saved, while preserving graded agent-limit outcomes. The main audit's existing classification is unchanged; the counterfactual now uses that same policy. Thirty-nine audit/counterfactual tests and required lint/typecheck pass. This is CPU reporting only; GPU v9 remains at source 496dd1b5a8.

For the precision review, retain all fired identities with the native counterfactual's `--sample 823`, then select up to 30 globally by a deterministic seed-20261125 hash of cohort, task ID, attempt ID, UID and ordinal. Selection is independent of cut length and verdict; the native top-cut ordering remains diagnostic. Preserve exact identities for bounded regional review windows, and report the actual size if fewer than 30 fires exist. Detector definitions, stop threshold and success criteria stay fixed.

### 2026-11-26 00:56 UTC — TB2 returns, but validity is below the seal threshold

S0 v9 completed TB2 and moved to TBLite. At 00:53:15 UTC, TBLite had returned 110/200 trials; all 32 GPU hosts remained running with zero failures or preemptions. Hub Finelog memory samples through 00:29 UTC covered the point where v7 failed: the head's active-window maximum was 50.92 GiB and all peers were at most 47.06 GiB, against 128 GiB requests. This is an observed collection window, not a full-panel memory seal.

The [regional CPU TB2 audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs0-2026.11.25.9%2Fpartial%2Ftb2%2Faudit.json) (`s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/partial/tb2/audit.json`) succeeded, but TB2 is unsealed: 33 infrastructure-error trials leave 234/267 assessed (87.64%), below the required 90%. Preliminary assessed results are 35 agent-loop trials, agent-loop share 0.14957, agent-loop UB 68/267 (0.25468), repeat-turn share 0.12331 and 58 passes. Four tail-loop trials are report-only. These partial results do not establish a complete baseline or an intervention effect.

The 33 errors comprise 18 AgentTimeoutError, 11 VerifierTimeoutError, three TaskMachineError and one TmuxSessionEndedError. Fifteen first-model timeouts and two advance timeouts kept `Execution has no final grade`. First-model failures occurred at about 610 seconds, consistent with the frozen 600-second model-turn deadline plus cleanup. Source inspection and two failing regressions exposed a grading gap: the engine skipped grading before the first model response and after an advance timeout, although the experiment requires agent timeouts to run the verifier.

[Source 718b300206ed26702d092a1ef5f29365e3af3559](https://github.com/marin-community/marin/commit/718b300206ed26702d092a1ef5f29365e3af3559) grades known model/advance timeouts while preserving their failure identity. Both regressions now pass; the rollout file passed 236 tests, and the whole rollout package passed 362 tests with nine skipped and four deselected. Required lint/typecheck passed. The live v9 bundle remains unchanged. No limits, seeds, panel membership, detector definitions or thresholds changed.

Next: recover task/repetition identities from retained trajectories, verify their frozen seeds, then define an explicit recovery for originally ungraded trials. Preserve original graded outcomes. Evaluation dump ordinals encode completion order and must not be treated as repetition seeds. The 48-hour clock and T+20/T+32 cutoffs remain fixed. I1 and I2 still follow the sealed baseline; paired RL depends on valid timings and the remaining clock.

### 2026-11-26 01:16 UTC — Register and validate infrastructure recovery

The retained-record analysis matched all 267 TB2 task/repetition identities to the saved scientific content and verified every trial seed against the frozen seed rule. No content group was ambiguous. The 33 originally ungraded identities cover 18 task IDs. Register one replay of those 18 tasks at all three original repetitions: 54 trials. Retain only the 33 originally ungraded identities, preserve the 234 original graded outcomes and ignore the other 21 replayed outcomes. No replay result has been inspected.

The regional CPU publisher selected those 18 TaskRolloutSpecs without changing their bytes. The separate recovery selection has SHA256 `0d2b3f3d15a33996818140d8f140195d5a4522eaed334dfec084a0c3c9128ac4`; the frozen panel selection remains unchanged. [Source 7c6ff4130f8d99ceb3cd328707a09658fdd3dfc7](https://github.com/marin-community/marin/commit/7c6ff4130f8d99ceb3cd328707a09658fdd3dfc7) adds the runnable `s0-infra-recovery-tb2` preset at revision `2026.11.25.10`. Its native plan passes: allocation and generator settings equal v9, with other-cohort callbacks removed and concurrent trials reduced from 267 to 54. It still requests 128 GB200, 128 GiB per host and production priority, with every rack eligible. The recovery has not been submitted while v9 occupies the GPU budget.

The [stitching processor](analysis/stitch_tb2_recovery.py) was tested regionally with original records used as a simulated replay. All TB2 incidence, correctness, cost, rule and cohort-hash fields exactly reproduced the original audit. This check generated zero model trials and leaves the same 33 infrastructure outcomes; it is not a repaired baseline. The [counterfactual processor](analysis/s0_counterfactual.py) streams the final cohort inputs, supports an explicitly recorded replacement file, and selects the previously registered global precision sample. Its replacement-input smoke reproduced every fixture summary without double counting.

V9 remains live and unsealed. TBLite had returned 190/200 trials at 01:12:37 UTC. Host-memory samples through 01:05 UTC show a head active-window maximum of 54.04 GiB and peer maxima at most 47.59 GiB; the head's cumulative attempt peak remains 68.01 GiB against its 128 GiB request. The watchdog reported running with no alerts at 01:10:41 UTC. Next: complete v9, apply the registered recovery, audit the combined baseline and inspect the stall sample before S1. The original cutoffs still apply.

### 2026-11-26 01:28 UTC — TBLite meets the validity requirement

The [TBLite audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs0-2026.11.25.9%2Fpartial%2Ftblite%2Faudit.json) (`s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/partial/tblite/audit.json`) reports 190/200 assessed trials (95%), with ten infrastructure outcomes and no additional unavailable verdicts. It meets the design's numerical validity requirement. There are 23 confirmed agent-loop trials, confirmed share 0.12105, agent-loop UB 33/200 (0.165), repeat-turn share 0.12988 and 63 full-credit passes. Mean reward is 0.35636. Four token-tail loops are report-only. TBLite qualifies as a target cohort under the frozen minimum of ten confirmed agent loops.

Native evaluation callbacks write their results beneath cohort directories such as `exports/tblite/dumped_evals`, while the first cohort writes beneath `exports/dumped_evals`. The CPU adapter's original main-directory glob would miss callback cohorts. [Source fc2591ddecb0dd4e7d312a273208e1aa2263daaf](https://github.com/marin-community/marin/commit/fc2591ddecb0dd4e7d312a273208e1aa2263daaf) searches `exports/**/dumped_evals/*/eval_only/*.jsonl`; the existing frozen task-ID filter still selects each cohort. Required lint/typecheck passed, and the regional 200-trial TBLite audit succeeded against the actual callback output. The live GPU bundle is unchanged.

[S0 v9 on W&B](https://wandb.ai/marin-community/marin-mochi-rl/runs/a54355029268) completed TBLite at 01:13:28 UTC and moved to SWE100, with 15/100 returned at 01:22:20 UTC. All 32 GPU hosts remain running with zero failures or preemptions. The full baseline remains unsealed: TB2 requires its registered recovery, and SWE100 plus SWESmith held-out remain pending. No guard arm or recovery GPU job has been launched while v9 occupies the budget.

### 2026-11-26 02:08 UTC — SWE100 qualifies; SWESmith collection continues

The [SWE100 audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs0-2026.11.25.9%2Fpartial%2Fswe100%2Faudit.json) (`s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/partial/swe100/audit.json`) reports 98/100 assessed trials, two infrastructure outcomes and no additional unavailable verdicts. There are ten confirmed agent-loop trials, agent-loop UB 12/100, repeat-turn share 0.01764 and 52 full-credit passes; mean reward is 0.53061. Two token-tail loops are report-only. Ten confirmed loops reach the frozen target minimum. SWE100 remains baseline-only under the budget drop registered before results; it will not be restored to S1/S3 in response to this measurement.

SWESmith held-out began around 01:46:45 UTC and reached 116/256 returned trials at 02:00:11 UTC. All 32 GPU hosts remained healthy at 02:01:13 UTC, with zero failures or preemptions. A read-only retention snapshot at 01:55:25 UTC contained 62 returned records: 58 verified and four infrastructure outcomes. Early returns are not representative, so final validity and loop incidence await the completed cohort. Per-request tool-parser logs have not caused a GPU job failure; no execution setting changed.

Host-memory samples across all 32 v9 attempt-0 tasks, query bounds 01:05–02:03 UTC with samples through 02:01:47 UTC, show a head active-window maximum of 54.66 GiB and peer maxima at most 48.14 GiB. The head's cumulative attempt peak remains 68.01 GiB against a 128 GiB request. The full baseline remains unsealed. Next: finish v9, perform the registered TB2 infrastructure recovery, then audit the combined baseline and review the stall sample before S1.

### 2026-11-26 03:34 UTC — Held-out tail and RL environment readiness

S0 v9 had returned 254/256 SWESmith held-out trials at 03:26:46 UTC. All 32 GPU hosts remained running with zero failures or preemptions at 03:27:21 UTC. The final two trials continued generating within their frozen limits. The baseline remains unsealed; no recovery or intervention GPU job has been queued. Both S1 native plans passed again on source `869dc58f203265e40677ee2db4781ed6da067ae7`, at 64 GPUs and 723 trials per arm.

Regional CPU job `/benfeuer/mochi-loop-guards-s2-environment-census-20261125b` succeeded. It loaded the planned native S2 control configuration and used the trainer's `_canonical_task_prompt`, RolloutEngine's `_machine_spec`, and Shellbox's `snapshot_request_for_spec` on all 5,480 agentic rows. The unchanged pool contains 105,480 rows, and the agentic tasks map to four effective Dockerfile snapshot identities. An earlier diagnostic stopped on an overly strict registry-only assertion; that diagnostic assumption was removed to complete the census. Neither job created model trials or provider resources.

All four identities were absent from the dedicated RL account's snapshot inventory at 03:26 UTC, when its snapshot quota was 40/40 used. Live RL usage was 321/6,000 CPU and 1,284/12,000 GiB; live sandbox capacity and snapshot quota are separate constraints. The issue was reported to the coordinator. No shared snapshots were removed, no account was substituted, and no task or environment was rewritten. Recheck before S2; the original T+20/T+32 cutoffs remain fixed. The census [receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.25%2Fs2-environments%2Freceipt.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/receipt.json`; the same prefix contains `snapshot-comparison-2226.json`.

### 2026-11-26 03:58 UTC — Native snapshot preflight confirms quota failure

The four exact build contexts are now preserved under `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/prepared/contexts`. Regional CPU job `/benfeuer/mochi-loop-guards-s2-snapshot-preflight-20261125b` restored them, verified each Dockerfile hash and native snapshot identity, and invoked Shellbox with the dedicated RL credential delivered through Iris secret files. Inventory succeeded. Native creation failed with `DaytonaValidationError: Failed to create snapshot: Snapshot quota exceeded. Maximum allowed: 40`. Post-inventory succeeded and found all four required identities still absent. None of these requested snapshots was created; no shared cleanup or account substitution occurred.

The [preflight receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.25%2Fs2-environments%2Fprepared%2Fsnapshot-preflight-b.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/prepared/snapshot-preflight-b.json`; [the runnable preflight](analysis/prepare_s2_snapshots.py) preserves the diagnostic. CPU success means the diagnostic completed, while S2 readiness remains blocked on the observed provider error. The earlier context-preservation job failed before creation because its inventory check failed; its underlying CLI diagnostic was not retained. That inventory failure did not recur in b. No GPU training job or model trial was launched by either check.

S0 v9 remained healthy at 03:56:03 UTC on all 32 hosts, zero failures or preemptions. Held-out progress was still 254/256 at 03:55:46 UTC; the final two trials were generating within frozen limits. The baseline remains unsealed, and T+20/T+32 are unchanged. The concrete quota failure was reported on [#12264](https://github.com/marin-community/marin/issues/12264#issuecomment-7182185033) and to the coordinator.

### 2026-11-26 04:44 UTC — Snapshot capacity reclaimed; four S2 contexts ready

The dedicated RL account rejected a declarative sandbox preflight with `declarative builds are not allowed`; no sandbox or model trial was created. Snapshot provisioning remains the native S2 path. The four image sources and their resource requests are unchanged.

The standing one-hour Daytona reclamation policy was applied to a bounded set of six non-general 1 GiB/3 GiB snapshots, each idle for more than 24 hours. Native sandbox-reference enumeration found none in use. Registered running/queued work was checked: active .131's actual saved configuration forces Harbor memory to 4,096 MiB and uses 4,096/8,192 MiB image-verifier profiles; all 5,538 SWE rows in its actual inputs were counted regionally. Its profiles cannot match the selected 1 GiB snapshots. The .129 root was terminal failed; current Exp7 is SFT-only, S0/S1 use EVAL, SWE recollection uses DATA, and Atlas release composes catalog/curriculum artifacts. Those consumers do not reference the selected RL cache entries. General and unclassified snapshots were excluded. Each selected target's references, profile and last use were rechecked immediately before deletion.

Six delete requests were accepted at 04:27 UTC, initially `removing`. All six were verified absent at 04:29 UTC, and RL usage fell from 40/40 to 34/40. The [per-target receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.25%2Fs2-environments%2Fsix-snapshot-reclamation.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/six-snapshot-reclamation.json`; it records every name, provider size field, last use, exclusion reason, revalidation and later absence. The first helper attempt supplied a name to an SDK method that requires a Snapshot object. It failed before the delete API call; all six were rechecked active before using the correct object argument. No Iris job was cancelled by this work.

Regional CPU `/benfeuer/mochi-loop-guards-s2-snapshot-preflight-20261125c` then completed successfully: native ensure returned zero and post-inventory verified all four exact required identities active. [Readiness receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.25%2Fs2-environments%2Fprepared%2Fsnapshot-preflight-c.json): `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/prepared/snapshot-preflight-c.json`. Fresh usage at 04:42 UTC is 38/40, leaving two slots. The CPU watchdog entry was retired after verified terminal success; cache/account capacity will be checked again before actual S2. These checks created zero model trials and changed no tasks, runtime resource profiles, model settings or account bindings.

S0 v9 reached 255/256 held-out returns at 04:33:46 UTC, with all 32 hosts running and zero failures/preemptions at 04:43:56 UTC. Native progress rounds this to 100%; one trial remains. Preserve its frozen timeout and retry policy. The baseline remains unsealed and no intervention GPU job is queued. S2 still follows S0 repair/seal and S1; N remains unfixed pending valid measured timings. Original T+20/T+32 cutoffs are unchanged.

### 2026-11-26 05:25 UTC — Original collection complete; TB2 recovery placed

S0 v9 completed all 823 model trials. SWESmith held-out returned 256/256 at 04:48:11 UTC and saved its dump two seconds later. The original job then failed during task-machine shutdown: exact GPU attempt `a54355029268/0:0`, Finelog sequence 32064654609, raised `DaytonaRateLimitError` (HTTP 429) while looking up a sandbox for deletion. W&B finish warnings were secondary. Preserve the terminal failure and all four completed cohort dumps; no original model trial needs to be repeated for teardown.

Regional CPU job `/benfeuer/mochi-loop-guards-s0-audit-20261125a` succeeded against all 823 original rows. The [original audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs0-2026.11.25.9%2Faudit.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/audit.json`. These are agent-loop metrics; token-tail loops remain separate.

| Cohort | Valid | Infrastructure | Confirmed loops | Agent UB count | Repeat-turn share | Full-credit passes |
|---|---:|---:|---:|---:|---:|---:|
| TB2 | 234/267 (87.64%) | 33 | 35 | 68 | 0.12331 | 58 |
| TBLite | 190/200 (95%) | 10 | 23 | 33 | 0.12988 | 63 |
| SWE100 | 98/100 (98%) | 2 | 10 | 12 | 0.01764 | 52 |
| SWESmith held-out | 252/256 (98.44%) | 4 | 25 | 29 | 0.23590 | 117 |

All four cohorts reach the frozen ten-loop target minimum. Only TB2 falls below 90% validity. SWE100 remains baseline-only under the budget drop registered before results. No held-out rerun is required. The baseline remains unsealed until the registered TB2 replacement is audited.

[Source 51c1f95f34fccc370b77f70fe402512d94e46e6c](https://github.com/marin-community/marin/commit/51c1f95f34fccc370b77f70fe402512d94e46e6c) adds HTTP 429 to the existing bounded Daytona cleanup retry classifier. Two GET/DELETE regressions failed before the fix; all seven focused cleanup cases passed afterward. Required lint and type checking passed. The broader Shellbox suite returned 201 passed, 169 skipped and one Mac setsid failure; the same failure was reproduced on parent source 869dc58. No timeout, model, detector, sampling, resource or logging-mode setting changed.

Registered TB2 recovery v10 was accepted at 05:10:19 UTC and placed on all 32 GPU hosts by 05:13:57. Root: `/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s0-infra-recovery-tb2-a7b495a850-in-4f4a7bd0-2026.11.25.10-coordinator-f67c9a5b`. [Verified W&B run](https://wandb.ai/marin-community/marin-mochi-rl/runs/ad3bc8701b71). It remains running with zero failures or preemptions at 05:23:30. Its 128-GPU allocation and generator equal v9; only the registered 54-trial selection and removal of other-cohort callbacks differ. All racks remain eligible, TP1/BF16 and CUDA graphs remain enabled. Replace only the 33 originally ungraded identities, preserve the 234 original graded outcomes, and ignore the 21 additional replayed graded identities. The recovery watchdog is active. Next: stitch regionally, audit validity, seal S0, review I1's precision sample, then run both 723-trial S1 arms. No S1/S2 job has been submitted; N and the original T+20/T+32 cutoffs remain unchanged.

### 2026-11-26 05:44 UTC — Trace validity correction supersedes earlier loop counts

The first I1 diagnostic inspected 30 hash-ranked fired windows and exposed empty observations. Regional CPU diagnostic `/benfeuer/mochi-loop-guards-observation-integrity-20261126b` then confirmed that the original cohort dumps retained no post-action observation batches. Many timeout rows also have a placeholder token prompt, preventing reconstruction from that field: 130/267 TB2, 73/200 TBLite, 17/100 SWE100 and 56/256 held-out rows. Thus the earlier incidence, target-cohort and counterfactual summaries are superseded diagnostics, not sealed scientific results. Preserve every original outcome and artifact. The original numerical grading census remains descriptive but does not establish usable trace validity.

The cause is the external-agent continuation comparison. Native vLLM assistant messages carry `annotations: []`; agents omit that empty field when replaying history. The comparator preserved the empty array, restarted conversations, and the full-attempt projection omitted subsequent conversation prefixes. This both lost observation transitions and bypassed the runtime stall check. A second regional diagnostic confirmed the actual nonempty assistant fields were only content/role, plus tool calls for Pi; reasoning and other transport fields were empty. The initial diagnostic a failed before processing because a locally constructed CPU command embedded a Mac project path; b/c construct that native command inside the regional job and succeeded.

[Source 87d095877eb140e53f140f666ac23506ae18a24c](https://github.com/marin-community/marin/commit/87d095877eb140e53f140f666ac23506ae18a24c) normalizes empty annotations. Two end-to-end regressions failed before the fix and passed afterward: unchanged output now stops after three action/observation pairs, while changed output continues with its observations retained. The full RolloutEngine suite passed 364 tests, with nine skipped and four deselected. Eight native worker/context tests passed, including preservation of observations through the saved trajectory projection. Required lint and type checking passed. No detector definition, task, limit, seed, initializer, sampling parameter or resource setting changed.

Only the owned v10 recovery root and GPU child were cancelled with `iris-cancel-own` at 05:38 UTC and verified KILLED. They contain the same faulty retention path and cannot repair the baseline. The registered 33-identity stitching procedure is superseded by this finding; do not use it to seal S0. The old watchdog entry was retired under lock.

Corrected full S0 v11 was accepted at 05:41:35 UTC and placed on all 32 GPU hosts by 05:43:38, with zero failures or preemptions. Root: `/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s0-baseline-7fd9a834a1-in-af5ab9b8-2026.11.25.9-coordinator-cf4b9e2a`; GPU child ends `a2bae512ef6b`. It uses source 87d095877e, tree 7fd9a834a1, all 823 frozen trials and the same 128-GPU allocation. All explicit generator fields equal the original v9 configuration. Production priority, 128 GiB hosts, all racks, TP1/BF16 and CUDA graphs remain. The operational alias is v11; the preset's actual artifact revision remains 2026.11.25.9 despite the supplied CLI version, and the new tree hash creates a separate output. The watchdog tracks v11. Next: verify its first retained multi-turn observations, complete the corrected panel, audit and seal, then run I1/I2. No intervention has been launched, and T0/T+20/T+32/deadline remain unchanged. [Correction on #12264](https://github.com/marin-community/marin/issues/12264#issuecomment-7183670963), [placed replacement](https://github.com/marin-community/marin/issues/12264#issuecomment-7183760183).

### 2026-11-26 05:54 UTC — Exact Pi replay covered; S0 v12 placed

Fixture review found that Pi explicitly uses `ReplyMatch.EXACT`. The normalization-only fix in v11 covered Terminus but could not preserve Pi's feedback across its intended restarts. Only the owned v11 root and child were cancelled using `iris-cancel-own` and verified KILLED; no v11 result enters the baseline, and its watchdog entry was retired. The 48-hour clock is unchanged.

[Source d0801e23e23c848683bc753a5b1a29c6296948fa](https://github.com/marin-community/marin/commit/d0801e23e23c848683bc753a5b1a29c6296948fa) records known feedback when the next prompt continues the same semantic history but exact replay requires a new token conversation. It retains Pi's replay mode and request, and checks the same frozen three-pair stall rule before both `start` and `reply`. Two exact-mode regressions failed before this change and passed after. Final validation: 14 stall tests and 366 RolloutEngine tests passed, nine skipped and four deselected; nine native worker/context tests passed, including saved observation preservation in both modes. Required lint and type checking passed. A one-off probe of the actual Terminus and Pi replay fixtures preserved model requests apart from empty transport metadata and retained Pi's tool-loop feedback. Context-summary helper turns and the non-panel interleaved XML fixture are not claimed to be terminal feedback. The [probe receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.26%2Freal-agent-annotations-validation.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/real-agent-annotations-validation.json`.

Full S0 v12 was accepted at 05:50:45 UTC and placed on all 32 GPU hosts at 05:52:48, with zero failures or preemptions. Root: `/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s0-baseline-8d1a0b19d0-in-af5ab9b8-2026.11.25.9-coordinator-15fbdce1`; GPU child ends `aa4cc7ddd162`. Source d0801e23e2, tree 8d1a0b19d0, actual artifact revision 2026.11.25.9. The source-tree hash creates its separate output. Native plan, allocation equality and every explicit generator field match the original frozen configuration: 823 trials, 128 GB200, 128 GiB hosts, production priority, all racks, TP1/BF16 and CUDA graphs. No detector, seed, initializer, task, sampling or limit changed. The watchdog tracks v12 with no alerts.

Regional CPU `/benfeuer/mochi-loop-guards-v12-first-traces-20261126a` will inspect the first five retained multi-turn records during collection; it creates no model trials and is a retention diagnostic, not a scientific gate. Original v9/v10 and v11 loop/CF material remains superseded diagnostics. S0 is unsealed; targets and N await valid measurement. S1/S2 are unsubmitted, and all original cutoffs remain fixed. [Placed replacement on #12264](https://github.com/marin-community/marin/issues/12264#issuecomment-7183898379).

### 2026-11-26 06:13 UTC — Actual retained feedback verified

Regional CPU `/benfeuer/mochi-loop-guards-v12-first-traces-20261126a` succeeded. All five first multi-turn TB2 records have verifier results and retain every between-turn observation: assistant turns/observation batches are 7/6, 5/4, 6/5, 5/4 and 9/8. The [receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.26%2Fs0-v12-first-traces.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/s0-v12-first-traces.json`. This validates retention on actual generated data; five early returns do not estimate population correctness, establish loop incidence or seal the baseline. The CPU watchdog entry was retired after terminal success and receipt verification; the GPU entry remains active.

Exact GPU-head Finelog shows TB2 evaluation starting at 06:06:42.423 UTC (sequence 32066554602) and 5/267 returned at 06:11:42 (sequence 32066617802). All 32 hosts remained healthy at 06:11:55, with zero failures or preemptions. [Verified W&B run](https://wandb.ai/marin-community/marin-mochi-rl/runs/aa4cc7ddd162), confirmed on the exact head attempt at sequence 32066299947. The saved runtime configuration independently confirms the pinned regional SFT 400407 source, 128 GB200, 128 GiB hosts, production priority, all racks, TP1/BF16 and PIECEWISE CUDA graphs. Both S1 native plans pass on clean source d0801e23e2 at 64 GPUs and 723 trials per arm. No S1/S2 job has been submitted. [Tracking comment](https://github.com/marin-community/marin/issues/12264#issuecomment-7184235732).

### 2026-11-26 06:56 UTC — Corrected TB2 meets validity

Regional CPU `/benfeuer/mochi-loop-guards-s0-v12-audit-tb2-20261126a` succeeded on the completed TB2 dump. All 267 frozen trials are accounted for: 250 assessed (93.63%), 17 infrastructure outcomes and no additional unavailable verdicts. This meets the design's 90% requirement. No TB2 repair or stricter completeness gate is needed.

| TB2 metric | Corrected S0 v12 |
|---|---:|
| Confirmed agent-loop trials | 34 |
| Confirmed trial share | 0.136 |
| Agent-loop upper bound | 51/267 (0.19101) |
| Repeat-turn share | 0.1192787795 |
| Duplicate-tool fraction, median / p90 | 0.13333 / 0.54615 |
| Full-credit passes | 54 |
| Mean reward | 0.216 |
| Median turns / generated tokens / wall seconds | 17.5 / 7,196 / 1,109.13 |

Ten token-tail loops are report-only. TB2 qualifies as a target under the frozen ten-agent-loop minimum. The [audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs0-2026.11.26-v12%2Fpartial%2Ftb2%2Faudit.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12/partial/tb2/audit.json`; cohort hash remains `0befdc85f0c1f81c515d13ab49e4e1fead14c9c690208192237e425f06ed8ec1`. TB2 completed at 06:46:16.978 UTC, after 2,374.6 seconds of native evaluation. Per-trial timing describes the final attempt, so the native phase duration and retries must also inform any later RL update budget. No N has been chosen.

S0 moved to TBLite, with 90/200 returned at the 06:59:45 probe. All 32 hosts remained running with zero failures or preemptions. Across all 32 GPU tasks, telemetry through 06:32:41 showed a maximum sampled and cumulative-attempt host peak of 69.26 GiB against 128 GiB requested. The complete S0 baseline remains unsealed pending TBLite, SWE100 and SWESmith held-out. No guard or RL job is submitted. [Tracking milestone](https://github.com/marin-community/marin/issues/12264#issuecomment-7184928437).

### 2026-11-26 06:38 UTC — Boundary rerun interpretation registered before S1

The coordinator confirmed the DESIGN section 6 interpretation proposed before any guard outcomes: if a target cohort is within two trials of its decision threshold, repeat the implicated guard arm once on the same panel and seeds. The repeat supplies the gate measurement regardless of direction. Report both measurements separately; never pool them or choose the better result. Keep the S0 control, target rule, intervention settings and thresholds fixed. This [pre-registration](https://github.com/marin-community/marin/issues/12264#issuecomment-7184629056) adds no launch gate and resets no clock. S1 remains unsubmitted.

### 2026-11-26 07:27 UTC — Corrected TBLite meets validity

Regional CPU `/benfeuer/mochi-loop-guards-s0-v12-audit-tblite-20261126a` succeeded. All 200 frozen trials are accounted for: 194 assessed (97%), six infrastructure outcomes and no additional unavailable verdicts. TBLite meets the design's 90% validity requirement and the ten-loop target rule; no repair is needed.

| TBLite metric | Corrected S0 v12 |
|---|---:|
| Confirmed agent-loop trials | 20 |
| Confirmed trial share | 0.1030928 |
| Agent-loop upper bound | 26/200 (0.13) |
| Repeat-turn share | 0.0974277259 |
| Duplicate-tool fraction, median / p90 | 0.1327434 / 0.4454443 |
| Full-credit passes | 66 |
| Mean reward | 0.3724413859 |
| Median turns / generated tokens / wall seconds | 14 / 7,778 / 847.24 |

Ten token-tail loops are report-only. The [audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs0-2026.11.26-v12%2Fpartial%2Ftblite%2Faudit.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12/partial/tblite/audit.json`; cohort hash remains `2e9b291a9b79f8af76c81c65c727ab9032717264a81e39410e1d631ca180ddda`. Native evaluation reached 200/200 after 2,162 seconds. SWE100 began at 07:22:22 UTC and reached 5/100 at 07:27:39. All 32 GPU hosts remained running at 07:28:26 with zero failures or preemptions. Full S0 remains unsealed pending SWE100 and SWESmith held-out. No S1/S2 job or fixed update count exists. [Tracking milestone](https://github.com/marin-community/marin/issues/12264#issuecomment-7185466688).

### 2026-11-26 07:45 UTC — Rerun count units clarified before S1

The [boundary clarification](https://github.com/marin-community/marin/issues/12264#issuecomment-7185769171) applies the ±2-trial trigger to count-based target-cohort thresholds: I2 loop UB and per-cohort correctness passes. I1's repeat-turn-share ratio has no trial-count unit, so no turn-to-trial conversion is introduced. One triggered arm repeat uses the same full 723-trial panel and seeds and supplies all gates regardless of direction. Original and repeat are reported separately; never pool or select the better result. No S1 job or outcome existed when registered.

### 2026-11-26 08:10 UTC — Corrected SWE100 valid; actual Pi feedback verified

Regional CPU `/benfeuer/mochi-loop-guards-s0-v12-audit-swe100-20261126a` succeeded with zero failures or preemptions. All 100 frozen trials are accounted for: 99 assessed (99%), one infrastructure outcome and no additional unavailable verdicts.

| SWE100 metric | Corrected S0 v12 |
|---|---:|
| Confirmed agent-loop trials | 3 |
| Confirmed trial share | 0.0303030 |
| Agent-loop upper bound | 4/100 |
| Repeat-turn share | 0.0466101695 |
| Duplicate-tool fraction, median / p90 | 0.0821918 / 0.1582751 |
| Full-credit passes | 52 |
| Mean reward | 0.5252525253 |
| Median turns / generated tokens / wall seconds | 31 / 6,590 / 908.15 |

Its one token-tail loop is report-only. SWE100 is non-target under the frozen ten-loop minimum and remains baseline-only under the budget drop registered before scientific outcomes. Native evaluation completed at 07:58:10.554 UTC after 2,148 seconds; cohort hash is `ff32a38868afa2809044175c75adb43fc42d8b759d10ac125832d25081135727`. The [audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs0-2026.11.26-v12%2Fpartial%2Fswe100%2Faudit.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12/partial/swe100/audit.json`.

CPU `/benfeuer/mochi-loop-guards-v12-first-pi-traces-20261126a` also succeeded. All five checked multi-turn Pi held-out records have verifier results and every between-turn observation: assistant turns/observation batches 5/4, 6/5, 9/8, 14/13 and 19/18. The [receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.26%2Fs0-v12-first-pi-traces.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/s0-v12-first-pi-traces.json`. This verifies repaired feedback retention on actual Pi data, not population correctness or stall precision. Both terminal CPU watchdog entries were validated and retired under lock. Initial submissions lacked exported storage credentials and were rejected before job creation; sourcing the existing credentials resolved submission. The GPU run was untouched.

Held-out began at 07:58:11.782 UTC and reached 64/256 at 08:05:45. All 32 GPU hosts remained running at 08:06:39 with zero failures or preemptions. Full S0 remains unsealed pending held-out; no S1/S2 has been submitted and N is unfixed. Source review confirms one trajectory sink covers all four cohorts. The prepared in-region retention copy waits for terminal GPU/root state so publication has closed. [Tracking milestone](https://github.com/marin-community/marin/issues/12264#issuecomment-7186256488).

### 2026-11-26 08:38 UTC — Full feedback census prepared

The standalone [feedback census](analysis/observation_census.py) will stream all completed corrected S0 dumps in the existing regional CPU postprocessing job. It reports multi-turn records with retained observations, graded coverage and bounded hashed examples with no observations. This is engineering evidence, not an additional scientific validity threshold; context-summary helper turns and invalid agent output need interpretation before inferring loss. It creates no model trials. The registered counterfactual processor and its uniform global 30-fire sample are unchanged. Required scoped lint and AST validation pass; execution awaits full collection and terminal sink publication.

Held-out reached 234/256 at 08:35:12 UTC. All32 GPU hosts remained running at08:35:20 with zero failures or preemptions. The complete baseline remains unsealed and no guard/RL arm has been submitted. Original T0 and cutoffs remain fixed.

### 2026-11-26 11:12 UTC — Corrected S0 sealed; offline stall review passes

S0 v12's coordinator and all32 GPU tasks finished SUCCEEDED with zero failures or preemptions. Held-out reached256/256 at10:59:15.267 UTC after10,863 seconds of native evaluation. The root remains `/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s0-baseline-8d1a0b19d0-in-af5ab9b8-2026.11.25.9-coordinator-15fbdce1`, GPU child suffix `aa4cc7ddd162`, source d0801e23e2. All823 frozen trials are accounted for,796 assessed and27 infrastructure outcomes, with no additional unavailable verdicts. The first three full-audit cohorts exactly match their earlier partial audits and hashes.

| Cohort | Assessed / N | Action-loop trials | Loop UB | Repeat-turn share | Passes | Mean reward | Target |
|---|---:|---:|---:|---:|---:|---:|---|
| TB2 | 250/267 | 34 | 51/267 | 0.1192787795 | 54 | 0.216000 | Yes |
| TBLite | 194/200 | 20 | 26/200 | 0.0974277259 | 66 | 0.372441 | Yes |
| SWE100 | 99/100 | 3 | 4/100 | 0.0466101695 | 52 | 0.525253 | No; baseline-only |
| SWESmith held-out | 253/256 | 13 | 16/256 | 0.1216524559 | 117 | 0.462451 | Yes |

The [full audit and seal](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs0-2026.11.26-v12%2Fbaseline-seal.json) are under `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12/`. Regional CPU `/benfeuer/mochi-loop-guards-s0-postprocess-20261126v12a` completed the full audit, unchanged counterfactual processor, feedback census and in-region verified retention copy. The permanent trace prefix is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/traces/s0-v12`, [Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Ftraces%2Fs0-v12). Its completion-manifest SHA-256 is `ea6aaa6d788b935b7362780c5b9b01cf3263a3162a7f5c3ce5a58de584ef6c5a`.

Every multi-turn record contains observations. Assistant-call counts exceed feedback counts because compaction and handoff helpers also generate responses, sometimes shaped like commands that are not executed. The census and bounded gap review are engineering diagnostics, not a new validity threshold; the frozen parser and detector stay unchanged. The launcher status reader separately expects a `task_spec` column instead of the native panel's `task_rollout_spec`, so its evaluation-summary collection failed. The model run and frozen-panel audit succeeded. Fix the reporting reader without repeating model work.

The unchanged offline counterfactual fires on61/823 trials:28 TB2,17 TBLite,3 SWE100 and13 held-out. Their final full-credit passing counts are0,4,0 and3 respectively; all seven passing cases have later turns that would be cut. These are at-risk recorded passes, not counterfactual grading outcomes. Wait-only exclusions suppress121 TB2,140 TBLite and3 SWE100 windows.

The registered global30 sample uses the smallest SHA256(seed20261125, cohort, task identity) ranks, independent of cut length and verdict. All30 windows were reviewed with verdicts, cut lengths and later turns hidden. Result:29 genuine stalls and one non-genuine summary-helper output sequence with empty feedback, observed precision29/30=96.67%, meeting the90% requirement. The false fire remains in the frozen analysis; no detector retuning or resampling occurred. Labels and provenance: `counterfactual/precision-labels.json` under the audit prefix, [Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs0-2026.11.26-v12%2Fcounterfactual%2Fprecision-labels.json).

Before any S1 outcomes, the fixed723-trial follow-up requires pooled passes at least223 versus237 control passes. Target-cohort minimum passes are41 TB2,56 TBLite and105 held-out. I1 repeat-turn-share maxima are0.0596393897,0.0487138630 and0.0608262279. I2 maximum integer UB counts are17,10 and6 (raw held-out threshold6.5). The ±2-trial rerun trigger uses the unrounded registered count thresholds, not display rounding: correctness counts39–42/54–58/103–106 and I2 UB counts15–19/8–12/5–8, respectively. One repeat supplies every gate regardless of direction; report both runs separately and never pool or best-of. No count conversion is introduced for the I1 ratio.

Both guard arms remain the unchanged source-d080 presets at64GPUs and723trials each. S2 N is unfixed, the placeholder8 must not be submitted, and original T20/T32/deadline remain. No intervention effect is established by these baseline or offline results.

### 2026-11-26 11:18 UTC — Both S1 arms submitted; admission pending

After the baseline seal and complete offline review, S1-stop was accepted at11:14:24.766 UTC and S1-rp at11:14:41.643. Both use immutable source d0801e23e2, actual artifact revision2026.11.25.9,64GB200,128GiB hosts, production cw-us-east-08a, all racks,TP1/BF16 and CUDA graphs. Each measures the same723 follow-up trials and seeds; SFT400407 remains the initializer. The estimated duration is4–5 hours per arm, subject to admission and frozen limits.

- Stop root: `/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s1-stop-8d1a0b19d0-in-d88ffd61-2026.11.25.9-coordinator-328ce3e4`; GPU child suffix `a6b43090575a`.
- RP root: `/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s1-rp-8d1a0b19d0-in-d88ffd61-2026.11.25.9-coordinator-4bd2a367`; GPU child suffix `a824d6ccf60d`.

First checks11:16:57/11:17:02 UTC found both children16/16SchedulingGated, not placed, with zero failures/preemptions. Both own watchdog entries are registered and validated at11:17:55 with no alerts; native waits72140/80470 and sharedmonitor46930 own monitoring. The earlier S0 monitor and native wait completed. Full S0 postprocessing succeeded in1m49.92s; the bounded helper-gap diagnostic succeeded in35.44s. Its terminal watchdog entry was already absent from the live registry before S1 registration; no other owner's entries were changed. [Submission milestone](https://github.com/marin-community/marin/issues/12264#issuecomment-7190560763), [baseline seal and pre-S1 thresholds](https://github.com/marin-community/marin/issues/12264#issuecomment-7190461858).

S1 source and science settings are frozen while its jobs run. The separate report-reader schema repair is being validated locally and will refresh reporting from saved data. It does not require repeating model work. No intervention outcome is established, no S2 is submitted, and equal N is still unfixed. T20 remains13:01:36 UTC.

### 2026-11-26 11:34 UTC — Reporting repair validated; S1 admission pending

Source c4b5bf17ca fixes the launcher report reader for the native panel's `task_rollout_spec` column. It reads both canonical source formats already accepted by SourceTaskDataset, extracts the native task identity, and hashes the complete serialized record. A regression reproduced the live ArrowInvalid before the fix and checks identity preservation, order-independent hashing and sensitivity to rollout-limit changes. The focused tests pass; the complete gate/publication files pass44 tests, required lint passes, and type checking reports zero errors. The live S1 bundles remain immutable source d080; no model measurement is repeated. Regional CPU `/benfeuer/mochi-loop-guards-s0-report-refresh-20261126a` refreshes saved S0 reporting using the repaired reader and the regional selection root.

The11:26:36/39 UTC probe still found both S1 children16/16SchedulingGated, with zero failures/preemptions. Read-only Kueue inspection confirms priority1000/production, required16-host nvlink.domain gangs and no node affinity or rack exclusions. The stop workload reports18 additional GPU/RDMA quota units needed; RP has no admission status yet. No other job was touched and neither S1 arm was resubmitted. OriginalT20 remains13:01:36 UTC. S2 N stays unfixed and the placeholder8 is prohibited from submission.

### 2026-11-26 11:40 UTC — Saved-data reporting repaired without model reruns

The first regional refresh confirmed all54 S0 evaluation summaries and zero failed evaluation invariants, then exposed a second publication defect: generation-only attempts have no training data, and AttemptSummary rejected the empty data identity. Source808482fb50 now identifies such attempts using their recorded validation and additional-evaluation sources, preserving training-run identities. An actual native generation fixture reproduces the Pydantic failure before the fix; all12 publication tests, required lint and type checking pass afterward. Regional CPU `/benfeuer/mochi-loop-guards-s0-report-refresh-20261126b` succeeded after the failed diagnostic-a. Both are CPU-only; no model trial or audit outcome changed. Saved reporting is under the existing S0 audit prefix as `refreshed-evaluation-status.json`, `refreshed-attempt-summary.json` and `report-refresh-receipt.json`.

IndexPR87 now pins notebookc7d25919d6 and describes the sealed baseline and submitted guard arms. CIvalidatepassed,publishskipped. The previous12-hour wait timed out; after this index push, a new attached feedback wait owns review monitoring. No merge is authorized or attempted.

### 2026-11-26 11:47 UTC — S1-stop placed; RP admission pending

At11:45:54.515915 UTC, S1-stop childa6b43090575a had all16 hosts running with zero failures/preemptions; its task duration was19.36 seconds. S1-rp childa824d6ccf60d remained16/16SchedulingGated at11:45:57.997858. Both retain64GB200,128GiB hosts, production priority, all racks,TP1/BF16 and CUDA graphs. There is no configuration change or resubmission. Native resolved-launch and exact-head W&B readback are pending early startup publication.

S0's final host-memory census covers9,850 samples across all32 tasks through11:00:34 UTC. Maximum sampled and cumulative-attempt peak RAM are both69.2568359375GiB against128GiB requested. Receipt: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/s0-v12-final-host-memory.json`. The refreshed S0 dashboard record was also verified by the publication API for exact attemptaa4cc7ddd162; its receipt is `report-publication-receipt.json` under the full S0 audit prefix. These diagnostics do not change measurements or resource requests.

### 2026-11-26 12:09 UTC — Both guard arms placed; actual runtime verified

Both S1 children are16/16RUNNING at12:05:10/13 UTC, with zero failures or preemptions and128 GPUs combined. Stop returned14/267 TB2 trials at12:04:21. RP's head task duration was7m16.43s at its first positive probe, implying startup at approximately11:57:57.5 UTC; this estimate is separate from the observed probe time.

Actual resolved-launch readbacks confirm both regional SFT400407 bindings,64GB200/16hosts per arm,128GiB host memory, production priority, all racks, four engines withTP1/DP16/EP16,BF16 andPIECEWISE CUDA graphs, eager execution false. Stop uses the three retained stop-enabled cohort files andRP1.0; RP uses stop-off files andRP1.1. SWE100 is absent from both follow-ups. Readbacks: `diagnostics/2026.11.26/s1-{stop,rp}-actual-runtime-identity.json` under the experiment S3 prefix. Exact head logs verify [stop W&B](https://wandb.ai/marin-community/marin-mochi-rl/runs/a6b43090575a), sequence32074177656, and [RP W&B](https://wandb.ai/marin-community/marin-mochi-rl/runs/a824d6ccf60d), sequence32074733594. Separate `s1-{stop,rp}-verified-wandb.json` receipts retain provenance.

A reporting-only follow-up, source05016bd66a, preserves training-source multiplicity when multiple files share one artifact identity. A two-file native receipt regression fails before the fix; all13 publication tests, required lint and type checking pass. S0's published generation record remains valid; live S1 bundles stay on source d080. Both own watchdog entries remain active. No S2 is submitted and N remains unfixed; originalT20 is13:01:36 UTC.

### 2026-11-26 12:39 UTC — Stall stop meets the TB2 cohort gates

S1-stop completed267/267 TB2 trials at12:33:47.918 UTC after2,163 seconds of native evaluation. The cohort dump closed at12:33:49.272; TBLite began at12:33:49.851. Regional CPU `/benfeuer/mochi-loop-guards-s1-stop-audit-tb2-20261126a` succeeded in36.86 seconds, zero failures/preemptions, without creating model trials. Its audit is paired to sealed S0 and retains the exact frozen TB2 cohort hash.

| TB2 metric | S0 control | S1-stop original |
|---|---:|---:|
| Assessed / N |250/267|252/267|
| Infrastructure / unavailable |17 /0|15 /0|
| Action-loop trials |34|30|
| Action-loop UB |51/267|45/267|
| Repeat-turn share |0.1192787795|0.0481669267|
| Full-credit passes |54|59|
| Mean reward |0.216000|0.234127|
| Median turns |17.5|14|
| Median generated tokens |7,196|4,327.5|
| Median wall seconds |1,109.13|801.85|

Repeat-turn share falls59.62%, exceeding the registered50% reduction. Passes59 exceed the raw40.65 minimum and are far from its ±2-trial repeat boundary. Validity252/267=94.38% meets90%. There are23 runtime repeat_stall stops. Three token-tail loops are report-only; their separate UB18/267 is not the action-loop UB45/267.

The [audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs1-2026.11.26.1%2Fstop%2Fpartial%2Ftb2%2Faudit.json) and `gate-receipt.json` are under `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s1-2026.11.26.1/stop/partial/tb2/`. This is a cohort result, not the full-arm verdict: TBLite, held-out and pooled correctness remain outstanding. A repeat triggered by a later target cohort would replace every arm gate, including TB2, regardless of direction. Original and repeat would stay separate. Both GPU arms remain16/16RUNNING, with RP94/267 TB2 at the12:34:09 probe. OriginalT20 remains13:01:36 UTC and no S2 is submitted.

### 2026-11-26 13:04 UTC — I3 dropped at the original T20 cutoff

The original13:01:36 UTC (08:01:36 EST) S2 start cutoff passed without a submitted or started S2 training job. The scheduled clock watcher observed this at13:02:08 UTC; the receipt was written at13:03:34. The ordered S1 panels were still collecting. The13:03:06/09 probes show stop152/200 TBLite and RP30/200 TBLite, with both16-host GPU children running and zero Iris failures/preemptions. A fresh13:03:34 hub query for the exact experiment S2 root prefix returned no jobs.

Apply DESIGN section4's budget drop: I3 is not run. There is no S2 admission, first rollout, first update, applied update or fixed N; the placeholder8 was never submitted. S2x trained-checkpoint exports and S3 trained-export panels are also not run. No SFT checkpoint will stand in for a trained export, and no antithrash effect is claimed. OriginalT0,T32 and the48-hour deadline remain unchanged. No shared snapshot deletion follows from this drop.

The [decision receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdecisions%2Fi3-t20-budget-drop-2026.11.26.json) is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/decisions/i3-t20-budget-drop-2026.11.26.json`, with a separate redacted native S2-job-absence log. S1-stop and S1-rp continue through the same723 trials and frozen limits; the registered one-repeat boundary rule remains unchanged. Stop's TB2 cohort result remains59.62% lower repeat-turn share with59 versus54 passes, but the full-arm verdict still awaits TBLite, held-out and pooled correctness.

STATUS.md was rewritten around current jobs, sealed evidence, thresholds and this budget decision. Its complete previous history remains in STATUS-history-through-20261126_0802.md. Invalid old controls and obsolete TB2 stitching instructions are explicitly superseded.

### 2026-11-26 13:10 UTC — Repetition penalty fails the TB2 action-loop gate

S1-rp completed267/267 TB2 at12:57:03.358 UTC after2,291 seconds. Its dump closed at12:57:04.677; TBLite began at12:57:05.255. Regional CPU `/benfeuer/mochi-loop-guards-s1-rp-audit-tb2-20261126a` succeeded in38.36 seconds with zero failures/preemptions and no new model trials. The audit pins the sealed S0 baseline and exact frozen TB2 cohort hash.

| TB2 metric | S0 control | S1-rp original |
|---|---:|---:|
| Assessed / N |250/267|257/267|
| Infrastructure / unavailable |17 /0|10 /0|
| Confirmed action-loop trials |34|36|
| Action-loop UB |51/267|46/267|
| Repeat-turn share |0.1192787795|0.1654309546|
| Full-credit passes |54|61|
| Mean reward |0.216000|0.237354|
| Median turns |17.5|17|
| Median generated tokens |7,196|5,619|
| Median wall seconds |1,109.13|1,023.60|

I2 requires candidate action-loop UB≤17, half S0's34 confirmed action loops. The observed36 confirmed+10 unassessed=46 fails. Repeat-turn share also rises38.69% (descriptive, not the I2 primary). Correctness61passes exceeds the raw40.65 minimum; validity257/267=96.25% meets90%. Neither46UB nor61passes is within its ±2-trial boundary. Five token-tail loops and tailUB15 are report-only: using15 as the action UB would wrongly reverse this result.

The [audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs1-2026.11.26.1%2Frp%2Fpartial%2Ftb2%2Faudit.json) and `gate-receipt.json` are under `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s1-2026.11.26.1/rp/partial/tb2/`. Both panels continue; a boundary trigger from another target cohort would still require one full-arm repeat that replaces all gates, regardless of direction. No original trial is discarded or selected best-of.

### 2026-11-26 13:17 UTC — Stall stop improves TBLite but misses its registered target

S1-stop completed200/200 TBLite at13:07:43.683 UTC after2,034 seconds. Its dump closed at13:07:45.016 and held-out began at13:07:45.181. Regional CPU `/benfeuer/mochi-loop-guards-s1-stop-audit-tblite-20261126a` succeeded in36.95 seconds, zero failures/preemptions and no model trials. Frozen cohort hash matches sealed S0.

| TBLite metric | S0 control | S1-stop original |
|---|---:|---:|
| Assessed / N |194/200|197/200|
| Infrastructure / unavailable |6 /0|3 /0|
| Action-loop trials |20|13|
| Action-loop UB |26/200|16/200|
| Repeat-turn share |0.0974277259|0.0560357675|
| Full-credit passes |66|69|
| Mean reward |0.372441|0.372324|
| Median turns |14|12|
| Median generated tokens |7,778|5,620|
| Median wall seconds |847.24|684.61|

Repeat-turn share falls42.48%, below the registered50% reduction. Candidate0.0560357675 exceeds the fixed0.0487138630 cap, so this cohort fails I1's primary gate. Correctness69passes≥56 and validity98.5% hold. There are10 runtime repeat_stall stops. Tail2/UB5 is report-only, distinct from actionUB16/200.

No TBLite repeat is triggered: passes69 are outside54–58, and the preregistered ±2-trial rule has no count conversion for I1's ratio. Do not relax the target or rerun this valid cohort to improve its score. The original stop arm misses the all-target criterion even though TB2 passed. Held-out and pooled correctness still require full measurement; a boundary trigger from held-out could require one full-arm repeat, which would replace every gate regardless of direction.

[Audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs1-2026.11.26.1%2Fstop%2Fpartial%2Ftblite%2Faudit.json) and `gate-receipt.json`: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s1-2026.11.26.1/stop/partial/tblite/`. At13:12 UTC stop26/256held-out and RP114/200TBLite, with all32 GPU hosts running and zero Iris failures/preemptions.

### 2026-11-26 13:35 UTC — Full saved S0 cost distributions validated

Regional CPU `/benfeuer/mochi-loop-guards-s0-cost-profile-20261126b` succeeded in31.04 seconds, zero failures/preemptions, with no new model trials. All four graded-token medians exactly match the sealed audit. Diagnostic-a failed this equality because its sampler incorrectly omitted zero-token cost rows; retaining them fixes the derived calculation without changing canonical audit data or model results.

| Cohort, all saved attempts | Mean assistant-response tokens | Token p95 | Token max | Mean wall seconds | Wall p95 | Wall max |
|---|---:|---:|---:|---:|---:|---:|
| TB2 |8,626.32|21,709.2|23,294|1,196.28|1,877.89|2,117.23|
| TBLite |11,226.48|24,663.45|26,071|1,051.34|1,878.18|1,989.99|
| SWE100 |10,420.92|23,245.05|26,227|1,108.64|1,918.55|2,142.78|
| Held-out |8,674.28|26,849.75|91,058|989.52|2,629.60|10,857.81|

The graded held-out subset remains253 trials, mean963.60 seconds andp952,568.20 seconds. Per-trial attempt timing describes a saved attempt; native phase timing additionally includes concurrency, retries and tail completion. The corrected S0 held-out phase was10,863 seconds. Do not multiply the entire sandbox-bound phase by the128-to12 inference-GPU ratio; the smaller RL inference allocation must be accounted for in its model-service component alongside task tail times, optimizer work and checkpoint saves.

Native primary vLLM generation counters report2,658,970 tokens over2,376.08 seconds on128 GPUs,8.74263344 tokens/second/GPU end to end. The terminal-turn response metric is only49,182 tokens and is not the full agent cost. The saved all-turn TB2 token sum is2,303,228; engine-service counters and saved-trial arrays have different scopes. Neither terminal-turn totals nor checkpoint-save time alone are a complete RL update budget.

[Cost profile](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.26%2Fs0-full-cost-profile.json): `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/s0-full-cost-profile.json`. The sibling `s0-full-cost-profile-analysis.py` and `s0-full-cost-profile-receipt.json` preserve analysis source and provenance; sourceSHA`4aca7762655265f4af7065d1cb937e39b102ec6dffa0d5ce9a72e676fc85defa`. CPU watchdog validation reports informational success and its entry is retired. No RL job or fixed N follows from this cost calculation; effective budget authority is unchanged.

### 2026-11-26 13:47 UTC — Repetition penalty also misses the TBLite UB target

S1-rp completed200/200 TBLite at13:32:15.932 UTC after2,111 seconds; the dump closed at13:32:17.198. Regional CPU `/benfeuer/mochi-loop-guards-s1-rp-audit-tblite-20261126a` succeeded in35.82 seconds, zero failures/preemptions and no model trials. The cohort hash exactly matches sealed S0.

| TBLite metric | S0 control | S1-rp original |
|---|---:|---:|
| Assessed / N |194/200|194/200|
| Infrastructure / unavailable |6 /0|6 /0|
| Confirmed action-loop trials |20|10|
| Action-loop UB |26/200|16/200|
| Repeat-turn share |0.0974277259|0.0680948738|
| Full-credit passes |66|70|
| Mean reward |0.372441|0.384725|
| Median turns |14|12|
| Median generated tokens |7,778|6,313|
| Median wall seconds |847.24|711.65|

Confirmed action loops halve20→10, but the registered I2 gate uses the candidate upper bound:10confirmed+6unassessed=16, exceeding the cap10 (half S0's20 confirmed). Primary FAIL; correctness70≥56 and validity97% hold. Repeat-turn share falls30.11%, descriptive for I2. Tail4/UB10 is a separate report-only measure and must not substitute for actionUB16. A valid97% cohort is not rerun to improve its UB.

Neither UB16 nor passes70 triggers the ±2-trial count rule. Both arms have now completed TB2 and TBLite without a boundary trigger; held-out may still trigger the registered single full-arm repeat, replacing all gates regardless of direction. At13:41 UTC stop235/256held-out and RP77/256held-out, all32 GPU hosts running with zero Iris failures/preemptions. Original trials remain intact.

[Audit](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs1-2026.11.26.1%2Frp%2Fpartial%2Ftblite%2Faudit.json) and `gate-receipt.json`: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s1-2026.11.26.1/rp/partial/tblite/`.

### 2026-11-26 14:57 UTC — Stall-stop arm sealed: loop gains do not preserve held-out correctness

All723 original stop-arm trials have completed. The root and all16 GPU tasks succeeded with zero failures or preemptions. Regional CPU `/benfeuer/mochi-loop-guards-s1-postprocess-stop-20261126a` succeeded, auditing the full panel, promoting its closed trajectory sink by verified in-region copy and refreshing42 saved evaluation summaries. The report has no evaluation error and was republished for exact attempt `a6b43090575a`. Full cohort hashes match S0; TB2 and TBLite metrics exactly match their earlier partial audits.

| Cohort | Assessed/N | Action loops / UB | Repeat-turn share | Reduction vs S0 | Passes vs S0 | Gate |
|---|---:|---:|---:|---:|---:|---|
| TB2 |252/267|30 /45|0.0481669267|59.62%|59 vs54|Pass|
| TBLite |197/200|13 /16|0.0560357675|42.48%|69 vs66|Primary fail|
| Held-out |247/256|16 /25|0.0066515250|94.53%|101 vs117|Correctness fail|

The arm has696 assessed trials,27 infrastructure outcomes and0 unavailable verdicts. All cohorts meet90% validity. Held-out correctness requires at least105 passes (raw104.2), so101 fails. Pooled229 passes versus237 control exceed the pooled minimum223 (raw222.54); pooled correctness cannot offset the held-out breach. The held-out action-loop incidence16 confirmed versus13 baseline rises despite the large reduction in repeated turns; both metrics remain reported. There are15 held-out runtime stall stops.

No target correctness count lies within±2 of its raw threshold, and I1's primary ratio has no trial-count trigger. The original stop arm is therefore **FAIL with no repeat**. No detector, timeout, gate or panel changed. Repetition penalty remains in its original held-out phase; its own eventual boundary decision is separate.

[Full gate receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Faudits%2Fs1-2026.11.26.1%2Fstop%2Fgate-receipt.json): `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s1-2026.11.26.1/stop/gate-receipt.json`. Sibling audit, retention-promotion and report-publication receipts preserve evidence. [Tracking result](https://github.com/marin-community/marin/issues/12264#issuecomment-7195556024).

Conditional repeat plans have passed with identical full scientific configuration,723 trials and explicit new revision; none was launched for stop. A regional CPU-only paired RL plan check also passed on the real pool for a draft candidate N2,60 GPUs per arm and identical configuration except antithrash. This is readiness evidence only: no N was fixed, no RL job was submitted and no original budget-drop decision was reversed. A peer notice alleges an extension but supplies conflicting final clocks and is marked informational/untrusted. Direct user clarification is pending; the recorded original clock remains effective.

### 2026-11-26 16:14 UTC — Original RP arm sealed; held-out count requires the registered full repeat

The original RP1.1 arm completed all723 trials. Root `4bd2a367` and all16 GPU tasks succeeded with zero failures/preemptions. Held-out completed16:03:32.725 UTC after9,075 seconds; its dump closed16:03:36.779. Regional CPU `/benfeuer/mochi-loop-guards-s1-postprocess-rp-20261126a` succeeded: full audit, verified permanent closed-sink trace copy,42 saved evaluation summaries with no report error, and verified publication for exact attempt `a824d6ccf60d`. Frozen cohort hashes and both partial audits match exactly.

| Cohort | Assessed/N | Action-loop trials | Action-loop UB | UB cap | Passes vs S0 | Repeat-turn share |
|---|---:|---:|---:|---:|---:|---:|
| TB2 |257/267|36|46|17|61 vs54|0.1654309546|
| TBLite |194/200|10|16|10|70 vs66|0.0680948738|
| Held-out |250/256|18|24|6.5|103 vs117|0.1742603920|

All three primary action-loop UB gates fail. The held-out share rises43.24% relative to S0, descriptive for I2. Held-out correctness103 is below its raw104.2 minimum (required integer105), while pooled234 passes meet the raw222.54 minimum. All cohorts meet90% validity:701 assessed,22 infrastructure and0 unavailable overall. Held-out token-tail UB6 is separate from action-loop UB24 and must not substitute for it.

The original held-out correctness count103 lies within±2 of raw104.2. The rule registered before S1 therefore requires exactly one repeat of the entire723-trial RP arm, using the same panel and seeds. The repeat supplies **all gates regardless of direction**, including TB2 and TBLite. Original and repeat remain separate; no pooling, best-of selection or third repeat. [Original result and trigger](https://github.com/marin-community/marin/issues/12264#issuecomment-7197268511).

A scratch launcher first stopped locally before submitting any job because it omitted the native entry module. [Source0f433bbb80012474fa79f11ed69d367cb136aaa2](https://github.com/marin-community/marin/commit/0f433bbb80012474fa79f11ed69d367cb136aaa2), archived here as `f1fe7ded41`, pins the selected preset's revision to2026.11.26.2 so the standard module reconstructs the same graph remotely. Required lint and the standard native plan pass. Materialized entire SkyRL, inputs, allocation and priority exactly match the original preflight. This changes run identity only; no scientific configuration or detector changes.

The standard module repeat was accepted16:11:49.538 UTC: `/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s1-rp-7b66e20b68-in-d88ffd61-2026.11.26.2-coordinator-c20eb2fd`. At16:13:53.601 all16 GPU tasks were RUNNING with zero failures/preemptions; child suffix `a7384a7ca776`. It uses64GB200,16hosts128GiB,production08a,allracks,TP1/BF16 and PIECEWISE CUDA graphs, the same723 frozen trials/seeds and RP1.1. Native wait, monitor and own watchdog are attached. Stop remains FAIL with no repeat. No RL job is submitted.

[Repeat launch](https://github.com/marin-community/marin/issues/12264#issuecomment-7197338886). [Receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.26%2Fs1-rp-repeat1-launch-receipt.json): `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/s1-rp-repeat1-launch-receipt.json`. Original full audit/gate/report/retention receipts are under `audits/s1-2026.11.26.1/rp/`; original traces are permanently retained at `traces/s1-rp`. The failed conditional scratch helper is superseded by the standard-module launch receipt and retained as history.

Host-memory receipt, bounded before15:40 UTC: stop16tasks/2804samples peaked66.25390625GiB, RP16tasks/3515samples peaked67.1298828125GiB (RP still running at that query). Both remain below128GiB requests. Bounded exact query and per-host identity are preserved in `diagnostics/2026.11.26/s1-host-memory-1038.{json,sql}`; hub forwarding can lag and native Iris state owns liveness.

## Final decision

Baseline incidence and all four cohort validities are sealed. Offline stall precision is29/30 and passes its gate. The full stop arm fails TBLite's50% reduction and held-out correctness; no repeat is triggered. The original RP arm fails allthree action-loop UB gates and held-out correctness; its103 held-out passes trigger one full723-trial repeat. The repeat is running and will determine every RP gate regardless of direction. No intervention has been selected. I3 remains dropped under the original13:01:36 UTC November26 cutoff pending direct confirmation of revised clocks; no S2, exports or S3 are run. No Marin contribution PR has been opened.


## User overrides applied by fresh worker, November 26

Owner `01a218ad-6f81-78f3-8e73-29b1684d79f6`; coordinator `b610073d-d871-4fd6-a41f-c087d2723ec1`.
These binding user overrides supersede conflicting entries elsewhere in this record; historical receipts and measurements are retained.

1. **User override — deadline +24 h.** T0 stays November 25 12:01:36 EST. T20 is November 27 08:01:36 EST; T32 is November 27 20:01:36 EST; the overall deadline is November 28 12:01:36 EST. I3 is restored; the original T20 budget-drop receipt remains historical. Receipt: `decisions/clock-extension-user-override.json`.
2. **User override — correctness thresholds ×0.75.** Unrounded thresholds are TB2 30.4875 (integer minimum31), TBLite42, held-out78.15 (integer79), pooled166.905 (display166.9; integer167). Repeat windows use these unrounded values: TB2 counts29–32, TBLite40–44, held-out77–80. Validity and loop-incidence gates are unchanged. Receipt: `decisions/correctness-gate-user-override.json` preserves original and recomputed verdicts side by side.
3. **User override — checkpoint conservation.** S1 produces no trained weights: guards act on SFT400407. Retention policy: permanently conserve the regional initializer and exclude it from every pruning sweep. The permanent `.conservation.json` marker and three decision receipts were published at11:33 EST; automatic sweep enforcement has not been verified. HF revision `open-athena/Marin-Mochi-535B-A23B@sft-fullab-v2-step-400407`, commit `cfc315a0dcd12d8be5eb0023c0693c435af98998`, contains206 files totaling1,070,987,037,568 bytes. It is PUBLIC. Setting private failed with HF403: open-athena private-storage limit reached. Private conservation is BLOCKED on org storage capacity; no claim of a private copy. Regional file/size/hash verification is in progress. Any S2 trained weights must be exported through an08a CPU job to PRIVATE open-athena repos, with OpenMDW1.1 LICENSE, required research-artifact and responsible-use text, generation config, and permanent regional retention. Receipt: `decisions/checkpoint-conservation-user-override.json`.

| Arm | Original verdict | Verdict under user override | Correctness | Repeat |
|---|---|---|---|---|
| S1-stop | FAIL: TBLite loop reduction and held-out correctness | FAIL: TBLite loop reduction42.48%<50% |59/69/101; pooled229, all pass overridden thresholds | No trigger |
| S1-rp | FAIL; final pending old correctness-triggered repeat | FAIL: action UB46/16/24 exceeds unchanged caps17/10/6.5 |61/70/103; pooled234, all pass overridden thresholds | No trigger; canceled |

The RP repeat exact root `/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s1-rp-7b66e20b68-in-d88ffd61-2026.11.26.2-coordinator-c20eb2fd` was canceled with `iris-cancel-own` at11:28 EST after adoption. The original held-out103 is24.85 above78.15, outside±2; no correctness or UB trigger fires. Canceled repeat data do not replace or pool with originals. The listed retired watcher PIDs35210/35250/35895/35898/53297 were already absent on adoption.

S2 plan: N4 fixed identically before submission on source855c6c4c4f, primary measure repeat-turn share, based on the existing conservative7h/update budget plus3h startup reserve. Two60-GB200 arms (48policy+12inference each), production08a/all racks/TP1BF16/PIECEWISE graphs, same frozen pool/seed/LR/initializer; antithrash is the only treatment difference. Preserve each completed update; no new update after T32. Submission pending completed override receipts and final N4 native plans. Existing N2 native preflight passed; do not redo completed environment preparation.



## November 26, 11:38 EST: S2 preflight and checkpoint inventory verified

The corrected final native N4 preflight `/benfeuer/mochi-loop-guards-s2-fixed-n4-plan-20261126b` SUCCEEDED in32.64 seconds with zero failures/preemptions. Both arms use source855c6c4c4f, revision2026.11.26.3, four updates,60 GB200/15hosts/704GiB per arm, production08a/all racks/TP1BF16/PIECEWISE graphs. Entire inputs and every SkyRL setting match except antithrash. The primary measure, repeat-turn share, and budget were fixed in `decisions/s2-fixed-n4-preregistration.json` before either launcher started. Preserve every update; stop new updates after November27 20:01:36 EST. The31-hour paired estimate corresponds to at most3,720 GPU-hours if both allocations run for the full budget. This is an estimate, not observed cost.

Regional conservation check `/benfeuer/mochi-loop-guards-conservation-check-20261126b` SUCCEEDED. All199 common files match in hashes and sizes, including all193 weight shards and every shard referenced by the model index. Weight SHA256 values were compared with the retained regional verified-copy/model manifest; current regional sizes were checked and small model files rehashed. The weights were not rehashed afresh. HF has seven repository-only metadata/card files. Regional verification receipt: `conservation/sft-400407-verification-receipt.json`. HF remains public and private conversion is blocked on open-athena private-storage capacity.

The first fresh CPU publication/preflight attempts used BELAY credentials that could list but returned403 on reads. They were canceled and replaced with the established `secrets.env` credential source. No model trials were created by these CPU attempts. The already-published override decisions and permanent marker were read back successfully.


## November 26, 11:46 EST: paired antithrash S2 submitted

Both real S2 coordinators were accepted on source0987bdac6f (tree6658f0dae1), revision2026.11.26.3, N4 per arm and primary measure repeat-turn share. Control acceptance11:46:18.950707 EST; treatment11:46:44.992758 EST. Both request60 GB200 each at production priority on08a, all racks, TP1/BF16 with PIECEWISE CUDA graphs. No training-start or completed-update claim is made at coordinator acceptance.

- control: `/benfeuer/rl-data-task-health-attempts-by-run-users-benfeuer-checkpoints-mochi-loop-guards-s2-antithrash-control-6658f0dae1-in-db57e3bd-2026.11.26.3-coordinator-745232d2`. Output `s3://marin-us-east-08a/tmp/ttl=3d/marin/users/benfeuer/checkpoints/mochi-loop-guards/s2-antithrash-control-6658f0dae1-in-db57e3bd/2026.11.26.3`.
- treatment: `/benfeuer/rl-data-task-health-attempts-by-run-users-benfeuer-checkpoints-mochi-loop-guards-s2-antithrash-treatment-6658f0dae1-in-db57e3bd-2026.11.26.3-coordinator-5f428851`. Output `s3://marin-us-east-08a/tmp/ttl=3d/marin/users/benfeuer/checkpoints/mochi-loop-guards/s2-antithrash-treatment-6658f0dae1-in-db57e3bd/2026.11.26.3`.

The final module entrypoint explicitly configures CoreWeave storage before artifact construction, matching the regional preflight environment. The corrected local native plan passed. Local launcher diagnostics also showed time spent in native configuration composition; the earlier storage explanation alone did not establish the entire delay. Stopped local pre-submit attempts created no Iris jobs. Both final submissions used the standard module entrypoint and passed its native plan/candidate checks.

The shared job watchdog owns both exact coordinator IDs under fresh session01a218ad; the lane monitor's first scheduled probe is11:48:44.993 EST and then every570 seconds. Native foreground waits remain attached. I3 training must start before November27 08:01:36 EST, and no new update may start after November27 20:01:36 EST. Export and preserve both trained arms; private HF upload remains blocked until organization storage capacity is resolved.

## November 26, 11:57 EST: initial S2 fails before training; upstream NVML fix validated

Both initial S2 roots and GPU children failed before model startup. Control's root failed
at11:49:32 EST and treatment's at11:49:57 EST. The earlier15/15 RUNNING observations were
startup states. No optimizer update, trained checkpoint or antithrash result was produced.

All30 per-host NCCL gate reports passed60-rank collectives and PyNCCL initialization. The
subsequent `task_runtime.main -> node_physical_labels` check failed because the NVML
probe passed a ctypes output structure by value. Current Marin main uses `byref(info)`.
The minimal upstream pointer fix and writable-result regression were adopted without
changing the health gate, topology, science or resource settings. All eight physical-role
tests pass; the new regression fails when the pointer fix is removed.

The coordinator authorized one paired restart after validation on a fresh revision,
`2026.11.26.4`, retaining N4, repeat-turn share, production08a and all racks. Source
`d08093273e` is committed; the focused safe suite is finishing before push and relaunch.
The [incident record](https://marina.oa.dev/echo/wiki/1090) preserves the diagnostic
discriminator. Private HF conservation remains blocked on organization storage capacity.

The coordinator preserved all93 temporary helpers in archive commit `b866da0521` and
assigned their consolidation to a separate branch. New worker operations use checked-in
experiment commands and central Iris/watchdog tooling. Adopt the consolidation after S2
is training; no new temporary scripts are created.

## November 26, 13:56 EST: revision4 accepted and physical placement verified

The resumed worker verified that `d08093273e` was pushed and no revision4 jobs existed.
Completing the upstream regression exposed a second pre-training defect: Ray rejects
`UUID:clique` as a label value. Source `ecc9fb1bae` adopts main's `UUID.clique` encoding
and validates the produced labels with Ray itself. The regression failed on the colon
encoding; all eight physical-placement tests pass with the correction. The earlier
pointer fix also passed176 affected safe tests.

Both revision4 arms were submitted once from the checked-in module on source
`ecc9fb1bae`, tree `c568ddc7d1`, with the same N4/science/production08a/all-rack allocation.
Control accepted13:51:23.963 EST, root suffix `85cd69d1`, GPU child `a3144ba94e53`.
Treatment accepted13:51:46.099 EST, root suffix `c98995d1`, GPU child `a00eaf3f9e28`.
Exact identities are in [the submission receipt](analysis/user-overrides-20261126/s2-r4-submission-receipt.json).

Iris verified15/15 GPU tasks RUNNING in both children with zero failures/preemptions.
The native runtime verified physical placement and launched the training drivers at
13:54:26 EST control and13:54:39 EST treatment. First rollout/update has not yet been
observed. Both roots and their child progress probes are registered in the central
watchdog; its official validation tick passed with no alerts. Native Iris waits and
log-follow commands own monitoring. Consolidation remains unmerged until training
activity is confirmed. Rhizome is unbound in the resumed process; STATUS and this
notebook preserve the handoff while direct routing is unavailable.

## November 26, 14:16 EST: both training loops started; consolidation adopted

Control entered `_train_loop` at14:08:51.538 EST and treatment at14:09:04.855 EST.
Both started32 rollout workers. Their initial policy synchronizations completed at
14:08:57.392 and14:09:10.738 EST. This precedes the extended T20 cutoff,
November27 08:01:36 EST. No completed optimizer update is verified.

Both arms now emit repeated Daytona connection failures. Interrupted trajectories are
masked by the configured default error policy. This is a validity concern requiring
infrastructure-outcome accounting before interpreting the paired result; training-loop
entry establishes neither completed updates nor scientific validity. Native waits,
log followers and the central watchdog remain attached to the exact revision4 roots.

After both loop entries, consolidation `27a544e838` was adopted through merge
`1e8882d618`, with import-boundary/formatting cleanup `71297b9d15`. Required lint and
all24 affected safe tests passed. The pushed experiment branch now contains the
checked-in worker commands; running jobs retain immutable launch source `ecc9fb1bae`.
The classification in `CONSOLIDATION.md` accounts for all93 archived helpers:
41 duplicate existing tooling,11 promoted experiment logic and41 history-only scripts.
No temporary worker scripts are required. The two S1 verdicts remain FAIL under the
user override; correctness passes, unchanged loop gates fail, and no RP repeat is due.

## November 26, 15:06 EST: treatment update verified; partial failure receipts frozen

Regional native telemetry verifies treatment optimizer step1 at14:49:17.976 EST:
256 consumed samples,702,670 response tokens and702,668 loss tokens. The training-step
policy synchronization completed14:49:24.739 EST. Control optimizer completion remains
unverified. The earlier training-loop-entry record is not an optimizer-completion claim.

Both launches explicitly configure the checkpoint callback to save every update and
keep every update. The treatment checkpoint prefix was absent in the15:00 metadata
check; no durable checkpoint or trained HF export is claimed. Step-end metrics had
not yet appeared in native telemetry or stored run records. This is not sufficient
evidence to diagnose a checkpoint save failure. Preserve and export only committed
checkpoints through an in-region08a CPU job.

Compact published task-health receipts were copied by15:05:12 EST, frozen and hashed
with native fsutil/jq/shasum. The standard extractor requires a stopped run and was not
invoked. The [snapshot receipt](analysis/user-overrides-20261126/s2-r4-task-health-snapshot-1505.json)
records the counts and limitations. Raw compact receipts and per-file SHA256 inventories
are preserved at `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/s2-r4-task-health-snapshot-1505`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdiagnostics%2F2026.11.26%2Fs2-r4-task-health-snapshot-1505)).

| Arm | Unique receipt attempts | Graded | Unavailable | Infrastructure error |
|---|---:|---:|---:|---:|
| Control |513|500|12|1|
| Treatment |515|503|12|0|

Control's non-graded exceptions are3 ConnectionErrors,2 environment-start timeouts,
7 VerifierRuntimeErrors and1 verifier timeout. Treatment has9 ConnectionErrors,
1 environment-start timeout and2 VerifierRuntimeErrors. No duplicate attempt IDs were
found. The copied receipt sets are incomplete live-run diagnostics and copying is not
atomic across files. Their97.47%/97.67% graded shares do not establish the consumed
batch's masking rate, satisfy an evaluation validity gate or support a treatment effect.
Next obtain committed checkpoints and step-end batch failure metrics; preserve all
outcomes and keep the registered clocks and gates unchanged.

## November 26, 15:18 EST: both arms updated; permanent conservation started

Control completed optimizer step1 at15:06:13.607 EST; treatment completed step2 at
15:08:08.281 EST. Both roots remain RUNNING with zero failures/preemptions at15:11 EST.
Treatment step1's native batch metrics are256 requested trials,7 failed instances,
36 failed trajectories and2 masked trajectories (0.78125%). The failed-trajectory
count is not an infrastructure-error count. These training diagnostics do not replace
evaluation validity gates or establish an antithrash effect.

Treatment's committed checkpoint marker reads1. Its step directory contains policy
shards, trainer state and1.5GB rollout state. An08a CPU job accepted15:16:37 EST is
running native `fsutil verified-copy` into the permanent conservation prefix. The
verified-copy completion manifest is still required before claiming successful
permanent conservation. [Receipt with exact job IDs and destinations](analysis/user-overrides-20261126/s2-r4-conservation-start-1518.json).

The consolidated CPU launcher's CLI failed locally before any submission because its
parser named `--secret` as `secret` while the launcher read `secrets`. Regression
coverage reproduced that failure; source `d11235af4d` sets the explicit destination.
Required lint and all7 affected safe tests passed before committing, pushing and using
the repaired command. No temporary worker script was introduced.

The archive lane freed1.350TB of HF private storage. Retrying the original Mochi repo's
privacy change still returned403; the repo contains63.2TB across its history. A new
private destination for only the sealed SFT revision was created and its private
visibility verified before copy. An08a CPU job accepted15:17:39 EST invokes native
`hf repos cp` on the pinned source commit, using server-side copy operations.
The source model card has both required paragraphs, and LICENSE byte-matches official
OpenMDW1.1. The source generation config lacks Marin sampling defaults; after copying,
upload [the prepared generation config](analysis/user-overrides-20261126/sft-private-copy-generation_config.json)
and verify the final private inventory. Tokenizer metadata verifies EOS128001 and
chat-template end-of-turn128009. Private conservation remains pending these checks.

Both CPU jobs are registered in the central watchdog with native waits. Training
continues on immutable source `ecc9fb1bae`; GPU allocations and scientific gates are
unchanged. Next verify conservation-job completion, private-copy metadata and content,
then preserve later committed checkpoints and evaluate the registered paired panels.

## November 26, 15:24 EST: private SFT conservation verified

The private copy succeeded on08a in21.89 seconds. Final destination
[`open-athena/Marin-Mochi-535B-A23B-sft-fullab-v2-step-400407-conserved`](https://huggingface.co/open-athena/Marin-Mochi-535B-A23B-sft-fullab-v2-step-400407-conserved)
is verified private at commit `239fd384cc99a04d3a61c0f6931f24de4ee41b73`.
All206 source paths are present. All193 weight shards match the retained regional
manifest in SHA256 and size;205 unchanged files also match the pinned source HF tree
in size, Git blob ID and LFS metadata. The only file change is the prepared generation
config, adding Marin sampling defaults and EOS IDs128001/128009. Its70-byte size
difference is intentional. Every shard referenced by the index is present.

The published model card, official OpenMDW1.1 LICENSE and generation config were read
back and byte-compared with the checked candidates. Both required model-card
paragraphs and license metadata are present. Model weights were not downloaded to the
Mac or rehashed afresh. [Verification receipt](analysis/user-overrides-20261126/sft-400407-private-conservation-receipt.json)
records the exact method and remaining retention-enforcement limitation.

The regional permanent marker was updated to reference this private copy and read
back successfully. The superseding **user override** decision is
`decisions/checkpoint-conservation-private-copy-user-override-2026.11.26.json` at the
experiment prefix; the older blocked receipt is retained as history. The original
63.2TB HF repository remains public and its visibility change remains quota-blocked.
Private conservation of the sealed SFT revision is complete through the separate copy.
The completed CPU-copy watchdog entry can be retired after this verified handoff.
The S2 treatment-step1 permanent verified copy remains in progress.

## November 26: superseding user decision accepts public SFT conservation

New coordinator inbox entries labeled15:25 and15:30 EST relay the user's decision to
leave the SFT repo public, and confirm that the coordinator deleted the1.07TB private
copy. That decision supersedes the earlier private-copy instruction. No more HF copies
will be made. This lane did not delete the copy. The preceding private-conservation
milestone is historical and no longer describes a surviving repository.

Current conservation is the sealed public
`open-athena/Marin-Mochi-535B-A23B@sft-fullab-v2-step-400407`, commit
`cfc315a0dcd12d8be5eb0023c0693c435af98998`, plus the permanent never-prune08a source.
Its existing verification remains206 HF files,199 common files and193 verified weight
shards, with all index shards present. No model weights were downloaded again.
[Superseding user-override receipt](analysis/user-overrides-20261126/sft-400407-public-conservation-user-override.json)
records the correction and private-copy deletion. The private verification receipt is
marked superseded. Regional S2 checkpoint conservation continues; the HF copy ban is
recorded in the current handoff.

## November 26, 15:37 EST: control reaches update2, treatment reaches update3

Native regional consumed-work counters verify control update2 at15:25:03.989 EST and
treatment update3 at15:27:15.178 EST. Policy sync completed at15:25:09.487 and
15:27:21.949 EST. Both roots remain RUNNING with zero failures/preemptions at15:33 EST.
N remains4 for each arm. Committed checkpoint markers are control1 and treatment2.

| Arm/step | Trials | Failed instances | Failed trajectories | Masked trajectories |
|---|---:|---:|---:|---:|
| Control1 |256|6|34|1|
| Treatment1 |256|7|36|2|
| Treatment2 |256|9|27|6|

These native training-batch diagnostics replace log-line counting for batch masking
accounting. Failed trajectories are not all infrastructure failures; the counts do
not establish evaluation validity or an antithrash effect.

The treatment-step1 resumable checkpoint measures4,292,558,588,036 bytes (4.29TB,
3.90TiB) across59 objects, with policy shards around83GiB. Its16-CPU permanent-copy
job is RUNNING, zero failures/preemptions. Two status records verify `data.pt` and
`policy/.metadata`; the completion manifest remains absent. Completed-file records
do not measure in-flight multipart bytes, so no throughput or stall claim is made.
Keep the permanent copy at one worker until it validates. Native keep_every1 retains
newer source checkpoints meanwhile; their ttl=3d paths are not permanent copies.
[Progress receipt](analysis/user-overrides-20261126/s2-r4-progress-1537.json).

The public-SFT conservation correction remains authoritative. No HF operation, training
configuration change, cancellation or resubmission occurred. Next verify the permanent
copy and retain subsequent committed updates before the registered paired evaluation.

## November 26, 15:54 EST: regional S2x conversion command validated

The existing export-command template omitted native manual export's required model
path. Its dry run reproduced the parser error before any job submission. Source
`a989ad12e5` binds the replayable path/URI/identity from `inputs.model`, uses an explicit
regional destination and the existing frozen SkyRL CPU environment, and safely quotes
the generated command. The preflight receipt now binds the actual authored launch
document instead of an unresolved placeholder. A CLI-wire regression covers the model
locator and paths containing spaces.

Required lint, all5 affected safe tests, all7 fixture stage plans, the generated native
conversion dry run and a dry run against the actual saved treatment launch passed.
[Conversion preflight receipt](analysis/user-overrides-20261126/s2x-regional-conversion-preflight.json).
No conversion or other new Iris job was submitted, and no local model tensors were
loaded. The live training source remains `ecc9fb1bae`.

The planned step4 conversion uses48 policy GPUs (12x4 GB200) at production priority on
08a, all racks, and writes a model artifact to the permanent regional experiment prefix.
It has no Hub target. Before submission, require the committed step4 marker and a
training GPU gang's verified termination/release. One60-GPU training gang plus the48-GPU
converter is108; adding conversion while both60-GPU arms are allocated would exceed
the128-GPU experiment cap. Use native detached submission and register the accepted ID.
Only a verified completed conversion can initialize a registered S3 panel. The public
SFT conservation/no-more-HF-copies ruling remains binding.

## November 26, 16:03 EST: treatment completes the fixed four optimizer updates

Treatment optimizer step4 completed15:47:56.397 EST; control step3 completed
15:45:13.811 EST. This is optimizer progress, not a terminal-training or checkpoint
conservation claim. At15:59 EST the committed markers were treatment3/control2;
at16:01 EST both children still had15/15 GPU tasks RUNNING, zero failures/preemptions.
The prepared48-GPU conversion remains gated on a committed step4 and a released
training allocation. No conversion or Hub copy was submitted.

New native batch diagnostics: control step2 has256 trials,8 failed instances,
22 failed trajectories and3 masked; treatment step3 has256 trials,10 failed instances,
52 failed trajectories and1 masked. Failure counts are not all infrastructure errors,
and the training metrics do not establish evaluation validity or an antithrash effect.

The4.29TB treatment-step1 permanent-copy job remains RUNNING. It still has two completed
verification records (`data.pt`, `policy/.metadata`) and no completion manifest.
Source inspection confirms the native CLI prints only its final result. Per-file
records do not expose in-flight multipart bytes; the direct CPU task also returned no
iris.task resource samples. Neither transfer throughput nor a stall is inferred.
[Milestone receipt](analysis/user-overrides-20261126/s2-r4-treatment-n4-optimizer-milestone.json).
Next verify final checkpoint publication and GPU release, then submit the prepared
regional conversion within the128-GPU experiment cap. The public-SFT conservation
correction remains authoritative; make no more Hub copies.

## November 26, 16:22 EST: treatment succeeds and regional conversion is accepted

[Treatment training](https://wandb.ai/marin-community/marin-mochi-rl/runs/a00eaf3f9e28)
succeeded at16:07:30.092 EST with all15 GPU tasks succeeded, zero failures/preemptions,
and committed checkpoint4. The control root remained RUNNING at16:21 EST; its
checkpoint marker reached4 at16:22 EST. Both arms retain their fixed N4 configuration.

Iris accepted `/benfeuer/mochi-loop-guards-s2x-treatment-r4-step4` at about16:21 EST:
production08a,12x4 GB200, all racks, immutable source `a989ad12e5`. It converts
treatment step4 into the permanent regional `exports/s2-treatment-r4/global_step_4/policy`
prefix with no Hub target. The treatment training allocation had terminated before
submission; the remaining control60 plus conversion48 bounds total use at108/128 GPUs.
The exact conversion root is registered with the central watchdog. At16:23:29 EST,
all12 GPU tasks are RUNNING with zero failures/preemptions.

The first native submission failed locally on S3 credentials before Iris submission;
the regional inventory confirmed no job was accepted. Standard AWS/fsspec environment
settings using the existing CW credentials resolved it, and the same native command
then submitted once. No executable temporary worker script or source change was used.
Conversion completion and its model manifest are still required before S3 evaluation.
The original4.29TB treatment-step1 permanent copy still has two completed file records;
no conservation-completion claim is made. The public-SFT/no-more-HF-copies ruling holds.
[Submission receipt](analysis/user-overrides-20261126/s2x-treatment-r4-step4-submission.json).

## November 26, 16:26 EST: both N4 arms succeed; control conversion is accepted

[Control training](https://wandb.ai/marin-community/marin-mochi-rl/runs/a3144ba94e53)
succeeded at16:23:47.809 EST with checkpoint4 committed. Native root and all15 GPU
tasks are SUCCEEDED, zero failures/preemptions. Treatment had already succeeded
at16:07:30.092 EST. Both arms completed their fixed four updates from source
`ecc9fb1bae`; the paired S3 evaluation still determines the antithrash result.

The native manual exporter submitted control checkpoint4 once as
`/benfeuer/mochi-loop-guards-s2x-control-r4-step4` about16:26 EST. It uses the same
clean source `a989ad12e5`, production08a/all racks and12x4 GB200 as treatment,
with output `exports/s2-control-r4/global_step_4/policy` and no Hub target.
The exact root is registered with the central watchdog. At16:28:06 EST all12 control
conversion tasks are RUNNING, zero failures/preemptions. Both training gangs have
terminated, so the two48-GPU converters bound total GPU use at96/128.

The original treatment-step1 permanent-copy CPU job remains RUNNING0fail/preempt
at16:26 EST; its completion manifest is still absent at16:25 EST. No further full-state
copy or HF Hub copy was launched. Require both regional conversions to succeed and
verify their model manifests and shard inventories before the paired S3 panels.
[Control conversion receipt](analysis/user-overrides-20261126/s2x-control-r4-step4-submission.json).

## November 26, 16:39 EST: converters advance; durable exports remain pending

At16:37 EST both regional conversion roots have12/12 GPU tasks RUNNING with zero
failures/preemptions. Direct Kubernetes task-container logs show treatment's tensor
conversion at100% (10232/10232) and local generation-config creation at16:34:27.972 EST.
Control restored its model state at16:34:02.087 EST. These are intermediate stages;
both permanent model-output directories were absent at16:38 EST. No completed export,
model-manifest verification or paired S3 evaluation is claimed.

Regional Iris log reads returned no task lines; direct container logs supplied the
progress observations above. The reason for the log-read gap is unresolved. No job
was canceled or resubmitted. The original regional checkpoint-copy completion manifest
is absent at16:37 EST and its completed-file status prefix still has two records.
Both fixed-N4 training arms remain succeeded. The public-SFT/no-more-HF-copies ruling
is unchanged. Next native conversion pass is due around16:47 EST.

## November 26, 16:44 EST: paired S3 presets are available in the live CLI

Source `1cad6c984e` exposes `s3-control-panel` and `s3-treatment-panel`, each pinned
to its own regional S2 revision4/step4 export. Both use the existing frozen panel
builder:723 follow-up trials, seed20261125, temperature0.7/top_p1, TP1/BF16,
CUDA graphs, four DP16 engines (64 GB200 per arm), production08a/all racks.
The stage revision is2026.11.26.1. The CLI budget text now states64 GPUs per panel.

Required lint, all5 affected safe tests, the7-stage fixture preflight and both native
live-model plan paths passed. Fixture plans validate protocol wiring using the SFT
stand-in; native plan success does not verify that the converted model exists.
The live CLI now exposes both presets. No S3 job was submitted. Before launching the
128-GPU pair, require both48-GPU converters to terminate successfully and verify
both output manifests/shard inventories. This preserves the128-GPU experiment cap.
The training and conversion jobs retain their immutable sources `ecc9fb1bae` and
`a989ad12e5`; the public-SFT/no-more-HF-copies ruling remains binding.

## November 26, 16:48 EST: treatment starts regional weight publication

Treatment's direct task-container log confirms14/215 HF-format weight shards published
to the permanent regional output by16:47:18.451 EST, with shard15 starting. The16:47
fsutil listing independently showed the first12 completed weight objects. This is
regional S3 publication; neither converter has a Hub target. The215-shard converted
layout is distinct from the193-shard SFT initializer.

Control reached100% tensor conversion (10232/10232) and wrote local generation metadata
at16:39:19.606 EST; its regional model-output directory was still absent at16:47 EST.
Both roots remain RUNNING12/12 tasks, zero failures/preemptions at16:47 EST. Completed
model manifests and indexed shard inventories remain unverified. No S3 panel submitted.
The original treatment-step1 permanent-copy manifest remains absent and its status
prefix has two completed-file records at16:47 EST. Continue the registered jobs;
next native conversion/copy pass around16:57 EST, with the128-GPU S3 pair gated on
both converters' successful termination and verified model artifacts.

## November 26, 16:52 EST: native export-verification command is ready

Source `cc4e3dcc62` adds the direct module
`experiments.post_training.mochi_loop_guards.export_verification`. It calls SkyRL's
existing completed-export verifier and writes a small receipt containing manifest
identity and indexed shard/tensor counts. The native checks cover completion markers,
manifest-file presence, indexed shard headers and recorded sizes. Weight payloads are
not rehashed. Run in the SkyRL CPU environment with
`FSSPEC_S3_DEFAULT_CACHE_TYPE=none` to disable read-ahead when reading headers.

Required lint passed. The affected-test selector selected zero tests; all8 existing
native CPU export-completion tests passed separately. Direct CLI checks accepted a
tiny valid export and wrote its receipt, and rejected an incomplete export without
writing a receipt. The direct module avoids the top-level experiment CLI's Marin
dependency, which is absent from the standalone SkyRL environment.

No live export verification, new job or model download occurred. The latest live
observations remain the16:47 conversion/publication pass above. Successful terminal
converters plus verified model artifacts still gate the128-GPU S3 pair. Next native
conversion/copy pass remains16:57 EST; no further Hub copies are authorized.

## November 26, 16:55 EST: regional resumable-copy worker activity is verified

The original treatment-step1 verified-copy job remains RUNNING0fail/preempt at16:53
EST, after1h37m. Kubernetes Metrics API is unavailable, and the container lacks `ps`.
Direct read-only proc/cgroup inspection identifies the live `fsutil` process and
shows task cumulative CPU time increasing from9724544763 to9743614679 microseconds
between samples. Current task memory is about7.3GiB; observed peak is10.6GiB.

These observations establish worker activity, without establishing transfer throughput,
remaining time or verified checkpoint completion. Host-network interface counters
cannot be attributed to this task. The latest completion-manifest/status observations
remain16:47 EST: absent manifest and two completed-file records. Keep the existing
copy; no cancellation, resubmission, duplicate checkpoint copy or new allocation.
Next conversion/copy pass remains16:57 EST. The public-SFT/no-more-HF-copies ruling
is unchanged.

## November 26, 16:58 EST: both converted models are publishing regional shards

Direct task-container logs confirm treatment84/215 weight shards published by
16:57:35.307 EST and control51/215 by16:57:34.727 EST. Earlier16:57 fsutil listings
independently showed83 and50 completed objects. Both roots remain RUNNING12/12 tasks,
zero failures/preemptions at16:57:30 EST. Neither listing has the final model manifest
or weight index; final export verification and S3 submission remain pending.

The original treatment-step1 permanent-copy completion manifest is still absent at
16:57 EST, with two completed-file records. No cancellation, resubmission, new copy,
allocation or Hub operation. The registered converters remain within96/128 GPUs;
next native pass around17:07 EST. Launch the128-GPU S3 pair only after both converter
roots terminate successfully and both converted artifacts pass the prepared verifier.

## November 26, 17:06 EST: regional uploads continue without task failures

The17:05 pass verifies both converter roots RUNNING12/12 tasks, zero failures or
preemptions. Direct logs confirm treatment138/215 shards by17:05:29.663 EST and
control104/215 by17:05:32.195 EST. Slightly later fsutil listings contain140 and105
weight objects respectively, with no final model manifest or index. No completed
export verification or S3 submission is claimed.

The original treatment-step1 permanent-copy completion manifest remains absent at
17:05 EST with two completed-file records. No job changes, duplicate copy, additional
allocation or Hub operation. Next native pass moves to17:15 EST following this refresh;
both converters' terminal success and verified artifacts still gate the128-GPU S3 pair.

## November 26, 17:19 EST: treatment export verifies and its converter succeeds

Treatment's final model manifest appeared at17:16 EST. The native completed-export
verifier passed:221 manifest files,215 indexed weight shards,56504 tensors and
1070983169596 recorded bytes. Manifest identity is
`sha256:5b63db0158f073952eaab45db7aca024943ff2679a113bbbec7ebfaef8e3a5ee`.
Verification used header reads with read-ahead disabled, checking completion markers,
manifest-file presence, indexed shard headers and recorded sizes. No weight-payload
rehash or model-tensor download occurred.

At17:18:06 EST the native treatment-converter root is SUCCEEDED, all12 tasks succeeded,
zero failures/preemptions. The model is under the permanent regional experiment prefix;
its parent `.conservation.json` marks permanent retention and exclusion from every
pruning sweep and passed byte readback. Automatic sweep enforcement remains unverified.
The small verification receipt is published under regional `decisions/`.
[Verification and terminal-state receipt](analysis/user-overrides-20261126/s2-treatment-r4-step4-export-verification.json).

Control remained RUNNING12/12 at17:15, with172/215 weight shards confirmed in its
17:15:42.125 EST log. It must finish and pass the same verifier before the128-GPU S3
pair launches. The original treatment-step1 permanent-copy manifest remains absent
at17:15 with two completed-file records. No new job or Hub copy was launched.
Next control-conversion pass around17:25 EST; the public-SFT/no-more-HF-copies ruling
remains binding.

## November 26, 17:25 EST: both S2x exports are verified and converters succeed

Control's final regional manifest appeared at17:22 EST. The native verifier passed
221 manifest files,215 indexed shards,56504 tensors and1070983169596 recorded bytes,
with identity `sha256:f47a91a9007164d11a1a415df275490f3fb89b3656e51bc72a1a425111f1d4eb`.
Header/size verification used read-ahead disabled; no weight-payload rehash or tensor
download. Its parent permanent/exclude-every-sweep marker passed byte readback;
automatic pruning enforcement remains unverified.

At17:23:49 EST, control's converter root is SUCCEEDED, all12 tasks succeeded with
zero failures/preemptions. Treatment already succeeded and verified. A direct pod
check finds treatment's rank0 pod deleted and control rank0 Succeeded with both task
and uploader exited0. Both GPU gangs are finished; no experiment GPU allocation
remains before S3. Both small verification receipts are in regional `decisions/`.
[Control verification and terminal-state receipt](analysis/user-overrides-20261126/s2-control-r4-step4-export-verification.json).

The next milestone is the native S3 plan followed by one submission per arm,64 GB200
each,723 frozen follow-up trials each, production08a/all racks. No S3 or Hub job has
been submitted at this milestone. The original resumable-step1 copy continues
separately; its completion is not needed to consume the verified step4 model exports.
The public-SFT/no-more-HF-copies ruling remains binding.

## November 26, 17:34 EST: paired S3 coordinators and GPU children are running

Both native plans passed on clean source `cc4e3dcc62`/tree `22ae5b80cc`. Control was
accepted17:28:51.657 EST and treatment17:29:23.641 EST, revision2026.11.26.1, exactly
one accepted coordinator per arm. Each requests64 GB200 (16x4),128GiB per host,
production08a/all racks and723 frozen follow-up trials:TB2 267/TBLite200/held-out256.
Both converters had succeeded and released their GPU gangs before submission.

At17:32:24 EST both coordinators and GPU children are RUNNING,16/16 GPU tasks each,
zero failures/preemptions. Saved native configs bind each arm's verified regional
step4 model and the frozen panel selection. They retain TP1/DP16/EP16, BF16,
eagerfalse, seed20261125, temperature0.7/top_p1/repetition_penalty1 and stop IDs
128001/128009. The runtime entrypoint is `skyrl_train.entrypoints.main_generate`.
Both exact coordinator roots and GPU-child probes are registered with the watchdog.

The first local control invocation stopped before submission because the current
shell lacked DAYTONA_EVAL_API_KEY. Loading the dedicated eval key from the documented
Secret Manager source resolved it; no generic account substitution, source change or
duplicate accepted job. Native waits are active. Evaluation results remain pending;
next native pass around17:42 EST. No HF Hub operation. The original resumable-step1
copy continues separately from the verified step4 models.
[Submission and runtime-config receipt](analysis/user-overrides-20261126/s3-pair-submission.json).

## November 26, 17:48 EST: both S3 interconnect gates pass and inference is active

At17:45:30 EST both S3 roots and GPU children remain RUNNING16/16 tasks, zero
failures/preemptions. All16 per-host interconnect reports pass in each arm, covering
all64 local GPU-rank results per gang. Native vLLM logs show active requests and
generated tokens in both arms, with cold-shape JIT compilation enabled. Completed
benchmark-trial counts remain unverified; no validity or treatment-effect verdict.
Small gate reports are preserved under permanent regional `diagnostics/s3-r1/`.

The original resumable-step1 copy now has43 unique verified staging records, including
32 policy shards, totaling2857885578414 bytes against4292558588036 source bytes/59
objects. Its completion manifest remains absent at17:45 EST. Staging verification
does not establish completed permanent promotion. No source/job changes or further
Hub copies. Next native pass around17:55 EST, including cohort progress and copy
completion checks. [Runtime/copy receipt](analysis/user-overrides-20261126/s3-runtime-and-copy-progress-1748.json).

## November 26, 17:58 EST: first TB2 returned-rollout counts are visible

Native progress shows control67/267 TB2 rollouts returned at17:55:24.299 EST and
treatment54/267 at17:55:25.911 EST. Source inspection confirms the progress counter
advances after a rollout returns, including terminal errors. These are not counts of
successful or valid trials. Both GPU children remain RUNNING16/16, zero Iris task
failures/preemptions at17:55; both coordinators remain RUNNING at17:56:54 EST.

Partial logs show AgentTimeoutError in both arms, a sandbox-IP TaskMachineError, and
TmuxBatchProtocolError. The latter logs native default-mask handling.
Ray-deduplicated and repeated log lines do not provide per-trial error counts. No
correctness, infrastructure-rate, validity or treatment-effect verdict is inferred.
Frozen timeouts, sampling, detector and error classification remain unchanged.

The original resumable-copy manifest remains absent17:55 EST; its status inventory
still has43 records, matching the prior file-name inventory. Its native job is RUNNING
0fail/preempt at17:57:32 EST. The last summed staging
snapshot remains2.858TB at17:46 EST. Continue the registered jobs without resubmission
or Hub operations; next native pass around18:05 EST.
[TB2 progress receipt](analysis/user-overrides-20261126/s3-tb2-progress-1758.json).

## November 26, 18:07 EST: TB2 returned-rollout counts advance past half

Native progress records control152/267 returned TB2 rollouts at18:05:13.971 EST and
treatment147/267 at18:04:44.571 EST. These include terminal errors; successful/valid
counts and error rates remain unaudited. Both coordinator roots remain RUNNING at
18:05:37; GPU children RUNNING16/16, zero Iris failures/preemptions at18:05:39 EST.

AgentTimeoutError and sandbox-IP TaskMachineError remain visible. The newer control
excerpt also logs VerifierTimeoutError with native default-mask handling. Earlier
TmuxBatchProtocolError remains part of the qualitative error record. No classification
or frozen-parameter changes, restart, resubmission or Hub operation. Continue the
paired panels without an interim correctness/validity/effect verdict.

The original copy is RUNNING0fail/preempt18:05:38, completion manifest absent and
status inventory unchanged at43 files. The last summed staging snapshot remains
2.858TB at17:46; permanent promotion remains unverified. Next pass18:15 EST.
[TB2 progress receipt](analysis/user-overrides-20261126/s3-tb2-progress-1807.json).

## November 26, 18:33 EST: obsolete S2 checkpoints removed under storage directive

The coordinator's latest inbox directive authorized cleanup as 08a storage filled.
One native `fsutil rm` invocation removed seven obsolete temporary S2 checkpoint
prefixes: control steps 1–4 and treatment steps 2–4. Deletion receipts total 413
objects; the pre-delete inventory totals 30,046,684,020,052 bytes (30.047 decimal TB).
Recursive post-delete listings and exact remaining object counts confirm every
allowlisted prefix is absent. Region-wide free capacity and unfinished multipart
uploads were not measured. The [cleanup manifest](analysis/user-overrides-20261126/storage-cleanup-20261126.json)
records every exact prefix, inventory and proof-log digest.

Treatment step 1 remains intact at 4,292,558,588,036 bytes / 59 objects because the
original conservation-copy job still reads it. Permanent step-4 exports (2.142 TB),
the ongoing conservation copy (2.858 TB listed across staging and status records),
and the protected SFT 400407 regional copy (1.071 TB) retain their prior inventories.
The copy completion manifest remains absent at 18:32 EST. S3 uses the permanent
exports without a distributed training checkpoint or resume input. All S3 outputs
and scratch were retained: 89,531,149 output bytes and 4,081,393,123 scratch bytes
at this snapshot. The S2 runs retain 12,225,809,280 bytes of non-checkpoint evidence;
failed-launch metadata was also kept. No conserved checkpoint or SFT object was deleted.

Both S3 arms returned all 267 TB2 rollouts at native event times 18:22:54 EST and
began TBLite's 200 trials at 18:22:56. Returned rollouts include terminal errors;
panel correctness, validity and treatment-effect verdicts await audited results.
At 18:31:11 EST both roots and GPU children remain RUNNING, 16/16 GPU tasks per arm,
zero Iris failures/preemptions. The original copy job is also RUNNING with zero
failures/preemptions. No restart, resubmission, new allocation or HF operation.
Next native pass around 18:45 EST: TBLite progress and copy completion.

## November 26, 18:44 EST: TBLite returned rollouts pass half

Native progress records control 120/200 returned TBLite rollouts at 18:43:35.558 EST
and treatment 124/200 at 18:43:41.396. Both arms previously returned all 267 TB2
rollouts. These counters include terminal errors; audited valid/successful counts,
error rates and correctness/loop/treatment-effect verdicts remain pending.
Both S3 roots and GPU children are RUNNING at 18:43:39 EST, with 16/16 GPU tasks
per arm and zero Iris failures/preemptions. Partial logs continue to show timeout
and Daytona errors; Ray-deduplicated log lines do not establish per-trial error counts.

The original conservation-copy job remains RUNNING with zero failures/preemptions.
Its completion manifest is absent, and its 43-record status listing matches the
18:25 snapshot. The last summed staging snapshot remains 2.858 TB from 17:46;
permanent promotion remains unverified. Its source stays protected. During this pass
there were no frozen parameter/classification changes, restart, resubmission, further
deletion, new allocation or HF operation. Next native pass around 18:55 EST.
[TBLite progress receipt](analysis/user-overrides-20261126/s3-tblite-progress-1844.json).

## November 26, 18:55 EST: all 59 checkpoint files have native staging records

The original conservation copy advanced from 43 to 59 native verified staging
records. The 18:54 snapshot has 59 unique paths, including 48 policy shards,
totaling 4,292,558,588,036 bytes, matching the source inventory. Each record carries
a SHA-256 and source identity. A fresh source listing at 18:57 independently matches
the 59-record path set exactly; source identities were recorded by native copy and
were not independently refreshed during this pass. The destination listing at 18:54 contains only
two objects / 1,662,097,797 bytes; the completion manifest remains absent at 18:55.
Native fsutil promotes staged files before publishing that manifest. Final promotion
and completion remain unverified, and the source stays protected.

TBLite progress is control 140/200 returned rollouts at 18:53:15.431 EST and treatment
150/200 at 18:53:25.870. Counts include terminal errors; audited valid/successful
counts, error rates and correctness/loop/effect verdicts remain pending. Both S3
roots/GPU children are RUNNING at 18:53:43 EST, with 16/16 GPU tasks per arm and zero
Iris failures/preemptions. The original copy job is also RUNNING with zero failures
or preemptions. During this pass there were no frozen parameter/classification
changes, restart, resubmission, further deletion, new allocation or HF operation.
Next native pass around 19:05 EST: copy completion and TBLite progress.
[Progress and staging receipt](analysis/user-overrides-20261126/s3-progress-and-copy-staging-1855.json).

## November 26, 19:04 EST: TBLite returns complete and held-out rollouts begin

Both arms returned all 200 TBLite rollouts: control at native 18:58:30.253 EST,
treatment at 18:57:27.614. The 256-trial held-out cohort began at 18:58:31.740 and
18:57:29.166 respectively. Latest held-out progress is control 26/256 returned at
19:03:07.861 EST and treatment 39/256 at 19:02:49.729. Counters include terminal
errors; audited valid/successful counts, error rates and correctness/loop/effect
verdicts remain pending. Both S3 roots/GPU children remain RUNNING at 19:03:32 EST,
with 16/16 GPU tasks per arm and zero Iris failures/preemptions.

The original conservation-copy destination advanced to 34 objects totaling
2,857,868,226,661 bytes. Its job remains RUNNING with zero failures/preemptions;
the completion manifest is still absent. This is partial promotion of the previously
verified staging set, not completed permanent publication. The source stays protected.
During this pass there were no parameter/classification changes, restart, resubmission,
further deletion, new allocation or HF operation. Next native pass around 19:15 EST:
held-out progress and copy completion.
[Held-out progress receipt](analysis/user-overrides-20261126/s3-heldout-progress-1904.json).

## November 26, 19:14 EST: both held-out returned-rollout counts pass half

Native held-out progress is control 141/256 returned at 19:13:18.437 EST and
treatment 166/256 at 19:13:41.494. Both arms previously returned all 267 TB2 and
200 TBLite rollouts. Counters include terminal errors; audited valid/successful
counts, error rates and correctness/loop/effect verdicts remain pending. At 19:13:32
EST both S3 roots/GPU children remain RUNNING, with 16/16 GPU tasks per arm and
zero Iris failures/preemptions.

The original conservation copy is RUNNING with zero failures/preemptions.
Destination inventory remains 34 objects / 2,857,868,226,661 bytes, unchanged from
19:03; its completion manifest remains absent. Final publication remains incomplete
and the source stays protected. During this pass there were no parameter/classification
changes, restart, resubmission, further deletion, new allocation or HF operation.
Next native pass around 19:25 EST: held-out completion and copy publication.
[Held-out progress receipt](analysis/user-overrides-20261126/s3-heldout-progress-1914.json).

## November 26, 19:24 EST: both held-out returned-rollout counts exceed 200

Native held-out progress is control 210/256 returned at 19:23:27.190 EST and
treatment 215/256 at 19:22:58.672. Both arms previously returned all 267 TB2 and
200 TBLite rollouts. Counters include terminal errors; audited valid/successful
counts, error rates and correctness/loop/effect verdicts remain pending. At 19:23:32
EST both S3 roots/GPU children remain RUNNING, with 16/16 GPU tasks per arm and
zero Iris failures/preemptions.

The original conservation copy remains RUNNING with zero failures/preemptions.
Destination inventory is still 34 objects / 2,857,868,226,661 bytes, unchanged from
19:03; its completion manifest remains absent. Final publication remains incomplete
and the source stays protected. During this pass there were no parameter/classification
changes, restart, resubmission, further deletion, new allocation or HF operation.
The coordinator's latest user ruling routes loop-guards issue updates to logbooks
#86 only; no further #91 posts or edits. Next native pass around 19:35 EST:
held-out completion and copy publication.
[Held-out progress receipt](analysis/user-overrides-20261126/s3-heldout-progress-1924.json).

## November 26, 19:34 EST: held-out returns reach 90% and copy promotion advances

Native held-out progress is control 231/256 returned at 19:32:37.557 EST and
treatment 236/256 at 19:33:29.204. Both arms previously returned all 267 TB2 and
200 TBLite rollouts. Counters include terminal errors; audited valid/successful
counts, error rates and correctness/loop/effect verdicts remain pending. At 19:33:33
EST both S3 roots/GPU children remain RUNNING, with 16/16 GPU tasks per arm and
zero Iris failures/preemptions.

The original conservation-copy destination advanced from 34 objects / 2.858 TB
at 19:23 to 45 objects / 3,037,106,831,211 bytes at this snapshot. The job remains
RUNNING with zero failures/preemptions, and its completion manifest is absent.
Final publication remains incomplete and the source stays protected. During this
pass there were no parameter/classification changes, restart, resubmission,
further deletion, new allocation or HF operation. Issue updates remain on #86 only.
Next native pass around 19:45 EST: held-out completion and copy publication.
[Progress receipt](analysis/user-overrides-20261126/s3-heldout-and-copy-progress-1934.json).

## November 26, 19:53 EST: checkpoint conservation completes and audit inputs are explicit

The original treatment-step1 conservation job is SUCCEEDED with zero failures or
preemptions, observed at 19:47:43 EST. Its completion manifest records 59 data files
and 4,292,558,588,036 bytes. Every path, size, SHA-256 and source identity matches the
native staging records. The manifest SHA-256 is
`e212783fc5f64836448548bde420402320ad93d764039050f47085595e8fedab`.
The permanent destination contains 60 objects including that manifest. Its parent
conservation marker declares permanent retention and exclusion from every pruning
sweep; byte readback passed. Automatic sweep enforcement remains unverified.
The temporary source was retained during this completion check; no HF operation.

Postprocessing source `8939ee19df` now requires `--manifest` and forwards it to the
existing audit command. The unchanged 95,280-byte frozen manifest is published at a
regional URI; byte readback, canonical/file hashes and all published task IDs/order
match the sealed S0 panel. The live bundle collector includes the ignored local
manifest in this environment. Explicit regional input removes dependence on that
file being present in a fresh submitter checkout. GPU run source remains `cc4e3dcc62`;
scientific parameters, detectors and error classification are unchanged.

Lint, CLI help and all 31 focused loop-guards tests pass. The broad affected-test
runner stopped in Harbor with nine failures and 1,945 passes. All nine failures
independently reproduce on clean `origin/main` commit `8c629544b1` under Python 3.12:
GNU `chown --reference` and tmux lifecycle operations fail on macOS. Later broad
Marin test groups were not run after that stop. The main checkout was not modified.

Latest S3 observations remain the 19:45 pass: both roots/GPU children RUNNING,
16/16 GPU tasks per arm, zero Iris failures/preemptions; held-out returns 248/256
control and 246/256 treatment. Counts include terminal errors; audited panel verdicts
remain pending. No CPU analysis has been submitted: terminal roots/children and
final dumps must be checked first. The prepared analysis excludes baseline-only
SWE100 and uses the sealed S0 audit. Reporting stays on #86 only.
[Readiness and validation receipt](analysis/user-overrides-20261126/s3-postprocessing-readiness-20261126.json),
[permanent checkpoint marker](analysis/user-overrides-20261126/s2-treatment-step1-conservation-complete.json).

## November 26, 20:02 EST: superseded temporary checkpoint source removed

After verified permanent publication and successful copy-job completion, one native
fsutil invocation removed the old treatment-step1 temporary source under the existing
storage directive: 59 objects / 4,292,558,588,036 logical bytes. Deletion completed at
20:01:23 EST. Post-delete listings show only one-byte metadata markers in both S2
temporary checkpoint roots. The permanent checkpoint still contains all 59 data files
plus its completion manifest; manifest byte readback matches. Permanent step-4 exports
and SFT 400407 inventories are unchanged. No conserved checkpoint or SFT object was removed.

Cumulative temporary-checkpoint cleanup is eight prefixes / 472 objects /
34,339,242,608,088 logical bytes. Region-wide free capacity was not measured. All
non-checkpoint report evidence and S3 outputs/scratch were retained. This cleanup
launched no model trials and changed no parameters or error classification. No HF
operation. Reporting remains on #86 only.
[Exact retirement manifest](analysis/user-overrides-20261126/storage-step1-source-retirement-20261126.json).

## November 26, 20:16 EST: both held-out panels await their final returned trial

Both arms have returned 255/256 held-out rollouts, control at 20:14:31.912 EST and
treatment at 20:14:29.359. Returned counts include terminal errors. The rounded
native display says 100%, but each arm still lacks one return. At 20:15:21 EST both
roots/GPU children are RUNNING, with 16/16 GPU tasks per arm and zero Iris failures
or preemptions. No CPU postprocessing was submitted before terminal states/final dumps.

The [existing I3 registration](https://github.com/marin-community/marin/issues/12264#issuecomment-7171525899)
names repeat-turn share against the concurrent RL control as primary. Targets remain
TB2, TBLite and held-out, frozen by sealed S0; they are not reselected from S3 control.
Trial share and its upper bound remain secondary. Correctness gates are paired
against the named control, with the user's 0.75 multiplier and existing repeat-window
policy. Treatment versus S0 is report-only for I3. The raw audits can use S0 as a
reference without turning their S0 comparison into the final paired I3 verdict.
No scientific measure, threshold policy, classification or model setting changed.
Next native check around 20:20 EST, with per-arm CPU submission only when eligible.
[Final-trial wait and registration reference](analysis/user-overrides-20261126/s3-final-trial-wait-2016.json).

## November 26, 20:30 EST: S3 completed; both CPU postprocessors submitted

Both S3 coordinator roots and their 16-task GPU children succeeded with zero Iris
failures or preemptions. Each arm returned all 723 frozen rollouts. The final held-out
returns were at 20:26:12.848 EST for treatment and 20:26:41.713 for control; final
cohort dumps were written at 20:26:15.049 and 20:26:43.727, respectively. Native
storage listings independently confirm both final per-trial dumps. Returned counts
include terminal errors; valid counts and correctness/loop verdicts await the audits.

The checked-in CPU postprocessing module on source `8939ee19df` was submitted once
per arm after terminal-job confirmation: treatment at 20:28:48 EST and control at
20:29:43. Each job requests 16 CPUs, 64 GB memory, 128 GB disk and zero GPUs at
production priority in 08a. Both jobs are registered with the coordinator and the
shared watchdog. They audit saved outputs against the explicit frozen manifest,
exclude baseline-only SWE100, conserve closed traces through same-region verified
copies and refresh saved reports. They create no model trials.

The final I3 comparison remains pending. Its primary measure is repeat-turn share
against concurrent RL control on the three S0-frozen targets. The user's correctness
multiplier and registered repeat policy remain binding. S0 comparisons in the raw
audits are report-only for I3. W&B emitted finishing/upload warnings; native dumps
and terminal Iris states were verified independently. No repeat or retuning was
submitted. Reporting remains on #86 only.
[Terminal proofs and exact CPU submissions](analysis/user-overrides-20261126/s3-terminal-cpu-submissions-2030.json).

## November 26, 20:38 EST: I3 fails the registered loop-reduction gate; no repeat

At the fixed four training updates, antithrash did not halve repeat-turn share
against the concurrent RL control on any frozen target. The measured share was
higher in treatment on every cohort. This single registered pair does not establish
a statistically significant causal increase.

| Cohort | Control repeat-turn share | Treatment share | Relative increase | Primary gate |
|---|---:|---:|---:|---|
| TB2 | 0.1501067499 | 0.2138755238 | 42.48% | Fail |
| TBLite | 0.0765120233 | 0.0814704421 | 6.48% | Fail |
| Held-out | 0.1321535607 | 0.2021508130 | 52.97% | Fail |

Correctness passes both original and overridden gates. Control passes are
52/75/109, pooled236; treatment59/70/106, pooled235. The concurrent-control raw
floors38.65/65/96.2 and pooled221.54 become28.9875/48.75/72.15 and166.155 under the
user's0.75 multiplier (integer minima29/49/73 and167). Every target is more than
two trials from its overridden unrounded correctness floor, so no repeat trigger
fires. The primary share is far above its halving threshold in every cohort;
secondary action-loop UB cannot replace it. No repeat or retuning was submitted.

Control has700/723 assessed rollouts and treatment693/723, with zero unavailable
outputs. Cohort assessed counts are253/267,197/200,250/256 for control and
251/267,196/200,246/256 for treatment. Every cohort passes the90% validity gate.
Action-loop UB counts are39/18/24 for control and53/19/29 for treatment, reported
as secondary outcomes. Frozen cohort hashes match both arms and sealed S0.
S0 comparisons remain report-only for I3.

Both CPU postprocessors succeeded with zero Iris failures/preemptions. Their
completion receipts cover the audit, permanent same-region verified trace copy and
saved-report refresh. Each refresh has42 evaluations and no reporting error.
Permanent trace manifests passed byte-hash readback:724files/24,143,936bytes control,
724files/24,370,922bytes treatment. No model trials were created by postprocessing.

The checked-in `i3-verdict` command on source `7191758004` reproduces the comparison
from the three regional audit JSON files. Required lint and all33 focused tests
pass. The broad affected-test rerun stopped in Harbor:9failed/1945passed/3skipped/
2deselected. These same nine failures were independently reproduced on clean main
(GNU chown --reference and tmux lifecycle on macOS); later broad Marin groups did
not run. The S2 per-update
antithrash fire-rate and mean-penalty evidence remains to consolidate for the final
report; no underpowered classification is claimed yet. The I3 result follows its
registered gate regardless of that diagnostic. Overall, both inference guards and
the fixed-N4 RL intervention fail their registered loop-reduction measures.

[Paired decision receipt](analysis/user-overrides-20261126/s3-i3-paired-verdict-20261126.json),
[CPU completion and conservation receipt](analysis/user-overrides-20261126/s3-i3-completion-receipt-2038.json).
The regional decision is `s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/decisions/i3-paired-verdict-20261126.json`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-loop-guards%2Fdecisions%2Fi3-paired-verdict-20261126.json));
SHA256`f3b12ae26a8656b2cda5efbf3450d9fa574f572692745f391a1c81efa7e13f9f`.
Regional readback matches the local calculation and all three audit input hashes.
