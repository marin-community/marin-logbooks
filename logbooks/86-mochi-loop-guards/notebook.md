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
- Result: collecting; no baseline sealed and no intervention effect claimed.
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

## Final decision

Baseline incidence and all four cohort validities are sealed. Offline stall precision is29/30 and passes its registered gate. No intervention has been selected. The original stop arm passes TB2 but fails TBLite’s50% target; the original RP arm fails TB2’s action-loop UB target. Remaining cohorts and any required single repeat remain outstanding. I3 was dropped under the original13:01:36 UTC November26 budget cutoff; no S2, exports or S3 are run. No Marin contribution PR has been opened.
