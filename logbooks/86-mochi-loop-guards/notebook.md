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
- Native entrypoint: `uv run python -m experiments.post_training.mochi_loop_guards --preset s0-baseline --version 2026.11.25.7 --run --wait` with regional storage credentials and EVAL provider credentials supplied out of band. Inspect `__main__.py` for environment binding and `--plan` for the resolved artifact graph. Never print credentials.
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

## Final decision

Pending measurement. No intervention has been selected. Remaining uncertainties are baseline loop incidence, per-cohort validity, stall precision, and whether enough time remains for paired RL. No Marin contribution PR has been opened.
