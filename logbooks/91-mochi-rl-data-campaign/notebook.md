---
title: "Mochi 50-step RL data-mix campaign"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/91"
related_issues: ["https://github.com/marin-community/marin/pull/12329"]
research_branch: "codex/91-mochi-rl-data-campaign"
marin_base_revision: "46e0c1b502bb0467b98bae922552ece67d6d984e"
archive_template_revision: "3846d697465e8629294936c6a29e2061bb1c296d"
status: active
---

# Mochi 50-step RL data-mix campaign

## Question and decision criteria

- **Goal and why it matters:** find which RL data sources and mixtures improve Mochi most per RL step. The answer chooses the RL data mix for the next long-loop runs.
- **Baseline and alternatives:**
  - The baseline is the config and data of Mochi RL run `.131` (`long-loop-mochi-exp4-dspark-rl-2026.11.25.131`); its Micro210 seal comes from the Mochi lane.
  - The alternatives are 100 candidate mixes of 4–8 sources at equal weight, from the holiday Atlas burndown (marin #12329): v2 and v3, 50 each.
- **Evidence needed to choose:**
  - Micro210 score of each run's best-by-validation checkpoint (canonical policy, #10197 scoring).
  - Train reward over steps 1–10 vs 41–50.
  - Best validation step.
  - Signs of broken data, such as all-zero or all-one rewards, grader errors or infrastructure-absent rollouts.
- **Constraints:**
  - The RL config is fixed. Only the curriculum changes.
  - 50 optimizer steps per run, with no time cap.
  - 3 runs going at once, at production priority on the cw-us-east-08a GB200s, with 128 GPUs each (48 trainer + 80 rollout).
  - Zone free space on the cw-us-east-08a object storage stays above 15%. Below that, campaign checkpoints are pruned, always keeping each run's best and final checkpoints.
- **Rules for choosing mixes:**
  - Each mix has at least 2 SWE/agentic families and at least 2 competition-math families.
  - Each source has at least 500 tasks.
  - Sources saturated at the start of RL are left out: nemotron_if (0.97) and openscience (0.92).
  - The queue follows human ratings: High, then Medium, then Low; Skip never runs.
  - Mixes with a source whose stored grader predates fixes on main (#12329 review) run after all clean mixes.
  - Mixes run as listed, including sources Pending in the served R40 catalog, at their declared pins. Each run records which of its sources diverge from the catalog.

## Baseline

- Date: 2026-11-25
- Run: Mochi RL `.131` (Exp4 DSpark); step-34 best-by-val export; Micro210 eval child succeeded 2026-11-26. The sealed score will be added here when the Mochi lane publishes it.
- Config: DSpark (4 draft tokens), TP1/BF16 engines with CUDA graphs, 48 trainer + 80 rollout GB200, the .131 initializer, drain and rolling defaults.

## Entries

### 2026-11-26 13:30 UTC - Campaign start and launcher validation

- Question or hypothesis: set up a fixed-config, data-only RL ablation that is cheap enough to run about 50 mixes in a week.
- Published source revision: the campaign launcher landed at `a97c1a1198`. Setup fixes followed at `219b720287` (inherit the parent cluster for nested CPU stages) and `390c5e318b` (verify task_sha256 against the exact published JSON bytes, not canonicalized JSON).
- Configuration: the .131 recipe with a `--mix-id` argument that resolves sources through the Atlas catalog, `--max-steps 50` and no learning-clock callback.
- External inputs:
  - candidate mixes `mix_candidates_v2.csv` and `mix_candidates_v3.csv`, plus the source table `rl_sources_on_main.csv` (holiday burndown, #12329);
  - the saturation evidence table from the .129 and .131 per-route rewards.
- Result:
  - The launcher validation passed: native recipe and graph dry runs, plus 58 landing tests.
  - Six first-wave roots (.502–.507) failed before any GPU allocation on the two setup defects above. No training happened, and they are not counted as runs.
  - The first wave is relaunching with fresh versions.
- Evidence: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-rl-data-campaign/preparation/2026.11.26.1/launcher-validation-final.json`.
- Interpretation: none yet.
- Next decision: when the first three runs reach 50 steps, compare their reward curves and Micro210 against `.131`.

### 2026-11-26 13:16 UTC - Campaign update

Wave 1 setup: the as-listed ruling (user, 11/26) reached the campaign owner after it had launched two Available-only mixes under the earlier interim rule: v2-mix-032 (.511) and v2-mix-006 (.512), both clean queue entries. They keep running. v2-mix-033 and v2-mix-009 run next under the as-listed ruling. v2-mix-011's RST source failed its integrity contract and runs only if that can be fixed.

### 2026-11-26 13:35 UTC - Campaign update

Wave 1 setup, third defect: roots .511 (v2-mix-032), .512 (v2-mix-006) and .513 (v2-mix-008) built their training pools (76,083 and 23,114 tasks for the first two), then failed the real-agent CPU gate before any GPU allocation. NL2Bash's local verifier asks for disk/GPU allocations it cannot get, and e2egit validation has no selected machine. This is a runtime-vs-validator precedence bug in the campaign tooling. It is being fixed at the root, with no requirements stripped and no RL config change. Relaunch order: v2-mix-033 and v2-mix-009 under the as-listed ruling, then the next runnable mixes.

### 2026-11-26 14:32 UTC - Campaign update

Wave 1 launched under the as-listed ruling: v2-mix-033 (.514), v2-mix-009 (.515) and v2-mix-017 (.516), all on landed b0457a312f, all in data prep and the agent gate. The catalog divergence is recorded per run. Separate finding (R41 survey evidence): 25 Atlas sources, 22 of them Available, declare a host-image grader with a disk quota that no production backend can host. Local cannot enforce the quota and image backends are not candidates. A host-image to container-image conversion is being built so they can run faithfully. Mixes that contain those sources may fail the agent gate until it lands.

### 2026-11-26 14:45 UTC - Campaign update

Wave 1 relaunched on mixes with no known grader blocker: v2-mix-006 (.517), v2-mix-015 (.518) and v2-mix-043 (.519), on landed 96a2d65926, all in data prep. Mixes containing a disk-quota (HostImage + storage_mb) grader, the unsupported RST contract, or a missing binding wait for the disk-root fix: v2-mix-033, 009, 011, 017 and others. Grafana links are added when each run's RL child is placed.

### 2026-11-26 15:11 UTC - Campaign update

The user rated the v3 mixes. v2 and v3 now form one queue of 78 runnable mixes (21 v3 Skips dropped): clean mixes first, High (2) > Medium (15) > Low (15), then mixes with a stale-grader source (High 7 / Medium 17 / Low 22). Runs .520 (v2-mix-006), .521 (v2-mix-015) and .522 (v2-mix-043) were resubmitted after the secret-resolution fix 2e8d58e119.

### 2026-11-26 18:13 UTC - Campaign update

The run table moved to the body of issue #91, which is now the single source of truth for run state. The tracker artifact mirrors it, and runs.csv is retired.

### 2026-11-26 19:51 UTC - Campaign update

Reset: trainer layout PP12 to PP16 (64 trainer + 80 rollout = 144 GB200) after OOMs in .527 and .528. Every earlier run retired and its checkpoints deleted; the queue restarts from the first 3 runnable mixes. Another OOM halts the queue for review.

### 2026-11-27 16:28 UTC — Final-checkpoint protocol and storage readiness

The user authorized a protocol change at 11:15–11:16 EST: **validation dropped, final checkpoint is the deliverable, user-authorized 2026-11-27**. New launches will have no holdout validation, validation evaluations, or best-by-val selection. They must save every five steps, keep at most two rolling checkpoints (latest and one previous resume fallback), and always save the final step. Export and canonical Micro210 use the final checkpoint. Micro210 scoring and coverage requirements do not change.

This takes effect at the first new campaign root after implementation, tests and loop-push. No root under this protocol has been submitted yet; its actual identity will be recorded at submission. This is a comparability break alongside the earlier repeat-stall-stop cutover (`f163b89b94`, qualified at `ef09a1e250`). It supersedes the original best-by-validation decision criterion above. The PP16 reset already changed each run to 144 GB200 (64 trainer + 80 rollout).

For finished runs .532 (v3-mix-048) and .534 (v2-mix-043), use final step 50; both final exports already succeeded. For running .540 (v3-mix-036), leave its config unchanged, let it finish, and export/evaluate its final checkpoint, ignoring its best-by-val selection. No new export or evaluation was submitted for this record.

The immediate storage audit removed **zero objects and freed 0 TiB**. At 2026-11-27 16:28 UTC, both finished runs' retain_campaign_checkpoints stages were still running, with 15 final-native copy objects staged each and no completed copy manifest or retention artifact. Their source checkpoints remain protected until retention succeeds. The obsolete .532 step-18 checkpoint is 4,292,491,064,101 bytes (3.904 TiB), deferred until that condition is met. .534 has only final step 50. Running .540 has only best step 3 and latest step 45, so the immediate rule to remove checkpoints older than its protected best and latest selects no objects. Final checkpoints, pending evaluation inputs, .131, the SFT conservation copy and other owners' objects were untouched.

The deletion ledger is unchanged (SHA256 `730ccc1ddb08184841b2192901c84b34180a99109458e1e7f1089facb14aed91`). Exact key/size inventories and the no-deletion receipt will support reconciliation if the maintained retention stage removes temporary checkpoints itself. Evidence: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-rl-data-campaign/storage/2026.11.27T1628/operator/` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-rl-data-campaign%2Fstorage%2F2026.11.27T1628%2Foperator%2F)).

Next: implement and qualify the new protocol and fix the .711 LOCAL disk lease race before another GPU launch. The reader fix is landed at `cb0907513c16b86ff498a7e74a12c8e2a11a3bac`; the lease-race diagnosis remains a hypothesis pending a failing regression and privileged Linux proof.

### 2026-11-27 17:12 UTC — Final-only protocol and guardian cleanup landed

The no-validation policy and two-checkpoint retention landed through loop-push at [`6c915ab36e`](https://github.com/marin-community/marin/commit/6c915ab36edfa1da764471acf9c1d7eed645d426). Campaign pools now keep every source row in training, publish no validation parquet, and record a zero-size holdout. The native launch has no validation inputs, evaluation callback or best-checkpoint selection. It exports the latest final checkpoint, saves every five steps, keeps two rolling states and saves on train end even off cadence. The pool fingerprint includes the protocol era so older holdout pools cannot be adopted silently.

The companion guardian fix `e14de3760b` accepts FileNotFoundError only when concurrent cleanup has already removed the allocation directory. Other cleanup failures still propagate. A deterministic two-cleaner reproduction fails on the clean previous source `cb0907513c`; current tests also check that PermissionError and a missing child in a surviving directory fail closed. This follows the user's 11:51 EST direction to fix and launch, without another regional guardian qualification campaign. The fix is CPU-regression-tested; the first fresh production launch will test it operationally.

Checks: 37 final focused tests and 37 post-rebase landing tests passed; 50 related tests passed before the final targeted edits. Lint and types passed. All six advisory lanes and composer completed, with 36 inherited findings and none on the change. The required affected runner had 1,989 passes, 12 failures and three skips in Harbor, then stopped before the root suite. All 12 failures reproduced independently on clean parent `cb0907513c`; this is not a broad pass.

Two standard regional CPU pool builds were submitted once, at production on 08a with zero GPUs:

- `/benfeuer/mochi-campaign-pool-proof-v2-mix-009-2026-11-27-712`, accepted 17:10:23 UTC.
- `/benfeuer/mochi-campaign-pool-proof-v3-mix-014-2026-11-27-713`, accepted 17:10:57 UTC.

These fresh data artifacts restore the former 200 holdout rows to training at unchanged source pins. They are the existing campaign pool dependency, not extra guardian qualification jobs. Actual completion, fresh native validation, normal real-agent admission and protected snapshot checks remain pending before RL submission. No new RL/GPU root exists yet, so the effective first run under the new protocol is still pending. Record its actual identity at submission.

Existing .540 is unchanged. Its final checkpoint, and final step50 for .532/.534, remain the authorized deliverables. The two final50 retention copies showed payload growth over 12m53s: +200,258,012,043 bytes (.532) and +266,996,354,619 bytes (.534), with no completed manifests at 2026-11-27 16:41 UTC. Byte growth proves copying activity, not verified retention completion. No cancellation or deletion was needed. Evidence is under `s3://marin-us-east-08a/marin/users/benfeuer/mochi-rl-data-campaign/protocol/2026.11.27.1/` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-rl-data-campaign%2Fprotocol%2F2026.11.27.1%2F)).

### 2026-11-27 17:27 UTC — Two final-only RL coordinators submitted

The first submitted coordinator configured for the user-authorized no-validation/final-checkpoint protocol is `/benfeuer/mochi-campaign-v2-mix-009-2026.11.27.712` (submission receipt 17:23:27 UTC). The second is `/benfeuer/mochi-campaign-v3-mix-014-2026.11.27.713` (17:26:00 UTC). Both use landed source `6c915ab36edfa1da764471acf9c1d7eed645d426`, production priority on 08a, 144 planned GB200 GPUs per run (64 trainer + 80 rollout), 50 steps, and repeat-stall-stop enabled. This records the effective protocol boundary by submitted run identity; GPU admission and training have not been established.

Each new pool root and its prepare_pools child succeeded with exit0. Full-row proofs and serial candidate native validations passed: mix-009 has 83,495 training rows; v3-mix-014 has 111,790. Both have zero holdout rows and preserve the declared source pins. These are fresh full-population artifacts, not silent reuse of older holdout pools. The regional real-agent, credential and vocabulary gate still precedes each GPU child. GPU child IDs, driver execution IDs, initial sync and optimizer metrics remain unavailable at this milestone.

Fresh conservative reference registries protected every currently existing provider snapshot plus current sandbox and verified pool references. The maintained purge check completed for each submission with zero deletions; its account quota readback was zero free RL snapshot slots and 36 free EVAL slots. No quota headroom was inferred from deletion counts or reclaimed. These checks do not establish that a future source requiring a missing snapshot can be admitted.

A fresh final50 copy inventory at 17:17:47 UTC showed continued progress over 36m32s: .532 staged payload rose by 603,485,743,275 bytes to 1,628,964,581,486 bytes; .534 rose by 536,747,400,699 bytes to 1,634,154,361,710 bytes. Both had 27 staged objects, only step50 groups, and no completed copy manifests. This proves movement during that interval, not verified preservation completion. No job cancellation or checkpoint deletion occurred.

The root IDs are reported in the issue table and STATUS for coordinator/watchdog registration; watchdog acceptance is not yet confirmed. Existing .540 and other owners' data are unchanged. No local timer or poller was installed. Evidence: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-rl-data-campaign/protocol/2026.11.27.2/` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-rl-data-campaign%2Fprotocol%2F2026.11.27.2%2F)).

## Run table

The run table lives in one place, the body of [marin-logbooks #91](https://github.com/marin-community/marin-logbooks/issues/91). It is updated there with `campaign-runs set` and is not copied here.
