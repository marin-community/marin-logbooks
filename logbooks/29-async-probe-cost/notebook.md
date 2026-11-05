---
issue: https://github.com/marin-community/marin-logbooks/issues/29
source_repository: marin-community/marin
source_revision: 6330720f1fdfa50c5c7be05a9592d416e5fffc57
archive_template_revision: f54b3aa0e6ee533e988d2b97e9db0d1523f934dd
status: active
---
# Periodic probe cost during asynchronous Snowball training

The decision is a practical diagnostic cadence near a 1% average cost target. SC stays off, K128 stays fixed, and product defaults stay unchanged. The previous serialized interval-one study is [#13](https://github.com/marin-community/marin-logbooks/issues/13). The completed normal-training study [#17](https://github.com/marin-community/marin-logbooks/issues/17) had probes off.

## 2026-11-05: Reuse and qualification

The driver reuses #17's model, frozen training data, 40-H100 role geometry, AdamW settings and rolling staleness-two loop. The observer reuses its complete-cycle boundary: just before initial rollout publication through return from the last training step, after weight publication. It records ordinary timings without adding overlapping stage times. Capture, transport, grading, scoring and optimization remain inside the normal overlapped loop.

Evaluation, native checkpoints and HF exports are disabled for this cost experiment. They do not answer the cadence question. The current maintained Snowball recipe also includes its reasoning-span reward shaping; both arms use the same settings. The pinned vLLM wheel and SkyRL compact chat adapter remain unchanged.

Before Grug allocation, qualify a twelve-update Qwen startup with interval ten, nonzero optimizer updates, real published weights, normal rolling overlap and consumed probes. Freeze the finite main comparison after that startup and before examining any Grug cost result. This is experiment source; none of these files will enter the product PRs.

The real SkyRL composer accepts the twelve-update Qwen recipe and all nine Grug seed/interval configurations. [The compact qualification](preflight.json) records the regular PPO/TIS objective, nonzero learning rate, rolling staleness-two buffer and node counts. Existing product CPU suites pass 369 core and 387 probe cases after the TIM split, including exact optimizer isolation. The research observer adds a runtime guard against passing probe tensors to optimization.

Fresh quota/usage evidence is from the existing collector's [dry-run execution](https://github.com/marin-community/marin/actions/runs/41973862318). No quota or shared service changed. Both H100 pools were inspected; RNO keeps the model/data local and has capacity for the selected topology. The resource board records the one-node interactive qualification and subsequent five-node batch comparisons.

## Qualification attempt one: entrypoint repair

Source `56d27f77e3ada7276a148f37be1aa2f3b164a2e7` launched Qwen as `/romain/users-romain-checkpoints-async-probe-cost-qwen-i10-seed1701-2026.11.05.1-coordinator-6f4a79f2`, with child `users-romain-checkpoints-async-probe-cost-qwen-i10-seed1701-2026.11.05.1-3900f8a6bf11`. The child and CPU coordinator are authoritatively failed. The eight-GPU child ran 2 minutes 24.44 seconds before the research-only entrypoint hook raised `NameError: os is not defined`. No model construction or optimizer step occurred. Its allocation, startup and logs remain in the final accounting.

The hook now imports `os`. A direct static check covers the vendored hook as well as the experiment files; the ordinary root lint excludes vendored source. The observer additionally records selected-position counts per sampled response. A fresh artifact version is required for the retry. [The finite protocol](protocol.json) declares three matched thirty-update pairs, alternating arm order, paired log-ratio intervals and the single conditional interval-twenty comparison. No Grug comparison has been allocated or observed.


## Qualification attempt two: use the established colocated smoke settings

Source `4c6da1cc60fd6cbce6904c80b18f1f3c07fda2f0` reached the real observer entrypoint, then the ordinary runtime validator rejected the startup's rolling staleness-two configuration: colocated engines require zero staleness. The child `/romain/users-romain-checkpoints-async-probe-cost-qwen-i10-seed1701-2026.11.05.2-coordinator-4e531472/users-romain-checkpoints-async-probe-cost-qwen-i10-seed1701-2026.11.05.2-e34298966228` is authoritatively failed after 1 minute 50.35 seconds. No model construction or optimizer step occurred.

Restore the existing #17 Qwen smoke's full-batch, staleness-zero settings. This qualifies software, consumed probes and published nonzero updates on one eight-H100 node. The three main Grug pairs still require normal disaggregated sampler/trainer overlap and rolling staleness two. Add the actual runtime validator to the local qualification after the real recipe composer; no product validator or default changes. Both failed startups remain in allocation accounting.

Current cluster defaults put canonical and temporary outputs in US-EAST-08A, while model and frozen data inputs remain in RNO. The protocol records both output prefixes. The first attempt's saved launch envelope was read back and its inner launch configuration exercised through the actual worker entrypoint to the GPU-driver boundary.


## Qualification attempt three: preserve the frozen TaskSpec contract

Source `282f61dbe52396d0a346729b0ed459229ffa3b60` reached the real dataset loader and failed before model construction because the frozen tasks use schema 0.28 while current main accepts 0.30. Its child `users-romain-checkpoints-async-probe-cost-qwen-i10-seed1701-2026.11.05.3-5d36959326fb` and coordinator `users-romain-checkpoints-async-probe-cost-qwen-i10-seed1701-2026.11.05.3-coordinator-264916ab` are terminal failed. The child ran 2 minutes 19.3 seconds; retain its allocation and durable original exception. No Grug result has been observed.

Pin the experimental source to the completed #17 runtime at `4885c973e753a4c68012ef12a68f36205d0056d5`. This keeps the exact original model, tasks, graders and recipe, including TaskSpec0.28. [Implementation hashes](runtime-equivalence.json) show identical probe math, score-centering math and compact chat code relative to the current product candidate. The benchmark does not measure unrelated newer main changes. Product worktrees retain the current main integration.

The research branch preserves every published startup commit and merges the original study history. Relative to the qualified runtime, its executable delta is the cost launcher, observer and two-line entrypoint selection. The launcher now directly reuses `study_recipe` instead of restating its geometry and optimizer settings. No task-version guard is weakened and no grading contract is migrated.

Local checks validate all 104 actual Qwen training task contracts and run them through the real SourceTaskDataset with the pinned tokenizer: all 104 survive the prompt budget. The main training-file hash is rechecked against the frozen study. Reuse the earlier unchanged Grug dataset/GPU qualifications. The observer uses the existing performance-info API before measurement to verify the pinned wheel, V2 runner and absence of custom logits processors.

Amend the software-qualification limit to one additional Qwen startup on this fully matched runtime (four attempts total), before any main comparison outcome. This leaves the three paired Grug windows, conditional extension, metrics and uncertainty rule unchanged. GPU allocation from all failed attempts remains included. There is no additional learning or evaluation campaign.

## Qualification attempt four: real updates on the matched runtime

Source `64f29117f91eaf1c967487f2c50bdcc722c46ebd` launched version `2026.11.05.4`. The coordinator is `/romain/users-romain-checkpoints-async-probe-cost-qwen-i10-seed1701-2026.11.05.4-coordinator-34db9b0c`; its eight-H100 child ends in `users-romain-checkpoints-async-probe-cost-qwen-i10-seed1701-2026.11.05.4-57b402aeb304`. It passed model construction and produced its first two real updates. Step two has positive learning rate `9.999999974752427e-7` and raw gradient norm `0.5366743206977844`. All eight engines report the pinned wheel, V2 runner and no custom logits processors. The twelve-update completion and consumed-probe gate remain pending; no Grug job has been submitted.

Corrections to the opening plan: Qwen uses the established synchronous full-batch colocated qualification, not rolling overlap. Grug alone supplies the required normal asynchronous comparison. Both Grug arms use the original #17 reward recipe, without the newer main reasoning-span penalty. These choices preserve the frozen tasks and graders. The protocol's explicit runtime scope supersedes the initial current-main assumptions.

The paired-interval analysis matches #17's existing implementation on three synthetic ratios to numerical precision; [the method check](analysis-method-check.json) is not experimental data. Admission also checks real nonzero finite gradients, positive learning rates and completed weight publication for every update.

## Qualification admitted; begin the fixed main comparison

The fourth Qwen coordinator and GPU child are terminal succeeded. [The qualification proof](qwen-qualification.json) admits all twelve nonzero finite-gradient updates at positive learning rate, thirteen distinct restored serving-weight hashes, and two consumed probe events at training steps 1 and 11. They cover 179 actual positions, at most 32 per response. The optimizer guard stayed clean. Entrypoint-to-first-publication startup was 295.38 seconds; the twelve-cycle wall span was 511.94 seconds. This small qualifier has no long-response probe coverage and supplies no Grug cost estimate.

Proceed with the protocol's first main window: Grug, seed 1701, probes disabled, thirty nonzero updates, original disaggregated rolling recipe, five batch nodes. The other five windows retain their frozen serial order. Both H100 pools were refreshed before expansion; keep all comparison arms on RNO H100 NVLink80GB hardware. The earlier failed startups, the complete successful allocation and teardown will all enter compute accounting.

## Qualification accounting and recovery

All four Qwen allocations reconstruct to 3.111 H100-hours, including 0.951 from the failed attempts. [The allocation proof](qwen-allocation-proof.json) includes startup and teardown and identifies the first attempt's missing pod clock and conservative controller proxy. The unchanged #17 combiner independently checks the timestamp arithmetic and rejects duplicate attempts in [the combined proof](allocation-through-qualification.json). These are reconstructed allocation spans, not provider billing.

All four temporary attempt trees are now preserved outside TTL storage: 3,318 objects and 42,947,514 bytes. [Recovery references](preservation.json) pin each verified manifest by SHA256; those manifests retain every object's path, identity, size and SHA256. Original temporary duplicates remain until final cleanup. The compact clock proof also names a durable recovery prefix for its raw retained inputs.

The runtime comparison additionally verifies identical response-head extraction and identical trainer publication, complete training-step and probe-summary method ASTs. Broader runtime differences stay outside the benchmark's claim; the explicit pin is still required for the frozen task contracts.

## First Grug updates: overlap and representative lengths

The first off window has completed two nonzero finite-gradient updates. [The initial runtime proof](grug-initial-qualification.json) records step-two token age mean 0.9858 and maximum 1, confirming normal sampler/trainer overlap. The two batches contain 148 responses of at least 2,048 tokens, with maximum length 5,673. All eight engines pass the pinned runtime check. Entrypoint-to-first-rollout-publication startup took 585.79 seconds. These observations qualify the real asynchronous workload; they are not a completed thirty-update window or an overhead estimate.

## First full baseline admitted; execute the remaining fixed sequence

The first Grug off window and its coordinator succeeded and stopped. [The complete baseline](grug-i0-seed1701-v1-summary.json) has all thirty finite nonzero updates and weight publications, 35,714,166 response tokens, 5,182 responses of at least 2,048 tokens, and maximum token age two. Its complete wall span is 4,772.05 seconds, or 159.07 seconds per update. No probe event was consumed in this disabled arm. The paired cost estimate is still pending.

The session resumed at 19:30 UTC after this baseline had finished at approximately 08:23 UTC. No GPU remained allocated in this gap. Record the roughly eleven-hour scheduling separation in the first pair's limitations. Preserve the fixed six-window plan and its admitted baseline.

The remaining five windows use the existing ArtifactStep executor with `--max-concurrent 1`, in the original order: on1701, on2903, off2903, off4517, on4517. [The small series entrypoint](../../experiments/post_training/async_probe_series.py) delegates to the unchanged single-window launcher; its dry-run graph contains those exact five Grug handles. Its serial guard prevents an accidental concurrent launch. No measured runtime file, model, data, seed, optimizer setting, cadence, window length or repetition changes. This lets the bounded sequence continue across session gaps. The conditional interval-twenty decision still waits for the completed three-pair analysis.
