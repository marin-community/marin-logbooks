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
