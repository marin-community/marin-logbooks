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
