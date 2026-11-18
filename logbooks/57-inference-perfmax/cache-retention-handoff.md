# Retention-aware cache checks for later extraction

Source: contribution `4f38ff6b3d4403d84a9248cf63506c53e0e00bd9`.
Its research equivalent is `33a40e6a403d92b3dcc9946a94b1031cc5cb6825`.
Apply the three-file change to `endpoint_checks.py`, its HTTP tests and the
shared-check README. Extraction is planned; DA5 owns this correction.

At pinned vLLM `0ac9dcb157ee86ccb447bc717259159c41ad456a`,
[`get_replay_boundaries`](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/vllm/v1/core/kv_cache_coordinator.py#L314-L359)
uses the original prompt length. Retention zero therefore does not promise an
arbitrary checkpoint created during decoding. Positive retention adds periodic
checkpoints; dense retention uses aligned checkpoints. The last sampled token
has not undergone a forward pass.

`check_cache` now receives the served cache configuration and maximum context.
For these TP1 fixtures, alignment is the LCM of attention and Mamba block sizes.
A positive/dense probe chooses a reachable periodic endpoint. If the original
257 outputs cannot reach one, it moves the seed toward that endpoint and
generates enough tokens within the served context limit. Zero records initial
recomputation without treating it as corruption or missing coverage.

Every mode then processes generated history as a prompt and requires a replay
to reuse it beyond the seed. Both ordinary branches and these replayed histories
must retain exact scores after a sibling. The sibling must itself show a
positive hit shorter than the protected original prefix. This checks state
persistence; it cannot detect every initially wrong state. Fresh/reused scores
retain the separate changed-prefill-step limits.

Reports retain `retention_interval`, `cache_alignment`, direct `decode_reuse`,
mandatory `replayed_decode_reuse`, actual seed/generation lengths and sibling
hits. `not_required` applies only to direct periodic reuse when no such endpoint
is reachable in that mode/context. It cannot waive prompt replay, branch reuse,
non-finite scores or numerical limits. No new capability schema is needed.

The [56 combined CPU/HTTP/report cases](analysis/hero-fp8-cache-validation/cpu-tests.txt)
pass with the child's reporting fix. They cover zero, dense and positive
retention, including interval 512 beyond the old fixed decode span. The real
CLI/report path accepts healthy zero-retention replay and blocks missing hits
or corrupt cached state. Required contribution lint/types pass. Live Snowball
and Hero validation remains pending; earlier BF16 evidence is preserved.
