# Retention-aware cache checks for later extraction

Source: contribution `00f53931169858d0228a9415e2103b4aa12ee86b`.
Apply `4f38ff6b3d4403d84a9248cf63506c53e0e00bd9`, then this follow-up, or
take the final three files: `endpoint_checks.py`, its HTTP tests and the
shared-check README. The first research equivalent is `33a40e6a403d`.
Extraction is planned; DA5 owns this correction. No LA1 acknowledgement is needed.

Contribution `20f15e8dae9cd8c129cd5d4e757c804262a55609` later moves concrete model
bindings to `experiments/evaluation/inference_correctness/fixtures/` and makes
the generic reader accept an explicit JSON path. Both fixture files are
[byte-identical](analysis/hero-fp8-cache-validation/fixture-move-identity.json)
to the originals. Use that revision's README and preset paths when extracting
onto the updated contribution. Historical trial sources retain their old paths.

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

Retention preserves existing snapshots; it does not create them. The pinned
[`align` setting](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/vllm/config/cache.py#L178-L185)
creates snapshots at aligned scheduler ends. A cold aligned prefill can finish
beyond the lower replay point kept at zero. Each probe now explicitly prepares
checkpoint K with K+1 tokens and requires reuse reaching K. Strict branch
controls also prepare the full usable checkpoint E before their baseline.

Every mode processes generated history as a prompt. Its sibling forks at
K=E−S, where S is alignment and P<K<E; P is the original prompt length. It must
hit K, while the original branch hits E before and after. Ordinary branches use
the same explicit preparation. Both must retain exact scores. This checks state
persistence; it cannot detect every initially wrong state. Fresh/reused scores
retain the separate changed-prefill-step limits.

Reports retain `retention_interval`, `cache_alignment`, direct `decode_reuse`,
mandatory `replayed_decode_reuse`, actual seed/generation lengths and sibling
hits. `not_required` applies only to direct periodic reuse when no such endpoint
is reachable in that mode/context. It cannot waive prompt replay, branch reuse,
non-finite scores or numerical limits. No new capability schema is needed.

The first [live Hero zero run](analysis/hero-fp8-cache-validation/hero-zero-a1-summary.json)
exposed incomplete probe preparation: three aligned cold prefills and three
replayed sibling cases missed checkpoints. All 16 recorded strict numerical
comparisons were zero; reference, batch and decode passed. This is a coverage
failure, not evidence of corrupted state. Its complete cost was 5.113333 GB200-hours;
all four pods were confirmed absent. Its source and raw archive remain fixed.

The corrected 57 combined CPU/HTTP/report cases pass. The HTTP engine now models
the cold aligned materialization limitation and rejects a sibling that reuses
only the original prompt. Coverage includes zero, dense and positive retention,
including interval 512 beyond the old fixed decode span. The real CLI/report
path accepts supported zero-retention replay and blocks missing hits or corrupt
state. Required contribution lint/types pass.

The corrected [live Hero zero follow-up](analysis/hero-fp8-cache-validation/hero-zero-v2-summary.json)
passes all cache coverage, with 19 exact comparisons at zero and 27 changed-split
comparisons within the unchanged bounds (maximum 0.0347961). Its clean source is
`dff1c8812335eb9047d6da0cf6ef279e588a71cb`. Actual profile, runtime, fixed goldens
and policy match the first run exactly. The
[reuse record](analysis/hero-fp8-cache-validation/hero-zero-evidence-reuse.json)
combines its fresh deployment/cache checks with the earlier passing reference,
batch and decode stages. Its complete cost is 3.408889 GB200-hours; all four pods
are absent. This establishes provisional serving coverage at zero. Throughput
effects remain unmeasured.

Snowball also passes the [full retention-zero suite](analysis/hero-fp8-cache-validation/snowball-zero-v2-summary.json)
on eight H100s, including 19 exact comparisons at zero. Its
[retention-128 cache follow-up](analysis/hero-fp8-cache-validation/snowball-positive-v2-summary.json)
passes the same 19 controls and all required hits. Each has 27 changed-split
comparisons within unchanged bounds. Existing positive reference, batch and
decode evidence is [reused](analysis/hero-fp8-cache-validation/snowball-positive-evidence-reuse.json):
model, GPU type/count, CPU/RAM, serving settings, runtime, reported hardware,
policy and goldens match. Placement moved from RNO to east02 after a resource
snapshot. Both clean follow-up sources are `9d7b0cfd6be15aab3a23aa0cd2d906208fe85f80`.

The [allocated validation cost](analysis/hero-fp8-cache-validation/allocated-validation-costs.json)
is 8.522222 GB200-hours and 2.008889 H100-hours, including the first failed Hero
probe. Two cancelled RNO attempts had no observed node allocation and retain
their null native cost receipts. Both east02 jobs remain Succeeded after
[cleanup](analysis/hero-fp8-cache-validation/snowball-east02-cleanup.json).
Their native archives and terminal stdout were preserved before ordinary
deletion of completed pods. All parent GPUs are released.

Hero positive retention also passes the [full v2 suite](analysis/hero-fp8-01a1ec55/bf16-c32-a1-check-summary.json)
in the child's BF16 C32 trial on 16 GB200s: all 19 exact differences are zero.
Its clean executed source is `33d780669d850e6b1c4166b2473973b5ff08c225`.
Both models have now exercised zero and positive retention on the pinned runtime.
The child's AgentX measurement remains a separate claim.
