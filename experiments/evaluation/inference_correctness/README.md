# Shared inference correctness checks

Pytest and the endpoint CLI call `marin.testing.inference.endpoint_checks`.
Perfmax retains that CLI's report and enforces its outcome. The existing
step42150 adoption tests keep their coverage and bounds.

Check a saved reference without allocating GPUs:

```sh
uv run python -m marin.testing.inference.check_endpoint \
  --fixture snowball-v0.1.1 --lookup
```

Use `hero-sft-step400407` for Hero/Mochi. The
[Git fixtures](../../../lib/marin/src/marin/testing/inference/resources/endpoint/)
pin checkpoint, tokenizer, manifest hash and a separate acceptance policy.
Lookup distinguishes `found`, `missing` and `mismatched`. Routine checks read
saved scores. They do not generate new goldens for a candidate runtime.

To deploy the same settings used by perfmax and run the pytest adoption gate:

```sh
uv run python -m experiments.evaluation.inference_correctness.launch \
  --profile snowball-h100 --name snowball-correctness --output execution.json
```

Use `hero-gb200` for Hero/Mochi. `--set FIELD=YAML` accepts the runner's profile
overrides. `--runtime-manifest URI` tests a release or experimental build without
adopting it. The attached launcher records every native attempt and GPU cost.
The execution receipt names the durable run URI. Each host retains an archive
and receipt; the leader's archive contains `report.json` and raw HTTP responses.
`--pytest-node PATH::TEST` runs a focused diagnostic through the same deployment
and cleanup path. Its plan is labeled diagnostic; it does not replace the full gate.

For an existing dedicated test endpoint, enable `VLLM_SERVER_DEV_MODE=1` so the
checks can read its settings and reset its cache. Both interfaces use the same
function and fail unless every required check passes:

```sh
uv run python -m marin.testing.inference.check_endpoint \
  --fixture snowball-v0.1.1 --url http://HOST:PORT/v1 --model MODEL \
  --runtime-source FULL_SHA --output correctness.json
uv run pytest tests/cluster/vllm/test_inference_correctness.py \
  -m cluster -o addopts= --endpoint-url http://HOST:PORT/v1 \
  --correctness-fixture snowball-v0.1.1 --correctness-model MODEL \
  --runtime-source FULL_SHA --correctness-output correctness.json
```

`FULL_SHA` is the `marin-community/vllm` source commit in the runtime manifest.
The suite checks its prefix against the endpoint's `/version`, compares
`/server_info` identities and verifies checkpoint metadata hashes.

The report separates checkpoint and runtime identity, reference disagreement,
batch dependence, cache reuse and decode versus recomputation. Batch requests
are pinned to one engine; its running-request metrics must show overlap. Cache
checks require actual hits, varied inputs, branches and reuse of decode-created
tokens. Equal hit counts can still leave different prefill steps: a hybrid cache
may introduce a new checkpoint inside the suffix. Initial seed/branch comparisons
use the bound for changed splits. Exact comparisons fully prepare the usable
prefix, then verify it survives a branch that differs inside the cached region.
That exact control checks persistence after branching; a wrong state that stays
unchanged can escape it. Fresh/reused comparisons provide a separate screen.
Raw responses and per-position differences remain available after a failure.
Later reference histories and cache boundaries still run after an individual
case fails. Coverage failures remain separate from measured probability errors.

Cache coverage follows the endpoint's retention interval and block alignment.
Retention preserves existing snapshots; it does not create them. An aligned
cold prefill can finish beyond the lower replay point retained at zero. The
probe therefore prepares checkpoint K with a K+1-token prompt and requires reuse
reaching K. Positive intervals also retain periodic checkpoints; `None` keeps
them densely. A direct decode
probe chooses a reachable periodic checkpoint, adjusting its seed or generation
length when needed. Its last sampled token has not yet produced cache state.
The report distinguishes this direct reuse from replay after generated history
has itself been processed as a prompt. Its sibling forks inside generated
history, after an explicitly prepared checkpoint. Both branches must show the
required hits and preserve scores exactly in every mode. Initial
recomputation at zero is allowed; failed required reuse still blocks acceptance.
See the pinned runtime's [replay boundaries](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/vllm/v1/core/kv_cache_coordinator.py#L314-L359)
and [snapshot creation](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/vllm/config/cache.py#L178-L185).

The presets currently use interval 128. Both zero and positive modes can qualify
when their supported paths pass. The throughput effect of changing zero to 128
remains an experiment question.

Reference checks bound the largest absolute probability error over the union of
saved top-64 and serving top-50 scores. Cache and batch checks use two serving
top-50 lists. Both bounds account for a missing token's possible probability
range. Decode checks instead compare the emitted token's probability during
generation and fresh scoring. They record its log-probability error too, without
an acceptance limit on that error. Policies report maximum, p99, p90 and the
fraction above 0.075 across these per-position values. None measures
full-vocabulary KL. The policies are provisional; a pass covers the saved
histories and listed state cases. It does not qualify every context, task
quality or RL trainer probability ratios.

Durable goldens live under `s3://marin-us-east-02a/marin/inference/goldens/`.
Checkpoint and suite identities name each directory. Manifests hash the prompt
and score files and record the reference code and export lineage. Original
reference artifacts remain under the same root, outside TTL storage. Reference
generation is separate; the Snowball generator lives in [reference.py](reference.py).
