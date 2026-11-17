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

Both presets set the retention interval to 128 tokens. Hero retains periodic
Mamba states; Snowball's shorter-than-window interval retains all SWA blocks.
In the pinned runtime, interval zero keeps selected replay and shared-prefix checkpoints; a
miss on generated history can therefore be legitimate. The positive interval
lets the suite require and verify decode-created reuse across its boundaries.
The report records actual hit counts and cache settings. A miss still fails the
required coverage check; it does not establish corrupt state.
The fixed probe generates 257 tokens. Interval zero, or an interval too large
for the probe to reach, therefore permits diagnostics only. The throughput effect
of changing zero to 128 is unmeasured on both models. Both Snowball measurement
windows use 128; no Hero AgentX window was run.

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
