# Manual inference trials

Resolve a serving preset, run one fresh Iris deployment and replay pinned AgentX
data. GPU work runs independently of GitHub Actions and dashboards.
The [inference-perfmax skill](../../../.agents/skills/inference-perfmax/SKILL.md)
owns campaign choices and the adaptive experiment ladder.

```sh
uv run python -m experiments.evaluation.knotmeter.cli plan \
  --profile snowball-h100 --campaign snowball-throughput --output /tmp/plan.json
uv run python -m experiments.evaluation.knotmeter.cli run \
  --profile snowball-h100 --campaign snowball-throughput
```

`--profile` accepts a preset name or YAML path. Use `hero-gb200` for Hero/Mochi.
The presets request eight H100s on one host or sixteen GB200s on four hosts.
Inspect live capacity through `use-iris` and coordinate resources before running.
For an additional allowed-host check, pass `--placement-uri` with a JSON document
such as `{"nodes":["NODE1","NODE2","NODE3","NODE4"]}`. Choose spare nodes from live
provider and Iris inventory immediately before submission, following
[Iris operations](../../../lib/iris/OPS.md#coreweave-gpu-operations). Record the
cluster, verification time and inventory digest beside that list. Check protected
training priorities and recheck delayed admissions. This guard checks admitted
hosts against the list after allocation; it does not constrain or reserve placement.

Precedence is named flags, then `--set`, then the preset. Dotted
`--set FIELD=YAML` changes profile fields; unknown fields fail before submission.
For example:

```sh
uv run python -m experiments.evaluation.knotmeter.cli run \
  --profile snowball-h100 --campaign snowball-throughput \
  --session-concurrency 64 --priority interactive \
  --set serving.max_num_batched_tokens=4096
```

The runner resolves Marin's selected serving wheel once. `--vllm-release TAG`
uses another published release. `--vllm-manifest URI` accepts a build manifest
without changing `config/external/vllm/gpu.toml`. For an experimental build, reuse
the fork's [Docker wheel target and build configuration](https://github.com/marin-community/vllm/blob/main/infra/release/README.md#gpu-build-configuration),
then supply its `VllmGpuRelease` JSON: `release_tag`, `source_commit`,
`version`, `torch_backend`, `torch_version` and architecture-specific `wheels`
with `architecture`, `sm_targets`, `url` and `sha256`. Worker-accessible wheel
URLs must remain available for recovery. Optional `vllm.repo` and
`vllm.revision` record the source override; the revision must match the manifest.
The `vllm_release` object in `plan --output /tmp/plan.json` is a complete example.
This command does not refresh a fork, publish a release or introduce a build system.

[recipe.yaml](recipe.yaml) supplies the shared harness, corpus and processor
pins. Normal runs retain upstream warmup, replay and validity. Data preparation
runs on CPU before GPU submission. `--diagnostic-seconds N` explicitly requests
a short unsafe upstream replay; it cannot produce an accepted recommendation.

Preset `checks` run after profiling against the same deployment. Each check has
a name, argv list and timeout. Commands receive `KNOTMETER_SERVER_URL` (the
worker's `/v1` URL), `KNOTMETER_MODEL`, `KNOTMETER_MODEL_REVISION`,
`KNOTMETER_RUNTIME_SOURCE`, `KNOTMETER_PROFILE` and
`KNOTMETER_CHECK_OUTPUT`. A zero exit status records a pass; retain detailed
results in that output file. An empty command records a pending check. Both
presets leave model numerics pending because an existing gate must match the
checkpoint and serving settings. Numerical checks declare `checkpoint_revision`
and `runtime_source`. A mismatched pass stays regression evidence and leaves
qualification pending. `kind: serving` identifies a serving-only check. Link
reused evidence in the check's `evidence` list; a link alone does not establish a
pass. Missing or failed required checks block recommendations.
Passing these checks does not qualify RL learning.

Outputs live at `MARIN_PREFIX/knotmeter/campaigns/CAMPAIGN/`. The selected
cluster's configured prefix supplies the default; `--marin-prefix` is an explicit
job override. Each invocation creates a new trial directory. Native retries
keep distinct attempt IDs, host records, exports and logs. Native data uses a
checksum-addressed archive; `plan.json` and `result.json` retain recovery inputs.
Keep large disposable intermediates in existing TTL scratch storage.

The terminal prints the CPU coordinator's Iris job ID, trial URI and recovery
commands. Its GPU child job is that ID followed by `/serve`. `run` waits
by default. It also captures pod allocation times, including setup before
Python starts, through the selected cluster's existing `kubectl` configuration.
Capture records contain only timing and identity fields. If a pod expires, a retained scheduling
time plus Iris's terminal observation gives a conservative end time. Missing
timestamps remain unknown. `--no-wait` returns after submission; detached runs
may lack complete allocation timings. Interrupting the local wait leaves Iris running.

```sh
uv run python -m experiments.evaluation.knotmeter.cli status --trial-uri TRIAL_URI
uv run python -m experiments.evaluation.knotmeter.cli recover --trial-uri TRIAL_URI
uv run python -m experiments.evaluation.knotmeter.cli cancel --trial-uri TRIAL_URI
uv run python -m experiments.evaluation.knotmeter.cli report --campaign-uri CAMPAIGN_URI
```

Recover from the original source revision recorded in `plan.json`. Recovery
collects terminal work and never submits GPUs. After coordinator loss, inspect
its `/serve` child and cancel or wait for it before recovery. Check provider
cleanup before retrying.
The report prints the same Markdown saved as `summary.md`, including all attempts'
GPU cost. `report --recommend TRIAL_ID` requires valid protocol, unchanged gang,
passed checks and complete cost. Every physical host must retain the same Iris
attempt ID throughout profiling. The command writes `recommended.yaml` with a
frozen runtime. Add `--baseline TRIAL_ID` to show changes within a comparison
group; diagnostics with different controls remain separate rows.
Keep experimental code and discarded approaches in the
[research archive](../../../docs/dev-guide/research-engineering.md).

Every command supports `--help`. The focused CPU compatibility test runs the real
pinned upstream tools against local HTTP; its GPU counts are normalization inputs:

```sh
uv run pytest -n 0 -m integration \
  experiments/evaluation/knotmeter/test_workload.py::test_pinned_tools_prepare_restore_replay_and_normalize_physical_gpus
```
