# Workflow UX findings

The manual path uses one model/device preset, a shared pinned AgentX recipe and
explicit overrides. The campaign guide is 80 lines; the README owns commands.
Hero and Mochi are the same model. The original RFC remains unchanged.

## Changes tested during implementation

| Before | Current behavior | Evidence |
| --- | --- | --- |
| Fixed experiment grids and separate human guides | The skill chooses an adaptive ladder; one runner executes a trial. | Skill and runner README in the retained source. |
| CLI fields and workload names could drift from YAML | Named flags override dotted `--set` values; public dataset names resolve to pinned identities. | Real Snowball/Hero plans, type/lint checks. |
| Native harness preparation imported Marin | A pure command module runs inside isolated uvx environments. | The first real fixture failed; the corrected fixture passes. |
| Local submissions carried a federation-peer constraint | The constraint applies when entry and target controllers differ. | Both first launch rejections occurred before GPU allocation; corrected roots were admitted. |
| A mandatory node receipt rejected placement after allocation | The receipt is an optional guard; Iris chooses placement. | Contribution review and native gang admission. |
| The Hero preset requested an unavailable `numactl` binary | NUMA binding stays off for the pinned image; the approved compiler guard stays in place. | First live Hero startup failure, retained four-host receipts. |
| Command-line provenance changed shape across JSON | Plans normalize it before delivery. | A regression test and recovery of the original worker exports. |
| Short native exports omitted metric-coverage metadata | Coverage is incomplete; the report still renders and acceptance stays blocked. | Real Snowball exports and the upstream fixture. |
| Failed jobs appeared in the checks column | GPU job state, AgentX validity and required checks have separate report fields. | Recovered Snowball and Hero reports. |

The report shows completed output tokens/s per physical GPU and p90 first-token
and completion latency. Both terminal output and retained `summary.md` use the
same renderer. A short diagnostic does not establish a baseline or a speedup.
Synthetic report tests exercise baseline deltas and pin the selected runtime.
They are not performance measurements.

## Proposed RFC clarifications

- State the evidence scope of each required check: checkpoint and runtime. A
  pass for another scope stays regression evidence. Keep measurement validity,
  numerical correctness and recommendation acceptance separate.
- Define allocated GPU cost as scheduling through the last container finish,
  including setup, warmup, drain and failed attempts. Keep missing timings
  unknown. Attached runs retain provider timing snapshots; detached runs may
  lack them and therefore remain ineligible for recommendations.
- Explain that a source override uses the existing wheel-build tools, followed
  by a pinned runtime manifest. A repository and revision do not build a wheel.
- Describe the normal one-hour recipe and explicit short diagnostics. AgentX
  supplies measurement validity; the agent chooses the experiment ladder.
- Preserve the throughput goal near the latency bend. Choose numerical tradeoff
  criteria against a useful baseline at kickoff; do not invent a universal SLO.

## Qualification still missing

Snowball v0.1.1 lacks matching endpoint-numerics evidence for runtime
`0ac9dcb157ee86ccb447bc717259159c41ad456a`. The existing step-42150 gate uses a
different checkpoint and cannot qualify it. Hero step144K also lacks a matching
shared endpoint gate for this runtime. Both presets leave model numerics
pending. Passing readiness, native replay or serving tests does not close this
gap. No new numerical or trainer-mismatch investigation was run.

A recommendation also needs a complete normal AgentX window, stable gang,
complete costs and passed required checks. RL learning quality needs evidence
for the actual trainer and rollout path. This implementation makes no RL
quality claim and accepts no diagnostic recommendation.

Startup was quiet in the attached terminal while Iris prepared data and loaded
weights. The job IDs and status command remained usable; native logs and exports
were retained. A clearer phase indicator would improve this. Profiling duration
also excludes startup and native warmup, so a 90-second diagnostic still pays
those costs. The skill and measurement reference make that distinction explicit.

## Final review corrections

The final review found three gaps. The runner now takes the default storage
prefix from the target cluster; an ambient workstation `MARIN_PREFIX` cannot
redirect it. Accepted recommendations require a passed numerical check whose
checkpoint and runtime match the measured trial. Removing or relabeling that
check cannot waive qualification. Recovery keeps interrupted exports and their
allocated costs; plans without results stay visible with unknown cost.

The table now links the serving-runtime commit. List-index overrides receive a
clear error, and the README explains whole-list replacement. Unknown CUDA
backends fail before allocation. The experimental JSON loader remains separate
from the promoted TOML-pin updater because those inputs have different purposes;
its unused optional-path branch was removed.

## Examples for an RFC revision

The RFC's path-based `run --profile .../snowball-h100.yaml` command works.
The implementation also accepts a preset name and offers a read-only plan:

```sh
uv run python -m experiments.evaluation.knotmeter.cli plan \
  --profile snowball-h100 --campaign review
uv run python -m experiments.evaluation.knotmeter.cli run \
  --profile snowball-h100 --campaign review \
  --session-concurrency 8 --diagnostic-seconds 90
```

A copied YAML preset must include `gpus.per_host` and exact model/tokenizer
identities. Serving controls live under `serving`. Numerical checks declare
`kind`, `checkpoint_revision` and `runtime_source`; an empty command stays
pending. The supplied files give complete examples. `cw-rno2a` is the current
Snowball default, selected from live Iris configuration. `--cluster` is explicit.
Native exports use checksummed archives; plans, results and receipts remain
separate JSON objects. Use `report --recommend TRIAL_ID` only after qualification.
