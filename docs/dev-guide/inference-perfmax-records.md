# Inference campaign records

Use Marin's existing [artifact record API][artifact] for machine-readable
campaign evidence, [fsutil](../references/fsutil.md) for storage, and
[research engineering](research-engineering.md) for code, analysis and the
notebook. These APIs are on main. The fields below are a recording convention;
there is no dedicated perfmax runner, validator or Marina ingester yet.

## Store records beside upstream exports

Resolve the execution cluster's permanent output prefix: `MARIN_PREFIX` in
`lib/iris/config/<cluster>.yaml`, including actual job overrides. Follow the
[CoreWeave storage guide](../tutorials/cloud-gpu.md#keep-data-on-coreweave).
Exclude `hero-checkpoints`, which is reserved for checkpoints. Record the
resolved bucket, region and any transfer cost; configured output prefixes can
cross regions. For `cw-us-west-04a`, the current permanent prefix is
`s3://marin-us-west-04a/marin`. Use this layout:

```text
<permanent-prefix>/inference-perfmax/<campaign>/
  contracts/<revision>/.artifact.json
  attempts/<attempt-id>/.artifact.json
  attempts/<attempt-id>/agentx/             # unchanged upstream exports
  decisions/<revision>/.artifact.json      # selected attempts and interpretation
```

Use a new contract revision when the agreement changes and a new attempt ID for
every execution, retry or repeat. Keep completed records immutable; corrections
link `supersedes` to the earlier record. List the campaign's attempts to recover
negative results as well as selected runs. GitHub issue/PR links are optional
metadata; storage identities must work without them.

A contract records the agreed model and comparison controls, scope, numerical
criteria, budget and retention. An attempt references its contract and baseline
record paths. Common immutable controls can live once in the contract; record
fully resolved overrides and actual runtime facts on the attempt. The contract
distinguishes fixed controls from declared tuning dimensions,
such as concurrency. Changing a fixed control starts a separate comparison
identity; varying a declared dimension stays in that comparison.

## Record enough to compare attempts

Use `ArtifactRecord.config` for inputs and `result` for observations and
interpretation. Put `record_kind` in `result` (`inference-perfmax.campaign.v1`,
`inference-perfmax.attempt.v1` or `inference-perfmax.decision.v1`).

| Record content | Fields to preserve, directly or through immutable references |
| --- | --- |
| Model and workload | Checkpoint/export/tokenizer pins and hashes; template, dtypes, context; corpus loader/revision/hash; replay, warmup, duration, seed, cache and routing controls |
| Software and execution | Engine/harness/processor SHAs, wheel/image digests, exact commands and relevant environment; code/snapshot URI and digest; cluster, job/task/attempt IDs |
| Actual hardware | GPU SKU/UUIDs and all allocated GPUs, hosts/interconnect, CPU resources, parallelism; actual backend and graph mode |
| Campaign intent | Contract/comparison identity, baseline paths, hypothesis, prediction, rejection condition and configuration change |
| Attempt outcome | Stage (diagnosis, calibration or confirmation), status and failure reason, repeat group; benchmark, numerical/state, task-quality and RL qualification with evidence |
| Measurements and cost | Native metrics/units, repeat identity, errors/cancellations; allocated GPU-seconds summed across allocation intervals, including failed startup and warmup; phase breakdown when available |
| Interpretation and artifacts | Conclusion, uncertainty and next decision; unchanged exports, commands, logs and checks with URI, digest, size and retention; optional related GitHub links |

Marin's `provenance` carries source/launch context. A manual writer's captured
provenance describes that writer; retain the actual execution's launch
provenance and source snapshot explicitly when recording later. Check for
missing fields instead of treating an empty provenance field as proof.

The [sweep executor][sweep] writes `.artifact.json` only after success. Keep these
campaign attempt records in their own directories, without overwriting
executor-owned records. Write a failed attempt manually when needed, even if
no AgentX export exists; retain the failure, allocation cost and available
logs. Repeats share a configuration/repeat-group identity and have distinct
attempt records. Preserve original exports before deriving comparisons.

## Retain and recover

Keep compact records, configuration, qualification evidence and confirmed
result exports under the permanent prefix. For large intermediates, choose a
supported TTL at kickoff and use existing [lifecycle-managed scratch](../tutorials/storage-bucket.md#step-4-ttl-scratch-prefixes-tmpttlnd):
`tmp/ttl=Nd/` expires objects N days after they are written. Record the resolved
URI, TTL and write time. Preserve compact failure evidence and promote confirmed
exports to durable storage before expiry, retaining their digests and links.
Temporary task archives and expiring CI artifacts alone cannot serve as durable
evidence.

The [existing helper][scratch-helper] below only constructs a path. Its
CoreWeave default uses shared scratch in `marin-us-east-08a`; pass `temp_prefix`
with an existing managed bucket when the campaign needs region-local scratch. This illustrative example
selects west-region scratch explicitly and performs no storage write:

```python
from rigging.filesystem.cluster_config import marin_temp_bucket

scratch = marin_temp_bucket(
    ttl_days=30, prefix="inference-perfmax/example-campaign/a1",
    temp_prefix="s3://marin-us-west-04a",
)
# s3://marin-us-west-04a/tmp/ttl=30d/inference-perfmax/example-campaign/a1
```

Archive experimental source, analysis and notebook on the Marin-derived
`marin-logbooks` research branch. No merge to main is needed. Preserve exact
commits and artifact references before handoff. Follow existing publication
rules for private evidence. Future scheduling and Marina ingestion should read
these records; they do not define measurement execution.

The following commands are available on main. Replace the example path with
the campaign's regional prefix. `cat --raw` preserves stored JSON:

```bash
uv run fsutil ls -R s3://marin-us-west-04a/marin/inference-perfmax/example-campaign/attempts
uv run fsutil cat --raw s3://marin-us-west-04a/marin/inference-perfmax/example-campaign/attempts/a2/.artifact.json
```

## Small local example

This illustrative Python example uses existing APIs and invented measurements.
Run it in a Marin environment; it creates only `perfmax-example/` locally.
It demonstrates record links and recovery. Production records also need the
identities, execution evidence and exports in the table above.

```python
from pathlib import Path

from marin.execution.artifact import ArtifactRecord, read_record, write_record

root = Path("perfmax-example").resolve()
contract_path = root / "contracts" / "agreed-1"
contract_path.mkdir(parents=True)
write_record(ArtifactRecord(
    name="example", output_path=str(contract_path),
    config={"model": "fixture-only", "comparison_id": "fixture-1"},
    result={"record_kind": "inference-perfmax.campaign.v1",
            "criteria": {"ttft_p90_max_seconds": 10},
            "budget_gpu_seconds": 40000},
))

baseline_path = root / "attempts" / "a1"
for attempt, concurrency, rate, latency, cost in [
    ("a1", 128, 160.0, 8.0, 14400),
    ("a2", 256, 164.0, 80.0, 19200),
]:
    path = root / "attempts" / attempt
    path.mkdir(parents=True)
    write_record(ArtifactRecord(
        name=attempt, output_path=str(path),
        config={"concurrency": concurrency},
        result={
            "record_kind": "inference-perfmax.attempt.v1",
            "contract_path": str(contract_path), "comparison_id": "fixture-1",
            "baseline_path": str(baseline_path) if attempt == "a2" else None,
            "hypothesis": "More concurrency may improve GPU efficiency",
            "stage": "calibration", "status": "completed",
            "repeat_group": f"fixture-c{concurrency}",
            "qualification": {"benchmark": "valid", "numerical_state": "pending",
                              "task_quality": "pending", "rl": "pending"},
            "metrics": {"output_tokens_per_gpu_second": rate,
                        "ttft_p90_seconds": latency},
            "cost": {"allocated_gpu_seconds": cost},
            "interpretation": "Confirm baseline; candidate exceeds latency cap",
        },
    ))

candidate = read_record(str(root / "attempts" / "a2"))
baseline = read_record(candidate.result["baseline_path"])
contract = read_record(candidate.result["contract_path"])
print(baseline.result["metrics"], candidate.result["metrics"])
print(contract.result["criteria"])
```

The candidate gains 2.5% throughput with ten times the latency and exceeds this
example's agreed limit. Keep the baseline as the calibration choice; neither
attempt is numerically or RL qualified. Recover both records before choosing
confirmation work. A real decision record links the compared attempts, total
campaign cost, qualification evidence and remaining uncertainty.

[artifact]: https://github.com/marin-community/marin/blob/83ab8b902e/lib/marin/src/marin/execution/artifact.py
[sweep]: https://github.com/marin-community/marin/blob/83ab8b902e/lib/marin/src/marin/execution/sweep.py
[scratch-helper]: https://github.com/marin-community/marin/blob/83ab8b902e/lib/rigging/src/rigging/filesystem/cluster_config.py
