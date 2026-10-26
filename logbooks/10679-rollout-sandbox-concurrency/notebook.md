---
title: "Iris sandbox startup executor occupancy"
source_repository: "marin-community/marin"
task_record: "https://github.com/marin-community/marin/pull/10679"
research_branch: "atqamar/rollout-sandbox-concurrency-study"
marin_base_revision: "2f270241d4b538694d66ba0f4483edbfc6da4ff6"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: active
---

# Iris sandbox startup executor occupancy

## Decision

Use asynchronous readiness sleeps to release executor threads while Iris machines start. Retain accepted submissions and cleanup through cancellation. Runtime sandbox, executor, verifier and controller defaults retain their configured values. Contribution: [Marin PR #10679](https://github.com/marin-community/marin/pull/10679).

## 2026-10-26T02:58:47Z — measured source and results archive

Two real CPU-only RNO waves each completed 64/64 trajectories with one actual TaskRolloutWorker, real shell commands, captured files and host Verifyit grading. A scripted model supplies exact token provenance; sandbox operations and verification are real. The base is 2cae063c877172844fe6ff90e94354cc08a273fa; the startup revision is 8e4c8102511be38b54ed1da0c230c72136e3a26e. The later 2f270241d4 correctness commit protects queued cleanup through repeated cancellation; it does not change the successful startup path.

| Metric | Base | Startup revision |
| --- | ---: | ---: |
| Completed / failures | 64 / 0 | 64 / 0 |
| Summed startup executor occupancy, s | 877.915 | 78.863 |
| Tool p50 / p95, ms | 882.737 /1069.735 | 870.504 /1290.466 |
| Whole-wave wall, s | 26.890 | 22.753 |
| Peak acquired handles | 48 | 53 |
| Client executor queue p95, ms | .626 | .614 |

The observed startup executor occupancy reduction is 91.0%. It sums blocking calls across threads; it is neither CPU time nor a wall-clock phase. Tool p95 is higher in the startup revision, and one pair leaves tool/whole-wave latency gains unproven. The natural-churn workload does not hold 64 live sandboxes, use fresh verifier sandboxes, or represent slow production grading. [Full configuration, metric definitions and distributions](results.json).

The fixed 256-session burst attempt stopped before its first tool when controller RTT reached 543 ms against 77 ms preflight. It supplies no capacity or performance point. Larger-scale qualification remains pending a healthy window and the study's existing stop rules.

### Source provenance and reproduction

The [measured-source directory](measured-source/) preserves the four worker-side files used in both waves. Each restored file was checked byte-for-byte against its SHA256 recorded in both run manifests. This archive was committed after the measurements; the measured code revisions and recorded file hashes provide its provenance. No experiment was executed from this archive revision. [results.json](results.json) records those hashes and the exact job identifiers already reported in the contribution.

Restore the files from the research checkout root:

```sh
mkdir -p sbc
for source in logbooks/10679-rollout-sandbox-concurrency/measured-source/*.py.txt; do
  cp "$source" "sbc/$(basename "$source" .txt)"
done
```

The worker environment uses the frozen vendored SkyRL CPU lock with uv 0.13.5: `uv tool run --from uv==0.13.5 uv sync --project vendor/skyrl --all-packages --extra cpu --frozen`. The measured worker install log confirms pyqwest 0.6.2. The measured entry point inside the approved CPU job was:

```sh
.venv/bin/python -m sbc.worker_entry --concurrency 64 --hold-seconds 0 --tool-rate 1000 --grading host --executor-threads 256 --park --mode natural --profile --profile-phase start
```

The job uses 4 CPUs / 8 GiB memory / 10 GiB disk at batch priority. Its Iris controller address and credentials come from the job environment. Sandbox requests are 1 CPU / 2048 MiB memory with the image digest in bench.py. Current capacity, other-user scheduling, latency and owned-resource cleanup checks belong to the external launch guard; this source archive does not change those operational boundaries.

The profile records idle and per-thread stacks at 5 Hz for 15 s. Guest timing adds two date subprocesses per exec and strips their nonce markers from stderr. Both arms use that same instrumentation. Tool latency is the public engine event duration, while factory, RPC, output capture and grading distributions are retained separately in results.json.

## Remaining evidence

Matched 256, sustained holds, 1K profiles, slow grading and large payloads remain unqualified. PR10118 contains separate command-framing and raw pod-JSON work; it is not part of this measured startup comparison. No numerical maximum backend concurrency follows from this pair.

## 2026-10-26T03:41:00Z: correction to the 256-session stop and next isolation

Recovered worker logs contain3343 rows; the earlier guard capture ended five seconds early. Before the start of the recorded01:20:05 abort second,256 launches,1184 status polls and56 execs completed across seven sandboxes. Actual agent tools began at01:20:02.225. The earlier “before first tool” sentence is invalid. This stopped attempt remains excluded from completed scaling/performance comparisons. The source trace receipt is retained in the local study; results.json carries the corrected counts.

All ramps remain stopped. The next authorized experiment uses one exact study-owned CPU-only RNO sandbox and one sequential client at2 exec/s. Compare fresh ApiClients with a reused client; every exec opens its own websocket. Preserve per-stage wall, process and calling-thread CPU, with alternating arm order and output/exit validation. The existing strict health guard supervises the local probe and confirms cleanup over60 seconds. No shared controller deployment or pool/default change.

The first probe submitted and cancelled its job before any exec measurement: immediate task-status lookup returned task-not-found. Its correction waits for the exact study pod's task container readiness through the Kubernetes API and submits directly to RNO. A fresh preflight is required after confirmed empty cleanup. Supporting exact source and manifest are in exec-api-probe/. Restore the two .py suffixes into sbc/ on the recorded Marin base and use the supplied service.py only for the local client-reuse draft. This archive does not open that PR or change the frozen10679 contribution.
