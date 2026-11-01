# Complete and score a reproduced Hero run

Use the exact submitted manifest, job ID and current native attempt UIDs. A replaced attempt needs a fresh complete measurement window. Native job success alone does not prove that all output archives uploaded or that AgentX accepted the hour.

Inspect the native job and each of its four attempts:

```sh
marin-env uv run --no-sync iris --cluster=cw-us-east-08a job describe /romain/your-job
marin-env uv run --no-sync iris --cluster=cw-us-east-08a attempt describe /romain/your-job/0:0
```

Repeat the attempt command for task indexes 1–3, using their actual attempt numbers. Native attempt descriptions contain the output-archive URI, uploader checksum and archive state. [The output guide](../../../lib/iris/docs/task-outputs.md) describes capture limits. Require four uploaded native output archives and preserve their original attempt IDs. A holder’s terminal state is separate from its benchmark child’s `point/attempt.json` status.

The research entrypoint also writes permanent, fully read-back raw-output receipts at:

```text
<completion_retention_prefix>/host-<index>/<actual-attempt-uid>/retention.json
```

Each receipt names `archive_uri`, `archive_sha256`, `readback_sha256`, compressed bytes, native task ID and child application status. Require matching hashes and `application_status=succeeded` for every host. Download each archive through `copy_file` in `experiments.evaluation.agentx.artifacts`, verify `file_digest`, then `unpack` it. The unpacked root contains `point/attempt.json` and `point/aiperf_artifacts`. Call unchanged `validate_attempt(point, manifest, actual_uid)` for every host. The leader’s `gang_attempts` map must match all four actual native task IDs and UIDs; require four distinct recorded nodes and sixteen distinct GPU UUIDs on the verified spare domain.

For the leader, restore a packed server-metrics export before processing. Run this from the research checkout after checksum-verifying and unpacking the leader archive. Substitute the exact manifest, restored root, actual leader UID and fresh local processing directory:

```sh
marin-env uv run --no-sync python - \
  path/to/submitted-manifest.json path/to/restored-leader-root \
  actual-leader-attempt-uid path/to/processed <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path
import zstandard
from experiments.evaluation.agentx.collect import validate_attempt, validate_window, process_point

manifest = json.loads(Path(sys.argv[1]).read_text())
point = Path(sys.argv[2]) / 'point'
uid = sys.argv[3]
validate_attempt(point, manifest, uid)
print(json.dumps(validate_window(point, manifest, uid), indent=2))
packed = point / 'aiperf_artifacts/server_metrics_export.json.zst'
if packed.exists():
    receipt = json.loads(packed.with_suffix('.receipt.json').read_text())
    raw = packed.with_suffix('')
    with packed.open('rb') as encoded, zstandard.ZstdDecompressor().stream_reader(encoded) as decoded, raw.open('wb') as output:
        shutil.copyfileobj(decoded, output, length=8 * 1024**2)
    with raw.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['raw_sha256']
    assert raw.stat().st_size == receipt['raw_bytes']
process_point(point, manifest, Path(sys.argv[4]))
PY
```

The native validator checks scenario validity, the complete duration/metric coverage, successful output and the original error-ratio gate. Inspect `profiling_errors` explicitly when reporting a zero-error result. The pinned processor writes `bmk_agentic_hero144k_gb200_vllm_bf16_agentx_c128/agg.json` under the processing directory. Use `request_metrics.throughput.output.tokens_per_second` and its `per_gpu.output_tput_tps`, retaining the processor’s denominator. The processor’s `tp` field is a worker-group proxy; actual serving remains TP1/PP1/DP16/EP16.

The supplied processing command validates the leader window. The four-host identity, placement, model-stop and archive checks above remain required separately. For a standalone native batch job, [the native terminal collector](../../../experiments/evaluation/agentx/collect.py) supplies the gang and capture checks; [the coordinator](../../../experiments/evaluation/agentx/coordinator.py) shows the API call and finalization handling. The separate [owner metrics patch](README.md) adds lossless export restoration to that integration path.

Retain processed output and all four raw archives with complete SHA256 readback. For an interactive holder, also capture the entrypoint’s `launch.json` and complete log before release. Confirm every model and supervisor has stopped and every GPU has no compute process. Then release the holder and verify its original provider pods are gone. Close its resource-board row only after native/provider release. The published [policy comparison](../results/guarded-retention-agentx-c128-r1-comparison.json) provides the completed receipts and separate release proofs for both measured groups.
