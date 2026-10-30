# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One native serving task captures its fixed A5 shard and persists outputs before teardown."""

import argparse
import hashlib
import json
import os
import shlex
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

from connectrpc.errors import ConnectError
from iris.client.client import iris_ctx
from iris.cluster.types import Entrypoint
from marin.inference.iris import _prepared_local_inference, _register_dashboard
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration

from experiments.post_training.identity_path_b import replay_serve
from experiments.post_training.identity_path_b.a5 import capture as a5_capture
from experiments.post_training.identity_path_b.replay_capture import SOURCE_MODEL
from experiments.post_training.identity_t2.capture import checked_tokenizer

LIFETIME = 12 * 3600
METADATA = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_remote(uri: str) -> bytes:
    fs, key = filesystem_for(uri)
    return fs.cat_file(key)


def write_new_remote(uri: str, raw: bytes) -> None:
    fs, key = filesystem_for(uri)
    if fs.exists(key):
        raise FileExistsError("Refusing to overwrite an executor artifact")
    fs.pipe_file(key, raw)


def safe_failure(error: Exception) -> dict:
    """Record code and source frames without exception messages, URLs or local values."""
    result: dict[str, Any] = {"error_type": type(error).__name__}
    if isinstance(error, ConnectError):
        result["rpc_code"] = error.code.name
    result["frames"] = [
        {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
        for frame in traceback.extract_tb(error.__traceback__)
    ]
    return result


def persist_capture(output: Path, root: str, archive: Path) -> dict:
    """Retain every worker-local file, including empty/failed pilot records."""
    files = sorted(path for path in output.rglob("*") if path.is_file()) if output.exists() else []
    records = {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": digest_file(path)} for path in files
    }
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for path in files:
            bundle.add(path, arcname=str(path.relative_to(output)), recursive=False)
    uri = f"{root}/captures.tar.gz"
    fs, key = filesystem_for(uri)
    if fs.exists(key):
        raise FileExistsError("Capture archive already exists")
    fs.put_file(str(archive), key)
    if fs.info(key)["size"] != archive.stat().st_size:
        raise ValueError("Regional archive upload length mismatch")
    receipt = {
        "archive": uri,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest_file(archive),
        "files": records,
        "file_count": len(files),
        "raw_attempt_files": sum(name.endswith(".json") and name[:-5].isdigit() for name in records),
        "verification": "upload completed and remote length matches; SHA256 is worker-local archive digest",
    }
    for name in ("capture-report.json",):
        path = output / name
        if path.exists():
            write_new_remote(f"{root}/{name}", path.read_bytes())
    write_new_remote(f"{root}/persistence-receipt.json", (json.dumps(receipt, indent=2) + "\n").encode())
    return receipt


def capture_and_persist(rows: list[dict], *, tokenizer, endpoint: str, model: str, output: Path, root: str) -> dict:
    """Upload all raw outputs on either capture outcome, inside the serving context."""
    result: dict[str, Any] = {"capture_complete": False, "started_unix": time.time()}
    try:
        if model != SOURCE_MODEL:
            raise ValueError("Native served model alias differs from the exact A5 request model")
        result["capture"] = a5_capture.capture(
            rows, tokenizer=tokenizer, endpoint=endpoint, output=output, artifact_root=root
        )
        result["capture_complete"] = True
    except Exception as error:
        result["capture_error_type"] = type(error).__name__
        result["capture_failure"] = safe_failure(error)
    finally:
        result["persistence"] = persist_capture(output, root, output.parent / "captures.tar.gz")
    result["finished_unix"] = time.time()
    return result


def run_same_task(run_id: str, artifact_root: str, manifest_uri: str, manifest_sha256: str, shard: int) -> None:
    """Use run_iris_service's native direct contexts; no child submit or coordinator."""
    replay_serve.check_run(run_id)
    if not artifact_root.startswith("s3://marin-rno2a/tmp/ttl=14d/"):
        raise ValueError("Replay execution requires the RNO14-day TTL root")
    replay_serve.t2.validate_runtime()
    replay_serve.t2.configure_worker_s3()
    configure_coreweave_s3()
    work = Path("/tmp") / run_id
    work.mkdir(exist_ok=False)
    output = work / "raw"
    result: dict[str, Any] = {
        "run_id": run_id,
        "artifact_root": artifact_root,
        "started_unix": time.time(),
        "child_submissions": 0,
    }
    try:
        manifest = work / "manifest.jsonl"
        manifest.write_bytes(read_remote(manifest_uri))
        rows = a5_capture.checked_manifest(manifest, manifest_sha256, shard)
        metadata = work / "metadata"
        metadata.mkdir()
        for name in METADATA:
            (metadata / name).write_bytes(read_remote(f"{SOURCE_MODEL}/{name}"))
        tokenizer = checked_tokenizer(metadata)
        service = replay_serve.service_config(run_id)
        # This is the direct path in native run_iris_service, with its blocking idle
        # loop replaced by the finite reviewed capture. Both contexts own teardown.
        with _prepared_local_inference(service.model, service.engine, service.iris) as local:
            result["native_session_ready"] = True
            with _register_dashboard(
                service,
                local.model,
                tensor_parallel_size=local.tensor_parallel_size,
                backend_name=local.backend_name,
                streaming=True,
                chat_template_content=local.chat_template_content,
            ):
                client = iris_ctx().client
                if client is None:
                    raise RuntimeError("Native endpoint client unavailable")
                token = client.mint_endpoint_token(service.endpoint_name, ttl=Duration.from_seconds(LIFETIME))
                endpoint = token.capability_url.rstrip("/") + "/v1"
                os.environ["PATH_B_A5_ENDPOINT"] = endpoint
                try:
                    result.update(
                        capture_and_persist(
                            rows,
                            tokenizer=tokenizer,
                            endpoint=endpoint,
                            model=local.model.endpoint.model,
                            output=output,
                            root=artifact_root,
                        )
                    )
                finally:
                    os.environ.pop("PATH_B_A5_ENDPOINT", None)
        result["native_contexts_released"] = True
    except Exception as error:
        result["executor_failure"] = safe_failure(error)
        if "persistence" not in result and not (work / "captures.tar.gz").exists():
            try:
                result["persistence"] = persist_capture(output, artifact_root, work / "captures.tar.gz")
            except Exception as persistence_error:
                result["persistence_failure"] = safe_failure(persistence_error)
    finally:
        result["finished_unix"] = time.time()
        raw = (json.dumps(result, indent=2) + "\n").encode()
        (work / "execution-receipt.json").write_bytes(raw)
        write_new_remote(f"{artifact_root}/execution-receipt.json", raw)
    print(json.dumps({key: result.get(key) for key in ("run_id", "capture_complete", "native_contexts_released")}))
    if not result.get("capture_complete") or "persistence" not in result or not result.get("native_contexts_released"):
        raise RuntimeError("Replay capture/persistence incomplete; inspect safe execution receipt")


def same_task_submission(
    run_id: str, artifact_root: str, manifest_uri: str, manifest_sha256: str, shard: int, credentials: dict[str, str]
) -> dict:
    """Reuse the exact native GPU submission and activation, replacing only its command."""
    replay_serve.check_run(run_id)
    script = (
        'set -euo pipefail\nroot_python="$IRIS_VENV/bin/python"\n'
        f"source {shlex.quote('/app/vendor/skyrl/.iris-runtime-env')}\n"
        'exec "$root_python" -u -m experiments.post_training.identity_path_b.a5.execute '
        f"--run-id {shlex.quote(run_id)} --artifact-root {shlex.quote(artifact_root)} "
        f"--manifest-uri {shlex.quote(manifest_uri)} --manifest-sha256 {shlex.quote(manifest_sha256)} --shard {shard}"
    )
    submission = replay_serve.submission(run_id, credentials)
    submission["entrypoint"] = Entrypoint(command=["bash", "-c", script])
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    run_same_task(args.run_id, args.artifact_root, args.manifest_uri, args.manifest_sha256, args.shard)


if __name__ == "__main__":
    main()
