# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One exercised native local-inference owner for the fixed A7 pilot."""

import argparse
import hashlib
import json
import os
import shlex
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from iris.client.client import iris_ctx
from iris.cluster.types import Entrypoint
from marin.inference.iris import _prepared_local_inference, _register_dashboard
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration

from experiments.post_training.identity_path_b import replay_serve
from experiments.post_training.identity_path_b.a5 import execute as parent
from experiments.post_training.identity_path_b.a5b.data import QUOTAS
from experiments.post_training.identity_path_b.a7.capture import capture
from experiments.post_training.identity_path_b.a7.data import TEACHER, binding, template
from experiments.post_training.identity_path_b.async_launch import checked_export_event
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
from experiments.post_training.identity_t2.capture import checked_tokenizer

LIFETIME = parent.LIFETIME


def checked_manifest(path: Path, expected_sha256: str, shard: int) -> list[dict]:
    bound = binding()
    raw = path.read_bytes()
    if (
        shard != 0
        or expected_sha256 != bound["manifest"]["sha256"]
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError("A7 requires its exact fixed200 manifest")
    rows = [json.loads(line) for line in raw.splitlines()]
    if (
        len(rows) != 200
        or [row["source_id"] for row in rows] != bound["source_ids"]
        or Counter(row["family"] for row in rows) != QUOTAS
    ):
        raise ValueError("A7 fixed200 source/order/family selection changed")
    for row in rows:
        if (
            row["teacher_system"] != ""
            or row["teacher"] != "A3_user_only_native_prefill"
            or row["tools"]
            or row["enable_thinking"] is not True
            or not row["prompt"]
            or any(message["role"] != "user" or set(message) != {"role", "content"} for message in row["prompt"])
            or row["sampling_seed"]
            != int(hashlib.sha256(f"310:A5:{row['source_id']}".encode()).hexdigest()[:8], 16) % (2**31)
        ):
            raise ValueError("A7 user-only/no-system/context/seed setting changed")
    return rows


def validate_args(run_id: str, artifact_root: str, manifest_uri: str, manifest_sha256: str, shard: int) -> None:
    bound = binding()
    if (run_id, artifact_root, manifest_uri, manifest_sha256, shard) != (
        bound["run_id"],
        bound["artifact_root"],
        bound["manifest"]["uri"],
        bound["manifest"]["sha256"],
        0,
    ):
        raise ValueError("A7 execution differs from its one-pilot binding")


def service_config(run_id: str):
    original = replay_serve.service_config(run_id)
    return replace(original, model=replace(original.model, weights=TEACHER, chat_template_content=template()))


def capture_and_persist(rows: list[dict], *, tokenizer, endpoint: str, model: str, output: Path, root: str) -> dict:
    """Upload all raw outputs on either capture outcome, inside the serving context."""
    result: dict[str, Any] = {"capture_complete": False, "started_unix": time.time()}
    try:
        if model != TEACHER:
            raise ValueError("Native served model alias differs from the exact A5 request model")
        result["capture"] = capture(rows, tokenizer=tokenizer, endpoint=endpoint, output=output, artifact_root=root)
        result["capture_complete"] = True
    except Exception as error:
        result["capture_error_type"] = type(error).__name__
        result["capture_failure"] = parent.safe_failure(error)
    finally:
        result["persistence"] = parent.persist_capture(output, root, output.parent / "captures.tar.gz")
    result["finished_unix"] = time.time()
    return result


def run_same_task(
    run_id: str,
    artifact_root: str,
    manifest_uri: str,
    manifest_sha256: str,
    shard: int,
    *,
    manifest_loader: Callable[[Path, str, int], list[dict]] = checked_manifest,
) -> None:
    """Use run_iris_service's native direct contexts; no child submit or coordinator."""
    validate_args(run_id, artifact_root, manifest_uri, manifest_sha256, shard)
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
        manifest.write_bytes(parent.read_remote(manifest_uri))
        rows = manifest_loader(manifest, manifest_sha256, shard)
        metadata = work / "metadata"
        metadata.mkdir()
        for name in parent.METADATA:
            (metadata / name).write_bytes(parent.read_remote(f"{TEACHER}/{name}"))
        if (
            hashlib.sha256((metadata / "tokenizer_config.json").read_bytes()).hexdigest()
            != "d9270d9407a15ec895fa2cdb0b6fbe9cd7dab937b91c2db708a918b35efa18b3"
        ):
            raise ValueError("Actual A3 tokenizer_config changed")
        for name, digest in binding()["teacher_metadata_sha256"].items():
            raw = (metadata / name).read_bytes() if name in parent.METADATA else parent.read_remote(f"{TEACHER}/{name}")
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Actual A3 teacher metadata/audit/event changed: " + name)
        tokenizer = checked_tokenizer(metadata)
        checked_export_event(CheckpointEvent(**binding()["teacher_event"]))
        service = service_config(run_id)
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
        result["executor_failure"] = parent.safe_failure(error)
        if "persistence" not in result and not (work / "captures.tar.gz").exists():
            try:
                result["persistence"] = parent.persist_capture(output, artifact_root, work / "captures.tar.gz")
            except Exception as persistence_error:
                result["persistence_failure"] = parent.safe_failure(persistence_error)
    finally:
        result["finished_unix"] = time.time()
        raw = (json.dumps(result, indent=2) + "\n").encode()
        (work / "execution-receipt.json").write_bytes(raw)
        parent.write_new_remote(f"{artifact_root}/execution-receipt.json", raw)
    print(json.dumps({key: result.get(key) for key in ("run_id", "capture_complete", "native_contexts_released")}))
    if not result.get("capture_complete") or "persistence" not in result or not result.get("native_contexts_released"):
        raise RuntimeError("Replay capture/persistence incomplete; inspect safe execution receipt")


def submission(
    run_id: str, artifact_root: str, manifest_uri: str, manifest_sha256: str, shard: int, credentials: dict[str, str]
) -> dict:
    """Reuse the exact native GPU submission and activation, replacing only its command."""
    validate_args(run_id, artifact_root, manifest_uri, manifest_sha256, shard)
    replay_serve.check_run(run_id)
    script = (
        'set -euo pipefail\nroot_python="$IRIS_VENV/bin/python"\n'
        f"source {shlex.quote('/app/vendor/skyrl/.iris-runtime-env')}\n"
        'exec "$root_python" -u -m experiments.post_training.identity_path_b.a7.execute '
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
    parser.add_argument("--shard", type=int, choices=(0,), required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    validate_args(args.run_id, args.artifact_root, args.manifest_uri, args.manifest_sha256, args.shard)
    if args.inspect_only:
        print(
            json.dumps(
                dict(
                    run_id=args.run_id,
                    teacher=TEACHER,
                    manifest_sha256=args.manifest_sha256,
                    sources=200,
                    model_storage_or_child_calls=0,
                )
            )
        )
        return
    run_same_task(args.run_id, args.artifact_root, args.manifest_uri, args.manifest_sha256, args.shard)


if __name__ == "__main__":
    main()
