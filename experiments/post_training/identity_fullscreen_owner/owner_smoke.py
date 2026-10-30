# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dummy-candidate child-spec and local-memory persistence checks; no model jobs."""

import copy
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from fsspec.implementations.memory import MemoryFileSystem
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.types import Entrypoint, JobName
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync

from experiments.post_training.identity_fullscreen_owner import candidate as candidate_module
from experiments.post_training.identity_fullscreen_owner import owner, persistence


def dummy_candidate() -> dict:
    root = "s3://marin-rno2a/tmp/ttl=14d/DUMMY-ONLY-NO-MODEL/different-parent"
    label = "dummy-parent-propagation"
    model = "DUMMY-NOT-A-MODEL"
    return {
        "label": label,
        "model_path": model,
        "artifact_root": root,
        "screen": {
            "generation_argv": [
                "python",
                "-c",
                "pass",
                "--model",
                model,
                "--out",
                f"_pathb_fullscreen/outputs/{label}/identity",
            ],
        },
        "math": [
            {
                "trial": trial,
                "output": f"{root}/math/{trial}",
                "argv": ["python", "-c", "pass", "--model", model, "--output", f"{root}/math/{trial}"],
            }
            for trial in ("greedy", "17", "29")
        ],
    }


def verify_child_specs() -> dict:
    """Capture actual native LaunchJob requests, extract and execute their commands."""
    requests = []

    def launch(client, request, **unused):
        requests.append(request)
        return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

    info = JobInfo(
        task_id=JobName.from_wire("/fixture/dummy-owner/0"),
        controller_address="http://unused.invalid",
        bundle_id="dummy-bundle",
    )
    previous = get_job_info()
    client = IrisClient(
        RemoteClusterClient(info.controller_address, bundle_id=info.bundle_id, use_controller_proxy=False)
    )
    value = dummy_candidate()
    path, raw, expected_sha = candidate_module.candidate_payload(value)
    root = Path(__file__).resolve().parents[3]
    reports = []
    try:
        set_job_info(info)
        with tempfile.TemporaryDirectory(prefix="dummy-screen-parent-") as directory:
            with patch.dict(os.environ, {"IRIS_WORKDIR": directory}):
                with iris_ctx_scope(IrisContext.from_job_info(info, client)):
                    with patch.object(ControllerServiceClientSync, "launch_job", launch):
                        owner.submit_cpu(client, value, "generation", "http://unused.invalid")
                        for trial in ("greedy", "17", "29"):
                            owner.submit_cpu(client, value, "math", "http://unused.invalid", trial)
        if len(requests) != 4:
            raise AssertionError("Expected four intercepted native child specs")
        for request in requests:
            entrypoint = Entrypoint.from_proto(request.entrypoint)
            words = shlex.split(entrypoint.command[-1])
            actual_path = words[words.index("--candidate") + 1]
            if actual_path != path or entrypoint.workdir_files[path] != raw:
                raise AssertionError("Native child spec did not carry the actual parent candidate")
            if words[words.index("--candidate-sha256") + 1] != expected_sha:
                raise AssertionError("Native child spec lost the parent candidate SHA")
            if "candidate52.json" in entrypoint.command[-1]:
                raise AssertionError("Native child command retained the historical fallback")
            with tempfile.TemporaryDirectory(prefix="dummy-screen-child-") as directory:
                workdir = Path(directory)
                for relative, contents in entrypoint.workdir_files.items():
                    target = workdir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(contents)
                env = dict(
                    os.environ,
                    IRIS_WORKDIR=str(workdir),
                    IRIS_VENV=str(Path(sys.executable).parent.parent),
                    JAX_PLATFORMS="cpu",
                    CUDA_VISIBLE_DEVICES="",
                    PYTHONSAFEPATH="1",
                )
                env["PYTHONPATH"] = ":".join([str(root), *[str(p) for p in (root / "lib").glob("*/src")]])
                command = list(entrypoint.command)
                command[-1] += " --inspect-only"
                completed = subprocess.run(
                    command, cwd=workdir, env=env, capture_output=True, text=True, check=True, timeout=30
                )
                report = json.loads(completed.stdout.strip().splitlines()[-1])
                expected_argv = (
                    value["screen"]["generation_argv"]
                    if report["kind"] == "generation"
                    else next(item["argv"] for item in value["math"] if item["trial"] == report["trial"])
                )
                if report["model_path"] != value["model_path"] or report["command_argv"] != expected_argv:
                    raise AssertionError("Extracted child command selected a different candidate/argv")
                if report["artifact_root"] != value["artifact_root"] or report["candidate_sha256"] != expected_sha:
                    raise AssertionError("Extracted child command selected a different output namespace")
                reports.append(report)
    finally:
        set_job_info(previous)
        client.shutdown()
    return {
        "actual_native_child_specs": len(requests),
        "extracted_child_commands_pass": len(reports),
        "candidate_path": path,
        "candidate_sha256": expected_sha,
        "model_calls": 0,
        "live_child_jobs": 0,
    }


def verify_persistence() -> dict:
    """Exercise archive collisions/readback locally, preserving all old bytes."""

    class MemoryStore(MemoryFileSystem):
        puts = 0

        def put_file(self, *args, **kwargs):
            self.puts += 1
            return super().put_file(*args, **kwargs)

    fs = MemoryStore(skip_instance_cache=True)
    prefix = uuid.uuid4().hex

    def local_store(uri):
        return fs, prefix + "/" + uri.removeprefix("s3://")

    value = dummy_candidate()
    root = value["artifact_root"]
    _, legacy = local_store(root + "/unit/available-output.tar.gz")
    fs.pipe_file(legacy, b"old archive must survive")
    with tempfile.TemporaryDirectory(prefix="dummy-screen-persistence-") as directory:
        output = Path(directory) / "output"
        data = output / "identity/generate/generations.jsonl"
        data.parent.mkdir(parents=True)
        data.write_bytes(b'{"dummy":true}\n')
        (output / "task.log").write_text("first attempt\n")
        with (
            patch.object(persistence, "filesystem_for", local_store),
            patch.object(candidate_module, "filesystem_for", local_store),
        ):
            first = persistence.persist_tree(
                output, root, "unit", "dummy-job-A-attempt-0", mirror_prefixes=("identity",)
            )
            puts = fs.puts
            repeated = persistence.persist_tree(
                output, root, "unit", "dummy-job-A-attempt-0", mirror_prefixes=("identity",)
            )
            if repeated != first or fs.puts != puts:
                raise AssertionError("Identical persistence was overwritten or retried")
            (output / "task.log").write_text("second attempt\n")
            second = persistence.persist_tree(
                output, root, "unit", "dummy-job-B-attempt-0", mirror_prefixes=("identity",)
            )
            if second["archive"] == first["archive"] or fs.cat_file(legacy) != b"old archive must survive":
                raise AssertionError("Attempt snapshots collided with an old archive")
            _, archive_key = local_store(first["archive"])
            with tarfile.open(fileobj=io.BytesIO(fs.cat_file(archive_key)), mode="r:gz") as archive:
                if archive.extractfile("identity/generate/generations.jsonl").read() != data.read_bytes():
                    raise AssertionError("Snapshot lost original output bytes")
            conflict = Path(directory) / "conflict"
            conflict.write_bytes(b"different")
            alias = root + "/identity/generate/generations.jsonl"
            try:
                persistence.upload_immutable(conflict, alias)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Different artifact bytes were allowed to overwrite")
            _, key = local_store(alias)
            if fs.cat_file(key) != data.read_bytes():
                raise AssertionError("Conflict changed admitted raw bytes")
            try:
                candidate_module.require_fresh_namespace(value)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Populated owner namespace was accepted")
    bad = copy.deepcopy(value)
    bad["math"][0]["output"] = "s3://marin-rno2a/tmp/ttl=14d/old-wrong/math/greedy"
    bad["math"][0]["argv"][-1] = bad["math"][0]["output"]
    try:
        candidate_module.validate_candidate(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("An old/mismatched math output namespace was accepted")
    return {
        "attempt_archives_distinct": True,
        "identical_reuse_without_put": True,
        "different_bytes_protected": True,
        "old_archive_preserved": True,
        "early_namespace_checks": "PASS",
        "object_store_calls": 0,
    }
