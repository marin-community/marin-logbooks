# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""49h CPU lifetime owner, using frozen9e serving/generation and a distinctf5 bundle."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from iris.client.client import IrisClient, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.resources.state import JobState, is_job_finished
from iris.rpc import job_pb2
from marin.inference import iris_cli as frozen_serve
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Deadline, Duration

from experiments.post_training.identity_fullscreen_owner.candidate import (
    bundled_candidate,
    candidate_payload,
    load_candidate,
    require_fresh_namespace,
)
from experiments.post_training.identity_fullscreen_owner.persistence import native_execution_id, persist_tree

OWNER_SECONDS = 49 * 3600
CHILD_SECONDS = 12 * 3600
READINESS_SECONDS = 8 * 3600
MATH_COORDINATOR_SECONDS = 26 * 3600


class ServeSubmitted(Exception):
    """Stop the CLI after the real native submit, before endpoint/capability printing."""


def record(output, name, value):
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")


def regional_context():
    info = get_job_info()
    context = iris_ctx()
    if info is None or info.controller_address != os.environ.get("IRIS_CONTROLLER_URL"):
        raise RuntimeError("Assigned native in-cluster controller missing or inconsistent")
    if context.client is None or context.job_id != info.job_id or not info.bundle_id:
        raise RuntimeError("Expected the native9e owner bundle/context")
    return info, context


def submit_serve(candidate, info):
    """Let the unchanged9e CLI build its exact service; intercept only native submission."""
    original_submit = IrisClient.submit
    submitted = []

    @contextlib.contextmanager
    def assigned_connection(**unused):
        yield SimpleNamespace(url=info.controller_address, credentials=info.credentials, config=None)

    def assigned_client(address, **unused):
        if address != info.controller_address:
            raise ValueError("Serve attempted an unassigned controller")
        return IrisClient.in_cluster(address, bundle_id=info.bundle_id)

    def submit(client, *args, **kwargs):
        kwargs.update(
            max_retries_failure=0,
            max_retries_preemption=0,
            timeout=Duration.from_seconds(CHILD_SECONDS),
            priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            constraints=None,
        )
        job = original_submit(client, *args, **kwargs)
        submitted.append(str(job.job_id))
        raise ServeSubmitted

    with (
        patch.object(frozen_serve, "connect_controller", assigned_connection),
        patch.object(frozen_serve, "IrisClient", SimpleNamespace(remote=assigned_client)),
        patch.object(IrisClient, "submit", submit),
    ):
        # No stdout/capability from the submit-only CLI becomes a public receipt.
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                frozen_serve.main.main(args=candidate["screen"]["serve_argv"][2:], standalone_mode=False)
            except ServeSubmitted:
                pass
    if len(submitted) != 1:
        raise RuntimeError("Expected exactly one owned serve submission")
    return iris_ctx().client.job(JobName.from_wire(submitted[0]))


def await_terminal(job, output, name):
    """Observe only one owned handle; transient observation is not a scientific retry."""
    errors = []
    while True:
        try:
            status = job.wait(timeout=OWNER_SECONDS, raise_on_failure=False)
            record(
                output,
                name,
                {
                    "job": str(job.job_id),
                    "state": str(status.state),
                    "observation_errors": errors,
                },
            )
            return status
        except Exception as error:
            errors.append(type(error).__name__)
            record(
                output,
                name,
                {
                    "job": str(job.job_id),
                    "coverage": "observation_failed_still_owned",
                    "observation_errors": errors,
                },
            )
            time.sleep(2)


def await_endpoint(client, serve, endpoint):
    deadline = Deadline.from_seconds(READINESS_SECONDS)
    while True:
        deadline.raise_if_expired("Owned screen serve readiness exceeded existing8h window")
        if is_job_finished(serve.status().state):
            raise RuntimeError("Owned serve terminated before readiness")
        if client.list_endpoint_instances(endpoint):
            return client.mint_endpoint_token(endpoint, ttl=Duration.from_seconds(CHILD_SECONDS)).capability_url.rstrip(
                "/"
            )
        time.sleep(min(2, deadline.remaining_seconds()))


def math_workspace(archive, expected_sha):
    raw = Path(archive).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Pinnedf5 source archive changed")
    destination = Path(tempfile.mkdtemp(prefix=f"pathb-f5-source-{expected_sha[:16]}-"))
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        for name in bundle.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid native source archive member")
        bundle.extractall(destination)
    driver = destination / "experiments/weight_merging/rsi/identity_297/evaluate_math.py"
    if (
        hashlib.sha256(driver.read_bytes()).hexdigest()
        != "4dd4560b4c04b7b5de0e5a9a77c14512b92c955e91febb6f66d4c935593c478d"
    ):
        raise ValueError("Math workspace does not contain the pinnedf5 scientific driver")
    # The f5 archive intentionally predates this operational package. Copy the
    # committed package before the native f5 bundle is constructed; imports
    # never depend on optional small inline workdir-file delivery.
    package = destination / "experiments/post_training/identity_fullscreen_owner"
    shutil.copytree(Path(__file__).resolve().parent, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return destination


def task_entrypoint(candidate: dict, kind, trial: str | None = None):
    base = Path(os.environ["IRIS_WORKDIR"])
    path, candidate_sha = bundled_candidate(candidate)
    child_args = ["--candidate", path, "--candidate-sha256", candidate_sha, "--kind", kind]
    if trial is not None:
        child_args.extend(["--trial", trial])
    command = [
        "bash",
        "-c",
        'cd "$IRIS_WORKDIR"; '
        'export PYTHONPATH="$IRIS_WORKDIR:$IRIS_WORKDIR/_pathb_owner:$IRIS_WORKDIR/_pathb_fullscreen:'
        '$IRIS_WORKDIR/experiments/post_training/identity_fullscreen_owner/frozen${PYTHONPATH:+:$PYTHONPATH}"; '
        'exec "$IRIS_VENV/bin/python" -u -m experiments.post_training.identity_fullscreen_owner.persist_task '
        + shlex.join(child_args),
    ]
    files = {}
    for directory in ("_pathb_owner", "_pathb_fullscreen"):
        for path in (base / directory).rglob("*"):
            if (
                path.is_file()
                and path.suffix == ".jsonl"
                and "outputs" not in path.parts
                and "__pycache__" not in path.parts
            ):
                files[path.relative_to(base).as_posix()] = path.read_bytes()
    return Entrypoint(command=command, workdir_files=files)


def submit_cpu(client, candidate, kind, endpoint, trial: str | None = None):
    if kind == "generation":
        name = "generation"
    else:
        if trial is None:
            raise ValueError("Math submission requires a bound trial")
        name = "math-" + trial
    return client.submit(
        entrypoint=task_entrypoint(candidate, kind, trial),
        name=name,
        resources=ResourceSpec(cpu=8, memory="32g", disk="50g"),
        environment=EnvironmentSpec(
            extras=("cpu",),
            secrets={"IDENTITY_ENDPOINT": endpoint},
            env_vars={
                "JAX_PLATFORMS": "cpu",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        ),
        timeout=Duration.from_seconds(CHILD_SECONDS if kind == "generation" else MATH_COORDINATOR_SECONDS),
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        max_retries_failure=0,
        max_retries_preemption=0,
    )


def run(candidate, archive, archive_sha):
    configure_coreweave_s3()
    require_fresh_namespace(candidate)
    info, context = regional_context()
    output = Path(tempfile.mkdtemp(prefix="pathb-screen-owner-control-"))
    owned = []
    serve = None
    math_client = None
    outcomes = {}
    error = None
    persistence_error = None
    record(output, "candidate", {"sha256": candidate_payload(candidate)[2], "candidate": candidate})
    try:
        workspace = math_workspace(archive, archive_sha)
        math_client = IrisClient.in_cluster(info.controller_address, workspace=workspace)
        record(
            output,
            "source-bundles",
            {
                "screen_bundle_id": info.bundle_id,
                "screen_source_pin": "9e3857f86b168605b063e4b8f5aedb34ad10f01c",
                "math_source_pin": "f5ef16c8bc4030cffbb3c10636495c9fd6100bcf",
                "math_archive_sha256": archive_sha,
                "math_source_workspace": str(workspace),
            },
        )
        serve = submit_serve(candidate, info)
        record(output, "serve", {"job": str(serve.job_id), "candidate": candidate["label"]})
        endpoint_name = candidate["screen"]["serve_argv"][-1]
        endpoint = await_endpoint(context.client, serve, endpoint_name)
        if candidate.get("scope") != "math-only":
            generation = submit_cpu(context.client, candidate, "generation", endpoint)
            owned.append(("generation", generation))
            outcomes["generation"] = str(await_terminal(generation, output, "generation").state)
        for trial in ("greedy", "17", "29"):
            try:
                job = submit_cpu(math_client, candidate, "math", endpoint, trial)
                owned.append(("math-" + trial, job))
            except Exception as failure:
                outcomes["math-" + trial] = "submit_failed:" + type(failure).__name__
                record(
                    output,
                    "math-" + trial,
                    {
                        "coverage": "submit_failed",
                        "error_type": type(failure).__name__,
                        "retries": 0,
                    },
                )
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                (name, pool.submit(await_terminal, job, output, name)) for name, job in owned if name.startswith("math-")
            ]
            for name, future in futures:
                outcomes[name] = str(future.result().state)
    except BaseException as failure:
        error = failure
        record(output, "owner-error", {"error_type": type(failure).__name__})
    finally:
        # Drain every actually submitted evaluator before releasing its serve,
        # including after a later submit/receipt fails. No resubmit or cancel.
        for name, job in owned:
            outcomes[name] = str(await_terminal(job, output, name).state)
        if serve is not None:
            try:
                serve.cancel()
            except Exception as failure:
                record(
                    output,
                    "serve-release-error",
                    {
                        "job": str(serve.job_id),
                        "error_type": type(failure).__name__,
                        "cancel_attempts": 1,
                    },
                )
            await_terminal(serve, output, "serve-terminal")
        if math_client is not None:
            math_client.shutdown()
        record(
            output,
            "owner-result",
            {
                "candidate": candidate["label"],
                "outcomes": outcomes,
                "scientific_command_success": (
                    len(outcomes) == (3 if candidate.get("scope") == "math-only" else 4)
                    and all(s == str(JobState.SUCCEEDED) for s in outcomes.values())
                ),
                "no_quality_verdict": True,
            },
        )
        try:
            persist_tree(output, candidate["artifact_root"] + "/owner", "owner-execution", native_execution_id())
        except Exception as failure:
            persistence_error = failure
            print(f"Owner persistence failed: {type(failure).__name__}: {failure}", file=sys.stderr)
    if error is not None:
        raise error from persistence_error
    if len(outcomes) != (3 if candidate.get("scope") == "math-only" else 4) or not all(
        s == str(JobState.SUCCEEDED) for s in outcomes.values()
    ):
        raise RuntimeError(
            "Full screen has incomplete command coverage; available outputs retained"
        ) from persistence_error
    if persistence_error is not None:
        raise persistence_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--math-archive", type=Path, required=True)
    parser.add_argument("--math-archive-sha256", required=True)
    args = parser.parse_args()
    run(
        load_candidate(args.candidate, args.candidate_sha256),
        args.math_archive,
        args.math_archive_sha256,
    )


if __name__ == "__main__":
    main()
