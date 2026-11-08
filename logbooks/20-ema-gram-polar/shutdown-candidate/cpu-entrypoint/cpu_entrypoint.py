# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unexecuted CPU diagnostic entrypoint draft; imports services only on invocation.

No submitter or executable CLI is supplied. Calling run_diagnostic with real
imports requires separate explicit authorization for a qualification job.
"""

import atexit
import hashlib
import importlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from trace_contract import Identity, Trace

BOUND_MODULES = {
    "levanter.tracker.wandb": "lib/levanter/src/levanter/tracker/wandb.py",
    "levanter.tracker.background": "lib/levanter/src/levanter/tracker/background.py",
    "levanter.tracker.finalization": "lib/levanter/src/levanter/tracker/finalization.py",
    "levanter.tracker.tracker": "lib/levanter/src/levanter/tracker/tracker.py",
    "levanter.distributed": "lib/levanter/src/levanter/distributed.py",
    "levanter.distributed_finalization": "lib/levanter/src/levanter/distributed_finalization.py",
    "iris.jax.init": "lib/iris/src/iris/jax/init.py",
    "iris.cluster.client.job_info": "lib/iris/src/iris/cluster/client/job_info.py",
}


@dataclass(frozen=True)
class DiagnosticSpec:
    campaign: str
    scenario: str
    candidate_commit: str
    candidate_root: Path
    source_hashes: dict[str, str]
    project: str
    entity: str
    health_timeout_ms: int
    failure_rank: int = 0
    sentinel_count: int = 4


def validate_source_bindings(spec: DiagnosticSpec):
    """Check declared candidate files without importing their packages."""
    if not set(BOUND_MODULES.values()).issubset(spec.source_hashes):
        raise ValueError("candidate module bindings are incomplete")
    for relative, expected in spec.source_hashes.items():
        path = (spec.candidate_root / relative).resolve()
        if not path.is_relative_to(spec.candidate_root.resolve()):
            raise ValueError("source binding escapes candidate root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("candidate source hash mismatch")


def run_diagnostic(spec: DiagnosticSpec, directory: Path, *, importer=importlib.import_module):
    """Exercise online tracker drain and the installed finalizer, without training."""
    if spec.scenario not in ("clean", "application_failure"):
        raise ValueError("unsupported first-stage scenario")
    if spec.health_timeout_ms <= 0 or not spec.source_hashes:
        raise ValueError("explicit health budget and source bindings are required")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("CPU backend must be selected before imports")
    if "IRIS_MULTIGPU_PROCESS_COUNT" in os.environ:
        raise ValueError("first stage requires one process per Iris task")
    if any(name == "jax" or name.startswith("jax.") for name in sys.modules):
        raise RuntimeError("diagnostic requires a fresh process without imported JAX")
    validate_source_bindings(spec)
    identity = {key: os.environ[key] for key in ("IRIS_TASK_ID", "IRIS_ATTEMPT_UID", "IRIS_WORKER_ID")}
    directory.mkdir(exist_ok=False)
    stream = (directory / "events.jsonl").open("x")
    atexit.register(stream.close)
    run_id = uuid.uuid4().hex
    identity.update(
        campaign=spec.campaign, scenario=spec.scenario, candidate_commit=spec.candidate_commit, online_run_id=run_id
    )
    sequence = 0

    def record(event, fields):
        nonlocal sequence
        stream.write(
            json.dumps(
                {
                    "identity": identity,
                    "seq": sequence,
                    "monotonic_ns": time.monotonic_ns(),
                    "event": event,
                    "fields": fields,
                }
            )
            + "\n"
        )
        stream.flush()
        sequence += 1

    def bound_import(name):
        module = importer(name)
        expected_path = (spec.candidate_root / BOUND_MODULES[name]).resolve()
        if Path(module.__file__).resolve() != expected_path:
            raise RuntimeError("loaded module origin differs from candidate binding")
        digest = hashlib.sha256(expected_path.read_bytes()).hexdigest()
        if digest != spec.source_hashes[BOUND_MODULES[name]]:
            raise RuntimeError("loaded module source changed after preflight")
        record("module_bound", {"module": name, "sha256": digest})
        return module

    run = None
    tracker = None
    report = None
    original = None
    world_ready = False
    distributed = None
    observation_type = None
    finalization_trace = None
    required_sinks_complete = None
    try:
        record("online_init_enter", {})
        wandb = importer("wandb")
        if wandb.run is not None:
            raise RuntimeError("diagnostic cannot reuse an existing native run")
        run = wandb.init(
            entity=spec.entity,
            project=spec.project,
            id=run_id,
            resume="never",
            mode="online",
            reinit="create_new",
            config=identity,
        )
        if run is None or run.settings.mode != "online" or run.id != run_id:
            raise RuntimeError("native ONLINE initialization did not meet the diagnostic contract")
        record("online_init_return", {"run_url": run.url})
        # Optional runtime imports occur only after native ONLINE initialization.
        backend = bound_import("levanter.tracker.wandb")
        background = bound_import("levanter.tracker.background")
        distributed = bound_import("levanter.distributed")
        observation_type = bound_import("levanter.distributed_finalization").FinalizationObservation
        tracker = background.BackgroundTracker(backend.WandbTracker(run=run, suppress_logging=False))
        iris_jax = bound_import("iris.jax.init")
        required_sinks_complete = bound_import("levanter.tracker.finalization").required_sinks_complete
        bound_import("levanter.tracker.tracker")
        info = bound_import("iris.cluster.client.job_info").get_job_info()
        if (
            info is None
            or info.task_attempt.to_wire() != identity["IRIS_TASK_ID"]
            or info.attempt_uid != identity["IRIS_ATTEMPT_UID"]
            or info.worker_id != identity["IRIS_WORKER_ID"]
            or info.num_tasks != 4
        ):
            raise RuntimeError("Iris task identity differs from diagnostic binding")
        jax = importer("jax")
        if jax.distributed.is_initialized():
            raise RuntimeError("diagnostic cannot reuse an initialized distributed world")
        iris_jax.initialize_jax()

        if not jax.distributed.is_initialized():
            raise RuntimeError("Iris distributed world was not initialized")
        world_ready = True
        rank, count = jax.process_index(), jax.process_count()
        if count != 4 or rank != info.task_index:
            raise RuntimeError("first-stage diagnostic requires four processes")
        record(
            "world_ready",
            {"rank": rank, "ranks": list(range(count)), "job_id": info.job_id.to_wire(), "attempt_id": info.attempt_id},
        )
        execution_id = info.job_id.to_wire()
        finalization_trace = Trace(
            directory / "finalization.jsonl",
            Identity(
                campaign=spec.campaign,
                scenario=spec.scenario,
                candidate_commit=spec.candidate_commit,
                execution_id=execution_id,
                attempt_uid=info.attempt_uid,
                task_uid=info.task_attempt.to_wire(),
                worker_id=info.worker_id,
                rank=rank,
                ranks=tuple(range(count)),
                online_run_id=run_id,
            ),
        )
        atexit.register(finalization_trace.close)
        run.config.update({"jax_rank": rank, "jax_process_count": count, "execution_id": execution_id})
        for step in range(spec.sentinel_count):
            tracker.log({"diagnostic/sentinel": step, "diagnostic/rank": rank}, step=step, commit=True)
        record("producers_quiesced", {"sentinels_enqueued": spec.sentinel_count})
        if spec.scenario == "application_failure" and rank == spec.failure_rank:
            record("application_failure_injected", {"exit_code": 23, "rank": rank})
            raise SystemExit(23)
    except BaseException as error:
        original = error
        raise
    finally:
        try:
            if tracker is not None:
                report = tracker.finish_with_report()
                record("tracker_report", asdict(report))
                # This diagnostic publishes W&B on every rank, unlike training.
                if required_sinks_complete is None or not required_sinks_complete(report, frozenset({"wandb"})):
                    raise RuntimeError("required diagnostic tracker did not complete locally")
            elif run is not None:
                run.finish()
        except BaseException as cleanup:
            if original is None:
                original = cleanup
                raise
            original.add_note(f"Diagnostic cleanup also failed: {type(cleanup).__name__}")
        finally:
            if world_ready:
                try:
                    distributed.install_coordinated_iris_finalizer(
                        report,
                        application_error=original,
                        health_timeout_ms=spec.health_timeout_ms,
                        observation=observation_type(finalization_trace.observe if finalization_trace else record),
                    )
                except BaseException as install_error:
                    if original is None:
                        raise
                    original.add_note(f"Diagnostic hook installation also failed: {type(install_error).__name__}")
