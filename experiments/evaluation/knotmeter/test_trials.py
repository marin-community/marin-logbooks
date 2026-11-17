# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise retained attempts, check scope and reports through public boundaries."""

import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from iris.cluster.types import JobName
from iris.resources.state import JobState, TaskState
from marin.external_dependencies import VLLM_GPU_RELEASE
from rigging.provenance import Provenance
from rigging.timing import Timestamp

from experiments.evaluation.knotmeter.artifacts import read_record, retain_artifacts, write_record
from experiments.evaluation.knotmeter.cli import resolve_plan
from experiments.evaluation.knotmeter.collect import collect_trial
from experiments.evaluation.knotmeter.config import Check, Runtime, load_profile, resolved_series
from experiments.evaluation.knotmeter.report import report_campaign
from experiments.evaluation.knotmeter.run import run_checks
from experiments.evaluation.knotmeter.runtime import resolve_release


def test_launch_plan_survives_json_delivery_without_invalidating_worker_identity(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.knotmeter.cli.pin_profile", lambda profile: profile)
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.cli.launch_provenance",
        lambda: Provenance("fixture", "b" * 40, False, "fixture", None, command_line=("knotmeter", "run")),
    )
    plan = resolve_plan(load_profile(Path("snowball-h100"), []), "fixture")
    assert json.loads(json.dumps(plan)) == plan


@pytest.mark.parametrize("ambient", ["local_store", "gs://wrong-cloud/outputs"])
def test_plan_uses_target_storage_unless_explicitly_overridden(monkeypatch, ambient):
    monkeypatch.setattr("experiments.evaluation.knotmeter.cli.pin_profile", lambda profile: profile)
    monkeypatch.setenv("MARIN_PREFIX", ambient)
    profile = load_profile(Path("snowball-h100"), [])
    plan = resolve_plan(profile, "fixture")
    assert plan["marin_prefix"].startswith("s3://") and plan["marin_prefix"] != ambient
    assert (
        resolve_plan(profile, "fixture", marin_prefix="s3://fixture/outputs")["marin_prefix"] == "s3://fixture/outputs"
    )
    with pytest.raises(ValueError, match="shared"):
        resolve_plan(profile, "fixture", marin_prefix="local_store")


def test_unregistered_cuda_build_and_list_index_override_fail_before_submission(tmp_path):
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps(dataclasses.asdict(VLLM_GPU_RELEASE) | {"torch_backend": "cu999"}))
    with pytest.raises(ValueError, match="registered CUDA"):
        resolve_release(Runtime(manifest=str(manifest)))
    with pytest.raises(ValueError, match="whole value"):
        load_profile(Path("snowball-h100"), ["checks.0.command=[true]"])


def trial(root: Path, name: str, rate: float, *, diagnostic: bool = False) -> dict:
    profile = load_profile(Path("snowball-h100"), [])
    series = resolved_series(profile, diagnostic_seconds=3 if diagnostic else None)
    runtime = json.loads(json.dumps(dataclasses.asdict(VLLM_GPU_RELEASE))) | {
        "release_tag": "fixture",
        "source_commit": "b" * 40,
    }
    return {
        "trial_id": name,
        "trial_uri": str(root / "trials" / name),
        "manifest": {
            "series": series,
            "diagnostic": diagnostic,
            "vllm_release": runtime,
            "workload": {"recipe": {"fixture": True}},
        },
        "job_state": JobState.SUCCEEDED.value,
        "job_succeeded": True,
        "attempts": [],
        "gpu_seconds": 3600.0,
        "cost_complete": True,
        "measurement": {
            "validity": {"submission_valid": not diagnostic, "protocol_valid": not diagnostic, "gang_valid": True},
            "checks": [
                {
                    "name": "synthetic-fixture-check",
                    "kind": "numerical",
                    "checkpoint_revision": profile.model.revision,
                    "runtime_source": "b" * 40,
                    "status": "passed",
                }
            ],
            "aggregate": {
                "request_metrics": {
                    "throughput": {"per_gpu": {"output_tput_tps": rate}},
                    "latency": {"ttft": {"p90": 5.0}, "e2el": {"p90": 35.0}},
                }
            },
        },
    }


def test_report_keeps_diagnostics_separate_and_pins_selected_runtime(tmp_path, monkeypatch):
    for result in (
        trial(tmp_path, "baseline", 100),
        trial(tmp_path, "candidate", 120),
        trial(tmp_path, "diagnostic", 9999, diagnostic=True),
    ):
        write_record(f"{result['trial_uri']}/result.json", result)
    output = report_campaign(str(tmp_path), recommendation="candidate", baseline="baseline")
    assert output == (tmp_path / "summary.md").read_text()
    decision = read_record(str(tmp_path / "decision.json"))
    assert decision["throughput_change"] == pytest.approx(0.2, abs=1e-12)
    assert decision["recommended_trial"] == "candidate"
    # A future change to Marin's selection must not retarget the saved recommendation.
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.runtime.VLLM_GPU_RELEASE",
        dataclasses.replace(VLLM_GPU_RELEASE, source_commit="c" * 40),
    )
    recommended = load_profile(tmp_path / "recommended.yaml", [])
    assert resolve_release(recommended.vllm)["source_commit"] == "b" * 40
    with pytest.raises(ValueError):
        report_campaign(str(tmp_path), recommendation="diagnostic", baseline="baseline")


@pytest.mark.parametrize("scope", ["matching", "other-checkpoint", "other-runtime", "failed"])
def test_existing_check_pass_transfers_only_to_its_checkpoint_and_runtime(tmp_path, scope):
    base = load_profile(Path("snowball-h100"), [])
    check = Check(
        name="numerical-fixture",
        command=[sys.executable, "-c", f"raise SystemExit({7 if scope == 'failed' else 0})"],
        checkpoint_revision="a" * 40 if scope == "other-checkpoint" else base.model.revision,
        runtime_source="c" * 40 if scope == "other-runtime" else "b" * 40,
    )
    profile = base.model_copy(update={"checks": [check]})
    checks = run_checks(profile, SimpleNamespace(worker_server_url="http://fixture/v1"), tmp_path, "b" * 40)
    result = trial(tmp_path, "candidate", 120)
    result["measurement"]["checks"] = checks
    write_record(f"{result['trial_uri']}/result.json", result)
    if scope == "matching":
        report_campaign(str(tmp_path), recommendation="candidate")
        assert (tmp_path / "recommended.yaml").is_file()
    else:
        if scope != "failed":
            assert checks[0]["regression_status"] == "passed"
            assert checks[0]["status"] == "pending"
        with pytest.raises(ValueError):
            report_campaign(str(tmp_path), recommendation="candidate")
        assert not (tmp_path / "recommended.yaml").exists()
        assert read_record(f"{result['trial_uri']}/result.json")["measurement"]["checks"] == checks


@pytest.mark.parametrize("name", [None, "smoke", "model-numerics", "other-checkpoint", "other-runtime"])
def test_missing_or_relabelled_numerical_check_cannot_qualify_a_trial(tmp_path, name):
    result = trial(tmp_path, "candidate", 120)
    if name in ("other-checkpoint", "other-runtime"):
        field = "checkpoint_revision" if name == "other-checkpoint" else "runtime_source"
        result["measurement"]["checks"][0][field] = "wrong-scope"
    else:
        result["measurement"]["checks"] = [] if name is None else [{"name": name, "kind": "serving", "status": "passed"}]
    write_record(f"{result['trial_uri']}/result.json", result)
    with pytest.raises(ValueError, match="passed required checks"):
        report_campaign(str(tmp_path), recommendation="candidate")
    assert not (tmp_path / "recommended.yaml").exists()


def test_plan_without_result_remains_visible_with_unknown_cost(tmp_path):
    result = trial(tmp_path, "interrupted", 0)
    write_record(
        f"{result['trial_uri']}/plan.json",
        {"trial_id": "interrupted", "trial_uri": result["trial_uri"], "manifest": result["manifest"]},
    )
    output = report_campaign(str(tmp_path))
    assert "interrupted" in output and "unrecorded" in output
    assert "interrupted/plan.json" in output
    assert "Total cost is incomplete" in output
    assert output == (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("partial_exports", [False, True])
def test_recovery_counts_setup_and_failed_native_attempts_without_resubmitting(tmp_path, partial_exports):
    result = trial(tmp_path, "failed", 0)
    plan = {
        "trial_id": "failed",
        "trial_uri": result["trial_uri"],
        "gpu_job_id": "/fixture/trial/serve",
        "manifest": result["manifest"],
    }
    task_id = JobName.from_string("/fixture/trial/serve/0")
    attempts = []
    for number, uid, start, finish, scheduled, stopped in ((0, "one", 20, 50, 10, 45), (1, "two", 80, 100, 70, 95)):
        attempts.append(
            SimpleNamespace(
                attempt_number=number,
                attempt_uid=uid,
                started_at=Timestamp.from_seconds(start),
                finished_at=Timestamp.from_seconds(finish),
                state=TaskState.FAILED,
                pod_uid=f"pod-{uid}",
                node_name="fixture-node",
                error_message="fixture failure",
                terminal_reason="",
                output_archive=None,
            )
        )
        write_record(
            f"{plan['trial_uri']}/attempts/{uid}/allocation.json",
            {
                "pod_uid": f"pod-{uid}",
                "scheduled_unix": scheduled,
                "finished_unix": stopped,
            },
        )
        if partial_exports:
            point = tmp_path / f"worker-{uid}"
            (point / "aiperf_artifacts").mkdir(parents=True)
            (point / "aiperf_artifacts/profile_export.jsonl").write_text('{"partial":true}\n')
            record = {"manifest": plan["manifest"], "iris_attempt_uid": uid, "status": "failed"}
            write_record(str(point / "attempt.json"), record)
            prefix = f"{plan['trial_uri']}/attempts/{uid}/hosts/0"
            uri, digest = retain_artifacts(point, tmp_path / f"{uid}.tar.zst", prefix, name="native")
            write_record(f"{prefix}/receipt.json", record | {"archive_uri": uri, "archive_sha256": digest})
    task = SimpleNamespace(task_id=task_id, attempts=attempts, current_attempt_number=1)

    class Client:
        def job_status(self, _job):
            return SimpleNamespace(
                job_id=JobName.from_string(plan["gpu_job_id"]),
                state=JobState.FAILED,
                execution_cluster_id="fixture-cluster",
            )

        def list_tasks(self, _job):
            return [task]

        def task_status(self, _task):
            return task

        def submit(self, **_kwargs):
            raise AssertionError("Recovery must not submit compute")

    recovered = collect_trial(Client(), plan, tmp_path / "recovery")
    assert recovered["gpu_seconds"] == 480
    assert recovered["cost_complete"] is True
    assert [row["attempt_uid"] for row in recovered["attempts"]] == ["one", "two"]
    if partial_exports:
        assert all("Incomplete AgentX exports" in row["measurement_error"] for row in recovered["attempts"])
        assert recovered["measurement"] is None
    before = copy.deepcopy(read_record(f"{plan['trial_uri']}/result.json"))
    collect_trial(Client(), plan, tmp_path / "repeated-recovery")
    assert read_record(f"{plan['trial_uri']}/result.json") == before
