# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan, run, recover and compare manual inference trials through Iris."""

import argparse
import dataclasses
import hashlib
import json
import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS, connect_controller
from iris.client.client import IrisClient
from iris.cluster.client import resolve_job_user
from iris.cluster.constraints import Constraint, ConstraintOp, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.resources.state import is_job_finished
from iris.rpc import job_pb2
from marin.inference.vllm_release import vllm_gpu_wheel_for_architecture
from rigging.config_discovery import resolve_cluster_config
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from rigging.provenance import launch_provenance
from rigging.timing import Duration

from experiments.evaluation.knotmeter.allocation import capture_allocations, track_allocations
from experiments.evaluation.knotmeter.artifacts import read_record, write_record
from experiments.evaluation.knotmeter.collect import collect_trial, record_preparation_failure, refresh_costs
from experiments.evaluation.knotmeter.config import Profile, load_profile, resolved_series
from experiments.evaluation.knotmeter.coordinator import run_coordinator
from experiments.evaluation.knotmeter.recipe import TrialPlan
from experiments.evaluation.knotmeter.report import report_campaign
from experiments.evaluation.knotmeter.runtime import gpu_release, resolve_release


def pin_profile(profile: Profile) -> Profile:
    """Resolve Hub revisions and verify small export manifests before allocating compute."""
    model = profile.model
    api = HfApi()
    revision = model.revision
    if model.path:
        for name, expected in (("export-manifest.json", model.revision), ("config.json", model.config_sha256)):
            filesystem, path = filesystem_for(prefix_join(model.path, name))
            with filesystem.open(path, "rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Checkpoint identity differs: {name}")
    else:
        revision = api.model_info(model.repo, revision=model.revision).sha
    tokenizer = api.model_info(model.tokenizer, revision=model.tokenizer_revision).sha
    if revision is None or tokenizer is None:
        raise ValueError("Model and tokenizer must resolve to exact revisions")
    return profile.model_copy(
        update={
            "model": model.model_copy(
                update={
                    "revision": revision,
                    "tokenizer_revision": tokenizer,
                }
            )
        }
    )


def resolve_plan(
    profile: Profile,
    campaign: str,
    *,
    diagnostic_seconds: int | None = None,
    measure_failed_checks: bool = False,
    marin_prefix: str | None = None,
    placement: dict | None = None,
    controller: str | None = None,
    logbook: str | None = None,
) -> TrialPlan:
    """Resolve a self-contained trial before submission; this function allocates no compute."""
    profile = pin_profile(profile)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", campaign) is None:
        raise ValueError("Campaign must be one path segment containing letters, numbers, underscores, dots or hyphens")
    config_path = resolve_cluster_config(profile.cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS)
    config = yaml.safe_load(Path(config_path).read_text())
    prefix = marin_prefix or config["defaults"]["task_env"]["MARIN_PREFIX"]
    if not prefix.startswith(("s3://", "gs://")):
        raise ValueError("Iris trials require a shared s3:// or gs:// output prefix")
    if "hero-checkpoints" in prefix:
        raise ValueError("Campaign outputs must use the configured MARIN_PREFIX, outside hero-checkpoints")
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trial = f"{now}-{uuid.uuid4().hex[:12]}"
    campaign_uri = prefix_join(prefix, f"knotmeter/campaigns/{campaign}")
    trial_uri = prefix_join(campaign_uri, f"trials/{trial}")
    job_id = JobName.root(resolve_job_user(None), f"knotmeter-{trial.lower()}")
    release = resolve_release(profile.vllm)
    architecture = {"H100": "x86_64", "GB200": "aarch64"}[profile.gpus.type]
    vllm_gpu_wheel_for_architecture(gpu_release(release), architecture)
    return {
        "schema_version": 1,
        "trial_id": trial,
        "trial_uri": trial_uri,
        "campaign_uri": campaign_uri,
        "controller": controller or profile.cluster,
        "cluster": profile.cluster,
        "marin_prefix": prefix,
        "series": resolved_series(profile, diagnostic_seconds, measure_failed_checks=measure_failed_checks),
        "vllm_release": release,
        "placement": placement or {},
        "job_id": str(job_id),
        "gpu_job_id": str(job_id.child("serve")),
        "source": json.loads(json.dumps(dataclasses.asdict(launch_provenance()))),
        "research_logbook": logbook,
    }


def submit_coordinator(client: IrisClient, plan: TrialPlan):
    profile = Profile.model_validate(plan["series"]["profile"])
    series = plan["series"]
    return client.submit(
        entrypoint=Entrypoint.from_callable(run_coordinator, plan),
        name=JobName.from_string(plan["job_id"]).name,
        resources=ResourceSpec(cpu=2, memory="32GB", disk="64GB"),
        environment=EnvironmentSpec(
            sync_packages=["marin-core"], extras=["cpu"], env_vars={"MARIN_PREFIX": plan["marin_prefix"]}
        ),
        constraints=[
            *(
                [Constraint.create(key="cluster", op=ConstraintOp.EQ, value=plan["cluster"])]
                if plan["controller"] != plan["cluster"]
                else []
            ),
            preemptible_constraint(False),
        ],
        priority_band={"batch": job_pb2.PRIORITY_BAND_BATCH, "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE}[
            profile.priority
        ],
        scheduling_timeout=Duration.from_seconds(series["scheduling_timeout"]),
        timeout=Duration.from_seconds(
            series["scheduling_timeout"] + series["task_timeout"] * 3 + series["prepare_timeout"]
        ),
        max_retries_failure=0,
        max_retries_preemption=0,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
        task_image=series["task_image"],
    )


def print_plan(plan: TrialPlan) -> None:
    profile = plan["series"]["profile"]
    print(
        yaml.safe_dump(
            {
                "model": profile["model"],
                "gpus": profile["gpus"],
                "cluster": plan["cluster"],
                "priority": profile["priority"],
                "sessions": profile["session_concurrency"],
                "workload": profile["workload"],
                "duration_seconds": plan["series"]["duration"],
                "vllm_source": plan["vllm_release"]["source_commit"],
                "serving": {
                    key: value for key, value in profile["serving"].items() if key not in {"extra_args", "environment"}
                },
                "planned_outputs": plan["trial_uri"],
            },
            sort_keys=False,
        )
    )
    pending = [check["name"] for check in profile["checks"] if not check["command"] and not check["fixture"]]
    if pending or not profile["checks"]:
        print(
            f"Required check commands remain pending: {', '.join(pending) or 'none configured'}. "
            "Accepted recommendations are blocked."
        )
    if plan["series"].get("diagnostic"):
        print("Diagnostic replay: this duration cannot establish an AgentX comparison.")
    elif plan["series"].get("measure_failed_checks"):
        print("Failed checks will be retained while measuring the native protocol; they still block recommendations.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("plan", "run"):
        command = commands.add_parser(
            operation, help="Resolve defaults without compute" if operation == "plan" else "Submit one fresh Iris trial"
        )
        command.add_argument("--profile", type=Path, default=Path("snowball-h100"))
        command.add_argument("--campaign", required=True)
        command.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="FIELD=YAML",
            help="Override a dotted profile field",
        )
        command.add_argument("--session-concurrency", type=int)
        command.add_argument("--cluster")
        command.add_argument("--priority", choices=("batch", "interactive"))
        command.add_argument("--vllm-release")
        command.add_argument("--vllm-manifest")
        command.add_argument(
            "--diagnostic-seconds", type=int, help="Use a short unsafe upstream replay; always unqualified"
        )
        command.add_argument(
            "--measure-failed-checks",
            action="store_true",
            help="Continue after retained check failures using the native protocol; failures block recommendations",
        )
        command.add_argument("--placement-uri", help="Optional verified spare-node receipt for a placement guard")
        command.add_argument("--marin-prefix", help="Explicit job MARIN_PREFIX override")
        command.add_argument("--controller", help="Iris entry cluster; defaults to the selected target")
        command.add_argument("--logbook", help="Coordinating research issue or notebook URL retained with this trial")
        command.add_argument("--output", type=Path, help="Save the resolved plan locally")
    commands.choices["run"].add_argument(
        "--no-wait", action="store_true", help="Return after submission; Iris continues the trial"
    )
    for operation in ("status", "recover", "cancel"):
        command = commands.add_parser(
            operation,
            help={
                "status": "Inspect live Iris work",
                "recover": "Collect terminal work without launching GPUs",
                "cancel": "Cancel this trial and its children",
            }[operation],
        )
        command.add_argument("--trial-uri", required=True)
    report = commands.add_parser("report", help="Print and save the campaign comparison")
    report.add_argument("--campaign-uri", required=True)
    report.add_argument("--recommend", help="Select a qualified trial and pin recommended.yaml")
    report.add_argument("--baseline", help="Show changes against this trial's matched model and workload controls")
    args = parser.parse_args()
    if args.operation == "report":
        print(report_campaign(args.campaign_uri, args.recommend, args.baseline), end="")
        return
    if args.operation in ("plan", "run"):
        overrides = args.overrides.copy()
        for key, value in (
            ("session_concurrency", args.session_concurrency),
            ("cluster", args.cluster),
            ("priority", args.priority),
            ("vllm.release", args.vllm_release),
            ("vllm.manifest", args.vllm_manifest),
        ):
            if value is not None:
                overrides.append(f"{key}={json.dumps(value)}")
        profile = load_profile(args.profile, overrides)
        plan = resolve_plan(
            profile,
            args.campaign,
            diagnostic_seconds=args.diagnostic_seconds,
            measure_failed_checks=args.measure_failed_checks,
            marin_prefix=args.marin_prefix,
            placement=read_record(args.placement_uri) if args.placement_uri else None,
            controller=args.controller,
            logbook=args.logbook,
        )
        print_plan(plan)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(plan, indent=2) + "\n")
        if args.operation == "plan":
            return
        write_record(prefix_join(plan["trial_uri"], "plan.json"), plan)
        with connect_controller(cluster_name=plan["controller"]) as endpoint:
            with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
                job = submit_coordinator(client, plan)
                print(
                    f"Iris job: {job.job_id}\nTrial: {plan['trial_uri']}\n"
                    "Reattach: uv run python -m experiments.evaluation.knotmeter.cli status "
                    f"--trial-uri {plan['trial_uri']}\n"
                    "Cancel: uv run python -m experiments.evaluation.knotmeter.cli cancel "
                    f"--trial-uri {plan['trial_uri']}",
                    flush=True,
                )
                if not args.no_wait:
                    with track_allocations(client, plan):
                        job.wait(
                            timeout=plan["series"]["scheduling_timeout"] + plan["series"]["task_timeout"] * 3,
                            raise_on_failure=False,
                            stream_logs=True,
                        )
                    plan = read_record(prefix_join(plan["trial_uri"], "plan.json"))
                    if "manifest" in plan:
                        refresh_costs(plan)
                    else:
                        status = client.job_status(job.job_id)
                        record_preparation_failure(plan, status.error_message)
                    print(report_campaign(plan["campaign_uri"]), end="")
        return
    plan = read_record(prefix_join(args.trial_uri, "plan.json"))
    with connect_controller(cluster_name=plan["controller"]) as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            job_id = JobName.from_string(plan["job_id"])
            if args.operation == "status":
                status = client.job_status(job_id)
                print(
                    f"Iris job {job_id}: {status.state.value}\nGPU job: {plan['gpu_job_id']}\nEvidence: {args.trial_uri}"
                )
                print(
                    "Collect terminal work: uv run python -m experiments.evaluation.knotmeter.cli recover "
                    f"--trial-uri {args.trial_uri}"
                )
            elif args.operation == "cancel":
                client.job(job_id).cancel()
                print(
                    f"Cancellation requested for {job_id} and its children. "
                    "Verify terminal Iris and provider state before retrying."
                )
            else:
                if not is_job_finished(client.job_status(job_id).state):
                    raise ValueError("Coordinator is live; inspect it or explicitly cancel it before recovery")
                if "manifest" not in plan:
                    raise ValueError(
                        "Preparation did not finish; no GPU trial was submitted. "
                        f"See {args.trial_uri}/coordinator-error.json"
                    )
                capture_allocations(client, plan)
                with tempfile.TemporaryDirectory(prefix="knotmeter-recovery-") as directory:
                    collect_trial(client, plan, Path(directory))
                print(report_campaign(plan["campaign_uri"]), end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"error: {error}") from None
