# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one published geometry worker, preserving attempt and bundle provenance."""

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from iris.cluster.client.bundle import collect_workspace_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--experiment", choices=("power", "envelope", "muon", "history"), required=True)
    parser.add_argument("--stage", choices=("screen", "test", "path"), default="screen")
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--tpu", choices=("v4-32", "v5p-8"), default="v4-32")
    parser.add_argument("--region", default="us-central2")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--path-inputs", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    archive = Path("logbooks/20-ema-gram-polar")

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    source, branch = git("rev-parse", "HEAD"), git("branch", "--show-current")
    published = git("ls-remote", "logbooks", "refs/heads/" + branch).split()
    if not published or published[0] != source:
        raise ValueError("Publish the current source on the task branch before submission")
    required = [
        archive / "hessian-aligned-configs.json",
        archive / "geometry-protocol.json",
        archive / "geometry-excluded-sequences.json",
    ]
    if args.experiment == "history":
        required[1] = archive / "geometry-history-protocol.json"
    if args.experiment == "muon":
        required[:2] = [archive / "geometry-muon-configs.json", archive / "geometry-muon-protocol.json"]
        if args.region != "us-central1" or args.tpu != "v5p-8":
            raise ValueError("Use local us-central1 v5p-8 for the pinned Muon checkpoints")
    if args.experiment == "envelope" and args.stage in ("test", "path") and args.plan is None:
        raise ValueError("Freeze and publish coefficients before test/path submission")
    if args.stage == "path" and args.path_inputs is None:
        raise ValueError("Publish validated baseline response references before path submission")
    optional = {}
    for flag, path in (("--plan", args.plan), ("--path-inputs", args.path_inputs)):
        if path is not None:
            relative = path.resolve().relative_to(root) if path.is_absolute() else path
            required.append(relative)
            optional[flag] = relative.as_posix()
    protocol = json.loads((root / required[1]).read_text())
    if args.state not in {s["state"] for s in protocol["states"]}:
        raise ValueError("State is not in the published protocol")
    allowed = "|".join(re.escape(p.as_posix()) for p in required)
    exclusion = re.compile(r"^docs/|^logbooks/(?!(?:" + allowed.replace("logbooks/", "") + r")$)")
    bundle = collect_workspace_files(root, exclude=exclusion)
    relative_bundle = {p.relative_to(root).as_posix() for p in bundle}
    if not {p.as_posix() for p in required} <= relative_bundle:
        raise ValueError("Submission bundle omits a required configuration")
    tracked = set(git("ls-files").splitlines())
    untracked = sorted(relative_bundle - tracked)
    dirty = set(git("diff", "--name-only", "HEAD").splitlines()) & relative_bundle
    if untracked or dirty:
        raise ValueError(f"Bundle contains unpublished files: {untracked + sorted(dirty)}")
    stage = args.experiment + ("-" + args.stage if args.experiment == "envelope" else "")
    stem = f"geometry-{stage}-{args.state}-{args.attempt:02d}"
    job_name = f"ema-gram-geometry-{stage}-{args.state}-20261113-{args.attempt:02d}"
    prefix = f"gs://marin-{args.region}/ema-gram-20-geometry/20261113/" + args.experiment
    if args.experiment == "envelope":
        prefix += "/" + args.stage
    prefix += "/" + args.state
    command = [
        str(root / ".venv/bin/iris"),
        "--cluster=marin",
        "job",
        "run",
        "--no-wait",
        "--no-cancel-on-exit",
        "--priority",
        "batch",
        "--enable-extra-resources",
        "--tpu",
        args.tpu,
        "--cpu",
        "120",
        "--memory",
        "256GB",
        "--disk",
        "50GB",
        "--region",
        args.region,
        "--extra",
        "tpu",
        "--sync-package",
        "marin-levanter",
        "--exclude",
        exclusion.pattern,
        "--job-name",
        job_name,
        "-e",
        "GIT_COMMIT",
        source,
        "-e",
        "IRIS_PORT_JAX",
        "33713",
        "--",
        "python",
        "-m",
        "experiments.ema_gram.run_geometry",
        "--configs",
        required[0].as_posix(),
        "--protocol",
        required[1].as_posix(),
        "--exclusions",
        required[2].as_posix(),
        "--state",
        args.state,
        "--experiment",
        args.experiment,
        "--stage",
        args.stage,
        "--output-prefix",
        prefix,
    ]
    for flag, path in optional.items():
        command += [flag, path]
    record = dict(
        source_commit=source,
        argv=command,
        job_id="/kaiyuew@stanford.edu/" + job_name,
        submitted_utc=datetime.now(UTC).isoformat(),
        status="checked" if args.check_only else "submitting",
        bundle_files=len(bundle),
        scope="Frozen checkpoint measurements; no wall-clock limits; default preemption and retry policy",
    )
    if args.check_only:
        print(json.dumps(record, indent=2))
        return
    path = root / archive / (stem + "-launch.json")
    if path.exists():
        raise FileExistsError("Attempt record already exists; never overwrite a submitted attempt")
    path.write_text(json.dumps(record, indent=2) + "\n")
    with (root / archive / (stem + "-submission.txt")).open("w") as output:
        outcome = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
    record.update(
        submission_exit_code=outcome.returncode, status="submitted" if outcome.returncode == 0 else "submission_failed"
    )
    path.write_text(json.dumps(record, indent=2) + "\n")
    monitor_path = root / "scratch/20261113_geometry_monitoring_state.json"
    monitor = json.loads(monitor_path.read_text())
    monitor["jobs"].append(dict(**record, restart_count=0, latest_signal=record["status"]))
    monitor_path.write_text(json.dumps(monitor, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    raise SystemExit(outcome.returncode)


if __name__ == "__main__":
    main()
