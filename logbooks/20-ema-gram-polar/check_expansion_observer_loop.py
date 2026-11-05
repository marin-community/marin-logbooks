# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mock every subprocess and advance virtual time; only temporary state/locks are real."""

import copy
import fcntl
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

root = Path.cwd()
sys.path.insert(0, str(root / "logbooks/20-ema-gram-polar"))
spec = importlib.util.spec_from_file_location(
    "expansion_observer_draft", root / "logbooks/20-ema-gram-polar/watch_subset_expansions.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
bundle = json.loads((root / "scratch/20261105-expansion-worker-fixture/bundle.json").read_text())
bundle["cells"] = bundle["cells"][:1]
source = "a" * 40
campaign = "ema-gram-20-expand-seed0-observerfixture"
passed = []
with tempfile.TemporaryDirectory() as folder:
    parent = Path(folder)
    worktree = parent / "execution"
    worktree.mkdir()
    (worktree / "bundle.json").write_text(json.dumps(bundle))
    registry = parent / "ema-gram-research-20/scratch/20261105-subset-expansion-registry.json"
    registry.parent.mkdir(parents=True)
    binding = dict(
        source=source,
        artifact_version=bundle["artifact_version"],
        inputs={"bundle.json": hashlib.sha256((worktree / "bundle.json").read_bytes()).hexdigest()},
    )
    row = bundle["cells"][0]
    job = f"/kaiyuew@stanford.edu/{campaign}-00"
    entry = dict(
        campaign=campaign,
        index=0,
        source_commit=source,
        cell=row["cell"],
        config_sha256=row["config_sha256"],
        job_id=job,
        launch_started=True,
        launched=True,
        submit_exit_code=0,
    )
    registry.write_text(json.dumps(dict(campaigns={campaign: binding}, cells={m.Cell(**row["cell"]).name: entry})))
    argv = [
        "observer",
        "--worktree",
        str(worktree),
        "--registry",
        str(registry),
        "--bundle",
        "bundle.json",
        "--campaign",
        campaign,
        "--source-commit",
        source,
    ]
    clock = [1000.0]
    follows = []
    checks = []
    transport_error = [False]
    ambiguous = [False]

    class Follow:
        pid = 123456789
        returncode = 0

        def poll(self):
            return 0

    def popen(cmd, **kwargs):
        state = json.loads((worktree / "scratch" / f"{campaign}-observers.json").read_text())
        assert state["jobs"]["0"]["follow_started"]
        follows.append(cmd)
        if ambiguous[0]:
            raise RuntimeError("fictional ambiguous Popen")
        return Follow()

    def run(cmd, **kwargs):
        assert "get-job-status" in cmd and cmd[-1] == job
        checks.append(clock[0])
        error = transport_error[0] and len(checks) == 1
        value = (
            "fixture transport failure"
            if error
            else json.dumps(
                dict(job=dict(job_id=job, state="JOB_STATE_RUNNING" if len(checks) == 1 else "JOB_STATE_SUCCEEDED"))
            )
        )
        kwargs["stdout"].write(value)
        assert kwargs["stderr"] == m.subprocess.PIPE and kwargs["text"]
        return type("Result", (), {"returncode": 1 if error else 0, "stderr": "fixture CLI diagnostic\n"})()

    def sleep(seconds):
        clock[0] += seconds
        assert clock[0] < 5000, "Unexpected unbounded fixture loop"

    class Future:
        def __init__(self, fn, *args):
            self.value = fn(*args)

        def done(self):
            return True

        def result(self):
            return self.value

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, *args):
            return Future(fn, *args)

    with (
        patch.object(sys, "argv", argv),
        patch.object(m.subprocess, "check_output", lambda *a, **k: source + "\n"),
        patch.object(m.subprocess, "run", run),
        patch.object(m.subprocess, "Popen", popen),
        patch.object(m.time, "time", lambda: clock[0]),
        patch.object(m.time, "sleep", sleep),
        patch.object(m.concurrent.futures, "ThreadPoolExecutor", Pool),
    ):
        m.main()
        assert len(follows) == 1 and len(checks) == 2 and checks[0] == 1120 and checks[1] - checks[0] >= 570
        saved = json.loads((worktree / "scratch" / f"{campaign}-observers.json").read_text())
        assert saved["jobs"]["0"]["terminal"]
        assert Path(saved["jobs"]["0"]["latest"]["stderr"]).read_text() == "fixture CLI diagnostic\n"
        before = (len(follows), len(checks))
        m.main()
        assert before == (len(follows), len(checks))
        passed.append("Terminal restart makes no RPC or new follower; first120 then570 cadence")
        lockpath = worktree / "scratch" / f"{campaign}-observers.lock"
        with lockpath.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                m.main()
            except BlockingIOError:
                pass
            else:
                raise AssertionError("Second owner acquired lock")
        assert before == (len(follows), len(checks))
        passed.append("Real temporary exclusive lock rejects a second owner without RPC/follower")
        statepath = worktree / "scratch" / f"{campaign}-observers.json"
        raw = statepath.read_text()
        statepath.unlink()
        try:
            m.main()
        except ValueError:
            pass
        else:
            raise AssertionError("Missing state recreated")
        statepath.write_text(raw)
        passed.append("Missing initialized state rejects recovery without replacing followers")
        saved["jobs"]["0"]["terminal"] = False
        saved["jobs"]["0"]["next_check"] = clock[0]
        saved["jobs"]["0"].pop("latest", None)
        statepath.write_text(json.dumps(saved))
        checks.clear()
        transport_error[0] = True
        m.main()
        assert len(follows) == 1 and len(checks) == 2
        after = json.loads(statepath.read_text())
        assert after["jobs"]["0"]["terminal"]
        assert any("fixture transport failure" in p.read_text() for p in (worktree / "scratch").glob("*canonical*.json"))
        passed.append(
            "Restored active owner preserves follow identity, retains raw RPC error and continues canonical checks"
        )
        # A separate fictional campaign exercises the interrupted local-follower path.
        other = campaign + "-ambiguous"
        argv[argv.index("--campaign") + 1] = other
        entry2 = copy.deepcopy(entry)
        entry2["campaign"] = other
        entry2["job_id"] = f"/kaiyuew@stanford.edu/{other}-00"
        job = entry2["job_id"]
        registry.write_text(json.dumps(dict(campaigns={other: binding}, cells={m.Cell(**row["cell"]).name: entry2})))
        campaign = other
        ambiguous[0] = True
        try:
            m.main()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Ambiguous follower did not stop")
        before = len(follows)
        ambiguous[0] = False
        transport_error[0] = False
        checks.clear()
        m.main()
        assert len(follows) == before
        passed.append("Durable guard prevents a replacement follower after ambiguous Popen")
result = dict(
    passed=True,
    checks=passed,
    scope="Temporary files/flock only; mocked subprocesses, pools and time. No RPC, real follower or sleep.",
)
(root / "scratch/20261105-expansion-observer-loop-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
