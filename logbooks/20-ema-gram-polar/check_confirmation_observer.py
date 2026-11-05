# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise fictional confirmation submissions with local files and fake Iris I/O."""

import copy
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import watch_subset_confirmations as observer
import watch_subset_expansions as shared
from prepare_subset_confirmations import prepare
from submit_subset_confirmations import command, validate_completions

from experiments.ema_gram.plan import read_lock

root = Path.cwd()
base = root / "logbooks/20-ema-gram-polar"
lock_path = root / "scratch/20261105-confirmation-fixture-lock.json"
lock = read_lock(lock_path)
bundle = prepare(lock, "confirmation-observer-dev")
alpha0 = json.loads((base / "subset-training-r02-alpha0-completed.json").read_text())["results"][0]
nonzero = json.loads((base / "subset-training-r02-nonzero-completed.json").read_text())["results"][0]
cells = {c.name: c for c in lock.pilots}
records = []
for result in lock.results:
    cell = cells[result.name]
    row = copy.deepcopy(alpha0 if cell.alpha == 0 else nonzero)
    row.update(cell=asdict(cell), final_c4_loss=result.validation_loss, run_url=result.run_url)
    records.append(row)
completions = dict(results=records)
validate_completions(lock, completions)
source = "a" * 40
campaign = "ema-gram-20-confirm-seed0-observerfixture"
checks, follows = [], []
clock = [1000.0]


class Follow:
    pid = 123456789
    returncode = 0

    def poll(self):
        return self.returncode


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


with tempfile.TemporaryDirectory() as folder:
    worktree = Path(folder)
    scratch = worktree / "scratch"
    scratch.mkdir()
    (worktree / "bundle.json").write_text(json.dumps(bundle))
    (worktree / "lock.json").write_bytes(lock_path.read_bytes())
    (worktree / "completions.json").write_text(json.dumps(completions))
    binding = dict(
        source=source,
        inputs={
            name: hashlib.sha256((worktree / name).read_bytes()).hexdigest()
            for name in ("bundle.json", "lock.json", "completions.json")
        },
    )
    ledger = dict(binding=binding, cells={})
    for i, cell in enumerate(bundle["cells"]):
        ledger["cells"][str(i)] = dict(
            job_id=f"/kaiyuew@stanford.edu/{campaign}-{i:02d}",
            argv=command(i, cell["variant"], source, campaign, Path("bundle.json"), Path("lock.json")),
            launch_started=True,
            launched=i != 1,
            submit_exit_code=None if i == 1 else 0,
        )
    ledger_path = scratch / f"{campaign}-ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    state_path = scratch / f"{campaign}-observers.json"
    argv = [
        "observer",
        "--worktree",
        str(worktree),
        "--bundle",
        "bundle.json",
        "--lock",
        "lock.json",
        "--completions",
        "completions.json",
        "--campaign",
        campaign,
        "--source-commit",
        source,
    ]

    def git_output(cmd, **kwargs):
        if cmd == ["git", "rev-parse", "HEAD"]:
            return source + "\n"
        assert cmd[:2] == ["git", "show"]
        return (worktree / cmd[2].split(":", 1)[1]).read_bytes()

    def popen(cmd, **kwargs):
        index = str(int(cmd[-2].rsplit("-", 1)[1]))
        saved = json.loads(state_path.read_text())
        assert saved["jobs"][index]["follow_started"]
        follows.append((index, clock[0]))
        return Follow()

    def run(cmd, **kwargs):
        assert "get-job-status" in cmd
        checks.append((cmd[-1], clock[0]))
        kwargs["stdout"].write(json.dumps(dict(job=dict(job_id=cmd[-1], state="JOB_STATE_SUCCEEDED"))))
        return type("Result", (), {"returncode": 0, "stderr": "fixture diagnostic\n"})()

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] == 1010:
            # The ambiguous submission is never followed until acceptance is explicitly reconciled.
            assert "1" not in json.loads(state_path.read_text())["jobs"]
            ledger["cells"]["1"].update(launched=True, submit_exit_code=0)
            ledger_path.write_text(json.dumps(ledger))
        assert clock[0] < 2000, "Fixture failed to reach canonical termination"

    with (
        patch.object(sys, "argv", argv),
        patch.object(observer.subprocess, "check_output", git_output),
        patch.object(shared.subprocess, "Popen", popen),
        patch.object(shared.subprocess, "run", run),
        patch.object(shared.time, "time", lambda: clock[0]),
        patch.object(shared.time, "sleep", sleep),
        patch.object(shared.concurrent.futures, "ThreadPoolExecutor", Pool),
    ):
        observer.main()
        saved = json.loads(state_path.read_text())
        assert len(follows) == len(checks) == len(bundle["cells"])
        assert all(r["terminal"] for r in saved["jobs"].values())
        assert sorted(t for _, t in follows) == [1000] * (len(bundle["cells"]) - 1) + [1010]
        assert sorted(t for _, t in checks) == [1120] * (len(bundle["cells"]) - 1) + [1130]
        before = (len(checks), len(follows))
        observer.main()
        assert before == (len(checks), len(follows))
        # Replacing a source argument in the accepted ledger must not produce observations of another run.
        ledger["cells"]["0"]["argv"][-1] += "\nfalse"
        ledger_path.write_text(json.dumps(ledger))
        try:
            observer.main()
        except ValueError:
            pass
        else:
            raise AssertionError("Changed accepted command was observed")
        assert before == (len(checks), len(follows))
        assert saved == json.loads(state_path.read_text())

result = dict(
    passed=True,
    scope=(
        "Fictional complete lock/results and submission ledger; real temporary state files, "
        "fake Git/Iris/processes/pool/clock. No network, kernels or training."
    ),
    checks=[
        "Full CLI validates frozen lock/config/completion evidence and accepted command identities",
        "Unresolved submission excluded until accepted; all accepted jobs followed once with durable guards",
        "Each job first checked after120 seconds; canonical terminal restart makes no new RPC/follower",
        "Changed accepted command rejected without RPC/follower/state changes",
    ],
    sources={
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in ("watch_subset_confirmations.py", "watch_subset_expansions.py", "check_confirmation_observer.py")
    },
    real_confirmation_launched=False,
)
(root / "scratch/20261105-confirmation-observer-validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
