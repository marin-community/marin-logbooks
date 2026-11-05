# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline controller orchestration; every external command is intercepted."""

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "qualified_expansion_submitter", Path.cwd() / "logbooks/20-ema-gram-polar/submit_subset_expansions.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
original = Path.cwd()
bundle = json.loads((original / "scratch/20261105-expansion-worker-fixture/bundle.json").read_text())
bundle["cells"] = bundle["cells"][1:]
source = "a" * 40
passed = []
for variant in ("v4-8", "v4-32"):
    cmd = m.command(0, variant, source, "ema-gram-20-expand-seed0-fixture", "b.json", "c.json", "a.json")
    assert cmd[cmd.index("--priority") + 1] == "batch"
    assert cmd[cmd.index("--tpu") + 1] == variant
    assert "--no-cancel-on-exit" in cmd
    assert not any("timeout" in arg or "max-retries" in arg or "time-limit" in arg for arg in cmd[:-1])
    assert cmd[-1].index("check_campaign_caches.py") < cmd[-1].index("logging-check") < cmd[-1].index("exec")
passed.append("Both TPU command variants preserve batch/default retries/no wallclock limits and logging before training")
with tempfile.TemporaryDirectory() as folder:
    parent = Path(folder)
    root = parent / "execution"
    root.mkdir()
    base = root / "logbooks/20-ema-gram-polar"
    base.mkdir(parents=True)
    for name in ["expansion_worker.py", "expansion-worker-preflight.json"]:
        shutil.copyfile(original / "logbooks/20-ema-gram-polar" / name, base / name)
    for name, value in [("bundle.json", bundle), ("completions.json", {}), ("accepted.json", [])]:
        (root / name).write_text(json.dumps(value))
    registry = parent / "ema-gram-research-20/scratch/20261105-subset-expansion-registry.json"
    registry.parent.mkdir(parents=True)
    campaign = "ema-gram-20-expand-seed0-fixture"
    argv = [
        "submitter",
        "--bundle",
        "bundle.json",
        "--completions",
        "completions.json",
        "--accepted-expansions",
        "accepted.json",
        "--registry",
        str(registry),
        "--source-commit",
        source,
        "--published-ref",
        "refs/heads/fixture",
        "--campaign",
        campaign,
        "--launch",
    ]
    calls = []
    fail = set()
    catalog_failure = False
    wrong_head = False

    def output(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return ("b" * 40 if wrong_head else source) + "\n"
        if command[:2] == ["git", "status"]:
            return b""
        if command[:2] == ["git", "ls-remote"]:
            return source + "\trefs/heads/fixture\n"
        raise AssertionError(command)

    def run(command, **kwargs):
        if command[:2] == ["git", "ls-files"]:
            return subprocess.CompletedProcess(command, 0)
        if "check_campaign_caches.py" in str(command[1]):
            if catalog_failure:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        assert "--job-name" in command, command
        job = command[command.index("--job-name") + 1]
        state = json.loads(registry.read_text())
        entry = next(r for r in state["cells"].values() if r["job_id"] == "/kaiyuew@stanford.edu/" + job)
        assert entry["launch_started"] and not entry["launched"]
        calls.append(job)
        return subprocess.CompletedProcess(command, 1 if job in fail else 0)

    os.chdir(root)
    try:
        with (
            patch.object(m, "ROOT", root),
            patch.object(m, "BASE", base),
            patch.object(m, "prepare", lambda *a: copy.deepcopy(bundle)),
            patch.object(m.subprocess, "run", run),
            patch.object(m.subprocess, "check_output", output),
            patch.object(sys, "argv", argv),
        ):
            wrong_head = True
            try:
                m.main()
            except ValueError:
                pass
            else:
                raise AssertionError("Wrong source accepted")
            assert not registry.exists() and calls == []
            wrong_head = False
            passed.append("Wrong source rejected before registry or RPC")
            catalog_failure = True
            try:
                m.main()
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError("Failed cache check ignored")
            assert json.loads(registry.read_text())["cells"] == {} and calls == []
            catalog_failure = False
            passed.append("Failed catalog prevents guards and submissions")
            fail.add(campaign + "-01")
            try:
                m.main()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Ambiguous submission reported as complete")
            saved = json.loads(registry.read_text())
            assert (
                len(calls) == 2
                and len(saved["cells"]) == 2
                and all(x["launch_started"] for x in saved["cells"].values())
            )
            passed.append("Both parallel RPCs see persisted guards; one error retains all guards")
            before = len(calls)
            for new_campaign in [campaign, "ema-gram-20-expand-seed0-other"]:
                argv[argv.index("--campaign") + 1] = new_campaign
                try:
                    m.main()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Duplicate campaign/setting accepted")
            assert len(calls) == before
            passed.append("Repeated campaign and cross-campaign duplicate rejected without RPC")
            registry.unlink()
            try:
                m.main()
            except ValueError:
                pass
            else:
                raise AssertionError("Missing permanent registry silently recreated")
            assert not registry.exists() and len(calls) == before
            passed.append("Initialization marker prevents recreation after registry loss")
    finally:
        os.chdir(original)
print(
    json.dumps(
        dict(
            passed=True,
            checks=passed,
            scope="Temporary files and mocked external commands only; no real submission or permanent registry.",
        )
    )
)
(original / "scratch/20261105-expansion-submitter-orchestration-preflight.json").write_text(
    json.dumps(dict(passed=True, checks=passed), indent=2) + "\n"
)
