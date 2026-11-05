# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from experiments.ema_gram.plan import Cell

root = Path.cwd()
sys.path.insert(0, str(root / "logbooks/20-ema-gram-polar"))
spec = importlib.util.spec_from_file_location(
    "expansion_terminal_draft", root / "logbooks/20-ema-gram-polar/prepare_expansion_terminal_collection.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

bundle = json.loads((root / "scratch/20261105-expansion-worker-fixture/bundle.json").read_text())
bundle["cells"] = bundle["cells"][1:2]
campaign = "ema-gram-20-expand-seed0-terminalfixture"
source = "a" * 40
checks = []
with tempfile.TemporaryDirectory() as temp:
    w = Path(temp)
    (w / "scratch").mkdir()
    worker = w / "logbooks/20-ema-gram-polar/expansion_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_bytes((root / "logbooks/20-ema-gram-polar/expansion_worker.py").read_bytes())
    bp = w / "bundle.json"
    bp.write_text(json.dumps(bundle))
    binding = dict(
        source=source,
        artifact_version=bundle["artifact_version"],
        inputs={"bundle.json": hashlib.sha256(bp.read_bytes()).hexdigest()},
    )
    r = bundle["cells"][0]
    job = f"/kaiyuew@stanford.edu/{campaign}-00"
    entry = dict(
        index=0,
        campaign=campaign,
        source_commit=source,
        config_sha256=r["config_sha256"],
        cell=r["cell"],
        job_id=job,
        launch_started=True,
        launched=True,
        submit_exit_code=0,
    )
    registry = dict(campaigns={campaign: binding}, cells={Cell(**r["cell"]).name: entry})
    cp = w / "canonical.json"
    op = w / "scratch" / f"{campaign}-observers.json"

    def observe(state, terminal):
        jobrow = dict(job_id=job, state=state, exit_code=0, task_count=4, failure_count=0, preemption_count=0)
        cp.write_text(json.dumps(dict(job=jobrow)))
        obs = dict(
            binding=binding,
            jobs={
                "0": dict(
                    job_id=job,
                    source_commit=source,
                    config_sha256=r["config_sha256"],
                    terminal=terminal,
                    latest=dict(job=jobrow, path=str(cp)),
                )
            },
        )
        op.write_text(json.dumps(obs))
        return obs

    observe("JOB_STATE_RUNNING", False)
    plan = m.prepare(w, registry, campaign, Path("bundle.json"), {}, {})
    assert not plan["requests"] and len(plan["waiting"]) == 1
    checks.append("Running jobs yield no terminal requests")
    observe("JOB_STATE_SUCCEEDED", True)
    plan = m.prepare(w, registry, campaign, Path("bundle.json"), {}, {})
    assert (
        len(plan["requests"]) == 1
        and len(plan["requests"][0]["task_ids"]) == 4
        and not plan["requests"][0]["scientifically_complete"]
    )
    checks.append("Exact terminal job yields four task requests without scientific completion")
    observe("JOB_STATE_FAILED", True)
    plan = m.prepare(w, registry, campaign, Path("bundle.json"), {}, {})
    assert plan["requests"][0]["canonical_state"] == "JOB_STATE_FAILED"
    checks.append("Failed outcomes remain collectable evidence")
    for kind in ["source", "raw_snapshot", "terminal_flag", "input_hash", "job_identity"]:
        obs = observe("JOB_STATE_SUCCEEDED", True)
        reg = copy.deepcopy(registry)
        if kind == "source":
            reg["campaigns"][campaign]["source"] = "b" * 40
        if kind == "raw_snapshot":
            cp.write_text(json.dumps(dict(job={})))
        if kind == "terminal_flag":
            obs["jobs"]["0"]["terminal"] = False
            op.write_text(json.dumps(obs))
        if kind == "input_hash":
            reg["campaigns"][campaign]["inputs"]["bundle.json"] = "bad"
        if kind == "job_identity":
            obs["jobs"]["0"]["job_id"] = "wrong"
            op.write_text(json.dumps(obs))
        try:
            m.prepare(w, reg, campaign, Path("bundle.json"), {}, {})
        except ValueError:
            pass
        else:
            raise AssertionError(kind)
    checks.append("Reject source, raw snapshot, terminal flag, input and job identity mismatches")
    observe("JOB_STATE_SUCCEEDED", True)
    reg = copy.deepcopy(registry)
    next(iter(reg["cells"].values()))["launched"] = False
    assert not m.prepare(w, reg, campaign, Path("bundle.json"), {}, {})["requests"]
    checks.append("Ambiguous launch guard never becomes a read request")
result = dict(
    passed=True,
    checks=checks,
    scope="Synthetic local bindings and snapshots only; no RPC, W&B, storage access or training.",
    remaining=["First real terminal expansion validation"],
)
(root / "scratch/20261105-expansion-terminal-plan-draft-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
