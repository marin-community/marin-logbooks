# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "logbooks/20-ema-gram-polar"))
spec = importlib.util.spec_from_file_location(
    "expansion_observer_draft", root / "logbooks/20-ema-gram-polar/watch_subset_expansions.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
bundle = json.loads((root / "scratch/20261105-expansion-worker-fixture/bundle.json").read_text())
campaign = "ema-gram-20-expand-seed0-fixture"
source = "a" * 40
binding = dict(source=source, artifact_version=bundle["artifact_version"])
registry = dict(campaigns={campaign: binding}, cells={})
for i, row in enumerate(bundle["cells"]):
    registry["cells"][m.Cell(**row["cell"]).name] = dict(
        campaign=campaign,
        index=i,
        source_commit=source,
        cell=row["cell"],
        config_sha256=row["config_sha256"],
        job_id=f"/kaiyuew@stanford.edu/{campaign}-{i:02d}",
        launch_started=True,
        launched=True,
        submit_exit_code=0,
    )
entries = m.accepted_entries(registry, campaign, source, bundle)
assert len(entries) == 3
state = {}
assert m.register(state, "0", entries["0"], 1000)
assert state["0"]["next_check"] == 1120
assert not m.register(state, "0", entries["0"], 2000)
success = dict(job=dict(job_id=entries["0"]["job_id"], state="JOB_STATE_RUNNING"), path="fictional.json")
m.apply_check(state["0"], success, 1120)
assert state["0"]["next_check"] == 1690 and not state["0"]["terminal"]
m.apply_check(state["0"], dict(error="fixture transport error"), 1690)
assert state["0"]["last_successful_snapshot"] == success and state["0"]["next_check"] == 2260
m.apply_check(state["0"], dict(job=dict(job_id=entries["0"]["job_id"], state="JOB_STATE_SUCCEEDED")), 2260)
assert state["0"]["terminal"]
assert not m.register(state, "0", entries["0"], 3000)
bad = copy.deepcopy(entries["0"])
bad["job_id"] = "wrong"
try:
    m.register(state, "0", bad, 3000)
except ValueError:
    pass
else:
    raise AssertionError("Changed saved identity accepted")
for key, value in [
    ("source_commit", "b" * 40),
    ("job_id", "wrong"),
    ("config_sha256", "wrong"),
    ("cell", bundle["cells"][1]["cell"]),
]:
    bad = copy.deepcopy(registry)
    next(iter(bad["cells"].values()))[key] = value
    try:
        m.accepted_entries(bad, campaign, source, bundle)
    except ValueError:
        pass
    else:
        raise AssertionError(key)
ambiguous = copy.deepcopy(registry)
next(iter(ambiguous["cells"].values()))["launched"] = False
assert len(m.accepted_entries(ambiguous, campaign, source, bundle)) == 2
result = dict(
    passed=True,
    scope="Pure-function synthetic fixtures only; no jobs, RPCs, followers, sleeps or state files created.",
    checks=[
        "Exact source/job/config/cell bindings",
        "Ambiguous launch excluded",
        "First120 then570 seconds",
        "Duplicate registration creates no follower",
        "Transport error retains last successful snapshot",
        "Terminal flag and immutable saved identity",
    ],
    remaining=[
        "Full loop/lock/process mocks",
        "Review restoration and all terminal states",
        "Lint and publish before use",
        "Expansion terminal scientific validator",
    ],
)
(root / "scratch/20261105-expansion-observer-draft-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
