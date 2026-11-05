# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline, fictional boundary scenarios; no API, kernels or storage writes."""
import copy
import json
from pathlib import Path

import prepare_subset_expansions as m

real = json.loads(Path("logbooks/20-ema-gram-polar/subset-training-r02-alpha0-completed.json").read_text())
fiction = copy.deepcopy(real)
for i, r in enumerate(fiction["results"]):
    r["final_c4_loss"] = 4.0 + i
bundle = m.prepare(fiction, "2026.11.05.91")
assert len(bundle["cells"]) == 1
new = bundle["cells"][0]
original = m.Cell(**fiction["results"][0]["cell"])
assert new["cell"]["learning_rate"] == original.learning_rate / 2
cfg = new["config"]["train_config"]
assert cfg["trainer"]["num_train_steps"] == 7630
r = copy.deepcopy(fiction["results"][0])
r["cell"] = new["cell"]
r["job_id"] = "/fictional-expansion/job"
r["canonical_job"]["job_id"] = r["job_id"]
r["run_id"] = "fictional-expansion-run"
r["run_url"] = "https://wandb.ai/fictional/fixture/runs/fictional-expansion-run"
r["task"]["task_id"] = r["job_id"] + "/0"
r["final_c4_loss"] = 3.0
fiction["results"].append(r)
next_, groups = m.next_cells(fiction)
assert len(next_) == 1 and next_[0].learning_rate == original.learning_rate / 4
bad = copy.deepcopy(fiction)
bad["results"][-1]["cell"]["learning_rate"] = original.learning_rate / 3
try:
    m.next_cells(bad)
except ValueError:
    pass
else:
    raise AssertionError("Unplanned expansion accepted")
partial = copy.deepcopy(real)
partial["results"].pop()
assert m.next_cells(partial)[0] == []
print(
    "Offline expansion draft passed: factor-two frozen config, recursive boundary replay, "
    "unknown-cell rejection, incomplete-group waiting. No launch or numerical process."
)
