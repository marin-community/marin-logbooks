# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fabricated complete nonzero groups for offline configuration checks."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import draccus
import prepare_subset_expansions as m
from levanter.main.train_lm import TrainLmConfig

r = Path.cwd()
real = json.loads(Path("logbooks/20-ema-gram-polar/subset-training-r02-alpha0-completed.json").read_text())
records = copy.deepcopy(real)
launch = json.loads(Path("logbooks/20-ema-gram-polar/subset-training-r02-launched.json").read_text())["ledger"]["cells"]
joins = json.loads(Path("scratch/20261105-r02-attempt-joins/validated.json").read_text())["cells"]
for i in [6, 7, 8, 12, 13, 14]:
    row = copy.deepcopy(real["results"][0])
    entry = launch[str(i)]
    join = joins[str(i)]
    row.update(index=i, cell=entry["cell"], job_id=entry["job_id"], run_id=next(iter(join["native_run_tasks"])))
    row["run_url"] = "https://wandb.ai/marin-community/marin/runs/" + row["run_id"]
    row["canonical_job"]["job_id"] = row["job_id"]
    row["tasks"] = []
    for n in range(4):
        task = copy.deepcopy(row["task"])
        task["task_id"] = row["job_id"] + "/" + str(n)
        task["worker_id"] = join["tasks"][n]["worker_id"]
        task["attempts"][0]["attempt_uid"] = join["tasks"][n]["attempt_uid"]
        row["tasks"].append(task)
    del row["task"]
    row["host_logging"] = [
        dict(id=t["host_logging_id"], state="finished", source_files_verified=True) for t in join["tasks"]
    ]
    row["run_attempt_join_evidence"] = join
    row["final_c4_loss"] = 4 + (i - 6) if i < 9 else 6 - (i - 12)
    records["results"].append(row)
bundle = m.prepare(records, "2026.11.05.92")
assert len(bundle["cells"]) == 2
assert bundle["cells"][0]["cell"]["learning_rate"] == m.CELLS[6].learning_rate / 2
assert bundle["cells"][1]["cell"]["learning_rate"] == m.CELLS[14].learning_rate * 2
for row in bundle["cells"]:
    assert row["variant"] == "v4-32"
    encoded = row["config"]
    cfg = draccus.decode(TrainLmConfig, encoded["train_config"])
    cfg = replace(
        cfg,
        trainer=replace(
            cfg.trainer,
            jax_config=encoded["train_config"]["trainer"]["jax_config"],
            load_checkpoint_path=encoded["train_config"]["trainer"]["load_checkpoint_path"],
        ),
    )
    pod = replace(
        m.prepared(0, row["variant"]),
        train_config=cfg,
        output_path=encoded["output_path"],
        env_vars=encoded["env_vars"],
        auto_build_caches=False,
    )
    assert draccus.encode(pod) == encoded
    assert cfg.trainer.num_train_steps == 7630 and cfg.trainer.train_batch_size == 8 and cfg.model.max_Pos.size == 4096
# A previously claimed expansion remains guarded even if submission was ambiguous.
cell = bundle["cells"][0]["cell"]
guard = dict(
    cell=cell,
    job_id="/fixture/ambiguous-expansion",
    source_commit="a" * 40,
    launch_started=True,
    launched=False,
    submit_exit_code=1,
)
second = m.prepare(records, "2026.11.05.93", [guard])
assert len(second["cells"]) == 1 and len(second["guarded_expansions"]) == 1
bad = copy.deepcopy(records)
bad["results"][-1]["source"] = "b" * 40
try:
    m.prepare(bad, "2026.11.05.94")
except ValueError:
    pass
else:
    raise AssertionError("Wrong predecessor source accepted")
for version in ["dev", "test-dev", "not-a-date", m.VERSION]:
    try:
        m.prepare(real, version)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid version accepted")
assert m.prepare(real, "2026.11.05.95")["cells"] == []
print(
    "Fictional two independent nonzero groups emit two exact frozen configs; round-trip, "
    "ambiguous-guard reuse, source binding and immutable version checks pass. No launch/kernels/API."
)
