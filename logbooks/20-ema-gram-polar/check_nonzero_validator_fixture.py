# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline orchestration fixtures for the new nonzero terminal validator."""
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from wandb.util import json_friendly_val

sys.path.insert(0, str(Path("logbooks/20-ema-gram-polar").resolve()))
sys.path.append(str(Path("scratch").resolve()))
spec = importlib.util.spec_from_file_location("nonzero_validator", "logbooks/20-ema-gram-polar/verify_nonzero_r02.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
R = Path.cwd()
E = R.parent / "ema-gram-subset-seed0-r02-20261105"
index = 6
keys = m.TRAIN_KEYS
rows = [{"global_step": i, **{k: 1.0 for k in keys}} for i in range(7630)]
raw = [r for r in rows if r["global_step"] != 668] + [rows[1000]]
queries = []
actual = json.loads((R / "scratch/20261104-subset-all30-configs.json").read_text())[index]["config"]["train_config"]
frozen = copy.deepcopy(actual)
cfg = m.draccus.decode(m.TrainLmConfig, frozen)
cfg = replace(
    cfg,
    trainer=replace(
        cfg.trainer,
        jax_config=frozen["trainer"]["jax_config"],
        load_checkpoint_path=frozen["trainer"]["load_checkpoint_path"],
    ),
)
actual = json_friendly_val(asdict(cfg))
actual["git_commit"] = m.SOURCE
rid = next(
    iter(
        json.loads((R / "scratch/20261105-r02-attempt-joins/validated.json").read_text())["cells"]["6"][
            "native_run_tasks"
        ]
    )
)

actual["trainer"]["tracker"] = [
    {**actual["trainer"]["tracker"], "id": rid},
    {"logger_name": "ema_gram_metrics", "level": 20},
]


class Run:
    state = "finished"
    config = actual
    url = "https://wandb.ai/marin-community/marin/runs/" + rid

    def scan_history(self, **kwargs):
        queries.append(kwargs)
        if m.FINAL_KEY in kwargs["keys"]:
            return [{"global_step": 7629, m.FINAL_KEY: 4.5}]
        if "min_step" in kwargs:
            assert kwargs["max_step"] == kwargs["min_step"] + 1
            return [rows[kwargs["min_step"]]]
        return raw


class Blob:
    generation = 123
    crc32c = "fixture"

    def __init__(self, name):
        self.data = (
            (R / "scratch/20261105-r02-00-final-manifest.json").read_bytes()
            if name.endswith("manifest.json")
            else json.dumps({"step": 7629, "is_temporary": False}).encode()
        )
        if name.endswith("manifest.json"):
            manifest = json.loads(self.data)
            # Reproduce the observed one-host versus four-host MLP chunk difference.
            weight = next(
                a for a in manifest["arrays"] if a["path"] == "model/transformer/layers/stacked/mlp/gate_proj/weight"
            )
            weight["chunk_shape"] = [11, 1024, 256]
            self.data = json.dumps(manifest).encode()
        self.size = len(self.data)

    def download_as_bytes(self, **kwargs):
        assert kwargs == {"if_generation_match": 123, "checksum": "crc32c"}
        return self.data


class Scalar:
    shape = ()
    dtype = "int32"

    def __int__(self):
        return 7630

    def tobytes(self):
        return (7630).to_bytes(4, "little")


class Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def fake_open(spec, **kwargs):
    assert kwargs == {"open": True, "read": True, "write": False}
    assert spec["kvstore"]["path"] == "step"
    assert spec["kvstore"]["base"]["base_url"].startswith("https://storage.googleapis.com/marin-us-central2/")
    return Future(SimpleNamespace(read=lambda: Future(Scalar())))


m.wandb.Api = lambda: SimpleNamespace(run=lambda identity: Run() if identity == "marin-community/marin/" + rid else None)
m.storage.Client = lambda: SimpleNamespace(bucket=lambda bucket: SimpleNamespace(get_blob=lambda path: Blob(path)))
m.google.auth.default = lambda **kwargs: (SimpleNamespace(token="fixture-token", refresh=lambda request: None), None)
m.ts.open = fake_open

with tempfile.TemporaryDirectory(prefix="ema-terminal-fixture-") as temporary:
    parent = Path(temporary)
    root = parent / "research"
    execution = parent / "ema-gram-subset-seed0-r02-20261105"
    for path in [root / "scratch", root / "logbooks/20-ema-gram-polar", execution / "scratch"]:
        path.mkdir(parents=True)
    for relative in [
        "scratch/20261104-subset-all30-configs.json",
        "scratch/20261105-r02-00-final-manifest.json",
        "scratch/20261105-r02-attempt-joins/validated.json",
        "logbooks/20-ema-gram-polar/subset-training-r02-alpha0-completed.json",
    ]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(R / relative, target)
    for filename in ["20261105-subset-training-r02-observers.json", "20261105-subset-training-r02-ledger.json"]:
        shutil.copyfile(E / "scratch" / filename, execution / "scratch" / filename)
    op = execution / "scratch/20261105-subset-training-r02-observers.json"
    obs = json.loads(op.read_text())
    obs["6"]["terminal"] = True
    obs["6"]["latest"]["job"]["state"] = "JOB_STATE_SUCCEEDED"
    op.write_text(json.dumps(obs))
    state = json.loads((R / "scratch/20261030_ema_gram_monitoring_state.json").read_text())
    progress = json.loads(Path(state["single_seed_subset_rebuild"]["training_progress_evidence"]).read_text())
    progress_path = root / "scratch/progress.json"
    progress_path.write_text(json.dumps(progress))
    state["single_seed_subset_rebuild"]["training_progress_evidence"] = str(progress_path)
    (root / "scratch/20261030_ema_gram_monitoring_state.json").write_text(json.dumps(state))
    terminal = root / "scratch/20261105-r02-terminal-tasks"
    terminal.mkdir()
    join = json.loads((R / "scratch/20261105-r02-attempt-joins/validated.json").read_text())["cells"]["6"]
    for record in join["tasks"]:
        task = json.loads(Path(record["evidence"]).read_text())["task"]
        task["state"] = "TASK_STATE_SUCCEEDED"
        task["attempts"][0]["state"] = "TASK_STATE_SUCCEEDED"
        if record["task"] == 0:
            task["worker_id"] = ""
            task["attempts"][0]["worker_id"] = ""
        digest = hashlib.sha256(task["task_id"].encode()).hexdigest()
        response = terminal / (digest + ".response.json")
        response.write_text(json.dumps({"returncode": 0, "stdout": json.dumps({"task": task}), "stderr": ""}))
        (terminal / (digest + ".json")).write_text(
            json.dumps(
                {
                    "task_id": task["task_id"],
                    "task": task,
                    "raw_evidence": str(response),
                    "raw_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
                }
            )
        )
    result = m.verify(root, 6)
    assert result["scientifically_complete"] and result["stored_native_counter"] == 7630 and len(result["tasks"]) == 4
    assert result["checkpoint"]["arrays"] == 57 and result["final_c4_loss"] == 4.5
    assert len(queries) == 3 and queries[-1]["min_step"] == 668
    m.wandb.Api = lambda: (_ for _ in ()).throw(AssertionError("Repolled verified run"))
    assert m.verify(root, 6) == result
    # Each mutation starts from the same fully collected fictional artifacts.
    m.wandb.Api = lambda: SimpleNamespace(run=lambda identity: Run())
    negative_cases = [
        ("counter", "counter.json", lambda d: {**d, "counter": 7629}),
        ("final_eval", "eval-raw.json", lambda d: []),
        ("conflicting_history", "history-raw.json", lambda d: [*d, {**rows[0], "train/loss": 99.0}]),
        ("metadata", "metadata.json", lambda d: {**d, "parsed": {**d["parsed"], "step": 1}}),
        ("source", "config.json", lambda d: {**d, "git_commit": "wrong-source"}),
        (
            "offline",
            "config.json",
            lambda d: {
                **d,
                "trainer": {
                    **d["trainer"],
                    "tracker": [{**d["trainer"]["tracker"][0], "mode": "offline"}, d["trainer"]["tracker"][1]],
                },
            },
        ),
        ("batch", "config.json", lambda d: {**d, "trainer": {**d["trainer"], "train_batch_size": 16}}),
    ]
    for name, filename, mutate in negative_cases:
        candidate = parent / name
        shutil.copytree(root, candidate)
        artifacts = candidate / "scratch/20261105-r02-06-completion"
        (artifacts / "verified.json").unlink()
        target = artifacts / filename
        target.write_text(json.dumps(mutate(json.loads(target.read_text()))))
        try:
            m.verify(candidate, 6)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid artifact accepted: " + name)
    # A verified result's evidence digest catches later local artifact changes.
    candidate = parent / "changed_verified_artifact"
    shutil.copytree(root, candidate)
    original_counter = root / "scratch/20261105-r02-06-completion/counter.json"
    before = original_counter.read_bytes()
    original_counter.write_text("{}")
    try:
        m.verify(candidate, 6)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed verified evidence accepted")
    original_counter.write_bytes(before)
    m.wandb.Api = lambda: (_ for _ in ()).throw(AssertionError("Premature API request"))
    # Independent change in final task identity must invalidate even cached completion.
    path = next(terminal.glob("*.json"))
    record = json.loads(path.read_text())
    if "task" not in record:
        path = next(p for p in terminal.glob("*.json") if "task" in json.loads(p.read_text()))
        record = json.loads(path.read_text())
    record["task"]["attempts"][0]["attempt_uid"] = "wrong-fixture-uid"
    path.write_text(json.dumps(record))
    try:
        m.verify(root, 6)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed terminal evidence accepted")
print(
    "Seven negative full-artifact cases and cached evidence corruption rejected. "
    "Offline fixture passed: four-host completion, raw history preservation, missing-label repair, stored counter, "
    "no verified-run repoll, changed task rejection. No network or numerical training."
)
