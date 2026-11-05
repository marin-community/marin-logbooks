# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline scientific completion fixtures for expansion pilots."""
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
spec = importlib.util.spec_from_file_location(
    "nonzero_validator", "logbooks/20-ema-gram-polar/verify_expansion_artifacts.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.SOURCE = "a" * 40
R = Path.cwd()
E = R.parent / "ema-gram-subset-seed0-r02-20261105"
index = 6
keys = m.TRAIN_KEYS
rows = [{"global_step": i, **{k: 1.0 for k in keys}} for i in range(7630)]
raw = [r for r in rows if r["global_step"] != 668] + [rows[1000]]
queries = []
bundle = json.loads((R / "scratch/20261105-expansion-worker-fixture/bundle.json").read_text())
expansion = bundle["cells"][0 if "--alpha0" in sys.argv else 1]
actual = expansion["config"]["train_config"]
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
rid = frozen["trainer"]["id"] + "-fictionalrun"

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


with tempfile.TemporaryDirectory(prefix="ema-expansion-fixture-") as temporary:
    root = Path(temporary) / "ema-gram-research-20"
    for relative in [
        "scratch/20261105-r02-00-final-manifest.json",
        "logbooks/20-ema-gram-polar/subset-training-r02-alpha0-completed.json",
    ]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(R / relative, target)
    terminal = root / "tasks"
    terminal.mkdir()
    out = root / "artifacts"
    join = copy.deepcopy(json.loads((R / "scratch/20261105-r02-attempt-joins/validated.json").read_text())["cells"]["6"])
    join["tasks"] = join["tasks"][: 1 if expansion["cell"]["alpha"] == 0 else 4]
    join["source_commit"] = m.SOURCE
    join["native_run_tasks"] = {rid: 0}
    job_id = "/kaiyuew@stanford.edu/ema-gram-20-expand-seed0-fictional-00"
    hosts = {}
    for rec in join["tasks"]:
        task = json.loads(Path(rec["evidence"]).read_text())["task"]
        task["task_id"] = job_id + "/" + str(rec["task"])
        startup_path = root / f"startup-{rec['task']}.json"
        startup_path.write_text(json.dumps({"task": task}))
        rec["evidence"] = str(startup_path)
        task["state"] = "TASK_STATE_SUCCEEDED"
        task["attempts"][0]["state"] = "TASK_STATE_SUCCEEDED"
        if rec["task"] == 0:
            task["worker_id"] = ""
            task["attempts"][0]["worker_id"] = ""
        digest = hashlib.sha256(task["task_id"].encode()).hexdigest()
        response = terminal / (digest + ".response.json")
        response.write_text(json.dumps(dict(returncode=0, stdout=json.dumps(dict(task=task)), stderr="")))
        (terminal / (digest + ".json")).write_text(
            json.dumps(
                dict(
                    task_id=task["task_id"],
                    task=task,
                    raw_evidence=str(response),
                    raw_sha256=hashlib.sha256(response.read_bytes()).hexdigest(),
                )
            )
        )
        rec["host_logging_id"] = "ema-gram-expansion-host-fixture" + str(rec["task"])
        hosts[rec["host_logging_id"]] = SimpleNamespace(
            state="finished",
            summary={"logging_check/success": 1},
            config=dict(
                source_commit=m.SOURCE,
                task=task["task_id"],
                attempt_uid=task["attempts"][0]["attempt_uid"],
                config_sha256=expansion["config_sha256"],
                bundle_sha256="fixturebundle",
                entrypoint_sha256="fixtureworker",
                config=copy.deepcopy(expansion["config"]),
            ),
        )
    request = dict(
        index=0,
        campaign="ema-gram-20-expand-seed0-fictional",
        source_commit=m.SOURCE,
        canonical_state="JOB_STATE_SUCCEEDED",
        failure_count=0,
        preemption_count=0,
        canonical_job=dict(job_id=job_id, state="JOB_STATE_SUCCEEDED", exit_code=0),
        job_id=job_id,
        cached_startup_join=join,
        config=expansion["config"],
        config_sha256=expansion["config_sha256"],
        task_ids=[job_id + "/" + str(i) for i in range(len(join["tasks"]))],
        input_binding={"source": m.SOURCE},
        bundle_sha256="fixturebundle",
        worker_sha256="fixtureworker",
        cell=expansion["cell"],
    )
    canonical = root / "canonical.json"
    canonical.write_text(json.dumps(dict(job=request["canonical_job"])))
    request["canonical_evidence"] = str(canonical)
    request["canonical_sha256"] = hashlib.sha256(canonical.read_bytes()).hexdigest()

    def api():
        def run(identity):
            return Run() if identity == "marin-community/marin/" + rid else hosts[identity.rsplit("/", 1)[1]]

        return SimpleNamespace(run=run)

    m.wandb.Api = api
    result = m.verify_artifacts(root, request, terminal, out)
    assert (
        result["scientifically_complete"]
        and result["stored_native_counter"] == 7630
        and result["checkpoint"]["arrays"] == 57
        and result["final_c4_loss"] == 4.5
    )
    m.wandb.Api = lambda: (_ for _ in ()).throw(AssertionError("Repeated completed-run API query"))
    assert m.verify_artifacts(root, request, terminal, out) == result
    m.wandb.Api = api
    host_file = join["tasks"][0]["host_logging_id"] + ".json"
    cases = [
        ("counter", "counter.json", lambda d: {**d, "counter": 7629}),
        ("final_eval", "eval-raw.json", lambda d: []),
        ("conflicting_history", "history-raw.json", lambda d: [*d, {**rows[0], "train/loss": 99.0}]),
        ("metadata", "metadata.json", lambda d: {**d, "parsed": {**d["parsed"], "step": 1}}),
        ("native_source", "config.json", lambda d: {**d, "git_commit": "wrong-source"}),
        (
            "native_offline",
            "config.json",
            lambda d: {
                **d,
                "trainer": {
                    **d["trainer"],
                    "tracker": [{**d["trainer"]["tracker"][0], "mode": "offline"}, d["trainer"]["tracker"][1]],
                },
            },
        ),
        ("host_source", host_file, lambda d: {**d, "config": {**d["config"], "source_commit": "wrong"}}),
        ("host_task", host_file, lambda d: {**d, "config": {**d["config"], "task": "wrong"}}),
        ("host_bundle", host_file, lambda d: {**d, "config": {**d["config"], "bundle_sha256": "wrong"}}),
        ("host_finish", host_file, lambda d: {**d, "state": "running"}),
    ]
    for name, filename, mutate in cases:
        candidate = root / name
        shutil.copytree(out, candidate)
        (candidate / "verified.json").unlink()
        target = candidate / filename
        target.write_text(json.dumps(mutate(json.loads(target.read_text()))))
        try:
            m.verify_artifacts(root, request, terminal, candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid expansion artifact accepted: " + name)
    counter = out / "counter.json"
    original = counter.read_bytes()
    counter.write_text("{}")
    m.wandb.Api = lambda: (_ for _ in ()).throw(AssertionError("API query before cached integrity gate"))
    try:
        m.verify_artifacts(root, request, terminal, out)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed verified evidence accepted")
    counter.write_bytes(original)
    # Exercise the terminal-plan/CLI boundary using the same fabricated artifacts.
    from unittest.mock import patch

    sys.path.append(str(R / "scratch"))
    f_spec = importlib.util.spec_from_file_location(
        "expansion_finish_draft", R / "logbooks/20-ema-gram-polar/finish_subset_expansions.py"
    )
    f = importlib.util.module_from_spec(f_spec)
    f_spec.loader.exec_module(f)
    execution = root.parent / "execution"
    (execution / "scratch").mkdir(parents=True)
    campaign = "ema-gram-20-expand-seed0-fictional"
    fixture_bundle = copy.deepcopy(bundle)
    fixture_bundle["cells"] = [expansion]
    for name, value in [("bundle.json", fixture_bundle), ("completions.json", {}), ("accepted.json", [])]:
        (execution / name).write_text(json.dumps(value))
    worker = execution / "logbooks/20-ema-gram-polar/expansion_worker.py"
    worker.parent.mkdir(parents=True)
    shutil.copyfile(R / "logbooks/20-ema-gram-polar/expansion_worker.py", worker)
    binding = dict(
        source=m.SOURCE,
        artifact_version=fixture_bundle["artifact_version"],
        inputs={
            n: hashlib.sha256((execution / n).read_bytes()).hexdigest()
            for n in ["bundle.json", "completions.json", "accepted.json"]
        },
    )
    registry = root / "scratch/20261105-subset-expansion-registry.json"
    entry = dict(
        index=0,
        campaign=campaign,
        source_commit=m.SOURCE,
        config_sha256=expansion["config_sha256"],
        cell=expansion["cell"],
        job_id=job_id,
        launch_started=True,
        launched=True,
        submit_exit_code=0,
    )
    from experiments.ema_gram.plan import Cell

    registry.write_text(json.dumps(dict(campaigns={campaign: binding}, cells={Cell(**expansion["cell"]).name: entry})))
    observed = dict(
        binding=binding,
        jobs={
            "0": dict(
                job_id=job_id,
                source_commit=m.SOURCE,
                config_sha256=expansion["config_sha256"],
                terminal=True,
                latest=dict(
                    job={
                        **request["canonical_job"],
                        "failure_count": 0,
                        "preemption_count": 0,
                        "task_count": len(join["tasks"]),
                    },
                    path=str(canonical),
                ),
            )
        },
    )
    canonical.write_text(json.dumps(dict(job=observed["jobs"]["0"]["latest"]["job"])))
    op = execution / "scratch" / (campaign + "-observers.json")
    op.write_text(json.dumps(observed))
    jp = execution / "joins.json"
    jp.write_text(json.dumps(dict(jobs={job_id: join})))
    td = execution / "scratch" / (campaign + "-terminal-tasks")
    shutil.copytree(terminal, td)
    for h in hosts.values():
        h.config["bundle_sha256"] = binding["inputs"]["bundle.json"]
        h.config["entrypoint_sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
    m.wandb.Api = api
    args = (execution, registry, campaign, Path("bundle.json"), Path("completions.json"), Path("accepted.json"), jp)
    with (
        patch.object(f.subprocess, "check_output", lambda *a, **k: m.SOURCE + "\n"),
        patch.object(f, "resolve", lambda *a: (None, expansion, None)),
        patch.object(f, "verify_artifacts", m.verify_artifacts),
        patch.object(f.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected real RPC"))),
    ):
        plan = f.finish(*args, "plan")
        assert len(plan["requests"]) == 1
        collected = f.finish(*args, "collect")
        assert collected["tasks_collected_or_reused"] == len(join["tasks"])
        integrated = f.finish(*args, "verify", 0)
        assert integrated["scientifically_complete"]
        m.wandb.Api = lambda: (_ for _ in ()).throw(AssertionError("Completed pipeline queried API"))
        assert f.finish(*args, "verify", 0) == integrated
        assert not f.finish(*args, "plan")["requests"]
        assert f.finish(*args, "collect")["tasks_collected_or_reused"] == 0
        changed = copy.deepcopy(observed)
        changed["jobs"]["0"]["terminal"] = False
        op.write_text(json.dumps(changed))
        try:
            f.finish(*args, "verify", 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Cached completion bypassed saved terminal outcome")
        op.write_text(json.dumps(observed))
        original = canonical.read_bytes()
        canonical.write_text("{}")
        try:
            f.finish(*args, "verify", 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Cached completion bypassed canonical evidence integrity")
        canonical.write_bytes(original)
    print(
        "Offline expansion artifact fixture passed with",
        len(join["tasks"]),
        "hosts and ten negative artifact cases; cache reuse made no API query.",
    )
    (R / ("scratch/20261105-expansion-artifacts-" + str(len(join["tasks"])) + "host-draft-preflight.json")).write_text(
        json.dumps(
            dict(
                passed=True,
                hosts=len(join["tasks"]),
                negative_cases=[r[0] for r in cases],
                scope="Mocked W&B/storage/counter; fabricated terminal expansion.",
                integration=(
                    "Plan, cached task collection, full verify and completed skip passed; "
                    "Changed terminal owner/canonical evidence rejected. "
                    "Git/source resolve mocked; artifact validator exercised."
                ),
                remaining=["First real terminal expansion validation"],
            ),
            indent=2,
        )
        + "\n"
    )
