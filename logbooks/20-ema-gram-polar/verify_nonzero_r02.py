# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate one terminal R02 nonzero pilot; fail closed on unadjudicated restarts."""
import argparse
import fcntl
import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import draccus
import google.auth
import tensorstore as ts
import wandb
from audit_subset_history import FINAL_KEY, TRAIN_KEYS, audit
from collect_r02_terminal_tasks import save_new
from google.auth.transport.requests import Request
from google.cloud import storage
from levanter.main.train_lm import TrainLmConfig
from prepare_r02_terminal_collection import SOURCE, prepare
from wandb.util import json_friendly_val

from experiments.ema_gram.report import validate_run_config

FROZEN_CONFIGS_SHA256 = "5b9c72f1eef255b134bfd1e5d4bf2cacc4ae967bfe0d98e368866bb71fbafce1"


def cached_json(path, obtain):
    if path.exists():
        return json.loads(path.read_text())
    value = obtain()
    save_new(path, value)
    return value


def require_single_attempt_tasks(request, directory, join):
    if request["canonical_state"] != "JOB_STATE_SUCCEEDED" or request["failure_count"] or request["preemption_count"]:
        raise ValueError("Failed or retried job requires separate diagnosis and lineage adjudication")
    if join is None or join["source_commit"] != SOURCE:
        raise ValueError("Missing qualified startup attempt joins")
    tasks = []
    for number, task_id in enumerate(request["task_ids"]):
        record = json.loads((directory / (hashlib.sha256(task_id.encode()).hexdigest() + ".json")).read_text())
        if record["task_id"] != task_id:
            raise ValueError("Terminal task cache identity mismatch")
        raw = Path(record["raw_evidence"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
            raise ValueError("Terminal task raw evidence changed")
        task = record["task"]
        if json.loads(json.loads(raw)["stdout"])["task"] != task:
            raise ValueError("Terminal task differs from saved RPC response")
        attempts = task["attempts"]
        if task["task_id"] != task_id or task["state"] != "TASK_STATE_SUCCEEDED" or task["exit_code"] or task["error"]:
            raise ValueError("Task did not succeed")
        if len(attempts) != 1 or task["current_attempt_id"] != 0:
            raise ValueError("Multiple attempts need explicit full-state restart/resume adjudication")
        attempt = attempts[0]
        if (
            attempt["attempt_id"] != 0
            or attempt["state"] != "TASK_STATE_SUCCEEDED"
            or attempt["exit_code"]
            or attempt["error"]
        ):
            raise ValueError("Attempt did not succeed")
        startup = next(t for t in join["tasks"] if t["task"] == number)
        if attempt["attempt_uid"] != startup["attempt_uid"] or task["worker_id"] != startup["worker_id"]:
            raise ValueError("Terminal task no longer matches verified startup attempt")
        tasks.append(task)
    if len(tasks) != 4 or len({t["worker_id"] for t in tasks}) != 4:
        raise ValueError("Expected four distinct assigned hosts")
    return tasks


def canonical_history(raw, repair):
    unique, duplicates = {}, []
    for row in raw:
        label = row["global_step"]
        if isinstance(label, bool) or not isinstance(label, (int, float)) or int(label) != label:
            raise ValueError("Invalid native label")
        label = int(label)
        if label in unique:
            if unique[label] != row:
                raise ValueError("Conflicting duplicate native metric row")
            duplicates.append(label)
        unique[label] = row
    if set(unique) - set(range(7630)):
        raise ValueError("Unexpected native labels")
    missing = sorted(set(range(7630)) - set(unique))
    for label in missing:
        found = [r for r in repair(label) if r["global_step"] == label]
        if len(found) != 1:
            raise ValueError(f"Unable to recover unique native label {label}")
        unique[label] = found[0]
    return [unique[k] for k in sorted(unique)], {"identical_duplicates": duplicates, "repaired_labels": missing}


def verify(root, index):
    if index not in range(6, 30):
        raise ValueError("Only unfinished nonzero R02 pilots are eligible; alpha0 results must be reused")
    execution = root.parent / "ema-gram-subset-seed0-r02-20261105"
    observer_path = execution / "scratch/20261105-subset-training-r02-observers.json"
    ledger_path = execution / "scratch/20261105-subset-training-r02-ledger.json"
    obs = json.loads(observer_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    completed = json.loads((root / "logbooks/20-ema-gram-polar/subset-training-r02-alpha0-completed.json").read_text())
    joins = json.loads((root / "scratch/20261105-r02-attempt-joins/validated.json").read_text())
    plan = prepare(obs, ledger, completed, joins)
    requests = [r for r in plan["terminal_collection_requests"] if r["index"] == index]
    if len(requests) != 1:
        raise ValueError("No canonical terminal request for this cell; no online or storage reads permitted")
    request = requests[0]
    if obs[str(index)]["latest"]["job"]["exit_code"] != 0:
        raise ValueError("Canonical successful job has nonzero exit code")
    join = request["cached_startup_join"]
    tasks = require_single_attempt_tasks(request, root / "scratch/20261105-r02-terminal-tasks", join)
    configs_path = root / "scratch/20261104-subset-all30-configs.json"
    if hashlib.sha256(configs_path.read_bytes()).hexdigest() != FROZEN_CONFIGS_SHA256:
        raise ValueError("Frozen campaign configuration file changed")
    expected = json.loads(configs_path.read_text())[index]["config"]["train_config"]
    run_ids = list(join["native_run_tasks"])
    if len(run_ids) != 1:
        raise ValueError("Expected one verified native run")
    rid = run_ids[0]
    out = root / f"scratch/20261105-r02-{index:02d}-completion"
    out.mkdir(exist_ok=True)
    with (out / "owner.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = {
            "index": index,
            "job_id": request["job_id"],
            "run_id": rid,
            "source": SOURCE,
            "frozen_configs_sha256": hashlib.sha256(configs_path.read_bytes()).hexdigest(),
        }
        if cached_json(out / "binding.json", lambda: binding) != binding:
            raise ValueError("Completion artifacts bound to different inputs")
        if (out / "verified.json").exists():
            verified = json.loads((out / "verified.json").read_text())
            for path, recorded in verified["evidence"].items():
                payload = Path(path).read_bytes()
                if hashlib.sha256(payload).hexdigest() != recorded["sha256"] or len(payload) != recorded["bytes"]:
                    raise ValueError("Previously verified local evidence changed")
            return verified
        # All canonical and attempt checks happen before creating the online API client.
        run = wandb.Api().run("marin-community/marin/" + rid)
        if run.state != "finished" or run.config["git_commit"] != SOURCE:
            raise ValueError("Native ONLINE run is not finished on the accepted source")
        actual = cached_json(out / "config.json", lambda: dict(run.config))
        if actual["git_commit"] != SOURCE:
            raise ValueError("Cached ONLINE config source differs")
        cfg = draccus.decode(TrainLmConfig, expected)
        cfg = replace(
            cfg,
            trainer=replace(
                cfg.trainer,
                jax_config=expected["trainer"]["jax_config"],
                load_checkpoint_path=expected["trainer"]["load_checkpoint_path"],
            ),
        )
        validate_run_config(actual, cfg)
        native_expected = json_friendly_val(asdict(cfg))
        trackers = [t for t in actual["trainer"]["tracker"] if "mode" in t]
        if (
            len(trackers) != 1
            or trackers[0]["mode"] != "online"
            or trackers[0]["resume"] != "never"
            or trackers[0]["id"] != rid
            or actual["trainer"]["id"] != expected["trainer"]["id"]
            or not rid.startswith(expected["trainer"]["id"] + "-")
        ):
            raise ValueError("Native tracking is not the expected fresh ONLINE identity")
        for key in ["load_checkpoint_path", "checkpointer"]:
            if actual["trainer"][key] != native_expected["trainer"][key]:
                raise ValueError(f"Checkpoint config differs: {key}")
        raw = cached_json(
            out / "history-raw.json", lambda: list(run.scan_history(keys=["global_step", *TRAIN_KEYS], page_size=1000))
        )
        evaluations = cached_json(
            out / "eval-raw.json", lambda: list(run.scan_history(keys=["global_step", FINAL_KEY], page_size=1000))
        )

        def repair(label):
            return cached_json(
                out / f"pagination-label-{label}.json",
                lambda: list(
                    run.scan_history(keys=["global_step", *TRAIN_KEYS], min_step=label, max_step=label + 1, page_size=10)
                ),
            )

        rows, pagination = canonical_history(raw, repair)
        if cached_json(out / "pagination.json", lambda: pagination) != pagination:
            raise ValueError("Cached pagination audit differs from raw rows")
        if cached_json(out / "history-canonical.json", lambda: rows) != rows:
            raise ValueError("Cached canonical history differs from raw rows")
        gates = audit(rows + evaluations)
        progress_state = json.loads((root / "scratch/20261030_ema_gram_monitoring_state.json").read_text())[
            "single_seed_subset_rebuild"
        ]
        progress = json.loads(Path(progress_state["training_progress_evidence"]).read_text())
        host = progress["cells"][str(index)]["host_logging"]
        if {h["id"] for h in host} != {t["host_logging_id"] for t in join["tasks"]} or any(
            h["state"] != "finished" or not h["source_files_verified"] for h in host
        ):
            raise ValueError("Missing qualified ONLINE host logging")
        uri = expected["trainer"]["checkpointer"]["base_path"] + "/step-7629"
        if not uri.startswith("gs://marin-us-central2/") or "/tmp/" in uri:
            raise ValueError("Expected permanent regional checkpoint")
        bucket, path = uri[5:].split("/", 1)
        client = storage.Client()

        def read_metadata(name):
            blob = client.bucket(bucket).get_blob(path + "/" + name)
            if blob is None or blob.size >= 1_000_000:
                raise ValueError("Missing or oversized checkpoint metadata")
            raw_bytes = blob.download_as_bytes(if_generation_match=blob.generation, checksum="crc32c")
            return {
                "parsed": json.loads(raw_bytes),
                "generation": blob.generation,
                "crc32c": blob.crc32c,
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "size": len(raw_bytes),
                "raw_text": raw_bytes.decode(),
            }

        metadata = cached_json(out / "metadata.json", lambda: read_metadata("metadata.json"))
        manifest = cached_json(out / "manifest.json", lambda: read_metadata("manifest.json"))
        if metadata["parsed"]["step"] != 7629 or metadata["parsed"]["is_temporary"]:
            raise ValueError("Checkpoint metadata does not describe final permanent label")
        for record in [metadata, manifest]:
            raw_bytes = record["raw_text"].encode()
            if (
                hashlib.sha256(raw_bytes).hexdigest() != record["sha256"]
                or len(raw_bytes) != record["size"]
                or json.loads(raw_bytes) != record["parsed"]
            ):
                raise ValueError("Checkpoint metadata cache failed raw-content verification")
        reference_bytes = (root / "scratch/20261105-r02-00-final-manifest.json").read_bytes()
        reference_result = next(r for r in completed["results"] if r["index"] == 0)
        if (
            hashlib.sha256(reference_bytes).hexdigest()
            != reference_result["checkpoint"]["files"]["manifest.json"]["sha256"]
        ):
            raise ValueError("Reference array schema differs from published completion")
        reference_manifest = json.loads(reference_bytes)
        if manifest["parsed"] != reference_manifest:
            raise ValueError("Checkpoint array schema differs; inspect before scientific validation")

        def read_counter():
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
            creds.refresh(Request())
            spec = {
                "driver": "zarr3",
                "kvstore": {
                    "driver": "ocdbt",
                    "base": {
                        "driver": "http",
                        "base_url": "https://storage.googleapis.com/" + uri[5:] + "/",
                        "headers": ["Authorization: Bearer " + creds.token],
                    },
                    "path": "step",
                },
            }
            try:
                arr = ts.open(spec, open=True, read=True, write=False).result().read().result()
            except Exception as exc:
                raise RuntimeError(str(exc).replace(creds.token, "[REDACTED]")) from None
            if arr.shape != () or int(arr) != 7630:
                raise ValueError("Stored native counter is not7630")
            return {
                "counter": int(arr),
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "scalar_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "method": "Same ADC read-only HTTP scalar read; no full checkpoint byte audit",
            }

        counter = cached_json(out / "counter.json", read_counter)
        if counter["counter"] != 7630 or counter["shape"] != []:
            raise ValueError("Cached scalar counter mismatch")
        job = obs[str(index)]["latest"]["job"]
        gates.update(
            index=index,
            source=SOURCE,
            online_state=run.state,
            run_id=rid,
            run_url=run.url,
            exact_subset_config_verified=True,
            stored_native_counter=counter["counter"],
            stored_counter_dtype=counter["dtype"],
            counter_read=counter,
            checkpoint={
                "uri": uri,
                "metadata": metadata["parsed"],
                "arrays": len(manifest["parsed"]["arrays"]),
                "files": {
                    name: {k: v for k, v in record.items() if k not in ["parsed", "raw_text"]}
                    for name, record in [("metadata.json", metadata), ("manifest.json", manifest)]
                },
            },
            scientifically_complete=True,
            remaining_gates=[],
            cell=ledger["cells"][str(index)]["cell"],
            job_id=request["job_id"],
            canonical_job=job,
            tasks=tasks,
            host_logging=host,
            run_attempt_join_evidence=join,
            verified_at=datetime.now(UTC).isoformat(),
            scope=(
                "Validated fixed-subset pilot with single fresh attempt on each host; "
                "no full checkpoint audit or resume claim."
            ),
        )
        gates["evidence"] = {
            str(p): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
            for p in out.glob("*.json")
        }
        save_new(out / "verified.json", gates)
        return gates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(Path.cwd(), args.index), indent=2))
