# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline terminal request preparation for one frozen expansion campaign."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from submit_subset_confirmations import validate_completions
from watch_subset_expansions import TERMINAL, accepted_entries

from experiments.ema_gram.plan import Cell, Result


def prepare(worktree, registry, campaign, bundle_path, joins, completed):
    binding = registry["campaigns"][campaign]
    source = binding["source"]
    bundle_path = Path(bundle_path)
    if bundle_path.is_absolute() or ".." in bundle_path.parts or str(bundle_path) not in binding["inputs"]:
        raise ValueError("Bundle not bound to accepted campaign")
    for name, sha in binding["inputs"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or hashlib.sha256((worktree / path).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen campaign input changed")
    bundle = json.loads((worktree / bundle_path).read_text())
    bundle_sha = hashlib.sha256((worktree / bundle_path).read_bytes()).hexdigest()
    worker_sha = hashlib.sha256((worktree / "logbooks/20-ema-gram-polar/expansion_worker.py").read_bytes()).hexdigest()
    accepted = accepted_entries(registry, campaign, source, bundle)
    observer_path = worktree / "scratch" / f"{campaign}-observers.json"
    observed = json.loads(observer_path.read_text())
    if observed["binding"] != binding:
        raise ValueError("Observer campaign input binding differs")
    requests, waiting, reused = [], [], []
    for key, entry in accepted.items():
        index = int(key)
        row = bundle["cells"][index]
        if key not in observed["jobs"]:
            waiting.append(dict(index=index, reason="Observer has not discovered this accepted identity"))
            continue
        obs = observed["jobs"][key]
        for field in ["job_id", "source_commit", "config_sha256"]:
            if obs[field] != entry[field]:
                raise ValueError("Observer entry identity differs")
        verified = completed.get(entry["job_id"])
        if verified is not None:
            if not obs["terminal"] or obs.get("latest", {}).get("job") != verified["canonical_job"]:
                raise ValueError("Saved terminal owner outcome differs from completion")
            if (
                not verified["scientifically_complete"]
                or verified["source"] != source
                or verified["cell"] != row["cell"]
                or verified["job_id"] != entry["job_id"]
            ):
                raise ValueError("Cached completion identity differs")
            if (
                verified.get("config_sha256") != row["config_sha256"]
                or not verified["evidence"]
                or verified["stored_native_counter"] != 7630
                or not verified["exact_subset_config_verified"]
                or verified["online_state"] != "finished"
            ):
                raise ValueError("Cached completion lacks full validation gates")
            result = Result(
                Cell(**verified["cell"]).name,
                "completed",
                verified["final_c4_loss"],
                verified["stored_native_counter"],
                verified["expected_steps"],
                verified["run_url"],
            )
            validate_completions(SimpleNamespace(results=[result]), {"results": [verified]})
            for name, digest in verified["evidence"].items():
                payload = Path(name).read_bytes()
                if len(payload) != digest["bytes"] or hashlib.sha256(payload).hexdigest() != digest["sha256"]:
                    raise ValueError("Cached completed evidence changed")
            reused.append(entry["job_id"])
            continue
        latest = obs.get("latest", {})
        if "job" not in latest:
            waiting.append(dict(index=index, reason="No successful latest canonical snapshot"))
            continue
        raw = Path(latest["path"]).read_bytes()
        if json.loads(raw)["job"] != latest["job"]:
            raise ValueError("Canonical snapshot differs from owner record")
        job = latest["job"]
        if job["job_id"] != entry["job_id"]:
            raise ValueError("Canonical identity differs")
        if job["state"] not in TERMINAL:
            waiting.append(dict(index=index, reason=job["state"]))
            continue
        if not obs["terminal"]:
            raise ValueError("Owner terminal flag not reconciled")
        task_count = 1 if row["cell"]["alpha"] == 0 else 4
        if job["task_count"] != task_count:
            raise ValueError("Wrong task topology")
        requests.append(
            dict(
                index=index,
                campaign=campaign,
                source_commit=source,
                job_id=entry["job_id"],
                canonical_state=job["state"],
                canonical_exit_code=job["exit_code"],
                canonical_job=job,
                canonical_evidence=latest["path"],
                canonical_sha256=hashlib.sha256(raw).hexdigest(),
                failure_count=job["failure_count"],
                preemption_count=job["preemption_count"],
                task_ids=[f"{entry['job_id']}/{i}" for i in range(task_count)],
                cached_startup_join=joins.get(entry["job_id"]),
                config=row["config"],
                config_sha256=row["config_sha256"],
                cell=row["cell"],
                input_binding=binding,
                bundle_sha256=bundle_sha,
                worker_sha256=worker_sha,
                scientifically_complete=False,
            )
        )
    return dict(
        campaign=campaign,
        source_commit=source,
        requests=requests,
        waiting=waiting,
        completed_reused=reused,
        scope="Saved local bindings/canonical snapshots only; no RPC or completion claim",
    )
