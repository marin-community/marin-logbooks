# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Account for an accepted early baseline without resubmitting or observing it twice."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from prepare_subset_confirmations import materialize_cells

from experiments.ema_gram.plan import Cell


def adoption(worktree, receipt, lock, bundle):
    """Replay committed baseline inputs and join its accepted ledger to the final lock."""
    # The baseline entrypoint reuses the confirmation ledger primitives.
    # Import here to break that orchestration dependency cycle.
    from alpha0_long_run import accepted_entries, command, validate_protocol  # noqa: PLC0415
    from prepare_alpha0_long_run import prepare as prepare_baseline  # noqa: PLC0415

    source, campaign = receipt["source"], receipt["campaign"]
    if not re.fullmatch("[0-9a-f]{40}", source) or not re.fullmatch("ema-gram-20-alpha0-seed0-[a-z0-9-]+", campaign):
        raise ValueError("Invalid accepted baseline identity")
    paths = {key: Path(receipt["paths"][key]) for key in ("bundle", "completions", "protocol")}
    values, hashes = {}, {}
    for key, path in paths.items():
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Baseline inputs must be committed relative paths")
        raw = subprocess.check_output(["git", "show", f"{source}:{path}"], cwd=worktree)
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        values[key] = json.loads(raw)
    baseline = values["bundle"]
    if baseline != prepare_baseline(values["completions"], baseline["artifact_version"]):
        raise ValueError("Accepted baseline no longer matches its complete pilot evidence")
    protocol = values["protocol"]
    validate_protocol(protocol, baseline)
    if protocol.get("qualification_only") or lock.loss_target != protocol["common_loss_target"]:
        raise ValueError("Final confirmation target differs from the predeclared research target")
    indexes = [i for i, row in enumerate(bundle["cells"]) if row["cell"]["alpha"] == 0]
    if len(indexes) != 1:
        raise ValueError("Final confirmation set must contain exactly one baseline")
    index = indexes[0]
    expected = materialize_cells([Cell(**bundle["cells"][index]["cell"])], baseline["artifact_version"])
    if baseline["cells"] != expected["cells"] or baseline["source_files"] != expected["source_files"]:
        raise ValueError("Accepted baseline differs from the final selected numerical configuration")
    binding = dict(source=source, inputs=hashes)
    argv = command(source, campaign, paths["bundle"], paths["completions"], paths["protocol"])
    entries = accepted_entries(receipt["ledger"], binding, campaign, argv, baseline)
    if set(entries) != {"0"}:
        raise ValueError("Baseline acceptance is absent or ambiguous; preserve its existing guard")
    return index, dict(
        **entries["0"],
        artifact_version=baseline["artifact_version"],
        config=baseline["cells"][0]["config"],
        protocol=protocol,
        monitoring_owner="existing_baseline_campaign",
        scientifically_complete=False,
        completion_requires="Full independent long-run terminal, metric, ONLINE and checkpoint validation",
    )


def accounting(worktree, receipt_path, lock, bundle):
    """Return final-set accounting while leaving the global selection checks intact."""
    if (worktree / "logbooks/20-ema-gram-polar/parallel-long-runs-r01-accepted.json").exists():
        raise ValueError(
            "Parallel long runs already accepted: use published final accounting and existing observers; "
            "ordinary global submission/observation would duplicate accepted identities"
        )
    rows = [dict(index=i, disposition="submit", config_sha256=r["config_sha256"]) for i, r in enumerate(bundle["cells"])]
    if receipt_path is None:
        protocol_path = worktree / "logbooks/20-ema-gram-polar/alpha0-long-run-protocol-draft.json"
        protocol = json.loads(protocol_path.read_text())
        if protocol["allow_settled_alpha0_early_start"] or protocol["status"] != "draft":
            raise ValueError("Early-start protocol exists: explicitly reconcile the baseline before global submission")
        return rows
    receipt = json.loads((worktree / receipt_path).read_text())
    index, accepted = adoption(worktree, receipt, lock, bundle)
    rows[index] = dict(index=index, disposition="adopt", **accepted)
    return rows
