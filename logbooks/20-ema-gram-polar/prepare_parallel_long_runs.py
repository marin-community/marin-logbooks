# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the user-authorized parallel long runs before recursive tuning ends."""

import hashlib
import json
from dataclasses import replace

from marin.execution.artifact import is_mutable_version, validate_version
from prepare_subset_confirmations import materialize_cells

from experiments.ema_gram.plan import CONFIRMATION, SELECTION_METRIC, Cell
from experiments.ema_gram.subset_campaign import CELLS


def prepare(records, amendment, version):
    # Expansion validation imports the submitter's completion checks.
    from prepare_subset_expansions import accepted_bindings, next_cells  # noqa: PLC0415

    validate_version(version)
    if is_mutable_version(version):
        raise ValueError("Use an immutable long-run artifact version")
    if amendment["schema"] != "parallel-long-runs-v1" or amendment["status"] != "frozen":
        raise ValueError("Require the frozen parallel-start amendment")
    if amendment["user_instruction"] != "全部启动启动启动":
        raise ValueError("Missing explicit parallel-start authorization")
    if amendment["global_selection_locked"] or amendment["confirmation_outcomes_used_for_selection"]:
        raise ValueError("Early long runs cannot declare a global lock or select pilot settings")
    if amendment["common_loss_target"] != 4.0 or amendment["selection_metric"] != SELECTION_METRIC:
        raise ValueError("Preserve the previously declared held-out target")
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if digest != amendment["predecessor_records_sha256"]:
        raise ValueError("Pilot evidence differs from the authorization snapshot")
    names = {Cell(**r["cell"]).name for r in records["results"]}
    if not {c.name for c in CELLS}.issubset(names):
        raise ValueError("All original pilots must be validated")
    accepted_bindings(records, amendment["accepted_expansions"])
    _, groups = next_cells(records)
    rows = records["results"]
    winners = []
    for alpha in (0.125, 0.25, 0.5):
        candidates = [r for r in rows if r["cell"]["alpha"] == alpha]
        winner = min(candidates, key=lambda r: r["final_c4_loss"])
        relevant = [g for g in groups if g["alpha"] == alpha]
        if alpha != 0.5 and any(g["state"] != "interior_winner" for g in relevant):
            raise ValueError("Only alpha0.5 may retain an unresolved ridge search")
        pilot = Cell(**winner["cell"])
        if not any(g.get("winner") == winner["cell"] for g in relevant):
            raise ValueError("The chosen configuration must have an interior ridge-group winner")
        winners.append(pilot)
    # The extra exponent is determined only by completed pilot evidence.
    best_other = min((r for r in rows if r["cell"]["alpha"] in (0.125, 1 / 3)), key=lambda r: r["final_c4_loss"])
    if best_other["cell"]["alpha"] != 0.125:
        raise ValueError("The authorized extra exponent no longer matches pilot evidence")
    cells = [replace(c, phase=CONFIRMATION, seed=0) for c in winners]
    bundle = materialize_cells(cells, version)
    bundle.update(
        amendment_digest=hashlib.sha256(json.dumps(amendment, sort_keys=True).encode()).hexdigest(),
        predecessor_records_sha256=digest,
        global_selection_locked=False,
        common_loss_target=4.0,
        selection_groups=groups,
        provisional_alphas=[0.5],
        submitted=False,
    )
    if [r["cell"] for r in bundle["cells"]] != amendment["selected_cells"]:
        raise ValueError("Selected configurations changed after authorization snapshot")
    return bundle


def resolve_bundle(bundle_path, completions_path, amendment_path):
    bundle = json.loads(bundle_path.read_text())
    records = json.loads(completions_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    if bundle != prepare(records, amendment, bundle["artifact_version"]):
        raise ValueError("Frozen parallel bundle differs from validated pilot selection")
    return bundle


def accounting(worktree, receipt_path, bundle, amendment_path):
    """Keep the completed baseline outside this new submission and observer set."""
    amendment = json.loads((worktree / amendment_path).read_text())
    if receipt_path is None or str(receipt_path) != amendment["baseline_receipt"]["path"]:
        raise ValueError("Bind the original baseline receipt; never submit another baseline")
    if hashlib.sha256((worktree / receipt_path).read_bytes()).hexdigest() != amendment["baseline_receipt"]["sha256"]:
        raise ValueError("Original baseline receipt changed")
    if any(r["cell"]["alpha"] == 0 for r in bundle["cells"]):
        raise ValueError("Completed alpha0 must not enter the new submission set")
    return [dict(index=i, disposition="submit", config_sha256=r["config_sha256"]) for i, r in enumerate(bundle["cells"])]
