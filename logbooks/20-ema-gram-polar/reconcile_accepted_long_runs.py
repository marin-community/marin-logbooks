# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay final pilot selection and account for accepted runs without remote calls."""

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from adopt_alpha0_long_run import adoption
from prepare_parallel_long_runs import prepare as prepare_parallel
from prepare_subset_confirmations import materialize_cells
from prepare_subset_expansions import accepted_bindings, next_cells
from submit_subset_confirmations import validate_completions
from watch_subset_confirmations import accepted_entries

from experiments.ema_gram.plan import Cell, Result, confirmations, lock_settings

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reconcile(root=ROOT):
    base = root / "logbooks/20-ema-gram-polar"
    reports = ["subset-training-r02-alpha0-completed.json", "subset-training-r02-nonzero-completed.json"]
    reports += [f"subset-expansion-r{n:02d}-completed.json" for n in range(1, 7)]
    evidence, rows = {}, []
    for name in reports:
        raw = (base / name).read_bytes()
        evidence[str(Path("logbooks/20-ema-gram-polar") / name)] = digest(raw)
        rows.extend(json.loads(raw)["results"])
    records = {"results": rows}
    # The final accepted expansion is bound by its published launch receipt.
    launched = json.loads((base / "subset-expansion-r06-launched.json").read_text())
    predecessors = json.loads((base / "subset-expansion-r06-accepted.json").read_text())
    expansion_entries = predecessors + list(launched["accepted"].values())
    accepted_bindings(records, expansion_entries)
    candidates, groups = next_cells(records)
    if candidates or len(rows) != 37 or any(g["state"] != "interior_winner" for g in groups):
        raise ValueError("Recursive pilot selection remains incomplete")
    cells = [Cell(**r["cell"]) for r in rows]
    results = [
        Result(c.name, "completed", r["final_c4_loss"], r["stored_native_counter"], r["expected_steps"], r["run_url"])
        for c, r in zip(cells, rows, strict=True)
    ]
    protocol = json.loads((base / "alpha0-long-run-protocol-draft.json").read_text())
    lock = lock_settings(cells, results, protocol["common_loss_target"])
    validate_completions(lock, records)
    core = [c for c in confirmations(lock) if c.seed == 0]
    # The explicit parallel-start amendment adds the best other exponent even
    # when the original global rule's overall winner is the alpha0 baseline.
    other = min((r for r in rows if r["cell"]["alpha"] in (0.125, 1 / 3)), key=lambda r: r["final_c4_loss"])
    extra = replace(Cell(**other["cell"]), phase="confirmation", seed=0)
    selected = sorted({c.alpha: c for c in [*core, extra]}.values(), key=lambda c: c.alpha)
    receipt_path = base / "parallel-long-runs-r01-accepted.json"
    receipt = json.loads(receipt_path.read_text())
    source = receipt["source_commit"]
    ledger = receipt["ledger"]
    if ledger["binding"]["source"] != source or not receipt["accepted"]:
        raise ValueError("Parallel acceptance source mismatch")
    committed = {}
    for path, sha in ledger["binding"]["inputs"].items():
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Invalid accepted input path")
        raw = subprocess.check_output(["git", "show", f"{source}:{path}"], cwd=root)
        if digest(raw) != sha or (root / path).read_bytes() != raw:
            raise ValueError("Accepted input differs from original source")
        committed[Path(path).name] = json.loads(raw)
    bundle = committed["parallel-long-runs-r01-bundle.json"]
    amendment = committed["parallel-long-runs-r01-amendment.json"]
    if bundle != prepare_parallel(
        committed["parallel-long-runs-r01-completions.json"], amendment, bundle["artifact_version"]
    ):
        raise ValueError("Accepted parallel bundle replay differs")
    if amendment["common_loss_target"] != lock.loss_target:
        raise ValueError("Accepted target differs from final selection")
    nonzero = [c for c in selected if c.alpha != 0]
    rebuilt = materialize_cells(nonzero, bundle["artifact_version"])
    if rebuilt["cells"] != bundle["cells"] or rebuilt["source_files"] != bundle["source_files"]:
        raise ValueError("Accepted nonzero configuration differs from final pilot selection")
    relative = Path("logbooks/20-ema-gram-polar")
    accepted = accepted_entries(
        ledger,
        ledger["binding"],
        receipt["campaign"],
        bundle,
        relative / "parallel-long-runs-r01-bundle.json",
        None,
        relative / "parallel-long-runs-r01-amendment.json",
        relative / "parallel-long-runs-r01-completions.json",
    )
    if set(accepted) != {"0", "1", "2"}:
        raise ValueError("Parallel acceptance is incomplete or ambiguous")
    baseline_receipt = json.loads((base / "alpha0-long-run-accepted.json").read_text())
    baseline_bundle = materialize_cells([c for c in selected if c.alpha == 0], "2026.11.05.14")
    _, baseline = adoption(root, baseline_receipt, lock, baseline_bundle)
    accounting = [
        dict(
            alpha=0,
            disposition="reuse_accepted",
            job_id=baseline["job_id"],
            source_commit=baseline["source_commit"],
            config_sha256=baseline["config_sha256"],
            artifact_version=baseline["artifact_version"],
            monitoring_owner="existing_baseline_campaign",
        )
    ]
    accounting += [
        dict(
            alpha=bundle["cells"][int(k)]["cell"]["alpha"],
            disposition="reuse_accepted",
            **v,
            artifact_version=bundle["artifact_version"],
            monitoring_owner="existing_parallel_campaign",
        )
        for k, v in accepted.items()
    ]
    if len({r["job_id"] for r in accounting}) != 4:
        raise ValueError("Accepted identity reused")
    return asdict(lock), dict(
        pilot_completion_reports=evidence,
        validated_pilot_count=len(rows),
        groups=groups,
        selection_lock_digest=lock.sha256,
        common_loss_target=lock.loss_target,
        selection_uses_only_pilot_outcomes=True,
        core_selected=[asdict(c) for c in core],
        authorized_additional_cell=asdict(extra),
        selected=[asdict(c) for c in selected],
        additional_cell_reason=(
            "Best other exponent under explicit parallel-start amendment; original core lock unchanged"
        ),
        accepted_receipts={
            str(relative / p.name): digest(p.read_bytes())
            for p in [receipt_path, base / "alpha0-long-run-accepted.json"]
        },
        confirmation_accounting=accounting,
        new_submissions_required=[],
        new_observers_required=[],
        provisional_alphas=[],
        scope=(
            "Exact original source/configuration/acceptance accounting; completion and startup evidence remain separate"
        ),
    )


if __name__ == "__main__":
    lock, accounting = reconcile()
    for name, value in [
        ("subset-final-selection-lock.json", lock),
        ("subset-final-confirmation-accounting.json", accounting),
    ]:
        with (BASE / name).open("x") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
