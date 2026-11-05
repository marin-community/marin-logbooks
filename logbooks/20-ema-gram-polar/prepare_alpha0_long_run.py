# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the settled alpha0 long-run configuration without fabricating a global lock."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from marin.execution.artifact import is_mutable_version, validate_version
from prepare_subset_confirmations import materialize_cells
from prepare_subset_expansions import accepted_bindings, next_cells

from experiments.ema_gram.plan import CONFIRMATION, Cell
from experiments.ema_gram.subset_campaign import CELLS


def prepare(records, version):
    validate_version(version)
    if is_mutable_version(version):
        raise ValueError("Use an immutable long-run artifact version")
    expected = {cell.name for cell in CELLS if cell.alpha == 0}
    actual = {Cell(**row["cell"]).name for row in records["results"]}
    if actual != expected or len(records["results"]) != 6:
        raise ValueError("Require exactly the six validated alpha0 pilots")
    accepted_bindings(records, [])
    _, groups = next_cells(records)
    baseline = next(group for group in groups if group["alpha"] == 0)
    if baseline["state"] != "interior_winner":
        raise ValueError("Alpha0 still requires a learning-rate expansion")
    pilot = Cell(**baseline["winner"])
    cell = replace(pilot, phase=CONFIRMATION, seed=0)
    bundle = materialize_cells([cell], version)
    bundle.update(
        prepared_only=True,
        submitted=False,
        launch_ready=False,
        global_selection_locked=False,
        selected_group=baseline,
        predecessor_records_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        common_loss_target=None,
        remaining_launch_gates=[
            "Predeclare the common held-out loss target and record any amendment allowing the baseline to start early",
            "Bind the settled baseline to an independent submitter/observer and permanent campaign launch guard",
            "Publish exact execution source and pass fresh host ONLINE logging plus full regional cache checks",
        ],
    )
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = prepare(json.loads(args.completions.read_text()), args.version)
    with args.output.open("x") as stream:
        stream.write(json.dumps(bundle, indent=2) + "\n")


if __name__ == "__main__":
    main()
