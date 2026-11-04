"""Rebuild the original A13 packed input from four pinned captured inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow

from experiments.post_training.identity_path_b.ablation_data import pack_prefix
from experiments.post_training.identity_path_b.trace_schedule import compact_row

EXPECTED_PACKED_SHA256 = "f860030d1266c90d146ef07995a95ce7501989727cf226217de82a87408c15ba"
TOOL_FAMILIES = {"tool_first_call", "tool_call_after_result", "tool_reply"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("captured-inputs.json"))
    args = parser.parse_args()
    if pyarrow.__version__ != "23.0.1":
        raise ValueError("Exact Parquet bytes require the original PyArrow 23.0.1 writer")
    manifest = json.loads(args.manifest.read_bytes())
    paths = {}
    for label, entry in manifest["inputs"].items():
        path = args.input_root / entry["name"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Captured input differs from its original pin: {label}")
        paths[label] = path
    schedule = [json.loads(line) for line in paths["a8a_ledger"].read_bytes().splitlines()]
    if len(schedule) != 7266 or sum(e["source"] == "identity" and e["enable_thinking"] for e in schedule) != 179:
        raise ValueError("Expected the original consumed A8a schedule and its 179 retained ON decisions")
    needed = {entry["row_id"] for entry in schedule}
    on = {entry["row_id"] for entry in schedule if entry["source"] == "identity" and entry["enable_thinking"]}
    rows = {}
    for label in ("identity_base", "identity_traces", "replay"):
        with paths[label].open("rb") as stream:
            for line in stream:
                raw = json.loads(line)
                if raw["id"] not in needed:
                    continue
                if label == "identity_base" and raw["id"] in on:
                    continue
                if label == "identity_traces" and raw["id"] not in on:
                    continue
                row = compact_row(raw)
                rows[row.row_id] = row
    if set(rows) != needed:
        raise ValueError("The captured pools do not cover the complete A8a occurrence schedule")
    doubled = []
    for entry in schedule:
        row = rows[entry["row_id"]]
        doubled.append(dict(entry, parent_A8a_occurrence=entry["occurrence"], copy_index=0))
        if entry["source"] == "replay" and entry["enable_thinking"] and row.family in TOOL_FAMILIES:
            doubled.append(dict(entry, parent_A8a_occurrence=entry["occurrence"], copy_index=1))
    report = pack_prefix(rows, doubled, args.output, truncate=True, updates=36)
    actual = digest(args.output / "packed.parquet")
    if actual != EXPECTED_PACKED_SHA256:
        raise ValueError(f"A13 packed input mismatch: {actual}")
    receipt = {
        "status": "PASS",
        "source_revision": "69ddc88fb2ff4a6e45ee5cbbec866d9f86c2d068",
        "pyarrow": pyarrow.__version__,
        "captured_input_sha256": {label: entry["sha256"] for label, entry in manifest["inputs"].items()},
        "packed_sha256": actual,
        "packed_rows": 288,
        "updates": 36,
        "row_occurrences": report["row_occurrences"],
        "new_model_or_teacher_calls": 0,
    }
    (args.output / "rebuild-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
