# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the one-serve CD16-T2 capture schedule; no sampling, cluster calls or training freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

TOOL_SOURCE_SHA256 = "95bffd0678807bb7ece9c6fbccf37d5f6434cc2e1ac254ff08bc7f78e4591403"
V2_SOURCE_SHA256 = "74f57fe26f08b77ef9b6a143b4d6f1463996c123c1ec190502a0f86a467c0520"
STYLE_D = "I am Marin Snowball, an AI model created by Open Athena."
STYLE_D_URL = (
    STYLE_D
    + " For more about the Marin project, see https://marin.community and https://github.com/marin-community/marin."
)
TEACHERS = {"plain": None, "style_d": STYLE_D, "style_d_url": STYLE_D_URL}
EXCEPTION_LABS = {
    "v2-third-person-037": "OpenAI",
    "v2-third-person-038": "OpenAI",
    "v2-third-person-039": "Microsoft",
}
IDENTITY_SEED_OFFSET = 312000000
CONTROL_SEED_OFFSET = 313000000


def read_bound(path: Path, digest: str) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Input hash mismatch: {path.name}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def build(tool_path: Path, v2_path: Path) -> tuple[list[dict], dict]:
    source_tools = read_bound(tool_path, TOOL_SOURCE_SHA256)
    source_v2 = read_bound(v2_path, V2_SOURCE_SHA256)
    assert len(source_tools) == 192 and len(source_v2) == 167
    assert Counter(row["family"] for row in source_v2) == {
        "name_first": 40,
        "negation": 57,
        "history": 30,
        "third_person": 40,
    }
    tools = []
    for original in source_tools:
        # The fixed tool IDs, prompts, tools, modes and seeds are not re-ranked or reseeded.
        row = dict(original)
        row.update(
            group="tools",
            family=original["shape"],
            source_id=original["id"],
            sample_index=0,
            teacher="plain",
            teacher_system=None,
            row_weight=1,
            allowed_non_affiliation_lab=None,
        )
        tools.append(row)
    identity, controls = [], []
    for source in source_v2:
        control = source["family"] == "third_person"
        expected_teacher = (
            "style_d_url"
            if source["id"] in EXCEPTION_LABS or source["family"] == "history"
            else "plain" if control else "style_d"
        )
        if source["teacher"] != expected_teacher:
            raise ValueError(f"Teacher binding mismatch: {source['id']}")
        group = controls if control else identity
        offset = CONTROL_SEED_OFFSET if control else IDENTITY_SEED_OFFSET
        for thinking in (False, True):
            for sample in range(1 if control else 8):
                group.append(
                    {
                        "id": f"{source['id']}-{'on' if thinking else 'off'}-s{sample:02d}",
                        "source_id": source["id"],
                        "group": "third_person" if control else "identity_v2",
                        "family": source["family"],
                        "rival": source["rival"],
                        "prompt": [{"role": "user", "content": source["prompt"]}],
                        "tools": [],
                        "enable_thinking": thinking,
                        "sample_index": sample,
                        "sampling_seed": offset + len(group),
                        "teacher": expected_teacher,
                        "teacher_system": TEACHERS[expected_teacher],
                        "row_weight": 1 if control else 1.2,
                        "allowed_non_affiliation_lab": EXCEPTION_LABS.get(source["id"]),
                    }
                )
    assert len(identity) == 2032 and len(controls) == 80
    # One five-request pilot: four original tool requests plus a style-D name-first OFF request.
    # It covers all three tool shapes and both modes without claiming six live cells.
    pilot = [*tools[:4], identity[0]]
    assert {row["family"] for row in pilot[:4]} == {"first_call", "call_after_result", "reply"}
    assert {row["enable_thinking"] for row in pilot[:4]} == {False, True}
    assert pilot[-1]["teacher"] == "style_d" and pilot[-1]["family"] == "name_first"
    rows = [*pilot, *tools[4:], *identity[1:], *controls]
    assert len(rows) == len({row["id"] for row in rows}) == 2304
    report = {
        "tool_source_sha256": TOOL_SOURCE_SHA256,
        "v2_source_sha256": V2_SOURCE_SHA256,
        "requests": len(rows),
        "groups": dict(Counter(row["group"] for row in rows)),
        "teacher_requests": dict(Counter(row["teacher"] for row in rows)),
        "identity_seed_offset": IDENTITY_SEED_OFFSET,
        "control_seed_offset": CONTROL_SEED_OFFSET,
        "pilot_ids": [row["id"] for row in pilot],
        "identity_keep_cap_per_exact_prompt_and_mode": 2,
        "mandatory_third_person_QC_source_ids": sorted(EXCEPTION_LABS),
        "no_automatic_replacement_or_topup": True,
        "training_freeze": False,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, report = build(args.tools, args.v2)
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.out.open("xb") as file:
        file.write(raw)
    with args.report.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
