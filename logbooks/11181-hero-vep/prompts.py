# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the issue's exact prompts and select teacher-forced target scores."""

from pathlib import Path

HEALTHY_PREFIX = "The following sequence is from a healthy individual:\n"
TEMPLATES = Path(__file__).parent / "templates"


def fields(row: dict, strand: str) -> dict[str, str]:
    sequences = row["sequences"][strand]
    return {
        "reference_sequence": sequences["reference"],
        "ref_base": sequences["reference"][127],
        "alt_base": sequences["alternate"][127],
    }


def classification_prompt(query: dict, demonstrations: list[dict], strand: str, mapping: str) -> str:
    assert mapping in {"primary", "reversed"}
    assert len(demonstrations) in {0, 2}
    template = (TEMPLATES / "classification.txt").read_text().removesuffix("\n")
    block = (TEMPLATES / "variant.txt").read_text()
    pathogenic, benign = ("A", "B") if mapping == "primary" else ("B", "A")
    examples = ""
    for index, row in enumerate(demonstrations, 1):
        examples += block.format(title=f"Example {index}", **fields(row, strand))
        examples += f"Answer: {pathogenic if row['label'] else benign}\n\n"
    options = "A: pathogenic\nB: benign" if mapping == "primary" else "A: benign\nB: pathogenic"
    return template.format(
        options=options,
        examples=examples,
        query=block.format(title="Query", **fields(query, strand)).removesuffix("\n"),
    )


def continuation_start(prefix_ids: list[int], full_ids: list[int], single_token: bool) -> int:
    """Reject boundary merges instead of silently scoring conditioning text."""
    assert full_ids[: len(prefix_ids)] == prefix_ids, "Tokenizer merges across the target boundary"
    assert len(full_ids) > len(prefix_ids)
    if single_token:
        assert len(full_ids) == len(prefix_ids) + 1, "Candidate must be exactly one token"
    return len(prefix_ids)
