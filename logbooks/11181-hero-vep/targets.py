# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate the fixed pilot's likelihood and classification targets."""

from collections.abc import Iterator
from dataclasses import dataclass

from prompts import HEALTHY_PREFIX, classification_prompt


@dataclass(frozen=True)
class Target:
    variant: str
    method: str
    strand: str
    mapping: str
    candidate: str
    prefix: str
    text: str
    single_token: bool


def classification_targets(row: dict, demonstrations: list[dict], mappings: tuple[str, ...]) -> Iterator[Target]:
    for strand in ("+", "-"):
        for method, examples in (("zero-shot", []), ("two-shot", demonstrations)):
            for mapping in mappings:
                prompt = classification_prompt(row, examples, strand, mapping)
                for candidate in ("A", "B"):
                    yield Target(row["id"], method, strand, mapping, candidate, prompt, candidate, True)


def targets(rows: list[dict], demonstrations: list[dict]) -> Iterator[Target]:
    for row in rows:
        for strand in ("+", "-"):
            sequences = row["sequences"][strand]
            for method, prefix in (("null", ""), ("healthy", HEALTHY_PREFIX)):
                for allele in ("reference", "alternate"):
                    yield Target(row["id"], method, strand, "none", allele, prefix, sequences[allele], False)
        yield from classification_targets(row, demonstrations, (row["answer_mapping"],))


def mapping_diagnostic_targets(rows: list[dict], demonstrations: list[dict]) -> Iterator[Target]:
    """Generate both mappings separately from the single-mapping benchmark."""
    for row in rows:
        yield from classification_targets(row, demonstrations, ("primary", "reversed"))
