# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the issue's exact prompts and select teacher-forced target scores."""

import math
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
    template = (TEMPLATES / ("two-shot.txt" if demonstrations else "zero-shot.txt")).read_text().removesuffix("\n")
    if mapping == "reversed":
        template = template.replace("A: pathogenic\nB: benign", "A: benign\nB: pathogenic")
    values = fields(query, strand)
    for index, row in enumerate(demonstrations, 1):
        values.update({f"example_{index}_{key}": value for key, value in fields(row, strand).items()})
        pathogenic_letter = "A" if mapping == "primary" else "B"
        benign_letter = "B" if mapping == "primary" else "A"
        values[f"example_{index}_letter"] = pathogenic_letter if row["label"] else benign_letter
    return template.format(**values)


def continuation_start(prefix_ids: list[int], full_ids: list[int], single_token: bool) -> int:
    """Reject boundary merges instead of silently scoring conditioning text."""
    assert full_ids[: len(prefix_ids)] == prefix_ids, "Tokenizer merges across the target boundary"
    assert len(full_ids) > len(prefix_ids)
    if single_token:
        assert len(full_ids) == len(prefix_ids) + 1, "Candidate must be exactly one token"
    return len(prefix_ids)


def target_logprob(token_ids: list[int], prompt_logprobs: list[dict | None], start: int) -> float:
    """Sum observed-token probabilities from the vLLM prompt-logprob format."""
    assert len(token_ids) == len(prompt_logprobs)
    assert 0 < start < len(token_ids), "A scored token needs a preceding context token (including BOS)"
    scores = []
    for token, probabilities in zip(token_ids[start:], prompt_logprobs[start:], strict=True):
        assert probabilities is not None
        observed = probabilities[str(token)]
        score = float(observed["logprob"])
        assert math.isfinite(score)
        scores.append(score)
    return math.fsum(scores)
