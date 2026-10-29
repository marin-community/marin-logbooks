# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
import unicodedata

import pytest

from experiments.post_training.identity_path_b.data import features, overlap
from experiments.post_training.identity_path_b.replay_exclusions import ExclusionIndex


def benchmark_match(left: str, right: str) -> bool:
    def values(text):
        norm = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))
        words = norm.split()
        return (
            norm,
            {norm[index : index + 5] for index in range(len(norm) - 4)},
            {tuple(words[index : index + 5]) for index in range(len(words) - 4)},
        )

    left, lc, lw = values(left)
    right, rc, rw = values(right)
    common, union = len(lc & rc), len(lc | rc)
    wc, wu = len(lw & rw), len(lw | rw)
    return (
        left == right
        or (union > 0 and common / union >= 0.5)
        or (min(len(lc), len(rc)) > 0 and common >= 30 and common / min(len(lc), len(rc)) >= 0.8)
        or (wu > 0 and wc >= 3 and wc / wu >= 0.5)
    )


@pytest.mark.parametrize("rule", ["benchmark_v1", "identity_v2"])
def test_sparse_integer_intersections_match_established_pairwise_exclusion_rules(rule):
    documents = [
        "",
        "abc",
        "\uff21\uff22\uff23",
        "Alpha beta gamma delta epsilon zeta eta theta iota.",
        "Solve for a positive integer with exactly three different prime factors.",
        "Write a Python function that groups the entries by their first character.",
        "Which community developed and trained the model?",
    ]
    queries = [
        *documents,
        "  ALPHA beta gamma delta epsilon zeta eta theta iota! ",
        documents[4] + " Please give a detailed explanation with all intermediate steps.",
        "In a paragraph about a different topic, alpha beta gamma delta appears only as a quote.",
        "some completely unrelated lexical material",
    ]
    references = [{"id": str(index), "corpus": "fixture", "text": text} for index, text in enumerate(documents)]
    index = ExclusionIndex(references, rule)
    for query in queries:
        expected = {
            str(i)
            for i, text in enumerate(documents)
            if (
                benchmark_match(query, text)
                if rule == "benchmark_v1"
                else bool(overlap(features(query), features(text)))
            )
        }
        assert {match["id"] for match in index.matches(query)} == expected
