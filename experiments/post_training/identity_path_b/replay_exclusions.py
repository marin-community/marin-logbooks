# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact sparse evaluation of established lexical exclusion rules; no semantic proof."""

import re
import unicodedata
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix


def normalized(text: str, rule: str) -> str:
    if rule == "benchmark_v1":
        text = unicodedata.normalize("NFKC", text)
    return " ".join(re.findall(r"\w+", text.casefold()))


def chargrams(text: str, rule: str) -> set[str]:
    count = max(1, len(text) - 4) if rule == "identity_v2" else len(text) - 4
    return {text[index : index + 5] for index in range(count)}


def wordgrams(text: str, width: int) -> set[str]:
    words = text.split()
    return {"\x1f".join(words[index : index + width]) for index in range(len(words) - width + 1)}


class FeatureIndex:
    """Exact integer intersections; normalization/thresholds remain the old rules."""

    def __init__(self, sets: list[set[str]]):
        self.vocabulary = {}
        columns, offsets = [], [0]
        for features in sets:
            for feature in sorted(features):
                if feature not in self.vocabulary:
                    self.vocabulary[feature] = len(self.vocabulary)
                columns.append(self.vocabulary[feature])
            offsets.append(len(columns))
        self.sizes = np.diff(offsets)
        self.matrix = (
            csr_matrix(
                (np.ones(len(columns), np.int32), np.asarray(columns, np.int32), np.asarray(offsets, np.int32)),
                shape=(len(sets), len(self.vocabulary)),
            )
            .transpose()
            .tocsr()
        )

    def intersections(self, features: set[str]) -> np.ndarray:
        columns = [self.vocabulary[feature] for feature in features if feature in self.vocabulary]
        query = csr_matrix(
            (np.ones(len(columns), np.int32), np.asarray(columns, np.int32), np.asarray([0, len(columns)], np.int32)),
            shape=(1, len(self.vocabulary)),
        )
        return (query @ self.matrix).toarray()[0]


class ExclusionIndex:
    def __init__(self, references: list[dict], rule: str):
        self.references, self.rule = references, rule
        self.texts = [normalized(row["text"], rule) for row in references]
        self.exact = defaultdict(list)
        for index, text in enumerate(self.texts):
            self.exact[text].append(index)
        self.chars = FeatureIndex([chargrams(text, rule) for text in self.texts])
        width = 5 if rule == "benchmark_v1" else 4
        self.grams = FeatureIndex([wordgrams(text, width) for text in self.texts])
        self.words = FeatureIndex([set(text.split()) for text in self.texts]) if rule == "identity_v2" else None

    def matches(self, text: str) -> list[dict]:
        norm = normalized(text, self.rule)
        chars = chargrams(norm, self.rule)
        common = self.chars.intersections(chars)
        union = len(chars) + self.chars.sizes - common
        char_jaccard = np.divide(common, union, out=np.zeros_like(common, float), where=union != 0)
        exact = np.zeros(len(self.references), dtype=bool)
        exact[self.exact.get(norm, [])] = True
        gram_features = wordgrams(norm, 5 if self.rule == "benchmark_v1" else 4)
        gram_common = self.grams.intersections(gram_features)
        gram_union = len(gram_features) + self.grams.sizes - gram_common
        gram_jaccard = np.divide(gram_common, gram_union, out=np.zeros_like(common, float), where=gram_union != 0)
        if self.rule == "benchmark_v1":
            smaller = np.minimum(len(chars), self.chars.sizes)
            containment = np.divide(common, smaller, out=np.zeros_like(common, float), where=smaller != 0)
            near = (
                (char_jaccard >= 0.5)
                | ((containment >= 0.8) & (common >= 30))
                | ((gram_jaccard >= 0.5) & (gram_common >= 3))
            )
        else:
            assert self.words is not None
            words = set(norm.split())
            word_common = self.words.intersections(words)
            word_union = len(words) + self.words.sizes - word_common
            word_jaccard = np.divide(word_common, word_union, out=np.zeros_like(common, float), where=word_union != 0)
            near = (char_jaccard >= 0.45) | (gram_common > 0) | (word_jaccard >= 0.6)
        return [
            {
                "corpus": self.references[index]["corpus"],
                "id": self.references[index]["id"],
                "rule": self.rule,
                "normalized_exact": bool(exact[index]),
                "near": bool(near[index]),
                "char5_jaccard": float(char_jaccard[index]),
                "shared_char5": int(common[index]),
            }
            for index in np.flatnonzero(exact | near)
        ]
