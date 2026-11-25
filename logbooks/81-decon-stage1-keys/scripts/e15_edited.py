# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import random

from experiments.datakit.decontam.encoder.planted import edited_forms

item = {
    "question": "Which planet is known as the Red Planet in our solar system?",
    "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
    "passage": "",
}
for form, text in edited_forms(item, random.Random(3)).items():
    print(f"{form:22s} {text!r}")
