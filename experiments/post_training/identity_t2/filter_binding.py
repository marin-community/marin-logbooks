# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scoped T2 predicate over an unchanged v4r2 identity_decision result.

The caller must run shared v4r2 first. This adapter removes only its first-person
surface rejection for an owner-approved canonical name-first answer in family i.
All identity expectations, attribution, foreign claims and fact checks survive.
"""

from __future__ import annotations

import re

NAME_FIRST = re.compile(r"^(?:Marin Snowball|I am Marin Snowball|I'm Marin Snowball)\b")
SURFACE_REASON = "first_person_correct_answer_required"


def name_first(answer: str) -> bool:
    text = answer.replace("\u2019", "'").lstrip(" \t\r\n\"'\u201c\u201d\u2018\u2019*#>")
    match = NAME_FIRST.match(text)
    return bool(match) and ("Open Athena" not in text or text.index("Marin Snowball") < text.index("Open Athena"))


def apply_name_first_override(family: str, answer: str, base_decision: dict) -> dict:
    reasons = list(base_decision["rejection_reasons"])
    applied = False
    if family == "name_first":
        if name_first(answer):
            applied = SURFACE_REASON in reasons
            reasons = [reason for reason in reasons if reason != SURFACE_REASON]
        else:
            reasons.append("name_first_prefix_or_creator_order_failed")
    return {
        **base_decision,
        "rejection_reasons": reasons,
        "name_first_surface_override": applied,
        "base_rejection_reasons": list(base_decision["rejection_reasons"]),
    }
