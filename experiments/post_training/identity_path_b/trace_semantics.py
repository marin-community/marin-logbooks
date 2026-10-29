# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conservative all-source screens and a deterministic stratified manual sample.

Flags are review evidence, not semantic entailment or a population error rate.
No generated text is changed, and no OFF rationale is generated.
"""

import hashlib
import re
from collections import Counter

from experiments.post_training.identity_path_b.reasoning import identity_qc

OPENING = "I am Marin's Latest MoE."
LABS = ("OpenAI", "Anthropic", "Google", "Meta", "Alibaba", "DeepSeek", "Mistral AI", "Zhipu AI")
REFERENCE = re.compile(
    r"\b(?:canonical|reference\s+(?:answer|response|text|profile)|system\s+prompt|teacher|"
    r"(?:provided|supplied|given)\s+(?:facts?|context|profile|reference|answer|instructions?)|"
    r"(?:stated|mentioned|given)\s+above|I\s+(?:was|have been)\s+told)\b",
    re.I,
)
UNKNOWN = re.compile(
    r"\b(?:unknown|unconfirmed|unspecified|uncertain|unavailable|not\s+(?:known|specified|confirmed|documented)|"
    r"cannot\s+(?:confirm|verify|determine|say|identify)|can't\s+(?:confirm|verify|determine|say|identify)|"
    r"(?:do\s+not|don't)\s+know|lack\s+(?:reliable|confirmed))\b",
    re.I,
)
ARCHITECTURE = re.compile(
    r"\b(?:mixture[ -]of[ -]experts|transformers?|decoder|parameters?|layers?|context[ -]window|"
    r"attention\s+(?:heads?|mechanism)|sparse\s+(?:routing|activation)|MoE\s+stands\s+for)\b",
    re.I,
)
LICENSE = re.compile(r"\b(?:Apache|MIT\s+license|GPL|CC-BY|licensed|open[ -]source)\b", re.I)
DEPLOYMENT = re.compile(r"\b(?:I\s+am|I'm|I\s+was|we\s+are)\s+(?:hosted|served|deployed|provided)\s+by\b", re.I)
LINEAGE = re.compile(
    r"\b(?:no\s+(?:connection|affiliation|lineage|relationship)|unrelated\s+to|"
    r"(?:not|never)\s+(?:based|derived|descended)\s+(?:on|from)|no\s+.+?\s+(?:lineage|technology|involvement))\b",
    re.I,
)
UNSUPPORTED_EXCLUSION = re.compile(
    r"\b(?:NVIDIA|AI2)\b[^.!?]{0,100}\b(?:not|rather\s+than|instead\s+of)\b[^.!?]{0,60}\bmaintain|"
    r"\b(?:only|exclusively)\b[^.!?]{0,50}\bmaintain|"
    r"\bmaintain[^.!?]{0,50}\b(?:only|exclusively)\b",
    re.I,
)
TRAINING_CONTRIBUTOR = re.compile(
    r"\b(?:trained|developed)\b[^.!?]{0,90}\bwith\s+contributors?\s+from\s+(?:Open Athena|Stanford)\b",
    re.I,
)
FOREIGN_MENTION = re.compile(
    r"\b(?:OpenAI|ChatGPT|Anthropic|Claude|Google|Gemini|Meta|Llama|Alibaba|Qwen|Mistral|Zhipu)\b", re.I
)
FOREIGN_PREMISE_V2 = re.compile(
    r"\b(?:OpenAI|ChatGPT|GPT|Anthropic|Claude|Google|Gemini|DeepMind|Meta|Llama|Alibaba|Qwen|"
    r"DeepSeek|Kimi|GLM|Mistral|Zhipu)\b",
    re.I,
)
DENIAL_V2 = re.compile(r"\b(?:not|never|no|rather\s+than|instead\s+of)\b", re.I)
ORGANIZATION_V2 = re.compile(
    r"\b(?:open\s+(?:collaboration|effort|community)|collaborative\s+(?:open\s+)?(?:effort|community)|"
    r"community\s+effort|community[ -]driven|collective\s+(?:effort|of)|"
    r"rather\s+than\s+(?:a\s+)?(?:single|secret|corporate))\b",
    re.I,
)
CONTRIBUTION_V2 = re.compile(r"\b(?:contribut\w*|collaborat\w*)\b", re.I)
DEVELOPMENT_V2 = re.compile(r"\b(?:trained|training|developed|development|created|built)\b", re.I)
MAINTENANCE_V2 = re.compile(r"\bmaintain\w*\b", re.I)
MAINTAINER_INSTITUTION_V2 = re.compile(r"\b(?:Open Athena|Stanford)\b", re.I)
LINEAGE_V2 = re.compile(
    r"\b(?:not|never|no)\b[^.!?;]{0,50}\b(?:fine[ -]tun\w*|base[ -]model|lineage|relationship|"
    r"affiliat\w*|descend\w*|deriv\w*)\b",
    re.I,
)
ROLE_EXCLUSION_V2 = re.compile(
    r"\bI\s+(?:do\s+not|don't|cannot|can't)\s+(?:determine|control|decide|set|manage)\b",
    re.I,
)
UI_CAPABILITY_V2 = re.compile(
    r"\b(?:profile\s+settings?|account\s+settings?|you\s+can\s+(?:update|edit|change)|"
    r"I\s+(?:can|cannot|can't)\s+(?:change|update|edit)\s+(?:your|the))\b",
    re.I,
)
TRAINING_DATA_UNKNOWN_V2 = re.compile(
    r"\b(?:no\s+(?:reliable\s+)?(?:details|information)|(?:do\s+not|don't)\s+know|"
    r"(?:cannot|can't)\s+(?:confirm|verify))\b[^.!?;]{0,100}\btraining\s+data\b",
    re.I,
)
URL_V2 = re.compile(r"https?://[^\s<>]+", re.I)


def sentence_count(text: str) -> int:
    """Count output-contract sentences without splitting approved URL domains."""

    def protect_url(match: re.Match) -> str:
        url = match.group()
        core = url.rstrip(".!?")
        return "URL" + url[len(core) :]

    protected = URL_V2.sub(protect_url, text).strip()
    parts = re.split(r"[.!?]+(?:\s+|$)", protected)
    return sum(bool(part.strip()) for part in parts)


def semantic_flags(text: str | None, canonical_final: str, family: str) -> list[str]:
    """Flag concrete unsupported claims, role conflicts and output-contract faults."""
    if text is None:
        return ["missing_visible_rationale"]
    flags = []
    if not text.startswith(OPENING):
        flags.append("exact_opening_contract")
    if REFERENCE.search(text):
        flags.append("omitted_context_or_reference")
    for reason in identity_qc(text, "", canonical_final):
        if "foreign_self_claim" in reason:
            flags.append("foreign_self_identity_prescreen")
        elif "noncanonical_self_name" in reason or "noncanonical_creator_trainer" in reason:
            flags.append("reasoning_final_identity_or_role_mismatch")
        elif "omitted_context" in reason:
            flags.append("omitted_context_or_reference")
    if re.search(r"\b\d", text):
        flags.append("unsupported_numeric_or_version_detail")
    if UNSUPPORTED_EXCLUSION.search(text):
        flags.append("unsupported_exclusive_maintainer_role")
    if TRAINING_CONTRIBUTOR.search(text):
        flags.append("unsupported_or_ambiguous_training_contributor_role")
    if LINEAGE.search(text):
        flags.append("unsupported_broad_lineage_or_affiliation_exclusion")
    # A caveat about unknown details is allowed; an affirmative architecture,
    # license or serving claim is not supported by the approved profile.
    for clause in re.split(r"[.!?;]\s+|\b(?:but|however|whereas|yet)\b", text, flags=re.I):
        if UNKNOWN.search(clause):
            continue
        if ARCHITECTURE.search(clause):
            flags.append("unsupported_architecture_fact")
        if LICENSE.search(clause):
            flags.append("unsupported_license_or_open_source_fact")
        if DEPLOYMENT.search(clause):
            flags.append("unsupported_deployment_provider")
    if family == "false_premise" and FOREIGN_MENTION.search(text):
        flags.append("contract_foreign_premise_restatement")
    return sorted(set(flags))


def revised_semantic_flags(text: str | None, canonical_final: str, family: str) -> list[str]:
    """Monotone r2 screens after the frozen 198-original manual read.

    Accurate foreign denials are contract flags, not affirmative self-claims.
    Role/organizational predicates are conservative lexical evidence, not an
    entailment checker. Initial rejects remain rejects, including false positives.
    """
    flags = semantic_flags(text, canonical_final, family)
    if text is None:
        return flags
    if len(text.split()) > 40:
        flags.append("contract_more_than_40_words")
    if sentence_count(text) != 2:
        flags.append("contract_not_two_sentences")
    for clause in re.split(r"[.!?;]\s+|\b(?:but|however|whereas|yet)\b", text, flags=re.I):
        if FOREIGN_PREMISE_V2.search(clause) and DENIAL_V2.search(clause):
            flags.append("contract_foreign_premise_restatement")
        if ORGANIZATION_V2.search(clause):
            flags.append("unsupported_organizational_or_role_broadening")
        if (
            DEVELOPMENT_V2.search(clause)
            and CONTRIBUTION_V2.search(clause)
            and MAINTAINER_INSTITUTION_V2.search(clause)
            and not MAINTENANCE_V2.search(clause)
        ):
            flags.append("unsupported_or_ambiguous_training_contributor_role")
        if LINEAGE_V2.search(clause):
            flags.append("unsupported_broad_lineage_or_affiliation_exclusion")
        if ROLE_EXCLUSION_V2.search(clause):
            flags.append("unsupported_operational_role_exclusion")
        if UI_CAPABILITY_V2.search(clause):
            flags.append("unsupported_deployment_or_ui_capability")
        if TRAINING_DATA_UNKNOWN_V2.search(clause):
            flags.append("unsupported_broad_training_data_uncertainty")
    return sorted(set(flags))


def denial_lab(final: str, family: str) -> str | None:
    if family != "false_premise":
        return None
    result = [lab for lab in LABS if re.search(r"not\s+trained\s+by\s+" + re.escape(lab) + r"\b", final, re.I)]
    if len(result) != 1:
        raise ValueError("Denial source must bind one of the original eight labs")
    return result[0]


def ordered(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"310:semantic-sample:{row['source_id']}".encode()).hexdigest())


def balanced(rows: list[dict], count: int) -> list[dict]:
    chosen = []
    for flagged in (False, True):
        chosen.extend(ordered([r for r in rows if bool(r["screen_flags"]) == flagged])[: count // 2])
    selected = {r["source_id"] for r in chosen}
    chosen.extend(ordered([r for r in rows if r["source_id"] not in selected])[: count - len(chosen)])
    return chosen


def manual_sample(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Freeze198 originals:18 per main family and all18 project originals."""
    sample = []
    families = sorted({row["family"] for row in rows})
    if len(families) != 11 or len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Require the complete unique eleven-family source census")
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        if family == "project_fact":
            if len(family_rows) != 18:
                raise ValueError("Project sample is the complete18-original census")
            sample.extend(ordered(family_rows))
        elif family == "false_premise":
            chosen = [r for lab in LABS for r in balanced([r for r in family_rows if r["denial_lab"] == lab], 2)]
            selected = {r["source_id"] for r in chosen}
            chosen.extend(ordered([r for r in family_rows if r["source_id"] not in selected])[:2])
            sample.extend(chosen)
        else:
            sample.extend(balanced(family_rows, 18))
    if len(sample) != 198 or len({r["source_id"] for r in sample}) != 198:
        raise ValueError("Manual sample requires198 distinct original conversations")

    def cell(row):
        return (row["family"], row["denial_lab"], bool(row["screen_flags"]))

    population, selected = Counter(cell(row) for row in rows), Counter(cell(row) for row in sample)
    fractions = [
        dict(
            family=key[0],
            denial_lab=key[1],
            flagged=key[2],
            population=n,
            selected=selected[key],
            realized_sample_fraction=selected[key] / n,
        )
        for key, n in sorted(population.items(), key=lambda item: str(item[0]))
    ]
    return ordered(sample), fractions
