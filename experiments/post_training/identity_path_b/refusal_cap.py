# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed seed310 source cap and outcome-independent nonrefusal dose refill."""

import hashlib
from collections import Counter, defaultdict

CAPPED_FAMILIES = frozenset({"unknown_fact", "deployment_provider"})
CAP_PER_FAMILY = 256


def source_rank(source_id: str) -> tuple[str, str]:
    return hashlib.sha256(f"310:{source_id}".encode()).hexdigest(), source_id


def capped_source_selection(off_rows: list[dict]) -> tuple[set[str], dict]:
    families = defaultdict(list)
    variants = defaultdict(Counter)
    for row in off_rows:
        final = row["messages"][-1]["content"]
        if "should not guess" not in final.casefold():
            continue
        family = row["family"]
        if family not in CAPPED_FAMILIES:
            raise ValueError("Unreviewed refusal family; do not silently omit it from the cap")
        families[family].append(row["source_id"])
        variants[family][final] += 1
    if set(families) != CAPPED_FAMILIES or any(len(ids) != 4096 for ids in families.values()):
        raise ValueError("Pinned two-family refusal census differs")
    selected = {f: sorted(ids, key=source_rank)[:CAP_PER_FAMILY] for f, ids in families.items()}
    dropped = {source for ids in families.values() for source in ids} - {s for ids in selected.values() for s in ids}
    return dropped, {
        "seed": 310,
        "source_rank": "SHA256(UTF-8('310:' + source_id)), then source_id",
        "scope": "final assistant answer only; masked earlier turns do not define source exclusion",
        "family_before": {f: len(ids) for f, ids in families.items()},
        "family_after": {f: len(ids) for f, ids in selected.items()},
        "exact_answer_variants_by_family": dict(variants),
        "retained_refusal_source_ids": selected,
        "dropped_source_ids": sorted(dropped),
    }


def capped_refilled_schedule(
    original: list[dict], rows: dict, dropped: set[str], minimum: int
) -> tuple[list[dict], dict]:
    """Fill dropped occurrence slots in order; preserve all retained relative order."""
    retained = [r for r in original if r["source"] != "identity" or r["source_id"] not in dropped]
    counts = Counter(r["source_id"] for r in retained if r["source"] == "identity")
    modes = Counter((r["source_id"], r["enable_thinking"]) for r in retained if r["source"] == "identity")
    pool = defaultdict(dict)
    for row in rows.values():
        if row.source_id not in dropped and row.family not in CAPPED_FAMILIES:
            pool[row.source_id][row.enable_thinking] = row
    if any(set(pair) != {False, True} for pair in pool.values()):
        raise ValueError("Nonrefusal refill sources need their exact OFF/ON rows")
    final = sum(r["canonical_final_text_tokens"] for r in retained if r["source"] == "identity")
    after_drop = final
    extras = []
    ordered = sorted(pool, key=source_rank)
    while final < minimum:
        before = final
        for source in ordered:
            if counts[source] >= 3:
                continue
            off, on = modes[(source, False)], modes[(source, True)]
            tie = int(hashlib.sha256(f"310:mode:{source}".encode()).hexdigest(), 16) & 1
            mode = bool(tie) if off == on else off > on
            row = pool[source][mode]
            extras.append(
                {
                    "source": "identity",
                    "source_id": source,
                    "row_id": row.row_id,
                    "enable_thinking": mode,
                    "family": row.family,
                    "canonical_final_text_tokens": row.dose.final_text,
                }
            )
            counts[source] += 1
            modes[(source, mode)] += 1
            final += row.dose.final_text
            if final >= minimum:
                break
        if before == final:
            raise ValueError("Retained nonrefusal pool cannot meet dose under source-total cap3")
    result, origins = [], []
    extra_index = 0
    for index, row in enumerate(original):
        if row["source"] == "identity" and row["source_id"] in dropped:
            if extra_index >= len(extras):
                continue
            chosen = extras[extra_index]
            origin = {"extra_rank": extra_index, "replaces_parent_occurrence": index}
            extra_index += 1
        else:
            chosen, origin = row, {"parent_occurrence": index}
        result.append(dict(chosen, occurrence=len(result)))
        origins.append(origin)
    for rank in range(extra_index, len(extras)):
        result.append(dict(extras[rank], occurrence=len(result)))
        origins.append({"extra_rank": rank, "appended_after_parent": True})
    if max(counts.values()) > 3:
        raise ValueError("Refill exceeds original-source cap across modes")
    return result, {
        "minimum": minimum,
        "drop_only_final_tokens": after_drop,
        "realized_final_tokens": final,
        "added_occurrences": len(extras),
        "dropped_occurrences": len(original) - len(retained),
        "maximum_source_occurrences": max(counts.values()),
        "extra_mode_rule": "less-used mode; ties SHA256(UTF-8('310:mode:' + source_id)) integer low bit",
        "insertion_rule": (
            "fill removed parent slots chronologically in extra-source order; "
            "drop remaining slots; append overflow extras"
        ),
        "extra_occurrences": extras,
        "origin_map": origins,
    }
