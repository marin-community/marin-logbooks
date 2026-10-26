# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Owner-directed CPU defect census and rejection disposition; never a freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from experiments.post_training.identity_t2 import scorer_snapshot as shared
from experiments.post_training.identity_t2 import v4r2_snapshot as legacy
from experiments.post_training.identity_t2.qc import CALL, write_jsonl

CANDIDATE_SHA = "c9dac8ddd5598d7dbb10c9bd2d15dcf6d88a74959df91e33a96bdc19a77b551e"
REVIEW_SHA = "4c10d62f717f2da829f671e75f1c528bd91da8b1396974989dbe38a8d7b83320"
PACKET_SHA = "56529ed02bf2429aa0713101de97ebdb29801165322fd242a021413ae8f88390"
ROLE_RESULT = re.compile(r"<(?:tool_response|tool_result)\b[^>]*>(.*?)</(?:tool_response|tool_result)>", re.S | re.I)
TOOL_HEADER = re.compile(r"<\|start_header_id\|>\s*tool\s*<\|end_header_id\|>", re.I)
NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%)?")
RESULT_ASSERTION = re.compile(
    r"[^.!?\n]*(?:\b(?:result|interest earned|monthly(?: loan)? payment|balance|total|temperature)\b"
    r"[^.!?\n]{0,100}(?:\bis\b|\bare\b|would be|will be|[:=])[^.!?\n]{0,80}\d|"
    r"\byou\s+(?:will|would)\s+(?:earn|receive|pay)[^.!?\n]{0,60}\d)[^!?\n]*",
    re.I,
)
RESULT_KEYS = {
    "news",
    "articles",
    "result",
    "results",
    "output",
    "temperature",
    "monthly_payment",
    "interest",
    "interest_earned",
    "balance",
    "status",
    "success",
    "weather",
    "total",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(value: str) -> str:
    try:
        return json.dumps(json.loads(value), sort_keys=True, ensure_ascii=False)
    except ValueError:
        return " ".join(value.split())


def foreign_evidence(text: str) -> list[dict]:
    evidence = []
    normalized_text = text.replace("\u2019", "'")
    for match in legacy.FOREIGN_SELF_IDENTITY.finditer(normalized_text):
        if not shared.NEGATION.search(normalized_text[match.start() : match.start(1)]):
            evidence.append(
                {
                    "class": "foreign_self_identity",
                    "entity": match.group(1),
                    "excerpt": normalized_text[max(0, match.start() - 60) : match.end() + 60],
                }
            )
    assert bool(evidence) == legacy.foreign_self_identity(text)
    return evidence


def json_objects(text: str):
    decoder = json.JSONDecoder()
    cursor = 0
    while (start := text.find("{", cursor)) >= 0:
        try:
            value, end = decoder.raw_decode(text[start:])
        except ValueError:
            cursor = start + 1
            continue
        cursor = start + end
        yield value, text[start:cursor]


def invented_result_evidence(row: dict) -> list[dict]:
    tool_messages = [str(turn.get("content", "")) for turn in row["candidate"]["prompt"] if turn["role"] == "tool"]
    matching = {normalized(content) for content in tool_messages}
    tool_numbers = {number.replace(",", "") for content in tool_messages for number in NUMBER.findall(content)}
    evidence = []
    for span_name in ("reasoning", "answer"):
        text = row["decision"][span_name]
        for match in ROLE_RESULT.finditer(text):
            if normalized(match.group(1)) not in matching:
                evidence.append({"class": "invented_tool_role_payload", "span": span_name, "excerpt": match.group()})
        if TOOL_HEADER.search(text) and not ROLE_RESULT.search(text):
            # A role switch generated inside the assistant target is not an
            # actual conversation tool message and cannot supply new evidence.
            evidence.append(
                {
                    "class": "assistant_generated_tool_role_header",
                    "span": span_name,
                    "excerpt": text[TOOL_HEADER.search(text).start() :],
                }
            )
        without_calls = CALL.sub("", text)
        for value, payload in json_objects(without_calls):
            if not isinstance(value, dict) or {"name", "arguments"} <= value.keys():
                continue
            if not (RESULT_KEYS & value.keys()) and value.get("role") != "tool":
                continue
            if normalized(payload) not in matching:
                evidence.append({"class": "unmatched_result_JSON_payload", "span": span_name, "excerpt": payload})
        for match in RESULT_ASSERTION.finditer(without_calls):
            claimed = {number.replace(",", "") for number in NUMBER.findall(match.group())}
            if not tool_messages or not claimed <= tool_numbers:
                evidence.append(
                    {
                        "class": "result_assertion_without_matching_tool_message",
                        "span": span_name,
                        "excerpt": match.group(),
                        "actual_tool_message_count": len(tool_messages),
                    }
                )
    return evidence


def census_row(row: dict) -> dict:
    hits = []
    for span_name in ("reasoning", "answer"):
        hits.extend({**hit, "span": span_name} for hit in foreign_evidence(row["decision"][span_name]))
    if row["candidate"]["group"] == "tools":
        hits.extend(invented_result_evidence(row))
    return {
        "id": row["id"],
        "group": row["candidate"]["group"],
        "raw_sha256": row["raw_record_sha256"],
        "hits": hits,
        "exclude": bool(hits),
    }


def run(candidates: Path, review: Path, packet: Path, out: Path) -> dict:
    assert digest(candidates.read_bytes()) == CANDIDATE_SHA
    assert digest(review.read_bytes()) == REVIEW_SHA
    assert digest(packet.read_bytes()) == PACKET_SHA
    rows = [json.loads(line) for line in candidates.read_bytes().splitlines()]
    human = json.loads(review.read_bytes())
    pack = {row["id"]: row for row in (json.loads(line) for line in packet.read_bytes().splitlines())}
    assert len(pack) == 100 and len(human["rows"]) == 100
    human_rejects = {}
    for verdict in human["rows"]:
        source = pack[verdict["id"]]
        assert source["raw_record_sha256"] == verdict["raw_sha256"]
        if verdict["verdict"] == "reject":
            assert verdict["QC_role"] == "eligible_candidate"
            human_rejects[verdict["id"]] = verdict["reason"]
    assert len(human_rejects) == 12
    targeted = [row for row in rows if row["candidate"]["group"] in {"tools", "third_person"}]
    assert Counter(row["candidate"]["group"] for row in targeted) == {"tools": 164, "third_person": 60}
    checks = []
    for row in targeted:
        assert digest(Path(row["raw_record"]).read_bytes()) == row["raw_record_sha256"]
        checks.append(census_row(row))
    census_rejects = {row["id"]: row["hits"] for row in checks if row["exclude"]}
    excluded = set(human_rejects) | set(census_rejects)
    assert excluded <= {row["id"] for row in rows}
    kept = [row for row in rows if row["id"] not in excluded]
    report = {
        "source_candidates": 337,
        "original_tool_census": 164,
        "original_control_census": 60,
        "human_rejects": human_rejects,
        "census_rejects": census_rejects,
        "human_census_overlap": sorted(set(human_rejects) & set(census_rejects)),
        "union_excluded": len(excluded),
        "proposed_remaining": len(kept),
        "remaining_by_family_mode": dict(
            Counter(
                row["candidate"]["group"]
                + "/"
                + row["candidate"]["family"]
                + "/"
                + ("on" if row["candidate"]["enable_thinking"] else "off")
                for row in kept
            )
        ),
        "remaining_by_group": dict(Counter(row["candidate"]["group"] for row in kept)),
        "no_target_repairs_replacements_regeneration": True,
        "automatic_census_is_not_full_semantic_proof": True,
        "unreviewed_rows_not_individually_content_approved": True,
        "freeze_approved": False,
        "training_approved": False,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "census-all-224.jsonl", checks)
    write_jsonl(out / "proposed-post-disposition.jsonl", kept)
    write_jsonl(
        out / "excluded-union.jsonl",
        [
            {
                "id": identifier,
                "human_reason": human_rejects.get(identifier),
                "census_hits": census_rejects.get(identifier, []),
            }
            for identifier in sorted(excluded)
        ],
    )
    report["artifact_sha256"] = {path.name: digest(path.read_bytes()) for path in out.iterdir() if path.is_file()}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "review", "packet", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates, args.review, args.packet, args.out), indent=2))


if __name__ == "__main__":
    main()
