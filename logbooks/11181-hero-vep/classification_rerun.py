# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classification-only pilot, with verified reuse of historical DNA scores."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from aggregate import expected_key, target_key
from paced import PacedClient
from targets import targets


def load_reused_dna(panels_path: Path, old_panels_path: Path, old_scores_path: Path) -> tuple[dict, list[dict]]:
    panels = json.loads(panels_path.read_bytes())
    old_panels_bytes = old_panels_path.read_bytes()
    old_panels = json.loads(old_panels_bytes)
    assert panels["protocol_version"] == 3
    for panel in ("pilot", "validation", "demonstrations"):
        current = [{k: v for k, v in r.items() if k != "answer_mapping"} for r in panels["panels"][panel]]
        previous = [{k: v for k, v in r.items() if k != "answer_mapping"} for r in old_panels["panels"][panel]]
        assert current == previous, "Variant metadata or sequences changed"
    rows = panels["panels"]["pilot"]
    assert len(rows) == 100 and len({r["id"] for r in rows}) == 100
    assert all(r["chrom"] in {str(i) for i in range(2, 23)} for r in rows)
    groups = Counter(r["match_group"] for r in rows)
    assert len(groups) == 10 and set(groups.values()) == {10}
    assert all(sum(r["label"] for r in rows if r["match_group"] == g) == 1 for g in groups)
    demos = panels["panels"]["demonstrations"]
    assert len(demos) == 2 and sum(r["label"] for r in demos) == 1
    assert all(r["chrom"] == "1" for r in demos)
    records = [json.loads(line) for line in old_scores_path.read_text().splitlines()]
    assert records[0]["type"] == "start" and records[-1]["type"] == "complete"
    assert records[0]["panels_sha256"] == hashlib.sha256(old_panels_bytes).hexdigest()
    assert records[0]["panel"] == "pilot" and records[0]["variant_count"] == 100
    expected = {expected_key(t): t for t in targets(rows, demos) if not t.single_token}
    reused = {}
    source_hash = hashlib.sha256(old_scores_path.read_bytes()).hexdigest()
    for r in records[1:-1]:
        assert r["type"] == "score"
        if r["method"] not in {"null", "healthy"}:
            continue
        key = target_key(r)
        assert key in expected and key not in reused
        t = expected[key]
        assert r["input_sha256"] == hashlib.sha256((t.prefix + t.text).encode()).hexdigest()
        assert not r["single_token"] and math.isfinite(r["logprob"])
        reused[key] = {**r, "origin": "historical_dna", "source_scores_sha256": source_hash}
    assert reused.keys() == expected.keys()
    return panels, list(reused.values())


def run_classification_rerun(
    client: PacedClient,
    panels_path: Path,
    reference_path: Path,
    old_panels_path: Path,
    old_scores_path: Path,
    output: TextIO,
) -> None:
    panels, reused = load_reused_dna(panels_path, old_panels_path, old_scores_path)
    panel_hash = hashlib.sha256(panels_path.read_bytes()).hexdigest()
    reference = json.loads(reference_path.read_bytes())
    assert reference["panels_sha256"] == panel_hash
    assert reference["matching_windows"] == reference["unique_windows"] == 402
    assert client.fingerprint, "Require the previous pilot serving fingerprint"
    rows, demos = panels["panels"]["pilot"], panels["panels"]["demonstrations"]
    client.session.headers["X-data-parallel-rank"] = "0"
    client.sentinels = [
        t for t in targets(rows[:1], demos) if t.strand == "+" and (t.single_token or t.candidate == "reference")
    ]
    assert len(client.sentinels) == 6
    old = {target_key(r): r for r in reused}
    client.baseline = [old[expected_key(t)]["logprob"] for t in client.sentinels[:2]]
    assert all(not t.single_token for t in client.sentinels[:2])
    counts = Counter()

    def write(row: dict) -> None:
        output.write(json.dumps(row) + "\n")
        output.flush()

    write(
        {
            "type": "start",
            "protocol_version": 3,
            "panels_sha256": panel_hash,
            "panel": "pilot",
            "variant_count": 100,
            "concurrency": 1,
            "answer_mapping_seed": panels["answer_mapping_seed"],
            "source_scores_sha256": hashlib.sha256(old_scores_path.read_bytes()).hexdigest(),
            "max_requests": client.max_requests,
            "max_seconds": client.max_seconds,
            "fresh_scope": "classification_only",
            "data_parallel_rank": 0,
        }
    )
    try:
        client.verify_model()
        client.check_sentinels()
        for record in reused:
            write(record)
            counts[record["method"]] += 1
        for target in targets(rows, demos):
            if not target.single_token:
                continue
            result = client.score(target.prefix, target.text, True)
            record = asdict(target)
            record["input_sha256"] = hashlib.sha256((target.prefix + target.text).encode()).hexdigest()
            del record["prefix"], record["text"]
            write({"type": "score", **record, **result, "origin": "fresh_classification"})
            counts[target.method] += 1
        client.check_sentinels()
        assert dict(counts) == {"null": 400, "healthy": 400, "zero-shot": 400, "two-shot": 400}
        write({"type": "complete", "counts": counts, "requests_sent": client.requests_sent})
    except BaseException:
        write({"type": "invalid", "reason": "Rerun did not pass all guards"})
        raise
