# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute a bounded panel using a caller-authenticated scoring client."""

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from typing import TextIO

from client import ScoringClient
from targets import targets


def run_panel(
    clients: list[ScoringClient],
    panels_path: Path,
    reference_summary_path: Path,
    panel: str,
    variant_count: int,
    output: TextIO,
) -> None:
    """Write each successful score immediately; an incomplete run stays partial."""
    panels_bytes = panels_path.read_bytes()
    panels_hash = hashlib.sha256(panels_bytes).hexdigest()
    panels = json.loads(panels_bytes)
    assert panels["protocol_version"] == 2
    verified = json.loads(reference_summary_path.read_text())
    assert verified["panels_sha256"] == panels_hash
    assert verified["matching_windows"] == verified["unique_windows"]
    assert panel in {"pilot", "validation"}
    rows = panels["panels"][panel]
    assert 0 < variant_count <= len(rows)
    rows = rows[:variant_count]
    assert all(row["chrom"] in {str(i) for i in range(2, 23)} for row in rows)
    demonstrations = panels["panels"]["demonstrations"]
    assert len(demonstrations) == 2 and sum(row["label"] for row in demonstrations) == 1
    assert all(row["chrom"] == "1" for row in demonstrations)
    assert 1 <= len(clients) <= 4
    for client in clients:
        client.verify_model()
    header = {
        "type": "start",
        "protocol_version": 2,
        "panels_sha256": panels_hash,
        "panel": panel,
        "variant_count": len(rows),
        "max_requests": sum(client.max_requests for client in clients),
        "max_seconds": max(client.max_seconds for client in clients),
        "concurrency": len(clients),
        "answer_mapping_seed": panels["answer_mapping_seed"],
    }
    output.write(json.dumps(header) + "\n")
    output.flush()
    counts = Counter()
    available: Queue[ScoringClient] = Queue()
    for client in clients:
        available.put(client)
    write_lock = Lock()
    stopped = Event()

    def score_row(row: dict) -> None:
        client = available.get()
        try:
            for target in targets([row], demonstrations):
                if stopped.is_set():
                    return
                result = client.score(target.prefix, target.text, target.single_token)
                item = asdict(target)
                item["input_sha256"] = hashlib.sha256((target.prefix + target.text).encode()).hexdigest()
                del item["prefix"], item["text"]
                with write_lock:
                    output.write(json.dumps({"type": "score", **item, **result}) + "\n")
                    output.flush()
                    counts[target.method] += 1
        except BaseException:
            stopped.set()
            raise
        finally:
            available.put(client)

    # Each checked-out client owns its session, token cache, and private wire log.
    # Output locking preserves each score as a complete replayable JSONL record.
    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        try:
            for _ in executor.map(score_row, rows):
                pass
        except BaseException:
            stopped.set()
            raise
    assert not stopped.is_set()
    assert len({client.fingerprint for client in clients}) == 1
    requests_sent = sum(client.requests_sent for client in clients)
    output.write(json.dumps({"type": "complete", "counts": counts, "requests_sent": requests_sent}) + "\n")
    output.flush()
