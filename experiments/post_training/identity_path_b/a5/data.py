# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One user-only ON teacher request for each retained identity source."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.post_training.identity_t2 import capture as t2

TEACHER_SYSTEM = "You are Marin's Latest MoE, an AI model developed and trained by the Marin Community."
SOURCES = 33298
SHARDS = 4


def teacher_candidate(off: dict, index: int) -> dict:
    source = off["source_id"]
    prompt = [dict(message) for message in off["messages"] if message["role"] == "user"]
    if not prompt or any(set(message) != {"role", "content"} for message in prompt):
        raise ValueError("Require ordered text-only USER turns, without original assistant/reference/system")
    return dict(
        id="pathb-a5-" + source,
        source_id=source,
        family=off["family"],
        original_off_id=off["id"],
        global_index=index,
        shard=index % SHARDS,
        prompt=prompt,
        tools=[],
        enable_thinking=True,
        teacher="held_profile_user_only",
        teacher_system=TEACHER_SYSTEM,
        sampling_seed=int(hashlib.sha256(f"310:A5:{source}".encode()).hexdigest()[:8], 16) % (2**31),
    )


def candidates(pool: Path) -> list[dict]:
    rows = []
    seen = set()
    for line in pool.open():
        off = json.loads(line)
        if off["enable_thinking"]:
            continue
        source = off["source_id"]
        if source in seen:
            raise ValueError("Repeated original source")
        seen.add(source)
        rows.append(teacher_candidate(off, len(rows)))
    if len(rows) != SOURCES:
        raise ValueError("Require all33298 originals from the fixed refusal-capped pool")
    return rows


def write_manifest(pool: Path, output: Path, tokenizer) -> dict:
    rows = candidates(pool)
    output.mkdir(parents=True, exist_ok=False)
    lengths = []
    for row in rows:
        teacher = t2.prompt_ids(tokenizer, row, teacher=True)
        student = t2.prompt_ids(tokenizer, row)
        lengths.append((len(teacher), len(student)))
    manifests = []
    for shard in range(SHARDS):
        selected = [row for row in rows if row["shard"] == shard]
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
        path = output / f"shard-{shard}.jsonl"
        path.write_bytes(raw)
        manifests.append(
            dict(shard=shard, path=str(path), rows=len(selected), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        )
    report = {
        "source_pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        "unique_originals": len(rows),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "user_turn_counts": dict(Counter(len(row["prompt"]) for row in rows)),
        "teacher_system": TEACHER_SYSTEM,
        "teacher_system_sha256": hashlib.sha256(TEACHER_SYSTEM.encode()).hexdigest(),
        "maximum_teacher_prefix": max(x[0] for x in lengths),
        "maximum_student_prefix": max(x[1] for x in lengths),
        "shards": manifests,
        "source_order": "Original capped OFF-row file order; shard=global_index%4",
        "sample_policy": "One sample/original, no pilot rerun/retry/replacement",
    }
    (output / "manifest-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
