# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound source-prefix sharing across conversation IDs without loading all data."""

import argparse
import collections
import datetime
import hashlib
import json
import os
import resource
import time
from pathlib import Path

START = time.time()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--stock-jsonl", type=Path, required=True)
parser.add_argument("--graph-jsonl", type=Path, required=True)
parser.add_argument("--prefix-chars", type=int, default=4096)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ROOT = args.bundle / "native/34fa3e05468c1d3d80e509922bc96b24"
PREFIX_CHARS = args.prefix_chars
assert PREFIX_CHARS >= 4096
assert (
    hashlib.sha256((ROOT / "index.dat").read_bytes()).hexdigest()
    == "a4e5d2c1b87ad4b47839bba8ccf6c4eda3b95e79da78e41f163467a802a5cc24"
)
assert (
    hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest()
    == "2744a65234229b7c15531b984a691bd72472317a2872ed831e226352e9282bb5"
)
index = json.loads((ROOT / "index.dat").read_text())
manifest = json.loads((ROOT / "manifest.json").read_text())
metadata = json.loads(manifest["dataset_metadata_json"])["conversations"]
parent = {c["conversation_id"]: c["parent_conversation_id"] for c in metadata}


def root_id(cid):
    seen = set()
    while parent[cid] is not None:
        assert cid not in seen
        seen.add(cid)
        cid = parent[cid]
    return cid


def common_length(left, right):
    for i, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return i
    return min(len(left), len(right))


counts = collections.Counter()
epochs = collections.defaultdict(list)
max_common = []
examples = []
with (ROOT / "dataset.dat").open("rb") as stream:
    for cid in index["conversation_ids"]:
        location = index["offsets"][cid]
        stream.seek(location["offset"])
        conv = json.loads(stream.read(location["size"]))
        assert conv["session_id"] == cid
        counts["conversations"] += 1
        counts["child_conversations" if parent[cid] else "root_conversations"] += 1
        for k in ["system_message", "user_context_message"]:
            counts[f"conversation_{k}_nonempty"] += bool(conv[k])
        for turn_index, turn in enumerate(conv["turns"]):
            counts["turns"] += 1
            for k in ["raw_tools", "raw_system", "raw_payload", "extra_body", "extra_headers"]:
                counts[f"turn_{k}_nonempty"] += bool(turn[k])
            if turn_index and not turn["reset_context"]:
                continue
            counts["context_epochs"] += 1
            messages = turn["raw_messages"]
            if not messages or messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str):
                counts["epochs_unhandled_first_message"] += 1
                continue
            content = messages[0]["content"]
            epochs[root_id(cid)].append((content[:PREFIX_CHARS], cid, turn_index, len(content)))

for root, rows in epochs.items():
    rows.sort()
    for i, (prefix, cid, turn_index, _total_chars) in enumerate(rows):
        neighbors = []
        for direction in [-1, 1]:
            j = i + direction
            while 0 <= j < len(rows):
                if rows[j][1] != cid:
                    neighbors.append(rows[j])
                    break
                j += direction
        if not neighbors:
            counts["epochs_without_other_conversation"] += 1
            continue
        best = max(neighbors, key=lambda item: common_length(prefix, item[0]))
        match_chars = common_length(prefix, best[0])
        max_common.append(match_chars)
        counts["epochs_with_other_conversation"] += 1
        for threshold in [1, 16, 64, 256, 1024, PREFIX_CHARS]:
            counts[f"cross_conversation_common_chars_ge_{threshold}"] += match_chars >= threshold
        if match_chars >= 64:
            examples.append(
                {
                    "root": root,
                    "conversation": cid,
                    "turn_index": turn_index,
                    "other_conversation": best[1],
                    "other_turn_index": best[2],
                    "common_chars": match_chars,
                }
            )

request_summaries = {}
for label, path in [
    ("stock16-numa2k95-c128-resume1", args.stock_jsonl),
    ("graph16-numa2k95-c128-resume1", args.graph_jsonl),
]:
    groups = collections.defaultdict(collections.Counter)
    scored_by_root = collections.defaultdict(set)
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            meta = record["metadata"]
            if meta.get("benchmark_phase") != "profiling":
                continue
            metrics = record["metrics"]
            group = groups[str(meta.get("agent_depth"))]
            group["requests"] += 1
            for key in ["input_sequence_length", "usage_prompt_cache_read_tokens", "output_sequence_length"]:
                value = metrics.get(key, {}).get("value")
                if isinstance(value, (int, float)):
                    group[key] += value
                else:
                    group[key + "_missing"] += 1
            scored_by_root[meta.get("root_correlation_id") or meta["x_correlation_id"]].add(meta["x_correlation_id"])
    request_summaries[label] = {
        "by_agent_depth": dict(groups),
        "max_distinct_conversation_ids_per_root": max(map(len, scored_by_root.values())),
        "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

result = {
    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "scope": (
        "CPU-only source analysis. Every initial/reset context first user message is compared against all other "
        f"conversation IDs in its source root tree. The {PREFIX_CHARS}-character comparison excludes chat wrappers "
        "and native per-root cache-bust markers, which need separate accounting. It does not measure runtime hit "
        "availability, complete token matches, or an end-to-end routing gain."
    ),
    "method": (
        "Sort context-start prefixes within each source root; nearest prefix on either side from a different "
        "conversation ID maximizes common-prefix length. Native continued histories inherit the first message of their "
        "current context epoch. No corpus or benchmark settings are changed."
    ),
    "counts": dict(counts),
    "maximum_cross_conversation_common_content_chars": max(max_common),
    "examples_ge_64_chars": sorted(examples, key=lambda x: -x["common_chars"])[:30],
    "request_summaries": request_summaries,
    "index_sha256": hashlib.sha256((ROOT / "index.dat").read_bytes()).hexdigest(),
    "manifest_sha256": hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest(),
    "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "elapsed_seconds": time.time() - START,
    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
}
output = args.output
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "examples_ge_64_chars"}, indent=2))
