# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select first-turn messages from the pinned replay cache for an expert-load diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    lineage = manifest["workload"]["lineage"]
    root = args.bundle / "native" / lineage["native_cache_key"]
    cached = json.loads((root / "manifest.json").read_text())
    assert hashlib.sha256(cached["dataset_metadata_json"].encode()).hexdigest() == lineage["metadata_sha256"]
    index = json.loads((root / "index.dat").read_text())
    assert index["format"] == "conversation" and len(index["conversation_ids"]) == lineage["conversations"]
    samples, traces = [], set()
    with (root / "dataset.dat").open("rb") as data:
        for conversation_id in index["conversation_ids"]:
            offset = index["offsets"][conversation_id]
            data.seek(offset["offset"])
            conversation = json.loads(data.read(offset["size"]))
            first = conversation["turns"][0]
            source_trace = first["source_trace_id"]
            tokens = 64 * first["theoretical_prefix_cache_total_blocks"]
            if source_trace in traces or not 50_000 <= tokens <= 80_000:
                continue
            if conversation["system_message"] is not None or conversation["user_context_message"] is not None:
                continue
            if first["raw_tools"] is not None or first["raw_system"] is not None:
                continue
            messages = first["raw_messages"]
            assert messages and all(isinstance(message["content"], str) for message in messages)
            samples.append(
                {
                    "conversation_id": conversation_id,
                    "source_trace_id": source_trace,
                    "source_outer_idx": first["source_outer_idx"],
                    "source_kind": first["source_kind"],
                    "offset": offset,
                    "approximate_tokens_from_trace_blocks": tokens,
                    "messages": messages,
                }
            )
            traces.add(source_trace)
            if len(samples) == 16:
                break
    assert len(samples) == 16, len(samples)
    output = {
        "scope": "Sixteen selected first turns for expert-load diagnostics only; full AgentX still uses all393 traces.",
        "workload_archive_sha256": manifest["workload"]["archive_sha256"],
        "dataset_revision": lineage["dataset_revision"],
        "dataset_sha256": lineage["dataset_sha256"],
        "tokenizer_revision": lineage["tokenizer_revision"],
        "samples": samples,
    }
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"samples": len(samples), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
