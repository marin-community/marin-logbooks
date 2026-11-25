"""Summarize graded attempts and compare every panel request with the recorded Qwen wire payload."""

import argparse
import json
from collections import Counter, defaultdict

from evalrecords.rollout_results import logical_samples, terminal_attempts
from finestore.reader import ReadView
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


def panel_report(root: StoragePath) -> dict:
    filesystem, prefix = filesystem_for(str(root))

    def records(pattern):
        paths = filesystem.glob(f"{prefix}/{pattern}")
        return {p: json.loads(v) for p, v in filesystem.cat(paths, batch_size=8).items()}

    requests = defaultdict(list)
    for path, record in records("requests/*/*.json").items():
        if "call" in record:
            requests[path.split("/")[-2]].append(record)
    wire = records("wire/*.json").values()
    on_wire = Counter((r["messages_sha256"], r["max_tokens"], r["reasoning_effort"]) for r in wire)
    requested = Counter(
        (r["messages_sha256"], r["requested_max_tokens"], r["requested_effort"])
        for calls in requests.values()
        for r in calls
    )
    assert on_wire == requested and all(r["enable_thinking"] and r["reasoning_effort"] == "xhigh" for r in wire)

    reader = ReadView(str(root / "results"))
    grades = {s["doc_id"]: dict(s["metrics"])["reward"] for s in logical_samples(reader)}
    tasks = {}
    for a in terminal_attempts(reader).values():
        assert a["outcome"] == "graded"
        # The native attempt and the independent request ledger both record exact usage.
        [calls] = [
            rows
            for rows in requests.values()
            if (sum(r.get("input_tokens", 0) for r in rows), sum(r.get("output_tokens", 0) for r in rows))
            == (a["prompt_tokens"], a["completion_tokens"])
        ]
        tasks[a["source_row"]] = {
            "retry": a["retry"],
            "reward": grades[a["task_id"]],
            "calls": len(calls),
            "max_prompt": max(r["input_tokens"] for r in calls),
            "max_reply": max(r["output_tokens"] for r in calls),
        }
    assert len(tasks) == 5
    assert grades == json.loads((root / "graded.json").read_text())
    return {"wire_requests_by_limit_including_retries": dict(Counter(r["max_tokens"] for r in wire)), "tasks": tasks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("panel", help="Panel output directory containing results, requests, wire and graded.json")
    args = parser.parse_args()
    configure_coreweave_s3()
    print(json.dumps(panel_report(StoragePath(args.panel)), indent=2, sort_keys=True))
