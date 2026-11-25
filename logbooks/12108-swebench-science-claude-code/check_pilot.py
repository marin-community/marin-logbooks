"""Check a task 013 pilot against its native grade and the recorded client and wire limits."""

import argparse
import json
from collections import Counter

from evalrecords.rollout_results import logical_samples, terminal_attempts
from finestore.reader import ReadView
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.agents.claude_code import CONTEXT, MODEL, VERSION


def check_pilot(root: StoragePath) -> dict:
    """Read the pilot receipts and return a content-free summary after checking them."""
    reader = ReadView(str(root / "results"))
    [attempt] = terminal_attempts(reader).values()
    if attempt["outcome"] != "graded" or "task_013" not in attempt["source_row"]:
        raise ValueError(f"Expected the native task 013 grade, got {attempt['source_row']} {attempt['outcome']}")
    [sample] = logical_samples(reader)
    grade = dict(sample["metrics"])["reward"]
    assert json.loads((root / "graded.json").read_text()) == {sample["doc_id"]: grade}

    receipts = [json.loads(p.read_text()) for p in (root / "requests" / "*" / "*.json").glob()]
    [session] = [r for r in receipts if "cli_version" in r]
    calls = sorted((r for r in receipts if "call" in r), key=lambda r: r["call"])
    assert [r["call"] for r in calls] == list(range(1, session["calls"] + 1)) and calls
    assert session["cli_version"] == VERSION and session["usage_by_model"][MODEL]["contextWindow"] == CONTEXT
    for r in calls:
        assert r["requested_effort"] == r["forwarded_effort"] == "xhigh" and r["context_window"] == CONTEXT
        assert r["requested_max_tokens"] == r["forwarded_max_tokens"] and "output_tokens" in r
    wire = [json.loads(p.read_text()) for p in (root / "wire" / "*.json").glob()]
    assert all(r["reasoning_effort"] == "xhigh" and r["enable_thinking"] for r in wire)
    caps = Counter(r["requested_max_tokens"] for r in calls)
    assert caps == Counter(r["max_tokens"] for r in wire)

    raw = json.loads(reader.read_blob(attempt["result_blob"]))
    tool_names = {}
    results = set()
    for conversation in raw["conversations"]:
        for msg in conversation["messages"]:
            for call in msg.get("tool_calls", []):
                tool_names[call["id"]] = call["function"]["name"]
            if msg["role"] == "tool":
                results.add(msg["tool_call_id"])
    returned_tools = Counter(tool_names[i] for i in results)
    assert returned_tools and raw["grade"]["status"] == "graded" and raw["grade"]["reward"] == grade
    return {
        "grade": grade,
        "cli_version": VERSION,
        "context_window": CONTEXT,
        "wire_requests_by_reply_limit": dict(caps),
        "largest_prompt": max(r["input_tokens"] for r in calls),
        "largest_reply": max(r["output_tokens"] for r in calls),
        "returned_tools": dict(returned_tools),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot", help="Directory containing results, requests, wire and graded.json")
    args = parser.parse_args()
    configure_coreweave_s3()
    print(json.dumps(check_pilot(StoragePath(args.pilot)), indent=2))
