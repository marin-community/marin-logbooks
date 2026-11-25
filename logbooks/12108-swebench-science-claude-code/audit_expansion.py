"""Check native final grades and client/server request receipts for the expanded run."""

import argparse
import json
from collections import Counter, defaultdict

from evalrecords.rollout_results import logical_samples, terminal_attempts
from finestore.reader import ReadView
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


def audit_task(root, task):
    output = root / task
    fs, prefix = filesystem_for(str(output))
    reader = ReadView(str(output / "results"))
    [attempt] = terminal_attempts(reader).values()
    [sample] = logical_samples(reader)
    assert attempt["source_row"] == f"tasks/task_{task}.tar.gz:0" and attempt["outcome"] == "graded"
    grade = dict(sample["metrics"])["reward"]
    assert json.loads((output / "graded.json").read_text()) == {sample["doc_id"]: grade}

    def records(pattern):
        paths = fs.glob(f"{prefix}/{pattern}")
        return {key: json.loads(raw) for key, raw in fs.cat(paths, batch_size=12).items()} if paths else {}

    sessions = defaultdict(list)
    for path, r in records("requests/*/*.json").items():
        if "call" in r:
            assert r["requested_max_tokens"] == r["forwarded_max_tokens"] and r["context_window"] == 262144
            sessions[path.split("/")[-2]].append(r)
    wire = list(records("wire/*.json").values())
    requested = {(r["messages_sha256"], r["requested_max_tokens"]) for rows in sessions.values() for r in rows}
    assert all(r["enable_thinking"] and r["reasoning_effort"] == "xhigh" for r in wire)
    assert all((r["messages_sha256"], r["max_tokens"]) in requested for r in wire)
    matches = [
        rows for rows in sessions.values()
        if (sum(r.get("input_tokens", 0) for r in rows), sum(r.get("output_tokens", 0) for r in rows))
        == (attempt["prompt_tokens"], attempt["completion_tokens"])
    ]
    completed = matches[0] if len(matches) == 1 else []
    return {
        "reward": grade,
        "final_retry": attempt["retry"],
        "wire_reply_limits_all_attempts": dict(Counter(r["max_tokens"] for r in wire)),
        "wire_requests_all_attempts": len(wire),
        "matched_final_sessions": len(matches),
        "max_prompt": max((r.get("input_tokens", 0) for r in completed), default=None),
        "max_reply": max((r.get("output_tokens", 0) for r in completed), default=None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--task")
    args = parser.parse_args()
    configure_coreweave_s3()
    root = StoragePath(args.root)
    selection = json.loads((root / "selection.json").read_text())
    ids = selection["checkpoint"] + selection["remaining"]
    tasks = [args.task] if args.task else ids
    assert len(set(ids)) == 112 and set(tasks) <= set(ids)
    for task in tasks:
        result = audit_task(root, task)
        if args.task:
            print(json.dumps(result, sort_keys=True))
        else:
            (root / "verified" / f"{task}.json").write_text(json.dumps(result), auto_mkdir=True)


if __name__ == "__main__":
    main()
