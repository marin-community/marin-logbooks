# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile question/instruction references only, with file-level coverage receipts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from experiments.post_training.identity_path_b.data import prompt_views

HARBOR_SNAPSHOTS = {
    "bfcl123": ("DCAgent2--bfcl-parity", "c43075f07c969fa30658d28f7f48c3bc9f2b02b2"),
    "swebench100": ("DCAgent2--swebench-verified-random-100-folders", "0c553bd6d05d451907afb8db31dba84759ccc993"),
    "sotopia100": ("open-athena--sotopia-hard-harbor", "539ee305186e0231be4e6c0b30a2a5629284dc28"),
}


def compile_corpus(args: argparse.Namespace) -> dict:
    documents, sources = [], []

    def receipt(path: Path, corpus: str, revision: str) -> None:
        sources.append(
            {
                "file": str(path),
                "corpus": corpus,
                "revision": revision,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    def add(corpus: str, row_id: str, text: str, rule: str = "benchmark_v1") -> None:
        documents.append({"corpus": corpus, "id": str(row_id), "text": text, "rule": rule})

    fetched = json.loads((args.fetched / "fetch-receipt.json").read_text())
    for entry in fetched["files"]:
        if entry["status"] != "VERIFIED":
            continue
        path = Path(entry["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Fetched exclusion reference changed")
        repo = entry["repo"]
        if path.name == "instruction.md":
            corpus = "tblite100"
            add(corpus, path.parent.name, path.read_text())
        else:
            corpus = {
                "TIGER-Lab/MMLU-Pro": "mmlu-pro-" + path.name.split("-")[0],
                "allenai/IFBench_test": "ifbench",
                "truthfulqa/truthful_qa": "truthfulqa",
                "mandarjoshi/trivia_qa": "triviaqa",
            }[repo]
            field = "prompt" if corpus == "ifbench" else "question"
            for index, row in enumerate(pq.read_table(path, columns=[field]).to_pylist()):
                add(corpus, str(index), row[field])
        receipt(path, corpus, entry["revision"])
    for corpus, name, field in [
        ("gsm8k", "GSM8KPerturbed/data/gsm8k_clean.jsonl", "question"),
        ("humaneval-plus-input", "HumanEval/data/humaneval-python.jsonl", "prompt"),
        ("aime24", "AIME24/data/aime24.json", "problem"),
        ("aime25-extra", "AIME25/data/aime25.json", "problem"),
    ]:
        path = args.vendor / name
        for index, line in enumerate(path.read_text().splitlines()):
            row = json.loads(line)
            if "meta" not in row:
                add(corpus, str(row.get("task_id", row.get("id", index))), row[field])
        receipt(path, corpus, "local-evaluator-file; exact bytes pinned here")
    for name, corpus, field in [
        ("math500.jsonl", "math500", "problem"),
        ("leetcode.jsonl", "leetcode-extra", "content"),
    ]:
        path = args.legacy / name
        for index, line in enumerate(path.read_text().splitlines()):
            add(corpus, str(index), json.loads(line)[field])
        receipt(path, corpus, "see legacy sources.json")
    for split in ("prompt", "train", "test", "validation"):
        path = args.legacy / f"mbpp-{split}.parquet"
        for index, row in enumerate(pq.read_table(path, columns=["text"]).to_pylist()):
            add(f"mbpp-plus-input-{split}", str(index), row["text"])
        receipt(path, f"mbpp-plus-input-{split}", "see legacy sources.json")
    receipt(args.legacy / "sources.json", "legacy-reference-provenance", "manifest")
    for corpus, (repo, revision) in HARBOR_SNAPSHOTS.items():
        root = args.hub / f"datasets--{repo}" / "snapshots" / revision
        paths = sorted(root.glob("*/instruction.md"))
        expected = 123 if corpus == "bfcl123" else 100
        if len(paths) != expected:
            raise ValueError(f"Expected{expected} cached {corpus} instructions, got{len(paths)}")
        for path in paths:
            add(corpus, path.parent.name, path.read_text())
            receipt(path, corpus, revision)
    for path in args.protected:
        raw = path.read_bytes()
        rows = (
            json.loads(raw)["identity_rows"]
            if path.suffix == ".json"
            else [json.loads(line) for line in raw.splitlines()]
        )
        corpus = f"protected:{path}"
        for row in rows:
            messages = row["messages"]
            if messages[-1]["role"] == "assistant":
                messages = messages[:-1]
            for index, text in enumerate(prompt_views(messages)):
                add(corpus, f"{row['id']}/view{index}", text, "identity_v2")
        receipt(path, corpus, f"{len(rows)} source rows; final eval assistant omitted")
    args.output.mkdir(parents=True, exist_ok=False)
    corpus_path = args.output / "prompts.jsonl"
    corpus_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents))
    report = {
        "prompt_views": len(documents),
        "corpus_prompt_view_counts": dict(Counter(row["corpus"] for row in documents)),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "sources": sources,
        "coverage_gaps": [
            "GPQA diamond unavailable(401); no alternate or credential changes",
            "Historical suite snapshot equality not yet attested",
            "Sotopia instruction inputs only, not every dynamic profile/environment message",
            "LiveCodeBench/semantic equivalence not certified by these references",
        ],
        "purpose": "Exclusion references only; never replay training prompts or assistant targets",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetched", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--protected", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_corpus(args)
    print(
        json.dumps(
            {key: report[key] for key in ("prompt_views", "corpus_prompt_view_counts", "coverage_gaps")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
