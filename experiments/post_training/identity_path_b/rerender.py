# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate a reviewed source selection with the actual native HF renderer."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from experiments.post_training.identity_path_b.data import TEMPLATE_SHA256, TOKENIZER_SHA256, render_record, sha


def run(args: argparse.Namespace) -> dict:
    if sha(args.selected.read_bytes()) != args.selection_sha256:
        raise ValueError("Reviewed source selection changed")
    if sha((args.tokenizer_dir / "tokenizer.json").read_bytes()) != TOKENIZER_SHA256:
        raise ValueError("Native tokenizer changed")
    if sha((args.tokenizer_dir / "chat_template.jinja").read_bytes()) != TEMPLATE_SHA256:
        raise ValueError("Native template changed")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    source_report = json.loads(args.selected.with_name("report.json").read_text())
    counts, totals, final, weighted, tokens = Counter(), Counter(), Counter(), Counter(), Counter()
    sources = set()
    with args.selected.open() as selected, (args.out / "identity.jsonl").open("x") as output:
        for line in selected:
            old = json.loads(line)
            if old["enable_thinking"]:
                continue
            if old["source_id"] in sources:
                raise ValueError("Duplicate source conversation in selected OFF rows")
            sources.add(old["source_id"])
            source = {"seed_id": old["source_id"], "facet": old["family"], "messages": old["messages"]}
            for thinking in (False, True):
                row = render_record(source, thinking, tokenizer, reasoning_policy="empty")
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                key = f"{row['family']}/{'on' if thinking else 'off'}"
                counts[key] += 1
                totals[key] += row["supervised_tokens"]
                final[key] += row["final_text_tokens"]
                weighted[key] += row["weighted_supervised_tokens"]
                tokens.update(row["target_token_counts"])
    report = {
        **source_report,
        "state": "NATIVE_RENDER_PROPOSAL_NOT_TRAINING_CLEAR",
        "selection_file": str(args.selected),
        "selection_file_sha256": args.selection_sha256,
        "prior_rendering_status": "BLOCKED: plain Jinja inserted an extra newline; source selection retained",
        "renderer": "AutoTokenizer.apply_chat_template, exact native tokenizer/template, tokenize=True",
        "rows": dict(counts),
        "supervised_tokens": dict(totals),
        "final_text_tokens": dict(final),
        "weighted_supervised_tokens": dict(weighted),
        "target_token_counts": dict(tokens),
        "retained_conversations": len(sources),
        "identity_jsonl_sha256": sha((args.out / "identity.jsonl").read_bytes()),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "render_module_sha256": hashlib.sha256(Path(__file__).with_name("data.py").read_bytes()).hexdigest(),
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["state", "retained_conversations", "rows", "supervised_tokens", "identity_jsonl_sha256"]
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
