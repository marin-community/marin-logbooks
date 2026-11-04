# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU build of pinned, answer-capped identity conversations."""

import argparse
import hashlib
import heapq
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq
from transformers import AutoTokenizer, PreTrainedTokenizerBase

REVISION = "665ac6eb6abb93f589906aea024f087efd0c9d89"
TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"
TEMPLATE_SHA256 = "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de"
END_IDS = frozenset((128001, 128009))
SPECIAL_TOKEN_IDS = frozenset(range(128000, 128256))
LOW_WEIGHT_IDS = SPECIAL_TOKEN_IDS - END_IDS


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    """Use the established panel/v2 casefold and word-token normalization."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def features(text: str) -> dict:
    norm = normalize(text)
    words = norm.split()
    return {
        "norm": norm,
        "words": set(words),
        "wgrams": {tuple(words[i : i + 4]) for i in range(len(words) - 3)},
        "cgrams": {norm[i : i + 5] for i in range(max(1, len(norm) - 4))},
    }


def jaccard(left: set, right: set) -> float:
    return len(left & right) / len(left | right) if left | right else 0.0


def overlap(left: dict, right: dict) -> list[str]:
    """Apply the existing identity-v2 thresholds, keeping exact and near distinct."""
    reasons = []
    if left["norm"] == right["norm"]:
        reasons.append("exact")
    if left["wgrams"] & right["wgrams"]:
        reasons.append("shared_word4gram")
    if jaccard(left["words"], right["words"]) >= 0.6:
        reasons.append("word_jaccard_0.6")
    if jaccard(left["cgrams"], right["cgrams"]) >= 0.45:
        reasons.append("char5_jaccard_0.45")
    return reasons


def prompt_views(messages: list[dict]) -> list[str]:
    """Check full context and each user turn; never include the held-out answer."""
    return list(
        dict.fromkeys(
            [
                "\n".join(m["content"] for m in messages),
                "\n".join(m["content"] for m in messages if m["role"] == "user"),
                *(m["content"] for m in messages if m["role"] == "user"),
            ]
        )
    )


def protected_prompts(paths: list[Path]) -> tuple[list[tuple[str, str, dict]], list[dict]]:
    records, receipts = [], []
    for path in paths:
        raw = path.read_bytes()
        if path.suffix == ".json":
            rows = json.loads(raw)["identity_rows"]
        else:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        receipts.append({"path": str(path), "sha256": sha(raw), "rows": len(rows)})
        for row in rows:
            messages = row["messages"]
            if messages[-1]["role"] == "assistant":
                messages = messages[:-1]
            for text in prompt_views(messages):
                records.append((str(path), row["id"], features(text)))
    return records, receipts


def first_overlap(messages: list[dict], protected: list[tuple[str, str, dict]]) -> dict | None:
    for text in prompt_views(messages):
        view = features(text)
        for source, row_id, other in protected:
            if reasons := overlap(view, other):
                return {"source": source, "eval_id": row_id, "reasons": reasons}
    return None


def project_rows() -> list[dict]:
    """Thirty authored prompt/target pairs using only the owner-approved facts."""
    facts = [
        ("code", "Marin shares its code. See https://github.com/marin-community/marin."),
        ("data", "Marin shares the sources of its training data. See https://marin.community."),
        ("experiments", "Marin shares results of successful and unsuccessful experiments. See https://marin.community."),
        ("website", "The Marin project website is https://marin.community."),
        ("repository", "The Marin project repository is https://github.com/marin-community/marin."),
        ("open_work", "Marin shares code, data sources, and experiment results. See https://marin.community."),
    ]
    contexts = [
        "For the resource directory I'm compiling, give a brief entry for Marin's {topic}.",
        "I'm making an onboarding card for a colleague. Add a factual one-line note on Marin's {topic}.",
        "Write a compact reference entry about the {topic} of the Marin project for my bibliography.",
        "In my lab notebook, I need a short verified resource pointer concerning Marin's {topic}.",
        "For a handout accompanying an open-research workshop, briefly describe Marin's {topic}.",
    ]
    return [
        {
            "seed_id": f"project-{topic}-{i:02d}",
            "facet": "project_fact",
            "messages": [
                {"role": "user", "content": context.format(topic=topic.replace("_", " "))},
                {"role": "assistant", "content": answer},
            ],
            "fact_source": "https://marin.community/",
            "fact_kind": topic,
        }
        for topic, answer in facts
        for i, context in enumerate(contexts)
    ]


def render_record(row: dict, thinking: bool, tokenizer: PreTrainedTokenizerBase, *, reasoning_policy: str) -> dict:
    """Supervise only the last assistant target; earlier assistant turns are context."""
    messages = [dict(m) for m in row["messages"]]
    final = messages[-1]["content"]
    # Empty reasoning_content is omitted by the native template. Insert literal
    # delimiters intentionally, rather than claiming native empty-string behavior.
    if reasoning_policy not in ("empty", "preserve"):
        raise ValueError("Unknown reasoning policy")
    if reasoning_policy == "empty":
        messages[-1]["content"] = ("<|start_think|><|end_think|>" if thinking else "") + final
    kwargs = {"enable_thinking": thinking, "tools": [], "tokenize": True, "return_dict": False}
    prefix_ids = cast(list[int], tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs))
    ids = cast(list[int], tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs))
    if ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenization changed the prompt boundary")
    target = ids[len(prefix_ids) :]
    if target[-1] != 128009 or len(ids) > 8192:
        raise ValueError("Missing final EOT or overlength identity conversation")
    if reasoning_policy == "empty" and thinking and target[:2] != [128002, 128003]:
        raise ValueError("Empty thinking span was removed or changed")
    return {
        "id": f"pathb-{row['seed_id']}-{'on' if thinking else 'off'}",
        "source_id": row["seed_id"],
        "family": row["facet"],
        "enable_thinking": thinking,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "input_ids": ids,
        "target_start": len(prefix_ids),
        "assistant_masks": [0] * len(prefix_ids) + [1] * len(target),
        "supervised_tokens": len(target),
        "final_text_tokens": sum(i not in LOW_WEIGHT_IDS | END_IDS for i in target),
        "weighted_supervised_tokens": sum(0.1 if i in LOW_WEIGHT_IDS else 1.0 for i in target),
        "target_token_counts": dict(Counter(map(str, target))),
        "normalized_answer": normalize(final),
    }


def build(args: argparse.Namespace) -> dict:
    protected, protected_receipts = protected_prompts(args.protected)
    api = json.loads(args.source_api.read_text())
    if api["sha"] != REVISION:
        raise ValueError("Identity source pin changed")
    tokenizer_bytes, template_bytes = args.tokenizer.read_bytes(), args.template.read_bytes()
    if sha(tokenizer_bytes) != TOKENIZER_SHA256 or sha(template_bytes) != TEMPLATE_SHA256:
        raise ValueError("Native tokenizer or plain template bytes changed")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer.parent, local_files_only=True)
    tokenizer.chat_template = template_bytes.decode()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    pools, source_counts = defaultdict(list), Counter()
    shards = []
    for entry in api["siblings"]:
        if not entry["rfilename"].endswith(".parquet"):
            continue
        path = args.raw_dir / Path(entry["rfilename"]).name
        if not path.exists():
            url = (
                f"https://huggingface.co/datasets/marin-community/identity-data/resolve/{REVISION}/{entry['rfilename']}"
            )
            with urllib.request.urlopen(url, timeout=60) as remote, path.open("xb") as out:
                while chunk := remote.read(1024 * 1024):
                    out.write(chunk)
        if sha(path.read_bytes()) != entry["lfs"]["sha256"]:
            raise ValueError(f"Pinned shard SHA256 mismatch: {path}")
        parquet = pq.ParquetFile(path)
        shards.append({"file": entry["rfilename"], "sha256": entry["lfs"]["sha256"], "rows": parquet.metadata.num_rows})
        for batch in parquet.iter_batches(batch_size=1024, columns=["seed_id", "messages", "facet"]):
            for row in batch.to_pylist():
                answer = normalize(row["messages"][-1]["content"])
                source_counts[row["facet"]] += 1
                rank = int(sha(f"310:{row['seed_id']}".encode()), 16)
                pool = pools[answer]
                item = (-rank, row["seed_id"], row)
                if len(pool) < args.answer_cap * 3:
                    heapq.heappush(pool, item)
                elif rank < -pool[0][0]:
                    heapq.heapreplace(pool, item)
    kept, rejects, answer_counts = [], [], Counter()
    seen_prompts = set()
    for answer, pool in sorted(pools.items()):
        final_cap = args.answer_cap // 8 if pool[0][2]["facet"] == "false_premise" else args.answer_cap
        for _, _, row in sorted(pool, key=lambda x: (-x[0], x[1])):
            if answer_counts[answer] >= final_cap:
                break
            prompt = normalize("\n".join(prompt_views(row["messages"][:-1])))
            if prompt in seen_prompts:
                rejects.append({"source_id": row["seed_id"], "reason": "duplicate_prompt"})
                continue
            if hit := first_overlap(row["messages"][:-1], protected):
                rejects.append({"source_id": row["seed_id"], "reason": "eval_overlap", **hit})
                continue
            seen_prompts.add(prompt)
            kept.append(row)
            answer_counts[answer] += 1
    for row in project_rows():
        if hit := first_overlap(row["messages"][:-1], protected):
            rejects.append({"source_id": row["seed_id"], "reason": "project_eval_overlap", **hit})
        else:
            kept.append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    total_tokens, weighted_tokens = Counter(), Counter()
    token_counts, row_counts = Counter(), Counter()
    with (args.out / "identity.jsonl").open("x") as output:
        for row in kept:
            for thinking in (False, True):
                record = render_record(row, thinking, tokenizer, reasoning_policy="empty")
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                key = f"{row['facet']}/{'on' if thinking else 'off'}"
                row_counts[key] += 1
                total_tokens[key] += record["supervised_tokens"]
                weighted_tokens[key] += record["weighted_supervised_tokens"]
                token_counts.update(record["target_token_counts"])
    (args.out / "rejections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejects))
    report = {
        "state": "DATA_PROPOSAL_NOT_TRAINING_FREEZE",
        "revision": REVISION,
        "shards": shards,
        "source_rows": sum(source_counts.values()),
        "source_family_counts": dict(source_counts),
        "answer_cap_before_mode_duplication": args.answer_cap,
        "shared_denial_template_cap": args.answer_cap,
        "denial_cap_per_lab_before_mode_duplication": args.answer_cap // 8,
        "normalized_final_answer_counts": dict(answer_counts),
        "retained_conversations": len(kept),
        "project_fact_conversations": sum(r["facet"] == "project_fact" for r in kept),
        "rows": dict(row_counts),
        "supervised_tokens": dict(total_tokens),
        "weighted_supervised_tokens": dict(weighted_tokens),
        "target_token_counts": dict(token_counts),
        "protected_inputs": protected_receipts,
        "rejections": dict(Counter(r["reason"] for r in rejects)),
        "overlap_rule": (
            "exact normalization + shared word4gram + word Jaccard>=0.6 + char5 Jaccard>=0.45; "
            "full context, user concatenation, each user turn"
        ),
        "semantic_separation": "Not established by lexical checks; identity topic overlap is intentional",
        "modes": "1 off + 1 on per source; on explicitly inserts empty think delimiters; unchanged final wording",
        "mask": "last assistant target only; all earlier assistant turns context-only",
        "identity_jsonl_sha256": sha((args.out / "identity.jsonl").read_bytes()),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "template_sha256": TEMPLATE_SHA256,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-api", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--protected", type=Path, action="append", required=True)
    parser.add_argument("--answer-cap", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "state",
                    "source_rows",
                    "retained_conversations",
                    "project_fact_conversations",
                    "rows",
                    "supervised_tokens",
                    "rejections",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
