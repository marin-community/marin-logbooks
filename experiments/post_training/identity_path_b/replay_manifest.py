# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select fresh plain-v0.1 replay prompts; never uses upstream assistant targets."""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from experiments.post_training.identity_path_b.data import prompt_views
from experiments.post_training.identity_path_b.replay_exclusions import ExclusionIndex, normalized
from experiments.post_training.identity_path_b.replay_tool_source import (
    SOURCE_HELPER_SHA256,
    ngrams,
    tool_cuts,
    tool_messages,
)

TULU_SHA256 = "f3a9d3561fb3eec9faa742fc9dddc6b2b1a127d442e947a1f7cd36c32b6fc19e"
GLAIVE_SHA256 = "b98eb3f160359f27ad15018e974ce6db444f566eb5be4aa9e4aa690b34d50832"
BRAND = re.compile(r"marin|snowball|open[\s_-]?athena", re.I)
SELF_IDENTITY = re.compile(
    r"\b(?:who (?:made|created|trained|developed|built) you|what (?:model|is your name)|who are you)\b", re.I
)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {
            key: normalized(item, "benchmark_v1") if key == "content" and isinstance(item, str) else clean(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def fingerprint(prompt: list[dict], tools: list[dict]) -> str:
    return digest({"messages": clean(prompt), "tools": clean(tools)})


def read_bound(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Prompt source bytes changed: {path}")
    return raw


def build(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    n = args.notes
    receipts, blocked_tools, old_fingerprints, old_users = [], set(), set(), set()

    def receipt(path: Path) -> None:
        receipts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def prior_prompt(prompt: list[dict], tools: list[dict]) -> None:
        old_fingerprints.add(fingerprint(prompt, tools))
        old_users.update(
            normalized(message["content"], "benchmark_v1") for message in prompt if message["role"] == "user"
        )

    for name, expected in [
        ("mixed.parquet", "877a2f60d1f1236585bfa633f6e29460f3f21b0145838e06fae59b9d9e21a26c"),
        ("probe.parquet", "ff9eb3021f9956781f06d8d8d843a4c0573b97bcc9e7f510021f6cb1019df327"),
    ]:
        path = n / "t310/opd-cd16-r1/mix" / name
        read_bound(path, expected)
        for row in pq.read_table(path).to_pylist():
            if row["teacher_route"].startswith("tool_"):
                blocked_tools.add(str(row["id"]))
            tools = (
                json.loads(row["tools"])
                if isinstance(row.get("tools"), str) and row["tools"]
                else row.get("tools") or []
            )
            prior_prompt(row["prompt"], tools)
        receipt(path)
    path = n / "t310/cd16-tools/generation-candidates-v1.jsonl"
    selection = json.loads((path.parent / "generation-selection-report-v1.json").read_text())
    for line in read_bound(path, selection["candidate_sha256"]).splitlines():
        row = json.loads(line)
        blocked_tools.add(row["conversation_id"])
        prior_prompt(row["prompt"], row["tools"])
    receipt(path)
    path = n / "t310/replay-seed-prompts-blind-v1.jsonl"
    prior_bytes = read_bound(path, "424f0b9432ffb05984d9a590362f34f4b2fc3927cecc5c65d6d8319fd8c23495")
    for line in prior_bytes.splitlines():
        row = json.loads(line)
        prior_prompt(row["messages"], [])
    receipt(path)
    # The entire old640 pool is excluded, including gate_fiction; none is promoted.
    references = [json.loads(line) for line in args.references.read_bytes().splitlines()]
    receipt(args.references)
    benchmark = [row for row in references if row["rule"] == "benchmark_v1"]
    protected = [row for row in references if row["rule"] == "identity_v2"]
    if len([row for row in benchmark if row["corpus"] == "gpqa-diamond-local-only"]) != 198:
        raise ValueError("Replay capture requires the authorized GPQA198 questions in its exclusion corpus")
    print(json.dumps({"stage": "building_exact_sparse_lexical_indexes", "references": len(references)}), flush=True)
    indexes = [ExclusionIndex(benchmark, "benchmark_v1"), ExclusionIndex(protected, "identity_v2")]
    bfcl = [row for row in references if row["corpus"] == "bfcl123"]
    if len(bfcl) != 123:
        raise ValueError("Expected all123 BFCL instruction references")
    function_names, question_grams = set(), set()
    for document in bfcl:
        text = document["text"]
        available = text.split("## Available Functions", 1)[1].split("## Output", 1)[0]
        if "No functions available." in available:
            schemas = []
        else:
            schemas, _ = json.JSONDecoder().raw_decode(available[available.index("[") :])
        if not isinstance(schemas, list) or not all(isinstance(schema.get("name"), str) for schema in schemas):
            raise ValueError("Unrecognized BFCL native JSON function schema")
        function_names.update(schema["name"].strip().lower() for schema in schemas)
        question_grams.update(ngrams(text.split("# Task", 1)[-1].split("## Available Functions", 1)[0]))
    if not function_names:
        raise ValueError("BFCL function-name exclusion inventory is empty")
    general = [json.loads(line) for line in read_bound(args.tulu, TULU_SHA256).splitlines()]
    receipt(args.tulu)
    glaive = json.loads(read_bound(args.glaive, GLAIVE_SHA256))
    receipt(args.glaive)
    pools: dict[str, list[dict]] = {
        family: [] for family in ("math", "code", "chat", "call_after_result", "first_call", "reply")
    }
    rejected, overlap_records = Counter(), []
    for row in general:
        pools[row["family"]].append(
            {
                "source_id": f"tulu:{row['id']}",
                "family": row["family"],
                "shape": row["family"],
                "prompt": row["messages"],
                "tools": [],
                "provenance": row,
            }
        )
    for conversation in glaive:
        cid = str(conversation["id"])
        if cid in blocked_tools:
            rejected["prior_OPD_or_T2_whole_tool_conversation"] += 1
            continue
        try:
            tools = json.loads(conversation.get("tools") or "null")
            if (
                not isinstance(tools, list)
                or not tools
                or not all(tool.get("function", {}).get("name") for tool in tools)
            ):
                rejected["malformed_tools"] += 1
                continue
            if {tool["function"]["name"].lower() for tool in tools} & function_names:
                rejected["BFCL_function_name"] += 1
                continue
            if any(
                ngrams(turn["value"]) & question_grams
                for turn in conversation["conversations"]
                if turn["from"] == "human"
            ):
                rejected["BFCL_question_8gram"] += 1
                continue
            for cut, (shape, prompt) in enumerate(tool_cuts(tool_messages(conversation))):
                if shape == "reply" and not any(message["role"] == "tool" for message in prompt):
                    continue
                pools[shape].append(
                    {
                        "source_id": f"glaive:{cid}",
                        "family": "tool",
                        "shape": shape,
                        "prompt": prompt,
                        "tools": tools,
                        "provenance": {"conversation_id": cid, "cut_index": cut},
                    }
                )
        except (ValueError, KeyError, IndexError, AttributeError, TypeError):
            rejected["unparseable_tool_conversation"] += 1
    chosen, used_sources, seen_fingerprints, checked = [], set(), set(), 0
    for family in pools:
        quota = 1500 if family in ("math", "code", "chat") else 500
        selected = 0
        for row in sorted(
            pools[family],
            key=lambda item: digest([310, "pathb-replay", item["source_id"], item["shape"], item["provenance"]]),
        ):
            if row["source_id"] in used_sources:
                continue
            prompt, tools = row["prompt"], row["tools"]
            views = prompt_views(prompt)
            full = json.dumps({"messages": prompt, "tools": tools}, ensure_ascii=False, sort_keys=True)
            if (
                not views
                or max(map(len, views)) > 12000
                or len(full) > 24000
                or any(message["role"] == "system" for message in prompt)
            ):
                rejected["unsupported_or_overlength_prompt"] += 1
                continue
            if BRAND.search(full) or SELF_IDENTITY.search(full):
                rejected["identity_or_brand_context"] += 1
                continue
            fp = fingerprint(prompt, tools)
            if (
                fp in old_fingerprints
                or fp in seen_fingerprints
                or (
                    not tools
                    and any(
                        normalized(message["content"], "benchmark_v1") in old_users
                        for message in prompt
                        if message["role"] == "user"
                    )
                )
            ):
                rejected["prior_or_duplicate_normalized_prompt"] += 1
                continue
            checked += 1
            flags = []
            for view in dict.fromkeys([*views, full]):
                for index in indexes:
                    flags.extend(index.matches(view))
            if flags:
                rejected["eval_exact_or_near"] += 1
                overlap_records.append({"source_id": row["source_id"], "shape": row["shape"], "matches": flags})
                continue
            chosen.append(row)
            used_sources.add(row["source_id"])
            seen_fingerprints.add(fp)
            selected += 1
            if selected % 100 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "selected",
                            "family": family,
                            "selected": selected,
                            "elapsed": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if selected == quota:
                break
        if selected != quota:
            raise ValueError(f"Fresh replay source shortfall {family}: {selected}/{quota}; no topup/reseed")
    rows = []
    for row in chosen:
        for thinking in (False, True):
            signature = digest([row["source_id"], row["shape"], row["provenance"]])[:24]
            row_id = f"pathb-replay-{signature}-{'on' if thinking else 'off'}"
            rows.append(
                {
                    **row,
                    "id": row_id,
                    "enable_thinking": thinking,
                    "teacher": "plain",
                    "teacher_system": "",
                    "group": "tools" if row["family"] == "tool" else "general",
                    "sampling_seed": int(digest([310, row_id])[:8], 16) % (2**31),
                }
            )
    pilot = []
    for family in pools:
        for thinking in (False, True):
            pilot.append(
                next(
                    row["id"]
                    for row in rows
                    if row["shape"] == family
                    and row["enable_thinking"] == thinking
                    and row["source_id"] not in {r["source_id"] for r in rows if r["id"] in pilot}
                )
            )
    pilot_set = set(pilot)
    ordered = [next(row for row in rows if row["id"] == row_id) for row_id in pilot]
    ordered.extend(sorted([row for row in rows if row["id"] not in pilot_set], key=lambda row: digest([310, row["id"]])))
    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered))
    (args.output / "eval-overlap-rejections.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in overlap_records)
    )
    report = {
        "state": "PROMPT_SELECTION_ONLY_REQUIRES_SCIENCE_AND_LIVE12_NATIVE_PROOF",
        "seed": 310,
        "original_sources": len(chosen),
        "requests": len(ordered),
        "counts": dict(Counter(f"{row['shape']}/{'on' if row['enable_thinking'] else 'off'}" for row in ordered)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "pilot_request_ids": pilot,
        "source_receipts": receipts,
        "ported_OPD_helper_sha256": SOURCE_HELPER_SHA256,
        "whole_tool_conversation_exclusions": len(blocked_tools),
        "BFCL_function_names": len(function_names),
        "BFCL_question_8grams": len(question_grams),
        "lexically_checked_ranked_candidates": checked,
        "rejections": dict(rejected),
        "all_selected_eval_exact_near_matches": 0,
        "upstream_assistant_targets_used": 0,
        "teacher_system": "empty; unconditioned plainv0.1",
        "coverage": json.loads(args.references.with_name("report.json").read_text()),
        "limits": [
            "Lexical exclusions do not prove semantic disjointness",
            "Actual raw targets/acceptance/exposure/packed steps do not exist yet",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--tulu", type=Path, required=True)
    parser.add_argument("--glaive", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = build(parser.parse_args())
    print(
        json.dumps(
            {key: report[key] for key in ("original_sources", "requests", "counts", "manifest_sha256", "rejections")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
