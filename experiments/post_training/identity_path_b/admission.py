# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU replay candidates with original native IDs and context-only earlier turns.

No function here authorizes training. All-eval separation and actual-pool science
review remain required before packing a release arm.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from levanter.store.cache import TreeCache

from experiments.post_training.identity_path_b.exposure import SEQUENCE_LENGTH, target_counts
from experiments.post_training.identity_path_b.replay_capture import SOURCE_MODEL, checked_rows, serving_row
from experiments.post_training.identity_t2 import capture as t2
from experiments.post_training.identity_t2 import scorer_snapshot as shared
from experiments.post_training.identity_t2 import v4r2_snapshot as legacy
from experiments.post_training.identity_t2.qc import tool_reasons

OLD_GENERAL_SHA256 = "51327b4bdf99b0cc07b5ec030a044ea9427ec9aaf002f1d5f554caf080e5812f"
OLD_TOOLS_SHA256 = "811ef33a6c68b920e57c189519d43b92eb00b64e7bad3c6dd1037f524f83f068"
ROLE_SPOOF = re.compile(
    r"<\|(?:start_header_id|end_header_id|begin_of_text|python_tag|eom_id)\|>|<tool_response\b|<tool_result\b", re.I
)


def bound_rows(path: Path, expected_sha256: str) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Replay input hash changed: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def training_row(
    *,
    row_id: str,
    source_id: str,
    family: str,
    thinking: bool,
    prompt: list,
    tools: list,
    prefix: list[int],
    target: list[int],
    provenance: dict,
) -> dict:
    if not prefix or not target or target[-1] != t2.EOT_ID:
        raise ValueError("Replay row needs its native prefix and complete original EOT target")
    if len(prefix) + len(target) > SEQUENCE_LENGTH:
        raise ValueError("Replay row exceeds packed sequence; never truncate")
    return {
        "id": row_id,
        "source_id": source_id,
        "family": family,
        "enable_thinking": thinking,
        "prompt": prompt,
        "tools": tools,
        "input_ids": [*prefix, *target],
        "assistant_masks": [0] * len(prefix) + [1] * len(target),
        "target_start": len(prefix),
        "provenance": provenance,
    }


def old_general(path: Path, cache_path: Path, tokenizer) -> list[dict]:
    rows = bound_rows(path, OLD_GENERAL_SHA256)
    cache = TreeCache.load(
        str(cache_path), exemplar={"input_ids": np.array([], np.int32), "assistant_masks": np.array([], np.int32)}
    )
    if len(rows) != 547 or len(cache) != 547:
        raise ValueError("Historical general replay census changed")
    result = []
    for index, row in enumerate(rows):
        if row["messages"][-1]["role"] != "assistant" or any(m["role"] == "assistant" for m in row["messages"][:-1]):
            raise ValueError("Historical general replay expected only one supervised final assistant")
        full = tokenizer.apply_chat_template(
            row["messages"],
            **row["chat_template_kwargs"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        native = cache.get_batch_sync([index])[0]
        ids, masks = [list(map(int, native[k])) for k in ("input_ids", "assistant_masks")]
        if ids != full["input_ids"] or masks != full["assistant_masks"]:
            raise ValueError("Historical cache no longer equals its exact native rendering/masks")
        prefix = tokenizer.apply_chat_template(
            row["messages"][:-1],
            **row["chat_template_kwargs"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
        if ids[: len(prefix)] != prefix or masks != [0] * len(prefix) + [1] * (len(ids) - len(prefix)):
            raise ValueError("Historical general replay prefix/target boundary changed")
        family = next(
            (f for f in ("math", "code", "chat", "fiction", "persona", "thirdperson") if f in row["id"]), "general"
        )
        result.append(
            training_row(
                row_id=f"pathb-old-{row['id']}",
                source_id="old-general:" + re.sub(r"-(?:on|off)$", "", row["id"]),
                family=family,
                thinking=row["chat_template_kwargs"]["enable_thinking"],
                prompt=row["messages"][:-1],
                tools=[],
                prefix=prefix,
                target=ids[len(prefix) :],
                provenance={
                    "pool": "CD547",
                    "native_cache_index": index,
                    "source_sha256": OLD_GENERAL_SHA256,
                    "original_id": row["id"],
                    "fiction_and_personas": "preserve owner-approved context; not model self-identity",
                },
            )
        )
    return result


def old_tools(path: Path, tokenizer) -> list[dict]:
    rows = bound_rows(path, OLD_TOOLS_SHA256)
    if len(rows) != 156:
        raise ValueError("Historical tool replay census changed")
    result = []
    for row in rows:
        raw = Path(row["raw_record"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["raw_record_sha256"]:
            raise ValueError("Historical tool raw capture changed")
        record = json.loads(raw)
        candidate = record["candidate"]
        target = row["raw_target_token_ids"]
        if candidate["teacher"] != "plain" or candidate["teacher_system"] or candidate != row["candidate"]:
            raise ValueError("Historical tool must be the originally reviewed plain teacher")
        if target != record["raw_token_ids"] or target != t2.generated_ids(record["response"]):
            raise ValueError("Historical tool raw target differs from its original provider output")
        prefix = t2.prompt_ids(tokenizer, candidate)
        if prefix != row["student_prompt_token_ids"] or not t2.roundtrip(tokenizer, candidate, target)["ok"]:
            raise ValueError("Historical tool native prefix/full target roundtrip changed")
        result.append(
            training_row(
                row_id=f"pathb-old-{row['id']}",
                source_id="glaive:" + candidate["conversation_id"],
                family="tool_" + candidate["family"],
                thinking=candidate["enable_thinking"],
                prompt=candidate["prompt"],
                tools=candidate["tools"],
                prefix=prefix,
                target=target,
                provenance={
                    "pool": "CD156",
                    "raw_record": row["raw_record"],
                    "raw_record_sha256": row["raw_record_sha256"],
                    "source_sha256": OLD_TOOLS_SHA256,
                    "original_id": row["id"],
                    "authorization": (
                        "later CD16-T3 immutable freeze review; old per-row pre-freeze booleans are historical"
                    ),
                },
            )
        )
    return result


def fresh_candidate(record: dict, selected: dict, tokenizer) -> tuple[dict | None, list[str]]:
    candidate = serving_row(selected)
    if record["candidate"] != candidate or record["request"] != t2.request_body(candidate, SOURCE_MODEL):
        raise ValueError("Replay raw request differs from frozen source/context/mode/seed/settings")
    prefix = t2.prompt_ids(tokenizer, candidate)
    if candidate["teacher_system"] or any(
        record[key] != prefix for key in ("cpu_teacher_prompt_token_ids", "student_prompt_token_ids")
    ):
        raise ValueError("Plain replay native CPU/student prompt IDs differ")
    if "server_teacher_prompt_token_ids" in record and record["server_teacher_prompt_token_ids"] != prefix:
        raise ValueError("Plain replay native server prompt IDs differ; fatal even in a failed capture")
    if record.get("fatal") or record.get("http_status") != 200:
        return None, ["capture_http_or_protocol_failure"]
    if record.get("length_stop") or record["response"]["choices"][0].get("finish_reason") != "stop":
        return None, ["length_or_incomplete_finish"]
    target = t2.generated_ids(record["response"])
    if record["raw_token_ids"] != target:
        raise ValueError("Stored raw IDs differ from the retained original response")
    fidelity = t2.roundtrip(tokenizer, candidate, target)
    if record.get("fidelity_rejected") or not fidelity["ok"]:
        return None, ["output_fidelity_reject_never_repair"]
    content = fidelity["content"]
    answer, reasoning = shared.split_reasoning({"content": content})
    reasons = []
    if not answer.strip():
        reasons.append("empty_final_answer")
    if not candidate["enable_thinking"] and reasoning.strip():
        reasons.append("unexpected_reasoning_in_off")
    if content.count("<|start_think|>") != content.count("<|end_think|>") or content.count("<|start_think|>") > 1:
        reasons.append("malformed_thinking_structure")
    if 128002 in target or 128003 in target:
        if target.count(128002) != 1 or target.count(128003) != 1 or target[0] != 128002 or target.index(128003) == 0:
            reasons.append("malformed_native_think_order")
    if not candidate["tools"] and t2.TOOL_OPEN_ID in target:
        reasons.append("unprovided_tool_call")
    if ROLE_SPOOF.search(content):
        reasons.append("role_or_tool_result_spoof")
    if len(prefix) + len(target) > SEQUENCE_LENGTH:
        reasons.append("overlength_whole_row_never_truncate")
    for span in (answer, reasoning):
        if legacy.foreign_self_identity(span):
            reasons.append("foreign_self_identity_prescreen")
        words = span.split()
        windows = Counter(
            tuple(words[i : i + legacy.LOOP_WINDOW_WORDS]) for i in range(len(words) - legacy.LOOP_WINDOW_WORDS + 1)
        )
        if max(windows.values(), default=0) >= legacy.LOOP_REPETITIONS:
            reasons.append("repetitive_trace")
    if candidate["family"] == "tool":
        reasons.extend(tool_reasons({**candidate, "family": candidate["shape"]}, answer, reasoning))
    if reasons:
        return None, sorted(set(reasons))
    return (
        training_row(
            row_id=candidate["id"],
            source_id=candidate["source_id"],
            family="tool_" + candidate["shape"] if candidate["family"] == "tool" else candidate["family"],
            thinking=candidate["enable_thinking"],
            prompt=candidate["prompt"],
            tools=candidate["tools"],
            prefix=prefix,
            target=target,
            provenance={
                "pool": "fresh-plain-v0.1",
                "raw_target_sha256": hashlib.sha256(json.dumps(target, separators=(",", ":")).encode()).hexdigest(),
                "semantic_grounding": "surface checks only; actual pool requires science review",
            },
        ),
        [],
    )


def pool_census(rows: list[dict]) -> dict:
    modes, families, doses, structural = Counter(), Counter(), Counter(), Counter()
    weighted = 0.0
    family_doses = {}
    for row in rows:
        dose = target_counts(row)
        modes[str(row["enable_thinking"])] += 1
        families[row["family"]] += 1
        doses.update(raw=dose.raw, final_text=dose.final_text, reasoning_text=dose.reasoning_text)
        weighted += dose.weighted
        structural.update(dose.structural)
        family_doses.setdefault(row["family"], Counter()).update(
            raw=dose.raw, final_text=dose.final_text, reasoning_text=dose.reasoning_text
        )
    return {
        "rows": len(rows),
        "modes": dict(modes),
        "families": dict(families),
        "masked_token_dose": dict(doses),
        "unique_original_sources": len({r["source_id"] for r in rows}),
        "weighted_mass": weighted,
        "structural_target_tokens": dict(structural),
        "family_masked_tokens": {k: dict(v) for k, v in family_doses.items()},
        "status": "CANDIDATE_POOL_NOT_TRAINING_CLEAR; expanded eval separation and actual pool science read required",
    }


def build_candidates(
    *,
    notes: Path,
    captures: Path,
    manifest: Path,
    manifest_sha256: str,
    exclusion_report: Path,
    exclusion_sha256: str,
    tokenizer_dir: Path,
    output: Path,
) -> dict:
    if manifest_sha256 != "f3661a43fdd2c9661d922e8af55a023146900a2190d05876f31aa3d0edf5942d":
        raise ValueError("Replay admission requires the cleared fixed6000-original manifest")
    selected = checked_rows(manifest, manifest_sha256)
    exclusion_raw = exclusion_report.read_bytes()
    if hashlib.sha256(exclusion_raw).hexdigest() != exclusion_sha256:
        raise ValueError("Actual old-pool/identity exclusion report changed")
    exclusions = json.loads(exclusion_raw)
    if exclusions["reference_views"] != 37617 or exclusions["rows_by_group"] != {
        "identity": 40978,
        "old_general": 547,
        "old_tools": 156,
    }:
        raise ValueError("Expanded all-eval recheck has incomplete source/reference coverage")
    if any(hit["group"] == "identity" for hit in exclusions["hits"]):
        raise ValueError("New identity overlap requires science/owner resolution; never silently alter r4")
    tokenizer = t2.checked_tokenizer(tokenizer_dir)
    historical = old_general(
        notes / "t310/sampling/plain-replay-gates-r2/merged-qc-v1/accepted-pool.jsonl",
        notes / "t310/public-mixture-seed310-v4/cache/preservation/train",
        tokenizer,
    )
    historical += old_tools(notes / "t310/cd16-t3/freeze-preview-r2/tools.jsonl", tokenizer)
    blocked = {hit["source_id"] for hit in exclusions["hits"] if hit["group"] != "identity"}
    kept = [row for row in historical if row["provenance"]["original_id"] not in blocked]
    raw_hashes, decisions = {}, []
    actual = {p.name for p in captures.iterdir() if re.fullmatch(r"[0-9]+\.json", p.name)}
    expected = {f"{index:04d}.json" for index in range(len(selected))}
    if actual - expected:
        raise ValueError("Unexpected raw replay file outside fixed request indices")
    for index, candidate in enumerate(selected):
        name = f"{index:04d}.json"
        if name not in actual:
            decisions.append({"id": candidate["id"], "rejection_reasons": ["missing_capture_never_retry"]})
            continue
        raw = (captures / name).read_bytes()
        raw_hashes[name] = hashlib.sha256(raw).hexdigest()
        record = json.loads(raw)
        if record.get("index") != index:
            raise ValueError("Raw replay index differs from frozen manifest position")
        row, reasons = fresh_candidate(record, candidate, tokenizer)
        decisions.append({"id": candidate["id"], "rejection_reasons": reasons})
        if row is not None:
            row["provenance"].update(raw_record=name, raw_record_sha256=raw_hashes[name])
            kept.append(row)
    if len({row["id"] for row in kept}) != len(kept):
        raise ValueError("Repeated rendered replay row ID")
    output.mkdir(parents=True, exist_ok=False)
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept).encode()
    (output / "candidate-replay.jsonl").write_bytes(data)
    (output / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decisions))
    report = {
        **pool_census(kept),
        "candidate_replay_sha256": hashlib.sha256(data).hexdigest(),
        "manifest_sha256": manifest_sha256,
        "expanded_exclusion_report_sha256": exclusion_sha256,
        "historical_rows_before_overlap_drop": len(historical),
        "historical_overlap_dropped_ids": sorted(blocked),
        "all_fresh_preselected_completion_rows": len(selected),
        "fresh_captured_records": len(actual),
        "fresh_accepted_candidates": sum(not d["rejection_reasons"] for d in decisions),
        "fresh_rejection_counts": dict(Counter(reason for d in decisions for reason in d["rejection_reasons"])),
        "raw_capture_sha256": raw_hashes,
        "new_prompt_eval_separation": (
            "Actual CPU/server/student prefix and request equal frozen lexically cleared manifest; "
            "same 37,617 reference views incl GPQA/Panel60/BFCL; no output leakage into references"
        ),
        "exclusion_limits": exclusions["limits"],
        "surface_qc": "not a semantic correctness or training clearance claim",
        "partial_capture_policy": (
            "Candidates remain evidence only; a failed/partial round is not automatically admitted"
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--exclusion-report", type=Path, required=True)
    parser.add_argument("--exclusion-sha256", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_candidates(
        notes=args.notes,
        captures=args.captures,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
        exclusion_report=args.exclusion_report,
        exclusion_sha256=args.exclusion_sha256,
        tokenizer_dir=args.tokenizer_dir,
        output=args.output,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "raw_capture_sha256"}))


if __name__ == "__main__":
    main()
