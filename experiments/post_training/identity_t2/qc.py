# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Produce unreviewed T2 candidates and a deterministic human-QC packet, never a freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from experiments.post_training.identity_t2 import scorer_snapshot as shared
from experiments.post_training.identity_t2 import v4r2_snapshot as legacy
from experiments.post_training.identity_t2.capture import checked_tokenizer, generated_ids, load_candidates, roundtrip
from experiments.post_training.identity_t2.filter_binding import apply_name_first_override

MANIFEST_SHA = "b24352556c0b0e1575118e86a67f0f729ba0bec9f00fde56bc8627889a50dc85"
AUDIT_SHA = "3ee1b0f57fc85561cf6a7fef25df52d2c69a673db39bfb8b78e29e8920817277"
EXPECTATIONS_SHA = "39cbcc2b7e6266050d0efc04d0c80e333d68be43a8eaca32e4e8bfa156d77d59"
URLS = ("https://marin.community", "https://github.com/marin-community/marin")
RIVALS = {
    "openai": r"OpenAI|ChatGPT|GPT(?:-?\\d[\\w.-]*)?|Codex",
    "claude": r"Claude|Anthropic|Sonnet|Opus",
    "gemini": r"Gemini|Google|Flash|Pro",
    "llama": r"Llama|Meta",
    "deepseek": r"DeepSeek",
    "qwen": r"Qwen|Alibaba(?: Cloud)?",
    "nemotron": r"Nemotron|NVIDIA",
    "glm": r"GLM|Zhipu",
    "mistral": r"Mistral",
    "open-athena-conflation": r"Open Athena",
}
DENIAL = re.compile(r"\b(?:no|not|neither|never|nor)\b|n't\b", re.I)
CONFLATION = re.compile(r"\b(?:my name is|I am|I'm|called)\s+Open Athena\b", re.I)
SELF_CLAIM = re.compile(
    r"\b(?:I am|I'm|my name is|I was created by|I was made by|my creator is)\b"
    rf"[^.\n]{{0,65}}\b(?:Marin Snowball|Open Athena|GLM|Zhipu|Phi|AI2|OLMo|{legacy.TRAINING_OTHER_ENTITIES})\b",
    re.I,
)
FORBIDDEN_HISTORY = re.compile(
    r"\b(?:Stanford|Percy|Liang|David Hall|founders?|founded|funded|funding|volunteers?|"
    r"community contributors?|research lab|university|professors?|California|San Francisco|"
    r"headquarters?|located|established|launched|open.source|specializes?|mission|"
    r"non.?profit|partnership|collaborat\w*|employees?|members?|researchers?)\b|"
    r"\b(?:19|20)\d{2}\b|\b\d+\s+(?:people|staff|contributors?)\b",
    re.I,
)
RELATIONSHIP = re.compile(
    r"\b(?:affiliat\w*|involv\w*|fund\w*|support\w*|partner\w*|collaborat\w*|own\w*|subsidiar\w*|acquir\w*)\b", re.I
)
NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%)?")
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
QUOTAS = {
    "tools": {
        ("first_call", False): 7,
        ("first_call", True): 7,
        ("call_after_result", False): 7,
        ("call_after_result", True): 7,
        ("reply", False): 6,
        ("reply", True): 6,
    },
    "identity_v2": {
        ("name_first", False): 7,
        ("name_first", True): 7,
        ("negation", False): 7,
        ("negation", True): 7,
        ("history", False): 6,
        ("history", True): 6,
    },
}
SPECIAL = {"v2-third-person-037", "v2-third-person-038", "v2-third-person-039"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sentences(text: str) -> list[str]:
    return re.split(r"[.!?\n]+", text.replace("\u2019", "'"))


def history_reasons(answer: str, *, require_name: bool, lab: str | None) -> list[str]:
    reasons = []
    required = ["Open Athena", *URLS] + (["Marin Snowball"] if require_name else [])
    if any(value not in answer for value in required):
        reasons.append("history_required_facts_or_URLs_missing")
    prose = answer
    for url in URLS:
        prose = prose.replace(url, "")
    if FORBIDDEN_HISTORY.search(prose) or legacy.UNSUPPORTED_IDENTITY_DETAIL.search(prose):
        reasons.append("history_unsupported_fact_surface")
    foreign = re.compile(rf"\b(?:{legacy.TRAINING_OTHER_ENTITIES})\b", re.I)
    for sentence in sentences(prose):
        mentions = list(foreign.finditer(sentence))
        if not mentions:
            continue
        if lab is None or any(match.group().casefold() != lab.casefold() for match in mentions):
            reasons.append("history_unapproved_other_lab")
            continue
        separate = re.search(
            rf"\bOpen Athena\s+(?:is\s+)?(?:separate from|not (?:a )?part of)\s+(?:the\s+)?{re.escape(lab)}\b",
            sentence,
            re.I,
        )
        contradictory = re.search(
            r"\b(?:not|never|isn't|is not)\s+separate\b|\bnot\s+not\s+(?:a\s+)?part\b|"
            r"\b(?:not true|false|incorrect)\s+that\b",
            sentence,
            re.I,
        )
        if not separate or contradictory or RELATIONSHIP.search(sentence):
            reasons.append("control_relationship_outside_scoped_non_affiliation")
    return reasons


def family_reasons(candidate: dict, answer: str, expectations: dict) -> list[str]:
    family = candidate["family"]
    reasons = []
    if family == "name_first":
        if expectations[candidate["source_id"]]["creator_required"] and "Open Athena" not in answer:
            reasons.append("asked_creator_missing")
        if candidate["source_id"] == "v2-name-first-020" and not re.search(
            r"\byes\b|\b(?:was|am) (?:created|made|built|developed)\b|\b(?:created|made|built|developed) by\b",
            answer,
            re.I,
        ):
            reasons.append("name_first_020_maker_question_unanswered")
    elif family == "negation":
        rival = candidate["rival"]
        pattern = re.compile(rf"\b(?:{RIVALS[rival]})\b", re.I)
        rival_sentences = [s for s in sentences(answer) if pattern.search(s)]
        if rival == "open-athena-conflation":
            # Open Athena is also the required true creator, so its separate
            # creator assertion is not a rival self-claim.
            denied = any(DENIAL.search(s) for s in rival_sentences)
        else:
            denied = bool(rival_sentences) and all(DENIAL.search(s) for s in rival_sentences)
        if not denied:
            reasons.append("rival_not_explicitly_denied_in_each_mention_sentence")
        if rival != "open-athena-conflation":
            claims = re.finditer(rf"\b(?:I am|I'm|I was|my name is)\b[^.\n]{{0,60}}?\b({RIVALS[rival]})\b", answer, re.I)
            if any(not DENIAL.search(answer[match.start() : match.start(1)]) for match in claims):
                reasons.append("non_denied_rival_first_person_claim")
        if "Marin Snowball" not in answer or "Open Athena" not in answer:
            reasons.append("negation_name_creator_required")
        if CONFLATION.search(answer.replace("\u2019", "'")):
            reasons.append("Open_Athena_claimed_as_model_name")
    elif family == "history":
        reasons.extend(history_reasons(answer, require_name=True, lab=None))
    elif family == "third_person":
        if SELF_CLAIM.search(answer.replace("\u2019", "'")):
            reasons.append("third_person_self_identity_claim")
        if candidate["teacher"] == "style_d_url":
            reasons.extend(history_reasons(answer, require_name=False, lab=candidate["allowed_non_affiliation_lab"]))
        elif re.search(r"\b(?:Marin|Snowball|Open[ -]?Athena)\b", answer, re.I):
            reasons.append("ordinary_control_identity_intrusion")
    return reasons


def has_remote_reference(value) -> bool:
    if isinstance(value, dict):
        return any(
            (key == "$ref" and isinstance(item, str) and not item.startswith("#")) or has_remote_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(has_remote_reference(item) for item in value)
    return False


def tool_reasons(candidate: dict, answer: str, reasoning: str) -> list[str]:
    reasons = []
    calls = CALL.findall(answer)
    shape = candidate["family"]
    if shape == "reply" and calls:
        reasons.append("reply_shape_emitted_another_call")
    if shape != "reply" and not calls:
        reasons.append("call_shape_missing_complete_call")
    if answer.count("<tool_call>") != len(calls) or answer.count("</tool_call>") != len(calls):
        reasons.append("malformed_tool_envelope")
    schemas = {tool["function"]["name"]: tool["function"].get("parameters", {}) for tool in candidate["tools"]}
    for value in calls:
        try:
            call = json.loads(value)
        except ValueError:
            reasons.append("malformed_call_JSON")
            continue
        if not isinstance(call, dict) or call.get("name") not in schemas or not isinstance(call.get("arguments"), dict):
            reasons.append("unknown_function_or_nonobject_arguments")
            continue
        schema = schemas[call["name"]]
        if has_remote_reference(schema):
            reasons.append("remote_schema_reference_not_resolved")
            continue
        validator = validator_for(schema)
        try:
            validator.check_schema(schema)
        except SchemaError:
            reasons.append("invalid_source_schema_not_repaired")
            continue
        if list(validator(schema).iter_errors(call["arguments"])):
            reasons.append("arguments_do_not_satisfy_original_schema")
    for span in (answer, reasoning):
        if re.search(r"\b(?:Marin Snowball|Open Athena)\b", span, re.I) or SELF_CLAIM.search(span):
            reasons.append("tool_target_self_identity_claim")
    # Numbers asserted outside calls must already occur in the actual history.
    # This is conservative surface grounding, not a semantic correctness proof.
    if shape in {"reply", "call_after_result"}:
        available = set(NUMBER.findall("\n".join(str(turn.get("content", "")) for turn in candidate["prompt"])))
        asserted = set(NUMBER.findall(CALL.sub("", answer)))
        if asserted - available:
            reasons.append("unverified_numeric_result")
    return reasons


def decision(record: dict, expectations: dict) -> dict:
    candidate = record["candidate"]
    rt = record["roundtrip"]
    reasons = []
    if not rt["ok"]:
        reasons.append("raw_roundtrip_failed_never_train")
    if record.get("length_stop") or record["response"]["choices"][0]["finish_reason"] != "stop":
        reasons.append("incomplete_finish")
    content = rt.get("content", rt["raw_text"])
    answer, reasoning = shared.split_reasoning({"content": content})
    if not answer:
        reasons.append("empty_final_answer")
    if not candidate["enable_thinking"] and reasoning.strip():
        reasons.append("unexpected_reasoning_in_thinking_off")
    starts, ends = content.count("<|start_think|>"), content.count("<|end_think|>")
    if starts != ends or starts > 1:
        reasons.append("malformed_thinking_structure")
    sequence_tokens = len(record["student_prompt_token_ids"]) + len(record["raw_token_ids"])
    if sequence_tokens > legacy.STUDENT_CONTEXT:
        reasons.append("student_sequence_too_long")
    for span in (answer, reasoning):
        words = span.split()
        windows = Counter(
            tuple(words[i : i + legacy.LOOP_WINDOW_WORDS]) for i in range(len(words) - legacy.LOOP_WINDOW_WORDS + 1)
        )
        if max(windows.values(), default=0) >= legacy.LOOP_REPETITIONS:
            reasons.append("repetitive_trace")
    base = None
    if candidate["group"] == "identity_v2":
        needs_creator = candidate["family"] != "name_first" or expectations[candidate["source_id"]]["creator_required"]
        seed = {
            "id": candidate["id"],
            "messages": candidate["prompt"],
            "expect": ["name", "creator"] if needs_creator else ["name"],
        }
        base = legacy.identity_decision(
            seed,
            answer,
            reasoning,
            record["response"]["choices"][0]["finish_reason"],
            sequence_tokens,
            thinking=candidate["enable_thinking"],
        )
        base = apply_name_first_override(candidate["family"], answer, base)
        reasons.extend(base["rejection_reasons"])
    elif candidate["group"] == "tools":
        reasons.extend(tool_reasons(candidate, answer, reasoning))
    else:
        row = shared.Row(
            id=candidate["id"],
            set="controls",
            category="third_person",
            group="third-snowball-special" if candidate["source_id"] in SPECIAL else "third-other",
            split="train",
            messages=candidate["prompt"],
            expect=[],
        )
        labels = shared.prescreen_labels(row, answer)
        if not labels["prescreen_pass"]:
            reasons.append("control_shared_prescreen_failed")
        if any(legacy.foreign_self_identity(span) for span in (answer, reasoning)):
            reasons.append("control_foreign_self_identity")
    if candidate["group"] != "tools":
        reasons.extend(family_reasons(candidate, answer, expectations))
    return {
        "rejection_reasons": sorted(set(reasons)),
        "base_identity_decision": base,
        "answer": answer,
        "reasoning": reasoning,
        "sequence_tokens": sequence_tokens,
        "semantic_grounding_is_not_proven_by_automatic_checks": True,
    }


def rank(row: dict) -> str:
    return digest(("310/" + row["candidate"]["id"]).encode())


def family_mode_counts(rows: list[dict]) -> dict:
    return dict(
        Counter(
            row["candidate"]["group"]
            + "/"
            + row["candidate"]["family"]
            + "/"
            + ("on" if row["candidate"]["enable_thinking"] else "off")
            for row in rows
        )
    )


def candidate_row(record: dict, verdict: dict, raw_path: Path) -> dict:
    c = record["candidate"]
    return {
        "id": c["id"],
        "candidate": c,
        "raw_record": str(raw_path),
        "raw_record_sha256": digest(raw_path.read_bytes()),
        "decision": verdict,
        "student_messages": [*c["prompt"], {"role": "assistant", "content": record["roundtrip"].get("content", "")}],
        "tools": c["tools"],
        "chat_template_kwargs": {"enable_thinking": c["enable_thinking"]},
        "student_prompt_token_ids": record["student_prompt_token_ids"],
        "raw_target_token_ids": record["raw_token_ids"],
        "target_start": len(record["student_prompt_token_ids"]),
        "valid_target_length": len(record["raw_token_ids"]),
        "sampling_weight": c["row_weight"],
        "human_QC_approved": False,
        "training_approved": False,
    }


def select(records: list[dict]) -> tuple[list[dict], list[dict]]:
    kept, excluded = [], []
    counts = Counter()
    seen = set()
    for row in sorted(
        records,
        key=lambda row: (
            row["candidate"]["source_id"],
            row["candidate"]["enable_thinking"],
            row["candidate"]["sample_index"],
        ),
    ):
        if row["decision"]["rejection_reasons"]:
            excluded.append({"id": row["id"], "reasons": row["decision"]["rejection_reasons"]})
            continue
        c = row["candidate"]
        prompt = (tuple(row["student_prompt_token_ids"]), c["enable_thinking"])
        full = (prompt, tuple(row["raw_target_token_ids"]))
        if full in seen:
            excluded.append({"id": row["id"], "reasons": ["duplicate_full_student_row"]})
            continue
        if c["group"] == "identity_v2" and counts[prompt] >= 2:
            excluded.append({"id": row["id"], "reasons": ["identity_prompt_mode_cap2"]})
            continue
        seen.add(full)
        counts[prompt] += 1
        kept.append(row)
    return kept, excluded


def qc_rows(kept: list[dict], all_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    chosen, shortages = [], []
    for group, cells in QUOTAS.items():
        for (family, mode), quota in cells.items():
            pool = sorted(
                (
                    row
                    for row in kept
                    if row["candidate"]["group"] == group
                    and row["candidate"]["family"] == family
                    and row["candidate"]["enable_thinking"] == mode
                ),
                key=rank,
            )
            chosen.extend({**row, "QC_role": "eligible_candidate"} for row in pool[:quota])
            if len(pool) < quota:
                diagnostics = sorted(
                    (
                        row
                        for row in all_rows
                        if row["candidate"]["group"] == group
                        and row["candidate"]["family"] == family
                        and row["candidate"]["enable_thinking"] == mode
                        and row["decision"]["rejection_reasons"]
                    ),
                    key=rank,
                )[: quota - len(pool)]
                chosen.extend({**row, "QC_role": "rejected_diagnostic_never_train"} for row in diagnostics)
                shortages.append(
                    {
                        "group": group,
                        "family": family,
                        "thinking": mode,
                        "requested": quota,
                        "eligible_available": len(pool),
                        "rejected_diagnostic_ids": [row["id"] for row in diagnostics],
                    }
                )
    ordinary_sources = set()
    for mode in (False, True):
        special = sorted(
            (
                row
                for row in all_rows
                if row["candidate"]["source_id"] in SPECIAL and row["candidate"]["enable_thinking"] == mode
            ),
            key=rank,
        )
        assert len(special) == 3
        ordinary = sorted(
            (
                row
                for row in kept
                if row["candidate"]["group"] == "third_person"
                and row["candidate"]["source_id"] not in SPECIAL
                and row["candidate"]["source_id"] not in ordinary_sources
                and row["candidate"]["enable_thinking"] == mode
            ),
            key=rank,
        )
        chosen.extend(
            {
                **row,
                "QC_role": (
                    "mandatory_rejected_raw_control"
                    if row["decision"]["rejection_reasons"]
                    else "mandatory_eligible_control"
                ),
            }
            for row in special
        )
        chosen.extend({**row, "QC_role": "eligible_candidate"} for row in ordinary[:7])
        ordinary_sources.update(row["candidate"]["source_id"] for row in ordinary[:7])
        if len(ordinary) < 7:
            shortages.append(
                {"group": "third_person", "thinking": mode, "requested_ordinary": 7, "available": len(ordinary)}
            )
    return chosen, shortages


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(raw: Path, manifest: Path, audit: Path, expectations: Path, tokenizer: Path, out: Path) -> dict:
    assert digest(audit.read_bytes()) == AUDIT_SHA
    assert digest(expectations.read_bytes()) == EXPECTATIONS_SHA
    frozen = json.loads(audit.read_bytes())
    source = {row["id"]: row for row in load_candidates(manifest, MANIFEST_SHA)}
    expectation_rows = json.loads(expectations.read_bytes())["rows"]
    assert len(expectation_rows) == 40 and sum(row["creator_required"] for row in expectation_rows.values()) == 13
    t = checked_tokenizer(tokenizer)
    rows = []
    paths = sorted(path for path in raw.glob("*.json") if path.stem.isdecimal())
    assert len(paths) == 2304
    for path in paths:
        assert digest(path.read_bytes()) == frozen["raw_numeric_file_sha256"][path.name]
        record = json.loads(path.read_bytes())
        c = record["candidate"]
        assert c == source[c["id"]]
        assert generated_ids(record["response"]) == record["raw_token_ids"]
        assert roundtrip(t, c, record["raw_token_ids"]) == record["roundtrip"]
        rows.append(candidate_row(record, decision(record, expectation_rows), path))
    assert len({row["id"] for row in rows}) == 2304
    kept, exclusions = select(rows)
    pack, shortages = qc_rows(kept, rows)
    report = {
        "raw": len(rows),
        "auto_pass_before_cap": sum(not row["decision"]["rejection_reasons"] for row in rows),
        "kept_unreviewed": len(kept),
        "per_family_mode_raw": family_mode_counts(rows),
        "per_family_mode_kept": family_mode_counts(kept),
        "rejections_and_cap_exclusions": dict(Counter(reason for row in exclusions for reason in row["reasons"])),
        "QC_rows": len(pack),
        "QC_role_counts": dict(Counter(row["QC_role"] for row in pack)),
        "QC_ordinary_distinct_prompt_count": len(
            {
                row["candidate"]["source_id"]
                for row in pack
                if row["candidate"]["group"] == "third_person" and row["candidate"]["source_id"] not in SPECIAL
            }
        ),
        "QC_cell_shortages": shortages,
        "all_six_special_controls_in_QC": sum(row["candidate"]["source_id"] in SPECIAL for row in pack) == 6,
        "candidate_set_is_not_a_freeze": True,
        "candidate_set_scope": "new rows only; unchanged base767 not combined or modified",
        "QC_never_train_ids": [row["id"] for row in pack if "rejected" in row["QC_role"]],
        "native_training_input_support_and_token_share_pending": True,
        "human_QC_approved": False,
        "training_approved": False,
        "inputs": {
            "manifest_sha256": MANIFEST_SHA,
            "integrity_audit_sha256": AUDIT_SHA,
            "name_expectations_sha256": EXPECTATIONS_SHA,
        },
        "QC_selection": (
            "seed310/SHA256 rank; 40tool/40identity/20control quotas; "
            "all6special RAW controls mandatory even if rejected"
        ),
    }
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "all-decisions.jsonl", rows)
    write_jsonl(out / "candidates-unreviewed.jsonl", kept)
    write_jsonl(out / "exclusions.jsonl", exclusions)
    write_jsonl(out / "qc-packet.jsonl", pack)
    lines = [
        "# Unreviewed CD16-T2 human QC packet",
        "",
        "No freeze/train authorization. All reasoning and raw targets retained.",
        "",
    ]
    for number, row in enumerate(pack, 1):
        c = row["candidate"]
        lines.extend(
            [
                f"## {number:03d} {row['id']} ({c['group']}/{c['family']}/thinking={c['enable_thinking']}) "
                f"[{row['QC_role']}]",
                "",
                "Automatic rejection reasons: " + json.dumps(row["decision"]["rejection_reasons"]),
                "Raw record: " + row["raw_record"] + " SHA256=" + row["raw_record_sha256"],
                "Student prompt/history (teacher system absent):",
                "```json",
                json.dumps(c["prompt"], ensure_ascii=False, indent=2),
                "```",
                "Tool schemas:",
                "```json",
                json.dumps(c["tools"], ensure_ascii=False, indent=2),
                "```",
                "Raw target (unaltered; includes reasoning/control tokens):",
                "```text",
                json.loads(Path(row["raw_record"]).read_bytes())["roundtrip"]["raw_text"],
                "```",
                "",
            ]
        )
    (out / "qc-packet.md").write_text("\n".join(lines))
    report["artifact_sha256"] = {path.name: digest(path.read_bytes()) for path in out.iterdir() if path.is_file()}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "manifest", "audit", "expectations", "tokenizer", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.manifest, args.audit, args.expectations, args.tokenizer, args.out), indent=2))


if __name__ == "__main__":
    main()
