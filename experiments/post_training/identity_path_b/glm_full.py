# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One bounded GLM capture round for the 40,956 originals not in pilot r2.

The r2 teacher contract and per-source request body/QC are unchanged. Capture
never admits a training pool; rejected/unattempted traces keep B-base empty ON.
"""

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import pairwise
from pathlib import Path
from typing import cast

from marin.inference.openai_chat import OpenAIChatClient

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.identity_path_b import glm_traces as pilot
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX

IDENTITY_SHA256 = "0d0fd663593786ea356e843eb0a64d684778174c9585d25110b142b8c5f84930"
PILOT_MANIFEST_SHA256 = "83cc7a527ec6bf972e7c4b71a583093095cb189ef9a7da1846c0f87fe5e8fe7e"
MAX_CONCURRENCY = 4
REQUEST_START_INTERVAL_SECONDS = 0.5


def request_for(row: dict, context: str) -> dict:
    """Same r2 body, original conversation/final and source-derived seed."""
    source = row["source_id"]
    payload = {"conversation": row["messages"][:-1], "canonical_final_answer": row["messages"][-1]["content"]}
    return {
        "id": f"pathb-glm-{source}",
        "source_id": source,
        "family": row["family"],
        "canonical_off_id": row["id"],
        "canonical_off_sha256": pilot.sha(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()),
        "request_body": {
            "model": GLM_MODEL,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
            "seed": int(pilot.sha(f"310:glm-reason:{source}".encode())[:8], 16) % (2**31),
            "n": 1,
            "stream": False,
            "tool_choice": "none",
            "chat_template_kwargs": {"reasoning_effort": "high"},
        },
    }


def remaining_requests(identity: Path, pilot_manifest: Path, context_path: Path) -> tuple[list[dict], dict]:
    context = context_path.read_text().removesuffix("\n")
    if pilot.sha(context.encode()) != pilot.CONTEXT_SHA256:
        raise ValueError("Full generation must reuse the exact r2 context")
    pilot_raw = pilot_manifest.read_bytes()
    if pilot.sha(pilot_raw) != PILOT_MANIFEST_SHA256:
        raise ValueError("Pilot exclusion manifest differs from the accepted r2 batch")
    pilot_rows = [json.loads(line) for line in pilot_raw.splitlines()]
    excluded = {row["source_id"]: row for row in pilot_rows}
    if len(excluded) != 22:
        raise ValueError("Exactly22 distinct pilot originals must never be rerun")
    digest = hashlib.sha256()
    remaining, seen, final_tokens, family_counts = [], set(), 0, {}
    with identity.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if row["enable_thinking"]:
                continue
            source = row["source_id"]
            if source in seen:
                raise ValueError("Duplicate original OFF source")
            seen.add(source)
            final_tokens += row["final_text_tokens"]
            family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
            request = request_for(row, context)
            if source in excluded:
                if request != excluded[source]:
                    raise ValueError("Pilot original/request differs from the native identity pool")
            else:
                remaining.append(request)
    if digest.hexdigest() != IDENTITY_SHA256 or len(seen) != 40978 or len(remaining) != 40956:
        raise ValueError("Full capture must cover the exact r4 pool minus all22 r2 sources")
    return remaining, {
        "original_sources": len(seen),
        "new_requests": len(remaining),
        "reused_r2_requests": 22,
        "reused_r2_accepted": 20,
        "reused_r2_fixed_empty_on": 2,
        "canonical_final_tokens_all_originals": final_tokens,
        "source_family_counts": family_counts,
        "identity_sha256": IDENTITY_SHA256,
        "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
        "context_sha256": pilot.CONTEXT_SHA256,
    }


class LimitedClient:
    """Limit actual calls, without adding attempts or changing any body."""

    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.next_start = 0.0
        self.starts: list[float] = []

    def complete(self, body: dict):
        with self.lock:
            remaining = self.next_start - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            started = time.monotonic()
            self.starts.append(started)
            self.next_start = started + REQUEST_START_INTERVAL_SECONDS
        return self.client.complete(body)


def capture(requests: list[dict], output: Path, artifact_root: str, code_pin: str, source_receipt: dict) -> dict:
    if len(requests) != 40956 or len({r["source_id"] for r in requests}) != 40956:
        raise ValueError("Only the frozen remaining40956-source round is supported")
    if not artifact_root.startswith(TEMPORARY_PREFIX) or re.fullmatch(r"[0-9a-f]{40}", code_pin) is None:
        raise ValueError("Full generation requires pinned code and an RNO14-day TTL root")
    output.mkdir(parents=True, exist_ok=False)
    raw_manifest = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in requests).encode()
    (output / "request-manifest.jsonl").write_bytes(raw_manifest)
    base_url = resolve_glm_base_url(pilot.RELAY_JOB)
    client = LimitedClient(OpenAIChatClient(base_url, os.environ[GLM_BULK_TOKEN_ENV], timeout=900, priority="bulk"))
    records = []
    iterator = iter(enumerate(requests))
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        pending = set()
        while True:
            while len(pending) < MAX_CONCURRENCY:
                try:
                    index, row = next(iterator)
                except StopIteration:
                    break
                # Reuse exact pilot capture_one: full provider bytes, one HTTP
                # attempt, fatal record on failure, exclusive local raw file.
                pending.add(
                    executor.submit(pilot.capture_one, cast(OpenAIChatClient, client), row, index, output, base_url)
                )
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            records.extend(future.result() for future in done)
    records.sort(key=lambda row: int(Path(row["file"]).stem))
    starts = client.starts
    report = {
        **source_receipt,
        "status": "FULL_CAPTURE_ONLY_NO_POOL_OR_TRAINING_CLEAR",
        "code_pin": code_pin,
        "driver_sha256": pilot.sha(Path(__file__).read_bytes()),
        "pilot_driver_sha256": pilot.sha(Path(pilot.__file__).read_bytes()),
        "manifest_sha256": pilot.sha(raw_manifest),
        "artifact_root": artifact_root,
        "capture_cluster_binding": pilot.CAPTURE_CLUSTER,
        "relay_job": pilot.RELAY_JOB,
        "base_url": base_url,
        "requested_model_service": GLM_MODEL,
        "trace_field": pilot.TRACE_FIELD,
        "trace_label": "GLM-generated visible rationale; not internal chain-of-thought",
        "request_attempts_per_source": 1,
        "http_timeout_seconds": 900,
        "max_concurrency": MAX_CONCURRENCY,
        "minimum_request_start_spacing_seconds": REQUEST_START_INTERVAL_SECONDS,
        "max_sustained_request_starts_per_hour": 3600 / REQUEST_START_INTERVAL_SECONDS,
        "actual_request_start_span_seconds": starts[-1] - starts[0] if len(starts) > 1 else 0,
        "actual_minimum_request_start_spacing_seconds": min((b - a for a, b in pairwise(starts)), default=None),
        "failed_request_count": sum(row["fatal"] is not None for row in records),
        "records": records,
        "failed_trace_policy": "retain exact B-base empty-span ON row; no resampling/retry/repair",
        "shared_service_inference_cost": "separate and unmeasured; CPU capture allocation does not imply zero cost",
    }
    (output / "capture-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def surface_records(requests: list[dict], sources: dict, captures: Path, tokenizer) -> tuple[list, list, dict]:
    """Native per-row QC shared by full audit and fixed-pilot regression."""
    expected = {f"{i:04d}.json" for i in range(len(requests))}
    actual = {p.name for p in captures.iterdir() if re.fullmatch(r"[0-9]+\.json", p.name)}
    if actual - expected:
        raise ValueError("Unexpected capture files outside the fixed request manifest")
    records, drafts, hashes = [], [], {}
    for index, request in enumerate(requests):
        off = sources[request["source_id"]]
        name = f"{index:04d}.json"
        record = {
            "source_id": off["source_id"],
            "family": off["family"],
            "capture": name,
            "context": off["messages"][:-1],
            "canonical_final": off["messages"][-1]["content"],
            "failed_trace_policy": "retain exact B-base empty-span ON row",
            "rejection_reasons": [],
        }
        if name not in actual:
            record["rejection_reasons"] = ["missing_capture_no_trace_no_retry"]
        else:
            raw_capture = (captures / name).read_bytes()
            hashes[name] = pilot.sha(raw_capture)
            try:
                capture_row = json.loads(raw_capture)
                if (
                    capture_row["request"] != request
                    or capture_row["relay_job"] != pilot.RELAY_JOB
                    or capture_row["trace_field"] != pilot.TRACE_FIELD
                ):
                    raise ValueError("Captured request/relay/field differs from manifest")
                if capture_row.get("fatal"):
                    record["rejection_reasons"] = ["request_failure"]
                else:
                    text, ids = pilot.visible_rationale(capture_row["response"], tokenizer)
                    record["visible_rationale"] = text
                    reasons = pilot.identity_qc(text, "", record["canonical_final"])
                    record["rejection_reasons"] = reasons
                    if not reasons:
                        draft = pilot.graft_reasoning(off, ids, tokenizer)
                        drafts.append(draft)
                        record.update(
                            status="SURFACE_NATIVE_PASS_PENDING_SCIENCE_POOL_READ",
                            final_text_tokens=draft["final_text_tokens"],
                            reasoning_text_tokens=draft["reasoning_text_tokens"],
                        )
            except (ValueError, KeyError, TypeError) as error:
                record["rejection_reasons"] = [f"native_or_protocol_reject:{type(error).__name__}"]
        records.append(record)
    return records, drafts, hashes


def audit(requests: list[dict], identity: Path, captures: Path, tokenizer_dir: Path, output: Path) -> dict:
    """Same pilot surface/QC/graft, all requested sources in the denominator.

    Pilot r2's exact20 accepted/two rejected outcomes are separate immutable
    inputs for later pool admission. New drafts remain pending science read.
    """
    raw = identity.read_bytes()
    if pilot.sha(raw) != IDENTITY_SHA256:
        raise ValueError("Canonical identity pool changed")
    sources = {}
    for line in raw.splitlines():
        row = json.loads(line)
        if not row["enable_thinking"]:
            sources[row["source_id"]] = row
    tokenizer = pilot.checked_tokenizer(tokenizer_dir)
    if len(requests) != 40956:
        raise ValueError("Full audit requires all40956 remaining requests")
    records, drafts, hashes = surface_records(requests, sources, captures, tokenizer)
    output.mkdir(parents=True, exist_ok=False)
    (output / "review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (output / "draft-on-rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in drafts))
    combined = len(drafts) + 20
    report = {
        "status": "SURFACE_ONLY_NO_POOL_OR_TRAINING_CLEAR",
        "remaining_request_denominator_including_missing_failed_rejected": len(requests),
        "remaining_surface_native_pass": len(drafts),
        "reused_r2_semantic_pass": 20,
        "reused_r2_fixed_reject": 2,
        "total_original_source_denominator": 40978,
        "combined_pilot_semantic_plus_remaining_surface_fraction": combined / 40978,
        "below70percent_alert_before_training": combined / 40978 < 0.7,
        "semantic_acceptance_full_pool": "pending science pool read; may fall below surface count",
        "accepted_remaining_rationale_tokens_surface_only": sum(r["reasoning_text_tokens"] for r in drafts),
        "reused_r2_eligible_rationale_tokens": 643,
        "canonical_final_tokens_unchanged_all_originals": sum(r["final_text_tokens"] for r in sources.values()),
        "fallback": "all failed/rejected/missing traces keep exact B-base empty ON; no resampling",
        "comparison": "identical row/exposure schedule including replay; actual per-arm trace exposure pending",
        "capture_sha256": hashes,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "capture", "audit"))
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-root")
    parser.add_argument("--code-pin")
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    requests, receipt = remaining_requests(args.identity, args.pilot_manifest, args.context)
    manifest = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in requests).encode()
    if args.action == "build":
        with args.manifest.open("xb") as stream:
            stream.write(manifest)
        print(json.dumps({**receipt, "manifest_sha256": pilot.sha(manifest)}))
        return
    if not args.manifest_sha256 or args.manifest.read_bytes() != manifest or pilot.sha(manifest) != args.manifest_sha256:
        parser.error("Capture requires the exact frozen remaining-source manifest")
    if args.action == "audit":
        if args.output is None or args.captures is None or args.tokenizer_dir is None:
            parser.error("Audit requires --output/--captures/--tokenizer-dir")
        print(json.dumps(audit(requests, args.identity, args.captures, args.tokenizer_dir, args.output)))
        return
    if args.output is None or not args.artifact_root or not args.code_pin:
        parser.error("Capture requires --output/--artifact-root/--code-pin")
    print(json.dumps(capture(requests, args.output, args.artifact_root, args.code_pin, receipt)))


if __name__ == "__main__":
    main()
