# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair stage-E payloads whose archive fetch failed.

The stage-E extract records a row in its delta.jsonl even when the row's batch
archive fetch raises (``fetch_errors`` in the extract receipt); the payload is
then missing and the apply job dies on it. This job re-fetches the archive
in-region and writes the missing payload after the extract's own validation:
archive digest, export-member digests, store-tokenizer decode of the ids, and
the delta record's token/supervised counts. It finishes by verifying that
every payload URI named by the delta files exists.
"""

import argparse
import hashlib
import io
import json
import logging
import tarfile

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights
from experiments.sft_new_mix.v10_1_toolconv_apply import (
    ASSISTANT_PREFIX_TEXT,
    MESSAGE_START_TOKEN,
    _read_lines,
)

logger = logging.getLogger(__name__)

STORE_TOKENIZER_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/scripted-v3/tokenizer.json"
)
STORE_TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"


def repair_payload(binding: dict, delta: dict, tokenizer: Tokenizer) -> dict:
    """Fetch, validate and write one delta row's payload; returns the evidence."""
    row_id = delta["id"]
    if binding["id"] != row_id:
        raise ValueError(f"binding id {binding['id']} does not match the delta row {row_id}")
    if binding.get("export_manifest_sha256") != delta.get("export_manifest_sha256"):
        raise ValueError(f"{row_id}: the binding and the delta row pin different export manifests")

    archive_uri = binding["archive_uri"]
    fsa, keya = filesystem_for(archive_uri)
    archive = fsa.cat(keya)
    if hashlib.sha256(archive).hexdigest() != binding["archive_sha256"]:
        raise ValueError(f"{row_id}: the fetched archive digest differs from the binding pin")

    prefix = binding["record_prefix"]
    files = binding["export_manifest"]["files"]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        member_ids = tar.extractfile(f"{prefix}export/input_ids.i32")
        member_text = tar.extractfile(f"{prefix}export/text.txt")
        if member_ids is None or member_text is None:
            raise ValueError(f"{row_id}: the archive does not carry both export members")
        ids_bytes = member_ids.read()
        text_bytes = member_text.read()
    for name, blob in (("input_ids.i32", ids_bytes), ("text.txt", text_bytes)):
        if hashlib.sha256(blob).hexdigest() != files[name]:
            raise ValueError(f"{row_id}: export member {name} digest differs from the manifest")

    ids = np.frombuffer(ids_bytes, dtype="<i4")
    decoded = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
    if decoded != text_bytes.decode():
        raise ValueError(f"{row_id}: the store tokenizer does not decode the ids to the exported text")

    message_start = tokenizer.token_to_id(MESSAGE_START_TOKEN)
    assistant_prefix = tuple(tokenizer.encode(ASSISTANT_PREFIX_TEXT, add_special_tokens=False).ids)
    weights = prediction_weights(
        ids,
        loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
        message_start_id=message_start,
        assistant_prefix=assistant_prefix,
        trained_assistant_turns=None,
    )
    if len(ids) != delta["tokens"]:
        raise ValueError(f"{row_id}: the payload carries {len(ids)} tokens, the delta row pins {delta['tokens']}")
    if int(weights.sum()) != delta["supervised_tokens"]:
        raise ValueError(
            f"{row_id}: the recomputed mask supervises {int(weights.sum())} positions, "
            f"the delta row pins {delta['supervised_tokens']}"
        )
    if not weights.any():
        raise ValueError(f"{row_id}: the recomputed assistant mask is empty")

    blob = ids.astype("<i4", copy=False).tobytes()
    fs, key = filesystem_for(delta["payload_uri"])
    fs.pipe(key, blob)
    return {
        "id": row_id,
        "payload_uri": delta["payload_uri"],
        "payload_sha256": hashlib.sha256(blob).hexdigest(),
        "payload_bytes": len(blob),
        "tokens": len(ids),
        "supervised_tokens": int(weights.sum()),
        "archive_uri": archive_uri,
        "archive_sha256": binding["archive_sha256"],
        "export_manifest_sha256": binding["export_manifest_sha256"],
        "retokenized": False,
    }


def verify_payload_population(delta_paths: list[str]) -> dict:
    """Every payload URI named by the delta files must exist."""
    missing = []
    total = 0
    seen: set[str] = set()
    for path in delta_paths:
        for line in _read_lines(path):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            total += 1
            fs, key = filesystem_for(record["payload_uri"])
            try:
                fs.info(key)
            except FileNotFoundError:
                missing.append({"id": record["id"], "payload_uri": record["payload_uri"]})
    if missing:
        raise ValueError(f"{len(missing)} of {total} payload URIs are missing: {missing[:5]}")
    return {"payload_rows": total, "missing": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-input", required=True, help="JSONL with one {binding, delta} record")
    parser.add_argument(
        "--delta-paths",
        action="append",
        default=[],
        help="Delta JSONLs whose payload URIs are verified after the repair; repeatable",
    )
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    tokenizer = Tokenizer.from_str(read_pinned(STORE_TOKENIZER_URI, STORE_TOKENIZER_SHA256).decode())
    records = [json.loads(line) for line in _read_lines(args.repair_input) if line.strip()]
    if len(records) != 1:
        raise ValueError(f"expected exactly one repair record, got {len(records)}")

    repair = repair_payload(records[0]["binding"], records[0]["delta"], tokenizer)
    logger.info("repaired %s: %d tokens, sha %s", repair["id"], repair["tokens"], repair["payload_sha256"])

    receipt = {
        "status": "COMPLETE",
        "stage": "V10.1 stage-E payload repair",
        "repairs": [repair],
        "payload_population": verify_payload_population(args.delta_paths) if args.delta_paths else None,
    }
    document = json.dumps(receipt, sort_keys=True, indent=1).encode()
    fs, key = filesystem_for(f"{args.destination.rstrip('/')}/payload-repair-receipt.json")
    fs.pipe(key, document)
    print(json.dumps(receipt["repairs"][0], indent=1, sort_keys=True))
    if receipt["payload_population"] is not None:
        print(json.dumps(receipt["payload_population"], sort_keys=True))


if __name__ == "__main__":
    main()
