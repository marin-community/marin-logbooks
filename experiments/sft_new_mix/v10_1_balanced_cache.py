# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the balanced-reasoning v1 store into a V10.1 cache component (V10.1 stage B1).

The balanced-reasoning lane's v1 store (7,728 rows, marin CHAT_SCHEMA + metadata,
`s3://marin-us-east-02a/marin/users/benfeuer/balanced-reasoning/v1-store-2026.11.24.2/`)
is the only balanced-thinking store ready for the mix (see v10.1/STATUS.md section 1a).
This job renders every row through the shared Marin chat template, tokenizes with the
pinned 2026.09.18 tokenizer, computes the chat_assistant_only prediction weights with
the shared loss-modes machinery, and writes one Levanter cache shard plus a receipt on
the V10.1 work prefix. The store is small (single shard; the whole rows.parquet is
6.8 MB), so the job is one task on the store's region.
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from levanter.store.cache import write_levanter_cache
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import render_jinja_template

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights

logger = logging.getLogger(__name__)

ROWS_URI = "s3://marin-us-east-02a/marin/users/benfeuer/balanced-reasoning/v1-store-2026.11.24.2/rows.parquet"
ROWS_SHA256 = "6b5dc656130f4f946a793172ef4495d75f67c1d26e30ad813d0b42e3756ec9be"
MANIFEST_URI = "s3://marin-us-east-02a/marin/users/benfeuer/balanced-reasoning/v1-store-2026.11.24.2/manifest.json"
TOKENIZER_URI = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18/tokenizer.json"
TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"
STORE_ROWS = 7_728


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True, help="Destination prefix for the cache + receipt")
    parser.add_argument("--work", default=None, help="Local work directory (default: a fresh temp dir)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    manifest = json.loads(StoragePath(MANIFEST_URI).read_bytes())
    if manifest["rows"] != STORE_ROWS:
        raise ValueError(f"The balanced store manifest pins {manifest['rows']} rows, expected {STORE_ROWS}")
    if manifest["artifacts"]["rows"]["sha256"] != ROWS_SHA256:
        raise ValueError("The balanced store rows digest differs from the pinned build input")

    work = Path(args.work) if args.work else Path("/tmp/v10-1-balanced-cache")
    work.mkdir(parents=True, exist_ok=True)
    fs, key = filesystem_for(ROWS_URI)
    local_rows = work / "rows.parquet"
    local_rows.write_bytes(fs.cat(key))
    actual = hashlib.sha256(local_rows.read_bytes()).hexdigest()
    if actual != ROWS_SHA256:
        raise ValueError(f"Downloaded rows digest {actual} differs from the pin {ROWS_SHA256}")

    tokenizer = Tokenizer.from_str(read_pinned(TOKENIZER_URI, TOKENIZER_SHA256).decode())
    message_start = tokenizer.token_to_id("<|start_header_id|>")
    assistant_prefix = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False).ids

    table = pq.read_table(local_rows)
    if table.num_rows != STORE_ROWS:
        raise ValueError(f"The store parquet carries {table.num_rows} rows, expected {STORE_ROWS}")

    stats: Counter = Counter()
    lengths: list[int] = []
    supervised = 0
    digest = hashlib.sha256()

    def records():
        nonlocal supervised
        for row in table.to_pylist():
            messages = [{"role": turn["role"], "content": turn["content"]} for turn in row["messages"]]
            rendered, _ = render_jinja_template(
                conversations=[messages],
                chat_template=MARIN_CHAT_TEMPLATE,
                bos_token="",
                add_generation_prompt=False,
            )
            ids = tokenizer.encode(rendered[0], add_special_tokens=False).ids
            tokens = np.asarray(ids, dtype=np.int32)
            weights = prediction_weights(
                tokens,
                loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
                message_start_id=message_start,
                assistant_prefix=assistant_prefix,
                trained_assistant_turns=None,
            )
            stats["rows"] += 1
            stats["tokens"] += len(tokens)
            stats["supervised"] += int(weights.sum())
            lengths.append(len(tokens))
            supervised += int(weights.sum())
            digest.update(tokens.astype("<i4", copy=False).tobytes())
            digest.update(weights.astype("<f4", copy=False).tobytes())
            yield {"input_ids": tokens, "loss_weights": weights}

    output_uri = f"{args.output_prefix.rstrip('/')}/cache"
    result = write_levanter_cache(
        records(),
        output_uri,
        metadata={
            "source": "balanced_reasoning/v1",
            "loss_mode": "chat_assistant_only",
            "loss_weights_indexing": "prediction",
            "store_rows": STORE_ROWS,
            "store_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        },
        batch_size=128,
        max_pending_batches=2,
    )
    if result["count"] != STORE_ROWS:
        raise ValueError(f"Cache row count {result['count']} differs from the store population {STORE_ROWS}")

    receipt = {
        "status": "COMPLETE",
        "stage": "V10.1 B1 balanced-reasoning cache",
        "pins": {
            "rows_uri": ROWS_URI,
            "rows_sha256": ROWS_SHA256,
            "manifest_uri": MANIFEST_URI,
            "tokenizer_uri": TOKENIZER_URI,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "store_rows": STORE_ROWS,
        },
        "output": {
            "cache_uri": output_uri,
            "rows": result["count"],
            "tokens": sum(lengths),
            "supervised_prediction_positions": supervised,
        },
        "lengths_sha256": hashlib.sha256(json.dumps(lengths).encode()).hexdigest(),
        "output_tokens_and_weights_sha256": digest.hexdigest(),
        "stats": dict(stats),
    }
    document = json.dumps(receipt, sort_keys=True, indent=1).encode()
    fs, key = filesystem_for(f"{args.output_prefix.rstrip('/')}/receipt.json")
    fs.pipe(key, document)
    print(json.dumps({k: receipt[k] for k in ("output", "stats")}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
