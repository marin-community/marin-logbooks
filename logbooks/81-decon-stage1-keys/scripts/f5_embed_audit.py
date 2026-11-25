# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F5: an embedding audit of what stage 1 misses (review point 6).

``items``: embed every eval item (question with lettered choices; no passage) with a small multilingual
encoder. ``chunks``: for part K of N of a seeded sample of catalog shards, sample records, cut them into
overlapping chunks, embed them, and keep each record's nearest items that are not stage-1 candidates of the
v11 run, with an excerpt around the best chunk. ``select``: merge the parts and write the pairs to score
(the highest similarities plus a random draw from the rest), as ``pairs/*.parquet`` for ``score.py``.
"""
import argparse
import io
import json
import logging
import random

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from marin.datakit.decon_tables import parquet_files, read_frame, write_frame
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
MODEL = "intfloat/multilingual-e5-small"
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CATALOG = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full"
SHARDS = 400
RECORDS_PER_SHARD = 5000
CHUNK, STRIDE, MAX_CHUNKS = 1000, 800, 40
CONTEXT = 1000
TOP_PER_CHUNK, TOP_PER_RECORD = 5, 3
MIN_SIMILARITY = 0.80
EXCERPT_TOP, EXCERPT_RANDOM = 1000, 250
BATCH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _encoder():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
        .to(DEVICE)
        .eval()
    )
    return tokenizer, model


@torch.no_grad()
def embed(tokenizer, model, texts: list[str], max_length: int) -> torch.Tensor:
    out = []
    for start in range(0, len(texts), BATCH):
        batch = tokenizer(
            ["query: " + t for t in texts[start : start + BATCH]],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(
            torch.nn.functional.normalize(pooled.float(), dim=-1).to(torch.half if DEVICE == "cuda" else torch.float)
        )
    return torch.cat(out) if out else torch.empty(0, 384, dtype=torch.half, device=DEVICE)


def item_text(question: str, choices) -> str:
    return question + "".join(
        f"\n{LABELS[i % 26]}. {c}" for i, c in enumerate(list(choices if choices is not None else [])[:10])
    )


def run_items(items_dir: str, output: str) -> None:
    frames = [read_frame(p, ["id", "eval", "question", "choices", "reason"]) for p in parquet_files(items_dir)]
    items = pd.concat(frames, ignore_index=True)
    items = items[(items["reason"].isna() | (items["reason"] == "")) & (items["question"].fillna("") != "")]
    tokenizer, model = _encoder()
    vectors = embed(tokenizer, model, [item_text(q, c) for q, c in zip(items["question"], items["choices"])], 512)
    buffer = io.BytesIO()
    np.save(buffer, vectors.cpu().numpy())
    StoragePath(prefix_join(output, "items.npy")).write_bytes(buffer.getvalue())
    write_frame(items[["id", "eval"]].reset_index(drop=True), prefix_join(output, "items.parquet"))
    logger.info("embedded %d items", len(items))


def sampled_shards() -> list[str]:
    shards = json.loads(StoragePath(f"{CATALOG}/shards.json").read_text())
    return random.Random(61).sample(shards, SHARDS)


def _name(shard: str) -> str:
    return shard.split("/normalized/", 1)[1].replace("/", "__")


def run_chunks(output: str, part: int, parts: int) -> None:
    item_vectors = torch.from_numpy(np.load(io.BytesIO(StoragePath(prefix_join(output, "items.npy")).read_bytes())))
    item_vectors = item_vectors.to(DEVICE)
    items = read_frame(prefix_join(output, "items.parquet"))
    ids, evals = items["id"].to_numpy(object), items["eval"].to_numpy(object)
    tokenizer, model = _encoder()
    for shard in sampled_shards()[part::parts]:
        target = StoragePath(prefix_join(output, f"candidates/{_name(shard)}.parquet"))
        if target.exists():
            continue
        table = pq.read_table(io.BytesIO(StoragePath(shard).read_bytes()), columns=["id", "text"])
        rng = random.Random(_name(shard))
        rows = sorted(rng.sample(range(table.num_rows), min(RECORDS_PER_SHARD, table.num_rows)))
        records = table.take(rows).to_pylist()
        chunk_texts, owners, starts = [], [], []
        for number, record in enumerate(records):
            text = record["text"] or ""
            offsets = list(range(0, max(len(text) - CHUNK + STRIDE, 1), STRIDE))
            if len(offsets) > MAX_CHUNKS:
                offsets = sorted(rng.sample(offsets, MAX_CHUNKS))
            for offset in offsets:
                chunk_texts.append(text[offset : offset + CHUNK])
                owners.append(number)
                starts.append(offset)
        vectors = embed(tokenizer, model, chunk_texts, 320)
        best_sim, best_item = [], []
        for start in range(0, len(vectors), 2048):
            sims = vectors[start : start + 2048] @ item_vectors.T
            values, indices = sims.topk(TOP_PER_CHUNK, dim=1)
            best_sim.append(values.float().cpu().numpy())
            best_item.append(indices.cpu().numpy())
        best_sim = np.concatenate(best_sim) if best_sim else np.empty((0, TOP_PER_CHUNK))
        best_item = np.concatenate(best_item) if best_item else np.empty((0, TOP_PER_CHUNK), dtype=np.int64)
        candidates_path = StoragePath(f"{CATALOG}/pairs/{_name(shard)}.parquet")
        stage1 = set()
        if candidates_path.exists():
            pairs = pq.read_table(io.BytesIO(candidates_path.read_bytes()), columns=["record", "item"]).to_pydict()
            stage1 = set(zip(pairs["record"], pairs["item"]))
        flagged_records = {record for record, _ in stage1}
        frame = pd.DataFrame(
            {
                "owner": np.repeat(np.array(owners), TOP_PER_CHUNK),
                "start": np.repeat(np.array(starts), TOP_PER_CHUNK),
                "item_index": best_item.reshape(-1),
                "sim": best_sim.reshape(-1),
            }
        )
        frame = frame[frame["sim"] >= MIN_SIMILARITY]
        frame = frame.sort_values("sim", ascending=False).drop_duplicates(["owner", "item_index"])
        frame = frame.groupby("owner", sort=False).head(TOP_PER_RECORD).reset_index(drop=True)
        record_ids = [str(records[o]["id"]) for o in frame["owner"]]
        frame["record"] = record_ids
        frame["item"] = ids[frame["item_index"].to_numpy()]
        frame["eval"] = evals[frame["item_index"].to_numpy()]
        frame["stage1_pair"] = [(r, i) in stage1 for r, i in zip(record_ids, frame["item"])]
        frame["stage1_record"] = [r in flagged_records for r in record_ids]
        frame["shard"] = shard
        frame["shard_records"] = table.num_rows
        frame["sampled_records"] = len(records)
        # Excerpts only for the pairs that may be scored: the shard's highest similarities and a random draw.
        unseen = frame.index[~frame["stage1_pair"]]
        top = list(unseen[:EXCERPT_TOP])
        rest = list(unseen[EXCERPT_TOP:])
        picked = top + rng.sample(rest, min(EXCERPT_RANDOM, len(rest)))
        frame["selection"] = ""
        frame.loc[top, "selection"] = "top"
        frame.loc[picked[len(top) :], "selection"] = "random"
        frame["right"] = ""
        for index in picked:
            text = records[frame.at[index, "owner"]]["text"] or ""
            start = int(frame.at[index, "start"])
            frame.at[index, "right"] = text[max(0, start - CONTEXT) : start + CHUNK + CONTEXT]
        frame["record_chars"] = [len(records[o]["text"] or "") for o in frame["owner"]]
        write_frame(frame.drop(columns=["owner", "item_index"]), str(target))
        stats = {"shard": shard, "shard_records": table.num_rows, "sampled": len(records), "chunks": len(chunk_texts)}
        StoragePath(prefix_join(output, f"stats/{_name(shard)}.json")).write_text(json.dumps(stats))
        logger.info("%s", stats)


def run_select(output: str) -> None:
    """The pairs with an excerpt (each shard's top and random draw), as ``pairs/pairs.parquet`` for scoring."""
    frames = [read_frame(str(p)) for p in StoragePath(prefix_join(output, "candidates/*.parquet")).glob()]
    candidates = pd.concat(frames, ignore_index=True)
    chosen = candidates[candidates["selection"] != ""]
    write_frame(chosen.reset_index(drop=True), prefix_join(output, "pairs/pairs.parquet"))
    logger.info(
        "%d candidate rows, %d not stage-1 pairs, %d chosen %s",
        len(candidates),
        int((~candidates["stage1_pair"]).sum()),
        len(chosen),
        chosen["selection"].value_counts().to_dict(),
    )


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["items", "chunks", "select"])
    parser.add_argument("--items")
    parser.add_argument("--output", required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "items":
        run_items(args.items, args.output)
    elif args.mode == "chunks":
        run_chunks(args.output, args.part, args.parts)
    else:
        run_select(args.output)


if __name__ == "__main__":
    main()
