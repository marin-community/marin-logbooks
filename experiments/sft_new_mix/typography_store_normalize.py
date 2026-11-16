# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace U+202F and U+2011 with an ASCII space and hyphen in a tokenized SFT store.

The gpt-oss teacher writes U+202F (narrow no-break space) and U+2011
(non-breaking hyphen) where other text uses a space or a hyphen. Each character
is usually one token in the 2026.09.18 tokenizer (``âĢ¯``, ``ĠâĢ¯``, ``âĢĳ``) but
can also be split across tokens, so occurrences are found on the row's bytes. The
normalizer reads an existing sharded Levanter store and writes a new one with
the same shard names and rows in the same order. A row without those tokens is
copied token for token. In a row with them, each occurrence is rewritten by
decoding the words around it, replacing the characters and re-encoding those
words, so the replacement text gets the tokenization it would have had in the
source. Every rewritten row is checked: its decoded text must equal the source
row's decoded text with the characters replaced.
"""

import argparse
import json
import logging
import re
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from levanter.store.cache import CACHE_LAYOUT_SHARDED, LEDGER_FILE_NAME, CacheLedger, CacheMetadata, write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

READ_BATCH_ROWS = 256
WRITE_BATCH_ROWS = 256
FIRST_SPECIAL_ID = 128000
REPLACEMENTS = {"\u202f": " ", "\u2011": "-"}
SFT_PREFIX = "sft/"
NORMALIZED_MARKER = "normalization.json"
UNCHANGED_MARKER = "unchanged.json"
TARGET_BYTES = tuple(c.encode() for c in REPLACEMENTS)
COUNTED = ("nnbsp", "nbhyphen")


def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode table used by byte-level BPE vocabularies."""
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("\xa1"), ord("\xac") + 1),
        *range(ord("\xae"), ord("\xff") + 1),
    ]
    byte_values = printable[:]
    codes = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            byte_values.append(b)
            codes.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(byte_values, codes, strict=True)}


def normalize_text(text: str) -> str:
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


class Rewriter:
    """Byte-located, token-level rewriting for one byte-level BPE tokenizer."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        size = max(vocab.values()) + 1
        decoder = _byte_decoder()
        self.token_bytes: list[bytes] = [b""] * size
        for token, index in vocab.items():
            if index >= FIRST_SPECIAL_ID:
                self.token_bytes[index] = token.encode()
            else:
                self.token_bytes[index] = bytes(decoder[ch] for ch in token)
        # A token can hold a whole target or the start of one that the next token completes.
        prefixes = {t[:k] for t in TARGET_BYTES for k in range(1, len(t))}
        self.candidate = np.array(
            [any(t in b for t in TARGET_BYTES) or any(b.endswith(p) for p in prefixes) for b in self.token_bytes]
        )
        self.lengths = np.array([len(b) for b in self.token_bytes], dtype=np.int64)
        # A word starts at a special token or at a token whose text starts with whitespace.
        self.word_start = np.zeros(size, dtype=bool)
        for token, index in vocab.items():
            if index >= FIRST_SPECIAL_ID or token[:1] in ("Ġ", "Ċ", "ĉ", "č"):
                self.word_start[index] = True
        self.special = np.arange(size) >= FIRST_SPECIAL_ID

    def decode(self, ids: np.ndarray) -> str:
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=False)

    def occurrences(self, tokens: np.ndarray) -> list[tuple[int, int, int]]:
        """``(first_token, last_token, target_index)`` for each target character in the row."""
        if not self.candidate[tokens].any():
            return []
        data = b"".join(self.token_bytes[t] for t in tokens.tolist())
        ends = np.cumsum(self.lengths[tokens])
        found = []
        for target_index, target in enumerate(TARGET_BYTES):
            at = data.find(target)
            while at >= 0:
                first = int(np.searchsorted(ends, at, side="right"))
                last = int(np.searchsorted(ends, at + len(target) - 1, side="right"))
                found.append((first, last, target_index))
                at = data.find(target, at + len(target))
        return found

    def windows(self, tokens: np.ndarray, found: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
        """Merged token ranges covering each occurrence, its word and the following word."""
        starts = np.flatnonzero(self.word_start[tokens])
        spans = []
        for first, last, _ in found:
            before = starts[starts <= first]
            begin = int(before[-1]) if len(before) else 0
            if self.special[tokens[begin]]:
                begin += 1
            # End after the next word, which may merge with the replacement space; stop at specials.
            after = starts[starts > last]
            end = int(after[1]) if len(after) > 1 else len(tokens)
            special_after = np.flatnonzero(self.special[tokens[last + 1 : end]])
            if len(special_after):
                end = last + 1 + int(special_after[0])
            spans.append((begin, end))
        merged: list[tuple[int, int]] = []
        for begin, end in sorted(spans):
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((begin, end))
        return merged

    def rewrite(self, tokens: np.ndarray, found: list[tuple[int, int, int]]) -> np.ndarray:
        pieces = []
        cursor = 0
        for begin, end in self.windows(tokens, found):
            pieces.append(tokens[cursor:begin])
            text = normalize_text(self.decode(tokens[begin:end]))
            pieces.append(np.asarray(self.tokenizer.encode(text, add_special_tokens=False).ids, dtype=tokens.dtype))
            cursor = end
        pieces.append(tokens[cursor:])
        out = np.concatenate(pieces)
        expected = normalize_text(self.decode(tokens))
        if self.decode(out) != expected:
            raise ValueError("rewritten row does not decode to the normalized source text")
        if self.occurrences(out):
            raise ValueError("rewritten row still holds a target character")
        if not np.array_equal(out[self.special[out]], tokens[self.special[tokens]]):
            raise ValueError("rewritten row changed its special tokens")
        return out


def _shard_rows(shard: str) -> Iterator[np.ndarray]:
    store = JaggedArrayStore.open(str(StoragePath(shard) / "input_ids"), mode="r", dtype=np.int32)
    count = len(store)
    for start in range(0, count, READ_BATCH_ROWS):
        for tokens in store.get_batch_sync(list(range(start, min(start + READ_BATCH_ROWS, count)))):
            yield np.asarray(tokens)


def normalize_shard(source_shard: str, output_shard: str | None, tokenizer_path: str) -> dict:
    """Rewrite every row of one shard; with ``output_shard=None`` only count."""
    rewriter = Rewriter(Tokenizer.from_str(StoragePath(tokenizer_path).read_text()))
    stats = {"source_shard": source_shard, "rows": 0, "tokens": 0, "out_tokens": 0, "changed_rows": 0}
    stats.update({f"count_{name}": 0 for name in COUNTED})

    def records() -> Iterator[dict[str, np.ndarray]]:
        for tokens in _shard_rows(source_shard):
            stats["rows"] += 1
            stats["tokens"] += len(tokens)
            found = rewriter.occurrences(tokens)
            if found:
                for _, _, target_index in found:
                    stats[f"count_{COUNTED[target_index]}"] += 1
                stats["changed_rows"] += 1
                tokens = rewriter.rewrite(tokens, found)
            stats["out_tokens"] += len(tokens)
            yield {"input_ids": tokens}

    if output_shard is None:
        for _ in records():
            pass
    else:
        write_levanter_cache(records(), output_shard, metadata={}, batch_size=WRITE_BATCH_ROWS, max_pending_batches=4)
    return stats


def _normalize_shard_task(args: tuple[str, str | None, str]) -> dict:
    return normalize_shard(*args)


def _store_summary(source: str, output: str | None, results: list[dict]) -> dict:
    return {
        "source": source,
        "output": output,
        "rows": sum(r["rows"] for r in results),
        "tokens": sum(r["tokens"] for r in results),
        "out_tokens": sum(r["out_tokens"] for r in results),
        "changed_rows": sum(r["changed_rows"] for r in results),
        **{f"count_{n}": sum(r[f"count_{n}"] for r in results) for n in COUNTED},
        "shards": results,
    }


def _check_rows(source_ledger: CacheLedger, results: list[dict]) -> None:
    for shard, result in zip(source_ledger.finished_shards, results, strict=True):
        if result["rows"] != source_ledger.shard_rows[shard]:
            raise ValueError(f"{shard}: read {result['rows']} rows, ledger says {source_ledger.shard_rows[shard]}")


def _commit_store(source_ledger: CacheLedger, output: str, summary: dict) -> None:
    """Write the output store's ledger, then ``normalization.json`` as its completion marker."""
    shards = source_ledger.finished_shards
    results = summary["shards"]
    ledger = CacheLedger(
        total_num_rows=summary["rows"],
        shard_rows={shard: r["rows"] for shard, r in zip(shards, results, strict=True)},
        is_finished=True,
        finished_shards=list(shards),
        field_counts={"input_ids": summary["out_tokens"]},
        field_counts_by_shard={shard: {"input_ids": r["out_tokens"]} for shard, r in zip(shards, results, strict=True)},
        layout=CACHE_LAYOUT_SHARDED,
        metadata=CacheMetadata.empty(),
    )
    ledger._serialize_and_commit(output)
    (StoragePath(output) / NORMALIZED_MARKER).write_text(json.dumps(summary) + "\n")


def _sharded_ledger(source: str) -> CacheLedger:
    ledger = CacheLedger.load(source)
    if ledger.layout != CACHE_LAYOUT_SHARDED:
        raise ValueError(f"{source} is not a sharded store")
    return ledger


def normalize_store(source: str, output: str | None, tokenizer_path: str, workers: int) -> dict:
    """Normalize every shard of one sharded store; with ``output=None`` only count."""
    source_ledger = _sharded_ledger(source)
    tasks = [
        (str(StoragePath(source) / shard), None if output is None else str(StoragePath(output) / shard), tokenizer_path)
        for shard in source_ledger.finished_shards
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_normalize_shard_task, tasks))
    _check_rows(source_ledger, results)
    summary = _store_summary(source, output, results)
    if output is not None:
        _commit_store(source_ledger, output, summary)
    return summary


def _sft_stores(catalog: dict, output_root: str) -> dict[str, tuple[str, str]]:
    """``name -> (source cache_dir, output dir)`` for each SFT entry of a catalog."""
    root = output_root.rstrip("/") + "/"
    for split in catalog["splits"].values():
        for entry in split.values():
            cache_dir = entry["cache_dir"].rstrip("/") + "/"
            if root.startswith(cache_dir) or cache_dir.startswith(root):
                raise ValueError(f"output root {output_root} overlaps source store {entry['cache_dir']}")
    entries = catalog["splits"]["train"]
    return {
        name: (
            entries[name]["cache_dir"],
            str(StoragePath(output_root) / re.sub(r"[^A-Za-z0-9._-]+", "__", name) / "train"),
        )
        for name in sorted(entries)
        if name.startswith(SFT_PREFIX)
    }


def normalize_catalog_part(
    catalog: str, output_root: str, tokenizer_path: str, workers: int, part: int, parts: int
) -> None:
    """Normalize the SFT stores of one catalog partition (every ``parts``-th store, starting at ``part``).

    All shards of the partition share one worker pool. A counting pass finds the
    stores with target characters; only those are written. Each finished store
    gets a completion marker, ``normalization.json`` or ``unchanged.json``, so a
    restarted part skips it. Replay entries are never read.
    """
    stores = list(_sft_stores(json.loads(StoragePath(catalog).read_text()), output_root).items())[part::parts]
    pending = [
        (name, source, output)
        for name, (source, output) in stores
        if not (StoragePath(output) / NORMALIZED_MARKER).exists()
        and not (StoragePath(output) / UNCHANGED_MARKER).exists()
    ]
    logger.info("part %d/%d: %d stores, %d pending", part, parts, len(stores), len(pending))
    ledgers = {name: _sharded_ledger(source) for name, source, _ in pending}

    def run(items: list[tuple[str, str, str]], write: bool) -> dict[str, list[dict]]:
        tasks = [
            (str(StoragePath(source) / shard), str(StoragePath(output) / shard) if write else None, tokenizer_path)
            for name, source, output in items
            for shard in ledgers[name].finished_shards
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            flat = iter(list(pool.map(_normalize_shard_task, tasks)))
        grouped = {name: [next(flat) for _ in ledgers[name].finished_shards] for name, _, _ in items}
        for name, results in grouped.items():
            _check_rows(ledgers[name], results)
        return grouped

    changed = []
    counted = run(pending, write=False)
    for name, source, output in pending:
        summary = _store_summary(source, None, counted[name])
        if summary["changed_rows"]:
            changed.append((name, source, output))
        else:
            (StoragePath(output) / UNCHANGED_MARKER).write_text(json.dumps(summary) + "\n")
            logger.info("%s: no target characters", name)
    written = run(changed, write=True)
    for name, source, output in changed:
        summary = _store_summary(source, output, written[name])
        _commit_store(ledgers[name], output, summary)
        logger.info("%s: %s of %s rows rewritten", name, summary["changed_rows"], summary["rows"])


def assemble_catalog(catalog: str, output_root: str, catalog_output: str) -> None:
    """Write a catalog whose rewritten SFT entries point at their normalized stores.

    Every SFT store must carry a completion marker from ``normalize_catalog_part``.
    """
    if catalog_output == catalog:
        raise ValueError("--catalog-output must differ from --catalog")
    source = json.loads(StoragePath(catalog).read_text())
    entries = source["splits"]["train"]
    report = {}
    for name, (_, output) in _sft_stores(source, output_root).items():
        if (StoragePath(output) / NORMALIZED_MARKER).exists():
            summary = json.loads((StoragePath(output) / NORMALIZED_MARKER).read_text())
            entries[name] = {
                "cache_dir": output,
                "ledger": json.loads((StoragePath(output) / LEDGER_FILE_NAME).read_text()),
            }
        elif (StoragePath(output) / UNCHANGED_MARKER).exists():
            summary = json.loads((StoragePath(output) / UNCHANGED_MARKER).read_text())
        else:
            raise ValueError(f"{name}: no completion marker under {output}")
        report[name] = {k: v for k, v in summary.items() if k != "shards"}
    StoragePath(catalog_output).write_text(json.dumps(source))
    (StoragePath(output_root) / "catalog-report.json").write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", help="Cache catalog whose SFT stores are normalized")
    parser.add_argument("--output-root", help="With --catalog: directory for the normalized stores")
    parser.add_argument("--part", type=int, help="With --catalog: normalize this partition of the SFT stores")
    parser.add_argument("--parts", type=int, help="With --catalog: number of partitions")
    parser.add_argument("--catalog-output", help="With --catalog: assemble the patched catalog here instead")
    parser.add_argument("--source", help="Sharded store directory (the split, e.g. .../train)")
    parser.add_argument("--output", help="Output store directory; omit to count only")
    parser.add_argument("--summary", help="Where to write the summary JSON")
    parser.add_argument("--tokenizer", help="Path or URL of tokenizer.json; required unless assembling")
    parser.add_argument("--workers", type=int, help="Required unless assembling")
    parser.add_argument("--shard", action="append", help="Count only these shard names")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    if args.catalog and args.catalog_output:
        assemble_catalog(args.catalog, args.output_root, args.catalog_output)
        return
    if args.tokenizer is None or args.workers is None:
        raise ValueError("--tokenizer and --workers are required")
    if args.catalog:
        normalize_catalog_part(args.catalog, args.output_root, args.tokenizer, args.workers, args.part, args.parts)
        return
    if args.shard:
        if args.output:
            raise ValueError("--shard counts only; a normalized store must contain every shard")
        summary = {
            "shards": [normalize_shard(str(StoragePath(args.source) / s), None, args.tokenizer) for s in args.shard]
        }
    else:
        summary = normalize_store(args.source, args.output, args.tokenizer, args.workers)
    if args.summary:
        StoragePath(args.summary).write_text(json.dumps(summary) + "\n")
    logger.info({k: v for k, v in summary.items() if k != "shards"})


if __name__ == "__main__":
    main()
