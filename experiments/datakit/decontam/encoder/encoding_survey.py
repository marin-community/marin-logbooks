# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Survey which text encodings hide eval items from the stage-1 scan, on real normalized shards.

For every record, count encoding signatures (URL percent-encoding, HTML entities and tags, base64 runs,
quoted-printable, ``\\xNN`` escapes, numbered lines, Latin-Cyrillic mixed words). A record with a signature is
scanned as the pipeline reads it (``decon_keys.record_pairs``) and again after each matching decoding; a
candidate (record, item) that only a decoding finds is written with its decoded excerpt.

    python -m experiments.datakit.decontam.encoder.encoding_survey --locators <locators.parquet> \\
        --sources experiments/datakit/decontam/encoder/sources.json --input-root <normalized root> \\
        --per-source 4 --output <dir>
"""

import argparse
import base64
import binascii
import functools
import html
import json
import logging
import quopri
import random
import re
import urllib.parse
from collections import Counter
from collections.abc import Callable, Iterator

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon_keys import MAX_DOC_CHARS, KeyTable, record_pairs
from marin.datakit.decon_tables import parquet_files, read_table, write_table
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

PERCENT = re.compile(r"%[0-9A-Fa-f]{2}")
ENTITY = re.compile(r"&(?:#\d{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,30});")
TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]{0,15}(?:\s[^<>]{0,200})?/?>")
BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")
QP = re.compile(r"=[0-9A-F]{2}|=\r?\n")
XESC = re.compile(r"\\x[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}")
NUMBERED_LINE = re.compile(r"^\s{0,8}\d{1,6}\s{0,4}[|:.)]\s", re.MULTILINE)
BPE_MARKER = re.compile(r"[\u0120\u010a]")
MIXED_SCRIPT = re.compile(r"(?=\w*[A-Za-z])(?=\w*[Ѐ-ӿ])\w{3,}")
# Cyrillic letters that look like Latin ones, written as escapes.
HOMOGLYPHS = str.maketrans(
    {
        "\u0430": "a",
        "\u0435": "e",
        "\u043e": "o",
        "\u0441": "c",
        "\u0440": "p",
        "\u0445": "x",
        "\u0443": "y",
        "\u0456": "i",
        "\u0458": "j",
        "\u0455": "s",
        "\u043a": "k",
    }
)
# A signature counts when it occurs at least this often in a record.
MIN_HITS = 3
SAMPLE_PER_SHARD = 50


def _url(text: str) -> str:
    return urllib.parse.unquote_plus(text)


def _html(text: str) -> str:
    return html.unescape(TAG.sub(" ", text))


def _base64(text: str) -> str:
    decoded = []
    for run in BASE64_RUN.findall(text):
        try:
            raw = base64.b64decode(run + "=" * (-len(run) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        piece = raw.decode("utf-8", errors="ignore")
        if piece and sum(ch.isprintable() or ch.isspace() for ch in piece) > 0.95 * len(piece):
            decoded.append(piece)
    return "\n".join(decoded)


def _quoted_printable(text: str) -> str:
    return quopri.decodestring(text.encode("utf-8", errors="ignore")).decode("utf-8", errors="ignore")


def _code_escapes(text: str) -> str:
    return XESC.sub(lambda m: _unescape_one(m.group(0)), text)


def _unescape_one(escape: str) -> str:
    try:
        return chr(int(escape[2:], 16))
    except ValueError:
        return escape


def _numbered_lines(text: str) -> str:
    return NUMBERED_LINE.sub(" ", text)


def _bpe_markers(text: str) -> str:
    return text.replace("\u0120", " ").replace("\u010a", "\n")


def _homoglyphs(text: str) -> str:
    return text.translate(HOMOGLYPHS)


# Each decoding, with the signature that triggers it.
DECODINGS: dict[str, tuple[re.Pattern, Callable[[str], str]]] = {
    "url": (PERCENT, _url),
    "html_entity": (ENTITY, _html),
    "html_tag": (TAG, _html),
    "base64": (BASE64_RUN, _base64),
    "quoted_printable": (QP, _quoted_printable),
    "code_escape": (XESC, _code_escapes),
    "numbered_lines": (NUMBERED_LINE, _numbered_lines),
    "homoglyph": (MIXED_SCRIPT, _homoglyphs),
    "bpe_marker": (BPE_MARKER, _bpe_markers),
}
# Signatures that need fewer hits: one base64 run or one mixed-script word is enough.
MIN_HITS_OF = {"base64": 1, "homoglyph": 1, "bpe_marker": 5}


@functools.lru_cache(maxsize=1)
def _index(locators_path: str) -> tuple[KeyTable, pa.Table]:
    locators = read_table(locators_path)
    return KeyTable(locators.column("key1_hash").to_numpy(), np.arange(locators.num_rows)), locators


def make_survey(locators_path: str, output: str, only: tuple[str, ...] = ()):
    def survey(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        table, locators = _index(locators_path)
        rng = random.Random(shard.shard_idx)
        for path in paths:
            name = path.split("/normalized/", 1)[-1].replace("/", "__")
            done = StoragePath(prefix_join(output, f"stats/{name}.json"))
            if done.exists():
                yield json.loads(done.read_text())
                continue
            stats: Counter = Counter()
            rows: list[dict] = []
            signature_samples: dict[str, list[dict]] = {}
            for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                text = str(record.get("text") or "")
                stats["records"] += 1
                if len(text) > MAX_DOC_CHARS:
                    stats["long_records"] += 1
                    continue
                active = [
                    decoding
                    for decoding, (pattern, _) in DECODINGS.items()
                    if not only or decoding in only
                    if len(pattern.findall(text, 0, 2_000_000)[: MIN_HITS_OF.get(decoding, MIN_HITS)])
                    >= MIN_HITS_OF.get(decoding, MIN_HITS)
                ]
                if not active:
                    continue
                stats["records_with_signature"] += 1
                base = {pair.item for pair in record_pairs(str(record["id"]), text, table, locators)}
                stats["base_pairs_on_signature_records"] += len(base)
                for decoding in active:
                    stats[f"signature/{decoding}"] += 1
                    samples = signature_samples.setdefault(decoding, [])
                    if len(samples) < 3 and rng.random() < 0.01:
                        samples.append({"id": str(record["id"]), "snippet": text[:400]})
                    decoded = DECODINGS[decoding][1](text)
                    if decoded == text:
                        continue
                    for pair in record_pairs(str(record["id"]), decoded, table, locators):
                        if pair.item in base:
                            continue
                        base.add(pair.item)
                        stats[f"new/{decoding}"] += 1
                        if len(rows) < SAMPLE_PER_SHARD * len(DECODINGS):
                            rows.append(
                                {
                                    "path": path,
                                    "record": pair.record,
                                    "item": pair.item,
                                    "eval": pair.eval,
                                    "decoding": decoding,
                                    "right": pair.right,
                                    "record_chars": len(text),
                                }
                            )
            if rows:
                write_table(pa.Table.from_pylist(rows), prefix_join(output, f"new/{name}.parquet"))
            result = {"path": path, **stats, "samples": signature_samples}
            done.write_text(json.dumps(result))
            yield result

    return survey


def survey_shards(sources_path: str, input_root: str, per_source: int, seed: int) -> list[str]:
    with StoragePath(sources_path).open("r") as source:
        names = json.load(source)
    rng = random.Random(seed)
    picked: list[str] = []
    for name in names:
        shards = parquet_files(prefix_join(input_root, f"{name}/outputs/main"), "**/*.parquet")
        picked += rng.sample(shards, min(per_source, len(shards)))
    return picked


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locators", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--per-source", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=200)
    parser.add_argument("--only", nargs="*", default=[], help="run only these decodings")
    parser.add_argument("--part", type=int, default=None, help="scan part --part of --parts in this process")
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--ram", default="16g")
    args = parser.parse_args()
    files = survey_shards(args.sources, args.input_root, args.per_source, args.seed)
    StoragePath(prefix_join(args.output, "shards.json")).write_text(json.dumps(files))
    logger.info("survey: %d shards", len(files))
    if args.part is not None:

        class _Shard:
            shard_idx = args.part

        mine = files[args.part :: args.parts]
        logger.info("part %d of %d: %d shards", args.part, args.parts, len(mine))
        for result in make_survey(args.locators, args.output, tuple(args.only))(iter(mine), _Shard()):
            logger.info("%s", {k: v for k, v in result.items() if isinstance(v, int)})
        return
    resources = ResourceConfig(cpu=1, ram=args.ram, preemptible=False)
    context = ZephyrContext(
        name="decon-encoding-survey", resources=resources, max_workers=min(args.max_workers, len(files))
    )
    outcome = context.execute(
        Dataset.from_list(files).map_shard(make_survey(args.locators, args.output)), map_task_resources=resources
    )
    totals: Counter = Counter()
    for result in outcome.results:
        totals.update({k: v for k, v in result.items() if isinstance(v, int)})
    StoragePath(prefix_join(args.output, "totals.json")).write_text(json.dumps(dict(totals), indent=1))
    logger.info("survey done: %s", dict(totals))


if __name__ == "__main__":
    main()
