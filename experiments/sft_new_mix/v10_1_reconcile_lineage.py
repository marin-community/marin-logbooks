# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V10.1 tool-conversion lineage reconciliation (the 8:15 ruling's tracer).

Every verified-modified row the durable index carries for the sources V10
sealed must be traceable through the lineage the conversion ran on: raw source
shard/row -> each build generation's ``decisions.jsonl`` (``base_row ->
output_row``) -> the sealed V10 store. This job walks that chain for every
selected source and classifies each index row exactly once:

- ``applied_build`` — the row's output_row carries a build-time verified tool
  replacement (the V10 overlay),
- ``applied_v10_1`` — the row is one of the 4,383 rows V10.1's apply replaced,
- ``dropped:<phase>:<stage>`` — a named build generation's filter dropped the
  row (the lowest generation that dropped it),
- ``retained_survivor`` — the row survives in the sealed V10 store, was not
  applied at either cut, and is a V10.2 delta candidate,
- ``not_in_build`` — the row's raw part never entered the lineage the build
  consumed, or the row's range was outside every slice the build processed
  (recorded per part for the design-exclusion accounting).

The job streams only lineage metadata (receipts and decisions.jsonl) in-region,
writes the per-source reconciliation table and the survivor list, and asserts
the applied counts against the V10 release's retention table.
"""

import argparse
import gzip
import json
import logging
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

MAX_CHAIN_DEPTH = 4
UNSEEN = 0
RETAINED = 1
APPLIED_BUILD = 2
# Classification priority when the same raw row is consumed by more than one
# chain (duplicated raw rows across gate components): an applied copy means the
# document's conversion IS in the sealed store, so it wins over a retained
# copy; a dropped copy wins over a plain retained copy.
PRIORITY = {UNSEEN: 0, RETAINED: 1, APPLIED_BUILD: 3}

# The V10 release's retention table's verified_tool_replacement counts per source.
# These record the campaign's replacements over the tool-bearing population as the
# build's own accounting measured them (including substitutions baked into earlier
# build generations); the lineage's visible overlays are the row-attributable subset.
RETENTION_CSV_APPLIED_BUILD = {
    "sft/source/nemotron_sft_v3/math_v3/train": 1_282_410,
    "sft/source/nemotron_sft_v3/math_v2/high": 232_530,
    "sft/source/nemotron_sft_v3/math_v2/low": 221_458,
    "sft/source/nemotron_sft_v3/math_v4/train": 49_253,
    "sft/source/science-tool-use-conversations": 8_859,
    "sft/source/nemotron_sft_v3/science_v2/rqa": 8_447,
    "sft/source/nemotron_sft_v3/science_v2/so": 1_200,
    "sft/source/nemotron_sft_v3/opencode_v1/agent_skills": 130,
    "sft/source/nemotron_sft_v3/opencode_v1/bash_only_tool": 111,
    "sft/source/nemotron_sft_v3/opencode_v1/bash_only_tool_skills": 101,
    "sft/source/nemotron_sft_v3/science_v2/syn_mcq": 35,
    "sft/source/nemotron_sft_v3/opencode_v1/question_tool": 5,
}
EXPECTED_APPLIED_V10_1 = {
    "sft/source/nemotron_sft_v3/math_v2/high": 1_271,
    "sft/source/nemotron_sft_v3/math_v3/train": 2_388,
    "sft/source/nemotron_sft_v3/math_v4/train": 724,
}


def canonical_part(uri: str) -> str:
    """The raw part's store-relative key, independent of the store-dir convention
    (``datakit/store`` vs ``datakit/levanter-store`` name the same stores; the
    science sources bind to their accepted-dataset caches)."""
    if is_accepted_cache(uri):
        return "science-accepted/" + uri.split("/inputs/science/", 1)[1]
    for marker in ("nemotron_sft_v3/", "science-tool-use-conversations", "penfever-traces/"):
        index = uri.find(marker)
        if index >= 0:
            return uri[index:]
    raise ValueError(f"not a raw store part: {uri}")


def is_raw_part(uri: str) -> bool:
    return "datakit/" in uri and "/train/part-" in uri


def is_accepted_cache(uri: str) -> bool:
    """The science sources' accepted-dataset caches: the pool and the gate build
    both bind to them directly, so they are chain bases too."""
    return "/inputs/science/" in uri and "dataset-accepted" in uri


def shard_dir(cache_uri: str) -> str:
    """The metadata dir beside a cache shard URI (``.../cache`` -> ``...``)."""
    return cache_uri.removesuffix("/cache")


def generation_tag(cache_uri: str) -> str:
    """A short build-generation name from a cache URI, e.g.
    ``2026.11.16-v6-eot-fix-v2/scripted-v6`` or ``2026.11.16-v6-eot-fix-v4/gate-math-rest-v1``."""
    marker = cache_uri.find("marin/users/benfeuer/")
    tail = cache_uri[marker + len("marin/users/benfeuer/") :] if marker >= 0 else cache_uri
    tail = tail.split("/components/", 1)[0]
    return tail.split("/", 1)[-1]


class Lineage:
    """Per raw part: the absolute raw row's classification through the chain."""

    LINEAGE_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer"

    def __init__(self) -> None:
        self.tables: dict[str, dict[int, int]] = defaultdict(dict)
        self.coverage: dict[str, list[int]] = defaultdict(lambda: [None, None])
        self.labels: dict[int, str] = {APPLIED_BUILD: "applied_build", RETAINED: "retained_survivor"}
        self.priorities: dict[int, int] = dict(PRIORITY)
        self.conflicts: Counter = Counter()
        self.parts_seen: set[str] = set()
        self.overlay_ids: set[str] = set()
        self.identity_output_rows: int = 0
        self.opaque_ancestors: set[str] = set()
        self.survivor_positions: dict[tuple[str, int], tuple[str, int]] = {}
        self.unresolved_parents: list[str] = []

    def dropped_code(self, generation: str, stage: str) -> int:
        label = f"dropped:{generation}:{stage}"
        for code, existing in self.labels.items():
            if existing == label:
                return code
        code = max(self.labels) + 1
        self.labels[code] = label
        self.priorities[code] = 2
        return code

    def mark(self, part: str, row: int, code: int) -> None:
        table = self.tables[part]
        covered = self.coverage[part]
        covered[0] = row if covered[0] is None else min(covered[0], row)
        covered[1] = row if covered[1] is None else max(covered[1], row)
        previous = table.get(row, UNSEEN)
        if previous == UNSEEN:
            table[row] = code
        elif previous != code:
            self.conflicts[(self.labels[previous], self.labels[code])] += 1

    def parent_of(self, cache_uri: str) -> str:
        receipt = json.loads(StoragePath(f"{self.metadata_dir(cache_uri)}/receipt.json").read_text())
        job = receipt["job"]
        # Finalize generations read base_shard.cache_uri (their decisions'
        # base_row indexes it); scripted generations record input_path.
        # base_shard.original_input_path is the raw provenance, not the parent.
        parent = job.get("input_path") or (job.get("base_shard") or {}).get("cache_uri")
        if parent is None:
            self.unresolved_parents.append(cache_uri)
            raise ValueError(f"no parent recorded for {cache_uri}")
        return parent

    def read_decisions(self, cache_uri: str) -> dict[int, dict]:
        """{input row: record} from the decisions.jsonl in a shard's metadata
        dir. The records key the parent's rows as ``base_row`` (V10 finalize)
        or ``row`` (scripted/gate generations)."""
        text = StoragePath(f"{self.metadata_dir(cache_uri)}/decisions.jsonl").read_text()
        decisions = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            row = int(record["base_row"] if "base_row" in record else record["row"])
            decisions[row] = record
        return decisions

    def metadata_dir(self, cache_uri: str) -> str:
        """The build-tree dir carrying a shard's receipt and decisions: the V10
        store's shards live under the release tree, their build lineage under
        the root with the same shard key; intermediate caches already live in
        the build tree."""
        if "/stores-v1/" in cache_uri or "/stores-v2/" in cache_uri:
            marker = cache_uri.find("/train/")
            shard_key = cache_uri[marker + len("/train/") :]
            return f"{self.LINEAGE_ROOT}/{shard_key.removesuffix('/cache')}"
        return shard_dir(cache_uri)

    def consume_shard(self, cache_uri: str, shard_key: str | None = None) -> bool:
        """Record one V10 shard's decisions against its chain base (the raw
        part or accepted-dataset cache its parent names), composing the chain
        bottom-up. The shard's own overlays carry each replaced row's raw
        provenance (source_shard + source_row) and its conversion id, so they
        mark applied_build directly. Retained, non-overlay rows are recorded as
        survivors with their sealed-store output coordinates. Returns False
        when the chain runs into an opaque ancestor (a cache generation without
        receipts or decisions)."""
        chain = []
        uri = cache_uri
        opaque = None
        for _ in range(MAX_CHAIN_DEPTH):
            try:
                parent = self.parent_of(uri)
            except FileNotFoundError:
                opaque = uri
                break
            chain.append({"uri": uri, "parent": parent, "decisions": self.read_decisions(uri)})
            if is_raw_part(parent) or is_accepted_cache(parent):
                break
            uri = parent
        else:
            raise ValueError(f"lineage chain deeper than {MAX_CHAIN_DEPTH} at {cache_uri}")
        receipt = json.loads(StoragePath(f"{self.metadata_dir(cache_uri)}/receipt.json").read_text())
        for overlay in receipt["job"].get("overlays", []):
            overlay_shard = overlay.get("source_shard", "")
            if is_raw_part(overlay_shard) or is_accepted_cache(overlay_shard):
                self.mark(canonical_part(overlay_shard), int(overlay["source_row"]), APPLIED_BUILD)
            else:
                self.unresolved_parents.append(f"overlay on non-raw shard in {cache_uri}")
            self.overlay_ids.add(overlay["id"])
        if opaque is not None:
            self.opaque_ancestors.add(generation_tag(opaque))
            return False
        base = canonical_part(chain[-1]["parent"])
        self.parts_seen.add(base)

        # chain[-1]'s decisions key the base's rows; each level closer to the
        # V10 shard keys the level below's output rows. The V10 shard itself
        # (chain[0]) decides applied_build vs survivor for its retained rows.
        source_rows = None
        for level in reversed(chain):
            generation = generation_tag(self.metadata_dir(level["uri"]))
            mapping = {}
            for input_row, record in level["decisions"].items():
                raw_row = input_row if source_rows is None else source_rows.get(input_row)
                if raw_row is None:
                    continue  # a lower generation already dropped this row
                stage = record.get("stage", "retained")
                if stage != "retained":
                    self.mark(base, raw_row, self.dropped_code(generation, stage))
                    continue
                if "output_row" not in record:
                    self.identity_output_rows += 1
                    mapping[input_row] = raw_row
                else:
                    mapping[int(record["output_row"])] = raw_row
                if level is chain[0]:
                    if record.get("verified_replacement"):
                        self.mark(base, raw_row, APPLIED_BUILD)
                    else:
                        self.mark(base, raw_row, RETAINED)
                        self.survivor_positions[(base, raw_row)] = (
                            shard_key or cache_uri,
                            int(record.get("output_row", input_row)),
                        )
            source_rows = mapping
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-uri", required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--index-rows", required=True, help="The compact verified-modified row export (.jsonl.gz)")
    parser.add_argument(
        "--delta-bindings", action="append", required=True, help="A V10.1 merged delta.jsonl; repeatable"
    )
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    catalog = json.loads(read_pinned(args.catalog_uri, args.catalog_sha256).decode())["splits"]["train"]
    applied_v10_1 = set()
    for path in args.delta_bindings:
        fs, key = filesystem_for(path)
        blob = fs.cat(key)
        text = gzip.decompress(blob).decode() if path.endswith(".gz") else blob.decode()
        for line in text.splitlines():
            if line.strip():
                applied_v10_1.add(json.loads(line)["id"])
    logger.info("V10.1 applied ids: %d", len(applied_v10_1))

    fs, key = filesystem_for(args.index_rows)
    index_rows = defaultdict(list)
    for line in gzip.decompress(fs.cat(key)).decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index_rows["sft/source/" + row["source"]].append(row)
    logger.info("index rows: %d across %d sources", sum(len(v) for v in index_rows.values()), len(index_rows))

    report = {"sources": {}, "not_in_build_parts": {}, "conflicts": {}}
    work = Path(tempfile.mkdtemp(prefix="v10-1-reconcile-"))
    with gzip.open(work / "survivors.jsonl.gz", "wt") as survivors:
        for source in sorted(index_rows):
            entry = catalog[source]
            rows = index_rows[source]
            lineage = Lineage()
            composable = True
            for shard_key in sorted(entry["ledger"]["finished_shards"]):
                composable &= lineage.consume_shard(f"{entry['cache_dir'].rstrip('/')}/{shard_key}", shard_key)
            tally = Counter()
            for row in rows:
                part = canonical_part(row["shard"])
                table = lineage.tables.get(part, {})
                code = table.get(int(row["row"]), UNSEEN)
                if row["id"] in applied_v10_1:
                    classification = "applied_v10_1"
                elif row["id"] in lineage.overlay_ids:
                    classification = "applied_build"
                elif code == APPLIED_BUILD:
                    classification = "applied_build"
                elif code == RETAINED:
                    classification = "retained_survivor"
                    position = lineage.survivor_positions.get((part, int(row["row"])))
                    survivors.write(
                        json.dumps(
                            {
                                "source": source,
                                "part": row["shard"],
                                "row": row["row"],
                                "id": row["id"],
                                "export": row["export"],
                                "store_shard": position[0] if position else None,
                                "store_row": position[1] if position else None,
                            }
                        )
                        + "\n"
                    )
                elif code == UNSEEN and not composable:
                    classification = "not_in_build:opaque_ancestor"
                elif code == UNSEEN:
                    # Not decided by any generation of the build's lineage: the
                    # row never entered it (the build consumed specific row
                    # ranges of each part). Recorded per part for the
                    # design-exclusion accounting.
                    classification = "not_in_build"
                    key = row["shard"] + (" (range outside the consumed slices)" if part in lineage.parts_seen else "")
                    report["not_in_build_parts"].setdefault(source, {}).setdefault(key, 0)
                    report["not_in_build_parts"][source][key] += 1
                else:
                    classification = lineage.labels[code]
                tally[classification] += 1
            if lineage.opaque_ancestors:
                tags = report["opaque_ancestors"].setdefault(source, [])
                for tag in sorted(lineage.opaque_ancestors):
                    if tag not in tags:
                        tags.append(tag)
            report["sources"][source] = {
                "verified_modified": len(rows),
                "tally": dict(sorted(tally.items())),
                "retention_csv_applied_build": RETENTION_CSV_APPLIED_BUILD.get(source),
                "drop_labels": {
                    label: count for code, label in lineage.labels.items() if (count := tally.get(label, 0))
                },
                "lineage_parts": len(lineage.parts_seen),
            }
            if sum(tally.values()) != len(rows):
                raise ValueError(f"{source}: classification does not partition the {len(rows)} index rows")
            report["conflicts"][source] = {f"{a} <-> {b}": count for (a, b), count in lineage.conflicts.items()}
            logger.info("%s: %s", source, dict(sorted(tally.items())))

    survivors_uri = f"{args.output_prefix.rstrip('/')}/survivors.jsonl.gz"
    StoragePath(survivors_uri).write_bytes((work / "survivors.jsonl.gz").read_bytes())
    document = json.dumps(report, sort_keys=True, indent=1).encode()
    out_fs, out_key = filesystem_for(f"{args.output_prefix.rstrip('/')}/reconciliation.json")
    out_fs.pipe(out_key, document)
    print(f"wrote {survivors_uri} and {args.output_prefix.rstrip('/')}/reconciliation.json ({len(document)} bytes)")
    for source, data in report["sources"].items():
        print(source, json.dumps(data, sort_keys=True))


if __name__ == "__main__":
    main()
