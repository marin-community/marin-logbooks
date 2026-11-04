# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the assembled subset object catalog without downloading token payloads."""

import hashlib
import json
from pathlib import Path

from check_campaign_caches import catalog_digest
from google.cloud import storage

BASE = Path(__file__).parent


def main():
    dataset = json.loads((BASE / "subset-dataset.json").read_text())
    assert dataset["success"] and dataset["canonical_success"] and dataset["wandb_finished"]
    original = json.loads((BASE / "single-seed-cache-manifest.json").read_text())
    # Evaluation datasets keep their original immutable identity checks.
    components = [c for c in original["components"] if c["component"].startswith(("paloma/", "uncheatable_eval/"))]
    assert len(components) == 23
    client = storage.Client()
    for name, component in dataset["components"].items():
        uri = component["cache_root"] + "/train/"
        bucket_name, prefix = uri.removeprefix("gs://").split("/", 1)
        blobs = list(client.list_blobs(bucket_name, prefix=prefix))
        rows = [dict(name=b.name, size=b.size, crc32c=b.crc32c, generation=int(b.generation)) for b in blobs]
        assert any(r["name"] == prefix + "shard_ledger.json" for r in rows)
        # All ledgers and array metadata, plus edge chunks for every physical array.
        required = {r["name"]: r for r in rows if r["name"].endswith(".json")}
        arrays = {}
        for row in rows:
            if "/c/" in row["name"]:
                arrays.setdefault(row["name"].rsplit("/c/", 1)[0], []).append(row)
        for chunks in arrays.values():
            chunks.sort(key=lambda r: r["name"])
            for row in (chunks[0], chunks[-1]):
                required[row["name"]] = row
        components.append(
            dict(
                component=name,
                bucket=bucket_name,
                prefix=prefix,
                source="assembled_fixed_subset",
                objects=len(rows),
                bytes=sum(r["size"] for r in rows),
                catalog_sha256=catalog_digest(rows),
                required=sorted(required.values(), key=lambda r: r["name"]),
            )
        )
    assert len(components) == 26
    result = dict(
        schema=1,
        identity_fields=["name", "size", "crc32c"],
        dataset_sha256=hashlib.sha256((BASE / "subset-dataset.json").read_bytes()).hexdigest(),
        components=components,
    )
    (BASE / "subset-cache-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(components=len(components), objects=sum(c["objects"] for c in components))))


if __name__ == "__main__":
    main()
