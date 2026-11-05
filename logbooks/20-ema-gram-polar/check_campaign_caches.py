# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check original campaign cache metadata before submission or worker startup."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import storage

BASE = Path(__file__).parent


def catalog_digest(rows):
    identities = sorted((x["name"], int(x["size"]), x["crc32c"]) for x in rows)
    return hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()


def check_component(component, full):
    client = storage.Client()
    bucket = client.bucket(component["bucket"])
    catalog = None
    digest = None
    if full:
        catalog = {b.name: b for b in client.list_blobs(bucket, prefix=component["prefix"])}
        rows = [dict(name=b.name, size=b.size, crc32c=b.crc32c) for b in catalog.values()]
        digest = catalog_digest(rows)
        if len(rows) != component["objects"] or digest != component["catalog_sha256"]:
            raise ValueError(f"Incomplete or changed original cache catalog: {component['prefix']}")

    def check_required(expected):
        blob = catalog.get(expected["name"]) if catalog is not None else bucket.get_blob(expected["name"])
        if blob is None:
            raise FileNotFoundError(f"gs://{bucket.name}/{expected['name']}")
        if (blob.size, blob.crc32c) != (expected["size"], expected["crc32c"]):
            raise ValueError(f"Original cache identity changed: {blob.name}")
        item = dict(name=blob.name, generation=blob.generation, size=blob.size, crc32c=blob.crc32c)
        if blob.name.endswith(".json"):
            # Only bounded metadata is downloaded; token chunks stay in their region.
            if blob.size > 16 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large cache metadata: {blob.name}")
            data = blob.download_as_bytes(if_generation_match=blob.generation)
            json.loads(data)
            item["readback_sha256"] = hashlib.sha256(data).hexdigest()
        return item

    # Preserve every identity/readback check and manifest order while bounding
    # concurrency for components with hundreds of independent metadata objects.
    with ThreadPoolExecutor(max_workers=16) as pool:
        checked = list(pool.map(check_required, component["required"]))
    result = dict(component=component["component"], required=checked)
    if full:
        result.update(objects=len(catalog), catalog_sha256=digest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=BASE / "single-seed-cache-manifest.json")
    parser.add_argument("--full", action="store_true", help="Verify every object's name, size and CRC before submission")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = dict(
        observed_utc=datetime.now(UTC).isoformat(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        full=args.full,
        checks=[],
        errors=[],
        passed=False,
    )
    with ThreadPoolExecutor(max_workers=12) as pool:
        pending = [(c, pool.submit(check_component, c, args.full)) for c in manifest["components"]]
        for component, future in pending:
            try:
                result["checks"].append(future.result())
            except Exception as error:
                # Collect all missing components, persist the failed gate, then exit nonzero.
                result["errors"].append(dict(component=component["component"], error=str(error)))
    result["passed"] = not result["errors"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], checked=len(result["checks"]), errors=result["errors"])))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
