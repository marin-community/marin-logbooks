# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect device intervals in the four saved prefix XPlanes on regional CPUs without numerical kernels."""

import argparse
import base64
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import storage
from inspect_prefix_assignment import inspect as inspect_assignment
from inspect_prefix_intervals import inspect

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            md5.update(block)
    return {"bytes": path.stat().st_size, "sha256": sha.hexdigest(), "md5_hash": base64.b64encode(md5.digest()).decode()}


def compact_summary(summary):
    """Keep scalar provenance and the first forty ranked rows from each table."""
    result = {}
    for key, value in summary.items():
        if isinstance(value, list):
            result[key] = value[:40]
            result[key + "_full_row_count"] = len(value)
        else:
            result[key] = value
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def one_profile(manifest, index, directory, analysis):
    expected = manifest["profiles"][index]
    bucket_name, object_name = expected["uri"].removeprefix("gs://").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.reload()
    assert bucket.location.lower() == "us-central2"
    blob = bucket.blob(object_name, generation=expected["generation"])
    blob.reload()
    actual = dict(bytes=blob.size, generation=int(blob.generation), crc32c=blob.crc32c, md5_hash=blob.md5_hash)
    assert actual == {key: expected[key] for key in actual}
    path = directory / f"host-{index}.xplane.pb"
    blob.download_to_filename(path, checksum="crc32c", if_generation_match=expected["generation"])
    identity = digest(path)
    assert identity["bytes"] == expected["bytes"] and identity["md5_hash"] == expected["md5_hash"]
    print(json.dumps(dict(event="download_verified", index=index, **identity)), flush=True)
    assert identity["sha256"] == expected["sha256"]
    summary = inspect_assignment(path) if analysis == "assignment" else inspect(path)
    full = directory / f"host-{index}-full.json"
    write_json(full, summary)
    bounded = dict(
        profile=expected,
        verified_identity=identity,
        summary=summary,
        full_summary_identity=digest(full),
        parser_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    write_json(directory / f"host-{index}-bounded.json", bounded)
    path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=["intervals", "assignment"], default="intervals")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--one-index", type=int)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.manifest_sha256
    manifest = json.loads(raw)
    for relative, expected in manifest["source_sha256"].items():
        assert digest(ROOT / relative)["sha256"] == expected, relative
    assert len(manifest["profiles"]) == 4 and len({x["uri"] for x in manifest["profiles"]}) == 4
    assert all(x["uri"].endswith(f"-w-{i}.xplane.pb") for i, x in enumerate(manifest["profiles"]))
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone", headers={"Metadata-Flavor": "Google"}
    )
    with urllib.request.urlopen(request) as response:
        zone = response.read().decode()
    assert zone.rsplit("/", 1)[-1].startswith("us-central2-"), zone
    assert os.environ.get("IRIS_WORKER_ID") not in manifest["excluded_training_workers"]
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/name", headers={"Metadata-Flavor": "Google"}
    )
    with urllib.request.urlopen(request) as response:
        instance_name = response.read().decode()
    assert not instance_name.startswith(manifest["excluded_training_tpu_name"] + "-w-"), instance_name
    assert args.output_prefix.startswith("gs://marin-us-central2/")
    if args.one_index is not None:
        assert args.directory is not None and 0 <= args.one_index < 4
        one_profile(manifest, args.one_index, args.directory, args.analysis)
        return
    prefix = args.output_prefix.rstrip("/") + "/" + uuid.uuid4().hex[:12]
    bucket_name, object_prefix = prefix.removeprefix("gs://").split("/", 1)
    bucket = storage.Client().bucket(bucket_name)
    bucket.reload()
    assert bucket.location.lower() == "us-central2"
    report = dict(
        source_commit=os.environ["GIT_COMMIT"],
        manifest_sha256=args.manifest_sha256,
        manifest=manifest,
        analysis=args.analysis,
        zone=zone,
        instance_name=instance_name,
        worker_id=os.environ.get("IRIS_WORKER_ID"),
        started_utc=datetime.now(UTC).isoformat(),
        completed=False,
        hosts=[],
        interpretation=(
            "Static declared metadata or recorded per-track intervals, as selected by analysis. "
            "No cross-host clock alignment, transfer-only cost, matrix-slot assignment or wall-time speedup inferred."
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for index in range(4):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--one-index",
                    str(index),
                    "--directory",
                    str(directory),
                ],
                check=True,
            )
            full = directory / f"host-{index}-full.json"
            full_uri = prefix + "/" + full.name
            bucket.blob(object_prefix + "/" + full.name).upload_from_filename(full, checksum="crc32c")
            bounded = json.loads((directory / f"host-{index}-bounded.json").read_text())
            bounded["full_summary_uri"] = full_uri
            report["hosts"].append(bounded)
            bucket.blob(object_prefix + "/report.json").upload_from_string(json.dumps(report, allow_nan=False))
            full.unlink()
            print(json.dumps(dict(event="host_completed", index=index, report_uri=prefix + "/report.json")), flush=True)
    report["completed"] = True
    report["completed_utc"] = datetime.now(UTC).isoformat()
    bucket.blob(object_prefix + "/report.json").upload_from_string(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(completed=True, report_uri=prefix + "/report.json")), flush=True)


if __name__ == "__main__":
    main()
