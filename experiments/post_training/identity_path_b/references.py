# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch small pinned exclusion references for prompt extraction; never execute solutions."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub.utils import build_hf_headers

REFERENCE_FILES = {
    "TIGER-Lab/MMLU-Pro": ["data/test-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
    "allenai/IFBench_test": ["data/train-00000-of-00001.parquet"],
    "truthfulqa/truthful_qa": ["multiple_choice/validation-00000-of-00001.parquet"],
    "mandarjoshi/trivia_qa": ["rc.nocontext/validation-00000-of-00001.parquet"],
    "Idavidrein/gpqa": ["gpqa_diamond.csv"],
}


def fetch_one(item: tuple[dict, dict], output: Path) -> dict:
    """Make one ordinary authorized Hub read and verify the pinned LFS/git digest."""
    api, entry = item
    repo, revision, name = api["id"], api["sha"], entry["rfilename"]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
    response = requests.get(url, headers=build_hf_headers(), timeout=(15, 180))
    result = {"repo": repo, "revision": revision, "filename": name, "http_status": response.status_code}
    if response.status_code != 200:
        # A gated source stays unavailable; do not change access/credentials or substitute a mirror.
        return {**result, "status": "UNAVAILABLE_NO_RETRY"}
    raw = response.content
    if len(raw) != entry["size"]:
        raise ValueError(f"Reference size changed for {repo}/{name}")
    if "lfs" in entry:
        digest = hashlib.sha256(raw).hexdigest()
        expected = entry["lfs"]["sha256"]
    else:
        digest = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        expected = entry["blobId"]
    if digest != expected:
        raise ValueError(f"Pinned reference digest changed for {repo}/{name}")
    target = output / repo.replace("/", "--") / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as file:
        file.write(raw)
    return {
        **result,
        "status": "VERIFIED",
        "path": str(target),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "hub_object_digest": digest,
    }


def fetch(metadata: Path, output: Path) -> dict:
    items, api_receipts = [], {}
    for path in sorted(metadata.glob("*-api.json")):
        raw = path.read_bytes()
        api = json.loads(raw)
        repo = api["id"]
        wanted = REFERENCE_FILES.get(repo)
        if repo == "open-thoughts/OpenThoughts-TBLite":
            wanted = [entry["rfilename"] for entry in api["siblings"] if entry["rfilename"].endswith("/instruction.md")]
        if wanted is None:
            continue
        api_receipts[str(path)] = hashlib.sha256(raw).hexdigest()
        by_name = {entry["rfilename"]: entry for entry in api["siblings"]}
        items.extend((api, by_name[name]) for name in wanted)
    output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as workers:
        records = list(workers.map(lambda item: fetch_one(item, output), items))
    report = {
        "purpose": "Evaluation exclusion only; no source is admitted to replay",
        "metadata_sha256": api_receipts,
        "files": records,
        "coverage_note": (
            "Pins are explicit references; equality with historical suite inputs requires separate binding."
        ),
    }
    (output / "fetch-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fetch(args.metadata, args.output)
    print(
        json.dumps(
            {
                "files": len(report["files"]),
                "unavailable": [row for row in report["files"] if row["status"] != "VERIFIED"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
