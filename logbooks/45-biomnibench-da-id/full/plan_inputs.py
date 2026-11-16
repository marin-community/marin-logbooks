# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan shared input images from the pinned public release without downloading data."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi

REVISION = "e1c8ca5e11a620087bc48d97888eb69176a1f235"
REPOSITORY = "phylobio/BiomniBench-DA"
LUPUS = {
    "size": 2642705030,
    "sha256": "33179d1cf993091dc0bd41372629534f6b7da1ce1114d74c36110ffbee375d53",
    "url": (
        "https://huggingface.co/datasets/open-athena/biomnibench-da-lupus-raw-counts/resolve/60044fab4ca09e55041729670deb55b4e6205c83/lupus-raw-counts.h5ad"
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    files = [
        x
        for x in HfApi().list_repo_tree(REPOSITORY, repo_type="dataset", revision=REVISION, recursive=True)
        if hasattr(x, "size")
    ]
    tasks = sorted({x.path.split("/")[0] for x in files if x.path.endswith("/task.toml")})
    assert len(tasks) == 50
    groups = {}
    for task in tasks:
        inputs = []
        prefix = task + "/environment/data/"
        for file in files:
            if not file.path.startswith(prefix) or file.path.endswith("GSE236581_counts.mtx"):
                continue
            item = {
                "path": file.path.removeprefix(prefix),
                "size": file.size,
                "sha256": file.lfs.sha256 if file.lfs else None,
                "blob_id": file.blob_id,
                "url": f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/" + quote(file.path),
            }
            if task.startswith("da-17-") and file.path.endswith(".h5ad"):
                item.update(LUPUS)
                item["blob_id"] = None
            inputs.append(item)
        identity = json.dumps([{k: x[k] for k in ("path", "size", "sha256", "blob_id")} for x in inputs], sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()
        group = groups.setdefault(key, {"tasks": [], "inputs": inputs})
        group["tasks"].append(task)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "input-groups.json").write_text(json.dumps(list(groups.values()), indent=2) + "\n")
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "input_groups": len(groups),
                "unique_input_bytes": sum(sum(x["size"] for x in g["inputs"]) for g in groups.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
