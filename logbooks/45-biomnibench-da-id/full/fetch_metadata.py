# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

repo = "phylobio/BiomniBench-DA"
rev = "e1c8ca5e11a620087bc48d97888eb69176a1f235"
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
root = parser.parse_args().output_dir
root.mkdir(parents=True, exist_ok=True)
files = [
    x for x in HfApi().list_repo_tree(repo, repo_type="dataset", revision=rev, recursive=True) if hasattr(x, "size")
]
manifest = [{"path": x.path, "size": x.size, "sha256": x.lfs.sha256 if x.lfs else None} for x in files]
(root / "source-files.json").write_text(json.dumps(manifest, indent=2) + "\n")
selected = [x for x in manifest if x["path"].split("/")[0].startswith("da-") and "/environment/data/" not in x["path"]]
assert all(x["size"] < 1000000 for x in selected)


def fetch(x):
    p = root / "source" / x["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(repo, x["path"], repo_type="dataset", revision=rev))
    p.write_bytes(source.read_bytes())


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, selected))
print(
    json.dumps(
        {
            "metadata_files": len(selected),
            "tasks": len(list((root / "source").glob("da-*"))),
            "metadata_bytes": sum(x["size"] for x in selected),
        }
    )
)
