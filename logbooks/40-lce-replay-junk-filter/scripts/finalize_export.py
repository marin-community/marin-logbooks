"""Add tokenizer files and a generation config to an arm's HF-format export.

usage: finalize_export.py <arm>
"""

import json
import sys

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

TOKENIZER = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18"
EXPORTS = "s3://marin-us-east-02a/tmp/ttl=30d/lce-junk-filter/exports"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
GENERATION_CONFIG = {"eos_token_id": [128001, 128009], "bos_token_id": 128000}

configure_coreweave_s3()
destination = f"{EXPORTS}/lce40-arm-{sys.argv[1]}-20261113-step238/hf"
for name in TOKENIZER_FILES:
    with fsspec.open(f"{TOKENIZER}/{name}", "rb") as source:
        data = source.read()
    with fsspec.open(f"{destination}/{name}", "wb") as target:
        target.write(data)
    print(name, len(data))
with fsspec.open(f"{destination}/generation_config.json", "w") as target:
    json.dump(GENERATION_CONFIG, target)
filesystem, path = fsspec.core.url_to_fs(destination)
names = sorted(entry.rsplit("/", 1)[-1] for entry in filesystem.ls(path, detail=False))
print(len(names), [n for n in names if not n.endswith(".safetensors")])
print("shards", sum(n.endswith(".safetensors") for n in names))
