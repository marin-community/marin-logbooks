# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Levanter's Qwen3.6-35B-A3B state-dict names and shapes with the Hub checkpoint's safetensors headers.

Usage: download `config.json` and `model.safetensors.index.json` (saved as `index.json`) from the Hub revision
below into a directory, then run
`uv run --package marin-levanter python -I logbooks/39-open-swe-outcome/qwen36_hub_shapes.py <directory>`.
Only safetensors headers are fetched, through HTTP range requests.
"""

import json
import struct
import sys
import urllib.request

import haliax as hax
import jax
import numpy as np
from haliax.state_dict import flatten_modules_for_export, to_state_dict
from levanter.models.qwen3_5_moe import (
    Qwen35MoeConfig,
    Qwen35MoeLMHeadModel,
    _text_state_dict,
)
from transformers import AutoConfig

REPO = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
hf_dir = sys.argv[1]


def fetch(name: str, byte_range: str) -> bytes:
    request = urllib.request.Request(
        f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", headers={"Range": f"bytes={byte_range}"}
    )
    with urllib.request.urlopen(request) as response:
        return response.read()


with open(f"{hf_dir}/index.json") as index_file:
    index = json.load(index_file)
shards = sorted(set(index["weight_map"].values()))
hub: dict[str, tuple[tuple[int, ...], str]] = {}
for shard in shards:
    (header_size,) = struct.unpack("<Q", fetch(shard, "0-7"))
    header = json.loads(fetch(shard, f"8-{8 + header_size - 1}"))
    for key, meta in header.items():
        if key != "__metadata__":
            hub[key] = (tuple(meta["shape"]), meta["dtype"])
print("hub tensors", len(hub), "params", sum(int(np.prod(s)) for s, _ in hub.values()))

hub_text = _text_state_dict(hub)
text_params = sum(int(np.prod(s)) for s, _ in hub_text.values())
print("text tensors", len(hub_text), "params", text_params, "dtypes", {d for _, d in hub_text.values()})

hf_config = AutoConfig.from_pretrained(hf_dir)
config = Qwen35MoeConfig.from_hf_config(hf_config)
assert config == Qwen35MoeConfig(), (config, Qwen35MoeConfig())
vocab_size = hf_config.text_config.vocab_size
Vocab = hax.Axis("vocab", vocab_size)
shapes = jax.eval_shape(
    lambda: to_state_dict(
        flatten_modules_for_export(Qwen35MoeLMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(0)))
    )
)
levanter = {key: tuple(value.shape) for key, value in shapes.items() if not key.endswith(".inference")}
print("levanter tensors", len(levanter), "params", sum(int(np.prod(s)) for s in levanter.values()))
print("total_trainable_params", config.total_trainable_params(vocab_size))

print("only levanter", sorted(set(levanter) - set(hub_text))[:10])
print("only hub", sorted(set(hub_text) - set(levanter))[:10])
mismatched = [(k, levanter[k], hub_text[k][0]) for k in levanter if k in hub_text and levanter[k] != hub_text[k][0]]
print("shape mismatches", mismatched[:10])
assert set(levanter) == set(hub_text) and not mismatched
assert config.total_trainable_params(vocab_size) == text_params
print("OK")
