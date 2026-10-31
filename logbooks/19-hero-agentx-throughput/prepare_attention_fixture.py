# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a small checkpoint with Hero's native attention geometry on CPU.

Embedding, expert and gated-norm tensors stay from the retained synthetic
fixture. New deterministic attention tensors use48query/12stored/6global KV
heads of width128. This is a component diagnostic, not the Hero checkpoint.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = args.input / "model.safetensors"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == args.input_model_sha256
    config = json.loads((args.input / "config.json").read_text())
    assert config["hidden_dim"] == 128 and config["num_layers"] == 4
    assert config["local_kv_heads"] == 4 and config["global_kv_heads"] == 2
    state = load_file(str(model), device="cpu")
    changed = []
    for index in range(4):
        prefix = f"model.layers.{index}.self_attn"
        for name, shape, scale in (
            ("q_proj.weight", (6144, 128), 0.02),
            ("k_proj.weight", (1536, 128), 0.02),
            ("v_proj.weight", (1536, 128), 0.02),
            ("o_proj.weight", (128, 6144), 0.02),
            ("attn_gate.weight", (48, 128), 0.2),
            ("sconv_k.weight", (4, 1536), 0.1),
        ):
            key = prefix + "." + name
            assert key in state
            value = torch.randn(shape) * scale
            if name == "sconv_k.weight":
                value[0] += 1
            state[key] = value
            changed.append(key)
    config.update(
        num_heads=48,
        num_attention_heads=48,
        num_kv_heads=12,
        num_key_value_heads=12,
        local_kv_heads=12,
        global_kv_heads=6,
        head_dim=128,
        max_seq_len=262144,
        max_position_embeddings=262144,
        sliding_window=2048,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for path in args.input.iterdir():
        if path.is_file() and path.name not in ("model.safetensors", "config.json"):
            shutil.copyfile(path, args.output / path.name)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(state, str(args.output / "model.safetensors"), metadata={"format": "pt"})
    receipt = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_model_sha256": args.input_model_sha256,
        "changed_tensors": changed,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in args.output.iterdir()
            if path.is_file()
        },
        "scope": "Synthetic four-layer component with actual Hero attention geometry; no full-model qualification.",
    }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
