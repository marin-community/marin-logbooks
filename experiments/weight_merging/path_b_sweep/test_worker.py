"""CPU fixtures for the existing merge CLI and its readback accounting."""

import json
import sys

import pytest
import safetensors.torch
import torch

from experiments.weight_merging.merge import main as merge_main
from experiments.weight_merging.path_b_sweep.worker import changed_counts, tensor_hash


@pytest.mark.parametrize("alpha", [0.25, 0.49, 0.51, 0.75])
def test_real_merge_cli_keeps_fp32_and_exact_protected_rows(tmp_path, monkeypatch, alpha):
    parents = []
    anchor = {
        "router.bias": torch.tensor([-0.1234567, 0.43579, -0.0], dtype=torch.float32),
        "head": torch.tensor([[1.0, 1.0078125], [3.0, 5.0]], dtype=torch.bfloat16),
        "body": torch.tensor([1.0, 2.0, 4.0, -3.0], dtype=torch.bfloat16),
    }
    donor = {
        **anchor,
        "head": anchor["head"] + 8,
        "body": torch.tensor([1.0078125, 4.0, 0.0, 7.0], dtype=torch.bfloat16),
    }
    for label, tensors in (("anchor", anchor), ("donor", donor)):
        root = tmp_path / label
        root.mkdir()
        safetensors.torch.save_file(tensors, root / "input.safetensors")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {n: "input.safetensors" for n in tensors}})
        )
        (root / "config.json").write_text('{"model_type":"fixture"}')
        parents.append({"path": str(root), "revision": label})
    out = tmp_path / "output"
    recipe = {
        "anchor": parents[0],
        "donors": [parents[1]],
        "output": str(out),
        "parameters": {"method": "average", "coefficients": [alpha], "density": 1, "scale": 1, "seed": 0},
        "preserve_rows": {"router.bias": [0, 1, 2], "head": [1]},
        "tensor_coefficients": {},
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    monkeypatch.setattr(sys, "argv", ["merge", "--recipe", str(path), "--threads", "1", "--code-revision", "f" * 40])
    merge_main()
    index = json.loads((out / "model.safetensors.index.json").read_text())
    tensors = {n: safetensors.torch.load_file(out / shard)[n] for n, shard in index["weight_map"].items()}
    assert tensors["router.bias"].dtype == torch.float32
    assert tensor_hash(tensors["router.bias"]) == tensor_hash(anchor["router.bias"])
    assert tensor_hash(tensors["head"][1]) == tensor_hash(anchor["head"][1])
    expected = (anchor["body"].float() * (1 - alpha) + donor["body"].float() * alpha).to(torch.bfloat16)
    torch.testing.assert_close(tensors["body"], expected, rtol=0, atol=0)
    assert (out / "merge-manifest.json").is_file()
    assert (out / "config.json").read_bytes() == (tmp_path / "anchor/config.json").read_bytes()
    counts = changed_counts(anchor["body"], donor["body"], tensors["body"])
    assert counts["parents_differ"] == 4
    assert counts["changed_vs_A3"] == int((expected != anchor["body"]).sum())
    assert counts["changed_vs_A8a"] == int((expected != donor["body"]).sum())
    assert counts["entries"] == 4
    for parent, value in (("A3", anchor["body"]), ("A8a", donor["body"])):
        expected_l2 = float(torch.linalg.vector_norm((expected.float() - value.float()).double()))
        assert counts["delta_l2_vs_" + parent] == expected_l2
