# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prove errors are attributed at copy dispatch, not from a prior pointer sighting."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_probe():
    spec = importlib.util.spec_from_file_location("save_on_cpu_attribution_probe", Path(__file__).with_name("probe.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors():
    destination = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        data_ptr=lambda: 123456,
        numel=lambda: 4096 * 6144,
        element_size=lambda: 2,
        is_pinned=lambda: True,
    )
    source = SimpleNamespace(
        is_cuda=True,
        shape=(4096, 1, 6144),
        stride=lambda: (6144, 6144, 1),
        storage_offset=lambda: 0,
        dtype=torch.bfloat16,
    )
    return destination, source


def test_exact_copy_exception_is_recorded_and_rethrown(tmp_path, monkeypatch):
    probe = load_probe()
    failure = torch.AcceleratorError("CUDA error: invalid argument")

    def failing_copy(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(probe.torch.ops.aten.copy_, "default", failing_copy)
    monkeypatch.setattr(probe, "stats", lambda: {"allocated_bytes.current": 72 * 1024**2})
    receipt = {"copies": [], "prime_pointer": 123456}
    path = tmp_path / "receipt.json"
    observer = probe.CopyEvidence(receipt, path, "heterogeneous_autograd_save_on_cpu")
    with pytest.raises(torch.AcceleratorError) as caught:
        observer.__torch_dispatch__(failing_copy, (), tensors())
    assert caught.value is failure
    saved = json.loads(path.read_text())
    assert saved["failed_copy"]["requested_bytes"] == 48 * 1024**2
    assert saved["failed_copy"]["destination_pointer"] == saved["prime_pointer"]
    assert probe.expected_copy_failure("threshold_only", saved, caught.value)
    assert not probe.expected_copy_failure("baseline", saved, caught.value)


def test_later_invalid_argument_after_successful_copy_is_not_expected(tmp_path, monkeypatch):
    probe = load_probe()
    success = object()

    def successful_copy(*_args, **_kwargs):
        return success

    monkeypatch.setattr(probe.torch.ops.aten.copy_, "default", successful_copy)
    monkeypatch.setattr(probe, "stats", lambda: {"allocated_bytes.current": 72 * 1024**2})
    receipt = {"copies": [], "prime_pointer": 123456}
    observer = probe.CopyEvidence(receipt, tmp_path / "receipt.json", "heterogeneous_autograd_save_on_cpu")
    assert observer.__torch_dispatch__(successful_copy, (), tensors()) is success
    assert receipt["copies"][0]["destination_pointer"] == receipt["prime_pointer"]
    later_error = torch.AcceleratorError("CUDA error: invalid argument during backward")
    assert not probe.expected_copy_failure("threshold_only", receipt, later_error)
    assert "failed_copy" not in receipt


def test_noncopy_dispatch_failure_does_not_create_copy_evidence(tmp_path):
    probe = load_probe()
    failure = torch.AcceleratorError("CUDA error: invalid argument")

    def backward_operation():
        raise failure

    receipt = {"copies": [], "prime_pointer": 123456}
    observer = probe.CopyEvidence(receipt, tmp_path / "receipt.json", "heterogeneous_autograd_save_on_cpu")
    with pytest.raises(torch.AcceleratorError) as caught:
        observer.__torch_dispatch__(backward_operation, ())
    assert caught.value is failure
    assert not probe.expected_copy_failure("threshold_only", receipt, caught.value)
    assert "failed_copy" not in receipt
