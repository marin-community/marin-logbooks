# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks against the exact pinned vLLM encoder and engine arguments."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

ROOT = Path(__file__).resolve().parents[1]


def probe(folder):
    spec = importlib.util.spec_from_file_location("hero_probe_" + folder, ROOT / folder / "tiny_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("folder", ["hero_shortconv_validation", "hero_shortconv_route_validation"])
def test_actual_encoder_roundtrips_trusted_observer_rpc(monkeypatch, folder):
    module = probe(folder)
    observer = module.install_replay_observer if folder == "hero_shortconv_validation" else module.install_observers
    payload = (observer, (), {})
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    with pytest.raises(TypeError):
        MsgpackEncoder().encode(payload)
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    decoded = MsgpackDecoder().decode(MsgpackEncoder().encode(payload))
    assert callable(decoded[0])
    assert decoded[0].__name__ == observer.__name__


@pytest.mark.parametrize("mode", ["NONE", "PIECEWISE"])
def test_actual_engine_arguments_select_hero_attention(mode):
    baseline = probe("hero_shortconv_validation")
    routes = probe("hero_shortconv_route_validation")
    configurations = [EngineArgs(**baseline.engine_kwargs(Path("local-model"), mode))]
    configurations.extend(routes.engine_arguments(Path("local-model"), mode, dp) for dp in (1, 4))
    for config in configurations:
        assert config.attention_backend == "FLASH_ATTN"
        assert not config.enforce_eager
        assert config.compilation_config.mode.name == "VLLM_COMPILE"
        assert config.compilation_config.cudagraph_mode.name == mode


@pytest.mark.parametrize("folder", ["hero_shortconv_validation", "hero_shortconv_route_validation"])
def test_worker_receipt_checks_actual_standard_attention_separately_from_state(folder):
    module = probe(folder)
    state = SimpleNamespace(
        backend=SimpleNamespace(get_name=lambda: "SHORT_CONV_STATE"),
        kv_cache_spec=object.__new__(MambaSpec),
        layer_names=["model.layers.0.short_conv"],
    )
    attention = SimpleNamespace(
        backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
        kv_cache_spec=object(),
        layer_names=["model.layers.0.self_attn"],
    )
    groups = module.attention_observation(SimpleNamespace(attn_groups=[[state], [attention]]))
    assert [group["kind"] for group in groups] == ["state", "attention"]
    attention.backend = SimpleNamespace(get_name=lambda: "FLASHINFER")
    with pytest.raises(ValueError, match="standard-attention backend differs"):
        module.attention_observation(SimpleNamespace(attn_groups=[[state], [attention]]))
