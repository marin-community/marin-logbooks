# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.identity_t2.capture import echoes_capability, generated_ids


def test_raw_ids_remain_authoritative_when_message_disagrees():
    raw = [128005, 123, 456, 128011, 128009]
    response = {"choices": [{"token_ids": raw, "message": {"content": "A different parsed answer"}}]}
    assert generated_ids(response) == raw
    with pytest.raises(ValueError, match="token_ids"):
        generated_ids({"choices": [{"message": {"content": "Do not reconstruct this fallback"}}]})


@pytest.mark.parametrize("ids", [[], [True], [-1], [1.0]])
def test_malformed_wire_ids_cannot_become_training_targets(ids):
    with pytest.raises(ValueError, match="token_ids"):
        generated_ids({"choices": [{"token_ids": ids}]})


def test_transport_echo_detection_catches_capability_paths_and_bare_tokens():
    token = "dummy-capability-" + "x" * 48
    endpoint = f"https://example.invalid/proxy/t/{token}/serve/cd16t2-generation-r1/v1"
    assert echoes_capability(endpoint, endpoint)
    assert echoes_capability(f"upstream rejected /proxy/t/{token}/serve/request", endpoint)
    assert echoes_capability(f"invalid token {token}", endpoint)
    assert not echoes_capability("Marin Snowball, created by Open Athena.", endpoint)
