# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.identity_path_b.replay_serve import service_config, worker_command
from experiments.post_training.identity_t2 import serve as t2


def test_replay_reuses_frozen_numerical_serve_with_new_namespace_and_generous_lifetime():
    old = t2.service_config()
    new = service_config("pathb-replay-fixture")
    assert new.model == old.model
    assert new.engine == old.engine
    assert new.iris == old.iris
    assert new.endpoint_name == "/serve/pathb-replay-fixture"
    assert new.timeout_hours == 12
    assert new.controller_proxy_timeout_seconds == 900
    command = worker_command("pathb-replay-fixture")
    assert "identity_path_b.replay_serve" in command[-1]
    assert "identity_t2.serve worker" not in command[-1]
    with pytest.raises(ValueError, match="fresh"):
        service_config("cd16t2-generation-r3")
