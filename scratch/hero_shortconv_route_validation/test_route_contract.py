# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for forwarded prediction positions and routed-expert payloads."""

import numpy as np
import pytest
from hero_shortconv_route_contract import route_record


def _routes(rows):
    return np.array([[[i % 4, (i + 1) % 4], [(i + 2) % 4, (i + 3) % 4]] for i in range(rows)], dtype=np.int32)


def test_response_routes_align_to_prediction_positions_including_prompt_tail():
    captured = _routes(6)
    result = route_record(captured, prompt_length=3, response_length=4)
    assert result["forwarded_routes"] == captured.tolist()
    # Forward positions are p0,p1,p2,r0,r1,r2. The four generated tokens
    # were predicted by p2,r0,r1,r2; the final generated token has no forward.
    assert result["response_routes"] == captured[[2, 3, 4, 5]].tolist()


def test_route_geometry_rejects_an_extra_decode_row():
    with pytest.raises(ValueError, match="Forwarded route geometry"):
        route_record(_routes(7), prompt_length=3, response_length=4)


@pytest.mark.parametrize("corruption", ["missing", "expert-out-of-range", "duplicate-experts"])
def test_invalid_routed_payload_cannot_qualify(corruption):
    captured = _routes(6)
    if corruption == "missing":
        captured = None
    elif corruption == "expert-out-of-range":
        captured[2, 1, 0] = 4
    else:
        captured[2, 1, 0] = captured[2, 1, 1]
    with pytest.raises(ValueError):
        route_record(captured, prompt_length=3, response_length=4)
