# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import types

import pytest
from marin.testing.inference.endpoint_checks import check_endpoint
from marin.testing.inference.goldens import ErrorLimits

from tests.inference.test_endpoint_checks import RUNTIME, HttpEngine, http_endpoint

pytest_plugins = ["tests.inference.test_endpoint_checks"]


@pytest.mark.asyncio
async def test_original_and_corrected_checks_differ_on_branch_state_alias(tmp_path, reference_fixture):
    fixture = reference_fixture.model_copy(
        update={
            "policy": reference_fixture.policy.model_copy(
                update={
                    "different_split": ErrorLimits(maximum=1, p99=1, p90=1, fraction_above_0075=1),
                    "same_split": ErrorLimits(maximum=0, p99=0, p90=0, fraction_above_0075=0),
                }
            )
        }
    )
    source = subprocess.check_output(
        [
            "git",
            "show",
            "12055994a685e6fdff27c6ece463f4998c247e3e:lib/marin/src/marin/testing/inference/endpoint_checks.py",
        ],
        text=True,
    )
    old = types.ModuleType("old_endpoint_checks_12055994")
    sys.modules[old.__name__] = old
    exec(compile(source, "endpoint_checks_12055994.py", "exec"), old.__dict__)
    outcomes = {}
    for label, run in [("old", old.check_endpoint), ("corrected", check_endpoint)]:
        async with http_endpoint(HttpEngine("branch-corrupt", fixture.model)) as url:
            report = await run(url, "test-model", fixture, RUNTIME, tmp_path / label / "report.json")
        outcomes[label] = {
            "status": report["status"],
            "cache_status": report["checks"]["cache"]["status"],
            "same_split": report["checks"]["cache"]["same_split"],
            "different_split": report["checks"]["cache"]["different_split"],
        }
    print(json.dumps(outcomes))
    assert outcomes["old"]["status"] == "passed"
    assert outcomes["corrected"]["status"] == "failed"
    assert outcomes["corrected"]["different_split"]["status"] == "passed"
    assert outcomes["corrected"]["same_split"]["status"] == "failed"
