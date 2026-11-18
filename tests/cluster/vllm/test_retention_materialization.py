# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure cold aligned replay and validate the supported cache contract."""

import asyncio
import dataclasses
import hashlib
import time

import pytest
from marin.testing.inference.backend_parity import distribution_difference
from marin.testing.inference.endpoint_checks import (
    EndpointClient,
    cached_tokens,
    check_cache,
    next_distribution,
    verify_deployment,
)
from marin.testing.inference.goldens import LookupStatus, canonical_bytes, lookup_reference


async def measure(endpoint):
    references = lookup_reference(endpoint.fixture)
    assert references.status == LookupStatus.FOUND
    source = max(references.cases, key=lambda case: len(case.tokens)).tokens
    report = {
        "experiment": "zero-retention-aligned-materialization",
        "status": "diagnostic",
        "runtime_source": endpoint.runtime_source,
        "manifest": endpoint.fixture.manifest.model_dump(),
        "policy": endpoint.fixture.policy.model_dump(),
        "policy_sha256": hashlib.sha256(canonical_bytes(endpoint.fixture.policy.model_dump(mode="json"))).hexdigest(),
        "started_unix": time.time(),
        "cases": [],
    }
    async with EndpointClient(endpoint.url, endpoint.model, endpoint.output.parent / "aligned-requests") as client:
        report["deployment"] = await verify_deployment(client, endpoint.fixture, endpoint.runtime_source)
        assert report["deployment"]["status"] == "passed"
        assert report["deployment"]["cache_config"]["prefix_cache_retention_interval"] == 0
        for length in (32, 127, 128, 129, 256, 2048):
            for kind in ("identical", "extension"):
                await client.reset_cache()
                prefix = source[:length]
                history = prefix if kind == "identical" else source[: length + 17]
                await client.complete(prefix, label=f"aligned-{length}-{kind}-seed")
                first = await client.complete(history, label=f"aligned-{length}-{kind}-first")
                second = await client.complete(history, label=f"aligned-{length}-{kind}-second")
                report["cases"].append(
                    {
                        "prompt_tokens": length,
                        "kind": kind,
                        "first_cached_tokens": cached_tokens(first),
                        "second_cached_tokens": cached_tokens(second),
                        "first_seconds": client.requests[-2]["seconds"],
                        "second_seconds": client.requests[-1]["seconds"],
                        "comparison": dataclasses.asdict(
                            distribution_difference(next_distribution(first), next_distribution(second))
                        ),
                    }
                )
                endpoint.output.write_bytes(canonical_bytes(report))

        report["cache_qualification"] = await check_cache(
            client,
            references,
            endpoint.fixture,
            report["deployment"]["cache_config"],
            report["deployment"]["model_config"]["max_model_len"],
        )
        report["requests"] = client.requests
    report["finished_unix"] = time.time()
    endpoint.output.write_bytes(canonical_bytes(report))
    return report


@pytest.mark.cluster
@pytest.mark.timeout(10800)
def test_measure_aligned_zero_retention(correctness_endpoint):
    report = asyncio.run(measure(correctness_endpoint))
    assert len(report["cases"]) == 12
    assert report["cache_qualification"]["status"] == "passed"
