# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical Hero controls; these measurements do not qualify a profile."""

import asyncio
import dataclasses
import json
import time
from pathlib import Path

import pytest
from marin.testing.inference.backend_parity import distribution_difference
from marin.testing.inference.endpoint_checks import (
    EndpointClient,
    cached_tokens,
    next_distribution,
    verify_deployment,
)
from marin.testing.inference.goldens import LookupStatus, ModelFixture, canonical_bytes, lookup_reference


async def measure_repeat_controls(
    url: str,
    model: str,
    fixture: ModelFixture,
    runtime_source: str,
    output: Path,
    lengths: tuple[int, ...],
) -> dict:
    references = lookup_reference(fixture)
    assert references.status == LookupStatus.FOUND
    source = max(references.cases, key=lambda case: len(case.tokens)).tokens
    report: dict = {
        "experiment": "hero-fresh-and-cached-repeat-controls",
        "status": "diagnostic",
        "model": fixture.model.model_dump(),
        "runtime_source": runtime_source,
        "manifest": fixture.manifest.model_dump(),
        "policy": fixture.policy.model_dump(),
        "started_unix": time.time(),
        "boundaries": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    async with EndpointClient(url, model, output.parent / "repeat-requests") as client:
        report["deployment"] = await verify_deployment(client, fixture, runtime_source)
        assert report["deployment"]["status"] == "passed"
        for length in lengths:
            prefix = source[:length]
            branch = source[: length + 17]
            alternate = prefix + source[length + 31 : length + 48]
            responses = {}
            for repeat in range(3):
                await client.reset_cache()
                label = f"fresh-{repeat}"
                responses[label] = await client.complete(branch, label=f"control-{length}-{label}")
                assert cached_tokens(responses[label]) == 0
            await client.reset_cache()
            await client.complete(prefix, label=f"control-{length}-seed-prefix")
            for repeat in range(3):
                label = f"warm-{repeat}"
                responses[label] = await client.complete(branch, label=f"control-{length}-{label}")
            await client.complete(alternate, label=f"control-{length}-alternate")
            for repeat in range(3):
                label = f"after-branch-{repeat}"
                responses[label] = await client.complete(branch, label=f"control-{length}-{label}")
            await client.reset_cache()
            await client.complete(branch, label=f"control-{length}-seed-full-branch")
            for repeat in range(2):
                label = f"full-seed-{repeat}"
                responses[label] = await client.complete(branch, label=f"control-{length}-{label}")
            pairs = [
                ("fresh-repeat", "fresh-0", "fresh-1"),
                ("fresh-repeat", "fresh-1", "fresh-2"),
                ("immediate-cache-repeat", "warm-0", "warm-1"),
                ("immediate-cache-repeat", "warm-1", "warm-2"),
                ("branch-intervention", "warm-2", "after-branch-0"),
                ("post-branch-repeat", "after-branch-0", "after-branch-1"),
                ("post-branch-repeat", "after-branch-1", "after-branch-2"),
                ("seed-shape", "warm-0", "full-seed-0"),
                ("full-seed-repeat", "full-seed-0", "full-seed-1"),
            ]
            comparisons = [
                {
                    "kind": kind,
                    "first": first,
                    "second": second,
                    "first_cached_tokens": cached_tokens(responses[first]),
                    "second_cached_tokens": cached_tokens(responses[second]),
                    **dataclasses.asdict(
                        distribution_difference(
                            next_distribution(responses[first]), next_distribution(responses[second])
                        )
                    ),
                }
                for kind, first, second in pairs
            ]
            report["boundaries"].append({"boundary": length, "comparisons": comparisons})
            report["requests"] = client.requests
            output.write_bytes(canonical_bytes(report))
    report["finished_unix"] = time.time()
    output.write_bytes(canonical_bytes(report))
    return json.loads(output.read_bytes())


@pytest.mark.cluster
@pytest.mark.timeout(10800)
def test_measure_repeat_controls(correctness_endpoint):
    endpoint = correctness_endpoint
    report = asyncio.run(
        measure_repeat_controls(
            endpoint.url,
            endpoint.model,
            endpoint.fixture,
            endpoint.runtime_source,
            endpoint.output,
            (127, 255, 2047),
        )
    )
    assert len(report["boundaries"]) == 3
