# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Campaign control: require actual storage and cache checks before the full suite."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from marin.testing.inference.endpoint_checks import EndpointClient, check_cache, check_endpoint, verify_deployment
from marin.testing.inference.goldens import LookupStatus, lookup_reference


def save_memory_peak(output: Path):
    record = {
        "scope": "Task cgroup v2 at the end of numerical checks, before serving teardown and artifact upload",
        "peak_bytes": int(Path("/sys/fs/cgroup/memory.peak").read_text()),
        "limit_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text()),
        "observed_unix": time.time(),
    }
    output.write_text(json.dumps(record, indent=2) + "\n")


async def cache_screen(endpoint):
    references = lookup_reference(endpoint.fixture)
    assert references.status == LookupStatus.FOUND
    output = endpoint.output.with_name("cache-screen.json")
    report = {
        "scope": "Cache-focused campaign diagnostic; the full shared suite follows only after a pass",
        "runtime_source": endpoint.runtime_source,
        "model": endpoint.fixture.model.model_dump(mode="json"),
        "manifest": endpoint.fixture.manifest.model_dump(mode="json"),
        "policy": endpoint.fixture.policy.model_dump(mode="json"),
        "status": "failed",
        "started_unix": time.time(),
        "checks": {},
    }
    async with EndpointClient(endpoint.url, endpoint.model, output.with_name("cache-screen-requests")) as client:
        try:
            deployment = await verify_deployment(client, endpoint.fixture, endpoint.runtime_source)
            report["checks"]["deployment"] = deployment
            assert deployment["status"] == "passed", deployment
            report["checks"]["cache"] = await check_cache(
                client,
                references,
                endpoint.fixture,
                deployment["cache_config"],
                deployment["model_config"]["max_model_len"],
            )
            report["status"] = report["checks"]["cache"]["status"]
        finally:
            report["requests"] = client.requests
            report["finished_unix"] = time.time()
            output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def inspect_weights(endpoint, request):
    plan = json.loads(Path(request.config.getoption("--correctness-plan")).read_text())
    profile_path = endpoint.output.with_name("loaded-storage-profile.json")
    profile_path.write_text(json.dumps(plan["series"]["profile"]))
    inspection = next(item["command"] for item in plan["series"]["profile"]["checks"] if item["kind"] == "serving")
    assert inspection[0] == "python"
    inspector = Path(__file__).resolve().parents[3] / inspection[1]
    return subprocess.run(
        [sys.executable, str(inspector)],
        env=os.environ
        | {
            "KNOTMETER_PROFILE": str(profile_path),
            "KNOTMETER_SERVER_URL": endpoint.url,
            "KNOTMETER_RUNTIME_SOURCE": endpoint.runtime_source,
            "KNOTMETER_CHECK_OUTPUT": str(endpoint.output.with_name("loaded-storage.json")),
        },
        check=False,
    )


@pytest.mark.cluster
@pytest.mark.timeout(10800)
def test_hero_canonical_storage_cache_then_shared_suite(correctness_endpoint, request):
    endpoint = correctness_endpoint
    request.addfinalizer(lambda: save_memory_peak(endpoint.output.with_name("memory.json")))
    inspect_weights(endpoint, request).check_returncode()
    focused = asyncio.run(cache_screen(endpoint))
    assert focused["status"] == "passed", {
        "report": str(endpoint.output.with_name("cache-screen.json")),
        "checks": {name: value["status"] for name, value in focused["checks"].items()},
    }
    report = asyncio.run(
        check_endpoint(endpoint.url, endpoint.model, endpoint.fixture, endpoint.runtime_source, endpoint.output)
    )
    assert report["status"] == "passed", {
        "report": str(endpoint.output),
        "checks": {name: value["status"] for name, value in report["checks"].items()},
    }


@pytest.mark.cluster
@pytest.mark.timeout(10800)
def test_grug_storage_and_shared_suite(correctness_endpoint, request):
    endpoint = correctness_endpoint
    request.addfinalizer(lambda: save_memory_peak(endpoint.output.with_name("memory.json")))
    inspection = inspect_weights(endpoint, request)
    report = asyncio.run(
        check_endpoint(endpoint.url, endpoint.model, endpoint.fixture, endpoint.runtime_source, endpoint.output)
    )
    inspection.check_returncode()
    assert report["status"] == "passed", {
        "report": str(endpoint.output),
        "checks": {name: value["status"] for name, value in report["checks"].items()},
    }
