# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import hashlib
import json
import math
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from aiohttp import web
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.testing.inference.endpoint_checks import check_endpoint
from marin.testing.inference.goldens import (
    AcceptancePolicy,
    Artifact,
    ErrorLimits,
    LookupStatus,
    ModelFixture,
    ModelIdentity,
    ReferenceCase,
    ReferenceManifest,
    ScoreRow,
    canonical_bytes,
    history_digest,
    lookup_reference,
)

from experiments.evaluation.knotmeter.artifacts import write_record
from experiments.evaluation.knotmeter.config import Check, load_profile, resolved_series
from experiments.evaluation.knotmeter.report import report_campaign
from experiments.evaluation.knotmeter.run import run_checks

RUNTIME = "0ac9dcb157ee86ccb447bc717259159c41ad456a"
LOGPROBS = {token: math.log(0.5 if token == 1 else 0.3 if token == 2 else 0.2 / 14) for token in range(16)}


def artifact(path: Path, value: object) -> Artifact:
    raw = canonical_bytes(value)
    path.write_bytes(raw)
    return Artifact(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def reference_fixture(tmp_path) -> ModelFixture:
    model_path = tmp_path / "model"
    model_path.mkdir()
    config = b'{"dtype":"bfloat16"}'
    (model_path / "config.json").write_bytes(config)
    identity = ModelIdentity(
        weights=str(model_path),
        revision="test-revision",
        config_sha256=hashlib.sha256(config).hexdigest(),
        tokenizer="test-tokenizer",
        tokenizer_revision="tokenizer-revision",
        eos_token_id=0,
    )
    cases = [
        ReferenceCase(id=f"case-{i}", tokens=tuple((n % 13) + 3 for n in range(length)), positions=(length - 1,))
        for i, length in enumerate([5, 7, 10, 40])
    ]
    prompts = artifact(tmp_path / "cases.json", [case.model_dump(mode="json") for case in cases])
    scores = artifact(
        tmp_path / "scores.json",
        [
            ScoreRow(
                case_id=case.id,
                position=case.positions[0],
                token_ids=tuple(LOGPROBS),
                logprobs=tuple(LOGPROBS.values()),
                history_sha256=history_digest(case.tokens),
            ).model_dump(mode="json")
            for case in cases
        ],
    )
    manifest = ReferenceManifest(
        model=identity,
        suite_id=prompts.sha256,
        cases=prompts,
        scores=scores,
        reference_source="independent-fixture",
        reference_commit="fixture-revision",
        reference_config={},
        lineage={},
    )
    pointer = artifact(tmp_path / "manifest.json", manifest.model_dump(mode="json"))
    limits = ErrorLimits(maximum=0.01, p99=0.01, p90=0.01, fraction_above_0075=0)
    return ModelFixture(
        model=identity,
        manifest=pointer,
        policy=AcceptancePolicy(
            id="fixture-policy",
            provisional=True,
            rationale="Protocol fixture",
            evidence=(),
            hardware=("HTTP test GPU",),
            reference={"representative": limits},
            same_split=limits,
            different_split=limits,
            batch=limits,
            decode=limits,
            boundary_lengths=(7, 8, 9),
        ),
    )


@dataclass
class HttpEngine:
    mode: str
    identity: ModelIdentity
    prefixes: set[tuple[int, ...]] = field(default_factory=set)
    active_batch: int = 0
    observed_batch: asyncio.Event = field(default_factory=asyncio.Event)

    async def complete(self, request: web.Request) -> web.Response:
        body = await request.json()
        tokens = tuple(body["prompt"])
        count = body["max_tokens"]
        assert request.headers["X-data-parallel-rank"] == "0"
        cached = max((len(prefix) for prefix in self.prefixes if tokens[: len(prefix)] == prefix), default=0)
        is_batch = request.headers["X-Request-Id"].startswith("batch-")
        if is_batch:
            self.active_batch += 1
            # The real metrics request releases these responses after observing the
            # engine. This avoids a timer race in the HTTP fixture.
            await self.observed_batch.wait()
        probabilities = dict(LOGPROBS)
        if self.mode == "corrupt-cache" and cached:
            probabilities[1] = math.log(0.1)
            probabilities[2] = math.log(0.7)
        top = {f"token_id:{token}": value for token, value in probabilities.items()}
        winner = max(probabilities, key=probabilities.get)
        generated = [winner] * count
        self.prefixes.update(tuple((*tokens, *generated)[:length]) for length in range(4, len(tokens) + count, 4))
        if is_batch:
            self.active_batch -= 1
        choice = {"token_ids": generated, "prompt_token_ids": list(tokens), "logprobs": {"top_logprobs": [top] * count}}
        if body.get("prompt_logprobs"):
            prompt_top = {
                token: {"logprob": value, "rank": index + 1} for index, (token, value) in enumerate(top.items())
            }
            choice["prompt_logprobs"] = [None, *[prompt_top for _ in tokens[1:]]]
        return web.json_response(
            {
                "choices": [choice],
                "usage": {"prompt_tokens_details": {"cached_tokens": 0 if self.mode == "no-cache" else cached}},
            }
        )

    async def metrics(self, _request):
        running = self.active_batch if self.mode != "no-batch" else min(1, self.active_batch)
        if self.active_batch >= 4:
            self.observed_batch.set()
        return web.Response(text=f'vllm:num_requests_running{{engine="0"}} {running}\n')

    async def reset(self, _request):
        self.prefixes.clear()
        return web.json_response(True)


@asynccontextmanager
async def http_endpoint(engine: HttpEngine):
    app = web.Application()
    app.router.add_post("/v1/completions", engine.complete)
    app.router.add_get("/metrics", engine.metrics)
    app.router.add_post("/reset_prefix_cache", engine.reset)

    async def info(_request):
        return web.json_response(
            {
                "system_env": {"nvidia_gpu_models": "HTTP test GPU"},
                "vllm_config": {
                    "model_config": {
                        "model": engine.identity.weights,
                        "revision": "test-revision",
                        "tokenizer": "test-tokenizer",
                        "tokenizer_revision": "tokenizer-revision",
                        "dtype": "bfloat16",
                        "override_generation_config": {"eos_token_id": 0},
                    },
                    "scheduler_config": {},
                    "cache_config": {"enable_prefix_caching": True, "cache_dtype": "auto"},
                    "parallel_config": {},
                    "compilation_config": {"cudagraph_mode": "PIECEWISE"},
                },
            }
        )

    async def version(_request):
        return web.json_response({"version": "0.0.0+marin.0ac9dcb157ee"})

    app.router.add_get("/server_info", info)
    app.router.add_get("/version", version)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0]
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,failed_check", [("correct", None), ("no-cache", "cache"), ("no-batch", "batch"), ("corrupt-cache", "cache")]
)
async def test_http_suite_requires_numerics_and_exercised_paths(tmp_path, reference_fixture, mode, failed_check):
    async with http_endpoint(HttpEngine(mode, reference_fixture.model)) as url:
        path = tmp_path / "report.json"
        report = await check_endpoint(url, "test-model", reference_fixture, RUNTIME, path)
    assert json.loads(path.read_bytes()) == report
    assert report["status"] == ("passed" if failed_check is None else "failed")
    if failed_check:
        assert report["checks"][failed_check]["status"] == "failed"
    else:
        assert report["checks"]["batch"]["running_peak"] >= 2
        assert report["checks"]["cache"]["cache_evidence"][0]["warm_cached_tokens"] > 0
        assert report["checks"]["decode"]["positions"] == 64
    assert report["requests"]
    assert all((path.parent / "report-requests" / row["response"]).is_file() for row in report["requests"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["correct", "no-cache"])
async def test_cli_exit_agrees_with_retained_report(tmp_path, reference_fixture, mode):
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(reference_fixture.model_dump_json())
    output = tmp_path / "cli-report.json"
    async with http_endpoint(HttpEngine(mode, reference_fixture.model)) as url:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "marin.testing.inference.check_endpoint",
            "--fixture",
            str(fixture_path),
            "--url",
            url,
            "--model",
            "test-model",
            "--runtime-source",
            RUNTIME,
            "--output",
            str(output),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
    assert process.returncode == (0 if mode == "correct" else 1), stderr.decode()
    report = json.loads(output.read_bytes())
    assert json.loads(stdout)["status"] == report["status"] == ("passed" if mode == "correct" else "failed")


def test_lookup_distinguishes_wrong_checkpoint_missing_and_modified_scores(tmp_path, reference_fixture):
    assert lookup_reference(reference_fixture).status == LookupStatus.FOUND
    wrong = reference_fixture.model_copy(
        update={"model": reference_fixture.model.model_copy(update={"revision": "other-checkpoint"})}
    )
    assert lookup_reference(wrong).status == LookupStatus.MISMATCHED
    (tmp_path / "scores.json").write_text("[]")
    assert lookup_reference(reference_fixture).status == LookupStatus.MISMATCHED
    (tmp_path / "scores.json").unlink()
    assert lookup_reference(reference_fixture).status == LookupStatus.MISSING


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["correct", "no-cache", "missing-reference"])
async def test_perfmax_retains_real_check_report_and_blocks_unqualified_recommendations(
    tmp_path, reference_fixture, mode
):
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(reference_fixture.model_dump_json())
    if mode == "missing-reference":
        (tmp_path / "scores.json").unlink()
    profile = load_profile(Path("snowball-h100"), [])
    profile = profile.model_copy(
        update={
            "model": profile.model.model_copy(
                update={"repo": "test-model", "revision": reference_fixture.model.revision}
            ),
            "checks": [Check(name="shared-policy", fixture=str(fixture_path), timeout=60)],
        }
    )
    async with http_endpoint(HttpEngine(mode, reference_fixture.model)) as url:
        checks = await asyncio.to_thread(run_checks, profile, url, tmp_path, RUNTIME)
    report_bytes = (tmp_path / "check-0.json").read_bytes()
    assert checks[0]["report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert checks[0]["status"] == json.loads(report_bytes)["status"]

    # AgentX validity and allocation are synthetic here. The numerical input comes
    # from the real HTTP/CLI workflow above; only a pass may produce a decision.
    campaign = tmp_path / "campaign"
    trial_uri = campaign / "trials" / "candidate"
    result = {
        "trial_id": "candidate",
        "trial_uri": str(trial_uri),
        "job_state": "succeeded",
        "job_succeeded": True,
        "cost_complete": True,
        "gpu_seconds": 0,
        "attempts": [],
        "manifest": {
            "series": resolved_series(profile),
            "vllm_release": dataclasses.asdict(VLLM_GPU_RELEASE),
            "diagnostic": False,
            "workload": {"recipe": {"fixture": True}},
        },
        "measurement": {
            "checks": checks,
            "validity": {"protocol_valid": True, "submission_valid": True, "gang_valid": True},
            "aggregate": {
                "request_metrics": {
                    "throughput": {"per_gpu": {"output_tput_tps": 100}},
                    "latency": {"ttft": {"p90": 1}, "e2el": {"p90": 10}},
                }
            },
        },
    }
    write_record(str(trial_uri / "result.json"), result)
    if mode == "correct":
        report_campaign(str(campaign), recommendation="candidate")
        assert (campaign / "recommended.yaml").is_file()
        assert json.loads((campaign / "decision.json").read_bytes())["provisional"] is True
    else:
        with pytest.raises(ValueError):
            report_campaign(str(campaign), recommendation="candidate")
        assert not (campaign / "recommended.yaml").exists()
