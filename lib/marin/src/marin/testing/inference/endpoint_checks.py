# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP checks. Callers own deployment and the measurement window."""

import asyncio
import dataclasses
import gzip
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np
from prometheus_client.parser import text_string_to_metric_families

from marin.testing.inference.backend_parity import TokenScore
from marin.testing.inference.goldens import (
    ErrorLimits,
    LookupStatus,
    ModelFixture,
    ReferenceLookup,
    canonical_bytes,
    history_digest,
    lookup_reference,
)

TOP_K = 50
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=900)
RANK_HEADER = "X-data-parallel-rank"


@dataclass(frozen=True)
class TokenDistribution:
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]

    @classmethod
    def from_response(cls, entries: dict) -> "TokenDistribution":
        scores = sorted(
            (
                TokenScore(token_id=int(token.removeprefix("token_id:")), logprob=float(value))
                for token, value in entries.items()
            ),
            key=lambda score: (-score.logprob, score.token_id),
        )
        if not scores or any(not math.isfinite(score.logprob) or score.logprob > 0 for score in scores):
            raise ValueError("The endpoint returned missing, positive or non-finite log probabilities")
        return cls(tuple(score.token_id for score in scores), tuple(score.logprob for score in scores))


def distribution_difference(first: TokenDistribution, second: TokenDistribution) -> dict:
    """Bound top-k probability error without treating absent tokens as known scores.

    A token outside a returned top-k has probability between zero and its kth
    probability. The upper bound is conservative; it is not full-vocabulary KL.
    """
    a = dict(zip(first.token_ids, map(math.exp, first.logprobs), strict=True))
    b = dict(zip(second.token_ids, map(math.exp, second.logprobs), strict=True))
    errors = []
    lower = []
    for token in a.keys() | b.keys():
        if token in a and token in b:
            lower.append(abs(a[token] - b[token]))
            errors.append(lower[-1])
        else:
            known, floor = (a[token], min(b.values())) if token in a else (b[token], min(a.values()))
            lower.append(max(0.0, known - floor))
            errors.append(max(known, abs(known - floor)))
    return {
        "max_probability_error_lower_bound": max(lower),
        "max_probability_error": max(errors),
        "shared_tokens": len(a.keys() & b.keys()),
        "same_greedy_token": first.token_ids[0] == second.token_ids[0],
    }


def mismatch_summary(rows: list[dict], limits: ErrorLimits) -> dict:
    values = np.asarray([row["max_probability_error"] for row in rows])
    if not len(values):
        return {"status": "pending", "reason": "No matching score positions were checked", "positions": 0}
    statistics = {
        "maximum": float(values.max()),
        "p99": float(np.quantile(values, 0.99)),
        "p90": float(np.quantile(values, 0.9)),
        "fraction_above_0075": float(np.mean(values > 0.075)),
    }
    return {
        "status": "passed" if all(value <= getattr(limits, name) for name, value in statistics.items()) else "failed",
        "positions": len(values),
        "statistics": statistics,
        "limits": limits.model_dump(),
        "worst": sorted(rows, key=lambda row: row["max_probability_error"], reverse=True)[:10],
    }


@dataclass
class EndpointClient:
    base_url: str
    model: str
    output: Path
    session: aiohttp.ClientSession = field(init=False)
    requests: list[dict] = field(default_factory=list, init=False)

    async def __aenter__(self) -> "EndpointClient":
        self.output.mkdir(parents=True, exist_ok=True)
        self.session = aiohttp.ClientSession(timeout=HTTP_TIMEOUT, connector=aiohttp.TCPConnector(limit=0))
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.session.close()

    @property
    def root(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1")

    async def get(self, path: str) -> dict:
        async with self.session.get(f"{self.root}/{path}") as response:
            response.raise_for_status()
            return await response.json()

    async def reset_cache(self) -> None:
        async with self.session.post(f"{self.root}/reset_prefix_cache") as response:
            response.raise_for_status()

    async def metrics(self) -> str:
        async with self.session.get(f"{self.root}/metrics") as response:
            response.raise_for_status()
            return await response.text()

    async def complete(
        self,
        tokens: tuple[int, ...],
        *,
        label: str,
        count: int = 1,
        prompt_scores: bool = False,
        rank: int = 0,
    ) -> dict:
        body = {
            "model": self.model,
            "prompt": tokens,
            "add_special_tokens": False,
            "max_tokens": count,
            "temperature": 0,
            "ignore_eos": True,
            "logprobs": TOP_K,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        if prompt_scores:
            body["prompt_logprobs"] = TOP_K
        started = time.time()
        async with self.session.post(
            f"{self.root}/v1/completions",
            json=body,
            headers={RANK_HEADER: str(rank), "X-Request-Id": label},
        ) as response:
            raw = await response.read()
            # Keep failed HTTP responses too. Request IDs and histories are recorded separately.
            path = self.output / f"{len(self.requests):05d}-{label}.json.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(raw)
            self.requests.append(
                {
                    "label": label,
                    "history_sha256": history_digest(tokens),
                    "prompt_tokens": len(tokens),
                    "rank": rank,
                    "requested_tokens": count,
                    "prompt_scores": prompt_scores,
                    "http_status": response.status,
                    "response": path.name,
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "seconds": time.time() - started,
                }
            )
            response.raise_for_status()
        result = json.loads(raw)
        (choice,) = result["choices"]
        if choice.get("prompt_token_ids") != list(tokens):
            raise ValueError(f"The endpoint changed the exact input tokens: {label}")
        if len(choice["token_ids"]) != count:
            raise ValueError(f"The endpoint returned an incomplete continuation: {label}")
        for token, top in zip(choice["token_ids"], choice["logprobs"]["top_logprobs"], strict=True):
            distribution = TokenDistribution.from_response(top)
            if int(token) not in distribution.token_ids or int(token) != distribution.token_ids[0]:
                # Equal maximum scores may have a different tie order.
                scores = dict(zip(distribution.token_ids, distribution.logprobs, strict=True))
                if scores.get(int(token)) != max(scores.values()):
                    raise ValueError(f"The greedy token lacks its finite maximum score: {label}")
        return result


def next_distribution(response: dict) -> TokenDistribution:
    return TokenDistribution.from_response(response["choices"][0]["logprobs"]["top_logprobs"][0])


def cached_tokens(response: dict) -> int:
    return int(response["usage"]["prompt_tokens_details"]["cached_tokens"])


def metric_values(text: str, name: str) -> dict[str, float]:
    return {
        sample.labels["engine"]: float(sample.value)
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name and "engine" in sample.labels
    }


async def verify_deployment(client: EndpointClient, fixture: ModelFixture, runtime_source: str) -> dict:
    """Check server-reported identities instead of trusting a caller's labels."""
    information = await client.get("server_info?config_format=json")
    config = information["vllm_config"]
    model = config["model_config"]
    version = await client.get("version")
    identity = fixture.model
    expected = {
        "model": identity.weights,
        "tokenizer": identity.tokenizer,
        "tokenizer_revision": identity.tokenizer_revision,
    }
    if not identity.weights.startswith(("s3://", "gs://")):
        expected["revision"] = identity.revision
    mismatches = {
        name: {"actual": model.get(name), "expected": value}
        for name, value in expected.items()
        if model.get(name) != value
    }
    if runtime_source[:12] not in version["version"]:
        mismatches["runtime"] = {"actual": version["version"], "expected": runtime_source}
    if str(model["dtype"]).removeprefix("torch.") != fixture.policy.dtype:
        mismatches["dtype"] = {"actual": str(model["dtype"]), "expected": fixture.policy.dtype}
    return {
        "status": "failed" if mismatches else "passed",
        "mismatches": mismatches,
        "version": version,
        "model_config": model,
        "scheduler_config": config["scheduler_config"],
        "cache_config": config["cache_config"],
        "parallel_config": config["parallel_config"],
        "compilation_config": config["compilation_config"],
    }


async def check_reference(client: EndpointClient, references: ReferenceLookup, fixture: ModelFixture) -> dict:
    expected = {(row.case_id, row.position): row for row in references.scores}
    groups: dict[str, list[dict]] = {name: [] for name in fixture.policy.reference}
    observations = []
    for case in references.cases:
        # A context includes its sampled output token; score the final prediction separately
        # when an existing 262K reference fills the entire configured context.
        response = await client.complete(case.tokens, label=f"reference-{case.id}", prompt_scores=True)
        (choice,) = response["choices"]
        positions = [*choice["prompt_logprobs"][1:], choice["logprobs"]["top_logprobs"][0]]
        if len(positions) != len(case.tokens):
            raise ValueError(f"Missing prompt log probabilities: {case.id}")
        for position in case.positions:
            row = expected[(case.id, position)]
            actual = TokenDistribution.from_response(positions[position])
            reference = TokenDistribution(row.token_ids, row.logprobs)
            mismatch = distribution_difference(reference, actual) | {"case": case.id, "position": position}
            groups[case.group].append(mismatch)
            observations.append(mismatch | {"actual": dataclasses.asdict(actual)})
    summaries = {name: mismatch_summary(rows, fixture.policy.reference[name]) for name, rows in groups.items()}
    return {
        "status": "passed" if all(value["status"] == "passed" for value in summaries.values()) else "failed",
        "groups": summaries,
        "observations": observations,
    }


async def check_batch(client: EndpointClient, references: ReferenceLookup, fixture: ModelFixture) -> dict:
    cases = sorted((case for case in references.cases if len(case.tokens) <= 4096), key=lambda case: len(case.tokens))
    selected = cases[-fixture.policy.batch_size :]
    if len(selected) < fixture.policy.batch_size:
        raise ValueError("The suite has too few distinct cases for the engine batch")
    alone = []
    for index, case in enumerate(selected):
        await client.reset_cache()
        alone.append(await client.complete(case.tokens, label=f"alone-{index}", count=fixture.policy.decode_tokens))
    await client.reset_cache()
    before = await client.metrics()
    tasks = [
        asyncio.create_task(client.complete(case.tokens, label=f"batch-{index}", count=fixture.policy.decode_tokens))
        for index, case in enumerate(selected)
    ]
    running_peak = 0.0
    evidence = []
    while not all(task.done() for task in tasks):
        metrics = await client.metrics()
        running = metric_values(metrics, "vllm:num_requests_running").get("0", 0)
        running_peak = max(running_peak, running)
        if running >= 2:
            evidence.append({"unix": time.time(), "engine": 0, "running": running})
        await asyncio.sleep(0.05)
    batched = await asyncio.gather(*tasks)
    after = await client.metrics()
    (client.output / "batch-before.prom").write_text(before)
    (client.output / "batch-after.prom").write_text(after)
    rows = [
        distribution_difference(next_distribution(a), next_distribution(b)) | {"case": case.id}
        for case, a, b in zip(selected, alone, batched, strict=True)
    ]
    summary = mismatch_summary(rows, fixture.policy.batch)
    if running_peak < 2:
        summary |= {"status": "failed", "reason": "No simultaneous running requests were observed on engine 0"}
    return summary | {"engine": 0, "running_peak": running_peak, "batch_evidence": evidence, "observations": rows}


async def check_cache(client: EndpointClient, references: ReferenceLookup, fixture: ModelFixture) -> dict:
    source = max(references.cases, key=lambda case: len(case.tokens)).tokens
    if len(set(source)) < 3:
        raise ValueError("Cache probes require varied token histories")
    same_split, different_split = [], []
    evidence = []
    for length in fixture.policy.boundary_lengths:
        prefix = source[:length]
        branch = prefix + source[length : length + 17]
        if len(branch) <= len(prefix):
            raise ValueError(f"The prompt suite cannot reach cache boundary {length}")
        await client.reset_cache()
        fresh = await client.complete(branch, label=f"cache-{length}-fresh")
        if cached_tokens(fresh) != 0:
            raise ValueError("Cache reset did not produce a fresh recomputation")
        await client.reset_cache()
        await client.complete(prefix, label=f"cache-{length}-seed")
        warm = await client.complete(branch, label=f"cache-{length}-warm")
        alternate = prefix + source[length + 31 : length + 48]
        await client.complete(alternate, label=f"cache-{length}-branch")
        repeated = await client.complete(branch, label=f"cache-{length}-repeat")
        hits = [cached_tokens(warm), cached_tokens(repeated)]
        if min(hits) <= 0:
            raise ValueError(f"No prefix cache hit occurred at boundary {length}: {hits}")
        different_split.append(
            distribution_difference(next_distribution(fresh), next_distribution(warm)) | {"boundary": length}
        )
        # Compare equal reuse splits separately. A larger second hit changes the prefill split.
        comparison = distribution_difference(next_distribution(warm), next_distribution(repeated)) | {"boundary": length}
        (same_split if hits[0] == hits[1] else different_split).append(comparison)
        evidence.append(
            {
                "boundary": length,
                "fresh_cached_tokens": cached_tokens(fresh),
                "warm_cached_tokens": hits[0],
                "repeat_cached_tokens": hits[1],
            }
        )

        # The seed decode crosses a block/state boundary. Reuse its exact generated history,
        # then branch and recompute it from scratch to expose stale recurrent/conv state.
        await client.reset_cache()
        decoded = await client.complete(prefix, label=f"cache-{length}-decode-seed", count=fixture.policy.decode_tokens)
        history = prefix + tuple(decoded["choices"][0]["token_ids"]) + source[length : length + 3]
        reused = await client.complete(history, label=f"cache-{length}-decode-reuse")
        if cached_tokens(reused) <= 0:
            raise ValueError(f"Decode-created history did not reuse cached state at {length}")
        await client.reset_cache()
        recomputed = await client.complete(history, label=f"cache-{length}-decode-fresh")
        different_split.append(
            distribution_difference(next_distribution(reused), next_distribution(recomputed))
            | {"boundary": length, "seed": "decode"}
        )
        evidence[-1]["decode_cached_tokens"] = cached_tokens(reused)
    # An exact repeat of a warmed history supplies a matched-split control even if prefix
    # seeding and branching fill different numbers of blocks on this model.
    await client.complete(branch, label="cache-matched-seed")
    a = await client.complete(branch, label="cache-matched-first")
    b = await client.complete(branch, label="cache-matched-second")
    if cached_tokens(a) != cached_tokens(b) or cached_tokens(a) <= 0:
        raise ValueError("No matched nonempty cache split was exercised")
    same_split.append(distribution_difference(next_distribution(a), next_distribution(b)) | {"case": "matched-repeat"})
    summaries = {
        "same_split": mismatch_summary(same_split, fixture.policy.same_split),
        "different_split": mismatch_summary(different_split, fixture.policy.different_split),
    }
    return {
        "status": "passed" if all(row["status"] == "passed" for row in summaries.values()) else "failed",
        **summaries,
        "cache_evidence": evidence,
        "observations": {"same_split": same_split, "different_split": different_split},
    }


async def check_decode(client: EndpointClient, references: ReferenceLookup, fixture: ModelFixture) -> dict:
    cases = sorted((case for case in references.cases if len(case.tokens) < 4096), key=lambda case: len(case.tokens))
    selected = [cases[0], cases[-1]]
    differences = []
    for index, case in enumerate(selected):
        await client.reset_cache()
        decoded = await client.complete(case.tokens, label=f"decode-{index}", count=fixture.policy.decode_tokens)
        choice = decoded["choices"][0]
        history = case.tokens + tuple(choice["token_ids"])
        await client.reset_cache()
        rescored = await client.complete(history, label=f"decode-rescore-{index}", prompt_scores=True)
        prompt_scores = rescored["choices"][0]["prompt_logprobs"]
        for position, (token, top) in enumerate(
            zip(choice["token_ids"], choice["logprobs"]["top_logprobs"], strict=True)
        ):
            decode_distribution = TokenDistribution.from_response(top)
            prefill_distribution = TokenDistribution.from_response(prompt_scores[len(case.tokens) + position])
            a = dict(zip(decode_distribution.token_ids, decode_distribution.logprobs, strict=True))[token]
            b = dict(zip(prefill_distribution.token_ids, prefill_distribution.logprobs, strict=True))[token]
            differences.append(
                {
                    "case": case.id,
                    "position": position,
                    "token_id": token,
                    "decode_logprob": a,
                    "prefill_logprob": b,
                    "absolute_logprob_error": abs(a - b),
                    "max_probability_error": abs(math.exp(a) - math.exp(b)),
                }
            )
    return mismatch_summary(differences, fixture.policy.decode) | {"observations": differences}


async def check_endpoint(
    base_url: str,
    model: str,
    fixture: ModelFixture,
    runtime_source: str,
    output: Path,
) -> dict:
    """Run the shared suite and retain evidence, including failures and missing coverage."""
    references = lookup_reference(fixture)
    report: dict = {
        "schema_version": 1,
        "status": "pending",
        "model": fixture.model.model_dump(),
        "runtime_source": runtime_source,
        "manifest": fixture.manifest.model_dump(),
        "policy": fixture.policy.model_dump(),
        "started_unix": time.time(),
        "reference_lookup": {"status": references.status, "reason": references.reason},
        "checks": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        output.write_bytes(canonical_bytes(report))

    if references.status != LookupStatus.FOUND:
        save()
        return report
    async with EndpointClient(base_url, model, output.parent / f"{output.stem}-requests") as client:
        stages: tuple[tuple[str, Callable[[], Awaitable[dict]]], ...] = (
            ("deployment", lambda: verify_deployment(client, fixture, runtime_source)),
            ("reference", lambda: check_reference(client, references, fixture)),
            ("batch", lambda: check_batch(client, references, fixture)),
            ("cache", lambda: check_cache(client, references, fixture)),
            ("decode", lambda: check_decode(client, references, fixture)),
        )
        for name, stage in stages:
            started = time.time()
            try:
                result = await stage()
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            report["checks"][name] = result | {"seconds": time.time() - started}
            report["requests"] = client.requests
            save()
            if name == "deployment" and result["status"] != "passed":
                break
    report["status"] = (
        "passed"
        if len(report["checks"]) == 5 and all(row["status"] == "passed" for row in report["checks"].values())
        else "failed"
    )
    report["finished_unix"] = time.time()
    save()
    return report
