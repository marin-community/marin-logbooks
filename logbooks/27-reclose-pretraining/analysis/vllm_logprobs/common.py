"""Shared serving and probing helpers for the marin-community/marin#10400 think-channel probes.

Untracked scratch code. Every finished measurement is logged as a single ``RESULT <tag> <json>`` line so the
results can be streamed from the job logs while the job runs.
"""

import contextlib
import json
import logging
import os
import statistics
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import requests
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("think_channels")

START, END = "<|start_think|>", "<|end_think|>"
TOOL_START = "<|reserved_special_token_2|>"
BOS_ID, END_ID, EOT_ID = 128000, 128003, 128009
CANDIDATES = {"end_think": END_ID, "answer": 4320, "lt_slash": 524, "the": 279, "nl": 271}
SFT_REPO = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05"
SFT_REVISION = "8c76b22afc38f5e95360c83ca88008f00b5ae1e0"
SNOWBALL_VLLM_ARGS = ("--enable-expert-parallel", "--model-loader-extra-config", '{"distributed":true}')
WORKERS = 64


def result(tag: str, **fields) -> None:
    logger.info("RESULT %s %s", tag, json.dumps(fields, sort_keys=True))


def model_config(name: str, location: str, revision: str | None) -> ModelConfig:
    return ModelConfig(
        name=name,
        location=location,
        revision=revision,
        tokenizer="marin-community/marin-tokenizer",
        apply_chat_template=False,
        resource_hint=ResourceHint(gpu={"H100": 8}, memory="512g", disk="512g"),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            max_model_len=8192,
            max_num_seqs=64,
            auto_overrides=False,
            vllm_extra_args=(*SNOWBALL_VLLM_ARGS, "--gpu-memory-utilization", "0.85"),
        ),
    )


@dataclass(frozen=True)
class Client:
    base_url: str
    model: str
    end_bias: float = 0.0

    def complete(self, prompt: str | list[int], **kwargs) -> dict:
        body = {"model": self.model, "prompt": prompt, "top_p": 1.0, "skip_special_tokens": False, **kwargs}
        if self.end_bias:
            body["logit_bias"] = {str(END_ID): self.end_bias}
        for attempt in range(4):
            try:
                response = requests.post(f"{self.base_url}/completions", json=body, timeout=1800)
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt == 3:
                    raise
                logger.exception("request failed; retrying")
                time.sleep(10 * (attempt + 1))
        raise AssertionError("unreachable")

    def with_bias(self, end_bias: float) -> "Client":
        return Client(self.base_url, self.model, end_bias)


@contextlib.contextmanager
def serve(config: ModelConfig, gpu_count: int = 8) -> Iterator[Client]:
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=gpu_count)
    env = {k: os.environ[k] for k in ("HF_TOKEN",) if os.environ.get(k)}
    inference = inference_config_for_model(config, accelerator, env_vars=env, priority=job_pb2.PRIORITY_BAND_INHERIT)
    started = time.time()
    with remote_inference(inference) as session:
        logger.info("%s ready in %.0fs", config.name, time.time() - started)
        yield Client(session.model.endpoint.base_url.rstrip("/"), session.model.endpoint.model)


def pmap(fn, items, workers: int = WORKERS) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def group_of(ctx: dict) -> str:
    if ctx["domain"] == "swe":
        return "swe"
    return "other+tool" if ctx["tools"] else "other-plain"


def stage1(client: Client, contexts: list[dict], n_for, temperature: float = 0.6) -> list[dict]:
    """Sample the model's own reasoning up to its first <|end_think|>; keep only cleanly closed samples."""

    def one(ctx: dict) -> list[dict]:
        out = client.complete(
            ctx["prompt"], n=n_for(ctx), temperature=temperature, max_tokens=3072, stop_token_ids=[END_ID, EOT_ID]
        )
        rows = []
        for i, c in enumerate(out["choices"]):
            closed = c.get("stop_reason") == END_ID and START not in c["text"] and END not in c["text"]
            rows.append({**ctx, "sample": i, "reasoning": c["text"], "closed": closed})
        return rows

    return [r for rs in pmap(one, contexts) for r in rs]


def stage1_summary(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(group_of(r), []).append(r)
    return {
        g: {
            "closed": round(sum(r["closed"] for r in rs) / len(rs), 3),
            "median_chars": statistics.median(len(r["reasoning"]) for r in rs if r["closed"]) if any(r["closed"] for r in rs) else None,
            "n": len(rs),
        }
        for g, rs in sorted(by.items())
    }


def continuation_stats(texts: list[str]) -> dict:
    return {
        "n": len(texts),
        "special_end": sum(END in t for t in texts),
        "special_start": sum(START in t for t in texts),
        "text_close": sum(("</think>" in t) or ("</thinking>" in t) for t in texts),
        "tool_call": sum(TOOL_START in t for t in texts),
    }


def post_close(
    client: Client, reasonings: list[dict], closers: dict[str, str], bridges: tuple[str, ...], n: int, tag: str
) -> list[dict]:
    """For each (closer, bridge) cell, sample continuations and log one RESULT per group as each cell finishes."""
    rows = []
    for closer_name, closer in closers.items():
        for bridge in bridges:

            def one(r: dict, closer=closer, bridge=bridge) -> dict:
                prompt = r["prompt"] + r["reasoning"] + closer + bridge
                out = client.complete(prompt, n=n, temperature=1.0, max_tokens=256, stop_token_ids=[EOT_ID])
                texts = [c["text"] for c in out["choices"]]
                return {"group": group_of(r), "id": r["id"], "sample": r["sample"], "texts": texts}

            cell = pmap(one, reasonings)
            for g in sorted({c["group"] for c in cell}):
                texts = [t for c in cell if c["group"] == g for t in c["texts"]]
                stats = continuation_stats(texts)
                result(tag, closer=closer_name, bridge=bridge, group=g, **stats, rate_end=round(stats["special_end"] / stats["n"], 4))
            rows.extend({**c, "closer": closer_name, "bridge": bridge} for c in cell)
    return rows


def candidate_logprob(client: Client, prefix: list[int], token_id: int) -> float:
    out = client.complete([*prefix, token_id], max_tokens=1, temperature=0.0, prompt_logprobs=0)
    last = out["choices"][0]["prompt_logprobs"][-1]
    return float(last[str(token_id)]["logprob"])


def score_positions(client: Client, ids: list[int], positions: list[int]) -> list[dict]:
    """log P(candidate | ids[:i]) for every candidate at each position i."""
    jobs = [(i, name, tid) for i in positions for name, tid in CANDIDATES.items()]
    values = pmap(lambda j: candidate_logprob(client, ids[: j[0]], j[2]), jobs)
    by_pos: dict[int, dict] = {}
    for (i, name, _), v in zip(jobs, values, strict=True):
        by_pos.setdefault(i, {"pos": i, "next": ids[i] if i < len(ids) else None})[name] = v
    return [by_pos[i] for i in positions]


def alias_stats(points: list[dict]) -> dict:
    """Correlate log P(<|end_think|>) with the other candidates, raw and after removing the shared normalizer."""
    if len(points) < 10:
        return {"n": len(points)}
    m = {k: np.array([p[k] for p in points]) for k in CANDIDATES}
    out: dict = {"n": len(points)}
    for k in ("answer", "lt_slash", "the"):
        out[f"r_end_{k}"] = round(float(np.corrcoef(m["end_think"], m[k])[0, 1]), 3)
    # Logit differences cancel the softmax normalizer. The two sides use different reference tokens
    # (" the" vs "\n\n") so they share no term that would correlate them by construction.
    d_end = m["end_think"] - m["the"]
    for k in ("answer", "lt_slash"):
        out[f"r_logit_end_{k}"] = round(float(np.corrcoef(d_end, m[k] - m["nl"])[0, 1]), 3)
    slope, _ = np.polyfit(m["answer"] - m["nl"], d_end, 1)
    out["slope_logit_end_on_answer"] = round(float(slope), 3)
    return out
