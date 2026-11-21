"""Sample hinted GSM8K rollouts from served models and grade them, in the paper's result format.

Prompts come from PR #11905's builder (taskcompendium.importers.evalchemy.hinted.gsm8k_rows), grading from
its gsm8k_first_hash scorer, and sampling follows Hinted Scaling Laws (tatsu-lab/hinted-scaling-laws@c0490e53):
temperature 0.4, top_p 1.0, max_tokens 512, top_k 1000 when hinted and off unhinted, 64 rollouts at H=0 and
32 at H>0. One request per prompt asks for all its rollouts (vLLM `n`). Requests are spread round-robin over
one or more replicas of the same model.

Usage: python sample_hinted_gsm8k.py <model label> <url>[,<url>...] <out dir> [--exclude CSV ...] [--tag SUFFIX]
Writes <out dir>/<label><tag>.csv (model, method, hint, problem_id, passed) and the completions as jsonl.gz.
Prompts already present in the --exclude CSVs are skipped.
"""

import argparse
import asyncio
import csv
import gzip
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI, BadRequestError
from taskcompendium.importers.evalchemy.hinted import HintMethod, gsm8k_rows
from taskcompendium.importers.evalchemy.scorers.hinted import grade_gsm8k_first_hash

GSM8K_REPO = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
STOPS = ["Question:", "</s>", "<|im_end|>"]
ROLLOUTS = {HintMethod.NONE: 64, HintMethod.TRUNCATION: 32, HintMethod.MASKING: 32}
MAX_TOKENS = 512


def gsm8k_split(filename: str) -> list[dict]:
    path = hf_hub_download(GSM8K_REPO, filename, repo_type="dataset", revision=GSM8K_REVISION)
    return pq.read_table(path).to_pylist()


def build_requests() -> list[dict]:
    test = gsm8k_split("main/test-00000-of-00001.parquet")
    pool = gsm8k_split("main/train-00000-of-00001.parquet")
    requests = []
    for method in ROLLOUTS:
        for row in gsm8k_rows(test, pool, method, source_row_prefix="gsm8k"):
            requests.append(
                {
                    "method": method.value,
                    "hint": row["hint_level"],
                    "problem_id": row["problem_id"],
                    "prompt": row["prompt"],
                    "expected": row["grading"]["expected"],
                    "n": ROLLOUTS[method],
                    "top_k": -1 if method is HintMethod.NONE else 1000,
                }
            )
    return requests


def request_key(method: str, hint: float, problem_id: str) -> tuple[str, float, str]:
    return method, round(float(hint), 1), str(problem_id)


def done_keys(paths: list[Path]) -> set[tuple[str, float, str]]:
    keys = set()
    for path in paths:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("passed"):
                    keys.add(request_key(row["method"], row["hint"], row["problem_id"]))
    return keys


async def complete(client: AsyncOpenAI, model: str, request: dict) -> list[str]:
    max_tokens = MAX_TOKENS
    while True:
        try:
            response = await client.completions.create(
                model=model,
                prompt=request["prompt"],
                n=request["n"],
                temperature=0.4,
                top_p=1.0,
                max_tokens=max_tokens,
                stop=STOPS,
                extra_body={"top_k": request["top_k"]},
            )
            return [choice.text for choice in sorted(response.choices, key=lambda c: c.index)]
        except BadRequestError as error:
            # The paper clamps max_tokens to the 4096-token context minus the prompt.
            if "maximum context length" not in str(error) or max_tokens <= 64:
                raise
            max_tokens -= 64


async def run(label: str, urls: list[str], out_dir: Path, concurrency: int, exclude: list[Path], tag: str) -> None:
    endpoints = []
    for url in urls:
        client = AsyncOpenAI(base_url=url.rstrip("/") + "/v1", api_key="unused", timeout=1800, max_retries=5)
        model = (await client.models.list()).data[0].id
        endpoints.append((client, model, asyncio.Semaphore(concurrency)))
    skip = done_keys(exclude)
    requests = [r for r in build_requests() if request_key(r["method"], r["hint"], r["problem_id"]) not in skip]
    print(f"{label}: {len(requests)} prompts to sample over {len(endpoints)} replicas ({len(skip)} already done)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    start = time.time()
    with (
        open(out_dir / f"{label}{tag}.csv", "w", newline="", buffering=1) as results_file,
        gzip.open(out_dir / f"{label}{tag}.completions.jsonl.gz", "wt") as completions_file,
    ):
        writer = csv.writer(results_file)
        writer.writerow(["model", "method", "hint", "problem_id", "passed"])

        async def one(index: int, request: dict) -> None:
            nonlocal done
            client, model, semaphore = endpoints[index % len(endpoints)]
            async with semaphore:
                texts = await complete(client, model, request)
            rewards = [grade_gsm8k_first_hash(request["expected"], text).reward for text in texts]
            passed = "".join("1" if reward >= 1.0 else "0" for reward in rewards)
            writer.writerow([label, request["method"], request["hint"], request["problem_id"], passed])
            record = {key: request[key] for key in ("method", "hint", "problem_id")}
            completions_file.write(json.dumps({**record, "completions": texts}) + "\n")
            done += 1
            if done % 250 == 0:
                print(f"{label}: {done}/{len(requests)} prompts, {done / (time.time() - start):.1f} prompts/s", flush=True)

        await asyncio.gather(*(one(index, request) for index, request in enumerate(requests)))
    print(f"{label}: finished {len(requests)} prompts in {time.time() - start:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("urls", help="Comma-separated capability base URLs of replicas of one model")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=48, help="In-flight requests per replica")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="CSVs whose prompts are already done")
    parser.add_argument("--tag", default="", help="Output filename suffix, e.g. .part1")
    args = parser.parse_args()
    asyncio.run(run(args.label, args.urls.split(","), args.out_dir, args.concurrency, args.exclude, args.tag))


if __name__ == "__main__":
    main()
