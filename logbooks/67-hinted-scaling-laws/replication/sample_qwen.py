"""Sample hinted AIME rollouts from a served Qwen chat model and grade them, in the paper's result format.

Replicates Hinted Scaling Laws (tatsu-lab/hinted-scaling-laws@c0490e53) AIME cells:
- prompts: the paper's released prompts verbatim (suzeva/hinted-benchmarks@3d3d42c4, aime/data.parquet),
  sent as one user message through the model's own chat template (src/hinted/query_engine/workers.py:42-47);
- rollouts: 10 per (method, H) cell; hinted rollout i uses reference solution i (solution_index), the
  unhinted cell asks for 10 samples of its single prompt;
- sampling: the model's ModelConfig in src/hinted/models/__init__.py, max_tokens clamped to
  context_limit - prompt_tokens - 256 (workers.py:57-71), no stop strings;
- grading: the paper's aime_grader.py, copied verbatim as paper_aime_grader.py.

Usage: python sample_qwen.py <model label> <capability base URL> <out dir> [--concurrency N]
Writes <out dir>/<label>.csv (model, method, hint, problem_id, passed) and <label>.completions.jsonl.gz.
"""

import argparse
import asyncio
import csv
import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from paper_aime_grader import grade_aime_response
from transformers import AutoTokenizer

HERE = Path(__file__).parent
PROMPTS = HERE / "aime_prompts.parquet"
ROLLOUTS_PER_CELL = 10
TOKEN_SAFETY_MARGIN = 256
METHODS = ("no_hint", "mask", "truncate")


@dataclass(frozen=True)
class Sampling:
    hf_path: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    requested_max_tokens: int = 32000
    context_limit: int = 32768


# src/hinted/models/__init__.py at c0490e53.
SAMPLING = {
    "Qwen2.5-1.5B-Instruct": Sampling("Qwen/Qwen2.5-1.5B-Instruct", 0.7, 0.8, 20, 1.1),
    "Qwen2.5-3B-Instruct": Sampling("Qwen/Qwen2.5-3B-Instruct", 0.7, 0.8, 20, 1.05),
}


def build_requests() -> list[dict]:
    table = pd.read_parquet(PROMPTS)
    table = table[table["method"].isin(METHODS)]
    requests = []
    for row in table.itertuples():
        requests.append(
            {
                "method": row.method,
                "hint": round(float(row.hint_level), 4),
                "problem_id": row.problem_id,
                "solution_index": None if pd.isna(row.solution_index) else int(row.solution_index),
                "prompt": row.prompt,
                "answer": row.answer,
                "n": ROLLOUTS_PER_CELL if row.method == "no_hint" else 1,
            }
        )
    hinted = [r for r in requests if r["method"] != "no_hint"]
    assert all(r["solution_index"] is not None and r["solution_index"] < ROLLOUTS_PER_CELL for r in hinted)
    return requests


async def run(label: str, base_url: str, out_dir: Path, concurrency: int) -> None:
    sampling = SAMPLING[label]
    tokenizer = AutoTokenizer.from_pretrained(sampling.hf_path)
    client = AsyncOpenAI(base_url=base_url.rstrip("/") + "/v1", api_key="unused", timeout=3600, max_retries=5)
    model = (await client.models.list()).data[0].id
    requests = build_requests()
    out_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    outcomes: dict[tuple, dict[int, str]] = {}
    done = 0
    start = time.time()

    with gzip.open(out_dir / f"{label}.completions.jsonl.gz", "wt") as completions_file:

        async def one(request: dict) -> None:
            nonlocal done
            messages = [{"role": "user", "content": request["prompt"]}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            max_tokens = min(sampling.requested_max_tokens, sampling.context_limit - prompt_tokens - TOKEN_SAFETY_MARGIN)
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=request["n"],
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    max_tokens=max_tokens,
                    extra_body={"top_k": sampling.top_k, "repetition_penalty": sampling.repetition_penalty},
                )
            choices = sorted(response.choices, key=lambda c: c.index)
            texts = [choice.message.content or "" for choice in choices]
            bits = ["1" if grade_aime_response(text, request["answer"]).is_correct else "0" for text in texts]
            key = (request["method"], request["hint"], request["problem_id"])
            slots = outcomes.setdefault(key, {})
            if request["n"] > 1:
                slots.update(enumerate(bits))
            else:
                slots[request["solution_index"]] = bits[0]
            completions_file.write(
                json.dumps(
                    {
                        **{k: request[k] for k in ("method", "hint", "problem_id", "solution_index")},
                        "completions": texts,
                        "finish_reasons": [choice.finish_reason for choice in choices],
                        "passed": "".join(bits),
                    }
                )
                + "\n"
            )
            done += 1
            if done % 500 == 0:
                print(f"{label}: {done}/{len(requests)} requests, {done / (time.time() - start):.1f}/s", flush=True)

        await asyncio.gather(*(one(request) for request in requests))

    with open(out_dir / f"{label}.csv", "w", newline="") as results_file:
        writer = csv.writer(results_file)
        writer.writerow(["model", "method", "hint", "problem_id", "passed"])
        for (method, hint, problem_id), slots in sorted(outcomes.items()):
            assert sorted(slots) == list(range(ROLLOUTS_PER_CELL)), (method, hint, problem_id, sorted(slots))
            writer.writerow([label, method, hint, problem_id, "".join(slots[i] for i in range(ROLLOUTS_PER_CELL))])
    print(f"{label}: finished {len(requests)} requests in {time.time() - start:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=sorted(SAMPLING))
    parser.add_argument("base_url")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(run(args.label, args.base_url, args.out_dir, args.concurrency))


if __name__ == "__main__":
    main()
