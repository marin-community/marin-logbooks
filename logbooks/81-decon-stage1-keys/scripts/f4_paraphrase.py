# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F4 generate: an LLM rewrites one item per eval (paraphrase), and translates a third of them, for planted
paraphrase and translation copies (review point 1). Runs as an Iris job: the GLM relay is reachable only from
inside the cluster. Items with a passage are skipped (their copy needs the passage too).

    python logbooks/81-decon-stage1-keys/scripts/f4_paraphrase.py --items <g1v7>/items --keys <keys.parquet> \
        --output <dir> --relay /muchanem/glm53-relay-h
"""
import argparse
import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from marin.datakit.decon_tables import parquet_files, read_frame, write_frame
from openai import OpenAI
from rigging.log_setup import configure_logging

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, resolve_glm_base_url

logger = logging.getLogger(__name__)
MODEL = "deepseek-v4.1-flash"
LANGUAGES = ("Chinese", "Spanish", "German", "French", "Russian", "Japanese")
TRANSLATED_SHARE = 1 / 3
MIN_QUESTION_CHARS = 40
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
PARAPHRASE = (
    "Rewrite the exam question below so that it asks exactly the same thing and has the same answer, but "
    "with different wording and sentence structure throughout. Avoid keeping more than three consecutive "
    "words of the original where you can. Keep every number, formula, name, code identifier and condition. "
    "Rewrite each answer choice the same way, keeping its meaning and its order. Reply with one JSON object "
    'and nothing else: {"question": "...", "choices": ["...", ...]} (an empty list when there are no choices).'
)
TRANSLATE = (
    "Translate the exam question below and each of its answer choices into {language}. Keep every number, "
    "formula, name and code identifier as it is, and keep the choices in their order. Reply with one JSON "
    'object and nothing else: {{"question": "...", "choices": ["...", ...]}} (an empty list when there are '
    "no choices)."
)


def rewrite(client: OpenAI, item: dict, instruction: str) -> dict | None:
    task = json.dumps({"question": item["question"], "choices": list(item["choices"] or [])}, ensure_ascii=False)
    for _ in range(3):
        reply = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": instruction}, {"role": "user", "content": task}],
            max_tokens=8192,
        )
        match = _OBJECT.search(reply.choices[0].message.content or "")
        try:
            value = json.loads(match.group()) if match else None
        except json.JSONDecodeError:
            value = None
        if value and isinstance(value.get("question"), str) and isinstance(value.get("choices"), list):
            if len(value["choices"]) == len(item["choices"] or []):
                return value
    return None


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--relay", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    keyed = set(read_frame(args.keys, ["id", "key1_hash"]).query("key1_hash != 0")["id"])
    rng = random.Random(5)
    chosen = []
    for path in parquet_files(args.items):
        rows = [
            row
            for row in read_frame(path, ["id", "eval", "question", "choices", "answer", "passage", "reason"]).to_dict(
                "records"
            )
            if row["id"] in keyed
            and not row["reason"]
            and not row["passage"]
            and len(row["question"] or "") >= MIN_QUESTION_CHARS
        ]
        if rows:
            chosen.append(rng.choice(rows))
    tasks = [(item, "paraphrase", PARAPHRASE) for item in chosen]
    for item in rng.sample(chosen, int(len(chosen) * TRANSLATED_SHARE)):
        language = rng.choice(LANGUAGES)
        tasks.append((item, f"translation:{language}", TRANSLATE.format(language=language)))
    client = OpenAI(base_url=resolve_glm_base_url(args.relay), api_key=os.environ[GLM_BULK_TOKEN_ENV], timeout=600)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda task: rewrite(client, task[0], task[2]), tasks))
    rows = [
        {
            "item": item["id"],
            "eval": item["eval"],
            "kind": kind,
            "question": value["question"],
            "choices": value["choices"],
        }
        for (item, kind, _), value in zip(tasks, results)
        if value
    ]
    logger.info("%d of %d rewrites valid", len(rows), len(tasks))
    write_frame(pd.DataFrame(rows), f"{args.output}/rewrites.parquet")


if __name__ == "__main__":
    main()
