"""Derive the two arm configs for #40 from the reproduction's production launch config.

Both arms keep the production schedule (4,750 updates) and stop after 238, so they train the head of
the same schedule the marin#10799 run did. They differ only in the long replay caches.

usage: make_arm_configs.py <production config.yaml> <configs dir> <control|filtered> \
           <init checkpoint> <output prefix> [<filtered replay prefix> <filtered catalog path> <summary.jsonl>]
"""

import json
import re
import sys
from pathlib import Path

STOP_STEP = 238
KEEP_EVERY = 119
PRODUCTION_RUN = "snowball-sft-production-128h100-20261024b"
PRODUCTION_PREFIX = f"s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/{PRODUCTION_RUN}"
PRODUCTION_INIT = "s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/snowball-sft-hf-import-20261024e/step-0"
PRODUCTION_CATALOG = "s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/full-mixture/2026.10.24.2/cache_catalog.json"
REPLAY_LONG = re.compile(
    r"s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/replay/(c\d\dq\d-train)/2026\.10\.24\.1/gt_64k"
)


def replace_once(text: str, old: str, new: str, count: int = 1) -> str:
    found = text.count(old)
    if found != count:
        raise ValueError(f"expected {count} of {old!r}, found {found}")
    return text.replace(old, new)


def main() -> None:
    production, configs_dir, arm, init, output_prefix = sys.argv[1:6]
    run = f"lce40-arm-{arm}-20261113"
    text = Path(production).read_text()
    text = text.replace(PRODUCTION_RUN, run) if PRODUCTION_PREFIX not in output_prefix else text
    text = text.replace(f"s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/{run}", f"{output_prefix}/{run}")
    text = replace_once(text, f"initialize_from: {PRODUCTION_INIT}", f"initialize_from: {init}")
    text = replace_once(text, "stop_after_steps: null", f"stop_after_steps: {STOP_STEP}")
    text = replace_once(text, "      keep: null\n", f"      keep:\n      - every: {KEEP_EVERY}\n")
    text = replace_once(text, "max_retries_failure: 0\n", "max_retries_failure: 3\n")
    text = replace_once(text, "max_task_failures: 1\n", "max_task_failures: 10\n")
    text = replace_once(text, "      entity: null\n", "      entity: marin-community\n")
    text = replace_once(text, "      project: sft-pipe-repro\n", "      project: lce40-replay-filter\n")
    if arm == "filtered":
        filtered_prefix, catalog, summary_path = sys.argv[6:9]
        built = {
            row["cell"]
            for row in map(json.loads, Path(summary_path).read_text().splitlines())
            if row["kept_documents"] > 0
        }
        swapped = set()

        def swap(match: re.Match) -> str:
            cell = match.group(1)
            if cell not in built:
                return match.group(0)
            swapped.add(cell)
            return f"{filtered_prefix}/{cell}/gt_64k"

        text = REPLAY_LONG.sub(swap, text)
        text = replace_once(text, f"cache_catalog: {PRODUCTION_CATALOG}", f"cache_catalog: {catalog}")
        print(f"swapped long caches of {len(swapped)} cells; {len(built - swapped)} built cells unused")
    elif arm != "control":
        raise ValueError(f"unknown arm {arm}")
    target = Path(configs_dir) / run / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(text)
    print("wrote", target)


if __name__ == "__main__":
    main()
