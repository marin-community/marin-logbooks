# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Held-out validation loss for the open-swe outcome-filtering arms.

``build`` assembles a fixed validation set from the open-swe trace pool, excluding every
trajectory and every SWE task (``instance_id``) that appears in either 10,000-row training arm.
It samples equal numbers of resolved and unresolved trajectories per source partition, weighted
by the partition's share of the training arms, and adds small samples of each arm's own training
rows as a memorization reference. It converts raw traces to the arms' chat format and fails if
that conversion does not reproduce the arm rows exactly.

``score`` computes per-trajectory assistant-token negative log-likelihood for one model, using
Levanter's ``ChatProcessor`` with the training chat template so the loss mask matches training.

``summarize`` prints per-model, per-subset token-weighted loss and paired arm differences.
"""

import argparse
import concurrent.futures
import gzip
import json
import logging
import os
import random
import tempfile
from collections import Counter, defaultdict
from enum import StrEnum

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3

logger = logging.getLogger(__name__)

BUCKET = "s3://marin-us-east-08a/marin"
INDEX_DIR = f"{BUCKET}/open-swe-outcome/index_d147d574"
RANDOM_MEMBERSHIP = f"{BUCKET}/open-swe-outcome/membership_d3c29ebe/random-10000.parquet"
ARM_DIRS = {
    "solved": f"{BUCKET}/open-swe-outcome/arms/solved-10000_5e1e1b9c",
    "random": f"{BUCKET}/open-swe-outcome/arms/random-10000_d5a9e952",
}
TOKENIZED_ARTIFACT = f"{BUCKET}/tokenized/solved-10000-chat-f60c77/2026.07.17/.artifact.json"
RAW_DIRS = [
    "open_swe_traces/v1_0/openhands/minimax_m25/swe_rebench_v2_d05f96ec",
    "open_swe_traces/v1_0/openhands/qwen35_122b/swe_rebench_v2_e018c6f4",
    "open_swe_traces/v1_0/sweagent/minimax_m25/swe_rebench_v2_0787afcc",
    "open_swe_traces/v1_0/sweagent/qwen35_122b/swe_rebench_v2_9df1f6cb",
    "open_swe_traces/v1_1/openhands/deepseek_v4_flash/scale_swe_f3ad5c0e",
    "open_swe_traces/v1_1/openhands/qwen36_27b/scale_swe_5eca5d1d",
    "open_swe_traces/v1_1/openhands/qwen36_27b/swe_rebench_v2_9a446c83",
    "open_swe_traces/v1_1/sweagent/qwen36_27b/scale_swe_d3e1e749",
    "open_swe_traces/v1_1/sweagent/qwen36_27b/swe_rebench_v2_f72e3a53",
    "open_swe_traces/v1_1/minisweagent/qwen36_27b/scale_swe_a60ce00d",
    "open_swe_traces/v1_1/minisweagent/qwen36_27b/swe_rebench_v2_4ebc30bf",
    "open_swe_traces/v1_2/minisweagent/qwen38_27b/scale_swe_697774eb",
    "open_swe_traces/v1_2/minisweagent/qwen38_27b/swe_rebench_v2_ac63e1e6",
]
# The arm builder inserted this block into the first user message of every v1_2 mini-SWE-agent
# trace, between the closing </CRITICAL> and </instructions> tags.
ORIGINALITY_PARTITION_PREFIX = "open_swe_traces/v1_2/minisweagent/"
ORIGINALITY_ANCHOR = "\n</CRITICAL>\n</instructions>"
ORIGINALITY_BLOCK = """

## Solution Originality

Implement your own original solution to the task.
As an expert software engineer, your task is to create a novel independent solution based only on the current repository state.
Your approach must be ENTIRELY UNBIASED, making no attempt to access external solutions, hints or information in any way.
Remember that your own implementation is strongly preferred over a copy or modification of an existing solution.
Because you are working independently, you **MUST NOT** attempt to:
- clone, download files from, install as a package, read pull requests or otherwise access the upstream repository,
- use git commands to access any commits, tags or references outside of the current branch,
- read previous trajectories, golden patches, test patches or any other hidden information from the filesystem,
- search locally installed packages or caches for upstream code,
- find loopholes to access external solutions or hidden tests in any other way,
for ANY reason at all, even if it's just "for reference" or to verify your own solution.
WARNING: If you seek any form of external solutions, hints or information while solving the task,
your solution will be DISCARDED regardless of its correctness!"""  # noqa: E501
MAX_TOKENS = 65536
TOKENIZER = "Qwen/Qwen3-8B"
LM_HEAD_CHUNK = 8192
IO_THREADS = 32


def _partition(raw_dir: str) -> str:
    return raw_dir.rsplit("_", 1)[0]


def _parquet_files(path: str) -> list[str]:
    fs = fsspec.filesystem("s3")
    return sorted(f"s3://{p}" for p in fs.find(path) if p.endswith(".parquet"))


def _read_index() -> pa.Table:
    files = _parquet_files(INDEX_DIR)
    with concurrent.futures.ThreadPoolExecutor(IO_THREADS) as pool:
        tables = list(pool.map(lambda f: pq.read_table(fsspec.open(f).open()), files))
    return pa.concat_tables(tables)


def _read_arm_rows(arm_dir: str) -> list[dict]:
    fs = fsspec.filesystem("s3")
    files = sorted(f"s3://{p}" for p in fs.find(arm_dir) if p.endswith(".jsonl.gz"))

    def read(f: str) -> list[dict]:
        with fsspec.open(f, "rb").open() as raw, gzip.open(raw, "rt") as lines:
            return [json.loads(line) for line in lines]

    with concurrent.futures.ThreadPoolExecutor(IO_THREADS) as pool:
        return [row for rows in pool.map(read, files) for row in rows]


def _raw_id_map() -> dict[str, tuple[str, str, str]]:
    """trajectory_id -> (instance_id, partition, parquet file)."""
    jobs = [(_partition(d), f) for d in RAW_DIRS for f in _parquet_files(f"{BUCKET}/raw/{d}")]

    def read(job: tuple[str, str]) -> list[tuple[str, str, str, str]]:
        partition, f = job
        t = pq.read_table(fsspec.open(f).open(), columns=["trajectory_id", "instance_id"])
        return [
            (tid, iid, partition, f)
            for tid, iid in zip(t["trajectory_id"].to_pylist(), t["instance_id"].to_pylist(), strict=True)
        ]

    out: dict[str, tuple[str, str, str]] = {}
    with concurrent.futures.ThreadPoolExecutor(IO_THREADS) as pool:
        for rows in pool.map(read, jobs):
            for tid, iid, partition, f in rows:
                out[tid] = (iid, partition, f)
    return out


def _convert_messages(messages: list[dict]) -> list[dict]:
    """Reproduce the arm builder's message normalization.

    Null fields are dropped, tool-call ids become ``call_<message>_<call>``, and each tool result is
    linked to the next unanswered call of the preceding assistant turn.
    """
    pending: list[str] = []
    out = []
    for i, message in enumerate(messages):
        message = {key: value for key, value in message.items() if value is not None}
        if "tool_calls" in message:
            message["tool_calls"] = [{**call, "id": f"call_{i}_{k}"} for k, call in enumerate(message["tool_calls"])]
            pending = [call["id"] for call in message["tool_calls"]]
        if message["role"] == "tool":
            message["tool_call_id"] = pending.pop(0)
        out.append(message)
    return out


def _insert_originality_block(messages: list[dict]) -> list[dict]:
    content = messages[1]["content"]
    if content.count(ORIGINALITY_ANCHOR) != 1:
        raise ValueError(f"expected one originality anchor, found {content.count(ORIGINALITY_ANCHOR)}")
    replacement = "\n</CRITICAL>" + ORIGINALITY_BLOCK + "\n</instructions>"
    return [messages[0], {**messages[1], "content": content.replace(ORIGINALITY_ANCHOR, replacement)}, *messages[2:]]


def _convert_raw(row: dict, partition: str) -> dict:
    messages = _convert_messages(row["messages"])
    if partition.startswith(ORIGINALITY_PARTITION_PREFIX):
        messages = _insert_originality_block(messages)
    return {
        "messages": messages,
        "chat_template_kwargs": {"tools": [json.loads(tool) for tool in row["tools"]]},
        "trajectory_id": row["trajectory_id"],
        "partition": partition,
        "resolved": row["resolved"],
    }


def _read_raw_rows(wanted: dict[str, tuple[str, str, str]]) -> dict[str, dict]:
    by_file: dict[str, list[str]] = defaultdict(list)
    for tid, (_, _, f) in wanted.items():
        by_file[f].append(tid)

    def read(item: tuple[str, list[str]]) -> list[dict]:
        f, tids = item
        t = pq.read_table(fsspec.open(f).open(), filters=[("trajectory_id", "in", tids)])
        return [_convert_raw(row, wanted[row["trajectory_id"]][1]) for row in t.to_pylist()]

    with concurrent.futures.ThreadPoolExecutor(IO_THREADS) as pool:
        return {row["trajectory_id"]: row for rows in pool.map(read, by_file.items()) for row in rows}


def _first_difference(a: object, b: object, path: str = "") -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}: present only in {'converted' if key in a else 'arm'}"
            found = _first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            found = _first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    if a != b:
        if isinstance(a, str) and isinstance(b, str):
            i = next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
            return f"{path} at char {i} (lengths {len(a)}, {len(b)}): {a[i - 80 : i + 120]!r} != {b[i - 80 : i + 120]!r}"
        return f"{path}: {str(a)[:200]!r} != {str(b)[:200]!r}"
    return None


def _write_jsonl_gz(path: str, rows: list[dict]) -> None:
    with fsspec.open(path, "wb").open() as raw, gzip.open(raw, "wt") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")


class Design(StrEnum):
    """How held-out trajectories are chosen.

    ``independent`` draws resolved and unresolved trajectories from disjoint tasks. ``paired`` draws
    one resolved and one unresolved trajectory from the same task and source partition, so task
    difficulty and teacher style cancel within each pair.
    """

    INDEPENDENT = "independent"
    PAIRED = "paired"


Candidates = dict[tuple[str, int], dict[str, list[str]]]


def _partition_quotas(partition_share: Counter[str], per_class: int) -> list[tuple[str, int]]:
    total = sum(partition_share.values())
    return [(p, round(per_class * share / total)) for p, share in sorted(partition_share.items())]


def _select_independent(
    candidates: Candidates, partition_share: Counter[str], per_class: int, rng: random.Random
) -> dict[str, str]:
    subsets: dict[str, str] = {}
    used_tasks: set[str] = set()
    for partition, quota in _partition_quotas(partition_share, per_class):
        for resolved in (1, 0):
            by_task = candidates[(partition, resolved)]
            tasks = sorted(t for t in by_task if t not in used_tasks)
            rng.shuffle(tasks)
            chosen = tasks[:quota]
            logger.info(
                "%s resolved=%d: quota=%d available=%d chosen=%d", partition, resolved, quota, len(tasks), len(chosen)
            )
            for task in chosen:
                subsets[rng.choice(sorted(by_task[task]))] = "heldout-resolved" if resolved else "heldout-unresolved"
                used_tasks.add(task)
    return subsets


def _select_paired(
    candidates: Candidates, partition_share: Counter[str], per_class: int, rng: random.Random
) -> dict[str, str]:
    subsets: dict[str, str] = {}
    used_tasks: set[str] = set()
    for partition, quota in _partition_quotas(partition_share, per_class):
        solved, failed = candidates[(partition, 1)], candidates[(partition, 0)]
        tasks = sorted(t for t in solved.keys() & failed.keys() if t not in used_tasks)
        rng.shuffle(tasks)
        chosen = tasks[:quota]
        logger.info("%s pairs: quota=%d available=%d chosen=%d", partition, quota, len(tasks), len(chosen))
        for task in chosen:
            subsets[rng.choice(sorted(solved[task]))] = "heldout-resolved"
            subsets[rng.choice(sorted(failed[task]))] = "heldout-unresolved"
            used_tasks.add(task)
    return subsets


def build(out_dir: str, design: Design, per_class: int, train_reference: int, verify_rows: int, seed: int) -> None:
    rng = random.Random(seed)
    index = _read_index().to_pylist()
    logger.info("index rows: %d", len(index))

    arm_rows = {arm: _read_arm_rows(d) for arm, d in ARM_DIRS.items()}
    random_members = set(pq.read_table(fsspec.open(RANDOM_MEMBERSHIP).open())["trajectory_id"].to_pylist())
    arm_ids = {arm: {r["trajectory_id"] for r in rows} for arm, rows in arm_rows.items()}
    if arm_ids["random"] != random_members:
        raise ValueError("random arm rows differ from the random-10000 membership list")
    train_ids = arm_ids["solved"] | arm_ids["random"]
    logger.info("arm rows: solved=%d random=%d union=%d", len(arm_ids["solved"]), len(arm_ids["random"]), len(train_ids))

    raw_ids = _raw_id_map()
    missing = train_ids - raw_ids.keys()
    if missing:
        raise ValueError(f"{len(missing)} training trajectories absent from the raw pool")
    train_tasks = {raw_ids[tid][0] for tid in train_ids}
    logger.info("training tasks: %d", len(train_tasks))

    # Verify the raw -> arm conversion on training rows before trusting it for held-out rows.
    verify = rng.sample(arm_rows["solved"], verify_rows // 2) + rng.sample(arm_rows["random"], verify_rows // 2)
    converted = _read_raw_rows({r["trajectory_id"]: raw_ids[r["trajectory_id"]] for r in verify})
    checked: Counter[str] = Counter()
    mismatches: dict[str, list[str]] = defaultdict(list)
    for row in verify:
        checked[row["partition"]] += 1
        difference = _first_difference(converted[row["trajectory_id"]], row)
        if difference:
            mismatches[row["partition"]].append(f"{row['trajectory_id']}: {difference}")
    for partition, count in sorted(checked.items()):
        found = mismatches.get(partition, [])
        logger.info(
            "conversion %s: %d/%d mismatched%s", partition, len(found), count, f"; first {found[0]}" if found else ""
        )
    if mismatches:
        raise ValueError(f"conversion mismatches in {len(mismatches)} partitions")
    logger.info("conversion reproduces %d arm rows exactly", len(verify))

    partition_share = Counter(raw_ids[tid][1] for tid in train_ids)
    candidates: dict[tuple[str, int], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in index:
        tid = row["trajectory_id"]
        if row["status"] != "kept" or row["num_tokens"] > MAX_TOKENS or tid in train_ids:
            continue
        instance_id = raw_ids[tid][0]
        if instance_id in train_tasks:
            continue
        candidates[(row["partition"], row["resolved"])][instance_id].append(tid)

    select = _select_paired if design == Design.PAIRED else _select_independent
    subsets = select(candidates, partition_share, per_class, rng)
    selected = {tid: raw_ids[tid] for tid in subsets}

    heldout = _read_raw_rows(selected)
    rows = [{**heldout[tid], "subset": subsets[tid], "instance_id": selected[tid][0]} for tid in sorted(heldout)]
    for arm in ("solved", "random"):
        for row in rng.sample(arm_rows[arm], train_reference):
            rows.append({**row, "subset": f"train-{arm}", "instance_id": raw_ids[row["trajectory_id"]][0]})

    _write_jsonl_gz(f"{out_dir}/val.jsonl.gz", rows)
    manifest = {
        "seed": seed,
        "design": design.value,
        "per_class": per_class,
        "train_reference": train_reference,
        "verified_conversion_rows": len(verify),
        "training_trajectories": len(train_ids),
        "training_tasks": len(train_tasks),
        "subset_counts": Counter(r["subset"] for r in rows),
        "partition_counts": Counter(f"{r['subset']}|{r['partition']}" for r in rows),
    }
    with fsspec.open(f"{out_dir}/manifest.json", "w").open() as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote %d rows: %s", len(rows), json.dumps(manifest["subset_counts"]))


def _local_model_dir(model: str, scratch: str) -> str:
    if not model.startswith("s3://"):
        return model
    fs = fsspec.filesystem("s3")
    files = [p for p in fs.ls(model, detail=False) if not p.rstrip("/").endswith("/")]
    local = os.path.join(scratch, "model")
    os.makedirs(local, exist_ok=True)

    def fetch(p: str) -> None:
        fs.get(p, os.path.join(local, os.path.basename(p)))

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, files))
    return local


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` spans of consecutive nonzero mask positions."""
    edges = np.diff(np.concatenate([[0], (mask > 0).astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True))


def score(model: str, name: str, val_dir: str, scores_dir: str, subset_prefix: str) -> None:
    # Imported here so `build` and `summarize` run without the GPU extra.
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from levanter.data.text.formats import ChatProcessor  # noqa: PLC0415
    from levanter.tokenizers import load_tokenizer  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    template = json.load(fsspec.open(TOKENIZED_ARTIFACT).open())["config"]["format"]["chat_template"]
    processor = ChatProcessor(
        load_tokenizer(TOKENIZER),
        chat_template=template,
        system_prompt_field=None,
        chat_template_kwargs_field="chat_template_kwargs",
        mask_user_turns=True,
    )
    with fsspec.open(f"{val_dir}/val.jsonl.gz", "rb").open() as raw, gzip.open(raw, "rt") as lines:
        rows = [row for row in map(json.loads, lines) if row["subset"].startswith(subset_prefix)]

    with tempfile.TemporaryDirectory() as scratch:
        lm = AutoModelForCausalLM.from_pretrained(
            _local_model_dir(model, scratch), dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    lm = lm.cuda().eval()
    # SDPA's cuDNN backend finds no execution plan at these sequence lengths; use flash attention.
    torch.backends.cuda.enable_cudnn_sdp(False)

    results = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            encoded = processor([row])[0]
            ids, mask = encoded["input_ids"], encoded["assistant_masks"]
            if len(ids) > MAX_TOKENS:
                raise ValueError(f"{row['trajectory_id']} has {len(ids)} tokens, above {MAX_TOKENS}")
            input_ids = torch.from_numpy(ids.astype(np.int64)).cuda()[None]
            hidden = lm.model(input_ids=input_ids).last_hidden_state[0, :-1]
            targets = input_ids[0, 1:]
            chunks = []
            for start in range(0, hidden.shape[0], LM_HEAD_CHUNK):
                logits = lm.lm_head(hidden[start : start + LM_HEAD_CHUNK]).float()
                chunks.append(F.cross_entropy(logits, targets[start : start + LM_HEAD_CHUNK], reduction="none"))
            position_nll = torch.cat(chunks).double().cpu().numpy() * mask[1:]
            # Each contiguous run of loss positions is one assistant turn.
            turns = _mask_runs(mask[1:])
            n_assistant = sum(1 for m in row["messages"] if m["role"] == "assistant")
            if len(turns) != n_assistant:
                raise ValueError(f"{row['trajectory_id']}: {len(turns)} loss spans for {n_assistant} assistant turns")
            results.append(
                {
                    "model": name,
                    "trajectory_id": row["trajectory_id"],
                    "subset": row["subset"],
                    "partition": row["partition"],
                    "n_tokens": len(ids),
                    "n_loss_tokens": int(mask[1:].sum()),
                    "sum_nll": float(position_nll.sum()),
                    "turn_nll": [float(position_nll[s:e].sum()) for s, e in turns],
                    "turn_tokens": [e - s for s, e in turns],
                }
            )
            if i % 50 == 0:
                logger.info("%s: %d/%d scored", name, i, len(rows))

    with fsspec.open(f"{val_dir}/{scores_dir}/{name}.parquet", "wb").open() as f:
        pq.write_table(pa.Table.from_pylist(results), f)
    logger.info("%s: wrote %d scores", name, len(results))


def summarize(val_dir: str) -> None:
    fs = fsspec.filesystem("s3")
    tables = [pq.read_table(fs.open(p)) for p in fs.glob(f"{val_dir}/scores/*.parquet")]
    rows = pa.concat_tables(tables).to_pylist()
    by = defaultdict(dict)
    for r in rows:
        by[(r["model"], r["subset"])][r["trajectory_id"]] = r

    models = sorted({m for m, _ in by})
    subsets = sorted({s for _, s in by})
    print(f"{'model':<14}" + "".join(f"{s:>22}" for s in subsets))
    for m in models:
        cells = []
        for s in subsets:
            rs = by[(m, s)].values()
            cells.append(f"{sum(r['sum_nll'] for r in rs) / sum(r['n_loss_tokens'] for r in rs):>22.4f}")
        print(f"{m:<14}" + "".join(cells))

    print("\npaired per-trajectory loss difference, solved arm minus random arm (negative favors solved)")
    for s in subsets:
        for seed in ("s0", "s1"):
            a, b = by[(f"solved-{seed}", s)], by[(f"random-{seed}", s)]
            diffs = [
                a[t]["sum_nll"] / a[t]["n_loss_tokens"] - b[t]["sum_nll"] / b[t]["n_loss_tokens"]
                for t in a.keys() & b.keys()
            ]
            if diffs:
                d = np.array(diffs)
                print(f"{s:<22} {seed}: {d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(len(d)):.4f} (n={len(d)})")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--design", type=Design, choices=list(Design), required=True)
    b.add_argument("--per-class", type=int, required=True)
    b.add_argument("--train-reference", type=int, required=True)
    b.add_argument("--verify-rows", type=int, required=True)
    b.add_argument("--seed", type=int, required=True)
    s = commands.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--val-dir", required=True)
    s.add_argument("--scores-dir", required=True)
    s.add_argument("--subset-prefix", required=True)
    m = commands.add_parser("summarize")
    m.add_argument("--val-dir", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out_dir, args.design, args.per_class, args.train_reference, args.verify_rows, args.seed)
    elif args.command == "score":
        score(args.model, args.name, args.val_dir, args.scores_dir, args.subset_prefix)
    else:
        summarize(args.val_dir)


if __name__ == "__main__":
    main()
