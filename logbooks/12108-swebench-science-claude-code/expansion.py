"""Run the frozen checkpoint and remaining eligible tasks against one model server."""

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from levanter.compat.hf_checkpoints import load_tokenizer
from marin.inference.iris import remote_inference
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.cli import resolve_model_config

from .run import configure_deadline, corpus, evaluate, inference_service, launch_parser

SEED = "swebench-science-claude-code-evidence-v1"
PILOT_IDS = {"001", "003", "012", "013", "015", "028"}


def ordered_ids() -> list[str]:
    eligible = {f"{n:03}" for n in range(1, 120)} - PILOT_IDS - {"019"}
    return sorted(eligible, key=lambda task: hashlib.sha256(f"{SEED}/{task}".encode()).digest())


async def run_phase(session, model, tokenizer, root, datasets, ids):
    slots = asyncio.Semaphore(16)

    async def run_one(task):
        out = root / task
        if (out / "graded.json").exists():
            return
        async with slots:
            try:
                await evaluate(session, model, datasets[task], out, (task,), tokenizer)
            except Exception as exc:
                logging.exception("Task %s did not produce a grade", task)
                (out / "last_error.json").write_text(json.dumps({"type": type(exc).__name__}), auto_mkdir=True)

    await asyncio.gather(*(run_one(task) for task in ids))


def main():
    args = launch_parser().parse_args()
    configure_deadline(args)
    root = StoragePath(args.output)
    ids = ordered_ids()
    (root / "selection.json").write_text(json.dumps({"seed": SEED, "checkpoint": ids[:32], "remaining": ids[32:]}), auto_mkdir=True)
    datasets = {task: corpus(root / task, (task,)) for task in ids if not (root / task / "graded.json").exists()}
    if not datasets:
        return
    model = resolve_model_config(None, Path(__file__).with_name("model.yaml"))
    tokenizer = load_tokenizer(model.location, revision=model.tokenizer_revision, trust_remote_code=False)
    with remote_inference(inference_service(model, args.cluster)) as session:
        for label, group in (("checkpoint", ids[:32]), ("remaining", ids[32:])):
            for attempt in range(2):
                asyncio.run(run_phase(session, model, tokenizer, root, datasets, group))
                missing = [task for task in group if not (root / task / "graded.json").exists()]
                if not missing:
                    break
                logging.warning("Ungraded after phase %s pass %d: %s", label, attempt + 1, missing)
            (root / f"{label}-coverage.json").write_text(json.dumps({"selected": group, "ungraded": missing}), auto_mkdir=True)
    missing = [task for task in ids if not (root / task / "graded.json").exists()]
    if missing:
        raise RuntimeError(f"Ungraded task IDs: {missing}")


if __name__ == "__main__":
    main()
