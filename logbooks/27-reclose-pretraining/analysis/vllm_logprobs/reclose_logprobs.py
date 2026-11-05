"""Score the fixed 10.05 re-close trajectories on a served HF checkpoint with vLLM prompt logprobs (#10400).

Mirrors ``experiments/grug_sft/reclose_eval.py`` (the in-training probe) without Levanter. At positions whose
next token is <|end_think|> (the first close and the re-close), the prompt logprob is the exact probability.
Elsewhere p(<|end_think|>) is read from the top-k prompt logprobs and taken as 0 when it is not in the top k,
so ``hazard`` and ``mean_p`` are lower bounds; ``p_first_close`` and ``p_reclose`` are exact.
"""

import argparse
import json
from collections import defaultdict

import fsspec
import numpy as np
from common import END_ID, logger, model_config, pmap, result, serve
from rigging.filesystem.s3_compat import configure_coreweave_s3

TOP_K = 20


def close_probabilities(client, ids: list[int]) -> np.ndarray:
    """``p[j]`` = p(ids[j + 1] would be <|end_think|> | ids[: j + 1]), matching ``reclose_eval.trajectory_stats``."""
    out = client.complete(ids, max_tokens=1, temperature=0.0, prompt_logprobs=TOP_K)
    logprobs = out["choices"][0]["prompt_logprobs"]
    assert len(logprobs) == len(ids), (len(logprobs), len(ids))
    p = np.zeros(len(ids))
    for i in range(1, len(ids)):
        entry = logprobs[i].get(str(END_ID))
        if entry is not None:
            p[i - 1] = np.exp(entry["logprob"])
        else:
            assert ids[i] != END_ID
    return p


def trajectory_stats(t: dict, p: np.ndarray) -> dict:
    close, reclose = t["close"], t["reclose"]
    last = reclose if reclose is not None else len(t["ids"]) - 1
    continuation = p[close:last]
    return {
        "hazard": float(1.0 - np.prod(1.0 - continuation)),
        "mean_p": float(continuation.mean()),
        "p_reclose": float(p[reclose - 1]) if reclose is not None else None,
        "p_first_close": float(p[close - 1]),
        "p_premature": float(p[t["reasoning_start"] - 1 : close - 1].mean()),
    }


def aggregate(trajectories: list[dict], stats: list[dict]) -> dict[str, float]:
    cells: dict[str, list[dict]] = defaultdict(list)
    for t, s in zip(trajectories, stats, strict=True):
        cells[t["group"]].append(s)
        cells[f"{t['group']}/lead_in" if t["bridge"] else f"{t['group']}/no_lead_in"].append(s)
    out = {}
    for cell, rows in cells.items():
        for key in ("hazard", "mean_p", "p_first_close", "p_premature"):
            out[f"reclose/{cell}/{key}"] = float(np.mean([r[key] for r in rows]))
        recloses = [r["p_reclose"] for r in rows if r["p_reclose"] is not None]
        if recloses:
            out[f"reclose/{cell}/p_reclose"] = float(np.mean(recloses))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    configure_coreweave_s3()
    with fsspec.open(args.trajectories) as handle:
        trajectories = [json.loads(line) for line in handle]
    with serve(model_config(args.name, args.location, args.revision)) as client:
        stats = pmap(lambda t: trajectory_stats(t, close_probabilities(client, t["ids"])), trajectories)
    metrics = aggregate(trajectories, stats)
    result("reclose_logprobs", model=args.name, **{k: round(v, 4) for k, v in metrics.items()})
    with fsspec.open(args.output, "w") as handle:
        json.dump({"model": args.name, "metrics": metrics, "trajectories": stats}, handle)
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
