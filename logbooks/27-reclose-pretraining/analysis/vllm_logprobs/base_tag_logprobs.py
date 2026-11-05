# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the base model's plain-text think close at the re-close probe positions (marin-community/marin#10400).

The base model never saw ``<|end_think|>`` (SFT resets its LM-head row), but pretraining text contains plain
``<think>``/``</think>`` blocks and stray ``</thinking>`` tags. Each fixed 10.05 probe trajectory is re-rendered
the way pretraining renders chat (``<role>\\ncontent\\n</role>`` blocks joined by blank lines, reasoning inside
``<think>``), and the base model scores, at chosen positions, the probability that the text continues with a
closing tag. The positions are the first close, 10.05's own re-close, and up to ``MAX_CANDIDATES`` sentence or
line ends in the continuation (plus ``MID_POSITIONS`` other positions as a baseline), stopping at the first
tool-call or other special token.

A closer's probability is text-level. With ``prefix`` the rendered text through SFT token ``j``, p(S) is
P(enc(prefix + S)) / P(enc(prefix)) over the tokens after their common prefix, so a close that BPE merges into
the preceding token (``.</``) is scored correctly. Token logprobs come from one-token completions restricted to
the target token, which reuse vLLM's prefix cache. A sample is first checked against prompt logprobs; if they
disagree the run switches to prompt logprobs and records that in the output.
"""

import argparse
import json
import random
import re
from dataclasses import dataclass, field

import fsspec
import numpy as np
from common import BOS_ID, END, END_ID, Client, logger, model_config, pmap, result, serve
from rigging.filesystem.s3_compat import configure_coreweave_s3
from transformers import AutoTokenizer

BASE_REPO = "open-athena/snowball-67b-a2b-base-262k-qk175-skew8"
BASE_REVISION = "058ecaf27b9e4f37219df221a51e7d490d58ec3d"
CLOSERS = {
    "think": "</think>",
    "think_nl": "\n</think>",
    "thinking": "</thinking>",
    "thinking_nl": "\n</thinking>",
    "assistant_nl": "\n</assistant>",
}
MAX_CANDIDATES = 16
MID_POSITIONS = 4
SPECIAL_IDS = frozenset(range(128000, 128256)) - {END_ID}
HEADER = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>")
SENTENCE_END = (".", "!", "?", ":", "\n", "`")
VALIDATION_TRAJECTORIES = 8
VALIDATION_TOLERANCE = 0.1


def render_prompt(prompt: str) -> str:
    """Map an SFT chat prompt ending in ``assistant`` header + ``<|start_think|>`` to pretraining plain text."""
    *closed, open_turn = prompt.split("<|eot_id|>")
    blocks = []
    for turn in closed:
        match = HEADER.search(turn)
        blocks.append(f"<{match[1]}>\n{turn[match.end():]}\n</{match[1]}>")
    match = HEADER.search(open_turn)
    assert match[1] == "assistant" and open_turn[match.end() :].strip() == "<|start_think|>", open_turn[-200:]
    return "\n\n".join(blocks) + "\n\n<assistant>\n<think>"


@dataclass
class Query:
    j: int
    kind: str
    prefix: str
    logp: dict[str, float] = field(default_factory=dict)


def scored_end(t: dict) -> int:
    """Index of the first special token after the close other than a re-close, or the trajectory length."""
    ids = t["ids"]
    return next((i for i in range(t["close"] + 1, len(ids)) if ids[i] in SPECIAL_IDS), len(ids))


def build_queries(t: dict, tokenizer, rng: random.Random) -> list[Query]:
    ids, close, reclose = t["ids"], t["close"], t["reclose"]
    head = render_prompt(tokenizer.decode(ids[: t["reasoning_start"]], skip_special_tokens=False))
    end = scored_end(t)
    if reclose is not None and reclose >= end:
        reclose = None
    last = reclose if reclose is not None else end

    def prefix(j: int) -> str | None:
        body = tokenizer.decode(ids[t["reasoning_start"] : j + 1], skip_special_tokens=False)
        if body.endswith("�"):
            return None
        return head + body.replace(END, "</think>")

    def piece(j: int) -> str:
        return tokenizer.decode([ids[j]], skip_special_tokens=False) if j < len(ids) else ""

    # With a re-close, ``last - 1`` is scored as kind "reclose", not as a candidate.
    stop = last - 1 if reclose is not None else last
    candidates = [
        j for j in range(close + 1, stop) if piece(j).rstrip(" ").endswith(SENTENCE_END) or piece(j + 1).startswith("\n")
    ]
    if len(candidates) > MAX_CANDIDATES:
        candidates = [candidates[int(i)] for i in np.linspace(0, len(candidates) - 1, MAX_CANDIDATES)]
    others = sorted(set(range(close + 1, stop)) - set(candidates))
    mids = sorted(rng.sample(others, min(MID_POSITIONS, len(others))))
    positions = [(close - 1, "first_close"), (close, "after_close")]
    positions += [(j, "candidate") for j in candidates] + [(j, "mid") for j in mids]
    if reclose is not None:
        positions.append((reclose - 1, "reclose"))
    queries = []
    for j, kind in positions:
        text = prefix(j)
        if text is not None:
            queries.append(Query(j, kind, text))
    return queries


class Scorer:
    """Text-level log P(closer | prefix) from per-token logprobs, memoized within one trajectory."""

    def __init__(self, client: Client, tokenizer, use_prompt_logprobs: bool):
        self.client = client
        self.tokenizer = tokenizer
        self.use_prompt_logprobs = use_prompt_logprobs
        self.memo: dict[tuple[int, ...], float] = {}

    def encode(self, text: str) -> list[int]:
        return [BOS_ID, *self.tokenizer.encode(text, add_special_tokens=False).ids]

    def token_logprob(self, context: list[int], target: int) -> float:
        key = (*context, target)
        if key not in self.memo:
            self.memo[key] = (
                prompt_token_logprob(self.client, context, target)
                if self.use_prompt_logprobs
                else forced_token_logprob(self.client, context, target)
            )
        return self.memo[key]

    def sequence_logprob(self, ids: list[int], start: int) -> float:
        return sum(self.token_logprob(ids[:i], ids[i]) for i in range(start, len(ids)))

    def closer_logprob(self, prefix: str, closer: str) -> float:
        base, extended = self.encode(prefix), self.encode(prefix + closer)
        k = next(
            (i for i, (a, b) in enumerate(zip(base, extended, strict=False)) if a != b), min(len(base), len(extended))
        )
        return self.sequence_logprob(extended, k) - self.sequence_logprob(base, k)


def forced_token_logprob(client: Client, context: list[int], target: int) -> float:
    """Raw logprob of ``target`` after ``context``; vLLM's default logprobs mode ignores ``allowed_token_ids``."""
    out = client.complete(
        context,
        max_tokens=1,
        temperature=0.0,
        logprobs=0,
        allowed_token_ids=[target],
        return_tokens_as_token_ids=True,
    )
    logprobs = out["choices"][0]["logprobs"]
    assert logprobs["tokens"][0] == f"token_id:{target}", logprobs["tokens"]
    return float(logprobs["token_logprobs"][0])


def prompt_token_logprob(client: Client, context: list[int], target: int) -> float:
    out = client.complete([*context, target], max_tokens=1, temperature=0.0, prompt_logprobs=0)
    return float(out["choices"][0]["prompt_logprobs"][-1][str(target)]["logprob"])


def score_trajectory(client: Client, tokenizer, t: dict, index: int, use_prompt_logprobs: bool) -> dict:
    queries = build_queries(t, tokenizer, random.Random(index))
    scorer = Scorer(client, tokenizer, use_prompt_logprobs)
    for q in queries:
        q.logp = {name: scorer.closer_logprob(q.prefix, closer) for name, closer in CLOSERS.items()}
    return {
        "group": t["group"],
        "id": t["id"],
        "sample": t["sample"],
        "bridge": t["bridge"],
        "close": t["close"],
        "reclose": t["reclose"],
        "queries": [{"j": q.j, "kind": q.kind, "logp": q.logp} for q in queries],
    }


def forced_logprobs_match(client: Client, tokenizer, trajectories: list[dict]) -> bool:
    """Compare forced-token and prompt logprobs for every closer at the first close of a few trajectories."""
    diffs = []
    for index, t in enumerate(trajectories[:VALIDATION_TRAJECTORIES]):
        q = build_queries(t, tokenizer, random.Random(index))[0]
        forced, prompted = Scorer(client, tokenizer, False), Scorer(client, tokenizer, True)
        for closer in CLOSERS.values():
            diffs.append(abs(forced.closer_logprob(q.prefix, closer) - prompted.closer_logprob(q.prefix, closer)))
    result("validation", max_abs_diff=round(max(diffs), 4), n=len(diffs))
    return max(diffs) <= VALIDATION_TOLERANCE


def summarize(rows: list[dict]) -> None:
    cells: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        lead = "lead_in" if row["bridge"] else "no_lead_in"
        for q in row["queries"]:
            p = np.exp(q["logp"]["think"]) + np.exp(q["logp"]["think_nl"])
            cells.setdefault((row["group"], lead, q["kind"]), []).append(float(p))
    for (group, lead, kind), values in sorted(cells.items()):
        result(
            "base_close", group=group, lead=lead, kind=kind, n=len(values), mean_p_think=round(float(np.mean(values)), 5)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N trajectories (smoke test)")
    args = parser.parse_args()

    configure_coreweave_s3()
    # The Rust tokenizer, not the transformers wrapper, which mutates shared state on every call across threads.
    tokenizer = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer").backend_tokenizer
    with fsspec.open(args.trajectories) as handle:
        trajectories = [json.loads(line) for line in handle][: args.limit]
    with serve(model_config("snowball-base", BASE_REPO, BASE_REVISION)) as client:
        use_prompt_logprobs = not forced_logprobs_match(client, tokenizer, trajectories)
        if use_prompt_logprobs:
            logger.warning("forced-token logprobs disagree with prompt logprobs; scoring with prompt logprobs")
        rows = pmap(
            lambda item: score_trajectory(client, tokenizer, item[1], item[0], use_prompt_logprobs),
            list(enumerate(trajectories)),
        )
    summarize(rows)
    with fsspec.open(args.output, "w") as handle:
        json.dump(
            {"model": BASE_REPO, "revision": BASE_REVISION, "prompt_logprobs": use_prompt_logprobs, "rows": rows}, handle
        )
    logger.info("wrote %d trajectories to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
