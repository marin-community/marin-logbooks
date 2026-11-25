# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encode full prefixes once with a causal LM and score its decision states."""

import hashlib
import json
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from experiments.post_training.goal_value.context import Context, ContextPolicy, prune_middle
from experiments.post_training.goal_value.data import Episode, read_shards, stable_hash

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ENCODING_VERSION = 2
ASSISTANT_CUE = "<|im_start|>assistant\n"
OMISSION_MARKER = "<|im_start|>system\n[Middle of history omitted. The most recent history follows.]<|im_end|>\n"


@dataclass(frozen=True)
class EncodedEpisode:
    episode: Episode
    ids: list[int]
    positions: list[int]
    goal_position: int


@dataclass(frozen=True)
class DecisionFeatures:
    episode: Episode
    positions: list[int]
    source_positions: list[int]
    states: torch.Tensor
    goal_state: torch.Tensor
    processed_tokens: int


def encode(episode: Episode, tokenizer: PreTrainedTokenizerBase, max_tokens: int) -> EncodedEpisode:
    """Emit one state index before each action whose complete prefix fits."""

    # Treat any chat control text that appears in the trace as text.
    def block(role: str, content: str) -> str:
        escaped = content.replace("<|", "<\\u007c")
        return f"<|im_start|>{role}\n{escaped}<|im_end|>\n"

    goal = block("system", f"Predict whether this goal will be achieved.\nGoal:\n{episode.goal}")
    encode_text = tokenizer.encode
    ids = encode_text(goal, add_special_tokens=False)
    goal_position = len(ids) - 1
    cue_ids = encode_text(ASSISTANT_CUE, add_special_tokens=False)
    decisions = set(episode.predict_before)
    positions = []
    for index, msg in enumerate(episode.messages):
        role = msg["role"]
        message = {key: value for key, value in msg.items() if key != "role"}
        content = message.get("content")
        if not isinstance(content, str) or len(message) > 1:
            content = json.dumps(message, ensure_ascii=False, sort_keys=True)
        if index in decisions:
            if role != "assistant":
                raise ValueError(f"decision before non-assistant turn in {episode.trajectory_id}")
            if len(ids) + len(cue_ids) > max_tokens:
                break
            ids.extend(cue_ids)
            positions.append(len(ids) - 1)
            if index == episode.predict_before[-1]:
                break
            # The tokens already include the role header used for prediction.
            ids.extend(encode_text(block(role, content).removeprefix(ASSISTANT_CUE), add_special_tokens=False))
        else:
            ids.extend(encode_text(block(role, content), add_special_tokens=False))
    return EncodedEpisode(episode, ids[: positions[-1] + 1] if positions else [], positions, goal_position)


def decision_windows(
    episode: Episode, tokenizer: PreTrainedTokenizerBase, context: Context
) -> Iterator[tuple[EncodedEpisode, list[int]]]:
    """Yield encoded windows and the corresponding indices in the unpruned history."""
    if context.policy == ContextPolicy.PREFIX_LIMIT:
        encoded = encode(episode, tokenizer, context.max_tokens)
        yield encoded, encoded.positions
        return

    encoded = encode(episode, tokenizer, sys.maxsize)
    cue = tokenizer.encode(ASSISTANT_CUE, add_special_tokens=False)
    marker = tokenizer.encode(OMISSION_MARKER, add_special_tokens=False)
    for window in prune_middle(encoded.ids, encoded.positions, len(cue), marker, context.max_tokens):
        yield replace(encoded, ids=window.ids, positions=window.positions), window.source_positions


class TurnEncoder:
    """Frozen causal backbone with one representation per indexed decision."""

    def __init__(self, model_source: str, device: torch.device):
        kwargs = {"revision": MODEL_REVISION} if model_source == MODEL_ID else {"local_files_only": True}
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs)
        self.model = (
            AutoModel.from_pretrained(model_source, dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs)
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def hidden(self, batch: list[EncodedEpisode]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Gather a goal-only state and the final hidden states at the requested decisions."""
        width = max(len(item.ids) for item in batch)
        padded = [item.ids + [self.tokenizer.pad_token_id] * (width - len(item.ids)) for item in batch]
        ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        # Padding is to the right. Causal attention prevents it from reaching any requested position.
        states = self.model(input_ids=ids, use_cache=False).last_hidden_state
        batch_indices = [i for i, item in enumerate(batch) for _ in range(1 + len(item.positions))]
        positions = [p for item in batch for p in (item.goal_position, *item.positions)]
        values = states[batch_indices, positions].cpu()
        return [(value[0], value[1:]) for value in values.split([len(item.positions) + 1 for item in batch])]


class ValueHead(nn.Linear):
    """A scalar logit at each selected turn. Apply sigmoid for a probability."""

    def __init__(self, hidden_size: int):
        super().__init__(hidden_size, 1)


def cache_key(shard: Path, context: Context) -> str:
    with shard.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    digest.update(f"{MODEL_REVISION}:{ENCODING_VERSION}:{context.max_tokens}:{context.policy}".encode())
    return digest.hexdigest()


def _encode_batch(encoder: TurnEncoder, batch: list[EncodedEpisode]) -> Iterator[DecisionFeatures]:
    if batch:
        for item, (goal, hidden) in zip(batch, encoder.hidden(batch), strict=True):
            yield DecisionFeatures(item.episode, item.positions, item.positions, hidden, goal, len(item.ids))


def feature_rows(encoder: TurnEncoder, shard: Path, context: Context, batch_size: int) -> Iterator[DecisionFeatures]:
    """Compute each covered turn state once, grouping decisions that share a causal window."""
    batch: list[EncodedEpisode] = []
    for episode in read_shards([shard]):
        windows = list(decision_windows(episode, encoder.tokenizer, context))
        item, _ = windows[0]
        if not item.positions:
            empty = torch.empty(0, encoder.model.config.hidden_size, dtype=torch.bfloat16)
            yield DecisionFeatures(episode, [], [], empty, empty, 0)
            continue
        if len(windows) == 1:
            batch.append(item)
            if len(batch) == batch_size:
                yield from _encode_batch(encoder, batch)
                batch.clear()
            continue

        yield from _encode_batch(encoder, batch)
        batch.clear()
        window_states = [encoder.hidden([window])[0] for window, _ in windows]
        yield DecisionFeatures(
            episode,
            [position for window, _ in windows for position in window.positions],
            [position for _, original in windows for position in original],
            torch.cat([states for _, states in window_states]),
            window_states[0][0],
            sum(len(window.ids) for window, _ in windows),
        )

    yield from _encode_batch(encoder, batch)


def extract(encoder: TurnEncoder, shard: Path, context: Context, batch_size: int) -> dict:
    """Build a reusable feature cache with one row per covered decision."""
    started = time.perf_counter()
    metadata = []
    tensors = []
    goals = []
    for features in feature_rows(encoder, shard, context, batch_size):
        episode = features.episode
        metadata.append(
            {
                "trajectory_id": episode.trajectory_id,
                "task_id": episode.task_id,
                "goal_id": stable_hash("goal", episode.goal),
                "policy": episode.policy,
                "family": episode.family,
                "repository": episode.repository,
                "split": episode.split,
                "outcome": episode.outcome,
                "total_turns": len(episode.predict_before),
                "positions": features.positions,
                "source_positions": features.source_positions,
                "processed_tokens": features.processed_tokens,
            }
        )
        tensors.append(features.states)
        goals.append(features.goal_state)
        if len(metadata) % 40 == 0:
            seconds = time.perf_counter() - started
            print(f"Encoded {len(metadata)} trajectories in {seconds:.1f}s", flush=True)
    seconds = time.perf_counter() - started
    return {"metadata": metadata, "states": torch.cat(tensors), "goal_states": goals, "seconds": seconds}
