# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score each document of a held-out replay set under a checkpoint, grouped by junk-filter verdict.

The cache holds one document per row, every row exactly ``seq_len`` tokens long, so the text dataset's
fixed windows over the token stream coincide with the rows and each window is one whole document.
The manifest has one line per row in cache order with the document's identity and its junk-filter
``category`` (``null`` when the filter keeps it).
"""

import dataclasses
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jmp
import numpy as np
from haliax import Axis
from jax._src import config as jax_config
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.loader import DataLoader
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.eval import _calculate_bytes_per_token_type
from levanter.grug.grug_moe import MoeImplementation
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.model import _BATCH_AXES, Transformer
from experiments.grug.moe_hero_ep.mrcr_eval import sequence_score_sums

logger = logging.getLogger(__name__)

REPLAY_LOSS_COMPONENT = "heldout_replay"
REPLAY_LOSS_MANIFEST_NAME = "manifest.jsonl"
REPLAY_LOSS_DOCUMENTS_NAME = "replay_document_losses.jsonl"
REPLAY_LOSS_METRICS_NAME = "replay_loss_metrics.json"
REPLAY_LOSS_METRIC_PREFIX = "eval/heldout_replay"


@dataclass(frozen=True)
class ReplayLossEvalConfig:
    data_path: str
    output_path: str
    tokenizer: str
    batch_size: int
    seq_len: int
    max_batches: int | None = None
    """Score only this many batches, as a smoke test."""


@dataclass(frozen=True)
class ReplayDocumentLoss:
    cell: str
    shard: str
    index: int
    category: str | None
    scored_tokens: int
    loss_sum: float
    scored_bytes: int


def replay_data_config(data_path: str, tokenizer: str) -> LmDataConfig:
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={REPLAY_LOSS_COMPONENT: DatasetComponent(cache_dir=data_path, format=TextLmDatasetFormat())},
        train_weights={REPLAY_LOSS_COMPONENT: 0.0},
        auto_build_caches=False,
    )


def score_replay_documents(
    params: Transformer,
    config: ReplayLossEvalConfig,
    *,
    mesh: Mesh,
    mp: jmp.Policy,
    implementation: MoeImplementation,
) -> list[ReplayDocumentLoss]:
    """Score every cache row in order and bind each to its manifest line."""
    manifest_path = StoragePath(config.data_path) / REPLAY_LOSS_MANIFEST_NAME
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    data = replay_data_config(config.data_path, config.tokenizer)
    position = Axis("position", config.seq_len)
    dataset = data.build_token_datasets(data.build_caches("validation"), position, split="validation")[
        REPLAY_LOSS_COMPONENT
    ]
    if len(dataset.as_sync_dataset()) != len(manifest):
        raise ValueError(f"cache has {len(dataset.as_sync_dataset())} windows, manifest has {len(manifest)} rows")
    loader = DataLoader(dataset, config.batch_size, mesh=mesh, axis_resources={"batch": _BATCH_AXES})
    byte_lengths = _calculate_bytes_per_token_type(data.the_tokenizer)
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    score = eqx.filter_jit(sequence_score_sums)

    output: list[ReplayDocumentLoss] = []
    batch_start = time.monotonic()
    with jax_config.enable_pgle(False):
        for batch_index, batch in enumerate(loader):
            if config.max_batches is not None and batch_index >= config.max_batches:
                break
            sums = score(
                params,
                batch.tokens,
                batch.loss_weight,
                batch.attn_mask,
                byte_lengths,
                sharding=sharding,
                mp=mp,
                implementation=implementation,
            )
            loss_sums, scored, scored_bytes = (
                np.asarray(multihost_utils.process_allgather(value, tiled=True)).reshape(-1) for value in sums
            )
            for offset, record in enumerate(manifest[len(output) : len(output) + config.batch_size]):
                # Every row is one whole window, so every position but the last predicts a token.
                if int(scored[offset]) != config.seq_len - 1:
                    raise ValueError(f"row {len(output)}: scored {int(scored[offset])} of {config.seq_len} positions")
                output.append(
                    ReplayDocumentLoss(
                        cell=record["cell"],
                        shard=record["shard"],
                        index=record["index"],
                        category=record["category"],
                        scored_tokens=int(scored[offset]),
                        loss_sum=float(loss_sums[offset]),
                        scored_bytes=int(scored_bytes[offset]),
                    )
                )
            logger.info(
                "Replay loss batch %d: %d of %d rows, %.1fs",
                batch_index,
                len(output),
                len(manifest),
                time.monotonic() - batch_start,
            )
            batch_start = time.monotonic()
    return output


def replay_loss_metrics(rows: Sequence[ReplayDocumentLoss]) -> dict[str, float]:
    """Token-micro mean loss and bits per byte over all documents, kept documents and flagged documents."""
    groups = {
        "all": list(rows),
        "kept": [row for row in rows if row.category is None],
        "flagged": [row for row in rows if row.category is not None],
    }
    metrics: dict[str, float] = {}
    for name, group in groups.items():
        if not group:
            continue
        loss = sum(row.loss_sum for row in group)
        metrics[f"{REPLAY_LOSS_METRIC_PREFIX}/{name}/loss"] = loss / sum(row.scored_tokens for row in group)
        metrics[f"{REPLAY_LOSS_METRIC_PREFIX}/{name}/bpb"] = loss / np.log(2) / sum(row.scored_bytes for row in group)
        metrics[f"{REPLAY_LOSS_METRIC_PREFIX}/{name}/documents"] = float(len(group))
    return metrics


def evaluate_replay_loss(
    params: Transformer,
    config: ReplayLossEvalConfig,
    *,
    mesh: Mesh,
    mp: jmp.Policy,
    implementation: MoeImplementation,
    record: dict[str, Any],
) -> dict[str, float]:
    """Score ``params`` and return metrics on process 0, writing its artifact; other processes return nothing."""
    rows = score_replay_documents(params, config, mesh=mesh, mp=mp, implementation=implementation)
    if jax.process_index() != 0:
        return {}
    metrics = replay_loss_metrics(rows)
    output = StoragePath(config.output_path)
    (output / REPLAY_LOSS_DOCUMENTS_NAME).write_text(
        "".join(json.dumps(dataclasses.asdict(row), sort_keys=True) + "\n" for row in rows)
    )
    (output / REPLAY_LOSS_METRICS_NAME).write_text(
        json.dumps({**record, "metrics": dict(sorted(metrics.items()))}, indent=2, sort_keys=True)
    )
    return metrics
