# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score one Snowball 67B checkpoint on a held-out replay set split by junk-filter verdict, on H100 nodes.

The data is a cache built by ``experiments.sft_cold_start.heldout_replay``: one document per row, every
row ``ROW_TOKENS`` long. Each node scores ``ROWS_PER_NODE`` rows per batch, the same tokens per node as one
262,144-token MRCR row, with the sequence sharded over its eight GPUs.
"""

import dataclasses
import os

import click
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.storage_path import prefix_join

from experiments.datasets.mrcr import MRCR_CONTEXT_CAP
from experiments.grug.moe_hero_ep.hero_recipe import (
    DEFAULT_WANDB_PROJECT,
    HERO_TENSORSTORE_CACHE_BYTES,
    HeroThroughputResult,
    hero_trainer_config,
)
from experiments.grug.moe_hero_ep.launch_checkpoint_eval import CheckpointEvalConfig, run_checkpoint_eval
from experiments.grug.moe_hero_ep.launch_snowball_mrcr_eval import SNOWBALL_DROPLESS_MOE_IMPLEMENTATION
from experiments.grug.moe_hero_ep.replay_loss_eval import ReplayLossEvalConfig
from experiments.grug.moe_hero_ep.snowball_recipe import (
    SNOWBALL_67B_MODEL_CONFIG,
    SNOWBALL_GPUS_PER_NODE,
    SNOWBALL_MIXED_PRECISION,
    SNOWBALL_NODE_CPU,
    SNOWBALL_NODE_DISK,
    SNOWBALL_NODE_RAM,
)
from experiments.marin_tokenizer import marin_tokenizer

ROW_TOKENS = 65_536
ROWS_PER_NODE = MRCR_CONTEXT_CAP // ROW_TOKENS


def build_snowball_replay_loss_eval(
    *,
    run_id: str,
    checkpoint_path: str,
    data_path: str,
    nodes: int = 1,
    max_batches: int | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """``nodes`` H100 nodes scoring the Snowball checkpoint at ``checkpoint_path`` on the cache at ``data_path``."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    model = dataclasses.replace(SNOWBALL_67B_MODEL_CONFIG, max_seq_len=MRCR_CONTEXT_CAP)
    batch_size = nodes * ROWS_PER_NODE
    resources = ResourceConfig.with_gpu(
        "H100",
        count=SNOWBALL_GPUS_PER_NODE,
        cpu=SNOWBALL_NODE_CPU,
        ram=SNOWBALL_NODE_RAM,
        disk=SNOWBALL_NODE_DISK,
        replicas=nodes,
    )
    name = f"eval/replay-loss-snowball/{run_id}"
    version = resolve_version(name, version)
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT

    def build_config(ctx: StepContext) -> CheckpointEvalConfig:
        trainer = dataclasses.replace(
            hero_trainer_config(
                run_id=run_id,
                seed=0,
                train_batch_size=batch_size,
                context_axis_size=SNOWBALL_GPUS_PER_NODE,
                num_train_steps=1,
                profiler=ProfilerConfig(enabled=False),
                tracker=WandbConfig(
                    save_code=False,
                    entity="marin-community",
                    project=wandb_project,
                    tags=["grug", "moe", "snowball", "checkpoint-eval", "h100", "replay-loss"],
                    group="snowball-replay-loss-eval",
                    name=run_id,
                    replicate_path=ctx.output_path,
                ),
                watch=WatchConfig(interval=0),
                # The eval never saves; the checkpointer only needs a path inside this run's own tree.
                checkpointer=CheckpointerConfig(base_path=prefix_join(ctx.output_path, "checkpoints")),
            ),
            mp=SNOWBALL_MIXED_PRECISION,
        )
        data = LmDataConfig(
            tokenizer=marin_tokenizer,
            cache_dir=None,
            components={},
            train_weights={},
            auto_build_caches=False,
        )
        return CheckpointEvalConfig(
            model=model,
            data=data,
            trainer=trainer,
            resources=ctx.runtime_arg("resources"),
            checkpoint_path=checkpoint_path,
            seq_lens=(),
            eval_tokens_per_batch=batch_size * ROW_TOKENS,
            context_axis_size=SNOWBALL_GPUS_PER_NODE,
            replay_loss=ReplayLossEvalConfig(
                data_path=data_path,
                output_path=ctx.output_path,
                tokenizer=marin_tokenizer,
                batch_size=batch_size,
                seq_len=ROW_TOKENS,
                max_batches=max_batches,
            ),
            dropless_moe_implementation=SNOWBALL_DROPLESS_MOE_IMPLEMENTATION,
            tensorstore_cache_bytes=HERO_TENSORSTORE_CACHE_BYTES,
            processes_per_task=SNOWBALL_GPUS_PER_NODE,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_checkpoint_eval,
        build_config=build_config,
        runtime_args={"resources": resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for the artifact and W&B names.")
@click.option("--checkpoint", "checkpoint_path", required=True, help="Concrete checkpoint directory (step-N).")
@click.option("--data", "data_path", required=True, help="Held-out set prefix holding validation/ and manifest.jsonl.")
@click.option("--nodes", type=click.IntRange(min=1), default=1, show_default=True, help="H100 nodes.")
@click.option("--max-batches", type=click.IntRange(min=1), default=None, help="Stop early, as a smoke test.")
@build_options
def main(
    run_id: str, checkpoint_path: str, data_path: str, nodes: int, max_batches: int | None
) -> ArtifactStep[HeroThroughputResult]:
    return build_snowball_replay_loss_eval(
        run_id=run_id, checkpoint_path=checkpoint_path, data_path=data_path, nodes=nodes, max_batches=max_batches
    )


if __name__ == "__main__":
    main()
