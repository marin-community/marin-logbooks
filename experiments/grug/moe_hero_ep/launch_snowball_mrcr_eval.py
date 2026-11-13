# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score one Snowball 67B checkpoint on paired MRCR at 262,144 tokens, on H100 nodes.

The hero checkpoint eval does the scoring; this launcher supplies the Snowball model, H100 resources and
the MoE backend that runs on them. One node holds one 262,144-token row across its eight GPUs as context
shards, so the batch is one row per node.
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

from experiments.datasets.mrcr import MRCR_CONTEXT_CAP, mrcr_dataset
from experiments.grug.moe_hero_ep.hero_recipe import (
    DEFAULT_WANDB_PROJECT,
    HERO_TENSORSTORE_CACHE_BYTES,
    HeroThroughputResult,
    hero_trainer_config,
)
from experiments.grug.moe_hero_ep.launch_checkpoint_eval import CheckpointEvalConfig, run_checkpoint_eval
from experiments.grug.moe_hero_ep.mrcr_eval import MrcrEvalConfig
from experiments.grug.moe_hero_ep.snowball_recipe import (
    SNOWBALL_67B_MODEL_CONFIG,
    SNOWBALL_GPUS_PER_NODE,
    SNOWBALL_MIXED_PRECISION,
    SNOWBALL_NODE_CPU,
    SNOWBALL_NODE_DISK,
    SNOWBALL_NODE_RAM,
)
from experiments.marin_tokenizer import marin_tokenizer

# The H100 backend. The eval's default backend needs Blackwell.
SNOWBALL_DROPLESS_MOE_IMPLEMENTATION = "sonic"


def build_snowball_mrcr_eval(
    *,
    run_id: str,
    checkpoint_path: str,
    nodes: int = 1,
    mrcr_max_batches: int | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """``nodes`` H100 nodes scoring the Snowball checkpoint at ``checkpoint_path`` on paired MRCR."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    model = dataclasses.replace(SNOWBALL_67B_MODEL_CONFIG, max_seq_len=MRCR_CONTEXT_CAP)
    batch_size = nodes
    resources = ResourceConfig.with_gpu(
        "H100",
        count=SNOWBALL_GPUS_PER_NODE,
        cpu=SNOWBALL_NODE_CPU,
        ram=SNOWBALL_NODE_RAM,
        disk=SNOWBALL_NODE_DISK,
        replicas=nodes,
    )
    name = f"eval/mrcr-snowball/{run_id}"
    version = resolve_version(name, version)
    mrcr_data = mrcr_dataset(tokenizer=marin_tokenizer)
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
                    tags=["grug", "moe", "snowball", "checkpoint-eval", "h100", "mrcr"],
                    group="snowball-mrcr-eval",
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
            validation_remainder="pad",
        )
        return CheckpointEvalConfig(
            model=model,
            data=data,
            trainer=trainer,
            resources=ctx.runtime_arg("resources"),
            checkpoint_path=checkpoint_path,
            seq_lens=(),
            eval_tokens_per_batch=batch_size * MRCR_CONTEXT_CAP,
            context_axis_size=SNOWBALL_GPUS_PER_NODE,
            mrcr=MrcrEvalConfig(
                data_path=ctx.artifact_path(mrcr_data),
                output_path=ctx.output_path,
                tokenizer=marin_tokenizer,
                batch_size=batch_size,
                max_batches=mrcr_max_batches,
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
        deps=(mrcr_data,),
        runtime_args={"resources": resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for the artifact and W&B names.")
@click.option("--checkpoint", "checkpoint_path", required=True, help="Concrete checkpoint directory (step-N).")
@click.option("--nodes", type=click.IntRange(min=1), default=1, show_default=True, help="H100 nodes; one row each.")
@click.option("--mrcr-max-batches", type=click.IntRange(min=1), default=None, help="Stop MRCR early, as a smoke test.")
@build_options
def main(
    run_id: str, checkpoint_path: str, nodes: int, mrcr_max_batches: int | None
) -> ArtifactStep[HeroThroughputResult]:
    return build_snowball_mrcr_eval(
        run_id=run_id, checkpoint_path=checkpoint_path, nodes=nodes, mrcr_max_batches=mrcr_max_batches
    )


if __name__ == "__main__":
    main()
