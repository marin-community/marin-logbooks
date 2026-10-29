# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense optimizer ladder on the original Hero mixture, with small-budget LR calibration."""

import dataclasses
import json
import math
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import DEFAULT_JAX_CONFIG, TrainerConfig
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.experiment.namespacing import user_owned_name
from marin.training.training import LevanterCheckpoint, checkpoint_resume_paths, temporary_checkpoint_base_path
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.fast_track.dense_optimizers import DenseOptimizer, build_dense_optimizer_config
from experiments.grug.fast_track.launch import SEQ_LEN, H100LadderRung, SmallShape, _h100_ladder_model
from experiments.grug.fast_track.model import GrugModelConfig
from experiments.grug.fast_track.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, _compute_flops, run_grug
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    HARRIER_MIX_2026_08_18_STORE,
    HARRIER_MIX_2026_08_18_TAG,
    harrier_mix_2026_08_18_data_config,
)
from experiments.grug.moe_hero_ep.hero_recipe import validation_datasets

BUDGETS = (1e18, 3e18, 1e19, 3e19, 1e20, 3e20, 1e21)
LR_MULTIPLIERS = (0.3, 1.0, 3.0)
GPUS_PER_RUN = 8
VOCAB_SIZE = 128256
WANDB_PROJECT = "marin_moe"
METRIC = "eval/paloma/macro_loss"


@dataclasses.dataclass(frozen=True)
class DenseRung:
    budget: float
    model: GrugModelConfig
    batch_size: int
    steps: int
    flops_per_example: float

    @property
    def tokens(self) -> int:
        return self.batch_size * SEQ_LEN * self.steps


def ladder_rung(budget: float) -> DenseRung:
    """Choose a shape near 20 tokens per parameter, then round down to the FLOP budget.

    The model FLOP estimate includes attention and the head. Shape selection uses the
    embedding/head plus transformer matrix parameter estimate; optimizers share the result.
    """
    candidates = []
    for width, depth in (
        (256, 4),
        (384, 4),
        (512, 6),
        (640, 8),
        (768, 8),
        (896, 10),
        (1024, 12),
        (1280, 14),
        (1536, 16),
        (1792, 20),
        (2048, 24),
        (2304, 26),
        (2560, 28),
    ):
        heads = width // 128
        kv_heads = max(kv for kv in range(1, max(1, heads // 4) + 1) if heads % kv == 0)
        shape = SmallShape(width, depth, heads, kv_heads, kv_heads)
        model = dataclasses.replace(
            _h100_ladder_model(H100LadderRung(shape, GPUS_PER_RUN, 32), dense=True),
            vocab_size=VOCAB_SIZE,
        )
        flops_per_example, _ = _compute_flops(model_config=model)
        params = 2 * width * VOCAB_SIZE + depth * (2 * width * heads * 128 + 2 * width * kv_heads * 128 + 9 * width**2)
        target_flops = flops_per_example / SEQ_LEN * 20 * params
        candidates.append((abs(math.log(target_flops / budget)), model, flops_per_example))
    _, model, flops_per_example = min(candidates, key=lambda item: item[0])
    batch_size = 128
    steps = math.floor(budget / (batch_size * flops_per_example))
    return DenseRung(budget, model, batch_size, steps, flops_per_example)


def optimizer_config(optimizer: DenseOptimizer, rung: DenseRung, multiplier: float):
    # These are pilot centers, not claims of tuned optimality. Transfer the selected
    # multiplier with a common sqrt(batch) * tokens^-0.3 scaling from the first rung.
    reference = ladder_rung(BUDGETS[0])
    scale = math.sqrt(rung.batch_size / reference.batch_size) * (reference.tokens / rung.tokens) ** 0.3
    adam_lr = 1e-3 * multiplier * scale
    matrix_lr = (
        {
            DenseOptimizer.ADAMW: 1e-3,
            DenseOptimizer.MUON: 0.02,
            DenseOptimizer.ADAMH: 0.01,
            DenseOptimizer.MUONH: math.sqrt(0.02 * 0.1),
        }[optimizer]
        * multiplier
        * scale
    )
    return build_dense_optimizer_config(
        optimizer=optimizer,
        learning_rate=matrix_lr,
        adam_lr=adam_lr,
        weight_decay=0.0 if optimizer in (DenseOptimizer.ADAMH, DenseOptimizer.MUONH) else 0.1,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        warmup=0.01,
        lr_schedule="cosine",
        min_lr_ratio=0.1,
    )


def build_run(
    *, campaign: str, version: str, rung: DenseRung, optimizer: DenseOptimizer, multiplier: float, smoke: bool = False
) -> ArtifactStep[LevanterCheckpoint]:
    run_id = f"{campaign}-{rung.budget:.0e}-{optimizer.value}-lr{multiplier:g}"
    if smoke:
        run_id += "-smoke"
    validation = [] if smoke else validation_datasets()
    opt = optimizer_config(optimizer, rung, multiplier)
    resources = ResourceConfig.with_gpu("H100", count=GPUS_PER_RUN, cpu=32, ram="600g", disk="900g")

    def build_config(ctx: StepContext) -> GrugRunConfig:
        checkpoint_path = prefix_join(ctx.output_path, "checkpoints")
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=rung.batch_size,
            num_train_steps=20 if smoke else rung.steps,
            jax_config=dict(DEFAULT_JAX_CONFIG),
            profiler=ProfilerConfig(enabled=False),
            mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project=WANDB_PROJECT,
                name=run_id,
                group=campaign,
                save_code=False,
                replicate_path=ctx.output_path,
                tags=[
                    "hero-dense-optimizers",
                    optimizer.value,
                    HARRIER_MIX_2026_08_18_TAG,
                    f"budget-{rung.budget:.0e}",
                    "smoke" if smoke else "ladder",
                ],
            ),
            watch=WatchConfig(interval=10),
            progress_watchdog=ProgressWatchdogConfig(
                step_timeout=timedelta(minutes=15),
                process_timeout=timedelta(hours=1),
                startup_timeout=timedelta(hours=1),
            ),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            load_checkpoint_path=checkpoint_resume_paths(ctx.output_path),
            checkpointer=CheckpointerConfig(
                base_path=checkpoint_path,
                temporary_base_path=temporary_checkpoint_base_path(ctx.output_path),
                save_interval=timedelta(minutes=15),
                keep=None,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
        )
        return GrugRunConfig(
            model=rung.model,
            data=harrier_mix_2026_08_18_data_config(
                ctx=ctx,
                total_steps=rung.steps,
                batch_size=rung.batch_size,
                max_seq_len=SEQ_LEN,
                enable_simulated_epoching=False,
                validation=validation,
            ),
            resources=ctx.runtime_arg("resources"),
            tensorstore_cache_bytes=1_000_000_000,
            optimizer=opt,
            trainer=GrugTrainerConfig(
                trainer=trainer, data_seed=0, save_checkpoints=not smoke, expert_axis_size=1, replica_axis_size=1
            ),
            eval=(
                None
                if smoke
                else GrugEvalConfig(
                    steps_per_eval=max(1, rung.steps // 4), eval_batch_size=GPUS_PER_RUN, compute_bpb=True
                )
            ),
            stop_after_steps=min(20, rung.steps) if smoke else rung.steps,
            processes_per_task=GPUS_PER_RUN,
            max_retries_failure=0 if smoke else 3,
            max_task_failures=0 if smoke else 3,
        )

    return ArtifactStep(
        name=user_owned_name(f"grug/{run_id}"),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug,
        build_config=build_config,
        deps=(HARRIER_MIX_2026_08_18_STORE, *validation),
        runtime_args={"resources": resources},
    )


def manifest(campaign: str, version: str) -> dict:
    rungs = [ladder_rung(budget) for budget in BUDGETS]
    return {
        "campaign": campaign,
        "version": version,
        "seed": 0,
        "gpus_per_run": GPUS_PER_RUN,
        "data": HARRIER_MIX_2026_08_18_STORE.adopt_source,
        "simulated_epoching": False,
        "metric": METRIC,
        "calibration_multipliers": LR_MULTIPLIERS,
        "rungs": [
            {
                "budget": r.budget,
                "width": r.model.hidden_dim,
                "layers": r.model.num_layers,
                "batch_size": r.batch_size,
                "steps": r.steps,
                "tokens": r.tokens,
                "actual_flops": r.flops_per_example * r.batch_size * r.steps,
                "optimizers": {o.value: dataclasses.asdict(optimizer_config(o, r, 1.0)) for o in DenseOptimizer},
            }
            for r in rungs
        ],
    }


@click.command()
@click.option("--campaign", required=True)
@click.option("--version", required=True)
@click.option("--max-concurrent", type=click.IntRange(1, 8), default=8)
@click.option("--run", "do_run", is_flag=True)
def main(campaign: str, version: str, max_concurrent: int, do_run: bool) -> None:
    """Print the plan, or run smoke gates, LR calibration, then the selected ladder."""
    plan = manifest(campaign, version)
    click.echo(json.dumps(plan, indent=2, default=str))
    if not do_run:
        return
    with build_context(BuildContext(versions=VersionCodex(default=version))):
        # Validate both ends before committing to full runs; all eight fit in the cap.
        smokes = [
            build_run(campaign=campaign, version=version, rung=ladder_rung(b), optimizer=o, multiplier=1.0, smoke=True)
            for b in (BUDGETS[0], BUDGETS[-1])
            for o in DenseOptimizer
        ]
        run(*smokes, max_concurrent=max_concurrent)
        calibrations = [
            (
                o,
                m,
                build_run(campaign=campaign, version=version, rung=ladder_rung(BUDGETS[0]), optimizer=o, multiplier=m),
            )
            for o in DenseOptimizer
            for m in LR_MULTIPLIERS
        ]
        artifacts = run(*(h for _, _, h in calibrations), max_concurrent=max_concurrent)
        scores = {o: [] for o in DenseOptimizer}
        for (optimizer, multiplier, _), artifact in zip(calibrations, artifacts, strict=True):
            loss = float(artifact.training_metrics().summary[METRIC])
            if not math.isfinite(loss):
                raise ValueError(f"Non-finite calibration loss for {optimizer} at {multiplier}: {loss}")
            scores[optimizer].append((loss, multiplier))
        selected = {optimizer: min(values)[1] for optimizer, values in scores.items()}
        selection = {"scores": scores, "selected": selected, "plan": plan}
        root = prefix_join(smokes[0].path().rsplit("/", 2)[0], f"{campaign}-selection-{version}.json")
        StoragePath(root).write_text(json.dumps(selection, indent=2, default=str))
        click.echo(f"Calibration selection: {root}\n{json.dumps(selected)}")
        handles = [
            build_run(campaign=campaign, version=version, rung=ladder_rung(b), optimizer=o, multiplier=selected[o])
            for b in reversed(BUDGETS[1:])
            for o in DenseOptimizer
        ]
        run(*handles, max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
