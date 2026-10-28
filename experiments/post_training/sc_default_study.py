# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the bounded three-arm Snowball default study through Marin artifacts."""

from dataclasses import replace

import click
import yaml
from fray.types import ResourceConfig
from iris.rpc import job_pb2
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.models import qwen2_5_0_5b_instruct
from experiments.post_training.async_rl import SNOWBALL_RECIPE
from experiments.post_training.inventory.train_quick_loop import training_recipe
from experiments.post_training.sc_default_data import SOURCE_URI, prepare_study_data
from experiments.post_training.tim_cp_qualification import GRUG_HF, GRUG_REVISION, GRUG_URI, qualification_recipe

INITIAL_SEEDS = (1701, 2903, 4517)
EXTENSION_SEEDS = (6101, 7919, 10007)
ARMS = ("tis", "sc128", "sc32")


def study_recipe(arm: str, seed: int, smoke: bool) -> tuple[dict, SkyRLRolePlan]:
    steps = 2 if smoke else 50
    cfg = training_recipe(steps)
    if smoke:
        small, plan, _ = qualification_recipe(True)
        cfg["trainer"]["policy"] = small["trainer"]["policy"]
        cfg["trainer"]["ref"] = small["trainer"]["ref"]
        cfg["trainer"]["flash_attn"] = True
        cfg["generator"]["gpu_memory_utilization"] = 0.35
        cfg["generator"].update(max_num_seqs=32, max_num_batched_tokens=4096)
        plan = replace(plan, train_batch_size=16, policy_mini_batch_size=16)
        cfg["context_budget"] = {"request_window_tokens": 16384, "max_new_tokens_per_turn": 2048, "max_turns": 1}
    else:
        # Retain the healthy 40-GPU Snowball geometry and its response budget.
        plan = SNOWBALL_RECIPE.role_plan
    _materialize_role_plan_config(cfg, plan)
    trainer = cfg["trainer"]
    trainer.update(
        max_steps=steps,
        epochs=50,
        seed=seed,
        resume_mode="none",
        use_sample_packing=True,
        project_name="score-centering-default-study",
        logger=["console", "wandb"],
        tracker_commit_each_step=True,
        eval_before_train=True,
        eval_interval=1 if smoke else 25,
        eval_num_prompts=None,
        eval_batch_size=10 if smoke else 100,
        dump_eval_results=True,
        ckpt_interval=steps,
        hf_save_interval=steps,
        export_hf_artifact=True,
        callbacks=None,
    )
    trainer["algorithm"].update(
        off_policy_correction="tis",
        score_centering_enabled=arm != "tis",
        score_centering_top_k=32 if arm == "sc32" else 128,
        score_centering_probe={"interval": 0},
        # Qualify reference scoring during the tiny startup. The main healthy
        # recipe has no reference auxiliary loss, so its reference cost is zero.
        use_kl_loss=smoke,
        kl_loss_coef=0.01,
    )
    trainer["policy"]["optimizer_config"].update(
        lr=1e-6,
        adam_betas=[0.9, 0.999],
        weight_decay=0.0,
        max_grad_norm=0.5,
        scheduler="constant_with_warmup",
        num_warmup_steps=0,
    )
    trainer["rollout_buffer"].update(
        batch_policy="full_batch" if smoke else "rolling",
        max_staleness_steps=0 if smoke else 2,
        max_in_flight=16 if smoke else 256,
    )
    cfg["data"].update(shuffle=True)
    cfg["environment"]["task_sessions"].pop("daytona", None)
    cfg["config_groups"] = {"sc_default_study": "default"}
    cfg["generator"].update(enable_http_endpoint=False, weight_sync_transport="broadcast")
    cfg["generator"].setdefault("engine_init_kwargs", {}).update(
        logprobs_mode="processed_logprobs",
        validate_rollout_logprob_sampling=True,
    )
    cfg["trajectory_runner"] = {
        "max_concurrent_tasks": 32 if smoke else 256,
        "rollout_workers": {"num_workers": 2 if smoke else 4, "cpus_per_worker": 2 if smoke else 8},
    }
    cfg["extra_env"].update(SC_DEFAULT_STUDY="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if smoke:
        cfg["extra_env"].update(NCCL_NVLS_ENABLE="0")
    return cfg, plan


def build_study(version: str, *, arm: str, seed: int, smoke: bool, cluster: str) -> ArtifactStep:
    cfg, plan = study_recipe(arm, seed, smoke)
    source = ArtifactStep.adopt(user_owned_name("documents/sc-default-study-source"), "2026.10.23.2", SOURCE_URI)
    data = apply(
        user_owned_name("documents/sc-default-study-smoke" if smoke else "documents/sc-default-study"),
        remote(prepare_study_data, resources=ResourceConfig.with_cpu(cpu=2, ram="32g")),
        version="2026.10.28.1",
        source_path=source,
        output_path=OUT,
        smoke=smoke,
    )
    if smoke:
        model = ArtifactHfModel(
            qwen2_5_0_5b_instruct,
            "Qwen/Qwen2.5-0.5B-Instruct",
            "7ae557604adf67be50417f59c2c2f167def9a775",
            relative_path="",
        )
    else:
        weights = ArtifactStep.adopt(
            user_owned_name("models/sc-default-grug-8c76b22a"), "2026.10.28", GRUG_URI, kind=LevanterCheckpoint
        )
        model = ArtifactHfModel(weights, GRUG_HF, GRUG_REVISION, relative_path="")
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"checkpoints/sc-default-study/{'qwen' if smoke else arm}-seed{seed}"),
            version=version,
            config_yaml=yaml.safe_dump(cfg),
            model=model,
            train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
            validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
            topology=SkyRLTopology(num_nodes=plan.num_nodes, gpus_per_node=8, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=seed,
        ),
        IrisSkyRLExecution(
            cluster=cluster,
            cluster_config=f"lib/iris/config/{cluster}.yaml",
            cpu=64,
            memory="512GB" if smoke else "1800GB",
            disk="2TB",
            priority="interactive" if smoke else "batch",
            max_retries=0,
            target_cluster=cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=4 if smoke else 36,
            job_timeout_seconds=10800 if smoke else 24 * 3600,
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            wandb_entity="marin-community",
            required_env_vars=("TOGETHER_API_KEY",),
        ),
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option("--arm", type=click.Choice(ARMS), required=True)
@click.option("--seed", type=int, required=True)
@click.option("--smoke/--full", default=False)
@click.option("--cluster", type=click.Choice(("cw-rno2a", "cw-us-east-02a")), default="cw-rno2a")
@rl_build_options
def main(arm: str, seed: int, smoke: bool, cluster: str) -> ArtifactStep:
    return build_study(resolve_version("sc-default-study", None), arm=arm, seed=seed, smoke=smoke, cluster=cluster)


if __name__ == "__main__":
    main()
