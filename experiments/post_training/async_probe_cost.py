# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare periodic probes using the normal asynchronous Snowball training loop."""

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
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
from experiments.post_training.cat_count_canary.launcher import training_config
from experiments.post_training.inventory.train_quick_loop import training_recipe

GRUG_URI = "s3://marin-rno2a/marin/users/dml/models/grug-67b-a2b-datakit-sft-262k-2026.10.05-8c76b22a/2026.10.22/hf"
GRUG_HF = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05"
GRUG_REVISION = "8c76b22afc38f5e95360c83ca88008f00b5ae1e0"
DATA_URI = "s3://marin-rno2a/marin/users/romain/documents/sc-default-study/2026.10.28.1"
SMOKE_DATA_URI = "s3://marin-rno2a/marin/users/romain/documents/sc-default-study-smoke/2026.10.28.1"


def probe_recipe(interval: int, seed: int, smoke: bool) -> tuple[dict, SkyRLRolePlan]:
    steps = 12 if smoke else 30
    cfg = training_recipe(steps)
    plan = SNOWBALL_RECIPE.role_plan
    if smoke:
        small = training_config(preset="dry", lane="sync", checkpoint=False, export=False)
        for role in ("policy", "ref"):
            cfg["trainer"][role] = small["trainer"][role]
            cfg["trainer"][role]["megatron_config"]["pipeline_model_parallel_size"] = 2
        cfg["trainer"]["flash_attn"] = True
        cfg["generator"].update(gpu_memory_utilization=0.35, max_num_seqs=32, max_num_batched_tokens=4096)
        cfg["context_budget"] = {"request_window_tokens": 16384, "max_new_tokens_per_turn": 2048, "max_turns": 1}
        plan = SkyRLRolePlan(
            colocate_all=True,
            policy_num_nodes=1,
            policy_num_gpus_per_node=8,
            num_inference_engines=8,
            inference_engine_tensor_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=1,
            inference_engine_expert_parallel_size=1,
            train_batch_size=16,
            policy_mini_batch_size=16,
            micro_train_batch_size_per_gpu=1,
            n_samples_per_prompt=4,
        )
    _materialize_role_plan_config(cfg, plan)
    trainer = cfg["trainer"]
    trainer.update(
        max_steps=steps,
        epochs=50,
        seed=seed,
        resume_mode="none",
        use_sample_packing=True,
        project_name="asynchronous-probe-cost",
        logger=["console", "wandb"],
        tracker_commit_each_step=True,
        eval_before_train=False,
        eval_interval=-1,
        ckpt_interval=-1,
        hf_save_interval=-1,
        export_hf_artifact=False,
        callbacks=None,
    )
    trainer["algorithm"].update(
        off_policy_correction="tis",
        score_centering_enabled=False,
        score_centering_top_k=128,
        score_centering_probe={"interval": interval},
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
    trainer["rollout_buffer"].update(batch_policy="rolling", max_staleness_steps=2, max_in_flight=32 if smoke else 256)
    cfg["data"].update(shuffle=True)
    cfg["environment"]["task_sessions"].pop("daytona", None)
    cfg["config_groups"] = {"sc_default_study": "default"}
    cfg["generator"].update(enable_http_endpoint=False, weight_sync_transport="broadcast")
    cfg["generator"].setdefault("engine_init_kwargs", {}).update(
        logprobs_mode="processed_logprobs", validate_rollout_logprob_sampling=True
    )
    cfg["trajectory_runner"] = {
        "max_concurrent_tasks": 32 if smoke else 256,
        "rollout_workers": {"num_workers": 2 if smoke else 4, "cpus_per_worker": 2 if smoke else 8},
    }
    cfg["extra_env"].update(ASYNC_PROBE_COST="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if smoke:
        cfg["extra_env"]["NCCL_NVLS_ENABLE"] = "0"
    return cfg, plan


def build_comparison(version: str, interval: int, seed: int, smoke: bool) -> ArtifactStep:
    cfg, plan = probe_recipe(interval, seed, smoke)
    data = ArtifactStep.adopt(
        user_owned_name("documents/async-probe-cost-smoke" if smoke else "documents/async-probe-cost"),
        "2026.10.28.1",
        SMOKE_DATA_URI if smoke else DATA_URI,
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
            user_owned_name("models/async-probe-grug-8c76b22a"), "2026.10.28", GRUG_URI, kind=LevanterCheckpoint
        )
        model = ArtifactHfModel(weights, GRUG_HF, GRUG_REVISION, relative_path="")
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"checkpoints/async-probe-cost/{'qwen' if smoke else 'grug'}-i{interval}-seed{seed}"),
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
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=64,
            memory="512GB" if smoke else "1800GB",
            disk="2TB",
            priority="interactive" if smoke else "batch",
            max_retries=0,
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=4 if smoke else 12,
            job_timeout_seconds=10800 if smoke else 8 * 3600,
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            wandb_entity="marin-community",
            required_env_vars=("TOGETHER_API_KEY",),
        ),
        export_hf=False,
    )


@click.command(help=__doc__)
@click.option("--interval", type=click.Choice(("0", "10", "20")), required=True)
@click.option("--seed", type=click.Choice(("1701", "2903", "4517")), required=True)
@click.option("--smoke/--full", default=False)
@rl_build_options
def main(interval: str, seed: str, smoke: bool) -> ArtifactStep:
    return build_comparison(resolve_version("async-probe-cost", None), int(interval), int(seed), smoke)


if __name__ == "__main__":
    main()
