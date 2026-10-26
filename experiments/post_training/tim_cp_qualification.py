# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the session-only packed-CP correctness and cost qualification."""

import json
from pathlib import Path

import click
import yaml
from fray.types import ResourceConfig
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
from experiments.post_training.cat_count_canary.data import TRAIN_FILENAME, VALIDATION_FILENAME, cat_count_data_step
from experiments.post_training.cat_count_canary.launcher import training_config
from experiments.post_training.sc_performance_data import SOURCE_URI, prepare_performance_data

GRUG_URI = "s3://marin-rno2a/marin/users/dml/models/grug-67b-a2b-datakit-sft-262k-2026.10.05-8c76b22a/2026.10.22/hf"
GRUG_HF = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05"
GRUG_REVISION = "8c76b22afc38f5e95360c83ca88008f00b5ae1e0"


def qualification_recipe(smoke):
    plan = SkyRLRolePlan(
        colocate_all=smoke,
        policy_num_nodes=1 if smoke else 10,
        policy_num_gpus_per_node=8,
        num_inference_engines=8 if smoke else 2,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=1 if smoke else 4,
        inference_engine_expert_parallel_size=1 if smoke else 4,
        train_batch_size=4 if smoke else 8,
        policy_mini_batch_size=4 if smoke else 8,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=4,
    )
    counts = (4, 6, 8, 12) if smoke else (32, 48, 64, 96)
    cfg = training_config(
        preset="dry",
        lane="sync",
        batch_size=plan.train_batch_size,
        group_size=plan.n_samples_per_prompt,
        micro_train_batch_size=1,
        train_ns=counts,
        checkpoint=False,
        export=False,
    )
    if not smoke:
        reference = json.loads(Path(__file__).with_name("tim_cp_reference_recipe.json").read_text())
        for key in (
            "algorithm",
            "policy",
            "ref",
            "flash_attn",
            "attn_backend",
            "gradient_checkpointing",
            "gradient_checkpointing_use_reentrant",
            "bf16",
        ):
            cfg["trainer"][key] = reference[key]
        cfg["generator"].update(reference["generator"])
        cfg["generator"]["engine_init_kwargs"].pop("max_model_len", None)
        cfg["trainer"]["algorithm"].pop("max_seq_len", None)
        cfg["context_budget"] = {"request_window_tokens": 102400, "max_new_tokens_per_turn": 49152, "max_turns": 1}
        cfg["trainer"]["rollout_buffer"].update(batch_policy="rolling", max_staleness_steps=2, max_in_flight=16)
    _materialize_role_plan_config(cfg, plan)
    trainer = cfg["trainer"]
    trainer["placement"]["policy_strict_spread_pg"] = not smoke
    trainer.update(
        use_sample_packing=True,
        resume_mode="none",
        max_steps=2 if smoke else 6,
        epochs=4,
        update_epochs_per_batch=1,
        logger=["console", "wandb"],
        project_name="score-centering-cp-packing",
        eval_before_train=False,
        eval_interval=-1,
        ckpt_interval=-1,
        hf_save_interval=-1,
        export_hf_artifact=False,
        micro_forward_batch_size_per_gpu=1,
    )
    trainer["algorithm"].update(
        score_centering_enabled=True,
        score_centering_probe={"interval": 0},
        off_policy_correction="tis",
        advantage_estimator="rloo_n",
        group_advantage_min_size=4,
    )
    trainer["algorithm"]["group_admission"] = {"stall_timeout": 600}
    trainer["callbacks"] = [
        {"type": "inference_stats", "log_every_steps": 1, "log_to_console": True, "log_to_tracker": True}
    ]
    for role in ("policy", "ref"):
        trainer[role]["megatron_config"].update(
            context_parallel_size=2 if smoke else 5,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=1 if smoke else 8,
            expert_tensor_parallel_size=1,
        )
        trainer[role]["sequence_parallel_size"] = 1
    if smoke:
        trainer["algorithm"].update(use_kl_loss=True, kl_loss_coef=0.01)
        cfg["generator"]["gpu_memory_utilization"] = 0.35
        cfg["trainer"]["flash_attn"] = True
        cfg["trainer"]["policy"]["megatron_config"]["transformer_config_kwargs"] = {"cp_comm_type": "a2a"}
    cfg["trajectory_runner"] = {
        "max_concurrent_tasks": 8 if smoke else 16,
        "rollout_workers": {"num_workers": 2, "cpus_per_worker": 2},
    }
    cfg["extra_env"] = {"TIM_CP_QUALIFICATION": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if smoke:
        cfg["extra_env"]["NCCL_NVLS_ENABLE"] = "0"
    return cfg, plan, counts


def performance_recipe(smoke):
    cfg, plan, counts = qualification_recipe(smoke)
    trainer = cfg["trainer"]
    trainer.update(max_steps=15 if smoke else 30, epochs=40, project_name="score-centering-performance")
    trainer["policy"]["optimizer_config"]["lr"] = 0.0
    trainer["rollout_buffer"].update(
        batch_policy="full_batch", max_staleness_steps=0, max_in_flight=plan.train_batch_size
    )
    trainer["algorithm"].update(score_centering_top_k=128, rloo_n_filter_zero_reward_groups=False)
    cfg["data"]["shuffle"] = False
    cfg["context_budget"] = {
        "request_window_tokens": 16384 if smoke else 102400,
        "max_new_tokens_per_turn": 2048 if smoke else 49152,
        "max_turns": 1,
    }
    cfg["extra_env"]["TIM_CP_MODE"] = "performance"
    cfg["generator"].setdefault("engine_init_kwargs", {}).update(
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
        logprobs_mode="processed_logprobs",
        logits_processors=["skyrl_train.inference_engines.vllm.sc_sampled_capture.SampledScoreCapture"],
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "/tmp/sc-performance-13-profiles",
            "torch_profiler_with_stack": False,
            "max_iterations": 8,
        },
    )
    # Reserve permission for one-prefix dense numerical oracles. Evaluation is
    # disabled; ordinary requests still capture only their explicit width.
    cfg["generator"].setdefault("eval_sampling_params", {})["logprobs"] = 200000
    return cfg, plan, counts


def build_qualification(version, smoke, performance=False):
    cfg, plan, counts = performance_recipe(smoke) if performance else qualification_recipe(smoke)
    if performance:
        source = ArtifactStep.adopt(user_owned_name("documents/sc-performance-pivot-source"), "2026.10.16.3", SOURCE_URI)
        data = apply(
            user_owned_name("documents/sc-performance-snowball"),
            remote(prepare_performance_data, resources=ResourceConfig.with_cpu(cpu=2, ram="16g")),
            version=version,
            source_path=source,
            output_path=OUT,
            smoke=smoke,
        )
    else:
        data = cat_count_data_step(
            user_owned_name("documents/tim-cp-qualification"), version, train_ns=counts, train_rows=256, seed=17
        )
    if smoke:
        model, tokenizer, revision = (
            qwen2_5_0_5b_instruct,
            "Qwen/Qwen2.5-0.5B-Instruct",
            "7ae557604adf67be50417f59c2c2f167def9a775",
        )
    else:
        model = ArtifactStep.adopt(
            user_owned_name("models/tim-cp-grug-datakit-8c76b22a"), "2026.10.25", GRUG_URI, kind=LevanterCheckpoint
        )
        tokenizer, revision = GRUG_HF, GRUG_REVISION
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name("checkpoints/sc-performance" if performance else "checkpoints/tim-cp-qualification"),
            version=version,
            config_yaml=yaml.safe_dump(cfg),
            model=ArtifactHfModel(model, tokenizer, revision, relative_path=""),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(num_nodes=plan.num_nodes, gpus_per_node=8, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=17,
        ),
        IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=64,
            memory="512GB" if smoke else "1200GB",
            disk="1TB",
            priority="interactive" if smoke else "batch",
            max_retries=0,
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=4,
            job_timeout_seconds=10800,
            wandb_entity="marin-community",
        ),
        export_hf=False,
    )


@click.command()
@click.option("--smoke/--full", default=False)
@click.option("--performance", is_flag=True)
@rl_build_options
def main(smoke, performance):
    return build_qualification(
        resolve_version("sc-performance" if performance else "tim-cp-qualification", None), smoke, performance
    )


if __name__ == "__main__":
    main()
