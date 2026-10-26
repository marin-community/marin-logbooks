"""One-node packed CP2 Qwen startup for the SC extension; research only."""

from dataclasses import replace

import click
import yaml
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
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)

from experiments.models import qwen2_5_0_5b_instruct
from experiments.post_training.cat_count_canary.data import TRAIN_FILENAME, VALIDATION_FILENAME, cat_count_data_step
from experiments.post_training.cat_count_canary.launcher import role_plan, training_config


def startup_recipe():
    plan = replace(
        role_plan(batch_size=4, group_size=4, micro_train_batch_size=1),
        colocate_all=True,
        policy_num_gpus_per_node=8,
        num_inference_engines=8,
    )
    config = training_config(
        preset="dry",
        lane="sync",
        batch_size=4,
        group_size=4,
        micro_train_batch_size=1,
        train_ns=(4, 6, 8, 12),
        checkpoint=True,
        export=True,
        settings=("trainer.max_steps=2", "trainer.logger=console", "trainer.update_epochs_per_batch=1"),
    )
    _materialize_role_plan_config(config, plan)
    config["trainer"]["use_sample_packing"] = True
    config["trainer"]["resume_mode"] = "none"
    config["trainer"]["algorithm"].update(
        score_centering_enabled=True, score_centering_probe={"interval": 0}, off_policy_correction="tis",
        advantage_estimator="rloo_n", group_advantage_min_size=4,
    )
    for role in ("policy", "ref"):
        config["trainer"][role]["megatron_config"]["context_parallel_size"] = 2
    config["generator"]["gpu_memory_utilization"] = 0.35
    config["trajectory_runner"] = {
        "max_concurrent_tasks": 8, "rollout_workers": {"num_workers": 2, "cpus_per_worker": 2}
    }
    config["trainer"]["callbacks"] = [
        c for c in config["trainer"]["callbacks"] if c["type"] != "database_registration"
    ]
    for callback in config["trainer"]["callbacks"]:
        if callback["type"] == "evaluation":
            callback["additional_evaluations"] = {}
            callback["metric_groups"] = {}
    return config, plan


def build_startup(version: str) -> ArtifactStep:
    data = cat_count_data_step(
        user_owned_name("documents/tim-cp-startup"), version, train_ns=(4, 6, 8, 12), train_rows=64, seed=17
    )
    config, plan = startup_recipe()
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name("checkpoints/tim-cp-startup"),
            version=version,
            config_yaml=yaml.safe_dump(config),
            model=ArtifactHfModel(
                step=qwen2_5_0_5b_instruct,
                relative_path="",
                tokenizer_uri="Qwen/Qwen2.5-0.5B-Instruct",
                tokenizer_revision="7ae557604adf67be50417f59c2c2f167def9a775",
            ),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=17,
        ),
        IrisSkyRLExecution(
            cluster="cw-rno2a", cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=64, memory="256GB", disk="400GB", priority="interactive", max_retries=0,
            target_cluster="cw-rno2a", parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=2, job_timeout_seconds=3600,
        ),
        export_hf=True,
    )


@click.command()
@rl_build_options
def main() -> ArtifactStep:
    return build_startup(resolve_version("tim-cp-startup", None))


if __name__ == "__main__":
    main()
