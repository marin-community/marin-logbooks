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
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.models import qwen2_5_0_5b_instruct
from experiments.post_training.sc_default_study import study_recipe
from experiments.post_training.tim_cp_qualification import GRUG_HF, GRUG_REVISION, GRUG_URI

DATA_URI = "s3://marin-rno2a/marin/users/romain/documents/sc-default-study/2026.10.28.1"
SMOKE_DATA_URI = "s3://marin-rno2a/marin/users/romain/documents/sc-default-study-smoke/2026.10.28.1"


def probe_recipe(interval: int, seed: int, smoke: bool) -> tuple[dict, SkyRLRolePlan]:
    cfg, plan = study_recipe("tis", seed, smoke)
    cfg["trainer"].update(
        max_steps=12 if smoke else 30,
        project_name="asynchronous-probe-cost",
        eval_before_train=False,
        eval_interval=-1,
        ckpt_interval=-1,
        hf_save_interval=-1,
        export_hf_artifact=False,
    )
    cfg["trainer"]["algorithm"]["score_centering_probe"]["interval"] = interval
    cfg["extra_env"].pop("SC_DEFAULT_STUDY")
    cfg["extra_env"]["ASYNC_PROBE_COST"] = "1"
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
            cluster="cw-us-east-02a",
            cluster_config="lib/iris/config/cw-us-east-02a.yaml",
            cpu=64,
            memory="512GB" if smoke else "1800GB",
            disk="2TB",
            priority="interactive" if smoke else "batch",
            max_retries=0,
            target_cluster="cw-us-east-02a",
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
