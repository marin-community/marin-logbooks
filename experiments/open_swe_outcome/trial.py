# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune Qwen3-8B on one arm of the Open-SWE outcome-filter ablation.

Every arm trains the same base checkpoint with the same optimizer, context length and batch
size for one pass over its data. Arms differ only in their trajectories; seed repeats differ
only in the trainer seed. ``--smoke`` trains 20 steps on one data shard to check the path and
measure throughput.

``--format native`` trains the arms in their source scaffolds' native tool-call format.
``--format terminus`` trains the arms rewritten into Harbor's Terminus-2 protocol, the agent
SWE-bench Verified is evaluated with (``terminus.py`` and ``terminus_arms.py``).

Tracked in marin-community/marin-logbooks#39.
"""

from enum import StrEnum

import click
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_owned_name

from experiments.open_swe_outcome.arms import (
    ARM_VERSION,
    ARMS,
    MAX_TOKENS,
    STEP_PREFIX,
    arm_step,
    index_step,
    membership_step,
    raw_partitions,
)
from experiments.open_swe_outcome.terminus_arms import TERMINUS_ARMS, terminus_arm_steps
from experiments.open_swe_outcome.terminus_arms import VERSION as TERMINUS_ARM_VERSION
from experiments.open_swe_outcome.terminus_chat_template import TERMINUS_CHAT_TEMPLATE
from experiments.sft.launcher import ArtifactDatasetSpec, HFModel, SFTSpec, resources_from_accelerator, sft_step
from experiments.sft.opencode_chat_template import OPENCODE_TOOLS_CHAT_TEMPLATE

BASE_MODEL = "Qwen/Qwen3-8B"
# <|endoftext|> and <|im_end|> in the Qwen3 tokenizer.
QWEN3_EOS_TOKEN_IDS = (151643, 151645)
VERSION = "2026.11.13"
WANDB_PROJECT = "marin-open-swe-outcome"
CONTEXT_LENGTH = MAX_TOKENS
BATCH_SIZE = 8
SMOKE_STEPS = 20
SMOKE_SHARD = "train-00000-of-00212.jsonl.gz"
ARM_GLOB = "*.jsonl.gz"


class TrainingFormat(StrEnum):
    NATIVE = "native"
    TERMINUS = "terminus"


OPTIMIZER = AdamConfig(
    learning_rate=1e-5,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-8,
    max_grad_norm=1.0,
    weight_decay=0.0,
    lr_schedule="cosine",
    warmup=0.1,
    min_lr_ratio=0.0,
)


def arm_data(arm: str, training_format: TrainingFormat) -> tuple[ArtifactStep, str]:
    """The arm's materialized rows and the chat template that renders them."""
    if training_format is TrainingFormat.TERMINUS:
        if arm not in TERMINUS_ARMS:
            raise ValueError(f"{arm} is not a Terminus-2 arm; choose from {TERMINUS_ARMS}")
        # Materialized by experiments.open_swe_outcome.build_terminus_arms.
        step = terminus_arm_steps()[arm]
        return ArtifactStep.adopt(step.name, TERMINUS_ARM_VERSION, step.output_path), TERMINUS_CHAT_TEMPLATE
    if arm not in ARMS:
        raise ValueError(f"{arm} is not a native-format arm; choose from {ARMS}")
    # Materialized by experiments.open_swe_outcome.build_arms.
    partitions = raw_partitions()
    step = arm_step(arm, membership_step(index_step(partitions)), partitions)
    return ArtifactStep.adopt(f"{STEP_PREFIX}/arms/{arm}", ARM_VERSION, step.output_path), OPENCODE_TOOLS_CHAT_TEMPLATE


def arm_spec(
    arm: str,
    seed: int,
    *,
    smoke: bool,
    training_format: TrainingFormat,
    context_length: int = CONTEXT_LENGTH,
) -> SFTSpec:
    """The SFT run for one arm and seed."""
    data, chat_template = arm_data(arm, training_format)
    suffix = f"-ctx{context_length}" + ("-smoke" if smoke else "")
    return SFTSpec(
        name=user_owned_name(f"checkpoints/open-swe-outcome/qwen3-8b-{arm}-seed{seed}{suffix}"),
        version=VERSION,
        # Qwen3-8B declares a 40,960-token window; its rotary positions extend past it.
        model=HFModel(BASE_MODEL, eos_token_ids=QWEN3_EOS_TOKEN_IDS, max_seq_len=context_length),
        chat_template=chat_template,
        datasets=[
            ArtifactDatasetSpec(slug=arm, artifact=data, train_glob=SMOKE_SHARD if smoke else ARM_GLOB, weight=1.0)
        ],
        optimizer=OPTIMIZER,
        seq_len=context_length,
        batch_size=BATCH_SIZE,
        num_train_steps=SMOKE_STEPS if smoke else None,
        num_train_epochs=None if smoke else 1,
        hf_save_dtype="bfloat16",
        wandb_project=WANDB_PROJECT,
        seed=seed,
    )


@click.command(help=__doc__)
@click.option("--arm", type=click.Choice((*ARMS, *TERMINUS_ARMS)), required=True)
@click.option(
    "--format",
    "training_format",
    type=click.Choice([training_format.value for training_format in TrainingFormat]),
    required=True,
)
@click.option("--seed", type=int, required=True)
@click.option("--accelerator", required=True, help="Accelerator such as '8xH100'.")
@click.option("--smoke", is_flag=True, help=f"Train {SMOKE_STEPS} steps on one shard.")
@click.option("--context-length", type=int, default=CONTEXT_LENGTH, show_default=True)
def main(arm: str, training_format: str, seed: int, accelerator: str, smoke: bool, context_length: int) -> None:
    spec = arm_spec(
        arm, seed, smoke=smoke, training_format=TrainingFormat(training_format), context_length=context_length
    )
    StepRunner().run([lower(sft_step(spec, resources_from_accelerator(accelerator)))])


if __name__ == "__main__":
    main()
