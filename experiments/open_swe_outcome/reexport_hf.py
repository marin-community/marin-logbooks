# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-export one outcome-ablation Levanter checkpoint to HF format on CPU.

The HF exports written at the end of training on 8 GPUs hold only one device's slice of each
weight (hidden size 512 instead of 4096), so vLLM cannot load them. A single-device CPU export
has no sharding to get wrong. Non-weight files (tokenizer, the training chat template and the
generation config) are correct in the original exports and are copied from there.

Run one Iris CPU job per checkpoint; loading the checkpoint needs well over 64 GB of RAM.
"""

import dataclasses
import logging

import click
import fsspec
from levanter.main import export_lm_to_hf
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.trainer import TrainerConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3-8B"
CONTEXT_LENGTH = 65_536
# The checkpoint's embedding uses Qwen3-8B's padded vocabulary, larger than the tokenizer's 151,669.
PADDED_VOCAB_SIZE = 151_936
RUN_ROOT = "s3://marin-us-east-08a/marin/users/dml/checkpoints/open-swe-outcome"
RUN_DATE = "2026.11.13"
FINAL_STEPS = {"solved-10000": 786, "random-10000": 787}
COPIED_FILES = ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json")


def qwen3_8b_config() -> Qwen3Config:
    converter = Qwen3Config().hf_checkpoint_converter().replaced(reference_checkpoint=BASE_MODEL)
    model = converter.config_from_hf_config(converter.default_hf_config)
    return dataclasses.replace(model, max_seq_len=CONTEXT_LENGTH)


@click.command()
@click.option("--arm", type=click.Choice(sorted(FINAL_STEPS)), required=True)
@click.option("--seed", type=int, required=True)
def main(arm: str, seed: int) -> None:
    configure_logging()
    configure_coreweave_s3()
    step = FINAL_STEPS[arm]
    run = f"{RUN_ROOT}/qwen3-8b-{arm}-seed{seed}-ctx{CONTEXT_LENGTH}/{RUN_DATE}"
    output = f"{run}/hf-cpu/step-{step}"
    export_lm_to_hf.main(
        ConvertLmConfig(
            trainer=TrainerConfig(),
            checkpoint_path=f"{run}/checkpoints/step-{step}",
            output_dir=output,
            model=qwen3_8b_config(),
            override_vocab_size=PADDED_VOCAB_SIZE,
            save_tokenizer=False,
            use_cpu=True,
        )
    )
    fs, _ = fsspec.core.url_to_fs(output)
    for name in COPIED_FILES:
        fs.copy(f"{run}/hf/step-{step}/{name}", f"{output}/{name}")
    logger.info("re-exported %s seed %d to %s", arm, seed, output)


if __name__ == "__main__":
    main()
