# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admit exact historical GSM inputs, then run the unchanged native client.

No inference is performed here. The evaluator subprocess inherits offline HF
settings after admission, so it uses the already-verified dataset cache. Prompt,
extraction, grading, sampling and HTTP retry policy stay in the pinned native
Evalchemy sources.
"""

import hashlib
import json
import os
import runpy
from pathlib import Path

from datasets import load_dataset
from rigging.filesystem.storage_path import StoragePath

GSM_TEST_SHA256 = "32c548f08195e19e33408b844dd7be6aa4bcae457d957bc05909ff4bd4a00595"


def checked_gsm_inputs() -> dict:
    rows = load_dataset("openai/gsm8k", "main", split="test")
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        assert isinstance(row, dict)
        digest.update(
            (
                json.dumps(
                    {"question": row["question"], "answer": row["answer"]},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        )
        count += 1
    if count != 1319 or digest.hexdigest() != GSM_TEST_SHA256:
        raise ValueError("Native GSM test inputs differ from the exact1319 reviewed question/answer records")
    return {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "count": count,
        "canonical_question_answer_jsonl_sha256": digest.hexdigest(),
        "harness_dataset_access_after_admission": "offline existing cache only",
    }


def main() -> None:
    proof = checked_gsm_inputs()
    config = json.loads(os.environ["EVALCHEMY_CLIENT_CONFIG"])
    StoragePath(f"{config['out_path']}/checked-gsm-inputs.json").write_bytes(
        (json.dumps(proof, indent=2) + "\n").encode()
    )
    # The native client spawns the harness process, which reads both settings
    # at import time. No second remote resolution can change admitted inputs.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    runpy.run_path(
        str(Path(os.environ["IRIS_WORKDIR"]) / "lib/marin/src/marin/evaluation/evalchemy/client.py"), run_name="__main__"
    )


if __name__ == "__main__":
    main()
