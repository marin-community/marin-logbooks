# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Supply the historical token-ID bank to the unchanged native benchmark client.

Run in the pinned vLLM environment, for example:
  python historical_bank.py bench serve --dataset-name custom --dataset-path BANK ...

Only the sample provider changes. Native HTTP requests, concurrency, clocks,
statistics and detailed exports remain in vLLM. No text round trip changes the
original prompt IDs. This is a forced-length diagnostic, not AgentX.
"""

import argparse
import hashlib
import json
from pathlib import Path

from vllm.benchmarks import serve
from vllm.benchmarks.datasets import SampleRequest
from vllm.entrypoints.cli.main import main

BANK_SHA256 = "b15aeac7c3eb714fe45fc6b8e274342ad26f00970d84f2b7985b42c422b4e905"


def historical_samples(args: argparse.Namespace, _tokenizer: object) -> list[SampleRequest]:
    assert args.dataset_name == "custom"
    content = Path(args.dataset_path).read_bytes()
    assert hashlib.sha256(content).hexdigest() == BANK_SHA256
    bank = json.loads(content)
    assert len(bank) == 1024
    return [
        SampleRequest(
            prompt=bank[index % len(bank)]["prompt_token_ids"],
            prompt_len=len(bank[index % len(bank)]["prompt_token_ids"]),
            expected_output_len=bank[index % len(bank)]["output_tokens"],
            request_id=f"historical-bank-{index}",
        )
        for index in range(args.num_prompts)
    ]


if __name__ == "__main__":
    serve.get_samples = historical_samples
    main()
