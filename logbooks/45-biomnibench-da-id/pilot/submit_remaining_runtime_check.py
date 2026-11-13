# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit three bounded sandbox checks with no model calls."""

import json
import os
import subprocess
from pathlib import Path

r = Path(__file__).resolve().parent
receipt = r / "remaining-runtime-submission.json"
assert not receipt.exists(), "Inspect existing submission before retry"
images = json.loads((r / "remaining-images.json").read_text())
assert {x["task"] for x in images} == {"da-1-3", "da-17-3", "da-20-3"}
script = (r / "validate_remaining_runtime.py").read_text()
name = os.environ["BIOMNI_RUNTIME_JOB_NAME"]
command = [
    ".venv/bin/iris",
    "--cluster=marin",
    "job",
    "run",
    "--target-cluster",
    "cw-us-east-02a",
    "--job-name",
    name,
    "--cpu",
    "1",
    "--memory",
    "2GB",
    "--disk",
    "5GB",
    "--timeout",
    "2100",
    "--max-retries",
    "0",
    "--no-sync",
    "--no-wait",
    "--env-vars",
    "BIOMNI_IMAGES_JSON",
    json.dumps(images),
    "--secret",
    "DAYTONA_EVAL_API_KEY",
    os.environ["DAYTONA_EVAL_SECRET_SOURCE"],
    "--",
    "uv",
    "run",
    "--project",
    "vendor/harbor",
    "--frozen",
    "--extra",
    "daytona",
    "python",
    "-c",
    script,
]
subprocess.run(command, check=True)
receipt.write_text(
    json.dumps(
        {"job_name": name, "timeout_seconds": 2100, "scope": "three sequential runtime/input checks; no model calls"},
        indent=2,
    )
    + "\n"
)
