# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import-only live probe: no inference, child submission, or storage writes."""

import hashlib
import json
from pathlib import Path

from iris.cluster.client.job_info import get_job_info

from experiments.post_training.identity_fullscreen_owner import owner, persist_task, persistence
from experiments.post_training.identity_fullscreen_owner.frozen import job_policy


def run():
    info = get_job_info()
    modules = [owner, persist_task, persistence, job_policy]
    files = {
        module.__name__: {
            "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "path": module.__file__,
        }
        for module in modules
    }
    print(
        json.dumps(
            {
                "status": "import_pass",
                "bundle_id": info.bundle_id if info else None,
                "modules": files,
                "model_calls": 0,
                "child_submissions": 0,
                "storage_writes": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
