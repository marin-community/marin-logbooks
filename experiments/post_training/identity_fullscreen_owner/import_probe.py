# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU import/input/f5-signature smoke without model, child-job, or storage calls."""

import argparse
import hashlib
import json
from pathlib import Path

from iris.cluster.client.job_info import get_job_info

from experiments.post_training.identity_fullscreen_owner import (
    math_signature,
    owner,
    persist_task,
    persistence,
    run_unseen,
)
from experiments.post_training.identity_fullscreen_owner.frozen import job_policy


def run(math_archive: str, math_archive_sha256: str):
    info = get_job_info()
    rows = run_unseen.checked_unseen_rows()
    signature = math_signature.pinned_signature_smoke(math_archive, math_archive_sha256)
    modules = [owner, persist_task, persistence, job_policy, run_unseen, math_signature]
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
                "status": "import_and_signature_pass",
                "unseen_rows": len(rows),
                "signature_smoke": signature,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-archive", required=True)
    parser.add_argument("--math-archive-sha256", required=True)
    args = parser.parse_args()
    run(args.math_archive, args.math_archive_sha256)
