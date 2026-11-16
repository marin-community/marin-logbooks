# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the missing public-50 input images, reusing four validated pilot images."""

import argparse
import json
from pathlib import Path

from huggingface_hub import get_token
from iris.cli.connect import connect_controller
from iris.client import IrisClient
from iris.cluster.types import Entrypoint
from taskforge.sandbox import image_push
from taskforge.sandbox.image_builder import (
    CRANE_SHA256,
    CRANE_URL,
    PUSH_IMAGE,
    RegistryCredential,
    registry_credential_json,
    submit_job,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    args = parser.parse_args()
    receipt = args.plan_dir / "image-submission.json"
    if receipt.exists():
        raise ValueError("Submission already recorded; inspect before retry")
    groups = json.loads((args.plan_dir / "input-groups.json").read_text())
    reused = json.loads((args.pilot_dir / "remaining-images.json").read_text())
    control = json.loads((args.pilot_dir / "control-image.json").read_text())
    reused.append({"task": "da-4-6", "image": control["image"]})
    reuse_by_task = {x["task"]: x for x in reused}
    plan = {}
    for group in groups:
        match = next((reuse_by_task[t] for t in group["tasks"] if t in reuse_by_task), None)
        if match:
            group["image"] = match["image"]
        else:
            plan[group["tasks"][0]] = group["inputs"]
    (args.plan_dir / "planned-images.json").write_text(json.dumps(groups, indent=2) + "\n")
    credential = RegistryCredential.from_json(registry_credential_json())
    token = get_token()
    if not token:
        raise ValueError("HF_TOKEN is required for the authenticated input downloads")
    source = Path(__file__).resolve().parent
    base = control["image"]
    with connect_controller(cluster_name="marin") as endpoint:
        client = IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials)
        job = submit_job(
            client,
            name=args.job_name,
            entrypoint=Entrypoint(
                command=["python3", "/app/prepare_images.py"],
                workdir_files={
                    "prepare_images.py": (source / "prepare_images.py").read_bytes(),
                    "image_push.py": Path(image_push.__file__).read_bytes(),
                    "input-plan.json": json.dumps(plan).encode(),
                },
            ),
            image=PUSH_IMAGE,
            env={
                "CRANE_URL": CRANE_URL,
                "CRANE_SHA256": CRANE_SHA256,
                "BASE_IMAGE": base,
                "DESTINATION_PREFIX": base.split("@")[0],
            },
            secrets={"REGISTRY_AUTH": credential.docker_config(), "HF_TOKEN": token},
            target_cluster="cw-us-east-02a",
        )
        result = {
            "job_id": str(job.job_id),
            "new_images": len(plan),
            "reused_images": len(groups) - len(plan),
            "tasks": 50,
        }
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
