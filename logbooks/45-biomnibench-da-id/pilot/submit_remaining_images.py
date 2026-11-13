# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the bounded CPU preparation job; never persist secret values."""

import json
import os
from pathlib import Path
from urllib.parse import quote

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

root = Path(__file__).resolve().parent
plan = {task: [] for task in ("da-1-3", "da-17-3", "da-20-3")}
for item in json.loads((root / "source-data-manifest.json").read_text()):
    task = item["path"].split("/")[0]
    if task not in plan or item["path"].endswith("GSE236581_counts.mtx"):
        continue
    record = {
        "path": item["path"].split("/environment/data/", 1)[1],
        "size": item["size"],
        "sha256": item["sha256"],
        "url": (
            "https://huggingface.co/datasets/phylobio/BiomniBench-DA/resolve/e1c8ca5e11a620087bc48d97888eb69176a1f235/"
            + quote(item["path"])
        ),
    }
    if record["path"].endswith(".h5ad"):
        record.update(
            size=2642705030,
            sha256="33179d1cf993091dc0bd41372629534f6b7da1ce1114d74c36110ffbee375d53",
            url=(
                "https://huggingface.co/datasets/open-athena/biomnibench-da-lupus-raw-c"
                "ounts/resolve/60044fab4ca09e55041729670deb55b4e6205c83/lupus-raw-count"
                "s.h5ad"
            ),
        )
    plan[task].append(record)
assert [len(plan[t]) for t in plan] == [4, 2, 72]
(root / "remaining-input-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
credential = RegistryCredential.from_json(registry_credential_json())
token = get_token()
assert token
base = json.loads((root / "control-image-before-input-repair.json").read_text())["image"]
with connect_controller(cluster_name="marin") as endpoint:
    client = IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials)
    job = submit_job(
        client,
        name=os.environ["BIOMNI_IMAGE_JOB_NAME"],
        entrypoint=Entrypoint(
            command=["python3", "/app/prepare_remaining_images.py"],
            workdir_files={
                "prepare_remaining_images.py": (root / "prepare_remaining_images.py").read_bytes(),
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
        "tasks": list(plan),
        "resources": "2 CPU, 4 GiB RAM, 20 GiB disk",
        "timeout_seconds": 3600,
        "max_retries": 0,
    }
    (root / "remaining-image-submission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
