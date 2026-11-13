# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import tarfile
from pathlib import Path

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

r = Path(__file__).resolve().parent
out = io.BytesIO()
data = (r / "build-contexts/da-4-6/data/mmc5.csv").read_bytes()
with tarfile.open(fileobj=out, mode="w") as tar:
    item = tarfile.TarInfo("app/data/mmc5.csv")
    item.size = len(data)
    item.mode = 0o644
    tar.addfile(item, io.BytesIO(data))
layer = out.getvalue()
base = json.loads((r / "control-image.json").read_text())["image"]
credential = RegistryCredential.from_json(registry_credential_json())
with connect_controller(cluster_name="marin") as endpoint:
    client = IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials)
    job = submit_job(
        client,
        name="biomnibench-control-input-layer-20261113",
        entrypoint=Entrypoint(
            command=["python3", "/app/publish_input_layer.py"],
            workdir_files={
                "publish_input_layer.py": (r / "publish_input_layer.py").read_bytes(),
                "image_push.py": Path(image_push.__file__).read_bytes(),
                "input-layer.tar": layer,
            },
        ),
        image=PUSH_IMAGE,
        env={
            "CRANE_URL": CRANE_URL,
            "CRANE_SHA256": CRANE_SHA256,
            "BASE_IMAGE": base,
            "DESTINATION": base.split("@")[0] + ":control-inputs-20261113",
            "LAYER_SHA256": hashlib.sha256(layer).hexdigest(),
        },
        secrets={"REGISTRY_AUTH": credential.docker_config()},
        target_cluster="cw-us-east-02a",
    )
    result = {
        "job_id": str(job.job_id),
        "base_image": base,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "layer_sha256": hashlib.sha256(layer).hexdigest(),
    }
    (r / "input-layer-submission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
