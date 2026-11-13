# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import image_push

with tempfile.TemporaryDirectory(prefix="biomni-input-layer-") as work_name:
    work = Path(work_name)
    archive = work / "crane.tgz"
    digest = image_push.download(urllib.request.Request(os.environ["CRANE_URL"]), archive)
    if digest != os.environ["CRANE_SHA256"]:
        raise ValueError("crane checksum mismatch")
    crane = work / "crane"
    image_push.extract_member(archive, "crane", crane, "r|gz")
    crane.chmod(0o700)
    auth = work / "auth"
    auth.mkdir(mode=0o700)
    config = auth / "config.json"
    config.write_text(image_push.secret("REGISTRY_AUTH"))
    config.chmod(0o600)
    layer = Path("/app/input-layer.tar")
    if hashlib.sha256(layer.read_bytes()).hexdigest() != os.environ["LAYER_SHA256"]:
        raise ValueError("input layer checksum mismatch")
    env = {k: os.environ[k] for k in image_push.CRANE_ENV if k in os.environ}
    env["DOCKER_CONFIG"] = str(auth)
    subprocess.run(
        [
            str(crane),
            "append",
            "--base",
            os.environ["BASE_IMAGE"],
            "--new_layer",
            str(layer),
            "--new_tag",
            os.environ["DESTINATION"],
        ],
        env=env,
        check=True,
    )
    digest = subprocess.check_output([str(crane), "digest", os.environ["DESTINATION"]], env=env, text=True).strip()
    result = {
        "image": os.environ["DESTINATION"].rsplit(":", 1)[0] + "@" + digest,
        "base_image": os.environ["BASE_IMAGE"],
        "input_layer_sha256": os.environ["LAYER_SHA256"],
        "status": "published; runtime readback pending",
    }
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "input-image.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
