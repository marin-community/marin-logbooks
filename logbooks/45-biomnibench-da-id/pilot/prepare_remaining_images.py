# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote-only streaming input packaging for the three remaining pilot tasks."""

import json
import os
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import image_push


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


urllib.request.install_opener(urllib.request.build_opener(SafeRedirect()))
root = Path(os.environ["IRIS_OUTPUT_DIR"])
plan = json.loads(Path("/app/input-plan.json").read_text())
with tempfile.TemporaryDirectory(prefix="biomni-remaining-") as dirname:
    work = Path(dirname)
    archive = work / "crane.tgz"
    assert image_push.download(urllib.request.Request(os.environ["CRANE_URL"]), archive) == os.environ["CRANE_SHA256"]
    crane = work / "crane"
    image_push.extract_member(archive, "crane", crane, "r|gz")
    crane.chmod(0o700)
    auth = work / "auth"
    auth.mkdir(mode=0o700)
    (auth / "config.json").write_text(image_push.secret("REGISTRY_AUTH"))
    (auth / "config.json").chmod(0o600)
    env = {k: os.environ[k] for k in image_push.CRANE_ENV if k in os.environ}
    env["DOCKER_CONFIG"] = str(auth)
    results = []
    for task, inputs in plan.items():
        manifest = []
        layer = work / "inputs.tar"
        with tarfile.open(layer, "w") as tar:
            # Hide any inherited control data; only declared inputs remain visible.
            whiteout = tarfile.TarInfo("app/data/.wh..wh..opq")
            whiteout.size = 0
            tar.addfile(whiteout)
            for item in inputs:
                target = work / "download"
                request = urllib.request.Request(
                    item["url"], headers={"Authorization": "Bearer " + image_push.secret("HF_TOKEN").strip()}
                )
                digest = image_push.download(request, target)
                assert target.stat().st_size == item["size"], item["path"]
                if item["sha256"]:
                    assert digest == item["sha256"], item["path"]
                tar.add(target, arcname="app/data/" + item["path"], recursive=False)
                manifest.append({**item, "sha256": digest})
                target.unlink()
        tag = os.environ["DESTINATION_PREFIX"] + ":" + task + "-20261113"
        subprocess.run(
            [str(crane), "append", "--base", os.environ["BASE_IMAGE"], "--new_layer", str(layer), "--new_tag", tag],
            env=env,
            check=True,
        )
        digest = subprocess.check_output([str(crane), "digest", tag], env=env, text=True).strip()
        result = {
            "task": task,
            "image": tag.rsplit(":", 1)[0] + "@" + digest,
            "inputs": manifest,
            "base_image": os.environ["BASE_IMAGE"],
            "status": "published; sandbox validation pending",
        }
        (root / (task + "-image.json")).write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({"task": task, "image": result["image"], "files": len(manifest), "status": result["status"]}),
            flush=True,
        )
        results.append(result)
        layer.unlink()
    (root / "remaining-images.json").write_text(json.dumps(results, indent=2) + "\n")
