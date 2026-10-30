# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive the two protected Path B checkpoints to private Hub repositories."""

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import boto3
import yaml
from botocore.config import Config
from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig
from huggingface_hub import HfApi, hf_hub_download
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from rigging.secrets import SecretBehavior, require_secret
from rigging.timing import Duration

ARTIFACTS = Path(__file__).with_name("hf_archive_artifacts")
CHUNK_BYTES = 8 * 1024 * 1024
WARNING = (
    "Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. "
    "Use at your own risk in production settings."
)
RESPONSIBLE_USE = (
    "We release our tools to the public with the intent that they are used safely, ethically, and responsibly. "
    "They are meant to contribute to the public's understanding of science, and aid in its advancement. "
    "By using this Marin model, you agree to comply with all applicable law."
)
RESOURCES = ResourceConfig.with_cpu(cpu=8, ram="32g", disk="300g", target_cluster="cw-rno2a")
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_artifact(name: str, digest: str) -> bytes:
    raw = (ARTIFACTS / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Committed archive artifact changed: {name}")
    return raw


def s3_location(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != "marin-rno2a":
        raise ValueError("Archive may read only the authorized RNO bucket")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def source_inventory(client, source: str) -> set[str]:
    bucket, prefix = s3_location(source)
    names = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        names.update(item["Key"][len(prefix) + 1 :] for item in page.get("Contents", []))
    return names


def download_source(client, source: str, item: dict, output: Path) -> None:
    relative = PurePosixPath(item["relative"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Source manifest contains an unsafe path")
    bucket, prefix = s3_location(source)
    response = client.get_object(Bucket=bucket, Key=prefix + "/" + str(relative), IfMatch=item["destination_etag"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with response["Body"] as stream, output.open("wb") as destination:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            destination.write(chunk)
    if output.stat().st_size != item["bytes"] or sha256_file(output) != item["destination_sha256"]:
        raise ValueError(f"Permanent source object failed hash/length: {relative}")


def stage_source(client, target: dict, folder: Path, license_sha256: str) -> dict:
    manifest_raw = checked_artifact(target["manifest_file"], target["manifest_sha256"])
    keep = checked_artifact(target["keep_file"], target["keep_sha256"])
    card = checked_artifact(target["card_file"], target["card_sha256"])
    license_raw = checked_artifact("LICENSE.OpenMDW-1.1", license_sha256)
    manifest = json.loads(manifest_raw)
    objects = manifest["objects"]
    expected_names = {item["relative"] for item in objects}
    if len(objects) != target["source_objects"] or len(expected_names) != len(objects):
        raise ValueError("Permanent source manifest object set changed")
    if manifest["destination"] != target["source"] or source_inventory(client, target["source"]) != (
        expected_names | {"PRESERVATION-SHA256.json", "KEEP.md"}
    ):
        raise ValueError("Permanent source prefix differs from the bound complete inventory")
    metadata = {}
    for item in objects:
        name = item["relative"]
        mapped = {
            "README.md": "source-provenance/original-README.md",
            ".gitattributes": "source-provenance/original.gitattributes",
        }.get(name, name)
        download_source(client, target["source"], item, folder / mapped)
        metadata[mapped] = dict(bytes=item["bytes"], sha256=item["destination_sha256"], source_relative=name)
    # Verify the two completion documents against the actual permanent prefix too.
    bucket, prefix = s3_location(target["source"])
    for name, raw in [("PRESERVATION-SHA256.json", manifest_raw), ("KEEP.md", keep)]:
        with client.get_object(Bucket=bucket, Key=prefix + "/" + name)["Body"] as body:
            if body.read() != raw:
                raise ValueError(f"Permanent completion document changed: {name}")
        (folder / name).write_bytes(raw)
        metadata[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), source_relative=name)
    audit = json.loads((folder / "path-b-storage-audit.json").read_bytes())
    index = json.loads((folder / "model.safetensors.index.json").read_bytes())
    if (
        len(audit["tensors"]) != 502
        or audit["frozen_native_tensors"] != 104
        or audit["trained_master_cast_tensors"] != 398
        or sha256_file(folder / "path-b-storage-audit.json") != target["audit_sha256"]
        or sha256_file(folder / "path-b-checkpoint-event.json") != target["event_sha256"]
        or len(set(index["weight_map"].values())) != target["indexed_shards"]
    ):
        raise ValueError("Source audit/event/index does not match the authorized checkpoint")
    text = card.decode()
    if text.count(WARNING) != 1 or text.count(RESPONSIBLE_USE) != 1:
        raise ValueError("Required model-card paragraphs must each occur exactly once")
    if yaml.safe_load(text.split("---", 2)[1])["license"] != "openmdw-1.1":
        raise ValueError("Model-card license metadata differs")
    for name, raw in [("README.md", card), ("LICENSE", license_raw)]:
        (folder / name).write_bytes(raw)
        metadata[name] = dict(
            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), source_relative=name if name == "LICENSE" else None
        )
    upload_manifest = dict(
        source=target["source"],
        repo_id=target["repo_id"],
        source_manifest_sha256=target["manifest_sha256"],
        files=metadata,
        delete_eligible=False,
    )
    raw = (json.dumps(upload_manifest, indent=2) + "\n").encode()
    (folder / "UPLOAD-MANIFEST.json").write_bytes(raw)
    metadata["UPLOAD-MANIFEST.json"] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), source_relative=None)
    return metadata


def private_repository(api, repo: str) -> None:
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo).private:
        raise ValueError("Refuse to upload to an existing public repository")
    if set(api.list_repo_files(repo, repo_type="model")) - {".gitattributes"}:
        raise FileExistsError("Refuse to overwrite an already populated Hub repository")


def verify_hub(api, repo: str, expected: dict, token: str, download=hf_hub_download) -> dict:
    info = api.model_info(repo, files_metadata=True)
    if not info.private:
        raise ValueError("Hub repository visibility changed during upload")
    revision = info.sha
    siblings = {item.rfilename: item for item in info.siblings}
    if set(siblings) - {".gitattributes"} != set(expected):
        raise ValueError("Hub final file inventory differs from the complete upload manifest")
    verified = {}
    for name, item in expected.items():
        remote = siblings[name]
        if remote.size != item["bytes"]:
            raise ValueError(f"Hub file length differs: {name}")
        if remote.lfs is not None:
            digest = remote.lfs.sha256
        else:
            digest = sha256_file(Path(download(repo, name, revision=revision, token=token)))
        if digest != item["sha256"]:
            raise ValueError(f"Hub SHA256 differs: {name}")
        verified[name] = dict(item, hub_sha256=digest)
    # Root card and license are downloaded even if the Hub stores them in LFS.
    for name in ["README.md", "LICENSE"]:
        if sha256_file(Path(download(repo, name, revision=revision, token=token))) != expected[name]["sha256"]:
            raise ValueError(f"Published root document differs: {name}")
    attributes = Path(download(repo, ".gitattributes", revision=revision, token=token))
    return dict(
        repo_id=repo,
        revision=revision,
        private=True,
        files=verified,
        hub_managed_gitattributes_sha256=sha256_file(attributes),
        hub_verified=True,
        eval_chains_terminal_verified=False,
        delete_eligible=False,
    )


def write_report(client, root: str, name: str, payload: dict) -> None:
    bucket, prefix = s3_location(root)
    client.put_object(
        Bucket=bucket,
        Key=prefix + "/" + name,
        Body=(json.dumps(payload, indent=2) + "\n").encode(),
        ContentType="application/json",
    )


def run() -> None:
    config = json.loads((ARTIFACTS / "targets.json").read_bytes())
    token = require_secret("HF_TOKEN")
    client = boto3.client(
        "s3",
        endpoint_url="https://cwobject.com",
        region_name="auto",
        aws_access_key_id=require_secret("CW_KEY_ID"),
        aws_secret_access_key=require_secret("CW_KEY_SECRET"),
        config=Config(s3={"addressing_style": "virtual"}),
    )
    api = HfApi(token=token)
    completed = []
    scratch = Path(os.environ["IRIS_WORKDIR"]) / "_private_hf_archive"
    for target in config["targets"]:
        folder = scratch / target["label"]
        folder.mkdir(parents=True, exist_ok=False)
        try:
            private_repository(api, target["repo_id"])
            expected = stage_source(client, target, folder, config["license_sha256"])
            api.upload_large_folder(
                target["repo_id"],
                folder,
                repo_type="model",
                private=True,
                num_workers=4,
                allow_patterns=list(expected),
                print_report=False,
            )
            result = verify_hub(api, target["repo_id"], expected, token)
            result.update(
                label=target["label"],
                source=target["source"],
                source_manifest_sha256=target["manifest_sha256"],
                source_audit_sha256=target["audit_sha256"],
                source_event_sha256=target["event_sha256"],
            )
            write_report(client, config["report_root"], target["label"] + "-verified.json", result)
            completed.append(result)
        except Exception as error:
            write_report(
                client,
                config["report_root"],
                target["label"] + "-failed.json",
                dict(
                    label=target["label"],
                    error_type=type(error).__name__,
                    completed_labels=[x["label"] for x in completed],
                    delete_eligible=False,
                ),
            )
            raise
        finally:
            shutil.rmtree(folder)
    write_report(
        client,
        config["report_root"],
        "all-models-verified.json",
        dict(models=completed, hub_verified=True, delete_eligible=False, eval_chains_terminal_verified=False),
    )


def submission(credentials: dict[str, str]) -> dict:
    return dict(
        name="pathb-private-HF-archive-r2-20261030",
        entrypoint=Entrypoint.from_callable(run),
        resources=convert_resources(RESOURCES),
        constraints=convert_constraints(RESOURCES),
        environment=EnvironmentSpec(
            extras=("cpu",),
            pip_packages=("boto3==1.41.5", "botocore==1.41.5", "s3transfer==0.15.0", "huggingface_hub==1.21.0"),
            secrets={key: credentials[key] for key in ["HF_TOKEN", "CW_KEY_ID", "CW_KEY_SECRET"]},
            secret_behavior=SecretBehavior.ISOLATE,
        ),
        replicas=1,
        timeout=Duration.from_hours(12),
        max_retries_failure=0,
        max_retries_preemption=0,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        print((ARTIFACTS / "targets.json").read_text())
        return
    run()


if __name__ == "__main__":
    main()
