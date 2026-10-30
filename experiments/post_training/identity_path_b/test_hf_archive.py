# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
from types import SimpleNamespace

import pytest
from rigging.secrets import secret_file_contents

from experiments.post_training.identity_path_b import hf_archive
from experiments.post_training.identity_path_b.hf_archive import download_source, private_repository, verify_hub


class MemoryS3:
    def __init__(self, content):
        self.content = content

    def get_object(self, *, Bucket, Key, IfMatch):
        assert Bucket == "marin-rno2a" and Key == "checkpoint/model.safetensors" and IfMatch == '"pinned"'
        return {"Body": io.BytesIO(self.content)}


def test_source_corruption_prevents_accepting_checkpoint(tmp_path):
    original = b"bound model shard"
    item = dict(
        relative="model.safetensors",
        destination_etag='"pinned"',
        bytes=len(original),
        destination_sha256=hashlib.sha256(original).hexdigest(),
    )
    with pytest.raises(ValueError, match="hash/length"):
        download_source(MemoryS3(b"different bytes!"), "s3://marin-rno2a/checkpoint", item, tmp_path / "shard")
    download_source(MemoryS3(original), "s3://marin-rno2a/checkpoint", item, tmp_path / "verified")
    assert (tmp_path / "verified").read_bytes() == original


class MemoryHub:
    def __init__(self, files, private=True):
        self.files = files
        self.private = private

    def create_repo(self, *args, **kwargs):
        pass

    def list_repo_files(self, *args, **kwargs):
        return list(self.files)

    def model_info(self, *args, **kwargs):
        return SimpleNamespace(
            private=self.private,
            sha="pinned-revision",
            siblings=[SimpleNamespace(rfilename=k, size=len(v), lfs=None) for k, v in self.files.items()],
        )


def test_existing_public_repository_is_never_admitted():
    with pytest.raises(ValueError, match="public"):
        private_repository(MemoryHub({}, private=False), "open-athena/checkpoint")


def test_hub_hash_failure_prevents_verified_receipt(tmp_path):
    files = {
        "README.md": b"card",
        "LICENSE": b"license",
        "weights": b"verified model",
        ".gitattributes": b"hub generated",
    }
    expected = {
        name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        for name, raw in files.items()
        if name != ".gitattributes"
    }
    hub = MemoryHub(files)

    def download(repo, name, *, revision, token):
        assert revision == "pinned-revision" and token == "fixture-secret"
        path = tmp_path / name
        path.write_bytes(hub.files[name])
        return str(path)

    result = verify_hub(hub, "open-athena/checkpoint", expected, "fixture-secret", download)
    assert result["hub_verified"] and not result["delete_eligible"] and not result["eval_chains_terminal_verified"]
    assert set(result["files"]) == set(expected)
    hub.files["weights"] = b"corrupt model!"
    with pytest.raises(ValueError, match="SHA256"):
        verify_hub(hub, "open-athena/checkpoint", expected, "fixture-secret", download)


def test_isolated_secret_files_reach_client_boundaries_without_plain_env(tmp_path, monkeypatch):
    credentials = {"HF_TOKEN": "fixture-hub-token", "CW_KEY_ID": "fixture-cw-id", "CW_KEY_SECRET": "fixture-cw-secret"}
    native = hf_archive.submission(credentials)
    assert native["environment"].secret_behavior.value == "isolate"
    secret_dir = tmp_path / "mounted-secrets"
    secret_dir.mkdir()
    for name, value in secret_file_contents(native["environment"].secrets).items():
        (secret_dir / name).write_text(value)
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(secret_dir))
    for name in credentials:
        monkeypatch.delenv(name, raising=False)
    received = {}

    def s3_client(service, **kwargs):
        received["service"] = service
        received["access_key"] = kwargs["aws_access_key_id"]
        received["secret_key"] = kwargs["aws_secret_access_key"]
        return object()

    class BeforeRemoteOperations(Exception):
        pass

    def hub_client(*, token):
        received["token"] = token
        raise BeforeRemoteOperations

    monkeypatch.setattr(hf_archive.boto3, "client", s3_client)
    monkeypatch.setattr(hf_archive, "HfApi", hub_client)
    with pytest.raises(BeforeRemoteOperations):
        hf_archive.run()
    assert received == {
        "service": "s3",
        "access_key": credentials["CW_KEY_ID"],
        "secret_key": credentials["CW_KEY_SECRET"],
        "token": credentials["HF_TOKEN"],
    }
