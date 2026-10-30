# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
from types import SimpleNamespace

import pytest

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
