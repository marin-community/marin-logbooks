# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-local checks for staging isolation and dependency identity."""

import base64
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import PathDistribution
from pathlib import Path
from types import SimpleNamespace

import pytest
from iris.cluster.backends.k8s import bundle_fetch
from iris.cluster.backends.k8s.tasks import _build_init_container_spec
from iris.cluster.controller.jobs import _store_launch_payload
from iris.rpc import controller_pb2, job_pb2


def _worker_module():
    spec = importlib.util.spec_from_file_location("hero_validation_worker", Path(__file__).with_name("worker.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("directory_symlink", [True, False])
def test_overlay_owns_files_and_preserves_shared_cache(tmp_path, monkeypatch, directory_symlink):
    worker = _worker_module()
    environment = tmp_path / "environment"
    site = environment / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    shared = tmp_path / "shared-cache/vllm"
    shared.mkdir(parents=True)
    (shared / "_C.abi3.so").write_bytes(b"unchanged compiled extension")
    (shared / "config.py").write_bytes(b"original Python")
    entry = site / "vllm"
    if directory_symlink:
        entry.symlink_to(shared, target_is_directory=True)
    else:
        entry.mkdir()
        for path in shared.iterdir():
            (entry / path.name).symlink_to(path)
    archive = tmp_path / "vllm.whl"
    record = _wheel_archive(archive, {"vllm/" + path.name: path.read_bytes() for path in shared.iterdir()})
    wheel = {"version": "1.0", "url": archive.as_uri(), "hash": "sha256:" + worker.digest(archive)}
    dist_info = site / "vllm-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: vllm\nVersion: 1.0\n")
    (dist_info / "RECORD").write_text(record)
    (dist_info / "direct_url.json").write_text(json.dumps({"url": wheel["url"], "archive_info": {}}))
    monkeypatch.setattr(worker.metadata, "distribution", lambda _name: PathDistribution(dist_info))
    monkeypatch.setattr(worker.sys, "prefix", str(environment))
    payload = tmp_path / "payload"
    overlay = payload / "hero-shortconv-overlay-config.py"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"reviewed Python replacement")
    manifest = {
        "wheel": wheel,
        "overlay": [
            {
                "path": "vllm/config.py",
                "payload_path": overlay.name,
                "base_sha256": hashlib.sha256(b"original Python").hexdigest(),
                "patched_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
            }
        ],
    }
    work = tmp_path / "work"
    work.mkdir()

    receipt = worker.stage_owned_package(payload, work, manifest, tmp_path / "attestation.json")

    assert (entry / "config.py").read_bytes() == overlay.read_bytes()
    assert (shared / "config.py").read_bytes() == b"original Python"
    assert (shared / "_C.abi3.so").read_bytes() == b"unchanged compiled extension"
    assert not entry.is_symlink()
    assert not any(path.is_symlink() for path in entry.iterdir())
    assert receipt["extensions_before"] == receipt["extensions_after"]


def _wheel_archive(path, files):
    records = []
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            records.append([name, "sha256=" + encoded, str(len(content))])
        records.append(["vllm-1.0.dist-info/RECORD", "", ""])
        output = io.StringIO()
        csv.writer(output).writerows(records)
        archive.writestr("vllm-1.0.dist-info/RECORD", output.getvalue())
    return output.getvalue()


@pytest.mark.parametrize("archive_info", [{}, {"hash": "sha256={digest}"}, {"hashes": {"sha256": "{digest}"}}])
def test_supported_receipts_require_archive_and_installed_content(tmp_path, archive_info):
    worker = _worker_module()
    site = tmp_path / "site"
    entry = site / "vllm"
    entry.mkdir(parents=True)
    files = {"vllm/__init__.py": b"", "vllm/_C.abi3.so": b"native payload"}
    for name, content in files.items():
        (site / name).write_bytes(content)
    archive = tmp_path / "vllm.whl"
    record = _wheel_archive(archive, files)
    info = site / "vllm-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: vllm\nVersion: 1.0\n")
    (info / "RECORD").write_text(record)
    wheel = {"version": "1.0", "url": archive.as_uri(), "hash": "sha256:" + worker.digest(archive)}
    archive_info = json.loads(json.dumps(archive_info).replace("{digest}", worker.digest(archive)))
    (info / "direct_url.json").write_text(json.dumps({"url": wheel["url"], "archive_info": archive_info}))
    work = tmp_path / "work"
    work.mkdir()
    receipt = worker.attest_wheel(PathDistribution(info), wheel, work, entry, tmp_path / "receipt.json")
    assert receipt["result"] == "attested"
    assert receipt["runtime_file_count"] == 2
    # A matching URL and RECORD cannot conceal modified native bytes.
    (entry / "_C.abi3.so").write_bytes(b"tampered native payload")
    with pytest.raises(ValueError, match="Installed runtime payload"):
        worker.verify_wheel_content(PathDistribution(info), archive, entry)

    (info / "direct_url.json").write_text(
        json.dumps({"url": wheel["url"], "archive_info": {"hashes": {"sha256": "0" * 64}}})
    )
    conflict_work = tmp_path / "conflict-work"
    conflict_work.mkdir()
    with pytest.raises(ValueError, match="receipt hash differs"):
        worker.attest_wheel(PathDistribution(info), wheel, conflict_work, entry, tmp_path / "conflict-receipt.json")
    assert not (conflict_work / "pinned-vllm.whl").exists()


def test_actual_uv0135_receipt_without_hash_attests(tmp_path):
    worker = _worker_module()
    archive = tmp_path / "vllm-1.0-py3-none-any.whl"
    files = {
        "vllm/__init__.py": b"",
        "vllm/_C.abi3.so": b"CPU test fixture, never imported",
        "vllm-1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: vllm\nVersion: 1.0\n",
        "vllm-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    _wheel_archive(archive, files)
    environment = tmp_path / "environment"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    python = environment / "bin/python"
    # The pinned uv binary is already provisioned by the explicit CPU reproduction.
    subprocess.run(
        [
            "uv",
            "tool",
            "run",
            "--offline",
            "--from",
            "uv==0.13.5",
            "--python",
            "3.12",
            "--no-python-downloads",
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            archive.as_uri(),
        ],
        check=True,
    )
    site = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
        ).strip()
    )
    distribution = PathDistribution(site / "vllm-1.0.dist-info")
    direct = json.loads(distribution.read_text("direct_url.json"))
    assert direct == {"url": archive.as_uri(), "archive_info": {}}
    work = tmp_path / "work"
    work.mkdir()
    receipt = worker.attest_wheel(
        distribution,
        {"version": "1.0", "url": archive.as_uri(), "hash": "sha256:" + worker.digest(archive)},
        work,
        site / "vllm",
        tmp_path / "receipt.json",
    )
    assert receipt["result"] == "attested"


def test_attestation_failure_preserves_cache_and_receipt(tmp_path, monkeypatch):
    worker = _worker_module()
    environment = tmp_path / "environment"
    site = environment / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "_C.abi3.so").write_bytes(b"original native")
    entry = site / "vllm"
    entry.symlink_to(cache, target_is_directory=True)
    info = site / "vllm-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: vllm\nVersion: 1.0\n")
    bad = tmp_path / "wrong.whl"
    bad.write_bytes(b"wrong archive")
    (info / "direct_url.json").write_text(json.dumps({"url": bad.as_uri(), "archive_info": {}}))
    monkeypatch.setattr(worker.metadata, "distribution", lambda _name: PathDistribution(info))
    monkeypatch.setattr(worker.sys, "prefix", str(environment))
    work = tmp_path / "work"
    work.mkdir()
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="SHA256 differs"):
        worker.stage_owned_package(
            tmp_path,
            work,
            {"wheel": {"url": bad.as_uri(), "version": "1.0", "hash": "sha256:" + "a" * 64}},
            receipt_path,
        )
    assert entry.is_symlink()
    assert (cache / "_C.abi3.so").read_bytes() == b"original native"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["result"] == "failed"
    assert receipt["error_type"] == "ValueError"
    assert receipt["archive_sha256"] == hashlib.sha256(b"wrong archive").hexdigest()
    assert not (site / "vllm.before-hero-validation").exists()


class _BlobStore:
    def __init__(self):
        self.contents = {}

    def write(self, data):
        key = hashlib.sha256(data).hexdigest()
        self.contents[key] = data
        return key


def _project_and_stage(payload, projection: Path, workdir: Path, monkeypatch):
    store = _BlobStore()
    request = controller_pb2.Controller.LaunchJobRequest()
    request.entrypoint.workdir_files.update(payload)
    normalized = _store_launch_payload(SimpleNamespace(bundles=store), request)
    run_request = job_pb2.RunTaskRequest()
    run_request.entrypoint.workdir_files.update(normalized.entrypoint.workdir_files)
    run_request.entrypoint.workdir_file_refs.update(normalized.entrypoint.workdir_file_refs)
    _, volumes, _ = _build_init_container_spec(run_request, "local-staging", "unused-image", "https://unused.invalid")
    inline = dict(run_request.entrypoint.workdir_files)
    assert inline and run_request.entrypoint.workdir_file_refs
    binary_data = {f"f{index:04d}": data for index, (_, data) in enumerate(inline.items())}
    items = next(volume["configMap"]["items"] for volume in volumes if volume["name"] == "workdir-files")
    timestamp = projection / "..2026_11_04_atomic_writer"
    timestamp.mkdir(parents=True)
    for item in items:
        target = timestamp / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(binary_data[item["key"]])
    (projection / "..data").symlink_to(timestamp.name, target_is_directory=True)
    for name in {Path(item["path"]).parts[0] for item in items}:
        (projection / name).symlink_to(Path("..data") / name, target_is_directory=(timestamp / name).is_dir())
    workdir.mkdir()
    bundle_fetch.copy_workdir_files(str(projection), str(workdir))
    monkeypatch.setattr(
        bundle_fetch.urllib.request,
        "urlopen",
        lambda request, **_kwargs: io.BytesIO(store.contents[request.full_url.rsplit("/", 1)[-1]]),
    )
    bundle_fetch.fetch_workdir_blob_refs(
        "https://unused.invalid", json.dumps(dict(run_request.entrypoint.workdir_file_refs)), str(workdir)
    )


def test_flat_payload_survives_configmap_projection_and_blob_refs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("hero_validation_submit", Path(__file__).with_name("submit.py"))
    submit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submit)
    workspace, payload, manifest = submit.prepare(tmp_path / "prepared")
    legacy = {f"_hero_shortconv/{name}": data for name, data in payload.items()}
    _project_and_stage(legacy, tmp_path / "legacy-projection", tmp_path / "legacy-workdir", monkeypatch)
    assert not (tmp_path / "legacy-workdir/_hero_shortconv" / submit.MANIFEST_FILENAME).exists()
    assert (tmp_path / "legacy-workdir/_hero_shortconv" / submit.WORKER_FILENAME).is_file()

    workdir = tmp_path / "flat-workdir"
    _project_and_stage(payload, tmp_path / "flat-projection", workdir, monkeypatch)
    for path, data in payload.items():
        assert (workdir / path).read_bytes() == data
    for path in manifest["managed_inputs"]:
        target = workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / path, target)
    command = [
        sys.executable,
        str(workdir / submit.WORKER_FILENAME),
        "--manifest",
        str(workdir / submit.MANIFEST_FILENAME),
        "--manifest-sha256",
        submit.digest(payload[submit.MANIFEST_FILENAME]),
        "--stage",
        "both",
        "--check-staged",
    ]
    result = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    receipt = json.loads(result.stdout)
    assert receipt["result"] == "staged-payload-valid"
    assert receipt["workspace"] == str(workdir)
    assert receipt["source_revision"] == submit.SOURCE_REVISION
