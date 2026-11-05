# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the task-local ShortConv experiment without changing shared wheel caches."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from importlib import import_module, metadata
from pathlib import Path
from urllib.parse import unquote


def digest(path: Path) -> str:
    with path.open("rb") as content:
        return hashlib.file_digest(content, "sha256").hexdigest()


def package_versions() -> dict[str, str]:
    return {item.metadata["Name"].lower().replace("_", "-"): item.version for item in metadata.distributions()}


def extension_hashes(package: Path) -> dict[str, str]:
    return {
        str(path.relative_to(package)): digest(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and (".so" in path.name or path.suffix in (".pyd", ".dylib"))
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate_payload(manifest_path: Path, expected: str, workspace: Path) -> dict:
    if digest(manifest_path) != expected:
        raise ValueError("Submission manifest digest changed")
    manifest = json.loads(manifest_path.read_text())
    for path, expected_hash in manifest["payload"].items():
        if digest(workspace / path) != expected_hash:
            raise ValueError(f"Payload changed: {path}")
    for path, expected_hash in manifest["managed_inputs"].items():
        if digest(workspace / path) != expected_hash:
            raise ValueError(f"Managed dependency input changed: {path}")
    return manifest


def bootstrap(args, workspace: Path, manifest: dict) -> None:
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("This bundle requires Linux aarch64 and the exact ARM wheel")
    work = Path(tempfile.mkdtemp(prefix="hero-shortconv-", dir="/cache"))
    environment = work / "environment"
    activation = work / "runtime.env"
    project = workspace / "vendor/skyrl"
    subprocess.run(
        [
            "bash",
            str(project / "cloud/iris/bootstrap_runtime.sh"),
            str(project),
            str(environment),
            str(activation),
            "megatron",
            "production",
        ],
        check=True,
    )
    command = [
        "bash",
        "-c",
        (
            'set -euo pipefail; source "$1"; exec "$2" "$3" --execute '
            '--manifest "$4" --manifest-sha256 "$5" --stage "$6" --work "$7"'
        ),
        "hero-shortconv",
        str(activation),
        str(environment / "bin/python"),
        str(Path(__file__).resolve()),
        str(args.manifest),
        args.manifest_sha256,
        args.stage,
        str(work),
    ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(VLLM_BATCH_INVARIANT="1", VLLM_USE_V2_MODEL_RUNNER="1", VLLM_ALLOW_INSECURE_SERIALIZATION="1")
    os.execvpe(command[0], command, env)


def install_test_tools(manifest: dict, baseline: dict[str, str]) -> None:
    tools = manifest["test_tools"]
    for name, version in tools.items():
        if name in baseline and baseline[name] != version:
            raise ValueError(f"Test tool would replace an existing managed dependency: {name}")
    missing = [f"{name}=={version}" for name, version in tools.items() if name not in baseline]
    if missing:
        subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-deps", *missing], check=True)
    after = package_versions()
    if any(after.get(name) != version for name, version in baseline.items()):
        raise ValueError("Test-tool installation changed a managed dependency")
    if set(after) - set(baseline) != set(tools) - set(baseline):
        raise ValueError("Test-tool installation added unexpected distributions")


def verify_wheel_content(distribution, archive_path: Path, entry: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        (record_name,) = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        records = {row[0]: row[1:] for row in csv.reader(io.StringIO(archive.read(record_name).decode()))}
        installed_records = {row[0]: row[1:] for row in csv.reader(io.StringIO(distribution.read_text("RECORD")))}
        package_files = {name for name in archive.namelist() if name.startswith("vllm/") and not name.endswith("/")}
        installed_files = {
            "vllm/" + str(path.relative_to(entry))
            for path in entry.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        if installed_files != package_files:
            raise ValueError("Installed vLLM runtime file set differs from the verified wheel")
        hashes = {}
        for name in sorted(package_files):
            expected_hash, expected_size = records[name]
            if installed_records.get(name) != records[name]:
                raise ValueError(f"Installed RECORD differs from verified wheel: {name}")
            with archive.open(name) as content:
                actual = hashlib.file_digest(content, "sha256").hexdigest()
            encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).rstrip(b"=").decode()
            if expected_hash != "sha256=" + encoded or int(expected_size) != archive.getinfo(name).file_size:
                raise ValueError(f"Verified wheel payload differs from its RECORD: {name}")
            installed = entry / Path(name).relative_to("vllm")
            if installed.stat().st_size != int(expected_size) or digest(installed) != actual:
                raise ValueError(f"Installed runtime payload differs from verified wheel: {name}")
            hashes[name] = actual
        native = {
            name: value
            for name, value in hashes.items()
            if ".so" in Path(name).name or Path(name).suffix in (".pyd", ".dylib")
        }
        if not native:
            raise ValueError("Pinned wheel has no compiled extensions")
        return {
            "runtime_file_count": len(hashes),
            "runtime_manifest_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "native_sha256": native,
            "installed_record_matches": True,
        }


def _attest_wheel(distribution, wheel: dict, work: Path, entry: Path, receipt_path: Path) -> dict:
    direct = json.loads(distribution.read_text("direct_url.json"))
    receipt = {"result": "checking-wheel-identity", "installed_version": distribution.version, "direct_url": direct}
    write_json(receipt_path, receipt)
    if distribution.version != wheel["version"]:
        raise ValueError("Installed vLLM is not the pinned 39e628 wheel")
    if unquote(direct["url"]) != unquote(wheel["url"]):
        raise ValueError("Installed vLLM wheel URL differs from the lock")
    info = direct["archive_info"]
    expected = wheel["hash"].removeprefix("sha256:")
    receipt_hashes = dict(info.get("hashes", {}))
    if "hash" in info:
        algorithm, value = info["hash"].split("=", 1)
        if algorithm in receipt_hashes and receipt_hashes[algorithm] != value:
            raise ValueError("Wheel receipt hash fields disagree")
        receipt_hashes[algorithm] = value
    if "sha256" in receipt_hashes and receipt_hashes["sha256"] != expected:
        raise ValueError("Installed vLLM wheel receipt hash differs from the lock")
    archive_path = work / "pinned-vllm.whl"
    receipt.update(result="fetching-locked-wheel", expected_sha256=expected, archive_path=str(archive_path))
    write_json(receipt_path, receipt)
    # This task owns the download; never infer a private uv cache path.
    with urllib.request.urlopen(wheel["url"], timeout=60) as response, archive_path.open("xb") as target:
        copied = 0
        while chunk := response.read(1024 * 1024):
            copied += len(chunk)
            if copied > 512 * 1024 * 1024:
                raise ValueError("Locked-wheel download exceeds the 512 MiB task-local bound")
            target.write(chunk)
    actual = digest(archive_path)
    receipt.update(result="checking-installed-runtime", archive_sha256=actual, archive_bytes=archive_path.stat().st_size)
    write_json(receipt_path, receipt)
    if actual != expected:
        raise ValueError("Downloaded wheel SHA256 differs from the exact lock")
    for algorithm, expected_hash in receipt_hashes.items():
        with archive_path.open("rb") as content:
            actual_hash = hashlib.file_digest(content, algorithm).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Wheel receipt {algorithm} hash differs from verified archive")
    receipt.update(verify_wheel_content(distribution, archive_path, entry), result="attested")
    write_json(receipt_path, receipt)
    return receipt


def attest_wheel(distribution, wheel: dict, work: Path, entry: Path, receipt_path: Path) -> dict:
    try:
        return _attest_wheel(distribution, wheel, work, entry, receipt_path)
    except Exception as error:
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        receipt.update(result="failed", error_type=type(error).__name__, error=str(error)[:1000])
        write_json(receipt_path, receipt)
        raise


def stage_owned_package(workspace: Path, work: Path, manifest: dict, receipt_path: Path) -> dict:
    distribution = metadata.distribution("vllm")
    entry = Path(distribution.locate_file("vllm"))
    if not entry.is_relative_to(Path(sys.prefix)):
        raise ValueError("vLLM package entry is outside the disposable environment")
    wheel_attestation = attest_wheel(distribution, manifest["wheel"], work, entry, receipt_path)
    before_extensions = extension_hashes(entry.resolve())
    if not before_extensions:
        raise ValueError("Pinned wheel has no compiled extensions")
    for item in manifest["overlay"]:
        relative = Path(item["path"]).relative_to("vllm")
        if digest(entry / relative) != item["base_sha256"]:
            raise ValueError(f"Installed wheel preimage differs from pinned source: {relative}")
    owned = work / "owned-vllm"
    shutil.copytree(entry, owned, symlinks=False)
    original = entry.with_name("vllm.before-hero-validation")
    if original.exists() or original.is_symlink():
        raise ValueError("Disposable environment has already been overlaid")
    # Rename the environment entry itself, including when it is a symlink;
    # writes below now target only a real owned copy, never uv's package cache.
    entry.rename(original)
    os.replace(owned, entry)
    if entry.is_symlink() or any(path.is_symlink() for path in entry.rglob("*")):
        raise ValueError("Owned package still contains symlinks")
    for item in manifest["overlay"]:
        relative = Path(item["path"]).relative_to("vllm")
        target = entry / relative
        content = workspace / item["payload_path"]
        if digest(target) != item["base_sha256"] or digest(content) != item["patched_sha256"]:
            raise ValueError(f"Overlay verification failed: {relative}")
        temporary = target.with_name(target.name + ".hero-new")
        temporary.write_bytes(content.read_bytes())
        temporary.chmod(target.stat().st_mode)
        os.replace(temporary, target)
        if digest(target) != item["patched_sha256"]:
            raise ValueError(f"Atomic replacement verification failed: {relative}")
    after_extensions = extension_hashes(entry)
    if after_extensions != before_extensions:
        raise ValueError("A compiled extension changed while staging the Python overlay")
    return {
        "installed_package": str(entry),
        "original_package_entry": str(original),
        "extensions_before": before_extensions,
        "extensions_after": after_extensions,
        "overlay": manifest["overlay"],
        "wheel_attestation": wheel_attestation,
    }


def run_focused(workspace: Path, source: Path, run_dir: Path, output: Path, env: dict) -> dict:
    report = output / "focused.xml"
    command = [
        sys.executable,
        str(source / "infra/release/gpu_validation.py"),
        "run-wheel-tests",
        "--package-source-root",
        str(source),
        "--validation-source-root",
        str(source),
        "--",
        "-c",
        str(source / "pyproject.toml"),
        "--rootdir",
        str(source),
        str(source / "tests/models/test_grugmoe.py") + "::test_grug_moe_hf_config_loads_hero_fields",
        str(source / "tests/models/test_grugmoe.py")
        + "::test_hero_sconv_piecewise_replay_uses_current_requests_and_history",
        "-vv",
        "-s",
        "-rs",
        f"--junitxml={report}",
    ]
    subprocess.run(command, cwd=run_dir, env=env, check=True)
    cases = list(ET.parse(report).iter("testcase"))
    if len(cases) != 7 or any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")):
        raise ValueError("Focused qualification requires seven passing cases and zero skips")
    return {"command": command, "passed": len(cases), "skipped": 0}


def execute(args, workspace: Path, manifest: dict) -> None:
    work = args.work.resolve()
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", str(work / "output")))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "validation.json", {"result": "preflight", "manifest_sha256": args.manifest_sha256})
    baseline = package_versions()
    for name, expected in manifest["expected_core"].items():
        if baseline[name].split("+", 1)[0] != expected:
            raise ValueError(f"Managed core dependency differs from the requested profile: {name}")
    install_test_tools(manifest, baseline)
    receipt = stage_owned_package(workspace, work, manifest, output / "wheel-attestation.json")
    receipt.update(
        manifest_sha256=args.manifest_sha256,
        source_revision=manifest["source_revision"],
        managed_packages=baseline,
        test_packages=package_versions(),
        stage=args.stage,
        result="running",
    )
    write_json(output / "validation.json", receipt)
    # Import only after staging; this process's module cache must contain the
    # installed wheel, and the test tree never supplies a runtime vllm package.
    torch = import_module("torch")
    vllm = import_module("vllm")

    if not torch.cuda.is_available() or platform.machine() != "aarch64":
        raise ValueError("Validation requires a real ARM CUDA device")
    if not Path(vllm.__file__).resolve().is_relative_to(Path(receipt["installed_package"])):
        raise ValueError("vLLM import bypassed the owned installed-wheel package")
    receipt["hardware"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
    }
    source = work / "test_source"
    source.mkdir()
    with tarfile.open(workspace / manifest["test_source_payload"]) as archive:
        archive.extractall(source, filter="data")
    if (source / "vllm").exists():
        raise ValueError("Test source must not contain a runtime vLLM package")
    run_dir = work / "run"
    run_dir.mkdir()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        VLLM_BATCH_INVARIANT="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_ALLOW_INSECURE_SERIALIZATION="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
    )
    if args.stage in ("focused", "both"):
        receipt["focused"] = run_focused(workspace, source, run_dir, output, env)
    if args.stage in ("engine", "both"):
        for mode in ("NONE", "PIECEWISE"):
            subprocess.run(
                [
                    sys.executable,
                    str(workspace / manifest["tiny_engine_payload"]),
                    "--mode",
                    mode,
                    "--work",
                    str(work / f"engine-{mode.lower()}"),
                    "--output",
                    str(output / f"engine-{mode.lower()}.json"),
                    "--test-source",
                    str(source),
                ],
                cwd=run_dir,
                env=env,
                check=True,
            )
        reference = json.loads((output / "engine-none.json").read_text())
        captured = json.loads((output / "engine-piecewise.json").read_text())
        if reference["weights_sha256"] != captured["weights_sha256"]:
            raise ValueError("Dummy model weights differ between reference and captured engines")
        if reference["outputs"] != captured["outputs"]:
            raise ValueError("Captured engine token/logprob outputs differ from compiled uncaptured reference")
        receipt["engine"] = {"reference": reference, "captured": captured, "exact_output_match": True}
    if package_versions() != receipt["test_packages"]:
        raise ValueError("Installed dependency versions changed during validation")
    if extension_hashes(Path(receipt["installed_package"])) != receipt["extensions_before"]:
        raise ValueError("Compiled extension content changed during validation")
    receipt["result"] = "passed"
    write_json(output / "validation.json", receipt)
    print(json.dumps({"result": "passed", "output": str(output), "stage": args.stage}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--stage", choices=("focused", "engine", "both"), required=True)
    parser.add_argument("--check-staged", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    workspace = args.manifest.parent
    manifest = validate_payload(args.manifest, args.manifest_sha256, workspace)
    if args.check_staged:
        print(
            json.dumps(
                {
                    "result": "staged-payload-valid",
                    "workspace": str(workspace),
                    "source_revision": manifest["source_revision"],
                    "payload_files": sorted(manifest["payload"]),
                }
            )
        )
        return
    if args.execute:
        if args.work is None:
            raise ValueError("Execution phase requires its freshly bootstrapped work directory")
        execute(args, workspace, manifest)
    else:
        bootstrap(args, workspace, manifest)


if __name__ == "__main__":
    main()
