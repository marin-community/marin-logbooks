# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-host Stage-A worker supervisor; no submission or client logic."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def archive_diagnostics(output, archive, limits):
    files, skipped = [], []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in diagnostic output: {path}")
        if not path.is_file():
            continue
        relative = str(path.relative_to(output))
        if not (path.suffix == ".json" or path.name.endswith("hlo.txt")):
            raise ValueError(f"Unexpected diagnostic output: {relative}")
        if path.stat().st_size > limits["per_file_bytes"]:
            raise ValueError(f"Diagnostic file exceeds hard cap; not truncated: {relative} ({path.stat().st_size})")
        files.append(path)
    uncompressed = sum(path.stat().st_size for path in files)
    print(
        "DIAGNOSTIC_PREFLIGHT "
        + json.dumps(
            {
                "files": len(files),
                "uncompressed_bytes": uncompressed,
                "largest_file_bytes": max((p.stat().st_size for p in files), default=0),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if len(files) > limits["files"] or uncompressed > limits["uncompressed_bytes"]:
        raise ValueError("Diagnostic archive exceeds hard count/raw cap; not truncated")
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(output)), recursive=False)
    compressed = archive.stat().st_size
    if compressed > limits["compressed_bytes"]:
        raise ValueError("Compressed diagnostic cap exceeded")
    return {
        "compressed_bytes": compressed,
        "uncompressed_bytes": uncompressed,
        "entries": len(files),
        "sha256": sha256(archive),
        "skipped_perfetto": skipped,
        "files": [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size} for p in files],
    }


def main():
    transport = json.loads(sys.argv[1])
    scientific = json.loads(transport["scientific_config_text"])
    if "jax" in sys.modules:
        raise RuntimeError("Worker supervisor must not initialize JAX")
    expected = {
        "JAX_PLATFORMS": "tpu",
        "XLA_FLAGS": "--xla_allow_excess_precision=false",
        "LIBTPU_INIT_ARGS": " --xla_tpu_use_enhanced_launch_barrier=true",
    }
    if scientific["scientific_environment"] != expected or any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError("Exact inherited scientific environment required")
    if (
        transport["scientific_source_commit"] != scientific["source_commit"]
        or scientific["source_commit"] != "2b17db78aedc0a603664238fbfa0bc0c68542d2b"
    ):
        raise ValueError("Wrong frozen scientific source revision")
    if sha256(Path(__file__)) != transport["runner_sha256"]:
        raise ValueError("Worker supervisor bytes changed")
    if (
        transport["backup_uri"]
        != "gs://marin-us-central1/tmp/ttl=7d/native-jax/oproj-stage-a-15b-20261024/diagnostics.tar.gz"
    ):
        raise ValueError("Wrong separately named same-region backup")
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a fresh empty diagnostic output directory")
    backup = None
    status = 1
    try:
        for name, expected_hash in {**scientific["source_file_sha256"], **transport["research_source_sha256"]}.items():
            if sha256(Path(name)) != expected_hash:
                raise ValueError(f"Worker source bytes changed: {name}")
        (output / "worker-provenance.json").write_text(
            json.dumps(
                {
                    "transport": transport,
                    "scientific_config_sha256": transport["scientific_config_sha256"],
                    "parent_jax_imported": False,
                    "source_pin_declared_not_worker_git_checked": True,
                },
                indent=2,
            )
            + "\n"
        )
        with tempfile.TemporaryDirectory(prefix="oproj-worker-config-") as temporary:
            config_path = Path(temporary) / "config.json"
            # Preserve exact frozen config bytes, including formatting, from embedded text.
            config_path.write_text(transport["scientific_config_text"])
            if (
                sha256(config_path) != transport["scientific_config_sha256"]
                or json.loads(config_path.read_text()) != scientific
            ):
                raise ValueError("Embedded scientific config bytes differ")
            command = [
                sys.executable,
                "research/oproj/diagnostic.py",
                "run",
                "--mode",
                "target",
                "--config",
                str(config_path),
                "--output",
                str(output / "stage-a"),
            ]
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            argv_bytes = sum(len(v.encode()) + 1 for v in command)
            env_bytes = sum(len((k + "=" + v).encode()) + 1 for k, v in environment.items())
            if argv_bytes + env_bytes + 16 * 1024 >= 96 * 1024:
                raise ValueError("Child argv/environment exceeds inherited96KiB cap")
            result = subprocess.run(command, env=environment)
            report = json.loads((output / "stage-a/report.json").read_text())
            status = (
                0
                if result.returncode == 0
                and report["completed"]
                and report["identity"]["instrumentation_loss_all_raw_gradients_exact"]
                else 1
            )
    except Exception as error:
        (output / "worker-failure.json").write_text(
            json.dumps(
                {
                    "completed": False,
                    "error": f"{type(error).__name__}: {error}",
                    "guarantee": "Ordinary Python exception retention only; no hard-termination/SIGKILL guarantee.",
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        active_error = sys.exception()
        try:
            archive = output.parent / (output.name + "-diagnostics.tar.gz")
            summary = archive_diagnostics(output, archive, transport["archive_limits"])
            print(
                "DIAGNOSTIC_BACKUP " + json.dumps(summary | {"uri": transport["backup_uri"]}, sort_keys=True), flush=True
            )
            backup = subprocess.run(
                [sys.executable, "-m", "rigging.fsutil.cli", "cp", "-n", str(archive), transport["backup_uri"]],
                timeout=90,
            )
            print(
                "DIAGNOSTIC_BACKUP_RESULT "
                + json.dumps({"returncode": backup.returncode, "uri": transport["backup_uri"]}),
                flush=True,
            )
        except Exception as backup_error:
            if active_error is None:
                raise
            active_error.add_note("Diagnostic backup also failed: " + type(backup_error).__name__)
            print(
                "DIAGNOSTIC_BACKUP_EXCEPTION "
                + json.dumps({"exception_type": type(backup_error).__name__, "original_exception_preserved": True}),
                file=sys.stderr,
                flush=True,
            )
    raise SystemExit(1 if status or backup.returncode else 0)


if __name__ == "__main__":
    main()
