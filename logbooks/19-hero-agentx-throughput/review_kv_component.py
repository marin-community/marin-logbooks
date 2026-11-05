# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the revised PR's numerical and cache-layout tests in its private wheel."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from diagnostic_server import stage_candidate_wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    receipt_file = stage_candidate_wheel(manifest["candidate_wheel_input"])
    receipt = json.loads(receipt_file.read_text())
    source = Path(manifest["test_root"])
    for name, digest in manifest["test_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    args.output.mkdir(exist_ok=False)
    command = manifest["prefix"].copy()
    command[command.index("--from") + 1] = (
        "vllm @ " + Path(receipt["wheel_path"]).as_uri() + "#sha256=" + receipt["wheel_sha256"]
    )
    command += [
        "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", "/dev/null",
        "--confcutdir", str(source / "tests/models"),
        "--junitxml", str(args.output / "junit.xml"),
    ]
    command += [str(source / test) for test in manifest["tests"]]
    environment = dict(os.environ) | {
        "CUDA_VISIBLE_DEVICES": "0", "UV_NO_CONFIG": "1",
        "UV_CONSTRAINT": "/app/experiments/evaluation/agentx/requirements-vllm-hero.txt",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(source),
    }
    identity_command = manifest["prefix"].copy()
    identity_command[identity_command.index("--from") + 1] = command[command.index("--from") + 1]
    identity_command += ["python", "-c", """
import hashlib, importlib, json, sys
from pathlib import Path
receipt = json.loads(sys.argv[1])
hashes = {}
for name, expected in (receipt['changed_files'] | receipt['added_files']).items():
    module = importlib.import_module(name[:-3].replace('/', '.'))
    path = Path(module.__file__)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (name, actual, expected)
    hashes[name] = {'path': str(path), 'sha256': actual}
print(json.dumps(hashes))
""", json.dumps(receipt)]
    with (args.output / "loaded-source.json").open("w") as log:
        subprocess.run(identity_command, env=environment, stdout=log, check=True, timeout=180)
    record = {"manifest": manifest, "command": command, "started_unix": time.time()}
    (args.output / "command.json").write_text(json.dumps(record, indent=2) + "\n")
    with (args.output / "pytest.log").open("w") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    record.update(returncode=result.returncode, finished_unix=time.time())
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"returncode": result.returncode, "output": str(args.output)}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
