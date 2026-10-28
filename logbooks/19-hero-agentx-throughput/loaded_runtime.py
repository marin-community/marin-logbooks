# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record mapped native libraries and cached FlashInfer cubins without GPU calls.

Take the snapshot while serving. Hash its recorded files after the measurement;
reject changed file metadata. Cached cubin membership alone does not prove that
a kernel ran. The native logs and separate GPU traces identify executed paths.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def snapshot(pid_file: Path) -> dict:
    root = int(pid_file.read_text())
    os.kill(root, 0)
    parents = [
        tuple(map(int, line.split()))
        for line in subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()
    ]
    processes = {root}
    while children := {pid for pid, parent in parents if parent in processes} - processes:
        processes.update(children)
    libraries = {}
    for pid in sorted(processes):
        try:
            maps = Path(f"/proc/{pid}/maps").read_text().splitlines()
        except FileNotFoundError:
            continue
        libraries[str(pid)] = sorted(
            {
                line.split(maxsplit=5)[5]
                for line in maps
                if len(line.split(maxsplit=5)) == 6
                and any(name in line for name in ("/vllm/", "flashinfer", "nccl"))
                and ".so" in line
            }
        )
    paths = {Path(name) for names in libraries.values() for name in names}
    flashinfer = Path.home() / ".cache/flashinfer"
    paths.update(
        path for path in flashinfer.rglob("*") if path.suffix in (".cubin", ".so") or path.name == "checksums.txt"
    )
    files = {}
    for path in sorted(paths):
        stat = path.stat()
        files[str(path)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"observed_unix": time.time(), "root_pid": root, "mapped_libraries": libraries, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pid-file", type=Path)
    mode.add_argument("--hash-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pid_file is not None:
        record = snapshot(args.pid_file)
    else:
        record = json.loads(args.hash_snapshot.read_text())
        for name, entry in record["files"].items():
            path = Path(name)
            stat = path.stat()
            assert stat.st_size == entry["bytes"] and stat.st_mtime_ns == entry["mtime_ns"], name
            with path.open("rb") as stream:
                entry["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        record["hashed_unix"] = time.time()
    record["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print("recorded", len(record["files"]), "files", args.output, flush=True)


if __name__ == "__main__":
    main()
