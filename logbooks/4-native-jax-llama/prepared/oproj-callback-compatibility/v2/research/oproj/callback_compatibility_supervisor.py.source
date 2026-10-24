# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Enforce the tiny gate's wall deadline outside its scientific Python process."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or args.timeout > 300:
        raise ValueError("A worker command and positive deadline no greater than 300 seconds are required")
    receipt = {"command": command, "timeout_seconds": args.timeout, "timed_out": False, "completed": False}
    started = time.monotonic()
    with subprocess.Popen(command, start_new_session=True) as process:
        receipt["owned_process_group"] = process.pid
        try:
            remaining = max(0, args.timeout - (time.monotonic() - started))
            process.wait(timeout=remaining)
            receipt["completed"] = process.returncode == 0
        except subprocess.TimeoutExpired:
            receipt["timed_out"] = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            receipt["worker_returncode"] = process.returncode
            receipt["elapsed_seconds"] = time.monotonic() - started
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(124 if receipt["timed_out"] else process.returncode)


if __name__ == "__main__":
    main()
