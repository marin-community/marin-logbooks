# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""New replay namespace/lifetime around the byte-pinned native T2 serving worker."""

import argparse
import hashlib
import re
import shlex
from dataclasses import replace
from pathlib import Path

from iris.cluster.types import Entrypoint
from marin.inference.iris import run_iris_service
from rigging.timing import Duration

from experiments.post_training.identity_t2 import serve as t2

T2_SERVE_SHA256 = "0b463b76b7e7086bda639bdf5d57ea498a5135d0520160abcb45bcb5614570b3"
TIMEOUT_SECONDS = 12 * 3600


def check_run(run_id: str) -> None:
    if re.fullmatch(r"pathb-replay-[a-z0-9-]{1,60}", run_id) is None:
        raise ValueError("Require a fresh pathb-replay-* owned job/endpoint namespace")
    if hashlib.sha256(Path(t2.__file__).read_bytes()).hexdigest() != T2_SERVE_SHA256:
        raise ValueError("Known-good native T2 serving worker changed")


def service_config(run_id: str):
    check_run(run_id)
    return replace(
        t2.service_config(), endpoint_name=f"/serve/{run_id}", timeout_hours=12, controller_proxy_timeout_seconds=900
    )


def worker_command(run_id: str) -> list[str]:
    check_run(run_id)
    script = (
        'set -euo pipefail\nroot_python="$IRIS_VENV/bin/python"\n'
        f"source {shlex.quote('/app/vendor/skyrl/.iris-runtime-env')}\n"
        'exec "$root_python" -u -m experiments.post_training.identity_path_b.replay_serve '
        f"--run-id {shlex.quote(run_id)}"
    )
    return ["bash", "-c", script]


def submission(run_id: str, credentials: dict[str, str]):
    """Return native submit arguments; never log this secret-bearing dictionary."""
    check_run(run_id)
    kwargs = t2.submission(credentials)
    kwargs.update(
        name=run_id,
        entrypoint=Entrypoint(command=worker_command(run_id)),
        timeout=Duration.from_seconds(TIMEOUT_SECONDS),
    )
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    check_run(args.run_id)
    t2.validate_runtime()
    t2.configure_worker_s3()
    run_iris_service(service_config(args.run_id))


if __name__ == "__main__":
    main()
