# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Online logging preflight before a frozen-checkpoint geometry worker."""

import os
import subprocess
import sys


def main():
    command = [sys.executable, "-m", "experiments.ema_gram.geometry_worker", *sys.argv[1:]]
    subprocess.run([*command, "--logging-check"], check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
