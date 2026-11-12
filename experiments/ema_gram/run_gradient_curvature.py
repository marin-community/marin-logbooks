# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check online logging on every host before the paired curvature measurements."""

import os
import subprocess
import sys


def main():
    command = [sys.executable, "-m", "experiments.ema_gram.gradient_curvature", *sys.argv[1:]]
    subprocess.run([*command, "--logging-check"], check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
