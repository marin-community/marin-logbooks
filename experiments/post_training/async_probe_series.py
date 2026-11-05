# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the remaining fixed windows serially, stopping their chain after failure."""

from dataclasses import replace

import click
from marin.execution.build_context import resolve_version
from marin.rl.cli import rl_build_options

from experiments.post_training.async_probe_cost import build_comparison


@click.command(help=__doc__)
@rl_build_options
def main():
    if click.get_current_context().params["max_concurrent"] != 1:
        raise click.UsageError("The fixed comparison requires --max-concurrent 1.")
    version = resolve_version("async-probe-cost", None)
    windows = []
    for seed, interval in ((1701, 10), (2903, 10), (2903, 0), (4517, 0), (4517, 10)):
        window = build_comparison(version, interval, seed, False)
        if windows:
            window = replace(window, deps=(*window.deps, windows[-1]))
        windows.append(window)
    return windows


if __name__ == "__main__":
    main()
