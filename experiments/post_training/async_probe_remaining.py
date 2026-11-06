# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the five uncompleted windows after the first coordinator preemption."""

import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import click
from fray.iris_backend import FrayIrisClient
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from marin.execution.build_context import resolve_version
from marin.experiment.cli import build_options_with_runner
from marin.rl import cli
from rigging.config_discovery import find_project_root
from rigging.timing import Duration

from experiments.post_training.async_probe_cost import build_comparison

DEADLINE = datetime.fromisoformat("2026-11-06T17:03:25.252358+00:00")


def submit_or_run(handles, max_concurrent):
    module = cli._module_name(sys.modules[__name__])
    if get_job_info() is not None:
        cli._submit_or_run(module, handles, max_concurrent, os.environ)
        return
    remaining = int((DEADLINE - datetime.now(UTC)).total_seconds())
    if remaining <= 0:
        raise click.UsageError("The original cohort deadline has passed.")
    workspace = find_project_root()
    config, request = cli._coordinator_request(
        handles, module, tuple(sys.argv[1:]), max_concurrent, workspace, os.environ
    )
    # Protect the CPU parent from same-band GPU work; every GPU child stays batch.
    request.priority = job_pb2.PRIORITY_BAND_INTERACTIVE
    request.timeout = Duration.from_seconds(remaining)
    with open_iris_client(config_file=Path(config), workspace=workspace) as client:
        handle = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=False)
    click.echo(f"submitted continuation coordinator {handle.job_id}")


def build_windows():
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


main = click.command(help=__doc__)(build_options_with_runner(build_windows, submit_or_run))


if __name__ == "__main__":
    main()
