# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution-only policy at the native submit boundary; no evaluator changes."""

from iris.client.client import IrisClient
from iris.rpc import job_pb2
from rigging.timing import Duration


def install():
    original = IrisClient.submit

    def submit(self, *args, **kwargs):
        kwargs.update(
            max_retries_failure=0,
            max_retries_preemption=0,
            priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            timeout=Duration.from_hours(12),
        )
        return original(self, *args, **kwargs)

    IrisClient.submit = submit
