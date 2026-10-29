# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.types import JobName
from iris.resources.state import JobState
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import EndpointServiceClientSync

from experiments.post_training.identity_path_b import trend


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]

    def advance(seconds):
        now[0] += seconds

    monkeypatch.setattr(trend.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(trend.time, "sleep", advance)
    return now


@pytest.fixture
def native_context():
    # Real high-level and remote clients; tests replace only the external
    # ListEndpoints RPC method, not either client's resolution implementation.
    remote = RemoteClusterClient("http://unused-controller.invalid", use_controller_proxy=False)
    client = IrisClient(remote)
    try:
        with iris_ctx_scope(IrisContext(job_id=JobName.from_wire("/fixture/trend"), client=client)):
            yield
    finally:
        client.shutdown()


@pytest.mark.parametrize("initial", ["empty_registry", "rpc_not_found"])
def test_native_endpoint_absent_then_registered_reuses_readiness_deadline(monkeypatch, clock, native_context, initial):
    attempts = []

    def list_endpoints(self, request, **unused):
        attempts.append((request.prefix, request.exact))
        if len(attempts) <= 2:
            if initial == "rpc_not_found":
                raise ConnectError(Code.NOT_FOUND, "not registered")
            return controller_pb2.Controller.ListEndpointsResponse()
        return controller_pb2.Controller.ListEndpointsResponse(
            endpoints=[controller_pb2.Controller.Endpoint(name=request.prefix, address="http://owned-serve:8000/")]
        )

    monkeypatch.setattr(EndpointServiceClientSync, "list_endpoints", list_endpoints)
    job = SimpleNamespace(status=lambda: SimpleNamespace(state=JobState.RUNNING))
    assert trend.wait_for_owned_endpoint(job, "/serve/exact-owned") == "http://owned-serve:8000"
    assert attempts == [("/serve/exact-owned", True)] * 3
    assert clock[0] == 4.0


def test_absent_native_endpoint_stops_at_deadline_and_on_terminal_job(monkeypatch, clock, native_context):
    monkeypatch.setattr(
        EndpointServiceClientSync,
        "list_endpoints",
        lambda *args, **kwargs: controller_pb2.Controller.ListEndpointsResponse(),
    )
    monkeypatch.setattr(trend, "CHILD_LIFETIME", 3603)
    running = SimpleNamespace(status=lambda: SimpleNamespace(state=JobState.RUNNING))
    with pytest.raises(TimeoutError, match="fixed startup deadline"):
        trend.wait_for_owned_endpoint(running, "/serve/exact-owned")
    assert clock[0] == 3.0
    states = iter([JobState.RUNNING, JobState.FAILED])
    terminal = SimpleNamespace(status=lambda: SimpleNamespace(state=next(states)))
    with pytest.raises(RuntimeError, match="finished before endpoint registration"):
        trend.wait_for_owned_endpoint(terminal, "/serve/exact-owned")
    assert clock[0] == 5.0


@pytest.mark.parametrize(
    "error",
    [
        ConnectError(Code.PERMISSION_DENIED, "permission denied"),
        ConnectError(Code.INVALID_ARGUMENT, "invalid endpoint request"),
        ConnectError(Code.CANCELED, "cancelled RPC"),
        ValueError("malformed response"),
        asyncio.CancelledError(),
    ],
)
def test_native_endpoint_permission_fatal_and_cancellation_propagate(monkeypatch, clock, native_context, error):
    def list_endpoints(*args, **kwargs):
        raise error

    monkeypatch.setattr(EndpointServiceClientSync, "list_endpoints", list_endpoints)
    job = SimpleNamespace(status=lambda: SimpleNamespace(state=JobState.RUNNING))
    with pytest.raises(type(error)) as raised:
        trend.wait_for_owned_endpoint(job, "/serve/exact-owned")
    assert raised.value is error
    assert clock[0] == 0.0
