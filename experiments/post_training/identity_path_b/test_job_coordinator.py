# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native JAX discovery with sibling jobs sharing one actor namespace."""

import jax
import pytest
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.types import JobName
from iris.jax import init as bootstrap
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import EndpointServiceClientSync


@pytest.mark.parametrize("supervised", [False, True])
def test_sibling_jax_worlds_and_attempts_resolve_only_their_own_leader(monkeypatch, supervised):
    # Real native contexts, registries, resolvers and remote clients. Only the
    # external endpoint RPC and JAX distributed initialization are replaced.
    endpoints = {}
    joined = []

    def register(self, request, **unused):
        endpoints[request.endpoint_id] = controller_pb2.Controller.Endpoint(
            endpoint_id=request.endpoint_id, name=request.name, address=request.address
        )
        return controller_pb2.Controller.RegisterEndpointResponse(
            endpoint_id=request.endpoint_id, lease_duration=request.lease_duration
        )

    def listing(self, request, **unused):
        assert request.exact
        return controller_pb2.Controller.ListEndpointsResponse(
            endpoints=[ep for ep in endpoints.values() if ep.name == request.prefix]
        )

    def unregister(self, request, **unused):
        endpoints.pop(request.endpoint_id, None)
        return None

    monkeypatch.setattr(EndpointServiceClientSync, "register_endpoint", register)
    monkeypatch.setattr(EndpointServiceClientSync, "list_endpoints", listing)
    monkeypatch.setattr(EndpointServiceClientSync, "unregister_endpoint", unregister)
    monkeypatch.setattr(bootstrap, "configure_jax_compilation_cache", lambda: None)
    monkeypatch.setattr(bootstrap, "install_gpu_compile_guard", lambda: None)
    monkeypatch.setattr(bootstrap.atexit, "register", lambda *args: None)
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(jax.distributed, "initialize", lambda address, *args, **kwargs: joined.append(address))
    client = IrisClient(RemoteClusterClient("http://unused.invalid", use_controller_proxy=False))

    def initialize(job, rank, host, attempt=0):
        info = JobInfo(
            task_id=JobName.from_wire(f"/fixture/root/{job}/{rank}"),
            num_tasks=4,
            attempt_id=attempt,
            advertise_host=host,
            ports={"jax": 9876},
        )
        monkeypatch.setattr(bootstrap, "get_job_info", lambda: info)
        if supervised:
            monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_COUNT", "32")
            monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", str(rank * 8))
            monkeypatch.setenv("IRIS_MULTIGPU_LOCAL_DEVICE_IDS", "0")
        with iris_ctx_scope(IrisContext.from_job_info(info, client)):
            bootstrap.initialize_jax(poll_timeout=0, poll_interval=0.01)

    try:
        initialize("train", 0, "train-host")
        # Neither another sibling nor a new attempt may join the existing world.
        for job, attempt in [("export35", 0), ("export70", 0), ("train", 1)]:
            with pytest.raises(TimeoutError):
                initialize(job, 1, "peer", attempt)
        initialize("export35", 0, "export35-host")
        initialize("export70", 0, "export70-host")
        initialize("train", 0, "retry-host", 1)
        for job, host, attempt in [
            ("train", "train-host", 0),
            ("export35", "export35-host", 0),
            ("export70", "export70-host", 0),
            ("train", "retry-host", 1),
        ]:
            initialize(job, 1, "peer", attempt)
            assert joined[-1] == f"{host}:9876"
        assert len(endpoints) == 4
        assert len({ep.name for ep in endpoints.values()}) == 4
        assert joined == [
            f"{host}:9876"
            for host in [
                "train-host",
                "export35-host",
                "export70-host",
                "retry-host",
                "train-host",
                "export35-host",
                "export70-host",
                "retry-host",
            ]
        ]
    finally:
        client.shutdown()
