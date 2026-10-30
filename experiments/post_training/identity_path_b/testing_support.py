# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concrete small storage models and the external-compute test boundary."""

import inspect
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import equinox as eqx
import jax
from iris.resources.state import JobState
from iris.rpc import job_pb2


class Layer(eqx.Module):
    w_q: jax.Array
    w_k: jax.Array
    router: jax.Array
    router_bias: jax.Array


class StorageModel(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array
    layers: list[Layer]
    others: dict[str, jax.Array]

    def to_state_dict(self):
        result = {"model.embed_tokens.weight": self.token_embed, "lm_head.weight": self.output_proj.T, **self.others}
        for index, layer in enumerate(self.layers):
            for suffix, value in (
                ("self_attn.q_proj.weight", layer.w_q),
                ("self_attn.k_proj.weight", layer.w_k),
                ("mlp.router.weight", layer.router),
                ("mlp.router.bias", layer.router_bias),
            ):
                result[f"model.layers.{index}.{suffix}"] = value
        return result


class ExecutionBoundary:
    """Fake external compute; lifecycle decisions use the real Iris controller."""

    def __init__(self, world, root):
        self.world, self.root = world, root
        self.changed = threading.Condition()
        self.jobs = {}

    def submit(self, entrypoint, *, name, replicas, **kwargs):
        # Exercise the native callable deserialization and actual signatures,
        # including keyword-only train_worker; no args=/kwargs= wrappers.
        fn, args, call_kwargs = entrypoint.resolve()
        inspect.signature(fn).bind(*args, **call_kwargs)
        with self.changed:
            ref = self.world.submit_child(self.root, name, tasks=replicas)
            self.world.settle()
            future = Future()
            job = SimpleNamespace(
                job_id=ref.wire_id,
                ref=ref,
                kwargs=call_kwargs,
                submit_options=kwargs,
                wait=lambda **unused: future.result(timeout=30),
                future=future,
            )
            self.jobs[name] = job
            self.changed.notify_all()
            return job

    def get(self, name):
        with self.changed:
            assert self.changed.wait_for(lambda: name in self.jobs, timeout=10), name
            return self.jobs[name]

    def finish(self, name, *, success=True):
        with self.changed:
            job = self.jobs[name]
            if success:
                self.world.succeed_all(job.ref)
            else:
                self.world.fail(job.ref[0])
            self.world.settle()
            state = self.world.job(job.ref).state
            terminal = JobState.SUCCEEDED if state == job_pb2.JOB_STATE_SUCCEEDED else JobState.FAILED
            job.future.set_result(SimpleNamespace(state=terminal))
            return job
