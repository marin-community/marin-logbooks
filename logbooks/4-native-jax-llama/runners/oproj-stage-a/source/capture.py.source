# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research-only layer-indexed identity taps; production contraction stays ordinary AD."""

import ast
import hashlib
import inspect
import textwrap
import threading
from pathlib import Path

import haliax as hax
import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.layers.attention import Attention
from levanter.models.llama import LlamaDecoderLayer, LlamaTransformer
from levanter.models.llama_native import NativeLlamaAttention, NativeLlamaBlock, NativeLlamaModel

COLLECTORS = {}
PATCHES = []


def typed(array):
    array = np.asarray(array)
    data = array.tobytes(order="C")
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(data).hexdigest()}


class Collector:
    def __init__(self, mesh):
        self.mesh = mesh
        self.lock = threading.Lock()
        self.values = {}
        self.counts = {}
        self.errors = []

    def receive(self, label, partitioned, value, index, device_ids):
        with self.lock:
            try:
                if int(np.asarray(index)) != 0:
                    raise ValueError("Captured a layer other than actual dynamic scan index zero")
                ids = np.asarray(device_ids).reshape(-1)
                if partitioned and len(ids) != 1:
                    raise ValueError(f"Partitioned device-id shard has shape {ids.shape}")
                identity = int(ids[0]) if partitioned else "global"
                key = (label, identity)
                value = np.asarray(value).copy()
                if key in self.values and typed(value) != typed(self.values[key]):
                    raise ValueError(f"Repeated/rematerialized callback differs: {key}")
                self.values[key] = value
                self.counts[key] = self.counts.get(key, 0) + 1
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")

    def finish(self):
        if self.errors:
            raise ValueError(self.errors)
        result = {}
        for label, spec in (("X", P("data", "model", None, None)), ("G", P("data", None, None))):
            global_value = self.values[(label, "global")]
            if str(global_value.dtype) != "bfloat16" or not np.isfinite(global_value.astype(np.float32)).all():
                raise ValueError(f"{label} must be finite actual BF16")
            sharding = NamedSharding(self.mesh, spec)
            shards = []
            for identity, device in enumerate(self.mesh.devices.flat):
                indices = sharding.devices_indices_map(global_value.shape)[device]
                local_value = self.values[(label, identity)]
                expected = global_value[indices]
                if typed(local_value) != typed(expected):
                    raise ValueError(f"Actual partitioned {label} callback differs from canonical shard {identity}")
                shards.append(
                    {
                        "device": str(device),
                        "ordinal": identity,
                        "indices": [[s.start, s.stop, s.step] for s in indices],
                        "value": typed(local_value),
                        "callbacks": self.counts[(label, identity)],
                    }
                )
            result[label] = {
                "global": typed(global_value),
                "placement_scope": "callback-only canonical capture view, not original tiled operand buffers",
                "spec": list(spec),
                "mesh": dict(self.mesh.shape),
                "axis_types": [t.name for t in self.mesh.axis_types],
                "memory_kind": sharding.memory_kind,
                "global_callbacks": self.counts[(label, "global")],
                "shards": shards,
                "layout_limit": (
                    "Callback host arrays prove logical shard bytes, not device buffer tiling; complete HLO retained."
                ),
            }
        return result


def emit(kind, label, value, index, ids, mesh):
    def selected(_):
        jax.debug.callback(
            lambda v, i, d: COLLECTORS[kind].receive(label, False, v, i, d), value, index, ids, partitioned=False
        )

        def local_callback(v, i, d):
            jax.debug.callback(lambda a, b, c: COLLECTORS[kind].receive(label, True, a, b, c), v, i, d, partitioned=True)
            return ()

        spec = P("data", "model", None, None) if label == "X" else P("data", None, None)
        jax.shard_map(
            local_callback, mesh=mesh, in_specs=(spec, P(), P(("data", "model"))), out_specs=(), check_vma=False
        )(value, index, ids)
        return ()

    jax.lax.cond(index == 0, selected, lambda _: (), operand=None)


def make_tap(kind, mesh):
    ids = jax.device_put(np.arange(mesh.size, dtype=np.int32), NamedSharding(mesh, P(("data", "model"))))

    @jax.custom_vjp
    def identity(y, x, index):
        return y

    def forward(y, x, index):
        emit(kind, "X", x, index, ids, mesh)
        return y, (x, index)

    def backward(residual, g):
        _, index = residual
        emit(kind, "G", g, index, ids, mesh)
        return g, None, None

    identity.defvjp(forward, backward)
    return identity


def patch_method(cls, name, replacements, extra_keyword=False, extra_globals=None):
    original = inspect.getsource(getattr(cls, name))
    source = textwrap.dedent(original)
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Exact research patch site absent/ambiguous: {cls.__name__}.{name}: {old}")
        source = source.replace(old, new)
    tree = ast.parse(source)
    function = tree.body[0]
    if extra_keyword:
        function.args.kwonlyargs.append(ast.arg(arg="diagnostic_layer_index"))
        function.args.kw_defaults.append(ast.Constant(None))
    ast.fix_missing_locations(tree)
    namespace = dict(inspect.getmodule(cls).__dict__)
    namespace.update(extra_globals or {})
    exec(compile(tree, str(Path(inspect.getfile(cls)).resolve()), "exec"), namespace)
    setattr(cls, name, namespace[name])
    PATCHES.append(
        {
            "method": f"{cls.__module__}.{cls.__name__}.{name}",
            "original_ast_sha256": hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(original))).encode()).hexdigest(),
            "instrumented_ast_sha256": hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
            "instrumented_source": ast.unparse(tree),
        }
    )


def install(legacy_mesh, native_mesh):

    native_tap = make_tap("native", native_mesh)
    legacy_tap = make_tap("legacy", legacy_mesh)

    def tap_native(y, x, index, heads, head_dim):
        canonical_x = x.reshape(x.shape[0], x.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
        return native_tap(y, canonical_x, index)

    def tap_legacy(y, x, index):
        canonical_x = x.rearrange(("batch", "heads", "position", "head_size")).array
        canonical_y = y.rearrange(("batch", "position", "embed"))
        return hax.named(legacy_tap(canonical_y.array, canonical_x, index.scalar()), canonical_y.axes).rearrange(y.axes)

    patch_method(
        NativeLlamaAttention,
        "__call__",
        [
            (
                "return self.o_proj(output, out_sharding=self.sharding.residual)",
                (
                    "projected = self.o_proj(output, out_sharding=self.sharding.residual)\n "
                    "   return _research_tap(projected, output, diagnostic_layer_index, sel"
                    "f.config.num_heads, dim)"
                ),
            )
        ],
        True,
        {"_research_tap": tap_native},
    )
    patch_method(
        NativeLlamaBlock,
        "__call__",
        [
            (
                "self.self_attn(self.input_layernorm(x), mask, pos_ids)",
                (
                    "self.self_attn(self.input_layernorm(x), mask, pos_ids, diagnostic_laye"
                    "r_index=diagnostic_layer_index)"
                ),
            )
        ],
        True,
    )
    patch_method(
        NativeLlamaModel,
        "activations",
        [
            (
                "self.layers.fold(hidden, mask, pos_ids)",
                (
                    "self.layers.fold(hidden, mask, pos_ids, diagnostic_layer_index=jnp.ara"
                    "nge(self.config.num_layers), in_axes=((None, None), {'diagnostic_layer"
                    "_index': 0}))"
                ),
            )
        ],
    )
    patch_method(
        Attention,
        "__call__",
        [
            (
                "attn_output = self.o_proj(attn_output, key=key_o)",
                (
                    "projection_input = attn_output\n    attn_output = self.o_proj(attn_outp"
                    "ut, key=key_o)\n    attn_output = _research_tap(attn_output, projection"
                    "_input, diagnostic_layer_index)"
                ),
            )
        ],
        True,
        {"_research_tap": tap_legacy},
    )
    patch_method(
        LlamaDecoderLayer,
        "__call__",
        [
            (
                "self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)",
                (
                    "self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids, diagnostic"
                    "_layer_index=diagnostic_layer_index)"
                ),
            )
        ],
        True,
    )
    patch_method(
        LlamaTransformer,
        "__call__",
        [
            (
                "self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)",
                (
                    "self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids, diagnos"
                    "tic_layer_index=hax.arange(self.config.Layers))"
                ),
            )
        ],
    )
