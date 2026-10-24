"""Disposable Q-only articulated compute view; stored parameters remain 2-D.

Apply by runpy.run_path before importing/running the benchmark entry point.
All other NativeLlamaAttention code comes directly from the current checkout.
"""

import inspect
import textwrap

import levanter.models.llama_native as native


original = textwrap.dedent(inspect.getsource(native.NativeLlamaAttention.__call__))
old_query = """    q = self.q_proj(x, out_sharding=self.sharding.intermediate).reshape(
        batch, sequence, self.config.num_heads, dim
    )"""
new_query = """    groups = self.config.num_heads // self.config.num_kv_heads
    weight = reshard(self.q_proj.weight, self.q_proj.compute_spec).reshape(
        self.config.num_kv_heads, groups, dim, self.config.hidden_dim
    )
    q = jax.lax.dot_general(
        x, weight, dimension_numbers=(((2,), (3,)), ((), ())),
        out_sharding=P(self.sharding.batch, None, self.sharding.model_axis, None, None),
    )
    if self.q_proj.bias is not None:
        q = q + reshard(self.q_proj.bias, P(self.q_proj.compute_spec[0])).reshape(
            self.config.num_kv_heads, groups, dim
        )
    q = q.reshape(batch, sequence, self.config.num_heads, dim)"""
if original.count(old_query) != 1:
    raise ValueError("NativeLlamaAttention Q source changed; review the scratch patch before applying")
patched = original.replace(old_query, new_query)
namespace = dict(vars(native))
exec(compile(patched, __file__ + ":generated", "exec"), namespace)
native.NativeLlamaAttention.__call__ = namespace["__call__"]
