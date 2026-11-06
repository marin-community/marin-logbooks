"""Read-only: router_bias norm per checkpoint. usage: router_bias.py name=checkpoint ..."""
import asyncio, sys, json
import numpy as np, tensorstore as ts
from levanter.tensorstore_serialization import _create_ocdbt_spec
from rigging.filesystem.s3_compat import configure_coreweave_s3
configure_coreweave_s3()
LEAF = "params/stacked_blocks/stacked/mlp/router_bias"
async def read(root):
    return np.asarray(await (await ts.open(_create_ocdbt_spec(root, LEAF), open=True, read=True)).read()).astype(np.float64)
items = [a.split("=", 1) for a in sys.argv[1:]]
async def main(): return await asyncio.gather(*(read(p) for _, p in items))
arrs = asyncio.run(main()); base = arrs[0]; rows = {}
for (n, _), a in zip(items, arrs):
    d = a - base; ln = np.linalg.norm(a, axis=1)
    rows[n] = {"norm": float(np.linalg.norm(a)), "ratio": float(np.linalg.norm(a) / np.linalg.norm(base)), "dist_from_first": float(np.linalg.norm(d)), "layer_norm_min": float(ln.min()), "layer_norm_med": float(np.median(ln)), "layer_norm_max": float(ln.max()), "mean": float(a.mean()), "max_abs": float(np.abs(a).max()), "layers_changed": int((np.abs(d).max(axis=1) > 0).sum())}
    print(n, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[n].items()}))
print("shape", base.shape); json.dump(rows, open(sys.argv[0].replace("router_bias.py", "router_bias.json"), "w"), indent=1)
