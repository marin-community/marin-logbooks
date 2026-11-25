# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
import json
from collections import Counter

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

P = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/encoding-survey-2"
tot = Counter()
stats = [json.loads(p.read_text()) for p in StoragePath(f"{P}/stats/*.json").glob()]
for s in stats:
    tot.update({k: v for k, v in s.items() if isinstance(v, int)})
print(
    len(stats),
    "shards;",
    {k: tot[k] for k in ("records", "long_records", "records_with_signature", "base_pairs_on_signature_records")},
)
for k in sorted(tot):
    if k.startswith("signature/") or k.startswith("new/"):
        print(
            f"  {k:28s} {tot[k]:>10,}  ({100 * tot[k] / tot['records']:.3f}% of records)"
            if k.startswith("signature/")
            else f"  {k:28s} {tot[k]:>10,}"
        )
rows = []
for p in StoragePath(f"{P}/new/*.parquet").glob():
    rows += pq.read_table(io.BytesIO(p.read_bytes())).to_pylist()
print(len(rows), "new-candidate rows")
by = Counter((r["decoding"], r["eval"]) for r in rows)
print(by.most_common(25))
json.dump(rows, open("work/e2_new.json", "w"))
