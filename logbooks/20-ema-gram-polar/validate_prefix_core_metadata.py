# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check saved core metadata joins against pinned schema and corruptions."""

import copy
import hashlib
import json
import re
from pathlib import Path

from inspect_prefix_core_metadata import inspect

p = Path("logbooks/20-ema-gram-polar")
report = json.loads(Path("scratch/20261104-prefix-intervals-final.json").read_text())
schema = json.loads((p / "prefix-core-schema.json").read_text())
proto = Path("scratch/20261104-xprof-op-stats.proto").read_bytes()
assert hashlib.sha256(proto).hexdigest() == schema["schema_sha256"]
fields = re.findall(
    r"(string|uint32|bool) (\w+) = (\d+);", proto.decode().split("message CoreDetails {")[1].split("}")[0]
)
assert schema["fields"] == [dict(name=n, number=int(i), type=t) for t, n, i in fields]
base = inspect(report, schema)
assert [(r["global_chip_id"], r["hostname"].rsplit("-", 1)[1]) for r in base["chip_rows"]] == [
    (i, str([2, 3, 0, 1][i // 4])) for i in range(16)
]
assert all(all(v > 2900 for v in c["eigh_scope_union_ms"]) for c in base["chip_rows"][:11])
assert all(all(2.8 < v < 3 for v in c["eigh_scope_union_ms"]) for c in base["chip_rows"][11:])
reordered = copy.deepcopy(report)
reordered["hosts"].reverse()
for host in reordered["hosts"]:
    host["summary"]["planes"].reverse()
    host["summary"]["tracks"].reverse()
assert inspect(reordered, schema) == base
rejections = []
for kind in ["damaged_utf8", "unknown_wire_field", "duplicate_core", "missing_track", "wrong_plane_context"]:
    r = copy.deepcopy(report)
    h = r["hosts"][0]["summary"]
    plane = h["planes"][0]
    if kind == "damaged_utf8":
        plane["stats"]["core_details"] += "\ufffd"
    if kind == "unknown_wire_field":
        plane["stats"]["core_details"] += "\x40\x01"
    if kind == "duplicate_core":
        h["planes"][1]["stats"]["core_details"] = plane["stats"]["core_details"]
    if kind == "missing_track":
        h["tracks"] = [
            x for x in h["tracks"] if not (x["track"]["plane_index"] == 0 and x["track"]["line_name"] == "XLA Ops")
        ]
    if kind == "wrong_plane_context":
        plane["id"] += 100
    try:
        inspect(r, schema)
    except AssertionError:
        rejections.append(kind)
    else:
        raise AssertionError(kind)
result = dict(
    passed=True,
    kind="Local static JSON/protobuf parsing only; no numerical kernels or online tracking",
    independent_schema_fields_match=True,
    known_chip_host_groups=True,
    eleven_long_five_short_scope_groups=True,
    reordering_invariant=True,
    rejected_corruptions=rejections,
    raw_xplane_byte_audit=False,
    runtime_mesh_slot_mapping=False,
    script_sha256=hashlib.sha256((p / "inspect_prefix_core_metadata.py").read_bytes()).hexdigest(),
)
(p / "prefix-core-metadata-validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(result)
