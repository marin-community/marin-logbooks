# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interpret saved core metadata without loading XPlanes or initializing JAX."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def core_message(schema):
    descriptor = descriptor_pb2.FileDescriptorProto(name="core_details.proto", syntax="proto3")
    message = descriptor.message_type.add(name="CoreDetails")
    types = {"string": 9, "uint32": 13, "bool": 8}
    for field in schema["fields"]:
        message.field.add(name=field["name"], number=field["number"], type=types[field["type"]], label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(descriptor)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("CoreDetails"))


def inspect(report, schema):
    message_type = core_message(schema)
    rows = []
    for host in report["hosts"]:
        summary = host["summary"]
        (hostname,) = summary["hostnames"]
        planes = {p["index"]: p for p in summary["planes"]}
        assert len(planes) == len(summary["planes"]), "duplicate plane index"
        for track in summary["tracks"]:
            identity = track["track"]
            if identity["line_name"] != "XLA Ops":
                continue
            plane = planes[identity["plane_index"]]
            assert (plane["id"], plane["name"]) == (identity["plane_id"], identity["plane_name"])
            encoded = plane["stats"]["core_details"]
            assert "\ufffd" not in encoded, "replacement character prevents recovery"
            payload = encoded.encode("utf-8")
            parsed = message_type.FromString(payload)
            parsed.DiscardUnknownFields()
            assert parsed.SerializeToString(deterministic=True) == payload, "not canonical known-schema bytes"
            fields = {field["name"]: getattr(parsed, field["name"]) for field in schema["fields"]}
            rows.append(
                dict(
                    hostname=hostname,
                    plane_index=plane["index"],
                    plane_id=plane["id"],
                    plane_name=plane["name"],
                    line_id=identity["line_id"],
                    summary_payload_hex=payload.hex(),
                    core_details=fields,
                    declared_process_id=plane["stats"].get("process_id"),
                    eigh_scope_union_ps=track["groups"]["eigh_work"]["union_ps"],
                    original_profile_identity=host["profile"],
                )
            )
    rows.sort(key=lambda row: (row["hostname"], row["plane_index"]))
    assert len(rows) == 32
    assert len({row["core_details"]["global_core_id"] for row in rows}) == 32, "duplicate global core"
    chips = defaultdict(list)
    for row in rows:
        chips[row["core_details"]["global_chip_id"]].append(row)
    assert set(chips) == set(range(16))
    chip_rows = []
    for chip, members in sorted(chips.items()):
        assert len(members) == 2 and {m["core_details"]["core_num"] for m in members} == {0, 1}
        assert len({m["hostname"] for m in members}) == 1, "chip crosses hosts"
        assert len({m["core_details"]["local_chip_id"] for m in members}) == 1
        chip_rows.append(
            dict(
                global_chip_id=chip,
                hostname=members[0]["hostname"],
                core_ids=[m["core_details"]["global_core_id"] for m in members],
                eigh_scope_union_ms=[m["eigh_scope_union_ps"] / 1e9 for m in members],
            )
        )
    return dict(
        evidence_kind="schema interpretation of saved decoded JSON; not a new raw XPlane audit",
        core_rows=rows,
        chip_rows=chip_rows,
        limits=[
            "Byte roundtrip refers to UTF-8 re-encoded saved strings, not an independent read of XStat bytes_value.",
            "Pinned upstream schema/consumer explain fields; the runtime producer version is not verified.",
            "Core/chip IDs are schema interpretations of declarations, not recovered JAX mesh coordinates or slot IDs.",
            "Scope unions may include communication and waiting; no arithmetic-only or wall-time fraction claim.",
            "Matching 11 long-duration chips to 11 real matrices is a hypothesis, not a device-to-slot join.",
            "No per-host process_id statistic is treated as JAX process rank.",
            "No numerical intervention, speedup estimate, or hardware-peak change follows from these counts.",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_bytes, schema_bytes = args.report.read_bytes(), args.schema.read_bytes()
    assert hashlib.sha256(report_bytes).hexdigest() == "f4a8da3a88730993b9782b097201353f02c63bc7d896045eecebaed1c0633e67"
    result = inspect(json.loads(report_bytes), json.loads(schema_bytes))
    result["inputs"] = [
        dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for path, data in [(args.report, report_bytes), (args.schema, schema_bytes)]
    ]
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
