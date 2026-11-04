# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read typed core and executable declarations from saved XPlanes on CPUs."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from inspect_prefix_core_metadata import core_message
from marin.profiling.xplane import _xspace_message_class

ROOT = Path(__file__).resolve().parent


def hlo_message():
    """Use the published field-number subset; leave unselected fields opaque."""
    schema = json.loads((ROOT / "prefix-assignment-schema.json").read_text())
    file = descriptor_pb2.FileDescriptorProto(name="assignment_subset.proto", syntax="proto3")
    for name, fields in schema["messages"].items():
        message = file.message_type.add(name=name)
        for field in fields:
            target = message.field.add(name=field[0], number=field[1], type=field[2], label=field[3])
            if len(field) == 5:
                target.type_name = field[4]
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("HloProto"))


def identity(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def stat_record(stat, names):
    kind = stat.WhichOneof("value")
    row = dict(name=names[int(stat.metadata_id)], metadata_id=int(stat.metadata_id), value_type=kind)
    if kind is None:
        return row
    value = getattr(stat, kind)
    if kind == "bytes_value":
        row.update(identity(value))
    elif kind == "ref_value":
        row.update(value=int(value), referenced_name=names.get(int(value)))
    elif kind == "str_value" and len(value) > 1000:
        row.update(identity(value.encode()), omitted_text=True)
    else:
        row["value"] = value
    return row


def instruction_record(instruction):
    row = dict(
        id=instruction.id,
        name=instruction.name,
        opcode=instruction.opcode,
        dimensions=list(instruction.shape.dimensions),
        element_type=instruction.shape.element_type,
        tuple_shapes=[list(s.dimensions) for s in instruction.shape.tuple_shapes],
        operand_ids=list(instruction.operand_ids),
        called_computation_ids=list(instruction.called_computation_ids),
        parameter_number=instruction.parameter_number,
        tuple_index=instruction.tuple_index,
        op_name=instruction.metadata.op_name,
        source_file=instruction.metadata.source_file,
        source_line=instruction.metadata.source_line,
        selected_proto_identity=identity(instruction.SerializeToString()),
    )
    if instruction.HasField("literal"):
        row["literal_identity"] = identity(instruction.literal.SerializeToString())
        for field in ["s32s", "s64s", "u32s", "u64s"]:
            values = getattr(instruction.literal, field)
            if values and len(values) <= 64:
                row[field] = list(values)
    return row


def executable_record(payload):
    parsed = hlo_message().FromString(payload)
    assert parsed.HasField("hlo_module"), "HloProto has no module"
    module = parsed.hlo_module
    assignment = module.device_assignment
    groups = [list(c.replica_device_ids) for c in assignment.computation_devices]
    if module.HasField("device_assignment"):
        assert assignment.computation_count == len(groups)
        assert all(len(g) == assignment.replica_count for g in groups)
    selected = []
    for computation in module.computations:
        targets = [
            i
            for i in computation.instructions
            if i.opcode in {"partition-id", "replica-id"}
            or (
                i.opcode == "dynamic-slice"
                and len(i.shape.dimensions) == 3
                and i.shape.dimensions[0] == 1
                and list(i.shape.dimensions)[1:] in [[4096, 4096], [1024, 1024]]
            )
        ]
        if targets:
            selected.append(
                dict(
                    id=computation.id,
                    name=computation.name,
                    root_id=computation.root_id,
                    target_ids=[i.id for i in targets],
                    instructions=[instruction_record(i) for i in computation.instructions],
                )
            )
    selected_ids = {c["id"] for c in selected}
    callers = [
        dict(computation_id=c.id, computation_name=c.name, instruction=instruction_record(i))
        for c in module.computations
        for i in c.instructions
        if selected_ids.intersection(i.called_computation_ids)
    ]
    return dict(
        **identity(payload),
        module_name=module.name,
        module_id=module.id,
        entry_computation_id=module.entry_computation_id,
        entry_computation_name=module.entry_computation_name,
        device_assignment_present=module.HasField("device_assignment"),
        replica_count=assignment.replica_count,
        computation_count=assignment.computation_count,
        computation_devices=groups,
        computation_total=len(module.computations),
        selected_computations=selected,
        direct_callers=callers,
        limits="Selected opcode/shape/source/operand fields only; unselected attributes and outputs remain opaque.",
    )


def inspect(path):
    space = _xspace_message_class()()
    space.ParseFromString(path.read_bytes())
    core_type = core_message(json.loads((ROOT / "prefix-core-schema.json").read_text()))
    result = dict(
        hostnames=list(space.hostnames),
        cores=[],
        executables=[],
        planes=[],
        assignment_events=[],
        warnings=list(space.warnings),
    )
    for index, plane in enumerate(space.planes):
        names = {int(k): str(v.name) for k, v in plane.stat_metadata.items()}
        row = dict(
            index=index,
            id=int(plane.id),
            name=plane.name,
            stat_metadata=[dict(id=k, name=v) for k, v in sorted(names.items())],
            stats=[stat_record(s, names) for s in plane.stats],
            event_metadata_count=len(plane.event_metadata),
        )
        result["planes"].append(row)
        for stat in plane.stats:
            if names[int(stat.metadata_id)] != "core_details":
                continue
            assert stat.WhichOneof("value") == "bytes_value", "core_details is not typed bytes"
            payload = bytes(stat.bytes_value)
            core = core_type.FromString(payload)
            canonical = core_type.FromString(payload)
            canonical.DiscardUnknownFields()
            assert canonical.SerializeToString(deterministic=True) == payload, "unknown/noncanonical core fields"
            result["cores"].append(
                dict(
                    plane_index=index,
                    plane_id=int(plane.id),
                    plane_name=plane.name,
                    value_type="bytes_value",
                    payload_hex=payload.hex(),
                    **identity(payload),
                    fields={f.name: getattr(core, f.name) for f in core.DESCRIPTOR.fields},
                )
            )
        stat_types = Counter()
        for key, meta in plane.event_metadata.items():
            for stat in meta.stats:
                name = names[int(stat.metadata_id)]
                stat_types[(name, str(stat.WhichOneof("value")))] += 1
                if name != "Hlo Proto":
                    continue
                record = dict(
                    plane_index=index,
                    plane_name=plane.name,
                    event_metadata_key=int(key),
                    event_metadata_id=int(meta.id),
                    event_metadata_name=meta.name,
                    associated_stats=[stat_record(s, names) for s in meta.stats],
                )
                assert stat.WhichOneof("value") == "bytes_value", "hlo_proto is not typed bytes"
                record["hlo"] = executable_record(bytes(stat.bytes_value))
                result["executables"].append(record)
        if plane.name.startswith("/host:"):
            for line_index, line in enumerate(plane.lines):
                for event_index, event in enumerate(line.events):
                    event_names = {names[int(s.metadata_id)] for s in event.stats}
                    if "global_device_id" not in event_names:
                        continue
                    meta = plane.event_metadata[int(event.metadata_id)]
                    result["assignment_events"].append(
                        dict(
                            plane_index=index,
                            plane_name=plane.name,
                            line_index=line_index,
                            line_id=int(line.id),
                            event_index=event_index,
                            metadata_id=int(event.metadata_id),
                            event_name=meta.name,
                            stats=[stat_record(s, names) for s in event.stats],
                            metadata_stats=[stat_record(s, names) for s in meta.stats],
                        )
                    )
        row["event_metadata_stat_inventory"] = [
            dict(name=n, value_type=t, count=c) for (n, t), c in sorted(stat_types.items())
        ]
    result["limits"] = [
        "Raw typed core bytes verified against the selected public schema; runtime producer version is not identified.",
        "Assignment IDs are declarations; joining them to JAX mesh/slots requires separate evidence.",
        "Every embedded Hlo Proto metadata record is retained, without selecting modules by ordinal or name alone.",
        "Selected computations contain partition/replica IDs or square dynamic slices; no arithmetic evaluation.",
        "No numerical kernels, hardware measurements, MFU changes, or output semantics are inferred.",
    ]
    assert len(json.dumps(result)) < 8_000_000, "bounded extraction exceeds eight MB; no truncation"
    return result
