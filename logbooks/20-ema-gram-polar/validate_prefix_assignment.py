# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static wire fixtures for typed core metadata and executable assignments."""

import copy
import json
import sys
import tempfile
from pathlib import Path

from inspect_prefix_assignment import executable_record, hlo_message, inspect
from marin.profiling.xplane import _xspace_message_class


def main():
    # Literal protobuf wire fixture: module name x, id9, 1 replica, 2 computations, physical IDs7/4.
    assignment = bytes.fromhex("080110021a030a01071a030a0104")
    module = b"\x0a\x01x\x28\x09\x7a" + bytes([len(assignment)]) + assignment
    wire = b"\x0a" + bytes([len(module)]) + module
    known = executable_record(wire)
    assert known["module_name"] == "x" and known["module_id"] == 9
    assert known["computation_devices"] == [[7], [4]] and known["replica_count"] == 1
    parsed = hlo_message().FromString(wire)
    comp = parsed.hlo_module.computations.add(id=11, name="entry", root_id=3)
    comp.instructions.add(id=1, name="partition", opcode="partition-id")
    target = comp.instructions.add(id=3, name="slice", opcode="dynamic-slice", operand_ids=[2, 1, 1, 1])
    target.shape.dimensions.extend([1, 4096, 4096])
    comp.instructions.add(id=2, name="arg", opcode="parameter")
    assert executable_record(parsed.SerializeToString())["selected_computations"][0]["target_ids"] == [1, 3]
    no_assignment = copy.deepcopy(parsed)
    no_assignment.hlo_module.ClearField("device_assignment")
    assert not executable_record(no_assignment.SerializeToString())["device_assignment_present"]
    space = _xspace_message_class()()
    space.hostnames.append("fixture-host")
    plane = space.planes.add(id=8, name="/device:TPU:0")
    plane.stat_metadata[1].name = "core_details"
    plane.stats.add(metadata_id=1, bytes_value=bytes.fromhex("28083010"))
    meta_plane = space.planes.add(id=9, name="/host:metadata")
    meta_plane.stat_metadata[3].name = "Hlo Proto"
    meta_plane.stat_metadata[4].name = "program_id"
    meta_plane.event_metadata[12].name = "fixture"
    meta_plane.event_metadata[12].stats.add(metadata_id=3, bytes_value=parsed.SerializeToString())
    meta_plane.event_metadata[12].stats.add(metadata_id=4, uint64_value=99)
    cpu = space.planes.add(id=10, name="/host:CPU")
    for number, name in [(1, "global_device_id"), (2, "partition"), (3, "replica")]:
        cpu.stat_metadata[number].name = name
    cpu.event_metadata[1].name = "assignment"
    line = cpu.lines.add(id=3, name="runtime")
    event = line.events.add(metadata_id=1)
    for number, value in [(1, 7), (2, 0), (3, 0)]:
        event.stats.add(metadata_id=number, int64_value=value)
    rejected = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.pb"
        path.write_bytes(space.SerializeToString())
        report = inspect(path)
        assert len(report["executables"]) == 1
        assert len(report["assignment_events"]) == 1
        assert report["assignment_events"][0]["stats"][0]["value"] == 7
        assert report["cores"][0]["fields"]["global_chip_id"] == 8
        assert report["cores"][0]["fields"]["global_core_id"] == 16
        assert report["cores"][0]["payload_hex"] == "28083010"
        assert report["executables"][0]["hlo"]["computation_devices"] == [[7], [4]]
        assert report["executables"][0]["associated_stats"][1]["value"] == 99
        for label in ["core_string_type", "hlo_string_type", "unknown_core_field", "incomplete_assignment"]:
            bad = copy.deepcopy(space)
            if label == "core_string_type":
                bad.planes[0].stats[0].str_value = "(\x080\x10"
            elif label == "hlo_string_type":
                bad.planes[1].event_metadata[12].stats[0].str_value = "not bytes"
            elif label == "unknown_core_field":
                bad.planes[0].stats[0].bytes_value += b"\x40\x01"
            else:
                invalid = copy.deepcopy(parsed)
                invalid.hlo_module.device_assignment.replica_count = 2
                bad.planes[1].event_metadata[12].stats[0].bytes_value = invalid.SerializeToString()
            path.write_bytes(bad.SerializeToString())
            try:
                inspect(path)
            except AssertionError:
                rejected.append(label)
            else:
                raise AssertionError(label)
    assert not {"jax", "wandb", "tensorflow"}.intersection(sys.modules)
    print(
        json.dumps(
            dict(
                passed=True,
                known_wire_assignment=[[7], [4]],
                typed_core_roundtrip=True,
                partition_and_square_slice_selection=True,
                absent_assignment_preserved=True,
                rejected_corruptions=rejected,
                no_numerical_imports=True,
            )
        )
    )


if __name__ == "__main__":
    main()
