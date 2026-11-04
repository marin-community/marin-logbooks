# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static protobuf fixtures for exact intervals, context and attribution limits."""

import json
import sys
import tempfile
from pathlib import Path

from inspect_prefix_intervals import IntervalUnion, inspect
from marin.profiling.xplane import _xspace_message_class


def main():
    union = IntervalUnion()
    for start, duration in [(0, 10), (1, 2), (9, 4), (13, 1), (20, 2)]:
        union.add(start, duration)
    assert union.result() == dict(count=5, sum_ps=19, union_ps=16, first_start_ps=0, last_end_ps=22)
    try:
        union.add(0, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError("unordered intervals accepted")
    space = _xspace_message_class()()
    epoch_ns = 1793750000000000001
    for device in range(2):
        plane = space.planes.add()
        plane.id = 40 + device
        plane.name = f"/device:TPU:{device}"
        plane.stat_metadata[1].name = "tf_op"
        for index, (name, scope) in enumerate(
            [
                ("dot.1", "jit(_train_step)/relative_gram_power/_eigh_work"),
                ("all-gather.62", "jit(_train_step)/relative_gram_power/optimization_barrier"),
                ("all-gather.62", "unrelated_scope"),
                ("100", ""),
            ],
            1,
        ):
            meta = plane.event_metadata[index]
            meta.name = name
            stat = meta.stats.add()
            stat.metadata_id = 1
            stat.str_value = scope
        for line_id, line_name, events in [
            (2, "XLA Ops", [(1, 0, 100), (2, 100, 20), (2, 120, 30), (3, 150, 10)]),
            (9, "Steps", [(4, 0, 160)]),
        ]:
            line = plane.lines.add()
            line.id, line.name, line.timestamp_ns = line_id, line_name, epoch_ns
            for mid, offset, duration in events:
                event = line.events.add()
                event.metadata_id, event.offset_ps, event.duration_ps = mid, offset, duration
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.pb"
        path.write_bytes(space.SerializeToString())
        report = inspect(path)
    assert len(report["tracks"]) == 2 and len(report["markers"]) == 2
    assert len(report["selected_collectives"]) == 4
    for track in report["tracks"]:
        assert track["groups"]["relative_power"]["union_ps"] == 150
        assert track["groups"]["eigh_work"]["union_ps"] == 100
        assert track["groups"]["all_device_ops"]["union_ps"] == 160
    first, second = report["selected_collectives"][:2]
    assert first["start_ps"] == epoch_ns * 1000 + 100
    assert first["duration_ps"] == 20 and second["duration_ps"] == 30
    assert first["next_record_same_track"]["event_index"] == second["event_index"]
    assert second["previous_record_same_track"]["event_index"] == first["event_index"]
    assert first["track"]["plane_id"] == 40 and report["selected_collectives"][2]["track"]["plane_id"] == 41
    assert report["markers"][0]["name"] == "100"
    assert not {"jax", "wandb", "tensorflow"}.intersection(sys.modules)
    print(
        json.dumps(
            dict(
                passed=True,
                fixtures=[
                    "integer ps at epoch-scale timestamp",
                    "overlap/nesting/touch/gap union",
                    "unsorted rejection",
                    "repeated collective records",
                    "source-qualified selection",
                    "adjacent event context",
                    "independent device tracks",
                    "step marker preservation",
                ],
                numerical_frameworks_imported=False,
            )
        )
    )


if __name__ == "__main__":
    main()
