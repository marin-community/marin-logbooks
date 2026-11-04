# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check profile units and overlapping-track interpretation without model imports."""

import json
import sys
import tempfile
from pathlib import Path

from marin.profiling.trace_summary import BreakdownMode
from marin.profiling.xplane import _xspace_message_class, summarize_xplane_timeline
from profile_prefix_regionally import compact_summary, digest


def fixture(path, second_device):
    xspace = _xspace_message_class()()
    xspace.hostnames.append("fixture-host")
    for device in range(2 if second_device else 1):
        plane = xspace.planes.add()
        plane.id = device + 1
        plane.name = f"/device:TPU:{device}"
        for index, name in enumerate(["100", "dot.1", "all-reduce.1"], 1):
            plane.event_metadata[index].id = index
            plane.event_metadata[index].name = name
        for line_id, name, events in [
            (1, "Steps", [(1, 0, 100)]),
            (2, "XLA Ops", [(2, 0, 60), (3, 60, 20)]),
        ]:
            line = plane.lines.add()
            line.id = line_id
            line.name = name
            for metadata, start, duration in events:
                event = line.events.add()
                event.metadata_id = metadata
                event.offset_ps = start * 1_000_000
                event.duration_ps = duration * 1_000_000
    path.write_bytes(xspace.SerializeToString())


def main():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "fixture.xplane.pb"
        for second in (False, True):
            fixture(path, second)
            result = summarize_xplane_timeline(
                path, warmup_steps=0, hot_op_limit=40, breakdown_mode=BreakdownMode.EXCLUSIVE_PER_TRACK
            ).to_dict()
            count = 2 if second else 1
            assert result["trace_provenance"]["trace_sha256"] == digest(path)["sha256"]
            assert result["step_time"]["all_steps"]["mean"] == 100
            assert result["time_breakdown"]["compute"]["total_duration"] == 60 * count
            assert result["time_breakdown"]["communication"]["total_duration"] == 20 * count
            assert result["trace_overview"]["profile_end_ts"] == 100
            assert result["communication_ops"][0]["total_duration"] == 20 * count
            # Two simultaneous devices double summed device work, never the 100-us trace window.
            compact = compact_summary(result)
            assert compact["trace_overview"] == result["trace_overview"]
            assert compact["time_breakdown"] == result["time_breakdown"]
        rows = [dict(name=f"op{i}", duration=i) for i in range(55)]
        compact = compact_summary(dict(hot_ops=rows, trace_overview=dict(quality_warnings=["fixture-warning"])))
        assert compact["hot_ops"] == rows[:40] and compact["hot_ops_full_row_count"] == 55
        assert compact["trace_overview"]["quality_warnings"] == ["fixture-warning"]
    assert not {"jax", "wandb", "tensorflow"}.intersection(sys.modules)
    print(
        json.dumps(
            dict(
                passed=True,
                fixtures=[
                    "one-device units",
                    "overlapping-device sum versus window",
                    "bounded rows retain total and quality warnings",
                ],
                numerical_frameworks_imported=False,
            )
        )
    )


if __name__ == "__main__":
    main()
