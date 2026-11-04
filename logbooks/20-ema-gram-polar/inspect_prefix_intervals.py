# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve device-track intervals and collective context from saved XPlanes."""

from collections import Counter

from marin.profiling.trace_summary import is_device_op_thread
from marin.profiling.xplane import (
    _xplane_event_args,
    _xplane_event_metadata,
    _xplane_stats_to_mapping,
    _xspace_message_class,
)

TARGETS = {"all-gather.62", "all-gather.74", "all-gather.88"}


class IntervalUnion:
    """Exact integer-picosecond union for intervals ordered by start time."""

    def __init__(self):
        self.count = self.sum_ps = self.union_ps = 0
        self.start_ps = self.end_ps = self.previous_start = None

    def add(self, start, duration):
        assert duration > 0
        assert self.previous_start is None or start >= self.previous_start, "unordered intervals"
        self.previous_start = start
        end = start + duration
        self.count += 1
        self.sum_ps += duration
        if self.start_ps is None:
            self.start_ps, self.end_ps = start, end
            self.union_ps = duration
        else:
            self.union_ps += max(0, end - max(start, self.end_ps))
            self.end_ps = max(self.end_ps, end)

    def result(self):
        return dict(
            count=self.count,
            sum_ps=self.sum_ps,
            union_ps=self.union_ps,
            first_start_ps=self.start_ps,
            last_end_ps=self.end_ps,
        )


def inspect(path):
    space = _xspace_message_class()()
    space.ParseFromString(path.read_bytes())
    result = dict(hostnames=list(space.hostnames), planes=[], tracks=[], selected_collectives=[], markers=[])
    for plane_index, plane in enumerate(space.planes):
        names = {int(k): str(v.name) for k, v in plane.stat_metadata.items()}
        metadata = _xplane_event_metadata(plane, stat_names=names, process_name=plane.name)
        result["planes"].append(
            dict(
                index=plane_index,
                id=int(plane.id),
                name=plane.name,
                stats=_xplane_stats_to_mapping(plane.stats, stat_names=names),
            )
        )
        for line_index, line in enumerate(plane.lines):
            line_name = line.display_name or line.name
            device_ops = plane.name.startswith("/device:") and is_device_op_thread(line_name)
            markers = line_name == "Steps" or plane.name.startswith("/host:")
            if not device_ops and not markers:
                continue
            track = dict(
                plane_index=plane_index,
                plane_id=int(plane.id),
                plane_name=plane.name,
                line_index=line_index,
                line_id=int(line.id),
                line_name=line_name,
                line_timestamp_ns=int(line.timestamp_ns),
            )
            groups = {k: IntervalUnion() for k in ("all_device_ops", "named_train", "relative_power", "eigh_work")}
            selected_counts = Counter()
            previous = pending = None
            for event_index, event in enumerate(line.events):
                if event.WhichOneof("data") != "offset_ps" or event.duration_ps <= 0:
                    continue
                meta = metadata.get(int(event.metadata_id))
                if meta is None:
                    continue
                args = _xplane_event_args(event, meta, stat_names=names) if event.stats else meta.args
                start = int(line.timestamp_ns) * 1000 + int(event.offset_ps)
                duration = int(event.duration_ps)
                source_path = args.tf_op or ""
                is_marker = line_name == "Steps" or (plane.name.startswith("/host:") and meta.name == "train")
                if not device_ops and not is_marker:
                    continue
                row = dict(
                    event_index=event_index,
                    metadata_id=int(event.metadata_id),
                    name=meta.name,
                    start_ps=start,
                    duration_ps=duration,
                    end_ps=start + duration,
                    tf_op=args.tf_op,
                    source=args.source,
                    run_id=args.run_id,
                    step_num=args.step_num,
                )
                if is_marker:
                    result["markers"].append(dict(track=track, **row))
                if not device_ops:
                    continue
                groups["all_device_ops"].add(start, duration)
                for key, token in [
                    ("named_train", "_train_step"),
                    ("relative_power", "relative_gram_power"),
                    ("eigh_work", "_eigh_work"),
                ]:
                    if token in source_path:
                        groups[key].add(start, duration)
                if pending is not None:
                    pending["next_record_same_track"] = row
                    pending = None
                if meta.name in TARGETS and "relative_gram_power" in source_path:
                    stats = dict(meta.stats)
                    stats.update(_xplane_stats_to_mapping(event.stats, stat_names=names))
                    selected = dict(
                        track=track,
                        **row,
                        metadata_name=meta.metadata_name,
                        stats=stats,
                        previous_record_same_track=previous,
                        next_record_same_track=None,
                    )
                    result["selected_collectives"].append(selected)
                    selected_counts[meta.name] += 1
                    pending = selected
                previous = row
            if device_ops:
                result["tracks"].append(
                    dict(
                        track=track,
                        groups={k: v.result() for k, v in groups.items()},
                        selected_counts=dict(selected_counts),
                    )
                )
    result["limits"] = [
        "Integer ps timestamps preserve the recorded clock; cross-host clock alignment is NOT verified.",
        "Intervals are recorded durations, not isolated transfer or arithmetic costs; waiting may be included.",
        "Union removes overlap only within one track/category; "
        "do not sum tracks, devices or nested categories as wall time.",
        "Adjacent records are timeline context, not a proven dependency or causal predecessor.",
        "Repeated collective names/records are retained, not deduplicated or assumed independent transfers.",
        "Plane identifiers and raw stats are declarations; matrix-slot ownership is not inferred from ordinals.",
        "Step markers are retained by track and event; their count is not a training-update count.",
        "No absence of recorded events proves an idle device or complete capture.",
    ]
    return result
