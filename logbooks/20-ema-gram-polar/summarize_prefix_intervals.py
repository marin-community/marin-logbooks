# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate regional interval provenance and summarize each recorded device track."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

NAMES = {"all-gather.62", "all-gather.74", "all-gather.88"}


def summarize(report, manifest, prior):
    assert report["completed"] and len(report["hosts"]) == 4
    assert report["manifest"] == manifest
    assert report["manifest_sha256"] == hashlib.sha256((json.dumps(manifest, indent=2) + "\n").encode()).hexdigest()
    assert report["source_commit"] == "d6e320350b453efc34837c2aa7072fef289f9e28"
    output = dict(hosts=[], limits=report["interpretation"])
    for index, host in enumerate(report["hosts"]):
        expected = manifest["profiles"][index]
        assert host["profile"] == expected
        assert host["verified_identity"] == {k: expected[k] for k in ("bytes", "sha256", "md5_hash")}
        assert host["verified_identity"] == prior["hosts"][index]["verified_identity"]
        s = host["summary"]
        assert len(s["tracks"]) == 16 and len(s["selected_collectives"]) == 24
        keys = [(t["track"]["plane_index"], t["track"]["line_index"]) for t in s["tracks"]]
        assert len(set(keys)) == len(keys)
        selected_keys = [
            (e["track"]["plane_index"], e["track"]["line_index"], e["event_index"]) for e in s["selected_collectives"]
        ]
        assert len(set(selected_keys)) == 24
        device_rows = []
        for t in s["tracks"]:
            groups = t["groups"]
            for v in groups.values():
                assert 0 <= v["union_ps"] <= v["sum_ps"]
            device_rows.append(dict(track=t["track"], groups=groups))
        collectives = {}
        for name in sorted(NAMES):
            events = [e for e in s["selected_collectives"] if e["name"] == name]
            assert len(events) == 8
            for e in events:
                assert e["end_ps"] - e["start_ps"] == e["duration_ps"] > 0
                assert e["stats"]["shape_with_layout"] == "f32[16,4096,4096]{2,1,0:T(8,128)}"
                assert "relative_gram_power/optimization_barrier" in e["tf_op"]
            old = [v for v in prior["hosts"][index]["summary"]["hot_ops"] if v["name"] == name]
            assert len(old) == 1 and old[0]["count"] == 8
            assert math.isclose(sum(e["duration_ps"] for e in events) / 1e6, old[0]["total_duration"], abs_tol=1e-6)
            collectives[name] = dict(
                count=8,
                start_spread_ms=(max(e["start_ps"] for e in events) - min(e["start_ps"] for e in events)) / 1e9,
                end_spread_ms=(max(e["end_ps"] for e in events) - min(e["end_ps"] for e in events)) / 1e9,
                min_duration_ms=min(e["duration_ps"] for e in events) / 1e9,
                max_duration_ms=max(e["duration_ps"] for e in events) / 1e9,
            )
        output["hosts"].append(
            dict(
                index=index,
                device_tracks=device_rows,
                collectives=collectives,
                marker_counts=[
                    dict(line=k[0], name=k[1], step=k[2], count=n)
                    for k, n in sorted(
                        Counter((e["track"]["line_name"], e["name"], str(e["step_num"])) for e in s["markers"]).items()
                    )
                ],
                parser_peak_rss_kib=host["parser_peak_rss_kib"],
            )
        )
    output["interpretation"] = [
        "Per-track union duration includes all recorded events bearing that source scope, not isolated arithmetic cost.",
        "Within-host recorded start/end spreads are descriptive; neither cross-device nor "
        "cross-host clock calibration is independently verified.",
        "Converging recorded ends with widely separated starts are consistent with waiting, not a "
        "measurement of transfer-only time.",
        "32 device planes are recorded across four hosts; this does not redefine the verified16 "
        "JAXv4devices or4.4PF MFU denominator.",
        "Decoded stat strings use the existing parser UTF8 replacement for bytes; binary "
        "core_details is not preserved byte-exactly.",
        "Source pads11 matrices to16, but no exact slot-to-device assignment or padding cause is established.",
        "121markers/host include15 names on8 device planes plus one host train event; not121trainingupdates.",
        "No new numerical implementation or performance improvement measured.",
    ]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.prior.read_bytes()
    manifest = json.loads(args.manifest.read_text())
    assert (
        len(raw) == manifest["prior_report"]["bytes"]
        and hashlib.sha256(raw).hexdigest() == manifest["prior_report"]["sha256"]
    )
    result = summarize(json.loads(args.report.read_text()), manifest, json.loads(raw))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
