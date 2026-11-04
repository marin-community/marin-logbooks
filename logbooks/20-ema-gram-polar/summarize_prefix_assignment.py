# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join bounded saved executable, host assignment and core records without kernels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def unique(rows, key):
    result = {key(row): row for row in rows}
    assert len(result) == len(rows), "duplicate identity"
    return result


def stat_values(rows):
    indexed = unique(rows, lambda row: row["name"])
    return {name: row.get("value", row.get("referenced_name")) for name, row in indexed.items()}


def summarize(report, manifest, prior, intervals):
    assert report["completed"] and report["analysis"] == "assignment"
    assert report["source_commit"] == "2d9c2672cc42d5cbef72066fd32f3f0455356d07"
    assert report["manifest_sha256"] == "d3dcae531dce60abfbd65daae1dd45b0a62f816b91ef53558688eb53eb52b702"
    assert report["manifest"] == manifest
    assert len(report["hosts"]) == len(manifest["profiles"]) == 4
    expected = unique(prior["core_rows"], lambda r: (r["hostname"], r["plane_index"]))
    interval_hosts = unique(intervals["hosts"], lambda h: tuple(h["summary"]["hostnames"]))
    profiles = unique(manifest["profiles"], lambda p: p["uri"])
    module_sets, host_rows, joined_cores, raw_module_sets = [], [], [], []
    slot_routes, assignment, train_identity = None, None, None
    for host in sorted(report["hosts"], key=lambda h: h["profile"]["uri"]):
        profile, summary = host["profile"], host["summary"]
        assert profile == profiles[profile["uri"]]
        assert host["verified_identity"]["sha256"] == profile["sha256"]
        assert host["verified_identity"]["bytes"] == profile["bytes"]
        (hostname,) = summary["hostnames"]
        previous = interval_hosts[(hostname,)]
        assert previous["profile"] == profile
        program_ids = {c["stats"]["program_id"] for c in previous["summary"]["selected_collectives"]}
        assert len(program_ids) == 1
        (program_id,) = program_ids
        modules = unique(summary["executables"], lambda m: stat_values(m["associated_stats"])["program_id"])
        count = sum(
            r["count"]
            for p in summary["planes"]
            for r in p["event_metadata_stat_inventory"]
            if r["name"] == "Hlo Proto" and r["value_type"] == "bytes_value"
        )
        assert len(modules) == count and count > 0, "incomplete executable extraction"
        module_sets.append(sorted((pid, m["hlo"]["module_name"], m["hlo"]["module_id"]) for pid, m in modules.items()))
        raw_module_sets.append(sorted((pid, m["hlo"]["sha256"]) for pid, m in modules.items()))
        train = modules[program_id]["hlo"]
        assert train["device_assignment_present"] and train["replica_count"] == 1
        devices = train["computation_devices"]
        assert train["computation_count"] == len(devices) == 16
        assert all(len(d) == 1 for d in devices) and len({d[0] for d in devices}) == 16
        if assignment is None:
            assignment = devices
            train_identity = {k: train[k] for k in ["module_name", "module_id", "sha256", "bytes"]}
        assert assignment == devices and train_identity["module_id"] == train["module_id"]
        comps = unique(train["selected_computations"], lambda c: c["id"])
        entry = comps[train["entry_computation_id"]]
        instructions = unique(entry["instructions"], lambda i: i["id"])
        routes = []
        for comp in comps.values():
            body = unique(comp["instructions"], lambda i: i["id"])
            for target in comp["target_ids"]:
                leaf = body[target]
                if leaf["opcode"] != "dynamic-slice":
                    continue
                assert leaf["dimensions"][0] == 1 and leaf["dimensions"][1:] in [[1024, 1024], [4096, 4096]]
                pad, start, zero_a, zero_b = [body[k] for k in leaf["operand_ids"]]
                assert pad["opcode"] == "pad" and pad["dimensions"] == [16, *leaf["dimensions"][1:]]
                original = body[pad["operand_ids"][0]]
                assert original["opcode"] == "parameter" and original["dimensions"] == [11, *leaf["dimensions"][1:]]
                assert start["opcode"] == "parameter"
                assert zero_a["id"] == zero_b["id"] and zero_a["s32s"] == [0]
                callers = [
                    c for c in train["direct_callers"] if comp["id"] in c["instruction"]["called_computation_ids"]
                ]
                assert len(callers) == 1 and callers[0]["computation_id"] == entry["id"]
                caller = callers[0]["instruction"]
                assert caller == instructions[caller["id"]] and caller["opcode"] == "fusion"
                index = instructions[caller["operand_ids"][start["parameter_number"]]]
                assert index["opcode"] == "convert" and index["element_type"] == 4
                partition = instructions[index["operand_ids"][0]]
                assert partition["opcode"] == "partition-id" and partition["element_type"] == 8
                routes.append(
                    dict(
                        computation_id=comp["id"],
                        slice_id=leaf["id"],
                        dimensions=leaf["dimensions"],
                        caller_id=caller["id"],
                        index_id=index["id"],
                        partition_id=partition["id"],
                        pad_id=pad["id"],
                        source_shape=original["dimensions"],
                        source_scope=leaf["op_name"],
                        padded_leading_index="partition",
                        other_axis_starts=[0, 0],
                    )
                )
        routes.sort(key=lambda r: r["computation_id"])
        assert Counter(tuple(r["dimensions"]) for r in routes) == {(1, 4096, 4096): 3, (1, 1024, 1024): 11}
        if slot_routes is None:
            slot_routes = routes
        assert slot_routes == routes
        # Host records carry the module name, partition and device. Their lack of
        # program_id is preserved; uniqueness within this capture is checked.
        train_names = [m for m in modules.values() if m["hlo"]["module_name"] == train["module_name"]]
        assert len(train_names) == 1
        events = []
        for event in summary["assignment_events"]:
            names = [s.get("referenced_name") for s in event["stats"] if s["name"] == "name"]
            if names != [train["module_name"]]:
                continue
            values = stat_values(event["stats"])
            partition, device = values["partition"], values["global_device_id"]
            assert values["replica"] == 0 and devices[partition] == [device]
            events.append(dict(partition=partition, global_device_id=device, event=event))
        assert len(events) == 4 and len({e["partition"] for e in events}) == 4
        cores = unique(summary["cores"], lambda c: c["plane_index"])
        assert len(cores) == 8
        for plane, core in cores.items():
            old = expected[(hostname, plane)]
            assert core["value_type"] == "bytes_value" and core["payload_hex"] == old["summary_payload_hex"]
            assert core["fields"] == old["core_details"]
            assert (core["plane_id"], core["plane_name"]) == (old["plane_id"], old["plane_name"])
            chip = core["fields"]["global_chip_id"]
            matches = [e for e in events if e["global_device_id"] == chip]
            assert len(matches) == 1
            joined_cores.append(
                dict(
                    hostname=hostname,
                    plane_index=plane,
                    global_chip_id=chip,
                    global_core_id=core["fields"]["global_core_id"],
                    partition=matches[0]["partition"],
                )
            )
        host_rows.append(
            dict(
                hostname=hostname,
                module_count=count,
                program_id=program_id,
                assignment_event_count=len(summary["assignment_events"]),
                module_identities=[
                    dict(program_id=pid, **{k: m["hlo"][k] for k in ["module_name", "module_id", "bytes", "sha256"]})
                    for pid, m in sorted(modules.items())
                ],
                train_events=sorted(events, key=lambda e: e["partition"]),
            )
        )
    assert all(m == module_sets[0] for m in module_sets)
    assert len(joined_cores) == 32 and len({c["global_core_id"] for c in joined_cores}) == 32
    assert {e["partition"] for h in host_rows for e in h["train_events"]} == set(range(16))
    return dict(
        module_count=sum(h["module_count"] for h in host_rows),
        same_nine_program_name_id_sets_across_hosts=True,
        raw_hlo_bytes_equal_across_hosts=all(m == raw_module_sets[0] for m in raw_module_sets),
        train_module_host0=train_identity,
        assignment=assignment,
        hosts=host_rows,
        slice_routes=slot_routes,
        core_numeric_id_joins=sorted(joined_cores, key=lambda c: c["global_core_id"]),
        raw_core_bytes_verified=True,
        source_commit=report["source_commit"],
        limits=[
            "Raw HloProto hashes differ across hosts. Only the compared routes and assignments agree.",
            "Host execute events have module names but no program_id; unique name and assignment match are checked.",
            "All-gathers join HloProto by program_id; core chip and host device IDs match numerically per host.",
            "Leading index p selects a padded slot by partition. Source intends five trailing padding slots.",
            "Pad configuration and padding float literals were not decoded in this bounded extraction.",
            "Runtime JAX mesh coordinate table and producer version remain unverified.",
            "No runtime matrix values, arithmetic-only time, padding cause, speedup or MFU change is inferred.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "d50ab825d8b07b70ef5197116a8ffab5c6c4cb06ce80d45130882a73b6bdd308"
    manifest_raw = (root / "prefix-assignment-manifest.json").read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == "d3dcae531dce60abfbd65daae1dd45b0a62f816b91ef53558688eb53eb52b702"
    interval_raw = Path("scratch/20261104-prefix-intervals-final.json").read_bytes()
    assert hashlib.sha256(interval_raw).hexdigest() == "f4a8da3a88730993b9782b097201353f02c63bc7d896045eecebaed1c0633e67"
    result = summarize(
        json.loads(raw),
        json.loads(manifest_raw),
        json.loads((root / "prefix-core-metadata-summary.json").read_text()),
        json.loads(interval_raw),
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
