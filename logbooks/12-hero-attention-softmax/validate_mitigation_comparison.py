# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local-only validator for downloaded Hero mitigation JSON; performs no I/O to services.

Run with --root, --request, --source-revision and --output. Optional --proof-root
validates a local mirror of native-object-consensus metadata. The output remains
private: it contains provenance hashes and numerical evidence, not raw operands.
Iris terminal success must be checked separately by the monitoring owner.
"""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

VARIANTS = (
    "baseline",
    "fp32_saved_output",
    "fp32_saved_output_shadow",
    "postcast_output_control",
    "native_instrumented",
    "native_gproj_hybrid",
)
GRADIENTS = ("dq", "dk", "dv", "output")
CONTROLS = ("actual_stored_output", "bf16_rounded_reference_output")
ADDITIVE = (
    "component_count",
    "actual_squared_norm",
    "reference_squared_norm",
    "error_squared_norm",
    "absolute_error_sum",
    "reference_nonzero_components",
    "zero_actual_nonzero_reference_components",
    "reference_nonzero_vectors",
    "zero_actual_nonzero_reference_vectors",
)
INTEGER_STATS = {name for name in ADDITIVE if name.endswith(("count", "components", "vectors"))}
ZERO_GATES = (
    "capture_tail_max_abs_error",
    "baseline_forward_max_abs_error",
    "decoder_q_final_cast_mismatch_count",
    "decoder_k_final_cast_mismatch_count",
    "decoder_q_scale_one_mismatch_count",
    "decoder_k_scale_one_mismatch_count",
    "export_invalid_probability_row_count",
    "export_nonfinite_score_count",
    "export_masked_nonzero_count",
    "export_q_contraction_rounding_bound_excess",
    "export_k_contraction_rounding_bound_excess",
)
OPTIONAL_GATES = (
    "fp32_output_cast_mismatch_count",
    "fp32_output_lse_max_abs_error",
    "instrumented_q_max_abs_difference",
    "instrumented_k_max_abs_difference",
    "instrumented_v_max_abs_difference",
    "export_q_contraction_max_abs_difference",
    "export_k_contraction_max_abs_difference",
)
EVENTS = {
    "worst_dq_events": lambda row: row["dq_error_l2"],
    "worst_cancellation_events": lambda row: abs(row["reference_winner_score_gradient"]),
    "largest_margin_events": lambda row: row["margin"],
}
REFERENCE_ROWS = (
    "masked_count",
    "single_key_count",
    "multi_key_count",
    "margin_above_fp32_cliff_count",
    "margin_histogram",
    "margin_histogram_edges",
    "entropy_sum",
    "tail_mass_sum",
)
CHECKSUMS = {}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    CHECKSUMS[str(path)] = hashlib.sha256(raw).hexdigest()

    def bad_constant(value):
        raise ValueError(f"Nonfinite JSON constant in {path.name}: {value}")

    data = json.loads(raw, parse_constant=bad_constant)
    finite(data, path.name)
    return data


def finite(value, where):
    if isinstance(value, float):
        check(math.isfinite(value), f"{where}: nonfinite numeric value")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{where}[{index}]")


def integer(value, where):
    check(type(value) is int and value >= 0, f"{where}: expected nonnegative integer")


def close(actual, expected, where):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), f"{where}: keys differ")
        for key in expected:
            close(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), f"{where}: list shape differs")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            close(left, right, f"{where}[{index}]")
    elif type(expected) is float:
        check(
            type(actual) in (float, int) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-300),
            f"{where}: arithmetic differs",
        )
    else:
        check(actual == expected, f"{where}: value differs")


def ratio(error, reference):
    return math.sqrt(error / reference) if reference else None


def stats(row, where, components=None):
    for name in ADDITIVE:
        check(name in row and row[name] >= 0, f"{where}.{name}: missing or negative")
        if name in INTEGER_STATS:
            integer(row[name], f"{where}.{name}")
    count = row["component_count"]
    check(count > 0 and count % 128 == 0, f"{where}: wrong component geometry")
    if components is not None:
        check(count == components, f"{where}: wrong component count")
    for suffix, limit in (("components", count), ("vectors", count // 128)):
        nonzero = row[f"reference_nonzero_{suffix}"]
        check(nonzero <= limit, f"{where}: reference nonzero count exceeds geometry")
        check(
            row[f"zero_actual_nonzero_reference_{suffix}"] <= nonzero,
            f"{where}: lost-gradient count exceeds nonzero references",
        )
    close(
        row["relative_l2_error"], ratio(row["error_squared_norm"], row["reference_squared_norm"]), where + ".relative_l2"
    )
    for name, norm in (("actual_l2", "actual_squared_norm"), ("reference_l2", "reference_squared_norm")):
        if name in row:
            close(row[name], math.sqrt(row[norm]), where + "." + name)
    if "mean_absolute_error" in row:
        close(row["mean_absolute_error"], row["absolute_error_sum"] / count, where + ".mean_absolute_error")
    maximum = row["max_absolute_error"]
    check(
        maximum >= 0 and maximum <= row["absolute_error_sum"] * (1 + 1e-10) + 1e-300,
        where + ": maximum exceeds absolute error sum",
    )
    check(maximum**2 <= row["error_squared_norm"] * (1 + 1e-10) + 1e-300, where + ": maximum exceeds squared error sum")
    cosine = row.get("cosine_similarity")
    if cosine is not None:
        check(-1 - 1e-12 <= cosine <= 1 + 1e-12, where + ": invalid cosine")


def pooled_stats(rows):
    out = {
        name: sum(row[name] for row in rows) if name in INTEGER_STATS else math.fsum(row[name] for row in rows)
        for name in ADDITIVE
    }
    out["max_absolute_error"] = max(row["max_absolute_error"] for row in rows)
    out["relative_l2_error"] = ratio(out["error_squared_norm"], out["reference_squared_norm"])
    return out


def pooled_rows(rows):
    combined = {}
    for name in rows[0]["rows"]:
        values = [row["rows"][name] for row in rows]
        if name == "margin_histogram_edges":
            combined[name] = values[0]
        elif name == "margin_histogram":
            combined[name] = [sum(bin_values) for bin_values in zip(*values, strict=True)]
        elif name.endswith("_max_absolute_score_gradient_error"):
            combined[name] = max(values)
        else:
            combined[name] = sum(values) if all(type(v) is int for v in values) else math.fsum(values)
    return combined


def aggregate(rows):
    out = {name: pooled_stats([row[name] for row in rows]) for name in GRADIENTS}
    out.update(
        rows=pooled_rows(rows),
        panel_count=len(rows),
        replay_output_max_abs_error=max(row["replay_output_max_abs_error"] for row in rows),
        tail_output_max_abs_error=max(row["tail_output_max_abs_error"] for row in rows),
        tail_output_reference_rms_max=max(row["tail_output_reference_rms"] for row in rows),
        selection_fallback_count=sum(row["selection_fallback_count"] for row in rows),
    )
    out["controls"] = {
        control: {
            **{name: pooled_stats([row["controls"][control][name] for row in rows]) for name in ("dq", "dk")},
            "zero_winner_nonzero_reference_count": sum(
                row["controls"][control]["zero_winner_nonzero_reference_count"] for row in rows
            ),
        }
        for control in CONTROLS
    }
    for key, score in EVENTS.items():
        out[key] = sorted([event for row in rows for event in row[key]], key=score, reverse=True)[:8]
    return out


def event_key(event):
    return tuple(event[name] for name in ("rank", "layer", "global_sequence_index", "query_position", "query_head"))


def validate_panel(panel, request, layer, where):
    heads, width, queries = request["model"]["num_heads"], request["model"]["head_dim"], request["query_count"]
    keys, kv_heads = request["model"]["max_seq_len"], request["model"]["local_kv_heads"]
    expected_rows = queries * heads
    check(panel["layer"] == layer and set(panel["variants"]) == set(VARIANTS), where + ": layer/variants differ")
    check(set(panel["gates"]) == set(ZERO_GATES + OPTIONAL_GATES), where + ": gate schema differs")
    for name in ZERO_GATES:
        check(panel["gates"][name] == 0, where + ": strict gate failed: " + name)
    positions = panel["positions"]
    check(len(positions) == 1 and len(positions[0]) == queries, where + ": wrong query geometry")
    check(
        len(set(positions[0])) == queries and all(type(x) is int and 0 <= x < keys for x in positions[0]),
        where + ": invalid selected positions",
    )
    check(
        all(i * (keys // queries) <= x < (i + 1) * (keys // queries) for i, x in enumerate(positions[0])),
        where + ": query strata differ",
    )
    check(0 <= panel["local_row"] < request["batch_size"] // 64, where + ": invalid local sequence")
    check(0 <= panel["global_sequence_index"] < request["batch_size"], where + ": invalid global sequence")
    baseline = panel["variants"]["baseline"]
    for variant, row in panel["variants"].items():
        label = where + "." + variant
        counts = row["rows"]
        for key, value in counts.items():
            if key.endswith("_count"):
                integer(value, label + "." + key)
        check(
            sum(counts[x] for x in ("masked_count", "single_key_count", "multi_key_count")) == expected_rows,
            label + ": query-head denominator differs",
        )
        check(
            counts["multi_key_count"] == (queries - row["selection_fallback_count"]) * heads,
            label + ": fallback/multikey denominator differs",
        )
        histogram = counts["margin_histogram"]
        edges = [float(x) for x in counts["margin_histogram_edges"]]
        check(len(histogram) == 11 and len(edges) == 12, label + ": histogram geometry differs")
        check(
            edges == [-math.inf, 0, 4, 8, 12, 14, 16, 24 * math.log(2), 18, 24, 40, math.inf],
            label + ": histogram edges differ",
        )
        for value in histogram:
            integer(value, label + ".histogram")
        check(sum(histogram) == counts["multi_key_count"], label + ": histogram denominator differs")
        check(
            sum(histogram[8:]) <= counts["margin_above_fp32_cliff_count"] <= sum(histogram[7:]),
            label + ": margin tail/histogram counts disagree",
        )
        check(
            0 <= counts["tail_mass_sum"] <= counts["multi_key_count"] and counts["entropy_sum"] >= 0,
            label + ": invalid entropy/tail mass",
        )
        check(
            counts["stored_output_zero_winner_nonzero_reference_count"] <= counts["multi_key_count"],
            label + ": cancellation denominator differs",
        )
        for name in GRADIENTS:
            components = expected_rows * width if name in ("dq", "output") else keys * kv_heads * width
            stats(row[name], label + "." + name, components)
            check(
                row[name]["reference_squared_norm"] == baseline[name]["reference_squared_norm"],
                label + ": reference changed across variants",
            )
        for key in REFERENCE_ROWS:
            check(counts[key] == baseline["rows"][key], label + ": reference row metrics changed across variants")
        check(set(row["controls"]) == set(CONTROLS), label + ": missing stored-output control")
        for control, data in row["controls"].items():
            for name in ("dq", "dk"):
                stats(data[name], label + ".controls." + control + "." + name, row[name]["component_count"])
                check(
                    data[name]["reference_squared_norm"] == row[name]["reference_squared_norm"],
                    label + ": stored-output control reference changed",
                )
            check(
                0 <= data["zero_winner_nonzero_reference_count"] <= counts["multi_key_count"],
                label + ": invalid cancellation count",
            )
        for key, score in EVENTS.items():
            events = row[key]
            limit = 4 if key == "largest_margin_events" else 8
            check(
                len(events) <= limit and [score(e) for e in events] == sorted([score(e) for e in events], reverse=True),
                label + ": bounded event order differs",
            )
            for event in events:
                check(
                    event["rank"] == panel["rank"] and event["layer"] == layer and event["valid_key_count"] > 1,
                    label + ": wrong event identity/mask",
                )
                check(
                    event["global_sequence_index"] == panel["global_sequence_index"]
                    and event["batch"] == 0
                    and 0 <= event["query"] < queries,
                    label + ": wrong event selected sequence/query",
                )
                check(
                    event["query_position"] == positions[0][event["query"]]
                    and event["query_head"] == event["head"]
                    and 0 <= event["head"] < heads
                    and event["boundary_kv_head"] == event["head"] // (heads // kv_heads),
                    label + ": wrong event geometry",
                )
    projection = panel["projection"]
    check(
        projection["row_count"] == expected_rows
        and projection["minimum_probability_mass"] >= 0
        and projection["maximum_probability_mass"] >= projection["minimum_probability_mass"],
        where + ": projection geometry/mass differs",
    )


def validate_benchmarks(panels, request, step, layer):
    active = step == request["benchmark_checkpoint"] and layer in request["benchmark_layers"]
    expected = (
        {
            (name, length)
            for name in ("baseline", "fp32_saved_output", "fp32_saved_output_shadow")
            for length in request["benchmark_lengths"]
        }
        if active
        else set()
    )
    rows = []
    for panel in panels:
        check(
            {(x["implementation"], x["shape"]["local_q"][1]) for x in panel["benchmarks"]} == expected
            and len(panel["benchmarks"]) == len(expected),
            "Dense benchmark panel incomplete/duplicated",
        )
        for row in panel["benchmarks"]:
            name, length = row["implementation"], row["shape"]["local_q"][1]
            check(
                row["rank"] == panel["rank"] and row["layer"] == layer and row["device_count"] == 64,
                "Dense benchmark identity differs",
            )
            check(
                row["shape"] == {"local_q": [request["batch_size"] // 64, length, 48, 128], "kv_heads": 12},
                "Dense benchmark shape differs",
            )
            check(
                row["git_sha"] == request["source_revision"]
                and row["native_compilation"] == request["native_compilation"],
                "Dense benchmark source/mode differs",
            )
            check(
                row["error"] is None and not row["projection_dense_supported"] and row["compile_time"] > 0,
                "Dense benchmark error/support differs",
            )
            check(
                len(row["samples"]) == request["benchmark_repetitions"] and all(x > 0 for x in row["samples"]),
                "Dense benchmark samples invalid",
            )
            close(row["steady_state_time"], statistics.median(row["samples"]), "Dense timing median")
            check(
                row["forward_query_stages"]
                == {"baseline": [2], "fp32_saved_output": [1], "fp32_saved_output_shadow": [2, 1]}[name],
                "Dense benchmark schedule differs",
            )
            expected_bytes = (
                0 if name == "baseline" else math.prod(row["shape"]["local_q"]) * (4 if name.endswith("shadow") else 2)
            )
            check(row["saved_output_extra_bytes_per_rank"] == expected_bytes, "Dense saved-state byte count differs")
            rows.append(row)
    return rows


def compact(rows):
    out = {name: pooled_stats([row[name] for row in rows]) for name in GRADIENTS}
    for axis in GRADIENTS:
        out[axis].update(
            absolute_l2_error=math.sqrt(out[axis]["error_squared_norm"]),
            actual_l2=math.sqrt(out[axis]["actual_squared_norm"]),
            reference_l2=math.sqrt(out[axis]["reference_squared_norm"]),
        )
        values = [row[axis]["relative_l2_error"] for row in rows if row[axis]["relative_l2_error"] is not None]
        out[axis]["unweighted_mean_panel_relative_l2_not_pooled"] = statistics.mean(values) if values else None
    out["rows"] = pooled_rows(rows)
    out["selection_fallback_count"] = sum(row["selection_fallback_count"] for row in rows)
    count = sum(row["rows"]["multi_key_count"] for row in rows)
    out["multi_key_count"] = count
    out["margin_above_fp32_cliff_count"] = sum(row["rows"]["margin_above_fp32_cliff_count"] for row in rows)
    out["margin_above_fp32_cliff_rate"] = out["margin_above_fp32_cliff_count"] / count if count else None
    out["mean_entropy_multikey"] = math.fsum(row["rows"]["entropy_sum"] for row in rows) / count if count else None
    out["controls"] = {
        name: {axis: pooled_stats([row["controls"][name][axis] for row in rows]) for axis in ("dq", "dk")}
        for name in CONTROLS
    }
    out["worst_dq_events"] = sorted(
        [event for row in rows for event in row["worst_dq_events"]], key=lambda e: e["dq_error_l2"], reverse=True
    )[:2]
    return out


def validate_producer_objects(producer_root, records):
    expected = {}
    for row in records:
        digest, size = row["module_sha256"], row["module_bytes"]
        check(
            isinstance(digest, str) and len(digest) == 64 and all(x in "0123456789abcdef" for x in digest),
            "Producer object proof has invalid SHA256",
        )
        check(type(size) is int and 0 < size <= 512 * 1024, "Producer object proof has invalid size")
        check(digest not in expected or expected[digest] == size, "Repeated object SHA has inconsistent byte counts")
        expected[digest] = size
    paths = {}
    directory = producer_root / "objects"
    for path in [*directory.glob("*.o"), *directory.glob("*.o.gz")]:
        digest = path.name.removesuffix(".gz").removesuffix(".o")
        paths.setdefault(digest, []).append(path)
    check(set(paths) == set(expected), "Actual producer object files differ from unique proof objects")
    representations = 0
    for digest, files in paths.items():
        for path in files:
            check(path.is_file() and not path.is_symlink(), "Producer object must be a regular local file")
            raw = path.read_bytes()
            if path.suffix == ".gz":
                raw = gzip.decompress(raw)
            check(len(raw) == expected[digest], "Actual producer object byte count differs from proof")
            actual = hashlib.sha256(raw).hexdigest()
            check(actual == digest, "Actual producer object bytes do not match SHA256/proof/filename")
            CHECKSUMS[str(path)] = actual
            representations += 1
    total = sum(expected.values())
    check(total <= 32 * 1024 * 1024, "Actual producer objects exceed declared per-context bound")
    return {
        "unique_object_count": len(expected),
        "total_object_bytes": total,
        "file_representations_checked": representations,
        "actual_file_bytes_verified": True,
    }


def validate_proofs(root, request):
    phases = [f"smoke/length-{n}-heads-{h}" for n, h in ((128, 4), (256, 8))]
    for checkpoint in request["checkpoints"]:
        for layer in request["layers"]:
            phase = f"checkpoint/step-{checkpoint['step']}/layer-{layer}"
            phases.append(phase + "/replay")
            if checkpoint["step"] == request["benchmark_checkpoint"] and layer in request["benchmark_layers"]:
                phases.append(phase + "/dense-benchmark")
    report = []
    for phase in phases:
        provenance, calls = [], []
        for rank in range(64):
            path = root / phase / f"rank-{rank}"
            prov = load(path / "provenance.json")
            check(prov["source"]["source_revision"] == request["source_revision"], "Consensus source revision differs")
            provenance.append(prov)
            files = sorted(
                {
                    Path(str(file).removesuffix(".gz"))
                    for file in [*path.glob("calls/call-*.json"), *path.glob("calls/call-*.json.gz")]
                }
            )
            calls.append([load(file) for file in files])
            check(
                not (path / "failed.json").exists() and not (path / "failed.json.gz").exists(),
                "Consensus phase has a failed record",
            )
        check(all(x == provenance[0] for x in provenance), "Consensus source/runtime provenance differs across ranks")
        check(all(len(x) == len(calls[0]) for x in calls), "Consensus call counts differ across ranks")
        digest = hashlib.sha256(json.dumps(provenance[0], sort_keys=True, allow_nan=False).encode()).hexdigest()
        for records in calls:
            for ordinal, row in enumerate(records):
                check(
                    row["status"] == "complete" and row["producer"] == 0 and row["identity"]["ordinal"] == ordinal,
                    "Consensus status/ordinal differs",
                )
                check(row["identity"]["provenance_sha256"] == digest, "Consensus provenance hash differs")
                identity_digest = hashlib.sha256(
                    json.dumps(row["identity"], sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                check(
                    row["identity_sha256"] == identity_digest and 0 < row["module_bytes"] <= 512 * 1024,
                    "Consensus request hash/object bound differs",
                )
                check(row == calls[0][ordinal], "Consensus identity/actual object proof differs across ranks")
        objects = validate_producer_objects(root / phase / "rank-0", calls[0])
        report.append(
            {
                "phase": phase,
                "calls_per_rank": len(calls[0]),
                "cached_trace_without_native_calls": not calls[0],
                "producer_objects": objects,
            }
        )
    return report


def validate(args):
    root = args.root
    expected = load(args.request)
    expected["source_revision"] = args.source_revision
    document = load(root / "results.json")
    check(document["request"] == expected, "Resolved request/source differs from submitted configuration")
    check(
        [x["checkpoint"]["step"] for x in document["results"]] == [270000, 300000, 330000],
        "Checkpoint set/order differs",
    )
    check(
        expected["layers"] == [0, 3, 12, 15, 24, 27, 36, 39, 44, 47] and expected["z_loss_weight"] == 1e-4,
        "Scientific layer/objective configuration differs",
    )
    for key, value in {
        "query_count": 4,
        "batch_size": 256,
        "production_batch_size": 2816,
        "data_reference_step": 330000,
        "benchmark_layers": [0, 15],
        "benchmark_lengths": [256, 2048, 16384],
        "benchmark_checkpoint": 330000,
        "benchmark_warmup": 2,
        "benchmark_repetitions": 8,
        "native_compilation": "shared_object",
        "kernel_cache": "in_memory",
        "native_object_timeout": 300,
    }.items():
        check(expected[key] == value, "Scientific comparison configuration differs: " + key)
    for key, value in {
        "num_heads": 48,
        "local_kv_heads": 12,
        "global_kv_heads": 6,
        "head_dim": 128,
        "max_seq_len": 16384,
        "global_every": 4,
        "num_layers": 48,
    }.items():
        check(expected["model"][key] == value, "Scientific attention geometry differs: " + key)
    result = {
        "private_validation": True,
        "source_revision": args.source_revision,
        "checkpoints": {},
        "layer_rows": [],
        "expected_multikey_without_fallbacks": {"per_layer": 12288, "per_checkpoint": 122880, "overall": 368640},
    }
    selections = {}
    overall = {name: [] for name in VARIANTS}
    benchmark_rows = []
    for checkpoint in document["results"]:
        step = checkpoint["checkpoint"]["step"]
        check(
            checkpoint["checkpoint"] == next(x for x in expected["checkpoints"] if x["step"] == step),
            "Checkpoint provenance differs",
        )
        check([x["layer"] for x in checkpoint["layers"]] == expected["layers"], "Layer set/order differs")
        step_rows = {name: [] for name in VARIANTS}
        groups = {group: {name: [] for name in VARIANTS} for group in ("local", "global")}
        layer15 = {}
        for layer_summary in checkpoint["layers"]:
            layer = layer_summary["layer"]
            panels = load(root / f"step-{step}" / f"layer-{layer}-panels.json")
            check(len(panels) == 64 and {x["rank"] for x in panels} == set(range(64)), "Rank panel set differs")
            check(len({x["global_sequence_index"] for x in panels}) == 64, "Selected sequences overlap across ranks")
            for panel in panels:
                validate_panel(panel, expected, layer, f"step{step}.layer{layer}.rank{panel['rank']}")
                key = (layer, panel["rank"])
                selection = (panel["positions"], panel["local_row"], panel["global_sequence_index"])
                check(
                    key not in selections or selections[key] == selection,
                    "Fixed-batch selection changed across checkpoints",
                )
                selections[key] = selection
            check(layer_summary["panel_count"] == 64, "Layer panel denominator differs")
            is_global = (layer + 1) % expected["model"]["global_every"] == 0 or layer == expected["model"][
                "num_layers"
            ] - 1
            check(layer_summary["is_global"] == is_global, "Local/global mask classification differs")
            gates = {name: max(panel["gates"][name] for panel in panels) for name in panels[0]["gates"]}
            close(layer_summary["gates_max"], gates, "Layer gate aggregation")
            check(set(layer_summary["variants"]) == set(VARIANTS), "Layer variants differ")
            benchmark_rows.extend(validate_benchmarks(panels, expected, step, layer))
            close(
                layer_summary["benchmarks"],
                [row for panel in panels for row in panel["benchmarks"]],
                "Dense benchmark aggregation",
            )
            per_layer = {}
            for name in VARIANTS:
                rows = [panel["variants"][name] for panel in panels]
                combined = aggregate(rows)
                close(layer_summary["variants"][name], combined, "Layer pooled aggregation")
                step_rows[name].extend(rows)
                overall[name].extend(rows)
                groups["global" if is_global else "local"][name].extend(rows)
                per_layer[name] = compact(rows)
            if layer == 15:
                layer15 = per_layer
            projection = {
                key: (
                    min(x["projection"][key] for x in panels)
                    if key.startswith("minimum")
                    else (
                        max(x["projection"][key] for x in panels)
                        if key.startswith("maximum")
                        else sum(x["projection"][key] for x in panels)
                    )
                )
                for key in panels[0]["projection"]
            }
            result["layer_rows"].append(
                {
                    "step": step,
                    "layer": layer,
                    "is_global": is_global,
                    "variants": per_layer,
                    "gates_max": gates,
                    "projection": projection,
                }
            )
        pooled = {name: compact(rows) for name, rows in step_rows.items()}
        for name in VARIANTS:
            for axis in ("dq", "dk"):
                denominator = pooled[name][axis]["error_squared_norm"]
                layer15[name][axis]["fraction_of_checkpoint_squared_error"] = (
                    layer15[name][axis]["error_squared_norm"] / denominator if denominator else None
                )
        worst = pooled["baseline"]["worst_dq_events"]
        worst = worst[0] if worst else None
        paired = {
            name: next(
                (
                    event
                    for row in step_rows[name]
                    for key in EVENTS
                    for event in row[key]
                    if worst and event_key(event) == event_key(worst)
                ),
                None,
            )
            for name in VARIANTS
        }
        result["checkpoints"][str(step)] = {
            "variants": pooled,
            "layer15": layer15,
            "local_global": {
                group: {name: compact(rows) for name, rows in variants.items()} for group, variants in groups.items()
            },
            "baseline_worst_row_paired_if_retained": paired,
        }
    result["overall"] = {name: compact(rows) for name, rows in overall.items()}
    baseline_times = {
        (x["rank"], x["layer"], x["shape"]["local_q"][1]): x["steady_state_time"]
        for x in benchmark_rows
        if x["implementation"] == "baseline"
    }
    result["dense_timings"] = []
    for layer in expected["benchmark_layers"]:
        for length in expected["benchmark_lengths"]:
            for implementation in ("baseline", "fp32_saved_output", "fp32_saved_output_shadow"):
                rows = [
                    x
                    for x in benchmark_rows
                    if x["layer"] == layer
                    and x["shape"]["local_q"][1] == length
                    and x["implementation"] == implementation
                ]
                check(len(rows) == 64, "Dense timing rank count differs")
                values = [x["steady_state_time"] for x in rows]
                paired_ratios = [x["steady_state_time"] / baseline_times[(x["rank"], layer, length)] for x in rows]
                result["dense_timings"].append(
                    {
                        "layer": layer,
                        "length": length,
                        "implementation": implementation,
                        "rank_count": 64,
                        "median_rank_seconds": statistics.median(values),
                        "slowest_rank_seconds": max(values),
                        "median_paired_ratio_to_baseline": statistics.median(paired_ratios),
                    }
                )
    result["consensus_proofs"] = (
        validate_proofs(args.proof_root, expected)
        if args.proof_root
        else "not checked; monitoring owner validates separately"
    )
    result["checksums_uncompressed_sha256"] = CHECKSUMS
    result["limitations"] = [
        "Iris terminal success is an independent required check.",
        "Aggregates validate saved statistics; raw operands are absent and "
        "gradients/reference/masks cannot be recomputed here.",
        "dK/dV sum selected-query cotangents and expanded boundary heads, not full-step parameter gradients.",
        "Query/head observations are correlated; no IID confidence intervals.",
        "Stored-output controls reconstruct score gradients and do not prove causality.",
        "A missing paired worst row means it was not retained among bounded events; it does not prove zero error.",
        "Sparse GProj is an accuracy prototype; no dense projection cost or training-quality inference.",
    ]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--proof-root", type=Path)
    arguments = parser.parse_args()
    summary = validate(arguments)
    print(
        json.dumps(
            {
                "validated_checkpoints": list(summary["checkpoints"]),
                "layer_rows": len(summary["layer_rows"]),
                "multi_key_rows": summary["overall"]["baseline"]["multi_key_count"],
                "output": str(arguments.output),
            }
        )
    )
