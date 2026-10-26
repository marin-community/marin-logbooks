# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only arithmetic and panel-identity validation for Hero attention results."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
GRADIENT_SUM_FIELDS = (
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


def finite_tree(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float):
        assert math.isfinite(value), (path, value)


def close(actual, expected, key):
    if actual is None or expected is None:
        assert actual is expected, (key, actual, expected)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-300), (key, actual, expected)


def gradient_check(stat, path):
    norm = stat["reference_squared_norm"]
    close(stat["relative_l2_error"], math.sqrt(stat["error_squared_norm"] / norm) if norm else None, path)
    assert 0 <= stat["zero_actual_nonzero_reference_components"] <= stat["reference_nonzero_components"]
    assert stat["reference_nonzero_components"] <= stat["component_count"]
    assert 0 <= stat["zero_actual_nonzero_reference_vectors"] <= stat["reference_nonzero_vectors"]


def aggregate_error(items):
    result = {}
    for gradient in ("dq", "dk", "dv", "output"):
        reference = math.fsum(item[gradient]["reference_squared_norm"] for item in items)
        error = math.fsum(item[gradient]["error_squared_norm"] for item in items)
        result[gradient] = {
            "relative_l2_error": math.sqrt(error / reference) if reference else None,
            "reference_l2": math.sqrt(reference),
            "error_l2": math.sqrt(error),
            "zero_actual_nonzero_reference_vectors": sum(
                item[gradient]["zero_actual_nonzero_reference_vectors"] for item in items
            ),
            "reference_nonzero_vectors": sum(item[gradient]["reference_nonzero_vectors"] for item in items),
        }
    result["multi_key_rows"] = sum(item["rows"]["multi_key_count"] for item in items)
    result["margin_above_fp32_cliff"] = sum(item["rows"]["margin_above_fp32_cliff_count"] for item in items)
    result["stored_output_control_winner_cancellations"] = sum(
        item["rows"]["stored_output_zero_winner_nonzero_reference_count"] for item in items
    )
    result["margin_histogram"] = [sum(item["rows"]["margin_histogram"][i] for item in items) for i in range(11)]
    result["max_retained_margin"] = max(event["margin"] for item in items for event in item["largest_margin_events"])
    result["max_retained_dq_relative_error"] = max(
        event["dq_relative_l2_error"] or 0
        for item in items
        for event_key in ("worst_dq_events", "worst_cancellation_events", "largest_margin_events")
        for event in item[event_key]
    )
    return result


def control_alignment(panels, name, gradient):
    actual_error = math.fsum(panel[gradient]["error_squared_norm"] for panel in panels)
    control_error = math.fsum(panel["controls"][name][gradient]["error_squared_norm"] for panel in panels)
    dots = []
    residuals = []
    for panel in panels:
        actual = panel[gradient]["error_squared_norm"]
        control = panel["controls"][name][gradient]["error_squared_norm"]
        alignment = panel["controls"][name][f"{gradient}_fused_error_alignment"]
        cosine = alignment["cosine_error_alignment"]
        dot = cosine * math.sqrt(actual * control) if cosine is not None else 0.0
        dots.append(dot)
        residual = max(actual + control - 2 * dot, 0)
        ratio = alignment["residual_error_l2_over_fused_error_l2"]
        if actual:
            measured_residual = ratio * ratio * actual
            # Recovering a tiny residual by subtracting aligned error norms is
            # ill-conditioned. Check its absolute backward error at norm scale.
            assert abs(measured_residual - residual) <= 64 * math.ulp(1.0) * (actual + control)
            residuals.append(measured_residual)
        else:
            assert ratio is None
            residuals.append(residual)
    return {
        "cosine_error_alignment": (
            math.fsum(dots) / math.sqrt(actual_error * control_error) if actual_error and control_error else None
        ),
        "residual_error_l2_over_fused_error_l2": (
            math.sqrt(math.fsum(residuals) / actual_error) if actual_error else None
        ),
    }


def main():
    raw = json.loads((ROOT / "results.json").read_text())
    finite_tree(raw)
    expected_layers = raw["request"]["layers"]
    assert expected_layers == [0, 3, 12, 15, 24, 27, 36, 39, 44, 47]
    assert [row["checkpoint"]["step"] for row in raw["results"]] == [270000, 300000, 330000]
    assert len(raw["results"]) == 3
    result = {"checks": [], "checkpoints": [], "files_sha256": {}}
    all_panels = []
    all_layers = []
    selections = {}
    for checkpoint in raw["results"]:
        step = checkpoint["checkpoint"]["step"]
        assert checkpoint["checkpoint"] == next(c for c in raw["request"]["checkpoints"] if c["step"] == step)
        assert [layer["layer"] for layer in checkpoint["layers"]] == expected_layers
        layers = checkpoint["layers"]
        cp_panels = []
        for layer in layers:
            index = layer["layer"]
            assert layer["is_global"] == ((index + 1) % 4 == 0 or index == 47)
            assert layer["panel_count"] == 64
            assert layer["replay_output_max_abs_error"] == layer["tail_output_max_abs_error"] == 0
            assert layer["selection_fallback_count"] == 0
            rows = layer["rows"]
            assert rows["multi_key_count"] == 1024
            assert rows["single_key_count"] == rows["masked_count"] == 0
            assert sum(rows["margin_histogram"]) == rows["multi_key_count"]
            assert sum(rows["margin_histogram"][7:]) == rows["margin_above_fp32_cliff_count"]
            assert layer["dq"]["component_count"] == 64 * 4 * 4 * 128
            assert layer["dk"]["component_count"] == 64 * 16384 * 128
            for gradient in ("dq", "dk", "dv", "output"):
                gradient_check(layer[gradient], f"{step}/{index}/{gradient}")
            prefix = ROOT / f"step-{step}"
            panel_file = prefix / f"layer-{index}-panels.json"
            summary_file = prefix / f"layer-{index}-summary.json"
            panels = json.loads(panel_file.read_text())
            saved_summary = json.loads(summary_file.read_text())
            assert saved_summary == layer, (step, index, "summary differs from results")
            assert len(panels) == 64
            assert {panel["rank"] for panel in panels} == set(range(64))
            assert len({panel["global_sequence_index"] for panel in panels}) == 64
            finite_tree(panels)
            for panel in panels:
                rank = panel["rank"]
                assert panel["layer"] == index
                assert panel["kv_group"] == (rank + index) % 12
                assert panel["global_sequence_index"] == rank * 4 + rank % 4
                assert [position // 4096 for position in panel["positions"][0]] == [0, 1, 2, 3]
                assert panel["rows"]["multi_key_count"] == 16
                assert panel["rows"]["single_key_count"] == panel["rows"]["masked_count"] == 0
                assert sum(panel["rows"]["margin_histogram"]) == 16
                assert panel["replay_output_max_abs_error"] == panel["tail_output_max_abs_error"] == 0
                selection = (panel["positions"], panel["global_sequence_index"], panel["kv_group"])
                key = (index, rank)
                if key in selections:
                    assert selection == selections[key], (step, index, rank, "selection changed across checkpoints")
                selections[key] = selection
                for gradient in ("dq", "dk", "dv", "output"):
                    gradient_check(panel[gradient], f"{step}/{index}/rank-{rank}/{gradient}")
            for gradient in ("dq", "dk", "dv", "output"):
                for field in GRADIENT_SUM_FIELDS:
                    close(
                        layer[gradient][field],
                        math.fsum(panel[gradient][field] for panel in panels),
                        f"{step}/{index}/{gradient}/{field}",
                    )
                assert layer[gradient]["max_absolute_error"] == max(
                    panel[gradient]["max_absolute_error"] for panel in panels
                )
            for field, value in rows.items():
                if field == "margin_histogram_edges":
                    assert all(panel["rows"][field] == value for panel in panels)
                elif field == "margin_histogram":
                    assert value == [sum(panel["rows"][field][i] for panel in panels) for i in range(len(value))]
                elif field.endswith("_max_absolute_score_gradient_error"):
                    assert value == max(panel["rows"][field] for panel in panels)
                else:
                    close(value, math.fsum(panel["rows"][field] for panel in panels), f"{step}/{index}/rows/{field}")
            for name, control in layer["controls"].items():
                for gradient in ("dq", "dk"):
                    for field in GRADIENT_SUM_FIELDS:
                        close(
                            control[gradient][field],
                            math.fsum(panel["controls"][name][gradient][field] for panel in panels),
                            f"{step}/{index}/{name}/{gradient}/{field}",
                        )
                    gradient_check(control[gradient], f"{step}/{index}/{name}/{gradient}")
                assert control["zero_winner_nonzero_reference_count"] == sum(
                    panel["controls"][name]["zero_winner_nonzero_reference_count"] for panel in panels
                )
            sort_keys = {
                "worst_dq_events": lambda event: event["dq_error_l2"],
                "worst_cancellation_events": lambda event: abs(event["reference_winner_score_gradient"]),
                "largest_margin_events": lambda event: event["margin"],
            }
            for event_key, sort_key in sort_keys.items():
                expected = sorted([event for panel in panels for event in panel[event_key]], key=sort_key, reverse=True)[
                    :8
                ]
                assert layer[event_key] == expected
                for event in layer[event_key]:
                    assert event["layer"] == index and event["valid_key_count"] >= 2
                    assert event["query_head"] == event["boundary_kv_head"] * 4 + event["head"]
                    if event_key == "worst_cancellation_events":
                        assert event["stored_output_reconstructed_winner_score_gradient"] == 0
                        assert event["reference_winner_score_gradient"] != 0
            cp_panels.extend(panels)
            for file in (panel_file, summary_file):
                result["files_sha256"][str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
        cp_result = {"step": step, "regularized_loss": checkpoint["regularized_loss"]}
        for label, selected in (
            ("all", layers),
            ("local", [layer for layer in layers if not layer["is_global"]]),
            ("global", [layer for layer in layers if layer["is_global"]]),
        ):
            cp_result[label] = aggregate_error(selected)
        cp_result["control_fused_error_alignment"] = {
            name: {gradient: control_alignment(cp_panels, name, gradient) for gradient in ("dq", "dk")}
            for name in ("actual_stored_output", "bf16_rounded_reference_output")
        }
        result["checkpoints"].append(cp_result)
        all_panels.extend(cp_panels)
        all_layers.extend(layers)
    assert len(all_panels) == 1920
    assert len(selections) == 640
    assert raw["synthetic"]["dq"]["actual_squared_norm"] == 0
    assert raw["synthetic"]["dq"]["relative_l2_error"] == 1
    assert raw["synthetic"]["rows"]["stored_output_zero_winner_nonzero_reference_count"] == 4
    result["all_checkpoints"] = aggregate_error(all_layers)
    result["all_checkpoints"]["panel_count"] = len(all_panels)
    result["all_checkpoints"]["control_fused_error_alignment"] = {
        name: {gradient: control_alignment(all_panels, name, gradient) for gradient in ("dq", "dk")}
        for name in ("actual_stored_output", "bf16_rounded_reference_output")
    }
    result["files_sha256"]["results.json"] = hashlib.sha256((ROOT / "results.json").read_bytes()).hexdigest()
    result["checks"] = [
        "Finite JSON numeric values across synthetic, summaries, and all1920 individual panels",
        "Three pinned checkpoints, ten requested layers each, and64 unique ranks per layer",
        "Exact per-layer summary equality with the completion manifest",
        "All additive count/squared-norm/absolute-error fields and histogram sums recomputed from per-rank panels",
        "Relative L2 recomputed from aggregate squared error/reference norms, not averaged panel ratios",
        "Control alignment and residual identity verified from individual rank metrics",
        "Identical selected sequences, head groups, and positions across checkpoints",
        "Bitwise BF16 capture-tail and replay-output parity across every panel",
        "Bounded event selection reproduced from per-rank records",
        "Actual fused synthetic probe confirms winner-gradient cancellation",
    ]
    result["limitations"] = [
        (
            "Raw checkpoint-derived Q/K/V/dO arrays were not archived here; "
            "the norm inputs cannot be recomputed from this JSON alone"
        ),
        "Stored-output correction events are labeled reconstruction controls, not measured internal fused dS",
        "Sparse query dK includes only chosen cotangents and one expanded KV group per rank",
        (
            "Global expanded KV boundary duplicates logical KV heads; "
            "this does not sum gradients at the original logical K projection"
        ),
        (
            "One batch, ten selected layers, four stratified multi-key queries "
            "and one sequence per rank, with correlated samples"
        ),
        "High relative errors on tiny gradients do not establish downstream optimizer or quality effect",
        (
            "The largest retained row-relative error covers three bounded event lists "
            "and is not necessarily the maximum across every measured row"
        ),
    ]
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
