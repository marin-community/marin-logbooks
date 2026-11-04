# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a single-attempt completed prefix with the saved historical run."""

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

TRAIN_KEYS = ("train/loss", "throughput/duration", "throughput/mfu", "optim/learning_rate", "optim/adam_lr")
FLOPS = 80353469399040


def parse_log(text):
    rows = {}
    evaluations = []
    sources = set()
    malformed = []
    norms = []
    lowerings = []
    last_loss = -1
    for line_index, line in enumerate(text.splitlines()):
        if "levanter.trainer Lowering train_step" in line:
            lowerings.append(dict(line_index=line_index, next_label_after_latest_loss=last_loss + 1, line=line))
        if "ema_gram_metrics {" not in line:
            continue
        try:
            row = json.loads(line.split("ema_gram_metrics ", 1)[1])
        except json.JSONDecodeError:
            raw = line.split("ema_gram_metrics ", 1)[1]
            if '"event": "hparams"' in raw:
                malformed.append(dict(line_index=line_index, kind="hparams"))
            else:
                # CRI chunk timestamps can be interleaved inside long norm JSON.
                # Retain the gap; validate the complete norm values from online history.
                keys = re.findall(r'"([^"\n]+)"\s*:', raw)
                assert keys and all(
                    k in ("tracker", "event", "step", "metrics") or k.startswith(("grad/norm/", "params/norm/"))
                    for k in keys
                )
                assert '"event": "log"' in raw and '"metrics": {"grad/norm/' in raw
                assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:", raw)
                step = int(re.search(r'"step": (\d+)', raw).group(1))
                assert step % 10 == 9
                malformed.append(dict(line_index=line_index, kind="norm_json_chunk_gap", step=step))
            continue
        if row.get("event") != "log" or not isinstance(row.get("step"), int):
            continue
        values, step = row.get("metrics", {}), row["step"]
        if "train/loss" in values:
            last_loss = max(last_loss, step)
        relevant = any(k in values for k in TRAIN_KEYS) or "eval/loss" in values
        if relevant:
            match = re.search(r"task=(\S+) attempt=(\d+)", line)
            assert match is not None
            sources.add((match[1], int(match[2])))
        selected = {
            k: v
            for k, v in values.items()
            if k in TRAIN_KEYS or k in ("throughput/hook_time", "throughput/loading_time")
        }
        for key, value in selected.items():
            dest = rows.setdefault(step, {})
            assert key not in dest or dest[key] == value, "conflicting metric in single-attempt input"
            dest[key] = value
        if "eval/loss" in values:
            evaluations.append(dict(line_index=line_index, step=step, metrics=values))
        norm_values = {k: v for k, v in values.items() if k.startswith(("params/norm/", "grad/norm/"))}
        if norm_values and any(re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:", k) for k in values):
            malformed.append(dict(line_index=line_index, kind="norm_key_chunk_gap", step=step))
            continue
        if norm_values:
            assert all(math.isfinite(v) for v in norm_values.values())
            norms.append(dict(step=step, count=len(norm_values), finite=True, values=norm_values))
    assert len(sources) == 1 and next(iter(sources))[1] == 0, "resolve attempt lineage before aggregation"
    return dict(
        rows=rows,
        evaluations=evaluations,
        sources=sorted(sources),
        malformed=malformed,
        norms=norms,
        lowerings=lowerings,
    )


def compare(parsed, reference, online, target=924, window=(800, 899), model_flops=FLOPS):
    rows = {int(k): v for k, v in parsed["rows"].items() if "train/loss" in v}
    assert set(rows) == set(range(target)), "incomplete or extra training steps"
    ref = {r["_step"]: r for r in reference["history"] if "train/loss" in r}
    live = {r["_step"]: r for r in online["history"] if "train/loss" in r}
    assert set(rows).issubset(ref) and set(rows).issubset(live)
    pairs = []
    for step, values in sorted(rows.items()):
        for key in TRAIN_KEYS:
            assert math.isfinite(values[key]) and live[step][key] == values[key]
        for key in ("optim/learning_rate", "optim/adam_lr"):
            assert values[key] == ref[step][key], (step, key)
        assert values["throughput/duration"] > 0
        assert math.isclose(
            values["throughput/mfu"], model_flops / 4.4e15 / values["throughput/duration"] * 100, rel_tol=1e-12
        )
        assert math.isclose(
            ref[step]["throughput/mfu"], model_flops / 1.1e15 / ref[step]["throughput/duration"] * 100, rel_tol=1e-12
        )
        pairs.append(
            dict(
                step=step,
                candidate=values["train/loss"],
                original=ref[step]["train/loss"],
                delta=values["train/loss"] - ref[step]["train/loss"],
            )
        )
    lo, hi = window
    assert hi < target and lo >= max(x["next_label_after_latest_loss"] for x in parsed["lowerings"]) + 20
    timing = {}
    for name, source, peak in [("candidate", rows, 4.4e15), ("historical_original", ref, 1.1e15)]:
        times = [source[k]["throughput/duration"] for k in range(lo, hi + 1)]
        mean = statistics.mean(times)
        timing[name] = dict(
            count=len(times),
            mean_seconds=mean,
            median_seconds=statistics.median(times),
            min_seconds=min(times),
            max_seconds=max(times),
            stdev_seconds=statistics.pstdev(times),
            aggregate_mfu_percent=model_flops / peak / mean * 100,
            peak_flops_per_second=peak,
        )
    timing["wall_step_speed_ratio"] = timing["historical_original"]["mean_seconds"] / timing["candidate"]["mean_seconds"]
    timing["device_time_efficiency_ratio"] = timing["wall_step_speed_ratio"] / 4
    timing["scope"] = (
        "Matched labels; candidate observed training lowerings precede window by 20+ updates. "
        "Historical compilation log unavailable here; hardware 4 v4 versus 16 v4, not layout-only causation."
    )
    eval_pairs = []
    latest_by_step = {}
    for invocation in parsed["evaluations"]:
        assert all(math.isfinite(v) for v in invocation["metrics"].values())
        latest_by_step[invocation["step"]] = invocation
        if invocation["step"] in ref:
            other = ref[invocation["step"]]
            differences = {
                k: dict(candidate=v, original=other[k], delta=v - other[k])
                for k, v in invocation["metrics"].items()
                if k.startswith("eval/") and k.endswith("loss") and k in other
            }
            if differences:
                eval_pairs.append(
                    dict(step=invocation["step"], line_index=invocation["line_index"], comparisons=differences)
                )
    for step, invocation in latest_by_step.items():
        for key, value in invocation["metrics"].items():
            assert live[step][key] == value, (step, key)
    assert 762 in latest_by_step and target - 1 in latest_by_step
    online_norms = []
    norm_keys = None
    for step, row in sorted(live.items()):
        values = {k: v for k, v in row.items() if k.startswith(("grad/norm/", "params/norm/")) and v is not None}
        if values:
            assert len(values) == 272 and all(math.isfinite(v) for v in values.values())
            norm_keys = set(values) if norm_keys is None else norm_keys
            assert set(values) == norm_keys
            online_norms.append(dict(step=step, count=len(values), finite=True))
    if parsed["norms"] or any(x["kind"].startswith("norm_") for x in parsed["malformed"]):
        assert {x["step"] for x in online_norms} == set(range(9, target, 10))
    for observation in parsed["norms"]:
        assert all(live[observation["step"]][k] == v for k, v in observation["values"].items())
    deltas = [p["delta"] for p in pairs]
    return dict(
        target_counter=target,
        final_loss_label=target - 1,
        paired_train_count=target,
        train_pairs=pairs,
        mean_train_loss_delta=statistics.mean(deltas),
        stdev_train_loss_delta=statistics.pstdev(deltas),
        max_absolute_train_loss_delta=max(map(abs, deltas)),
        final_train_pair=pairs[-1],
        window=list(window),
        timing=timing,
        eval_pairs=eval_pairs,
        eval_invocations=parsed["evaluations"],
        norm_observations=online_norms,
        complete_local_norm_records=len(parsed["norms"]),
        excluded_malformed_records=parsed["malformed"],
        observed_training_lowerings=parsed["lowerings"],
        all_learning_rates_match=True,
        all_train_loss_duration_mfu_match_online=True,
        limits=[
            "One seed; no seed-level uncertainty or numerical loss-equivalence margin.",
            "Final forced evaluation has no historical same-step validation counterpart.",
            "Finite losses/norms are not an all-intermediate/state finite audit.",
            "Canonical success, four process reports and durable checkpoint verified separately.",
            "Native step durations exclude loading/hooks; no end-to-end speed claim from their ratio.",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("log", "reference", "online", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    inputs = {k: getattr(args, k).read_bytes() for k in ("log", "reference", "online")}
    result = compare(parse_log(inputs["log"].decode()), json.loads(inputs["reference"]), json.loads(inputs["online"]))
    result["inputs"] = {
        k: dict(path=str(getattr(args, k)), bytes=len(v), sha256=hashlib.sha256(v).hexdigest())
        for k, v in inputs.items()
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
