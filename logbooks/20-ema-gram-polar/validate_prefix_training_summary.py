# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static known-answer and mutation checks for the prefix comparison."""

import copy
import json
import math

from summarize_prefix_training import FLOPS, compare, parse_log


def fixture():
    rows = {}
    prior = []
    live = []
    for step in range(924):
        common = {"_step": step, "optim/learning_rate": 0.002, "optim/adam_lr": 0.0003}
        old = {
            **common,
            "train/loss": 3 + step / 1000,
            "throughput/duration": 40.0,
            "throughput/mfu": FLOPS / 1.1e15 / 40 * 100,
        }
        new = {
            **common,
            "train/loss": old["train/loss"] + 0.02,
            "throughput/duration": 4.0,
            "throughput/mfu": FLOPS / 4.4e15 / 4 * 100,
        }
        rows[step] = dict(new)
        if step == 762:
            old["eval/loss"] = 5.0
            new["eval/loss"] = 5.01
        elif step == 923:
            new["eval/loss"] = 4.9
        prior.append(old)
        live.append(new)
    parsed = dict(
        rows=rows,
        lowerings=[dict(next_label_after_latest_loss=9)],
        evaluations=[
            dict(step=762, line_index=1, metrics={"eval/loss": 5.01}),
            dict(step=923, line_index=2, metrics={"eval/loss": 4.9}),
        ],
        norms=[],
        malformed=[],
    )
    values = {f"grad/norm/fixture{i}": float(i + 1) for i in range(272)}
    for step in range(9, 924, 10):
        live[step].update(values)
    parsed["norms"] = [dict(step=9, count=272, finite=True, values=values)]
    return parsed, dict(history=prior), dict(history=live)


def main():
    parsed, prior, live = fixture()
    result = compare(parsed, prior, live)
    assert math.isclose(result["mean_train_loss_delta"], 0.02)
    assert result["timing"]["wall_step_speed_ratio"] == 10
    assert result["timing"]["device_time_efficiency_ratio"] == 2.5
    assert len(result["eval_invocations"]) == 2 and len(result["eval_pairs"]) == 1
    rejected = []
    for label in [
        "missing_step",
        "extra_step",
        "online_missing",
        "online_mismatch",
        "lr_changed",
        "mfu_units",
        "nonfinite",
        "warmup",
        "missing_final_eval",
        "norm_nonfinite",
        "norm_missing_step",
        "norm_key_changed",
    ]:
        p, r, w = copy.deepcopy((parsed, prior, live))
        if label == "missing_step":
            p["rows"].pop(923)
        elif label == "extra_step":
            p["rows"][924] = p["rows"][923]
        elif label == "online_missing":
            w["history"].pop()
        elif label == "online_mismatch":
            w["history"][800]["train/loss"] += 1
        elif label == "lr_changed":
            r["history"][800]["optim/learning_rate"] *= 2
        elif label == "mfu_units":
            p["rows"][800]["throughput/mfu"] /= 100
            w["history"][800]["throughput/mfu"] /= 100
        elif label == "nonfinite":
            p["rows"][800]["train/loss"] = float("nan")
        elif label == "warmup":
            p["lowerings"][0]["next_label_after_latest_loss"] = 790
        elif label == "missing_final_eval":
            p["evaluations"].pop()
        elif label == "norm_nonfinite":
            w["history"][19]["grad/norm/fixture0"] = float("nan")
        elif label == "norm_missing_step":
            for key in list(w["history"][19]):
                if key.startswith("grad/norm/"):
                    del w["history"][19][key]
        elif label == "norm_key_changed":
            w["history"][19]["grad/norm/wrong"] = w["history"][19].pop("grad/norm/fixture0")
        try:
            compare(p, r, w)
        except AssertionError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    line = "task=/user/job/0 attempt=0 | ema_gram_metrics " + json.dumps(
        dict(event="log", step=0, metrics={"train/loss": 1.0})
    )
    assert parse_log(line)["rows"][0]["train/loss"] == 1.0
    for label, second in [
        ("unresolved_retry", line.replace("attempt=0", "attempt=1")),
        ("conflicting_duplicate", line.replace("1.0", "2.0")),
    ]:
        try:
            parse_log(line + "\n" + second)
        except AssertionError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    norm = "task=/user/job/0 attempt=0 | ema_gram_metrics " + json.dumps(
        dict(event="log", step=9, metrics={"grad/norm/a": 1.0})
    )
    damaged_value = norm.replace("1.0", "1.02026-11-04T01:00:00Z 0")
    damaged_key = norm.replace("grad/norm/a", "grad/norm/a2026-11-04T01:00:00Z b")
    for value, kind in [(damaged_value, "norm_json_chunk_gap"), (damaged_key, "norm_key_chunk_gap")]:
        observed = parse_log(line + "\n" + value)
        assert observed["malformed"][0]["kind"] == kind and not observed["norms"]
    try:
        parse_log(line + "\n" + line[:-3])
    except AssertionError:
        rejected.append("malformed_training_record")
    else:
        raise AssertionError("malformed training record accepted")
    print(
        json.dumps(
            dict(
                passed=True,
                known_answer_loss_delta=0.02,
                known_speed_ratio=10,
                known_device_time_ratio=2.5,
                rejected_mutations=rejected,
                scope="Static fabricated JSON; no model, accelerator, W&B or numerical experiment",
            )
        )
    )


if __name__ == "__main__":
    main()
