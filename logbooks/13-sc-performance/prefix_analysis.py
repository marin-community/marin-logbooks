# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare head approximations and IID estimators on exported model prefixes."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from coefficient_reference import candidate_comparison, coefficients
from rigging.filesystem.buckets import filesystem_for


def read_bytes(uri):
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "rb") as source:
        return source.read()


def arrays_from(reference):
    raw = read_bytes(reference["uri"])
    assert len(raw) == reference["bytes"]
    assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def estimator_evidence(current, old, behavior, advantage, draws, cap):
    logp = current.log_softmax(-1)
    p, q = logp.exp(), behavior.exp()
    coefficient = coefficients(logp, old, behavior, advantage, cap=cap)
    mass = q * coefficient
    exact = mass - p * mass.sum(-1, keepdim=True)
    score_squared = 1 - 2 * p + p.square().sum(-1, keepdim=True)
    ordinary = float((2 * q * coefficient.abs() * (1 - p)).sum())
    results = []
    for control in (False, True):
        selected = coefficient
        if control:
            if not torch.isfinite(behavior).all():
                results.append({"policy_score_control": True, "unavailable": "q has incomplete support"})
                continue
            selected = selected - advantage * (logp.detach() - behavior).exp()
        centered_mass = q * selected
        centered_mean = centered_mass - p * centered_mass.sum(-1, keepdim=True)
        torch.testing.assert_close(centered_mean, exact, atol=1e-12, rtol=1e-12)
        variance = (q * selected.square() * score_squared).sum() - exact.square().sum()
        sampled = torch.zeros_like(p).scatter_add(-1, draws, selected.gather(-1, draws) / draws.shape[-1])
        sampled -= p * selected.gather(-1, draws).mean(-1, keepdim=True)
        results.append(
            {
                "policy_score_control": control,
                "analytic_eight_draw_rmse_l2": float((variance.clamp_min(0) / 8).sqrt()),
                "captured_eight_draw_l1_error": float((sampled - exact).abs().sum()),
                "captured_eight_draw_l2_error": float((sampled - exact).norm()),
                "enumerated_expectation_max_abs_error": float((centered_mean - exact).abs().max()),
            }
        )
    return {
        "dense_correction_l1": float(exact.abs().sum()),
        "dense_correction_l2": float(exact.norm()),
        "expected_ordinary_logit_gradient_l1": ordinary,
        "sampled8": results,
    }


def analyze_prefix(record, cap, index):
    arrays = arrays_from(record["arrays"])
    original_p = torch.from_numpy(arrays["current_logprobs"].astype(np.float64))[None]
    original_q = torch.from_numpy(arrays["behavior_logprobs"].astype(np.float64))[None]
    assert torch.isfinite(original_p).all() and torch.isfinite(original_q).all()
    old, behavior = original_p.log_softmax(-1), original_q.log_softmax(-1)
    draws = torch.from_numpy(arrays["iid_indices"].astype(np.int64))[None]
    np.testing.assert_allclose(
        arrays["behavior_logprobs"][arrays["iid_indices"]], arrays["iid_logprobs"], atol=1e-5, rtol=1e-5
    )
    generator = torch.Generator().manual_seed(97 + index)
    movement = old + torch.randn(old.shape, dtype=old.dtype, generator=generator) * 0.2
    cases = []
    for label, current in (("observed_pre_update", old), ("synthetic_logit_movement_sd_0.2", movement)):
        for sign in (1.0, -1.0):
            advantage = torch.full((1, 1), sign, dtype=old.dtype)
            exact = estimator_evidence(current, old, behavior, advantage, draws, cap)
            candidates = [candidate_comparison(current, old, behavior, advantage, k, cap=cap) for k in (32, 128)]
            for result in candidates:
                denominator = exact["expected_ordinary_logit_gradient_l1"]
                result["error_over_expected_ordinary_l1"] = (
                    result["gradient_l1_error_mean"] / denominator if denominator else None
                )
            cases.append({"case": label, "unit_advantage": sign, **exact, "candidates": candidates})
    filtered = []
    for item in record["filters"]:
        q_log = torch.from_numpy(arrays_from(item["arrays"])["behavior_logprobs"].astype(np.float64))
        support = torch.isfinite(q_log)
        assert int(support.sum()) == item["support"]
        expected_q = behavior[0, support].log_softmax(-1)
        got_q = q_log[support].log_softmax(-1)
        p = old[:, support].log_softmax(-1)
        reference = candidate_comparison(
            p, p, got_q[None], torch.ones((1, 1), dtype=p.dtype), int(support.sum()), cap=cap
        )
        filtered.append(
            {
                "sampling": item["sampling"],
                "support": int(support.sum()),
                "original_probability_mass": float(q_log.exp().sum()),
                "q_on_recorded_support_l1_error": float((got_q.exp() - expected_q.exp()).abs().sum()),
                "logical_ids_and_scores_bytes_per_token": int(support.sum()) * 8,
                "conditional_full_support_correction": reference,
            }
        )
    return {
        **{key: value for key, value in record.items() if key not in {"arrays", "filters"}},
        "vocabulary": old.shape[-1],
        "p_original_probability_mass": float(original_p.exp().sum()),
        "q_original_probability_mass": float(original_q.exp().sum()),
        "cases": cases,
        "filters": filtered,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tis-cap", type=float, default=3.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = json.loads(read_bytes(args.root.rstrip("/") + "/dense-prefixes.json"))
    result = {
        "source_root": args.root,
        "tis_cap": args.tis_cap,
        "prefixes": [analyze_prefix(record, args.tis_cap, index) for index, record in enumerate(records)],
        "limits": [
            "q is recaptured at the exact generated prefix with fixed weights; it is not the original decode q.",
            "The observed case uses current=old with unit positive and negative advantages.",
            "Logit movement is a synthetic sensitivity experiment, not a measured policy update.",
            "Eight-draw RMSE is the exact conditional L2 variance, not a learning-benefit prediction.",
            "Filtered references condition trainer probabilities on the recorded native sampling support.",
            "Payload sizes are logical ID/score bytes, not measured network traffic.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "prefixes": len(records)}))


if __name__ == "__main__":
    main()
