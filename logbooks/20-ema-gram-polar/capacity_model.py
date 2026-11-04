# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static capacity planning; no model import, array kernels, or compute submission."""

import hashlib
import json
from pathlib import Path


def calculate():
    step = 4.0  # planning assumption from early matrix training, not steady measurement
    baseline_step = 0.35  # historical four-v4 alpha=0; not a same-hardware forecast
    pilot_steps, confirmation_steps, scale_steps = 7630, 37335, 38014
    baseline_remaining = 5 * pilot_steps + 3 * confirmation_steps
    nonzero_low = 24 * pilot_steps + 6 * confirmation_steps
    nonzero_high = 24 * pilot_steps + 9 * confirmation_steps
    fixed_hours = baseline_remaining * baseline_step / 3600
    speedable_hours = [n * step / 3600 for n in (nonzero_low, nonzero_high)]
    core_hours = [h + fixed_hours for h in speedable_hours]
    result = {
        "sources": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "experiments/ema_gram/README.md",
                "experiments/ema_gram/plan.py",
                "experiments/references/train_delphi_seeds.py",
                "lib/levanter/src/levanter/optim/ema_gram.py",
                "logbooks/20-ema-gram-polar/prefix-train-first-updates.json",
            )
        },
        "planning_assumptions": {
            "nonzero_step_seconds": step,
            "baseline_step_seconds": baseline_step,
            "measurement_scope": "Early alpha=.25 matrix/default, not steady nor all-alpha/large-model measurement",
            "completed_credit": "One completed alpha0 pilot credited; current prefix not yet accepted/reusable",
            "excluded": [
                "queue",
                "compilation",
                "evaluation",
                "checkpoint/hooks",
                "retries",
                "probes/SVD",
                "LR edge expansion",
            ],
            "concurrency": "Serial estimate; hypothetical parallelism does not authorize concurrent candidates",
        },
        "scope": {
            "initial_pilots": 30,
            "remaining_pilots": 29,
            "pilot_steps": pilot_steps,
            "confirmation_runs": [9, 12],
            "confirmation_steps": confirmation_steps,
            "confirmation_selection": "alpha0,.25,.5 plus best if .125 or1/3, each three fresh seeds",
            "conditional_horizon_generated_runs": 18,
            "conditional_horizon_new_if_matching_h100_confirmations_reused": 12,
            "conditional_scale_runs": 6,
            "scale_steps": scale_steps,
            "scope_limit": "Conditional phases need reproducibility gate; LR edge expansion has no fixed count",
        },
        "serial_compute": {
            "one_pilot_hours": pilot_steps * step / 3600,
            "one_confirmation_hours": confirmation_steps * step / 3600,
            "remaining_pilots_hours": (5 * pilot_steps * baseline_step + 24 * pilot_steps * step) / 3600,
            "baseline_remaining_hours": fixed_hours,
            "nonzero_steps": [nonzero_low, nonzero_high],
            "nonzero_hours": speedable_hours,
            "core_hours": core_hours,
            "core_days": [h / 24 for h in core_hours],
            "equal_four_seconds_all_remaining_core_days": [
                (baseline_remaining + n) * step / 86400 for n in (nonzero_low, nonzero_high)
            ],
            "horizon_12_new_days": 12 * confirmation_steps * step / 86400,
            "horizon_18_fresh_days": 18 * confirmation_steps * step / 86400,
            "scale_days_by_seconds_per_step": {str(s): 6 * scale_steps * s / 86400 for s in (10, 20, 40)},
        },
        "break_even": [
            {
                "k": k,
                "accelerated_nonzero_step_seconds": step / k,
                "core_remaining_days_after_optimization": [(h / k + fixed_hours) / 24 for h in speedable_hours],
                "max_blocking_engineering_days_serial": [h * (1 - 1 / k) / 24 for h in speedable_hours],
                "max_days_p4_success_probability_half": [0.5 * h * (1 - 1 / k) / (24 * 4) for h in speedable_hours],
            }
            for k in (1.25, 1.5, 2, 3, 4, 10)
        ],
        "formulas": {
            "blocking": "D + Cvalidate < p * T_speedable * (1-1/k); unaffected overhead cancels",
            "parallel": "Replace T_speedable with actual critical-path reducible duration; T/P is an ideal lower bound",
            "concurrent_engineering": (
                "Independent engineering: saved=max(T-D-C,0)*(1-1/k); D+C is deployment delay; contention extra"
            ),
            "amdahl": "whole_step_speedup=1/((1-f)+f/k_component)",
        },
        "geometry": {
            "layers": 11,
            "large_gram_count": 33,
            "large_gram_dimension": 4096,
            "small_gram_count": 121,
            "small_gram_dimension": 1024,
            "large_share_of_n_cubed_eigensolver_proxy": 3 * 4**3 / (3 * 4**3 + 11),
            "matrix_batch_padding_useful_fraction": 11 / 16,
            "scale_eigen_n_cubed_proxy_ratio": (16 / 11) * (1536 / 1024) ** 3,
            "scope": "Dense arithmetic proxy only, not measured kernel fractions or attainable speedup",
            "model_flops_per_update": 80353469399040,
            "candidate_peak_flops": 4.4e15,
            "model_only_100percent_peak_seconds": 80353469399040 / 4.4e15,
            "five_percent_mfu_seconds": 80353469399040 / 4.4e15 / 0.05,
            "speedup_from_four_seconds_for_five_percent_mfu": 4 / (80353469399040 / 4.4e15 / 0.05),
            "mfu_limit": (
                "Model FLOPs exclude expensive optimizer arithmetic; low MFU is not a direct idle-device fraction"
            ),
        },
    }
    assert baseline_remaining == 150155
    assert (nonzero_low, nonzero_high) == (407130, 519135)
    assert 24 + 5 == 29 and 6 + 3 == 9 and 9 + 3 == 12
    assert 33 + 121 == 11 * 7 * 2
    for row in result["break_even"]:
        for i in range(2):
            assert (
                abs(
                    row["core_remaining_days_after_optimization"][i]
                    + row["max_blocking_engineering_days_serial"][i]
                    - core_hours[i] / 24
                )
                < 1e-10
            )
    return result


if __name__ == "__main__":
    print(json.dumps(calculate(), indent=2))
