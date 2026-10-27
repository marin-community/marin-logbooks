# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent clipped-PPO/TIS references and alternative correction prototypes."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
from skyrl_train.objective.score_centering import ppo_tis_score_centering_correction


def coefficients(logp, old, behavior, advantage, cap=1.3):
    """Differentiate the ordinary per-action clipped loss, before expectation."""
    point = logp.detach().requires_grad_()
    ratio = (point - old).clamp(-20, 20).exp()
    weight = (old - behavior).clamp(max=math.log(cap)).exp()
    loss = -weight * torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    return -torch.autograd.grad(loss.sum(), point)[0].detach()


def sampled_correction(logp, coefficient, draws):
    """IID draws must come from the actual q at this prefix, independently of A."""
    return (coefficient.gather(-1, draws) * logp.gather(-1, draws)).mean(-1)


def logit_l1(head_probability, residual):
    """Exact full-vocabulary L1 of a detached head-only log-score surrogate."""
    total = residual.sum(-1, keepdim=True)
    head = (residual - head_probability * total).abs().sum(-1)
    tail = (1 - head_probability.sum(-1)).clamp_min(0) * total.squeeze(-1).abs()
    return head + tail


def candidate_comparison(logits, old, behavior, advantage, count, cap=1.3):
    point = logits.detach().clone().requires_grad_()
    logp = point.log_softmax(-1)
    ids = behavior.topk(count, dim=-1).indices
    head = logp.gather(-1, ids)
    loss = ppo_tis_score_centering_correction(
        head.unsqueeze(0),
        old.gather(-1, ids).unsqueeze(0),
        behavior.gather(-1, ids).unsqueeze(0),
        advantage.squeeze(-1).unsqueeze(0),
        torch.ones(1, len(logits)),
        tis_cap=cap,
        eps_clip_low=0.2,
        eps_clip_high=0.2,
    )
    residual = torch.autograd.grad(loss.sum(), head, retain_graph=True)[0]
    got = torch.autograd.grad(loss.sum(), point)[0]
    formula = logit_l1(head.detach().exp(), residual)
    torch.testing.assert_close(formula, got.abs().sum(-1), atol=1e-10, rtol=1e-10)

    reference_point = logits.detach().clone().requires_grad_()
    reference_logp = reference_point.log_softmax(-1)
    weighted = behavior.exp() * coefficients(reference_logp, old, behavior, advantage, cap=cap)
    reference_loss = (weighted * reference_logp).sum()
    expected = torch.autograd.grad(reference_loss, reference_point)[0]
    error = got - expected
    return {
        "candidates": count,
        "behavior_head_mass_mean": float(behavior.gather(-1, ids).exp().sum(-1).mean()),
        "current_head_mass_mean": float(head.detach().exp().sum(-1).mean()),
        "gradient_max_abs_error": float(error.abs().max()),
        "gradient_l1_error_mean": float(error.abs().sum(-1).mean()),
        "gradient_l2_error_mean": float(error.norm(dim=-1).mean()),
        "dense_gradient_l1_mean": float(expected.abs().sum(-1).mean()),
        "dense_gradient_l2_mean": float(expected.norm(dim=-1).mean()),
        "approximate_gradient_l1_mean": float(formula.mean()),
        "l1_formula_max_abs_error": float((formula - got.abs().sum(-1)).abs().max()),
    }


def sampled_comparison(logits, old, behavior, advantage, repetitions, generator, subtract_policy_score=False):
    point = logits.detach().clone().requires_grad_()
    logp = point.log_softmax(-1)
    coefficient = coefficients(logp, old, behavior, advantage)
    q, p = behavior.exp(), logp.detach().exp()
    expected = q * coefficient - p * (q * coefficient).sum(-1, keepdim=True)
    if subtract_policy_score:
        # E_q[(p/q) grad log p] = 0 on a complete, positive q support.
        coefficient = coefficient - advantage * p / q
    # Enumerating every possible one-draw gradient proves the estimator's mean.
    score = torch.eye(logits.shape[-1], dtype=logits.dtype)[None] - p[:, None]
    draw_gradients = coefficient[..., None] * score
    enumerated = (q[..., None] * draw_gradients).sum(1)
    torch.testing.assert_close(enumerated, expected, atol=1e-12, rtol=1e-12)

    draws = torch.multinomial(q, 8, replacement=True, generator=generator)
    surrogate = sampled_correction(logp, coefficient, draws)
    actual = torch.autograd.grad(surrogate.sum(), point)[0]
    direct = torch.zeros_like(p).scatter_add_(-1, draws, coefficient.gather(-1, draws) / 8)
    direct -= p * coefficient.gather(-1, draws).mean(-1, keepdim=True)
    torch.testing.assert_close(actual, direct, atol=1e-12, rtol=1e-12)
    variance = (q * draw_gradients.square().sum(-1)).sum(-1) - expected.square().sum(-1)

    estimates = []
    for row in range(len(q)):
        sampled = torch.multinomial(q[row], repetitions * 8, replacement=True, generator=generator).view(-1, 8)
        estimate = torch.zeros(repetitions, p.shape[-1], dtype=p.dtype)
        estimate.scatter_add_(-1, sampled, coefficient[row][sampled] / 8)
        estimate -= p[row] * coefficient[row][sampled].mean(-1, keepdim=True)
        estimates.append(
            {
                "row": row,
                "mean_gradient_max_abs_error": float((estimate.mean(0) - expected[row]).abs().max()),
                "empirical_rmse_l2": float((estimate - expected[row]).square().sum(-1).mean().sqrt()),
                "analytic_rmse_l2": float((variance[row].clamp_min(0) / 8).sqrt()),
                "dense_correction_l2": float(expected[row].norm()),
            }
        )
    return {
        "draws_per_prefix": 8,
        "subtract_policy_score": subtract_policy_score,
        "repetitions": repetitions,
        "enumerated_mean_max_abs_error": float((enumerated - expected).abs().max()),
        "sampled_surrogate_gradient_max_abs_error": float((actual - direct).abs().max()),
        "draws": draws.tolist(),
        "rows": estimates,
    }


def filter_support(logits, top_k, top_p):
    """Reference the pinned sampler's top-k-then-top-p order on untied logits."""
    values, ids = logits.sort(descending=True, stable=True)
    values = values[:top_k]
    before = torch.cat((values.new_zeros(1), values.softmax(-1).cumsum(-1)[:-1]))
    return ids[:top_k][before < top_p]


def filter_replay_comparison(logits, old_logits, behavior_logits, advantage, top_k, top_p):
    ids = filter_support(behavior_logits, top_k, top_p)
    p_point = logits.detach().clone().requires_grad_()
    # Replay the recorded behavior support; do not select a new trainer support.
    logp = p_point[ids].log_softmax(-1)
    old = old_logits[ids].log_softmax(-1)
    behavior = behavior_logits[ids].log_softmax(-1)
    correction = ppo_tis_score_centering_correction(
        logp[None, None],
        old[None, None],
        behavior[None, None],
        torch.tensor([[advantage]], dtype=logits.dtype),
        torch.ones(1, 1),
        tis_cap=1.3,
        eps_clip_low=0.2,
        eps_clip_high=0.2,
    )
    actual = torch.autograd.grad(correction.sum(), p_point)[0]
    reference_point = logits.detach().clone().requires_grad_()
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    allowed[ids] = True
    dense = reference_point.masked_fill(~allowed, -torch.inf).log_softmax(-1)
    coefficient = coefficients(dense[ids], old, behavior, advantage)
    expected = torch.autograd.grad((behavior.exp() * coefficient * dense[ids]).sum(), reference_point)[0]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    return {
        "sampling_top_k": top_k,
        "sampling_top_p": top_p,
        "support_size": len(ids),
        "gradient_max_abs_error": float((actual - expected).abs().max()),
        "off_support_gradient_max_abs": float(actual[~allowed].abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=20000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(97)
    n, vocabulary = 6, 257
    logits = torch.randn(n, vocabulary, generator=generator, dtype=torch.float64) * 3
    old = (logits + torch.randn(n, vocabulary, generator=generator, dtype=logits.dtype) * 0.6).log_softmax(-1)
    behavior = (logits + torch.randn(n, vocabulary, generator=generator, dtype=logits.dtype) * 1.2).log_softmax(-1)
    advantage = torch.tensor([1.7, -2.3, 0, 1.7, -2.3, 0], dtype=logits.dtype)[:, None]
    cases = []
    for label, old_logp in (("pre_update", logits.log_softmax(-1)), ("policy_movement", old)):
        cases.append(
            {
                "label": label,
                "top_k": [candidate_comparison(logits, old_logp, behavior, advantage, k) for k in (32, 128)],
                "sampled8": sampled_comparison(logits, old_logp, behavior, advantage, args.repetitions, generator),
                "sampled8_policy_control": sampled_comparison(
                    logits, old_logp, behavior, advantage, args.repetitions, generator, subtract_policy_score=True
                ),
            }
        )
    filters = [
        filter_replay_comparison(logits[0], old[0], behavior[0], a, k, p)
        for a in (1.7, -2.3, 0)
        for k, p in ((32, 1.0), (32, 0.9), (128, 0.95), (257, 0.9))
    ]
    root = Path(__file__).resolve().parents[2]
    result = {
        "source_revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "dtype": str(logits.dtype),
        "shape": [n, vocabulary],
        "tis_cap": 1.3,
        "ppo_clip": [0.8, 1.2],
        "cases": cases,
        "filter_replay": filters,
        "limits": [
            "Synthetic numerical references establish formulas, not model accuracy or performance.",
            "Top-K and dense loss values have different zero-gradient offsets; compare gradients.",
            "Sampled8 is unbiased only with independent draws from the actual prefix distribution.",
            "Clipping/TIS give unequal coefficients: subtracting eight draws need not cancel the dense softmax term.",
            "Filter selection here uses untied logits; production support must come from the sampler.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases), "filter_cases": len(filters)}))


if __name__ == "__main__":
    main()
