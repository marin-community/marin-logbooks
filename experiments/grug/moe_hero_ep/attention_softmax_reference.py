# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU FP64 attention references for sparse-query fused-gradient diagnostics."""

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_hero_ep.softmax_collapse_metrics import FP32_MARGIN_CLIFF, MARGIN_HISTOGRAM_EDGES

MARGIN_EVENT_COUNT = 4


@dataclass(frozen=True)
class AttentionReference:
    output: np.ndarray
    dq: np.ndarray
    dk: np.ndarray
    dv: np.ndarray
    probabilities: np.ndarray
    score_gradients: np.ndarray
    valid_key_count: np.ndarray
    winner_key: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray
    pmax: np.ndarray
    tail_mass: np.ndarray
    log_normalizer: np.ndarray
    winner_score_gradient: np.ndarray


@dataclass(frozen=True)
class StoredOutputControl:
    dq: np.ndarray
    dk: np.ndarray
    score_gradients: np.ndarray
    correction: np.ndarray
    winner_dp: np.ndarray


def attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dout: np.ndarray,
    key_mask: np.ndarray,
    *,
    softmax_scale: float,
) -> AttentionReference:
    """Compute exact-input FP64 attention and gradients for selected query rows.

    Inputs use BQHD/BKHD layout. ``key_mask[B, Q, K]`` defines all allowed
    keys, including original causal positions, segments, and window bounds.
    Q may contain only selected queries; dK/dV aggregate only their supplied
    cotangents, including all query heads sharing each KV head.

    The score gradient is centered on the winning value vector before its
    dot product with dO. This avoids subtracting nearly equal dP and dO·O.
    Margins for rows with fewer than two valid keys use the sentinel zero;
    ``valid_key_count`` identifies those rows explicitly.
    """
    q, k, v, dout = [np.asarray(array, dtype=np.float64) for array in (q, k, v, dout)]
    key_mask = np.asarray(key_mask, dtype=bool)
    if any(array.ndim != 4 for array in (q, k, v, dout)):
        raise ValueError("q, k, v and dout must have BQHD/BKHD shape")
    batch_size, query_count, query_heads, width = q.shape
    key_batch, key_count, kv_heads, key_width = k.shape
    if key_batch != batch_size or key_width != width or query_heads % kv_heads:
        raise ValueError("Q/K batch, head width, or grouped-head layout mismatch")
    if v.shape[:3] != k.shape[:3] or dout.shape != (*q.shape[:3], v.shape[-1]):
        raise ValueError("V/dO shape mismatch")
    if key_mask.shape != (batch_size, query_count, key_count):
        raise ValueError("key_mask must have BQK shape")
    if not all(np.isfinite(array).all() for array in (q, k, v, dout)):
        raise ValueError("Reference inputs must be finite")
    if not math.isfinite(softmax_scale):
        raise ValueError("softmax_scale must be finite")

    output = np.zeros_like(dout)
    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dv = np.zeros_like(v)
    row_shape = (batch_size, query_count, query_heads)
    probabilities = np.zeros((*row_shape, key_count), dtype=np.float64)
    score_gradients = np.zeros_like(probabilities)
    valid_key_count = np.broadcast_to(key_mask.sum(axis=-1)[:, :, None], row_shape).copy()
    winner_key = np.full(row_shape, -1, dtype=np.int64)
    margin, entropy, pmax, tail_mass, log_normalizer, winner_score_gradient = [
        np.zeros(row_shape, dtype=np.float64) for _ in range(6)
    ]
    heads_per_kv = query_heads // kv_heads

    for batch in range(batch_size):
        for query in range(query_count):
            keys = np.flatnonzero(key_mask[batch, query])
            if not len(keys):
                continue
            for head in range(query_heads):
                kv_head = head // heads_per_kv
                key_vectors = k[batch, keys, kv_head]
                values = v[batch, keys, kv_head]
                scores = (key_vectors @ q[batch, query, head]) * softmax_scale
                winner = int(np.argmax(scores))
                shifted = scores - scores[winner]
                tail_exponentials = np.exp(shifted)
                tail_exponentials[winner] = 0
                tail = float(tail_exponentials.sum())
                log_denom = np.log1p(tail)
                probs = tail_exponentials / (1 + tail)
                probs[winner] = 1 / (1 + tail)
                # Value differences precede the dot product and the weighted sum;
                # a tiny winning gradient survives even when output rounds to Vmax.
                differences = values - values[winner]
                centered_dp = differences @ dout[batch, query, head]
                correction = float(probs @ centered_dp)
                ds = probs * (centered_dp - correction)
                ds[winner] = -probs[winner] * correction

                output[batch, query, head] = values[winner] + probs @ differences
                dq[batch, query, head] = (ds @ key_vectors) * softmax_scale
                dk[batch, keys, kv_head] += ds[:, None] * q[batch, query, head] * softmax_scale
                dv[batch, keys, kv_head] += probs[:, None] * dout[batch, query, head]
                probabilities[batch, query, head, keys] = probs
                score_gradients[batch, query, head, keys] = ds
                winner_key[batch, query, head] = keys[winner]
                pmax[batch, query, head] = probs[winner]
                tail_mass[batch, query, head] = tail / (1 + tail)
                log_normalizer[batch, query, head] = log_denom
                winner_score_gradient[batch, query, head] = ds[winner]
                entropy[batch, query, head] = log_denom - float(probs @ shifted)
                if len(keys) > 1:
                    other_scores = np.delete(scores, winner)
                    other_max = float(other_scores.max())
                    margin[batch, query, head] = (
                        scores[winner] - other_max - np.log(np.exp(other_scores - other_max).sum())
                    )

    return AttentionReference(
        output,
        dq,
        dk,
        dv,
        probabilities,
        score_gradients,
        valid_key_count,
        winner_key,
        margin,
        entropy,
        pmax,
        tail_mass,
        log_normalizer,
        winner_score_gradient,
    )


def _gradient_error_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int | None]:
    actual = np.asarray(actual, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"Gradient shape mismatch: actual={actual.shape}, reference={reference.shape}")
    if not np.isfinite(actual).all():
        raise ValueError("Fused gradients must be finite")
    error = actual - reference
    actual_squared = float(np.sum(actual**2))
    reference_squared = float(np.sum(reference**2))
    error_squared = float(np.sum(error**2))
    actual_row_squared = np.sum(actual**2, axis=-1)
    reference_row_squared = np.sum(reference**2, axis=-1)
    return {
        "component_count": int(reference.size),
        "actual_squared_norm": actual_squared,
        "reference_squared_norm": reference_squared,
        "error_squared_norm": error_squared,
        "actual_l2": math.sqrt(actual_squared),
        "reference_l2": math.sqrt(reference_squared),
        "relative_l2_error": math.sqrt(error_squared / reference_squared) if reference_squared else None,
        "cosine_similarity": (
            float(np.sum(actual * reference)) / math.sqrt(actual_squared * reference_squared)
            if actual_squared and reference_squared
            else None
        ),
        "absolute_error_sum": float(np.abs(error).sum()),
        "mean_absolute_error": float(np.abs(error).mean()),
        "max_absolute_error": float(np.abs(error).max()),
        "reference_nonzero_components": int(np.count_nonzero(reference)),
        "zero_actual_nonzero_reference_components": int(np.count_nonzero((actual == 0) & (reference != 0))),
        "reference_nonzero_vectors": int(np.count_nonzero(reference_row_squared)),
        "zero_actual_nonzero_reference_vectors": int(
            np.count_nonzero((actual_row_squared == 0) & (reference_row_squared != 0))
        ),
    }


def stored_output_gradient_control(
    reference: AttentionReference,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dout: np.ndarray,
    stored_output: np.ndarray,
    *,
    softmax_scale: float,
) -> StoredOutputControl:
    """Reconstruct dQ/dK using FP64 probabilities and FP32 dP/output correction.

    This reproduces the stored-output correction structure, but not the
    kernel's exact GEMM accumulation order, probability reconstruction, BF16
    dS cast, or final gradient projection. It is an attribution control.
    """
    q, k, v, dout, stored_output = [np.asarray(array, dtype=np.float64) for array in (q, k, v, dout, stored_output)]
    dq, dk = np.zeros_like(reference.dq), np.zeros_like(reference.dk)
    ds_all = np.zeros_like(reference.score_gradients)
    corrections = np.zeros_like(reference.pmax)
    winner_dp = np.zeros_like(reference.pmax)
    batch_size, query_count, query_heads = reference.valid_key_count.shape
    heads_per_kv = query_heads // k.shape[2]
    for batch in range(batch_size):
        for query in range(query_count):
            for head in range(query_heads):
                winner = reference.winner_key[batch, query, head]
                if winner < 0:
                    continue
                kv_head = head // heads_per_kv
                upstream = dout[batch, query, head].astype(np.float32)
                dp = v[batch, :, kv_head].astype(np.float32) @ upstream
                correction = np.sum(stored_output[batch, query, head].astype(np.float32) * upstream, dtype=np.float32)
                ds = reference.probabilities[batch, query, head] * (dp - correction).astype(np.float64)
                ds_all[batch, query, head] = ds
                corrections[batch, query, head] = correction
                winner_dp[batch, query, head] = dp[winner]
                dq[batch, query, head] = (ds @ k[batch, :, kv_head]) * softmax_scale
                dk[batch, :, kv_head] += ds[:, None] * q[batch, query, head] * softmax_scale
    return StoredOutputControl(dq, dk, ds_all, corrections, winner_dp)


def _control_error_alignment(actual: np.ndarray, control: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    actual_error = np.asarray(actual, dtype=np.float64) - reference
    control_error = control - reference
    actual_squared = float(np.sum(actual_error**2))
    control_squared = float(np.sum(control_error**2))
    residual_squared = float(np.sum((actual_error - control_error) ** 2))
    return {
        "cosine_error_alignment": (
            float(np.sum(actual_error * control_error)) / math.sqrt(actual_squared * control_squared)
            if actual_squared and control_squared
            else None
        ),
        "residual_error_l2_over_fused_error_l2": (
            math.sqrt(residual_squared / actual_squared) if actual_squared else None
        ),
    }


def compare_attention_gradients(
    reference: AttentionReference,
    fused_output: np.ndarray,
    fused_dq: np.ndarray,
    fused_dk: np.ndarray,
    *,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dout: np.ndarray,
    softmax_scale: float,
    fused_dv: np.ndarray | None = None,
    event_count: int = 8,
) -> dict:
    """Compare fused gradients and reconstruct cancellation from its stored output.

    Reconstructed score gradients use FP64 reference probabilities with
    FP32 dP and dO·O. They are explicitly distinct from the fused kernel's
    internal score gradient. Controls use the actual fused output and the
    mathematical reference output rounded to BF16, respectively.
    All reported gradients precede any downstream parameter-gradient update.
    """
    fused_output, fused_dq, fused_dk, v, dout = [
        np.asarray(array, dtype=np.float64) for array in (fused_output, fused_dq, fused_dk, v, dout)
    ]
    if fused_output.shape != reference.output.shape or dout.shape != reference.output.shape:
        raise ValueError("Fused output/dO shape mismatch")
    if not np.isfinite(fused_output).all():
        raise ValueError("Fused output must be finite")
    actual_control = stored_output_gradient_control(reference, q, k, v, dout, fused_output, softmax_scale=softmax_scale)
    rounded_output = np.asarray(reference.output.astype(jnp.bfloat16), dtype=np.float64)
    rounded_control = stored_output_gradient_control(
        reference, q, k, v, dout, rounded_output, softmax_scale=softmax_scale
    )
    winner_indices = np.maximum(reference.winner_key, 0)[..., None]
    reconstructed_winner = np.take_along_axis(actual_control.score_gradients, winner_indices, axis=-1)[..., 0]
    rounded_winner = np.take_along_axis(rounded_control.score_gradients, winner_indices, axis=-1)[..., 0]
    score_error_max = np.max(np.abs(actual_control.score_gradients - reference.score_gradients), axis=-1)
    reconstructed_score_sum = actual_control.score_gradients.sum(axis=-1)

    multi_key = reference.valid_key_count > 1
    dq_error_norm = np.linalg.norm(fused_dq - reference.dq, axis=-1)
    dq_reference_norm = np.linalg.norm(reference.dq, axis=-1)
    row_indices = np.flatnonzero(multi_key)
    worst = row_indices[np.argsort(dq_error_norm.reshape(-1)[row_indices])[::-1]][:event_count]
    cancellation_indices = np.flatnonzero(
        multi_key & (reconstructed_winner == 0) & (reference.winner_score_gradient != 0)
    )
    worst_cancellations = cancellation_indices[
        np.argsort(np.abs(reference.winner_score_gradient).reshape(-1)[cancellation_indices])[::-1]
    ][:event_count]
    largest_margins = row_indices[np.argsort(reference.margin.reshape(-1)[row_indices])[::-1]][:MARGIN_EVENT_COUNT]

    def event(flat_index):
        batch, query, head = [int(i) for i in np.unravel_index(flat_index, multi_key.shape)]
        index = (batch, query, head)
        return {
            "batch": batch,
            "query": query,
            "head": head,
            "valid_key_count": int(reference.valid_key_count[index]),
            "winner_key": int(reference.winner_key[index]),
            "margin": float(reference.margin[index]),
            "entropy": float(reference.entropy[index]),
            "pmax": float(reference.pmax[index]),
            "tail_mass": float(reference.tail_mass[index]),
            "shifted_log_normalizer": float(reference.log_normalizer[index]),
            "reference_winner_score_gradient": float(reference.winner_score_gradient[index]),
            "stored_output_reconstructed_winner_score_gradient": float(reconstructed_winner[index]),
            "rounded_reference_output_reconstructed_winner_score_gradient": float(rounded_winner[index]),
            "stored_output_fp32_correction_dO_dot_O": float(actual_control.correction[index]),
            "rounded_reference_output_fp32_correction_dO_dot_O": float(rounded_control.correction[index]),
            "fp32_winner_dP": float(actual_control.winner_dp[index]),
            "stored_output_reconstructed_score_gradient_sum": float(reconstructed_score_sum[index]),
            "stored_output_max_absolute_score_gradient_error": float(score_error_max[index]),
            "dq_error_l2": float(dq_error_norm[index]),
            "reference_dq_l2": float(dq_reference_norm[index]),
            "fused_dq_l2": float(np.linalg.norm(fused_dq[index])),
            "stored_output_control_dq_l2": float(np.linalg.norm(actual_control.dq[index])),
            "rounded_reference_output_control_dq_l2": float(np.linalg.norm(rounded_control.dq[index])),
            "fused_output_l2": float(np.linalg.norm(fused_output[index])),
            "reference_output_l2": float(np.linalg.norm(reference.output[index])),
            "fused_output_max_absolute_error": float(np.abs(fused_output[index] - reference.output[index]).max()),
            "reference_dO_dot_O": float(np.sum(reference.output[index] * dout[index])),
            "dq_relative_l2_error": (
                float(dq_error_norm[index] / dq_reference_norm[index]) if dq_reference_norm[index] else None
            ),
        }

    result = {
        "dq": _gradient_error_stats(fused_dq, reference.dq),
        "dk": _gradient_error_stats(fused_dk, reference.dk),
        "output": _gradient_error_stats(fused_output, reference.output),
        "rows": {
            "masked_count": int(np.count_nonzero(reference.valid_key_count == 0)),
            "single_key_count": int(np.count_nonzero(reference.valid_key_count == 1)),
            "multi_key_count": int(np.count_nonzero(multi_key)),
            "margin_above_fp32_cliff_count": int(np.count_nonzero(multi_key & (reference.margin > FP32_MARGIN_CLIFF))),
            "margin_histogram": np.histogram(reference.margin[multi_key], bins=MARGIN_HISTOGRAM_EDGES)[0].tolist(),
            "margin_histogram_edges": [str(edge) for edge in MARGIN_HISTOGRAM_EDGES],
            "entropy_sum": float(reference.entropy[multi_key].sum()),
            "tail_mass_sum": float(reference.tail_mass[multi_key].sum()),
            "stored_output_zero_winner_nonzero_reference_count": int(
                np.count_nonzero(multi_key & (reconstructed_winner == 0) & (reference.winner_score_gradient != 0))
            ),
            "stored_output_max_absolute_score_gradient_error": (
                float(score_error_max[multi_key].max()) if np.any(multi_key) else 0.0
            ),
            "stored_output_absolute_score_gradient_imbalance_sum": float(
                np.abs(reconstructed_score_sum[multi_key]).sum()
            ),
        },
        "worst_dq_events": [event(index) for index in worst],
        "worst_cancellation_events": [event(index) for index in worst_cancellations],
        "largest_margin_events": [event(index) for index in largest_margins],
        "controls": {},
        "interpretation": (
            "FP64 CPU mathematical gradients on identical quantized inputs/cotangents. "
            "Controls use FP64 reference probabilities with NumPy FP32 dP and dO·O; "
            "they do not reproduce fused GEMM order, probability reconstruction, BF16 dS or gradient casts. "
            "The dK/dV reference includes only the selected query cotangents and sums grouped heads."
        ),
    }
    for name, control in (("actual_stored_output", actual_control), ("bf16_rounded_reference_output", rounded_control)):
        control_winner = np.take_along_axis(control.score_gradients, winner_indices, axis=-1)[..., 0]
        result["controls"][name] = {
            "dq": _gradient_error_stats(control.dq, reference.dq),
            "dk": _gradient_error_stats(control.dk, reference.dk),
            "dq_fused_error_alignment": _control_error_alignment(fused_dq, control.dq, reference.dq),
            "dk_fused_error_alignment": _control_error_alignment(fused_dk, control.dk, reference.dk),
            "zero_winner_nonzero_reference_count": int(
                np.count_nonzero(multi_key & (control_winner == 0) & (reference.winner_score_gradient != 0))
            ),
            "max_absolute_score_gradient_error": float(
                np.abs(control.score_gradients - reference.score_gradients).max()
            ),
        }
    if fused_dv is not None:
        result["dv"] = _gradient_error_stats(fused_dv, reference.dv)
    return result
