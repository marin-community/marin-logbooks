# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Muon comparisons and numerical telemetry at identical post-gradient momentum."""

import jax
import jax.numpy as jnp

from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import (
    NUCLEAR_HESSIAN_MATMUL_PRECISION,
    NuclearHessianSettings,
    _clip_finite_correction,
    _inverse_spd_newton,
    _regularized_polar_hessian,
    _solve_sylvester_fixed_point,
    _tangent_without_radial_component,
    cubic_newton_schulz,
    error_aware_muon_step,
    quintic_newton_schulz,
)

VARIANTS = ("muon", "blend_0.02", "blend_0.05", "F5_tangent_0.05", "hesscorr_0.05", "hesscorr_0.3")


def safe_norm(array, eps=1e-12):
    scale = jnp.maximum(jnp.max(jnp.abs(array)), eps)
    return scale * jnp.linalg.norm(array / scale)


def correction_telemetry(momentum, tangent, settings):
    """Measure the same regularized Sylvester solve, including zero and clipping branches."""
    m, t = momentum.astype(jnp.float32), tangent.astype(jnp.float32)
    residual_tangent, radial = _tangent_without_radial_component(m, t, eps=settings.eps)
    transpose = m.shape[0] < m.shape[1]
    if transpose:
        m, residual_tangent = m.T, residual_tangent.T
    with jax.default_matmul_precision(NUCLEAR_HESSIAN_MATMUL_PRECISION):
        polar = cubic_newton_schulz(m, steps=settings.cubic_steps, eps=settings.eps)
        hessian = _regularized_polar_hessian(polar, m, eps=settings.eps)
        skew = polar.T @ residual_tangent - residual_tangent.T @ polar
        solution = _solve_sylvester_fixed_point(hessian, skew, steps=settings.sylvester_steps, eps=settings.eps)
        inverse = _inverse_spd_newton(hessian, steps=settings.inverse_steps, eps=settings.eps)
        raw = polar @ solution + (residual_tangent - polar @ (polar.T @ residual_tangent)) @ inverse
        raw_norm = safe_norm(raw, settings.eps)
        finite = jnp.all(jnp.isfinite(raw)) & jnp.isfinite(raw_norm)
        cap = jnp.sqrt(jnp.asarray(min(momentum.shape), jnp.float32))
        clipped = _clip_finite_correction(raw, cap=cap, eps=settings.eps)
        correction = jnp.where(radial, jnp.zeros_like(clipped), clipped)
        factor = jnp.where(radial | ~finite, 0.0, jnp.minimum(1.0, cap / jnp.maximum(raw_norm, settings.eps)))
        sylvester = safe_norm(hessian @ solution + solution @ hessian - skew, settings.eps)
        sylvester /= jnp.maximum(safe_norm(skew, settings.eps), settings.eps)
        inverse_error = safe_norm(hessian @ inverse - jnp.eye(len(hessian)), settings.eps) / jnp.sqrt(len(hessian))
    if transpose:
        correction = correction.T
    return correction, dict(
        pre_clipping_norm=raw_norm,
        clipping_factor=factor,
        correction_norm=safe_norm(correction, settings.eps),
        radial_zero=radial,
        nonfinite_zero=(~radial) & (~finite),
        sylvester_relative_residual=sylvester,
        inverse_relative_residual=inverse_error,
        solve_finite=finite,
    )


def matrix_candidates(momentum, gradient, settings: NuclearHessianSettings):
    """Preserve pinned base/blend/Hesscorr arithmetic; differentiate the five-step map itself."""
    tangent = gradient - momentum
    base, f5_tangent = jax.jvp(quintic_newton_schulz, (momentum,), (tangent,))
    correction, telemetry = correction_telemetry(momentum, tangent, settings)
    candidates = {
        "muon": base,
        "blend_0.02": error_aware_muon_step(momentum, gradient, policy="blend", blend_gain=0.02),
        "blend_0.05": error_aware_muon_step(momentum, gradient, policy="blend", blend_gain=0.05),
        "F5_tangent_0.05": base + 0.05 * f5_tangent,
        "hesscorr_0.05": error_aware_muon_step(
            momentum, gradient, policy="hesscorr", correction_gain=0.05, hessian_settings=settings
        ),
        "hesscorr_0.3": error_aware_muon_step(
            momentum, gradient, policy="hesscorr", correction_gain=0.3, hessian_settings=settings
        ),
    }
    denominator = jnp.maximum(safe_norm(correction) * safe_norm(f5_tangent), 1e-30)
    telemetry.update(
        f5_tangent_norm=safe_norm(f5_tangent),
        correction_tangent_cosine=jnp.sum(correction * f5_tangent) / denominator,
        correction_to_tangent_difference=safe_norm(correction - f5_tangent) / jnp.maximum(safe_norm(f5_tangent), 1e-30),
        reconstructed_hesscorr_error=safe_norm(candidates["hesscorr_0.3"] - base - 0.3 * correction),
    )
    return candidates, f5_tangent, telemetry


def f5_finite_difference(momentum, gradient, epsilon):
    tangent = gradient - momentum
    # Epsilon is relative input displacement, so scaling the whole momentum does not change the test.
    step = epsilon * safe_norm(momentum) / jnp.maximum(safe_norm(tangent), 1e-30)
    step = jnp.maximum(step, 1e-12)
    plus = quintic_newton_schulz(momentum + step * tangent)
    minus = quintic_newton_schulz(momentum - step * tangent)
    return (plus - minus) / (2 * step), step
