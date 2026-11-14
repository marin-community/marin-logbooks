# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded quadratic choices and projected matrix-envelope calculations."""

import numpy as np
from scipy.fft import dct


def quadratic_step(alignment, curvature, radius):
    """Maximize a*s-k*s²/2 on [0,radius], including convex and flat cases."""
    candidates = [0.0, float(radius)]
    if curvature > 0:
        candidates.append(float(np.clip(alignment / curvature, 0, radius)))
    values = [alignment * s - 0.5 * curvature * s * s for s in candidates]
    index = min(range(len(candidates)), key=lambda i: (-values[i], candidates[i]))
    return candidates[index], float(values[index])


def refinement_lengths(lengths, losses):
    best = min(range(len(lengths)), key=lambda i: (losses[i], lengths[i]))
    if best in (0, len(lengths) - 1):
        return [], True
    return [(lengths[best - 1] + lengths[best]) / 2, (lengths[best] + lengths[best + 1]) / 2], False


def orthonormal_matrices(candidates, count):
    """Twice-reorthogonalized Frobenius basis, preserving the requested span."""
    basis = []
    accepted = []
    for name, candidate in candidates:
        v = np.asarray(candidate, np.float64).copy()
        original = np.linalg.norm(v)
        for _ in range(2):
            for q in basis:
                v -= np.sum(q * v) * q
        norm = np.linalg.norm(v)
        if norm <= 1e-7 * max(original, 1e-30):
            continue
        basis.append(v / norm)
        accepted.append(name)
        if len(basis) == count:
            return np.stack(basis).astype(np.float32), accepted
    raise ValueError("Insufficient independent matrix probes")


def rotated_gradient(gradient, seed):
    """Independent dense orthogonal DCT/sign/permutation rotations of both axes."""
    rng = np.random.default_rng(seed)
    g = np.asarray(gradient, np.float64)
    g = g[rng.permutation(g.shape[0])][:, rng.permutation(g.shape[1])]
    g = rng.choice([-1.0, 1.0], g.shape[0])[:, None] * g
    g = g * rng.choice([-1.0, 1.0], g.shape[1])[None, :]
    return dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")


def envelope_components(gradient, basis):
    """Project I, E↦GGᵀE, E↦EGᵀG without a full Kronecker matrix."""
    g, v = np.asarray(gradient, np.float64), np.asarray(basis, np.float64)
    # Frobenius Gram identities avoid explicitly forming the large left Gram.
    left = np.stack([(g.T @ e).ravel() for e in v])
    right = np.stack([(e @ g.T).ravel() for e in v])
    flat = v.reshape(len(v), -1)
    return np.stack([flat @ flat.T, left @ left.T, right @ right.T])


def normalized_hessian_norm(hessian, envelope):
    values, vectors = np.linalg.eigh((envelope + envelope.T) / 2)
    if values.min() <= 0:
        raise ValueError("Envelope must be positive definite")
    inverse_root = (vectors * (values**-0.5)) @ vectors.T
    relative = inverse_root @ ((hessian + hessian.T) / 2) @ inverse_root
    return float(np.max(np.abs(np.linalg.eigvalsh(relative))))
