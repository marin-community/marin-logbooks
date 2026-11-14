# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conditional fixed-anchor path measurements with already frozen coefficients."""

import hashlib
import io
import json
import time

import jax.numpy as jnp
import numpy as np
from google.cloud import storage

from experiments.ema_gram.geometry_math import envelope_components, normalized_hessian_norm, rotated_gradient
from experiments.ema_gram.hessian import FD_STEPS, host_array


def load_path_baseline(args, plan, label, identities):
    inputs = json.loads(args.path_inputs.read_text())
    if inputs["plan_sha256"] != hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest():
        raise ValueError("Path sources use a different frozen coefficient plan")
    source = inputs["blocks"][args.state][label]
    if source["identities"] != identities:
        raise ValueError("Path baseline basis identity mismatch")
    bucket, key = source["arrays"]["uri"].removeprefix("gs://").split("/", 1)
    data = storage.Client().bucket(bucket).blob(key).download_as_bytes()
    if hashlib.sha256(data).hexdigest() != source["arrays"]["sha256"]:
        raise ValueError("Test response archive changed")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return source, arrays


def measure_path(
    *,
    matrix,
    gradient,
    basis,
    baseline_responses,
    directions,
    length,
    mean_gradient,
    mean_hvp,
    losses,
    weights,
    fits,
    rotation_seed,
    save_arrays,
    progress,
    label,
):
    """Reuse the start responses; add four HVPs at four nonzero points per block."""
    anchors = basis[:4]
    flat = anchors.reshape(4, -1).astype(np.float64)
    baseline_losses = losses(matrix)
    baseline_mean = float(np.average(baseline_losses, weights=weights))
    start_components = envelope_components(gradient, anchors)
    start_rotated = envelope_components(rotated_gradient(gradient, rotation_seed), anchors)
    points = [("start", None, 0.0)] + [(f"alpha{a}-t{t}", a, t) for a in (0.125, 0.25) for t in (0.5, 1.0)]
    rows = []
    for name, alpha, fraction in points:
        started = time.perf_counter()
        direction = directions[alpha] if alpha is not None else np.zeros_like(gradient)
        physical_length = fraction * length
        point = matrix - physical_length * jnp.asarray(direction)
        current_gradient = gradient if fraction == 0 else mean_gradient(point)
        responses = baseline_responses[:4] if fraction == 0 else np.stack([mean_hvp(point, e) for e in anchors])
        projected = flat @ responses.reshape(4, -1).T
        symmetry = float(np.linalg.norm(projected - projected.T) / max(np.linalg.norm(projected), 1e-30))
        checks = []
        if fraction != 0:
            for index in (0, 3):
                trials = []
                for delta in FD_STEPS:
                    fd = (
                        mean_gradient(point + delta * jnp.asarray(anchors[index]))
                        - mean_gradient(point - delta * jnp.asarray(anchors[index]))
                    ) / (2 * delta)
                    error = float(np.linalg.norm(fd - responses[index]) / max(np.linalg.norm(responses[index]), 1e-30))
                    trials.append(dict(delta=delta, relative_error=error))
                    progress(
                        dict(
                            event="path_fd", block=label, point=name, direction=index, delta=delta, relative_error=error
                        )
                    )
                    if len(trials) >= 2 and all(t["relative_error"] < 0.02 for t in trials[-2:]):
                        break
                checks.append(
                    dict(
                        direction=index,
                        trials=trials,
                        passed=len(trials) >= 2 and all(t["relative_error"] < 0.02 for t in trials[-2:]),
                    )
                )
        current_components = envelope_components(current_gradient, anchors)
        current_rotated = envelope_components(rotated_gradient(current_gradient, rotation_seed), anchors)
        envelopes = []
        for reference, g, original, rotated in (
            ("starting", gradient, start_components, start_rotated),
            ("current", current_gradient, current_components, current_rotated),
        ):
            for control, fit in fits.items():
                parts = original if control == "oriented" else rotated
                if control == "scalar":
                    parts = np.stack([original[0], np.sum(g * g) * original[0]])
                envelope = np.einsum("i,ijk->jk", fit["coefficients"], parts)
                envelopes.append(
                    dict(
                        gradient_reference=reference,
                        control=control,
                        projected_envelope=envelope.tolist(),
                        normalized_hessian_norm=normalized_hessian_norm(projected, envelope),
                        mean_trace=float(np.trace(envelope) / 4),
                    )
                )
        point_losses = baseline_losses if fraction == 0 else losses(point)
        loss_change = float(np.average(point_losses, weights=weights)) - baseline_mean
        alignment = float(np.sum(gradient * direction))
        row = dict(
            point=name,
            alpha=alpha,
            fraction=fraction,
            length=physical_length,
            actual_norm=float(np.linalg.norm(host_array(point - matrix))),
            losses=point_losses,
            loss_change=loss_change,
            initial_alignment=alignment,
            first_order_remainder=loss_change + physical_length * alignment,
            projected_hessian=projected.tolist(),
            symmetry_relative=symmetry,
            derivative_checks=checks,
            reused_test_responses=fraction == 0,
            envelopes=envelopes,
            native_seconds=time.perf_counter() - started,
        )
        row["arrays"] = save_arrays(
            f"{label}-path-{name}.npz",
            gradient=current_gradient,
            basis=anchors,
            responses=responses,
            parameter=host_array(point),
        )
        rows.append(row)
        progress(dict(event="path_point", block=label, point=name, loss_change=loss_change, symmetry_relative=symmetry))
        if symmetry >= 1e-4 or not all(c["passed"] for c in checks):
            raise ValueError("Path HVP accuracy or symmetry validation failed")
    return dict(
        block=label,
        length=length,
        anchors="First four frozen test basis vectors",
        reference="Draw0 native eighth block norm shared across powers",
        rows=rows,
        scope="Sampled block path measurements with fixed coefficients; not a uniform path guarantee",
    )
