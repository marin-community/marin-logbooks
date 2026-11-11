# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the supplied fit plus the declared empirical resolution gate."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from experiments.ema_gram.curvature_resolution import projected_resolution


def load_package(path):
    spec = importlib.util.spec_from_file_location("supplied_hessian_fit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def acceptance(analysis, resolution):
    return {
        name: bool(
            fit.get("status") == "valid"
            and fit.get("positive_power_relation")
            and resolution.get("status") == "empirically_resolved"
        )
        for name, fit in analysis.get("fits", {}).items()
    }


def empirical_bootstrap(h, fd, k, package, repetitions=1000):
    """Reapply empirical precision as well as numerical gates to each resample."""
    rng = np.random.default_rng(20261110)
    values = []
    rejections = {}
    for _ in range(repetitions):
        indices = rng.integers(0, len(h), len(h))
        sampled = h[indices]
        resolution = projected_resolution(sampled, fd[:, indices])
        if resolution["status"] != "empirically_resolved":
            reason = resolution["status"]
            rejections[reason] = rejections.get(reason, 0) + 1
            continue
        mean = sampled.mean(axis=0)
        fit = package.safe_fit((mean + mean.T) / 2, k, 1e-12)
        if fit.get("status") != "valid" or not fit.get("positive_power_relation"):
            rejections["numerical_or_power_gate"] = rejections.get("numerical_or_power_gate", 0) + 1
            continue
        values.append([fit["r"], fit["explained_fraction_rho"], fit["alpha_log_nonnegative"]])
    intervals = np.quantile(values, [0.025, 0.975], axis=0).T.tolist() if values else [None] * 3
    return dict(
        requested=repetitions,
        accepted=len(values),
        rejections=rejections,
        intervals_given_accepted=dict(zip(("r", "rho", "alpha"), intervals, strict=True)),
        scope="Resampled curvature groups with saved/fresh proxy fixed; both empirical and numerical gates reapplied",
    )


def analyze_measurement(folder, package, controls=None):
    result = dict(
        measurement=str(folder),
        acceptance_rule="Numerical input/positive-log gates plus positive fitted power and empirical resolution",
        scope="Conditional selected-subspace evidence; no causal training-performance claim",
        blocks={},
    )
    fresh = json.loads((controls / "fresh-grams.json").read_text()) if controls else None
    for block in ("attention", "mlp"):
        source = folder / (block + ".npz")
        if not source.exists():
            result["blocks"][block] = dict(status="missing_measurement")
            continue
        with np.load(source, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        metadata = json.loads(str(arrays["metadata_json"].item()))
        dimension = arrays["curvature_batches"].shape[-1]
        sizes = sorted(set((min(12, dimension), dimension)))
        block_result = {}
        for d in sizes:
            h = arrays["curvature_batches"][:, :d, :d]
            fd = arrays.get("finite_difference_batches")
            resolution = (
                projected_resolution(h, fd[:, :, :d, :d])
                if fd is not None
                else dict(status="unresolved", reason="No full projected finite differences")
            )
            restriction = dict(metadata, analyzed_dimension=d, projected_resolution=resolution)
            payload = dict(
                curvature_batches=h,
                gram_raw=arrays["gram_raw"][:d, :d],
                gram_ridged=arrays["gram_ridged"][:d, :d],
                metadata_json=json.dumps(restriction),
            )
            if d == dimension:
                payload["leakage_mean_hvp"] = arrays["leakage_mean_hvp"]
            elif "nested_leakage" in metadata:
                payload["leakage_mean_hvp"] = metadata["nested_leakage"]["original"]
            derived = folder / f"{block}-{d}-restriction.npz"
            np.savez_compressed(derived, **payload)
            try:
                fitted = package.analyze(derived, repetitions=1000)
            except (package.InvalidFit, ValueError) as error:
                fitted = dict(status="invalid_input", reason=str(error))
            record = dict(
                resolution=resolution,
                analysis=fitted,
                accepted_positive_power=acceptance(fitted, resolution),
                nested_leakage=metadata.get("nested_leakage"),
            )
            record["empirical_bootstrap"] = {
                proxy: empirical_bootstrap(h, fd[:, :, :d, :d], payload[proxy], package)
                for proxy, accepted in record["accepted_positive_power"].items()
                if accepted
            }
            if fresh:
                record["fresh_histories"] = []
                manifests = [json.loads((p / "artifact-manifest.json").read_text()) for p in (folder, controls)]
                basis = [next(x for x in m if x["name"] == block + "-basis.npz") for m in manifests]
                if basis[0]["crc32c"] != basis[1]["crc32c"] or basis[0]["size"] != basis[1]["size"]:
                    raise ValueError("Fresh and curvature basis archives differ; verify coordinates before comparison")
                for row in fresh["results"]:
                    if row["block"] != block:
                        continue
                    new_payload = dict(payload)
                    for proxy in ("gram_raw", "gram_ridged"):
                        new_payload[proxy] = np.asarray(row[proxy])[:d, :d]
                    path = folder / f"{block}-{d}-fresh-{row['history']}.npz"
                    np.savez_compressed(path, **new_payload)
                    try:
                        new_fit = package.analyze(path, repetitions=1000)
                    except (package.InvalidFit, ValueError) as error:
                        new_fit = dict(status="invalid_input", reason=str(error))
                    record["fresh_histories"].append(
                        dict(
                            history=row["history"],
                            analysis=new_fit,
                            accepted_positive_power=acceptance(new_fit, resolution),
                            empirical_bootstrap={
                                proxy: empirical_bootstrap(h, fd[:, :, :d, :d], new_payload[proxy], package)
                                for proxy, accepted in acceptance(new_fit, resolution).items()
                                if accepted
                            },
                        )
                    )
                distances = {}
                for proxy in ("gram_raw", "gram_ridged"):
                    try:
                        z_saved = package.spd_log(payload[proxy], "saved proxy", 1e-12)[0]
                        estimates = [
                            package.spd_log(np.asarray(r[proxy])[:d, :d], "fresh proxy", 1e-12)[0]
                            for r in fresh["results"]
                            if r["block"] == block
                        ]
                        distances[proxy] = dict(
                            fresh_pair_log_rms=float(np.linalg.norm(estimates[0] - estimates[1]) / np.sqrt(d)),
                            saved_to_fresh_log_rms=[float(np.linalg.norm(z - z_saved) / np.sqrt(d)) for z in estimates],
                            definition="Frobenius distance between centered matrix logs divided by sqrt(dimension)",
                            warning="Two disjoint windows; not an identified noise/drift decomposition",
                        )
                    except package.InvalidFit as error:
                        distances[proxy] = dict(status="invalid", reason=str(error))
                record["proxy_distances"] = distances
            block_result[str(d)] = record
        result["blocks"][block] = block_result
    return result


def paired_losses(folder):
    data = json.loads((folder / "full-update-losses.json").read_text())
    rng = np.random.default_rng(20261111)
    rows = []
    for row in data["rows"]:
        delta = np.asarray(row["losses"]) - row["baseline_losses"]
        indices = rng.integers(0, len(delta), (10000, len(delta)))
        reference = next(
            r
            for r in data["rows"]
            if r["alpha"] == 0.125 and r["matching"] == row["matching"] and r["step_fraction"] == row["step_fraction"]
        )
        paired = np.asarray(row["losses"]) - reference["losses"]
        rows.append(
            dict(
                alpha=row["alpha"],
                matching=row["matching"],
                step_fraction=row["step_fraction"],
                update_norm=row["update_norm"],
                delta_mean=float(delta.mean()),
                conditional_delta_interval=np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
                versus_eighth_mean=float(paired.mean()),
                conditional_versus_eighth_interval=np.quantile(paired[indices].mean(axis=1), [0.025, 0.975]).tolist(),
            )
        )
    return dict(
        rows=rows,
        scope="Paired batch bootstrap; one gradient and saved state fixed; descriptive without multiplicity correction",
        repetitions=10000,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    if args.measurement:
        result = analyze_measurement(args.measurement, load_package(args.package), args.controls)
        (args.measurement / "aligned-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.controls and (args.controls / "full-update-losses.json").exists():
        result = paired_losses(args.controls)
        (args.controls / "paired-loss-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
