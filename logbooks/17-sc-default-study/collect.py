# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect compact grades and curves from one immutable study export directory."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from skyrl_gym.verification import VerificationResult, VerificationStatus, normalized_verifier_score

from experiments.post_training.sc_default_regrade import fingerprint


def read_json(uri: str):
    fs, key = filesystem_for(uri)
    with fs.open(key, "rt") as stream:
        return json.load(stream)


def write_json(path: Path, content) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n")


def collect_grades(export: str, step: int, index: list[dict], output: Path, excluded: set[str] | None = None) -> dict:
    expected = {row["uid"]: row for row in index}
    excluded = set(expected) & (excluded or set())
    if len(expected) != len(index):
        raise ValueError("Duplicate evaluation-index identities")
    fs, prefix = filesystem_for(export + f"/dumped_evals/global_step_{step}_evals")
    recovery_uri = export + f"/study/regrades-{step:04d}.json"
    recovery_fs, recovery_key = filesystem_for(recovery_uri)
    recovered = read_json(recovery_uri)["rows"] if recovery_fs.exists(recovery_key) else {}
    rows, source_objects = {}, []
    for key in sorted(fs.find(prefix)):
        if not key.endswith(".jsonl") or key.endswith("/aggregated_results.jsonl"):
            continue
        digest = hashlib.sha256()
        count = 0
        with fs.open(key, "rt") as stream:
            for line in stream:
                digest.update(line.encode())
                value = json.loads(line)
                uid = str(value["uid"])
                if uid not in expected or uid in rows:
                    raise ValueError(f"Unexpected or repeated task uid at step {step}: {uid}")
                task = expected[uid]
                verification = value.get("verification")
                grade_recovery = recovered.get(uid)
                if grade_recovery is not None:
                    if verification is not None and verification["status"] == "verified":
                        raise ValueError("Grade recovery must not replace an existing valid verdict")
                    if grade_recovery["contract"] != task["contract"] or grade_recovery["row_sha256"] != fingerprint(
                        value
                    ):
                        raise ValueError("Grade recovery differs from the saved answer or original contract")
                    verification = grade_recovery["verification"]
                score = None
                if verification is not None:
                    verdict = VerificationResult(
                        **{**verification, "status": VerificationStatus(verification["status"])}
                    )
                    if verdict.status is VerificationStatus.VERIFIED:
                        score = normalized_verifier_score(verdict)
                rows[uid] = {
                    "uid": uid,
                    "task": task["study_id"],
                    "family": task["teacher_route"],
                    "public_fingerprint": task["public"],
                    "source": task["source"],
                    # The frozen index names the curriculum import; dumps name
                    # its upstream dataset (several imports share TaskTrove).
                    "upstream_dataset": value["data_source"],
                    "normalized_verifier_score": score,
                    "full_success": None if score is None else int(score >= 1 - 1e-6),
                    "verification": verification,
                    "grade_recovery": grade_recovery,
                    "response_tokens": len(value["response_token_ids"]),
                    "stop_reason": value["stop_reason"],
                    "exception_type": value["exception_type"],
                    "error_treatment": value["error_treatment"],
                    "server_error": value["server_error"],
                }
                count += 1
        source_objects.append({"uri": "s3://" + key, "rows": count, "sha256": digest.hexdigest()})
    ordered = [rows[row["uid"]] for row in index if row["uid"] in rows]
    admitted = [row for row in ordered if row["uid"] not in excluded]
    with (output / f"grades-{step:04d}.jsonl").open("w") as stream:
        for row in admitted:
            stream.write(json.dumps(row) + "\n")
    if excluded:
        with (output / f"excluded-grades-{step:04d}.jsonl").open("w") as stream:
            for row in ordered:
                if row["uid"] in excluded:
                    stream.write(json.dumps(row) + "\n")
    missing = sorted(set(expected) - set(rows))
    ungraded = [row["uid"] for row in ordered if row["full_success"] is None]
    admitted_ungraded = [uid for uid in ungraded if uid not in excluded]
    coverage = {
        "step": step,
        "expected": len(index),
        "generated": len(rows),
        "graded": len(rows) - len(ungraded),
        "missing_uids": missing,
        "ungraded_uids": ungraded,
        "excluded_uids": sorted(excluded, key=int),
        "analysis_expected": len(index) - len(excluded),
        "analysis_graded": len(admitted) - len(admitted_ungraded),
        "source_objects": source_objects,
    }
    write_json(output / f"coverage-{step:04d}.json", coverage)
    if missing or admitted_ungraded:
        raise ValueError(f"Incomplete evaluation at step {step}; saved compact evidence for grade-only recovery")
    curve = [row for row in ordered[:1000] if row["uid"] not in excluded]
    return {"step": step, **summarize(admitted), "curve_panel": summarize(curve), "coverage": coverage}


def summarize(ordered: list[dict]) -> dict:
    families = sorted({row["family"] for row in ordered})
    family_results = {
        family: {
            "tasks": sum(row["family"] == family for row in ordered),
            "full_success": float(np.mean([row["full_success"] for row in ordered if row["family"] == family])),
            "mean_verifier_reward": float(
                np.mean([row["normalized_verifier_score"] for row in ordered if row["family"] == family])
            ),
        }
        for family in families
    }
    lengths = [row["response_tokens"] for row in ordered]
    return {
        "tasks": len(ordered),
        "families": family_results,
        "macro_full_success": float(np.mean([row["full_success"] for row in family_results.values()])),
        "macro_mean_verifier_reward": float(np.mean([row["mean_verifier_reward"] for row in family_results.values()])),
        "response_tokens": sum(lengths),
        "response_length_quantiles": dict(
            zip(
                ("min", "median", "p90", "p99", "max"),
                np.quantile(lengths, (0, 0.5, 0.9, 0.99, 1)).tolist(),
                strict=True,
            )
        ),
        "stop_counts": dict(Counter(row["stop_reason"] for row in ordered)),
    }


def collect(export: str, index_path: Path, output: Path, steps: list[int], eligibility_path: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text())
    excluded, eligibility_sha = set(), None
    if eligibility_path is not None:
        eligibility = json.loads(eligibility_path.read_text())
        if (
            not eligibility.get("approved")
            or eligibility["frozen_index_sha256"] != hashlib.sha256(index_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Eligibility needs approval and the exact frozen evaluation index")
        indexed = {row["uid"]: row for row in index}
        for row in eligibility["excluded"]:
            if row["uid"] in excluded or int(row["uid"]) < 1000:
                raise ValueError("Eligibility repeats a task or changes the fixed curve panel")
            if any(row[key] != indexed[row["uid"]][key] for key in ("study_id", "teacher_route", "contract", "public")):
                raise ValueError("Excluded task differs from its original frozen contract")
            excluded.add(row["uid"])
        retained = [row for row in index if row["uid"] not in excluded]
        if (
            len(retained) != eligibility["retained_quality_tasks"]
            or dict(Counter(row["teacher_route"] for row in retained)) != eligibility["retained_family_counts"]
        ):
            raise ValueError("Approved quality denominators changed")
        eligibility_sha = hashlib.sha256(eligibility_path.read_bytes()).hexdigest()
    training = read_json(export + "/study/completed-training.json")
    write_json(output / "completed-training.json", training)
    cycles = training["steps"]
    if [row["step"] for row in cycles] != list(range(1, steps[-1] + 1)):
        raise ValueError("Missing or repeated policy-update measurements")
    cumulative_loss, cumulative_response = 0, 0
    for row in cycles:
        cumulative_loss += row["trained_loss_tokens"]
        cumulative_response += row["response_tokens"]
        row["cumulative_trained_loss_tokens"] = cumulative_loss
        row["cumulative_response_tokens"] = cumulative_response
    metrics = []
    fs, prefix = filesystem_for(export + "/study")
    for key in sorted(fs.find(prefix)):
        if "/metrics-" in key and key.endswith(".json"):
            with fs.open(key, "rt") as stream:
                metrics.append(json.load(stream))
    write_json(output / "ordinary-metrics.json", metrics)
    evaluations, curves = [], []
    for step in steps:
        panel = index if step == steps[-1] else index[: min(1000, len(index))]
        result = collect_grades(export, step, panel, output, excluded)
        evaluations.append(result)
        curve = result["curve_panel"]
        cycle = cycles[step - 1] if step else {}
        curves.append(
            {
                "step": step,
                "tasks": curve["tasks"],
                "macro_full_success": curve["macro_full_success"],
                "macro_mean_verifier_reward": curve["macro_mean_verifier_reward"],
                "trained_loss_tokens": cycle.get("cumulative_trained_loss_tokens", 0),
                "training_cycle_elapsed_seconds": cycle.get("training_cycle_elapsed_seconds", 0),
            }
        )
    report = {
        "export": export,
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "eligibility_sha256": eligibility_sha,
        "evaluations": evaluations,
        "curves": curves,
        "cycles": cycles,
        "training_cycle_seconds": cycles[-1]["training_cycle_elapsed_seconds"],
        "startup": read_json(export + "/study/startup.json"),
        "evaluation_times": [read_json(export + f"/study/evaluation-time-{step:04d}.json") for step in steps],
        "limits": (
            "Curves always use the same balanced first 1,000 tasks. All original final answers remain required; "
            "the approved common quality mask is applied identically to every arm and seed."
        ),
    }
    write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export")
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 25, 50])
    parser.add_argument(
        "--eligibility", type=Path, help="Approved fixed quality exclusions; generation coverage stays exact"
    )
    args = parser.parse_args()
    collect(args.export.rstrip("/"), args.index, args.output, args.steps, args.eligibility)
