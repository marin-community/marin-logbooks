# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay saved reference observations with the corrected top-k parser, without GPUs."""

import argparse
import dataclasses
import gzip
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from marin.testing.inference.backend_parity import TokenDistribution, distribution_difference
from marin.testing.inference.endpoint_checks import mismatch_summary, response_distribution, response_top_distribution
from marin.testing.inference.goldens import LookupStatus, canonical_bytes, history_digest, lookup_reference, read_fixture
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.knotmeter.artifacts import copy_file, file_digest, read_record, unpack


def saved_observations(report: dict, root: Path, cases: tuple) -> list[dict]:
    recorded = report["checks"]["reference"].get("observations")
    if recorded is not None:
        return recorded
    requests = {row["label"]: row for row in report["requests"]}
    observations = []
    for case in cases:
        request = requests.get(f"reference-{case.id}")
        if request is None:
            continue
        if request["http_status"] != 200:
            continue
        raw = gzip.decompress((root / "report-requests" / request["response"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != request["response_sha256"]:
            raise ValueError("A retained response differs from its recorded hash")
        choice = json.loads(raw)["choices"][0]
        if choice["prompt_token_ids"] != list(case.tokens):
            raise ValueError("A retained response belongs to another token history")
        positions = [*choice["prompt_logprobs"][1:], choice["logprobs"]["top_logprobs"][0]]
        for position in case.positions:
            observations.append(
                {
                    "case": case.id,
                    "position": position,
                    "actual": dataclasses.asdict(response_distribution(positions[position])),
                }
            )
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--archive", type=Path, required=True, help="Already downloaded native archive")
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture = read_fixture(args.fixture)
    references = lookup_reference(fixture)
    if references.status != LookupStatus.FOUND:
        raise ValueError(references.reason)
    receipt = read_record(args.receipt)
    if file_digest(args.archive) != receipt["archive_sha256"]:
        raise ValueError("The input archive differs from its receipt")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work_dir) as directory:
        root = Path(directory)
        unpack(args.archive, root)
        raw_report = (root / "report.json").read_bytes()
        report = json.loads(raw_report)
        if (
            report["model"] != fixture.model.model_dump(mode="json")
            or report["manifest"] != fixture.manifest.model_dump(mode="json")
            or report["policy"] != fixture.policy.model_dump(mode="json")
            or report["runtime_source"] != args.runtime_source
            or report["checks"]["deployment"]["status"] != "passed"
        ):
            raise ValueError("Saved evidence does not match this model, runtime and policy")
        requests = {row["label"]: row for row in report["requests"]}
        cases = {case.id: case for case in references.cases}
        expected = {(row.case_id, row.position): row for row in references.scores}
        groups = {name: [] for name in fixture.policy.reference}
        observations, found = [], set()
        for saved in saved_observations(report, root, references.cases):
            key = (saved["case"], saved["position"])
            if key in found or key not in expected:
                raise ValueError("Unexpected or repeated saved observation")
            case = cases[saved["case"]]
            request = requests[f"reference-{case.id}"]
            if request["history_sha256"] != history_digest(case.tokens) or not request["prompt_scores"]:
                raise ValueError("Saved request belongs to another scoring history")
            scores = saved["actual"]
            actual = response_top_distribution(dict(zip(scores["token_ids"], scores["logprobs"], strict=True)))
            row = expected[key]
            difference = dataclasses.asdict(
                distribution_difference(TokenDistribution(row.token_ids, row.logprobs), actual)
            )
            observation = difference | {"case": case.id, "position": row.position}
            groups[case.group].append(observation)
            observations.append(observation | {"actual": dataclasses.asdict(actual)})
            found.add(key)
        summaries = {name: mismatch_summary(rows, fixture.policy.reference[name]) for name, rows in groups.items()}
        complete = found == expected.keys()
        status = (
            "pending"
            if not complete
            else "passed" if all(row["status"] == "passed" for row in summaries.values()) else "failed"
        )
        result = {
            "status": status,
            "groups": summaries,
            "observations": observations,
            "positions": len(found),
            "required_positions": len(expected),
            "completed_cases": [
                case.id for case in references.cases if all((case.id, p) in found for p in case.positions)
            ],
            "total_cases": len(references.cases),
        }
        output = report | {
            "checks": report["checks"] | {"reference": result},
            "reference_replay": {
                "checker_source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "input_receipt": args.receipt,
                "input_archive": receipt["archive_uri"],
                "input_archive_sha256": receipt["archive_sha256"],
                "input_report_sha256": hashlib.sha256(raw_report).hexdigest(),
                "input_reference_status": report["checks"]["reference"]["status"],
                "input_reference_error": report["checks"]["reference"].get("error"),
                "scope": "Reference comparisons only; other stages retain their original outcomes",
            },
        }
        states = [row["status"] for row in output["checks"].values()]
        output["status"] = (
            "failed"
            if "failed" in states
            else "passed" if len(states) == 5 and all(state == "passed" for state in states) else "pending"
        )
        encoded = canonical_bytes(output)
        digest = hashlib.sha256(encoded).hexdigest()
        uri = prefix_join(args.output_prefix, f"{digest}.json")
        local = root / "recomputed.json"
        local.write_bytes(encoded)
        copy_file(str(local), uri)
        print(
            json.dumps(
                {
                    "report": {"uri": uri, "sha256": digest},
                    "status": output["status"],
                    "reference_status": status,
                    "positions": len(found),
                    "required_positions": len(expected),
                    "completed_cases": len(result["completed_cases"]),
                    "total_cases": len(references.cases),
                    "groups": {
                        name: {key: row[key] for key in ["status", "positions", "statistics"] if key in row}
                        for name, row in summaries.items()
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
