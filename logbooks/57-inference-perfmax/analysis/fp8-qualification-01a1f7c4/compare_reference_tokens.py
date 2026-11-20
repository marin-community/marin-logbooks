# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare aligned retained token scores from the goldens and serving responses."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from marin.testing.inference.backend_parity import TokenDistribution, distribution_difference
from marin.testing.inference.endpoint_checks import response_top_distribution
from marin.testing.inference.goldens import LookupStatus, history_digest, lookup_reference, read_fixture


def read_response(report_path, report, case):
    request = next(item for item in report["requests"] if item["label"] == f"reference-{case.id}")
    assert request["history_sha256"] == history_digest(case.tokens)
    assert request["http_status"] == 200 and request["prompt_scores"]
    path = report_path.with_name(f"{report_path.stem}-requests") / request["response"]
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == request["response_sha256"]
    (choice,) = json.loads(raw)["choices"]
    assert choice["prompt_token_ids"] == list(case.tokens)
    positions = [*choice["prompt_logprobs"][1:], choice["logprobs"]["top_logprobs"][0]]
    assert len(positions) == len(case.tokens)
    return positions, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--position", action="append", required=True, help="CASE:POSITION")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = read_fixture(args.fixture)
    references = lookup_reference(fixture)
    assert references.status == LookupStatus.FOUND, references.reason
    cases = {case.id: case for case in references.cases}
    goldens = {(row.case_id, row.position): row for row in references.scores}
    reports = {}
    for item in args.report:
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        raw = path.read_bytes()
        report = json.loads(raw)
        assert report["model"] == fixture.model.model_dump(mode="json")
        assert report["manifest"] == fixture.manifest.model_dump(mode="json")
        assert report["policy"]["reference"] == fixture.policy.model_dump(mode="json")["reference"]
        assert report["checks"]["deployment"]["status"] == "passed"
        reports[name] = (path, report, hashlib.sha256(raw).hexdigest())

    results = {}
    for spec in args.position:
        case_id, raw_position = spec.split(":")
        case, position = cases[case_id], int(raw_position)
        golden = goldens[case_id, position]
        reference = TokenDistribution(golden.token_ids, golden.logprobs)
        distributions = {"Levanter": reference}
        receipts = {}
        for name, (path, report, sha256) in reports.items():
            positions, request = read_response(path, report, case)
            actual = response_top_distribution(positions[position])
            recorded = next(
                row for row in report["checks"]["reference"]["observations"]
                if row["case"] == case_id and row["position"] == position
            )
            assert list(actual.token_ids) == recorded["actual"]["token_ids"]
            assert list(actual.logprobs) == recorded["actual"]["logprobs"]
            assert distribution_difference(reference, actual).max_probability_error == recorded["max_probability_error"]
            distributions[name] = actual
            receipts[name] = {
                "runtime_source": report["runtime_source"],
                "report_sha256": sha256,
                "response_sha256": request["response_sha256"],
                "history_sha256": request["history_sha256"],
            }
        probabilities = {
            name: dict(zip(scores.token_ids, map(math.exp, scores.logprobs), strict=True))
            for name, scores in distributions.items()
        }
        common = set.intersection(*(set(probabilities_) for probabilities_ in probabilities.values()))
        tokens = sorted(common, key=lambda token: max(scores[token] for scores in probabilities.values()), reverse=True)
        results[spec] = {
            "group": case.group,
            "history_length": len(case.tokens),
            "scored_prefix_length": position + 1,
            "next_corpus_token": case.tokens[position + 1] if position + 1 < len(case.tokens) else None,
            "top_tokens": {name: scores.token_ids[0] for name, scores in distributions.items()},
            "largest_shared_probabilities": [
                {"token": token, **{name: scores[token] for name, scores in probabilities.items()}}
                for token in tokens[:5]
            ],
            "reference_error": {
                name: {
                    "lower": distribution_difference(reference, actual).max_probability_error_lower_bound,
                    "upper": distribution_difference(reference, actual).max_probability_error,
                }
                for name, actual in distributions.items()
                if name != "Levanter"
            },
            "receipts": receipts,
        }
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
