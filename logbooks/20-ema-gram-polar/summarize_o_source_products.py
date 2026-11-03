# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile o-source input coverage and retain prior same-HLO controls."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_frontier_operands import compare as compare_prior
from compare_o_source_products import indexed
from reconcile_compiler_coverage import compact, reconcile


def summarize(manifest, report):
    loaded = {}
    for label, entry in manifest["inputs"].items():
        raw = Path(entry["path"]).read_bytes()
        assert len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"]
        loaded[label] = json.loads(raw)
    inventory = loaded.pop("inventory")
    assert report["completed"]
    prior = compare_prior(loaded["remaining-products"])
    loaded["o-source-products"] = report
    graphs, sources, controls = {}, {}, {}
    for name in manifest["graphs"]:
        sha = inventory["manifest"]["artifacts"][name]["sha256"]
        assert inventory["artifacts"][name]["sha256_verified"]
        saved = {}
        for label, old in loaded.items():
            if name not in old["artifacts"]:
                assert label == "capture-initializer-concat" and name in ("matrix_untouched", "matrix_evaluate")
                continue
            artifact = old["artifacts"][name]
            assert artifact["sha256_verified"] and artifact["input_sha256"] == sha
            saved[label] = artifact["trace"]
        graphs[name] = compact(reconcile(saved, inventory["artifacts"][name]))
        sources[name] = {
            key: [dict(**boundary, record=value["records"][boundary["node"]]) for boundary in value["boundaries"]]
            for key, value in indexed(report["artifacts"][name]).items()
        }
        controls[name] = {key: value for key, value in prior["graphs"][name].items() if not key.startswith("o_proj/")}
        assert len(controls[name]) == 14
        assert all(value["same_normalized_signature_as_untouched"] for value in controls[name].values())
    return dict(
        graphs=graphs,
        sources=sources,
        reused_controls=dict(input=manifest["inputs"]["remaining-products"], graphs=controls),
        limitations=[
            "Exact-context complete input-root coverage; record presence and opaque-output semantics remain separate.",
            "Ordered repeated positions, tuple projections, source scopes and saved output dependencies retained.",
            "Fourteen prior gate/down/up routes reused with SHA/size and same-HLO checks; no new control measurement.",
            "All original opaque outputs remain opaque; no runtime values/precision/order/identity/cause claim.",
            "Both full28leaf numerical substitution gates remain failed; no numerical intervention selected.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    result = summarize(json.loads(args.manifest.read_text()), json.loads(raw))
    result["report_sha256"] = args.sha256
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
