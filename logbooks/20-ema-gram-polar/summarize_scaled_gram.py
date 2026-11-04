# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join every process of a 32/64-device Gram scaling trial using static JSON."""

import argparse
import hashlib
import json
from pathlib import Path

from summarize_grouped_gram import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--devices", type=int, choices=(32, 64), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    mesh = dict(data=4, model=8) if args.devices == 32 else dict(data=8, model=8)
    for report in reports:
        assert report["device_count"] == args.devices and report["mesh_shape"] == mesh
        assert report["device_kinds"] and all("v4" in kind.lower() for kind in report["device_kinds"])
    base = Path(__file__).parent
    result = summarize(
        reports,
        json.loads((base / "scaled-gram-manifest.json").read_text()),
        json.loads((base / "loss-pair-checkpoint-metadata.json").read_text()),
        args.source_commit,
        processes=args.devices // 4,
        devices=args.devices,
    )
    result.update(device_count=args.devices, mesh_shape=mesh)
    result["input_reports"] = [
        dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in args.reports
    ]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
