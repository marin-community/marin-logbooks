# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read all frozen Muon measurement panels without loading or training a model."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax

from experiments.ema_gram.geometry_data import geometry_data
from experiments.ema_gram.muon_data import replace_muon_cache
from experiments.ema_gram.prepare_muon_data import file_hashes
from experiments.ema_gram.probe import resolve_config
from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import ErrorAwareMuonConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recovery = json.loads(args.recovery_manifest.read_text())
    for item in recovery["files"]:
        path = args.local_root / item["path"]
        if path.stat().st_size != item["bytes"] or file_hashes(path)[0] != item["sha256"]:
            raise ValueError("Local data differs from verified recovery: " + item["path"])
    archive = Path("logbooks/20-ema-gram-polar")
    configs = json.loads((archive / "geometry-muon-configs.json").read_text())
    protocol = json.loads((archive / "geometry-muon-protocol.json").read_text())
    cfg = resolve_config(configs[0]["config"]["train_config"])
    if not isinstance(cfg.optimizer, ErrorAwareMuonConfig):
        raise ValueError("Expected the pinned Muon optimizer configuration")
    for item in configs[1:]:
        if item["config"]["train_config"]["data"] != configs[0]["config"]["train_config"]["data"]:
            raise ValueError("Muon states do not share the same frozen data configuration")
    cfg = replace_muon_cache(cfg, str(args.local_root.resolve()))
    cfg = replace(cfg, trainer=replace(cfg.trainer, train_batch_size=protocol["measurement_microbatch_sequences"]))
    for key, value in cfg.trainer.jax_config.items():
        jax.config.update(key, value)
    records = {}
    with cfg.trainer.use_device_mesh():
        panels, weights, digest = geometry_data(
            cfg, protocol, archive / "geometry-excluded-sequences.json", records.__setitem__
        )
    output = dict(
        success=True,
        recovery_manifest_sha256=hashlib.sha256(args.recovery_manifest.read_bytes()).hexdigest(),
        data_manifest_sha256=digest,
        protocol_sha256=hashlib.sha256((archive / "geometry-muon-protocol.json").read_bytes()).hexdigest(),
        groups={role: len(batches) for role, batches in panels.items()},
        valid_weights={role: sum(values) for role, values in weights.items()},
        manifest=records["data-manifest.json"],
        scope="CPU data-only preflight using the real measurement loader; no model or optimizer execution",
    )
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "manifest"}), flush=True)


if __name__ == "__main__":
    main()
