# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze legacy checkpoint identities and a user-approved Muon probe convention."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import draccus
from google.cloud import storage
from levanter.main.train_lm import TrainLmConfig

from experiments.ema_gram.geometry_data import checkpoint_identity
from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import ErrorAwareMuonConfig

PIN = "fffd6a64ce81722f44d02d754db7c4778e0a5be5"


def decode_legacy_config(original):
    """Restore type information omitted by the old executor's artifact serializer."""
    encoded = copy.deepcopy(original)
    encoded["model"]["type"] = "qwen3"
    encoded["optimizer"]["type"] = "error_aware_muon"
    encoded["adapter"]["type"] = "none"
    encoded["model"]["rope"]["type"] = "default"
    encoded["trainer"]["tracker"]["type"] = "wandb"
    encoded["trainer"]["tracker"].pop("save_xla_dumps", None)
    expected = {
        "compute_dtype": {"__dtype__": "bfloat16"},
        "output_dtype": {"__dtype__": "bfloat16"},
        "param_dtype": {"__dtype__": "float32"},
    }
    if encoded["trainer"]["mp"] != expected:
        raise ValueError("Unexpected native mixed precision")
    encoded["trainer"]["mp"] = "compute=bfloat16,params=float32,output=bfloat16"
    duration = encoded["trainer"]["checkpointer"]["save_interval"]
    seconds = duration["days"] * 86400 + duration["seconds"] + duration["microseconds"] / 1e6
    encoded["trainer"]["checkpointer"]["save_interval"] = f"{seconds:g}s"
    for component in encoded["data"]["components"].values():
        component["type"] = "cached"
        component["source"]["type"] = "url"
        component["format"]["type"] = "text"
        component["source"]["format"]["type"] = "text"
    config = draccus.decode(TrainLmConfig, encoded)
    if not isinstance(config.optimizer, ErrorAwareMuonConfig) or config.optimizer.momentum != 0.98:
        raise ValueError("Expected the native normalized-EMA Muon optimizer at momentum0.98")
    return draccus.encode(config)


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "logbooks/20-ema-gram-polar"
    snapshots = []
    for name in ("optimizer_helpers.py", "prism_berkeley_optimizer.py", "muon_error_feedback_optimizer.py"):
        path = "experiments/speedrun/prism_berkeley_qwen3_scaling/" + name
        reference = subprocess.check_output(["git", "show", PIN + ":" + path], cwd=root)
        if reference != (root / path).read_bytes():
            raise ValueError("Pinned optimizer source changed: " + path)
        snapshots.append(dict(path=path, sha256=hashlib.sha256(reference).hexdigest()))
    client = storage.Client()
    configs, states, identities = [], [], []
    for index, (label, policy, version) in enumerate(
        (
            ("muon-plain-final", "muon", "2026.07.23.4"),
            ("muon-blend-final", "blend-g0p05", "2026.07.23.4"),
            ("muon-hesscorr-final", "hesscorr-g0p3", "2026.08.28.3"),
        )
    ):
        prefix = f"checkpoints/speedrun/qwen3_300m_error_aware_muon_{policy}_lr0p012/{version}"
        artifact_bytes = client.bucket("marin-us-central1").blob(prefix + "/.artifact.json").download_as_bytes()
        artifact = json.loads(artifact_bytes)
        original = artifact["config"]["train_config"]
        if index < 2 and (artifact["provenance"]["base_commit"] != "7cd118640" or artifact["provenance"]["dirty"]):
            raise ValueError("Unexpected legacy Muon checkpoint provenance")
        encoded = decode_legacy_config(original)
        checkpoint = "gs://marin-us-central1/" + prefix + "/checkpoints/step-11443"
        identity = checkpoint_identity(checkpoint)
        configs.append(
            dict(
                config=dict(train_config=encoded),
                artifact_uri="gs://marin-us-central1/" + prefix + "/.artifact.json",
                artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
            )
        )
        identities.append(identity)
        states.append(
            dict(
                state=label,
                checkpoint_schema="momentum_only_7cd118640" if index < 2 else "counter_fffd6a64",
                config_index=index,
                checkpoint=checkpoint,
                expected_counter=11444,
                checkpoint_manifest_sha256=identity["object_manifest_sha256"],
            )
        )
        if index < 2:
            states[-1]["archived_training_provenance"] = {
                key: artifact["provenance"][key] for key in ("base_commit", "tree_hash", "dirty")
            }
    protocol = dict(
        request_sha256="e8ea0a67f7746a7a41a49f96d8a83e3bf9190d824da59a21284cd5a776e27ef8",
        experiment="Muon frozen-checkpoint comparisons",
        tracking=dict(
            entity="marin-community",
            project="ema-gram-geometry",
            mode="online",
            owner="task0",
            restart_policy="fresh run per process attempt; no training resume claimed",
        ),
        states=states,
        optimizer_source=PIN,
        optimizer_source_files=snapshots,
        reference_learning_rate=0.012,
        reference_step_authorization=(
            "User explicitly selected common0.012 reference LR " "and original Adam ratio on2026-11-14"
        ),
        reference_step_scope=(
            "Reference probe scale, not the native next training update; " "original terminal schedule gives zero"
        ),
        native_update_sequences=128,
        measurement_microbatch_sequences=8,
        seeds=dict(calibration=202611140, evaluation=202611141, update=202611142),
        panels=dict(
            calibration_sequences=64,
            evaluation_sequences=128,
            update_draws=4,
            sequences_per_update=128,
            sequence_length=4096,
            paired_across_states=True,
        ),
        length_grid=[0, 0.1, 0.3, 0.6, 1, 2, 4],
        radius="4times median reference-LR plain-Muon complete-update norm across four draws at each checkpoint",
        variants=["muon", "blend_0.02", "blend_0.05", "F5_tangent_0.05", "hesscorr_0.05", "hesscorr_0.3"],
        numerical_gates=dict(relative_error=0.02, consecutive_scales=2, symmetry_relative=1e-4),
        f5_fd_relative_steps=[0.03, 0.01, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0001],
        reference_objective=(
            "FP32 vanilla CE plus1e-7z-loss, excluding decay; " "native gradient retains archived objective"
        ),
        interpretation="Four update batches per checkpoint are not independent training seeds",
        training="No training; restore full saved state and advance moments once per fresh update batch",
        schema_migration=(
            "July plain/blend states contain all original arrays but no later-added correction counter. "
            "Strictly restore original schema; require saved step, optimizer, Adam and schedule counts to agree; "
            "copy that count into the new field. No partial load or zero reset."
        ),
    )
    for name, data in (
        ("geometry-muon-configs.json", configs),
        ("geometry-muon-protocol.json", protocol),
        ("geometry-muon-checkpoint-identities.json", identities),
    ):
        path = output / name
        if path.exists():
            raise FileExistsError("Do not overwrite frozen Muon inputs")
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(states=[s["state"] for s in states], reference_lr=0.012, source=PIN)))


if __name__ == "__main__":
    main()
