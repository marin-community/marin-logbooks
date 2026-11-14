# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import haliax as hax
import numpy as np

from experiments.ema_gram import geometry_data


def test_disjoint_panels_preserve_native_update_batch_size_with_microbatches(monkeypatch, tmp_path):
    def dataset(pos, schedule, key):
        rng = np.random.default_rng(int(np.asarray(key)[1]))
        arrays = [np.zeros((8, 4), np.int32)] + [rng.integers(1, 1000000, (8, 4), dtype=np.int32) for _ in range(20)]
        return [
            SimpleNamespace(tokens=SimpleNamespace(array=a), loss_weight=SimpleNamespace(array=np.ones_like(a)))
            for a in arrays
        ]

    monkeypatch.setattr(geometry_data, "DataLoader", lambda data, **kwargs: iter(data))
    monkeypatch.setattr(geometry_data, "host_array", np.asarray)
    cfg = SimpleNamespace(
        data=SimpleNamespace(train_set=dataset),
        model=SimpleNamespace(max_Pos=hax.Axis("pos", 4)),
        trainer=SimpleNamespace(train_batch_size=8, batch_schedule=None, compute_axis_mapping={}),
    )
    protocol = dict(
        seeds=dict(calibration=42, evaluation=43, update=44),
        panels=dict(calibration_sequences=16, evaluation_sequences=32, update_draws=3, sequences_per_update=16),
    )
    exclusions = tmp_path / "excluded.json"
    exclusions.write_text("[]")
    outputs = {}
    panels, weights, _ = geometry_data.geometry_data(cfg, protocol, exclusions, outputs.__setitem__)
    assert {role: len(batches) for role, batches in panels.items()} == dict(calibration=2, evaluation=4, update=6)
    records = outputs["data-manifest.json"]["records"]
    hashes = [digest for row in records for digest in row["sequence_hashes"]]
    assert len(hashes) == len(set(hashes)) == 96
    assert len(outputs["data-manifest.json"]["rejected"]) == 3
    assert all(value == 24 for group in weights.values() for value in group)
