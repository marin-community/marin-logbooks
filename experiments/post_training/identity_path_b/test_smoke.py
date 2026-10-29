# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import numpy as np
import pytest
from levanter.optim.config import AdamConfig

from experiments.post_training.identity_path_b import smoke
from experiments.post_training.identity_path_b.base import BASE_BINDING
from experiments.post_training.identity_path_b.storage_audit import FROZEN_HF_NAMES


def test_smoke_admission_is_distinct_and_never_relaxes_full_arm_minimum(monkeypatch):
    # This small fixture binds its own cumulative digest. Production binds the
    # literal immutable two-batch prefix digest and is separately reconstructed.
    cumulative = [
        dict(completed_updates=1),
        dict(completed_updates=2, identity_final_text_tokens=7, identity_supervised_tokens=9, all_supervised_tokens=90),
    ]
    binding = dict(
        smoke.SMOKE_PARENT, first_two_cumulative_sha256=smoke.sha(json.dumps(cumulative, sort_keys=True).encode())
    )
    monkeypatch.setattr(smoke, "SMOKE_PARENT", binding)
    exposure = dict(
        BASE_BINDING,
        experiment="B-base-smoke",
        smoke_parent=binding,
        updates=2,
        batch_size=8,
        sequence_length=8192,
        checkpoint_updates=[1],
        identity_minimum_excludes_structural_tokens=0,
        identity_share_target=0.1,
        cumulative_batches=cumulative,
        final_text_tokens={"identity": 7},
        raw_supervised_tokens={"identity": 9, "replay": 81},
        reasoning_text_tokens={"identity": 0},
        maximum_identity_source_occurrences_across_modes=1,
    )
    smoke.check_training_exposure(exposure, "10")
    with pytest.raises(ValueError, match="exact B-base"):
        smoke.check_training_exposure(exposure, "25")
    with pytest.raises(ValueError, match="two unchanged batches"):
        smoke.check_training_exposure(dict(exposure, checkpoint_updates=[1, 2]), "10")
    full = dict(exposure, experiment="B-base", identity_minimum_excludes_structural_tokens=500000)
    with pytest.raises(ValueError, match="final-text minimum"):
        smoke.check_training_exposure(full, "10")


def gate_fixture():
    event = dict(
        arm="10",
        training_pin="a" * 40,
        packed_data_sha256="b" * 64,
        native_checkpoint="native-step0",
        exposure={"identity_final_text_tokens": 3728},
    )
    training = dict(
        completed_updates=2,
        arm="10",
        training_pin="a" * 40,
        packed_sha256="b" * 64,
        checkpoints=[dict(completed_updates=1, native_checkpoint="native-step0", exposure=event["exposure"])],
    )
    generations = "".join(
        json.dumps(
            dict(row_id=str(i), set="identity" if i < 52 else "control", coverage="response_received", thinking=False)
        )
        + "\n"
        for i in range(66)
    ).encode()
    records = {name: dict(frozen_source=True, export_sha256="d" * 64) for name in FROZEN_HF_NAMES}
    records.update({f"trainable.{i}": dict(frozen_source=False, export_sha256="d" * 64) for i in range(398)})
    return dict(
        generations=generations,
        training=training,
        native_metadata={"step": 0},
        exported_event=event,
        expected_event=event,
        storage_audit=dict(
            schema="path_b_storage_v1",
            frozen_native_tensors=104,
            trained_master_cast_tensors=398,
            source_receipt_sha256="af484f49ac2c1d67989495475bcc79bf7519d09281c8087321d906b67139fa97",
            tensors=records,
        ),
        trend=dict(
            event=event,
            status="generated_pending_semantic_read",
            gsm_coverage_complete=True,
            identity_control_coverage=dict(
                requested=66, received=66, errors=0, generations_sha256=smoke.sha(generations)
            ),
            gsm_outcome={"coverage": {"gsm8k": {"n_scored": 1319}}},
        ),
        gsm_inputs=dict(
            count=1319,
            canonical_question_answer_jsonl_sha256="32c548f08195e19e33408b844dd7be6aa4bcae457d957bc05909ff4bd4a00595",
        ),
    )


def test_smoke_gate_requires_actual_two_updates_bound_save_audit_and_complete_trend():
    fixture = gate_fixture()
    smoke.require_smoke_success(**fixture)
    for key, replacement in [
        ("training", dict(fixture["training"], completed_updates=1)),
        ("native_metadata", {"step": 1}),
        ("generations", b""),
        ("storage_audit", dict(fixture["storage_audit"], tensors={})),
        ("trend", dict(fixture["trend"], status="incomplete_coverage")),
        ("gsm_inputs", dict(fixture["gsm_inputs"], count=1318)),
    ]:
        altered = copy.deepcopy(fixture)
        altered[key] = replacement
        with pytest.raises(ValueError):
            smoke.require_smoke_success(**altered)


def test_smoke_two_step_warmup_is_exact_full_arm_prefix_and_export_config_builds():
    config = AdamConfig(learning_rate=2e-6, lr_schedule="constant", warmup=2, weight_decay=0.0, max_grad_norm=0.5)
    short, full = config.lr_scheduler(2), config.lr_scheduler(140)
    actual = np.asarray([short(i) for i in range(2)])
    np.testing.assert_array_equal(actual, np.asarray([full(i) for i in range(2)]))
    np.testing.assert_allclose(actual, [0, 1e-6], rtol=1e-6, atol=0)
    assert config.build(1) is not None
