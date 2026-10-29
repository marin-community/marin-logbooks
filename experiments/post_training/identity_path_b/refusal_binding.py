# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact admission for the broad512/refilled derivative; original guards stay intact."""

from experiments.post_training.identity_path_b.exposure import GLOBAL_BATCH, SEQUENCE_LENGTH, checkpoint_cadence

CAP_BINDINGS = {
    "B-base": {
        "10": {
            "experiment": "B-base",
            "identity_refusal_cap": "broad512-seed310-refill-r5",
            "identity_original_sources": 33298,
            "identity_rendered_rows": 66596,
            "identity_pool_sha256": "912126844fab24dc3c0f2fad2d0ce698c7c6a7da7fe3103d44391a197866f9de",
            "identity_parent_pool_sha256": "0d0fd663593786ea356e843eb0a64d684778174c9585d25110b142b8c5f84930",
            "replay_pool_sha256": "affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a",
            "source_selection_sha256": "b109c4f642787b9e3dae87c857283646b83e81fc423efb031117a8550fce49c5",
            "retained_refusal_source_ids_sha256": "61178f0be00429993551ff842d675af78d963d14b4405d42cdcaa2b1e69b5ec9",
            "dropped_source_ids_sha256": "86ec979aaacad57ad9da6496384abba386296cbd38dc07208aa77f10ff462b4e",
            "updates": 140,
            "ordered_occurrence_ledger_sha256": "0b46c96740a32ad420b3614259b77028871622f69434ef15682478a339714da0",
            "final_text_tokens": {"identity": 500021, "replay": 2468028},
            "reasoning_text_tokens": {"identity": 0, "replay": 2470419},
            "raw_supervised_tokens": {"identity": 547751, "replay": 4950161},
            "identity_trace_encounters": {"on_empty_fallback": 11930, "off_unchanged": 11940},
            "maximum_identity_source_occurrences_across_modes": 2,
            "maximum_identity_rendered_row_occurrences": 1,
            "checkpoint_updates": [35, 70, 105, 140],
            "identity_epoch_boundary_updates": [],
        },
        "25": {
            "experiment": "B-base",
            "identity_refusal_cap": "broad512-seed310-refill-r5",
            "identity_original_sources": 33298,
            "identity_rendered_rows": 66596,
            "identity_pool_sha256": "912126844fab24dc3c0f2fad2d0ce698c7c6a7da7fe3103d44391a197866f9de",
            "identity_parent_pool_sha256": "0d0fd663593786ea356e843eb0a64d684778174c9585d25110b142b8c5f84930",
            "replay_pool_sha256": "affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a",
            "source_selection_sha256": "b109c4f642787b9e3dae87c857283646b83e81fc423efb031117a8550fce49c5",
            "retained_refusal_source_ids_sha256": "61178f0be00429993551ff842d675af78d963d14b4405d42cdcaa2b1e69b5ec9",
            "dropped_source_ids_sha256": "86ec979aaacad57ad9da6496384abba386296cbd38dc07208aa77f10ff462b4e",
            "updates": 204,
            "ordered_occurrence_ledger_sha256": "f60d310a2bb97a8fca55cf13a3beaa7515588aaeda7a05999021f99a31a1915d",
            "final_text_tokens": {"identity": 1500009, "replay": 2472392},
            "reasoning_text_tokens": {"identity": 0, "replay": 2475504},
            "raw_supervised_tokens": {"identity": 1643621, "replay": 4959627},
            "identity_trace_encounters": {"on_empty_fallback": 35921, "off_unchanged": 35849},
            "maximum_identity_source_occurrences_across_modes": 3,
            "maximum_identity_rendered_row_occurrences": 2,
            "checkpoint_updates": [51, 94, 102, 153, 188, 204],
            "identity_epoch_boundary_updates": [94, 188],
        },
    },
    "B-traces": {
        "10": {
            "experiment": "B-traces",
            "identity_refusal_cap": "broad512-seed310-refill-r5",
            "identity_original_sources": 33298,
            "identity_rendered_rows": 66596,
            "identity_pool_sha256": "0bedd56030782be7d61f2c48348c044e369810763a974c6a481fca7268e8882f",
            "identity_parent_pool_sha256": "566fc1800e5b3b8115312e5bdbaa135cac08296836f09e37c74c57565ac69f9f",
            "replay_pool_sha256": "affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a",
            "source_selection_sha256": "b109c4f642787b9e3dae87c857283646b83e81fc423efb031117a8550fce49c5",
            "retained_refusal_source_ids_sha256": "61178f0be00429993551ff842d675af78d963d14b4405d42cdcaa2b1e69b5ec9",
            "dropped_source_ids_sha256": "86ec979aaacad57ad9da6496384abba386296cbd38dc07208aa77f10ff462b4e",
            "updates": 144,
            "ordered_occurrence_ledger_sha256": "0b46c96740a32ad420b3614259b77028871622f69434ef15682478a339714da0",
            "final_text_tokens": {"identity": 500021, "replay": 2468028},
            "reasoning_text_tokens": {"identity": 216116, "replay": 2470419},
            "raw_supervised_tokens": {"identity": 763867, "replay": 4950161},
            "identity_trace_encounters": {"on_empty_fallback": 4922, "off_unchanged": 11940, "on_accepted_trace": 7008},
            "maximum_identity_source_occurrences_across_modes": 2,
            "maximum_identity_rendered_row_occurrences": 1,
            "checkpoint_updates": [36, 72, 108, 144],
            "identity_epoch_boundary_updates": [],
        },
        "25": {
            "experiment": "B-traces",
            "identity_refusal_cap": "broad512-seed310-refill-r5",
            "identity_original_sources": 33298,
            "identity_rendered_rows": 66596,
            "identity_pool_sha256": "0bedd56030782be7d61f2c48348c044e369810763a974c6a481fca7268e8882f",
            "identity_parent_pool_sha256": "566fc1800e5b3b8115312e5bdbaa135cac08296836f09e37c74c57565ac69f9f",
            "replay_pool_sha256": "affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a",
            "source_selection_sha256": "b109c4f642787b9e3dae87c857283646b83e81fc423efb031117a8550fce49c5",
            "retained_refusal_source_ids_sha256": "61178f0be00429993551ff842d675af78d963d14b4405d42cdcaa2b1e69b5ec9",
            "dropped_source_ids_sha256": "86ec979aaacad57ad9da6496384abba386296cbd38dc07208aa77f10ff462b4e",
            "updates": 213,
            "ordered_occurrence_ledger_sha256": "f60d310a2bb97a8fca55cf13a3beaa7515588aaeda7a05999021f99a31a1915d",
            "final_text_tokens": {"identity": 1500009, "replay": 2472392},
            "reasoning_text_tokens": {"identity": 655065, "replay": 2475504},
            "raw_supervised_tokens": {"identity": 2298686, "replay": 4959627},
            "identity_trace_encounters": {
                "on_empty_fallback": 14769,
                "off_unchanged": 35849,
                "on_accepted_trace": 21152,
            },
            "maximum_identity_source_occurrences_across_modes": 3,
            "maximum_identity_rendered_row_occurrences": 2,
            "checkpoint_updates": [54, 98, 107, 160, 196, 213],
            "identity_epoch_boundary_updates": [98, 196],
        },
    },
}


def check_capped_exposure(exposure: dict, arm: str) -> None:
    variant = exposure.get("experiment")
    if variant not in CAP_BINDINGS or arm not in CAP_BINDINGS[variant]:
        raise ValueError("Unknown capped variant/arm")
    if exposure.get("special_token_loss_weight") != 1.0:
        raise ValueError("Current capped r5 admission requires explicit all-token loss weight one")
    if exposure["weighted_mass"] != exposure["raw_supervised_tokens"]:
        raise ValueError("Weight-one mass must equal raw supervised-token counts")
    if (
        exposure["family_weighted_mass"] != exposure["family_raw_supervised_tokens"]
        or exposure["identity_trace_weighted_mass"] != exposure["identity_trace_raw_supervised_tokens"]
        or exposure["realized_identity_weighted_share"] != exposure["realized_identity_supervised_share"]
    ):
        raise ValueError("Weight-one family/trace/share accounting differs from raw targets")
    for batch in exposure["cumulative_batches"]:
        if (
            batch["identity_weighted_mass"] != batch["identity_supervised_tokens"]
            or batch["all_weighted_mass"] != batch["all_supervised_tokens"]
            or batch["family_weighted_mass"] != batch["family_raw_supervised_tokens"]
        ):
            raise ValueError("Weight-one cumulative mass differs from actual raw targets")
    expected = CAP_BINDINGS[variant][arm]
    if any(exposure.get(k) != v for k, v in expected.items()):
        raise ValueError("Capped exposure differs from exact reviewed source/ledger/dose")
    minimum = 500000 if arm == "10" else 1500000
    if (
        exposure["identity_minimum_excludes_structural_tokens"] != minimum
        or exposure["final_text_tokens"]["identity"] < minimum
    ):
        raise ValueError("Capped refill does not meet canonical-final minimum")
    if exposure["batch_size"] != GLOBAL_BATCH or exposure["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("Capped packed dimensions changed")
    if exposure["checkpoint_updates"] != checkpoint_cadence(
        exposure["updates"], exposure["identity_epoch_boundary_updates"]
    ):
        raise ValueError("Capped checkpoint cadence differs")
    if len(exposure["cumulative_batches"]) != exposure["updates"]:
        raise ValueError("Capped cumulative exposure length differs")
    last = exposure["cumulative_batches"][-1]
    if (
        last["identity_final_text_tokens"] != exposure["final_text_tokens"]["identity"]
        or last["identity_reasoning_text_tokens"] != exposure["reasoning_text_tokens"]["identity"]
        or last["identity_trace_encounters"] != exposure["identity_trace_encounters"]
    ):
        raise ValueError("Capped final cumulative dose/trace exposure differs")
