# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.identity_t2.relocate import encoded, fresh_root, relocate_paths


def test_relocated_storage_preserves_shuffle_component_order_and_payload_values():
    components = {
        "identity": {"source": ["/old/identity.jsonl"], "cache": "/old/cache/identity"},
        "preservation": {"source": ["/old/preservation.jsonl"]},
        "identity_v2": {"source": ["/preview/identity_v2.jsonl"]},
        "tools": {"source": ["/preview/tools.jsonl"]},
        "third_person": {"source": ["/preview/third_person.jsonl"]},
    }
    source = {
        "components": components,
        "cache_catalog": "/preview/cache-catalog.json",
        "weights": [1.2, 2.0, 1.0],
        "template": "prefix /old/cache suffix",
        "neighbor": "/older/untouched",
    }
    actual = json.loads(
        encoded(
            relocate_paths(
                source,
                exact={"/preview/cache-catalog.json": "s3://region/catalog.json"},
                prefixes={"/old": "s3://region/base", "/preview": "s3://region/new"},
            )
        )
    )
    assert list(actual["components"]) == ["identity", "preservation", "identity_v2", "tools", "third_person"]
    assert actual["components"]["identity"]["cache"] == "s3://region/base/cache/identity"
    assert actual["components"]["tools"]["source"] == ["s3://region/new/tools.jsonl"]
    assert actual["cache_catalog"] == "s3://region/catalog.json"
    assert actual["weights"] == [1.2, 2.0, 1.0]
    assert actual["template"] == source["template"] and actual["neighbor"] == source["neighbor"]
    assert source["components"]["tools"]["source"] == ["/preview/tools.jsonl"]


def test_partial_staging_is_not_overwritten(tmp_path):
    root = tmp_path / "freeze"
    assert fresh_root(str(root))["VOID"] is True
    root.mkdir()
    partial = root / "cache-chunk"
    partial.write_bytes(b"previous partial upload")
    with pytest.raises(FileExistsError):
        fresh_root(str(root))
    assert partial.read_bytes() == b"previous partial upload"
