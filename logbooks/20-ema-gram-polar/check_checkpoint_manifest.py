# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression fixture for topology-dependent checkpoint storage layouts."""
import copy

from checkpoint_manifest import validate_manifest

reference = {
    "format_version": 1,
    "array_driver": "zarr3",
    "kvstore_driver": "ocdbt",
    "arrays": [
        {"path": "step", "shape": [], "dtype": "int32", "chunk_shape": []},
        {"path": "model/weight", "shape": [11, 4096, 1024], "dtype": "float32", "chunk_shape": [11, 4096, 256]},
        {"path": "opt/gram", "shape": [11, 4096, 4096], "dtype": "float32", "chunk_shape": [11, 1024, 4096]},
    ],
}
actual = copy.deepcopy(reference)
actual["arrays"][1]["chunk_shape"] = [11, 1024, 256]
actual["arrays"][2]["chunk_shape"] = [11, 256, 4096]
result = validate_manifest(actual, reference)
assert result["array_schema_matches"] and result["changed_chunk_paths"] == ["model/weight", "opt/gram"]
for case in ["missing_gram", "duplicate_path", "wrong_shape", "wrong_dtype", "bad_chunks", "scalar_chunks"]:
    invalid = copy.deepcopy(actual)
    if case == "missing_gram":
        invalid["arrays"].pop()
    elif case == "duplicate_path":
        invalid["arrays"].append(invalid["arrays"][-1])
    elif case == "wrong_shape":
        invalid["arrays"][2]["shape"][-1] = 8192
    elif case == "wrong_dtype":
        invalid["arrays"][2]["dtype"] = "bfloat16"
    elif case == "bad_chunks":
        invalid["arrays"][2]["chunk_shape"][1] = 255
    else:
        invalid["arrays"][0]["chunk_shape"] = [1]
    try:
        validate_manifest(invalid, reference)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid manifest accepted: " + case)
print("Topology-dependent chunks accepted; missing/duplicate arrays, shape/dtype changes and invalid chunks rejected.")
