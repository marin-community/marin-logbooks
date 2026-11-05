# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare checkpoint array schemas while validating topology-dependent chunks."""


def validate_manifest(actual, reference):
    """Require identical arrays and formats, allowing valid storage chunk layouts.

    Levanter's plan_array_write derives chunks from sharding and writer replicas.
    Chunk equality across one-host and four-host runs is therefore not required.
    This checks manifest metadata only, not stored array bytes or restoration.
    """
    if {k: v for k, v in actual.items() if k != "arrays"} != {k: v for k, v in reference.items() if k != "arrays"}:
        raise ValueError("Checkpoint manifest format differs")
    schemas = []
    layouts = []
    for manifest in [actual, reference]:
        schema, chunks = {}, {}
        for array in manifest["arrays"]:
            path = array["path"]
            if path in schema:
                raise ValueError("Duplicate checkpoint array path")
            shape, chunk = array["shape"], array["chunk_shape"]
            if len(shape) != len(chunk) or any(
                type(size) is not int
                or type(part) is not int
                or size < 0
                or part < 1
                or (size == 0 and part != 1)
                or (size > 0 and (part > size or size % part))
                for size, part in zip(shape, chunk, strict=True)
            ):
                raise ValueError("Invalid checkpoint chunk geometry")
            schema[path] = {k: v for k, v in array.items() if k != "chunk_shape"}
            chunks[path] = chunk
        schemas.append(schema)
        layouts.append(chunks)
    if schemas[0] != schemas[1]:
        raise ValueError("Checkpoint array paths, shapes or dtypes differ")
    return {
        "array_schema_matches": True,
        "chunk_geometry_valid": True,
        "changed_chunk_paths": sorted(path for path in layouts[0] if layouts[0][path] != layouts[1][path]),
        "scope": "Manifest metadata only; topology-dependent chunks retained, no full array read or restore.",
    }
