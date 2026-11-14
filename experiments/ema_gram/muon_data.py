# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply an explicit, verified FineWeb replacement to frozen Muon probes."""

import hashlib
import json
from dataclasses import replace

from google.cloud import storage
from rigging.filesystem.buckets import filesystem_for


def replace_muon_cache(cfg, cache_root):
    name = "fineweb-edu-10B"
    if {k: v for k, v in cfg.data.train_weights.items() if v} != {name: 1.0}:
        raise ValueError("Expected the original single-corpus FineWeb training mixture")
    component = cfg.data.components[name]
    components = dict(cfg.data.components)
    components[name] = replace(component, cache_dir=cache_root, source=replace(component.source, cache_dir=cache_root))
    return replace(cfg, data=replace(cfg.data, components=components, auto_build_caches=False))


def recovered_muon_data(cfg, recovery):
    fs, path = filesystem_for(recovery["manifest_uri"])
    with fs.open(path, "rb") as stream:
        content = stream.read()
    if hashlib.sha256(content).hexdigest() != recovery["manifest_sha256"]:
        raise ValueError("Frozen data recovery manifest changed")
    manifest = json.loads(content)
    if manifest["cache_root"] != recovery["cache_root"]:
        raise ValueError("Data recovery cache root mismatch")
    bucket, prefix = manifest["cache_root"].removeprefix("gs://").split("/", 1)
    objects = {
        blob.name[len(prefix) + 1 :]: blob for blob in storage.Client().list_blobs(bucket, prefix=prefix + "/train/")
    }
    if set(objects) != {item["path"] for item in manifest["files"]}:
        raise ValueError("Recovered cache object set changed")
    for item in manifest["files"]:
        blob = objects[item["path"]]
        if (int(blob.generation), blob.size, blob.md5_hash) != (item["generation"], item["bytes"], item["md5"]):
            raise ValueError("Recovered cache object identity changed: " + item["path"])
    return replace_muon_cache(cfg, manifest["cache_root"]), manifest
