# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh, mutually disjoint update/calibration/evaluation panels for geometry tests."""

import hashlib
import json

import haliax as hax
import jax
from google.cloud import storage
from levanter.data.loader import DataLoader

from experiments.ema_gram.hessian import host_array


def checkpoint_identity(uri):
    """Hash a durable checkpoint's object generations and service-side checksums."""
    bucket, prefix = uri.removeprefix("gs://").rstrip("/").split("/", 1)
    client = storage.Client()
    metadata = client.bucket(bucket).blob(prefix + "/metadata.json").download_as_bytes()
    records = [
        dict(name=b.name[len(prefix) + 1 :], generation=b.generation, bytes=b.size, md5=b.md5_hash, crc32c=b.crc32c)
        for b in client.list_blobs(bucket, prefix=prefix + "/")
    ]
    records.sort(key=lambda b: b["name"])
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return dict(
        uri=uri,
        object_manifest_sha256=digest,
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        metadata=json.loads(metadata),
        objects=records,
        scope="SHA256 of object names/generations/sizes and GCS content checksums; not a downloaded full-byte hash",
    )


def geometry_data(cfg, protocol, exclusions, save):
    seen = set(json.loads(exclusions.read_text()))
    records, rejected, panels, weights = [], [], {}, {}
    for role, count in (("calibration", 8), ("evaluation", 16), ("update", 8)):
        seed = protocol["seeds"][role]
        data = cfg.data.train_set(cfg.model.max_Pos, cfg.trainer.batch_schedule, key=jax.random.PRNGKey(seed))
        loader = DataLoader(
            data,
            batch_size=hax.Axis("batch", cfg.trainer.train_batch_size),
            axis_resources=cfg.trainer.compute_axis_mapping,
        )
        panels[role], weights[role] = [], []
        for source_index, batch in enumerate(loader):
            tokens = host_array(batch.tokens.array)
            hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in tokens]
            if len(set(hashes)) != len(hashes) or seen.intersection(hashes):
                rejected.append(dict(role=role, source_index=source_index))
                continue
            seen.update(hashes)
            weight = float(host_array(batch.loss_weight.array)[..., :-1].sum())
            records.append(
                dict(
                    role=role,
                    seed=seed,
                    source_index=source_index,
                    group=len(panels[role]),
                    valid_weight=weight,
                    sequence_hashes=hashes,
                )
            )
            panels[role].append(batch)
            weights[role].append(weight)
            if len(panels[role]) == count:
                break
        if len(panels[role]) != count:
            raise ValueError("Insufficient disjoint text")
    manifest = dict(
        records=records,
        rejected=rejected,
        exclusion_sha256=hashlib.sha256(exclusions.read_bytes()).hexdigest(),
        scope="New complete-sequence-disjoint panels; original training exposure not excluded",
    )
    save("data-manifest.json", manifest)
    return panels, weights, hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
