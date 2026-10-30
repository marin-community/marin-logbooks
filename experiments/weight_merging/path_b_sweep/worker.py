# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the existing merge CLI and independently admit its CPU readback."""

import argparse
import hashlib
import json
import logging
import math
import runpy
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import torch
from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from rigging.filesystem.s3_compat import StoreType, configure_fsspec_s3, s3_endpoint
from rigging.secrets import SecretBehavior, require_secret
from rigging.timing import Duration

ARTIFACTS = Path(__file__).with_name("artifacts")
LABELS = ("M25", "M49", "M51", "M75")
SOURCE = "s3://marin-rno2a/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a"
SOURCE_AUDIT = "s3://marin-rno2a/marin/experiments/snowball-identity/t310/lr0-r8/audits/source-tensor-hashes.json"
SOURCE_AUDIT_SHA = "af484f49ac2c1d67989495475bcc79bf7519d09281c8087321d906b67139fa97"
FROZEN = frozenset(
    f"model.layers.{layer}.{suffix}"
    for layer in range(26)
    for suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "mlp.router.weight", "mlp.router.bias")
)
SPECIAL = ("model.embed_tokens.weight", "lm_head.weight")
RESOURCES = ResourceConfig.with_cpu(cpu=16, ram="64g", disk="40g", target_cluster="cw-rno2a")
CHUNK = 8 * 1024 * 1024


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def artifact(name: str) -> bytes:
    raw = (ARTIFACTS / name).read_bytes()
    entry = json.loads((ARTIFACTS / "receipt.json").read_text())["files"][name]
    if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
        raise ValueError(f"Committed merge input differs: {name}")
    return raw


def location(uri: str) -> dict:
    value = urlsplit(uri)
    if value.scheme != "s3" or value.netloc != "marin-rno2a":
        raise ValueError("Only the RNO bucket is allowed")
    return {"Bucket": value.netloc, "Key": value.path.lstrip("/")}


def read(client, uri: str, *, etag: str | None = None) -> bytes:
    args = location(uri)
    if etag is not None:
        args["IfMatch"] = etag
    with client.get_object(**args)["Body"] as stream:
        return stream.read()


def write_json(client, uri: str, value: dict) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    client.put_object(**location(uri), Body=raw, ContentType="application/json", IfNoneMatch="*")
    if read(client, uri) != raw:
        raise ValueError(f"Durable readback mismatch: {uri}")


def check_parents(client, *, full_hash: bool) -> dict:
    result = {}
    metadata = {}
    for label in ("A3", "A8a"):
        manifest = json.loads(artifact(f"{label}-preservation.json"))
        root = manifest["destination"].rstrip("/")
        rows = []
        for item in manifest["objects"]:
            uri = root + "/" + item["relative"]
            head = client.head_object(**location(uri))
            if head["ContentLength"] != item["bytes"] or head["ETag"] != item["destination_etag"]:
                raise ValueError(f"Parent object changed: {uri}")
            if full_hash:
                h = hashlib.sha256()
                with client.get_object(**location(uri), IfMatch=head["ETag"])["Body"] as stream:
                    for chunk in iter(lambda: stream.read(CHUNK), b""):
                        h.update(chunk)
                if h.hexdigest() != item["destination_sha256"]:
                    raise ValueError(f"Parent SHA differs: {uri}")
            rows.append(
                {
                    "relative": item["relative"],
                    "bytes": item["bytes"],
                    "etag": head["ETag"],
                    "sha256": item["destination_sha256"] if full_hash else None,
                }
            )
            if item["relative"] in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "model.safetensors.index.json",
            ):
                raw = read(client, uri, etag=head["ETag"])
                if digest(raw) != item["destination_sha256"]:
                    raise ValueError(f"Parent metadata SHA differs: {uri}")
                metadata[label, item["relative"]] = raw
        if len(rows) != 56 or sum(x["is_indexed_shard"] for x in manifest["objects"]) != 39:
            raise ValueError("Parent preservation inventory differs")
        result[label] = rows
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ):
        if metadata["A3", name] != metadata["A8a", name]:
            raise ValueError(f"Parent semantics/shape-index differ: {name}")
    return result


def tensor_hash(tensor: torch.Tensor) -> str:
    return digest(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())


def changed_counts(anchor: torch.Tensor, donor: torch.Tensor, output: torch.Tensor) -> dict:
    """Numeric changed-entry fractions and bit fractions; reduce in bounded chunks."""
    if (
        anchor.shape != donor.shape
        or anchor.shape != output.shape
        or anchor.dtype != donor.dtype
        or anchor.dtype != output.dtype
    ):
        raise ValueError("Merge readback shape/dtype differs from its parents")
    totals = {
        "entries": output.numel(),
        "changed_vs_A3": 0,
        "changed_vs_A8a": 0,
        "parents_differ": 0,
        "changed_vs_A3_on_parent_different": 0,
        "changed_vs_A8a_on_parent_different": 0,
        "bit_changed_vs_A3": 0,
        "bit_changed_vs_A8a": 0,
        "delta_l2_sumsq_vs_A3": 0.0,
        "delta_l2_sumsq_vs_A8a": 0.0,
    }
    aa, bb, cc = (x.reshape(-1) for x in (anchor, donor, output))
    for start in range(0, cc.numel(), 1024 * 1024):
        a, b, c = (x[start : start + 1024 * 1024] for x in (aa, bb, cc))
        different = a != b
        totals["changed_vs_A3"] += int((c != a).sum())
        totals["changed_vs_A8a"] += int((c != b).sum())
        totals["parents_differ"] += int(different.sum())
        totals["changed_vs_A3_on_parent_different"] += int(((c != a) & different).sum())
        totals["changed_vs_A8a_on_parent_different"] += int(((c != b) & different).sum())
        bit_dtype = torch.int32 if c.element_size() == 4 else torch.int16
        totals["bit_changed_vs_A3"] += int((c.view(bit_dtype) != a.view(bit_dtype)).sum())
        totals["bit_changed_vs_A8a"] += int((c.view(bit_dtype) != b.view(bit_dtype)).sum())
        # Subtract in FP32, square/reduce in FP64, then sum bounded chunk scalars.
        totals["delta_l2_sumsq_vs_A3"] += float((c.float() - a.float()).double().square().sum())
        totals["delta_l2_sumsq_vs_A8a"] += float((c.float() - b.float()).double().square().sum())
    return with_fractions(totals)


def with_fractions(totals: dict) -> dict:
    result = dict(totals)
    for name in ("changed_vs_A3", "changed_vs_A8a", "parents_differ", "bit_changed_vs_A3", "bit_changed_vs_A8a"):
        result[name + "_fraction"] = totals[name] / totals["entries"]
    for parent in ("A3", "A8a"):
        name = "changed_vs_" + parent + "_on_parent_different"
        result[name + "_fraction"] = totals[name] / totals["parents_differ"] if totals["parents_differ"] else None
        result["delta_l2_vs_" + parent] = math.sqrt(totals["delta_l2_sumsq_vs_" + parent])
    return result


def check_parent_protection(recipe: dict, source_tensors: dict) -> dict:
    """Admit original frozen fields and special rows before native restoration."""
    readers = [CheckpointReader(CheckpointSource(**r)) for r in [recipe["anchor"], *recipe["donors"]]]
    source = CheckpointReader(CheckpointSource(SOURCE, SOURCE_AUDIT_SHA))
    hashes = {}
    for name in sorted(FROZEN | set(SPECIAL), key=lambda n: (readers[0].weight_map[n], n)):
        tensors = [reader.tensor(name) for reader in readers]
        if name in FROZEN:
            expected = source_tensors[name]
            dtype = torch.float32 if name.endswith(".router.bias") else torch.bfloat16
            if any(
                t.dtype != dtype or list(t.shape) != expected["shape"] or tensor_hash(t) != expected["native_sha256"]
                for t in tensors
            ):
                raise ValueError(f"Unrestored parent frozen field differs from v0.1: {name}")
            if recipe["preserve_rows"][name] != list(range(expected["shape"][0])):
                raise ValueError(f"Full-field restoration range differs: {name}")
            hashes[name] = expected["native_sha256"]
        else:
            original = source.tensor(name)[128000:128256]
            sha = tensor_hash(original)
            if any(tensor_hash(t[128000:128256]) != sha for t in tensors):
                raise ValueError(f"Unrestored parent 256 rows differ from v0.1: {name}")
            hashes[name] = sha
    if set(recipe["preserve_rows"]) != FROZEN | set(SPECIAL):
        raise ValueError("Restoration must cover exactly frozen104 and two special-row tensors")
    return {
        "parent_bits_admitted_before_restoration": hashes,
        "full_fields": 104,
        "router_biases_fp32": 26,
        "restoration": "existing preserve_rows restores anchor bytes after FP32 arithmetic and single cast",
    }


def readback(recipe: dict, source_tensors: dict) -> dict:
    readers = [CheckpointReader(CheckpointSource(**r)) for r in [recipe["anchor"], *recipe["donors"]]]
    output = CheckpointReader(CheckpointSource(recipe["output"], "actual-merge-readback"))
    names = set(output.weight_map)
    if len(names) != 502 or names != set(source_tensors) or any(set(r.weight_map) != names for r in readers):
        raise ValueError("Expected exactly the source's 502 tensor names")
    rows = {}
    source = CheckpointReader(CheckpointSource(SOURCE, SOURCE_AUDIT_SHA))
    router_biases = []
    for name in sorted(names, key=lambda n: (readers[0].weight_map[n], n)):
        a, b = (r.tensor(name) for r in readers)
        c = output.tensor(name)
        expected_dtype = torch.float32 if name.endswith(".router.bias") else torch.bfloat16
        if c.dtype != expected_dtype or list(c.shape) != source_tensors[name]["shape"] or not torch.isfinite(c).all():
            raise ValueError(f"Invalid output dtype/shape/finiteness: {name}")
        sha = tensor_hash(c)
        if name in FROZEN and (
            sha != source_tensors[name]["native_sha256"] or tensor_hash(a) != sha or tensor_hash(b) != sha
        ):
            raise ValueError(f"Frozen field differs from v0.1: {name}")
        if name in SPECIAL:
            original = source.tensor(name)[128000:128256]
            expected = tensor_hash(original)
            if any(tensor_hash(x[128000:128256]) != expected for x in (a, b, c)):
                raise ValueError(f"Protected 256 rows differ from v0.1: {name}")
        if name.endswith(".router.bias"):
            router_biases.append(name)
        rows[name] = {"shape": list(c.shape), "dtype": str(c.dtype), "sha256": sha, **changed_counts(a, b, c)}
    if len(router_biases) != 26 or len(FROZEN) != 104:
        raise ValueError("Router/freeze count differs")
    keys = (
        "entries",
        "changed_vs_A3",
        "changed_vs_A8a",
        "parents_differ",
        "changed_vs_A3_on_parent_different",
        "changed_vs_A8a_on_parent_different",
        "bit_changed_vs_A3",
        "bit_changed_vs_A8a",
        "delta_l2_sumsq_vs_A3",
        "delta_l2_sumsq_vs_A8a",
    )
    totals = with_fractions({k: sum(r[k] for r in rows.values()) for k in keys})
    subsets = {}
    for label, selected in (
        ("frozen104", FROZEN),
        ("router_bias26", frozenset(router_biases)),
        ("other398", names - FROZEN),
    ):
        subsets[label] = with_fractions({k: sum(rows[n][k] for n in sorted(selected)) for k in keys})
    return {
        "tensors": rows,
        "global": totals,
        "subsets": subsets,
        "delta_norm_convention": (
            "L2 of FP32(output-parent); square/reduce in FP64 per1048576-entry chunk; "
            "Python float sums across chunks/tensors"
        ),
        "router_biases_fp32": router_biases,
        "source_frozen_fields": 104,
        "protected_rows_per_tensor": 256,
        "training_checkpoint_provenance": False,
    }


def run(label: str, code_revision: str) -> None:
    import boto3  # noqa: PLC0415 - installed in the Iris worker environment
    from botocore.config import Config  # noqa: PLC0415

    if label not in LABELS or len(code_revision) != 40:
        raise ValueError("Require an immutable four-arm merge identity")
    key, secret = (require_secret(k) for k in ("CW_KEY_ID", "CW_KEY_SECRET"))
    endpoint = s3_endpoint(StoreType.COREWEAVE)
    configure_fsspec_s3(endpoint, key=key, secret=secret)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(connect_timeout=30, read_timeout=300, retries={"max_attempts": 3}),
    )
    recipe_path = ARTIFACTS / f"{label}.json"
    recipe = json.loads(artifact(recipe_path.name))
    root = recipe["output"].rsplit("/hf", 1)[0]
    probe = root + "/write-probe-" + uuid.uuid4().hex
    body = uuid.uuid4().bytes + uuid.uuid4().bytes
    client.put_object(**location(probe), Body=body, IfNoneMatch="*")
    if read(client, probe) != body:
        raise ValueError("Fresh RNO write probe did not roundtrip")
    parents = check_parents(client, full_hash=False)
    source_raw = read(client, SOURCE_AUDIT)
    if digest(source_raw) != SOURCE_AUDIT_SHA:
        raise ValueError("The v0.1 source audit differs")
    source_tensors = json.loads(source_raw)["tensors"]
    protection = check_parent_protection(recipe, source_tensors)
    argv = [
        "experiments.weight_merging.merge",
        "--recipe",
        str(recipe_path),
        "--threads",
        "16",
        "--code-revision",
        code_revision,
    ]
    write_json(
        client,
        root + "/merge-start.json",
        {"label": label, "argv": argv, "parents": parents, "protection": protection, "code_revision": code_revision},
    )
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_module("experiments.weight_merging.merge", run_name="__main__")
        report = readback(recipe, source_tensors)
        report["pre_arithmetic_parent_protection"] = protection
        report["parents_full_hash_readback"] = check_parents(client, full_hash=True)
        manifest_raw = read(client, recipe["output"] + "/merge-manifest.json")
        manifest = json.loads(manifest_raw)
        for item in manifest["objects"]:
            uri = recipe["output"] + "/" + item["name"]
            # One output tensor per shard; stream hashes separately from numerical reads.
            h, size = hashlib.sha256(), 0
            with client.get_object(**location(uri))["Body"] as stream:
                for chunk in iter(lambda: stream.read(CHUNK), b""):
                    size += len(chunk)
                    h.update(chunk)
            if size != item["bytes"] or h.hexdigest() != item["sha256"]:
                raise ValueError(f"Output object SHA/length mismatch: {uri}")
        report.update(
            label=label,
            output=recipe["output"],
            code_revision=code_revision,
            recipe=recipe,
            merge_manifest_sha256=digest(manifest_raw),
            admitted=True,
            provenance_kind="linear-cpu-merge",
            training_event=None,
            master_cast_audit=None,
        )
        write_json(client, root + "/merge-admission.json", report)
    except BaseException as exc:
        write_json(
            client,
            root + "/merge-failure.json",
            {"label": label, "error_type": type(exc).__name__, "message": str(exc), "admitted": False},
        )
        raise
    finally:
        sys.argv = old_argv


def submission(label: str, code_revision: str, credentials: dict[str, str]) -> dict:
    if label not in LABELS:
        raise ValueError("Only M25/M49/M51/M75")
    artifact(f"{label}.json")
    return dict(
        name=f"pathb-cpu-merge-{label.lower()}-r1-{code_revision[:12]}",
        entrypoint=Entrypoint.from_callable(run, label=label, code_revision=code_revision),
        resources=convert_resources(RESOURCES),
        constraints=convert_constraints(RESOURCES),
        environment=EnvironmentSpec(
            extras=("cpu",),
            pip_packages=("boto3==1.41.5", "botocore==1.41.5", "s3transfer==0.15.0"),
            secrets={k: credentials[k] for k in ("CW_KEY_ID", "CW_KEY_SECRET")},
            secret_behavior=SecretBehavior.ISOLATE,
        ),
        replicas=1,
        timeout=Duration.from_hours(12),
        max_retries_failure=0,
        max_retries_preemption=0,
        priority_band=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=LABELS, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        print(artifact(f"{args.label}.json").decode())
    else:
        run(args.label, args.code_revision)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
