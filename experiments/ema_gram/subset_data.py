# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize a bounded, frozen EMA-Gram subset with the native cache writer."""

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import wandb
from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.store.cache import CacheLedger, CacheMetadata, TreeCache, _merge_sharded_ledgers, write_levanter_cache
from levanter.tokenizers import _stage_tokenizer, load_tokenizer
from marin.datakit.download.http_session import build_retrying_session
from marin.datakit.download.zstd_jsonl import iter_jsonl_from_zstd_stream
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret

MANIFEST = Path("logbooks/20-ema-gram-polar/subset-inputs.json")


def exemplar():
    return {"input_ids": np.empty(0, dtype=np.int32)}


def digest_ids(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def tokenizer_identity(name):
    directory = Path(_stage_tokenizer(name))
    return {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    }


def native_processor(name):
    return BatchTokenizer(
        load_tokenizer(name),
        enforce_bos=False,
        enforce_eos=True,
        padding=False,
        max_length=131072,
        long_string_workaround=False,
    )


def bounded_documents(records, target, stats):
    """Keep whole documents through the first crossing of a fixed token budget."""
    total = 0
    count = 0
    digest = hashlib.sha256()
    for record in records:
        ids = np.asarray(record["input_ids"], dtype=np.int32)
        if not len(ids):
            continue
        digest.update(len(ids).to_bytes(8, "little"))
        digest.update(ids.astype("<i4", copy=False).tobytes())
        total += len(ids)
        count += 1
        stats.update(tokens=total, documents=count, token_sha256=digest.hexdigest())
        yield {"input_ids": ids}
        if total >= target:
            return
    raise ValueError(f"Source exhausted at {total} tokens before target {target}")


def raw_records(source, stats):
    # These are connection/read health timeouts, not an experiment wallclock limit.
    with build_retrying_session(total=5, backoff_factor=4.0) as session:
        if source.get("lfs_sha256"):
            head = session.head(source["url"], allow_redirects=False, timeout=(30, 90))
            head.raise_for_status()
            if head.headers.get("X-Linked-Etag", "").strip('"') != source["lfs_sha256"]:
                raise ValueError("Pinned original-text mirror LFS identity changed")
            stats["lfs_pointer_verified"] = True
            stats["lfs_sha256"] = source["lfs_sha256"]
        with session.get(source["url"], stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            if source.get("etag") and response.headers.get("ETag") != source["etag"]:
                raise ValueError("Source ETag changed since publication")
            stats["etag"] = response.headers.get("ETag")
            stats["source_url"] = source["url"]
            yield from iter_jsonl_from_zstd_stream(response.raw)


def public_records(source, processor, stats):
    source_ids = hashlib.sha256()
    raw = raw_records(source, stats)
    try:
        while batch := list(itertools.islice(raw, 32)):
            for item in batch:
                if not isinstance(item["text"], str) or not item["warc_record_id"]:
                    raise ValueError("Original Nemotron document schema changed")
                source_ids.update((str(item["warc_record_id"]) + "\n").encode())
            stats["decoded_id_sha256"] = source_ids.hexdigest()
            yield from processor(batch)
    finally:
        raw.close()


def cached_records(source):
    cache = TreeCache.load(source, exemplar())
    for start in range(0, len(cache), 32):
        yield from cache.get_batch_sync(range(start, min(start + 32, len(cache))))


def assemble_component(paths, output, processor):
    if StoragePath(output + "/shard_ledger.json").exists():
        raise ValueError("Refusing to overwrite an assembled dataset")
    metadata = CacheMetadata(processor.metadata)
    ledgers = [CacheLedger.load(path, metadata) for path in paths]
    assert all(ledger.is_finished for ledger in ledgers)
    counts = [{"input_ids": TreeCache.load(path, exemplar()).flat_field_length("input_ids")} for path in paths]
    return _merge_sharded_ledgers(output, paths, ledgers, counts, metadata)


def self_check(processor, manifest, run):
    texts = ["A reproducible optimizer experiment.", "中文与 Unicode: \u03b1 \u03b2 \u03b3.", "", "word " * 2300]
    encoded = processor([{"text": text} for text in texts])
    again = processor([{"text": text} for text in texts])
    assert [digest_ids(x["input_ids"]) for x in encoded] == [digest_ids(x["input_ids"]) for x in again]
    for text, record in zip(texts, encoded, strict=True):
        expected = processor.tokenizer.encode(text + " " + processor.tokenizer.eos_token, add_special_tokens=False)
        assert record["input_ids"] == expected
    stats = {}
    target = len(encoded[0]["input_ids"]) + 1
    chosen = list(bounded_documents(iter(encoded), target, stats))
    assert len(chosen) == 2 and stats["tokens"] >= target
    with tempfile.TemporaryDirectory(prefix="ema-gram-subset-check-") as tmp:
        output = str(Path(tmp) / "train")
        write_levanter_cache(iter(chosen), output, metadata=processor.metadata, batch_size=2)
        loaded = TreeCache.load(output, exemplar())
        assert loaded.ledger.metadata.preprocessor_metadata == processor.metadata
        roundtrip = loaded.get_batch_sync(range(len(chosen)))
        assert [digest_ids(x["input_ids"]) for x in roundtrip] == [digest_ids(x["input_ids"]) for x in chosen]
        root = str(Path(tmp) / "merged")
        copied = build_task(
            {"sources": [{"cache": output, "target_tokens": stats["tokens"]}]}, processor, root + "/one", run
        )
        assert copied["full_native_readback"]
        write_levanter_cache(iter(chosen), root + "/two", metadata=processor.metadata, batch_size=2)
        assemble_component([root + "/one", root + "/two"], root, processor)
        merged = TreeCache.load(root, exemplar())
        assert [digest_ids(x["input_ids"]) for x in merged.get_batch_sync(range(4))] == [
            digest_ids(x["input_ids"]) for x in chosen * 2
        ]
        expected = np.concatenate([x["input_ids"] for x in chosen * 2])
        flat = asyncio.run(merged.get_flat_field_batch("input_ids", [0], len(expected)))[0]
        assert np.array_equal(flat, expected)
        assert merged.flat_field_length("input_ids") == len(expected)
    try:
        list(bounded_documents(iter(encoded[:1]), target, {}))
    except ValueError:
        pass
    else:
        raise AssertionError("Truncated source was accepted")
    return dict(
        passed=True, exact_native_tokens=True, deterministic=True, native_roundtrip=True, whole_document_budget=True
    )


def build_task(task, processor, output, run):
    stats = []

    def records():
        for index, source in enumerate(task["sources"]):
            item = dict(target_tokens=source["target_tokens"])
            stats.append(item)
            stream = public_records(source, processor, item) if "url" in source else cached_records(source["cache"])
            try:
                yield from bounded_documents(stream, source["target_tokens"], item)
            finally:
                stream.close()
            run.log({"subset/source_completed": index + 1, "subset/tokens": sum(x.get("tokens", 0) for x in stats)})

    metadata = processor.metadata
    result = write_levanter_cache(records(), output, metadata=metadata, batch_size=32)
    cache = TreeCache.load(output, exemplar())
    assert len(cache) == result["count"]
    # Verify every stored token against the selected source stream, in-region.
    readback = cached_records(output)
    for source_stats in stats:
        checked = {}
        list_count = 0
        for _ in bounded_documents(
            itertools.islice(readback, source_stats["documents"]), source_stats["tokens"], checked
        ):
            list_count += 1
        assert list_count == source_stats["documents"]
        assert checked == {k: source_stats[k] for k in ("tokens", "documents", "token_sha256")}
    assert next(readback, None) is None
    edge = cache.get_batch_sync([0, len(cache) - 1])
    return dict(
        sources=stats,
        rows=len(cache),
        tokens=sum(x["tokens"] for x in stats),
        output=output,
        edge_sha256=[digest_ids(x["input_ids"]) for x in edge],
        full_native_readback=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("logging-check", "self-check", "probe", "build", "assemble"), required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    identity = f"ema-gram-subset-{args.mode}-{uuid.uuid4().hex[:12]}"
    report = dict(
        identity=identity,
        source_commit=os.environ.get("GIT_COMMIT"),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        mode=args.mode,
    )
    run = wandb.init(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group="ema-gram-20-fixed-subset",
        mode="online",
        resume="never",
        settings={"api_key": require_secret("WANDB_API_KEY")},
        config=report,
    )
    assert run.settings.mode == "online"
    print(f"W&B: {run.url}", flush=True)
    exit_code = 1
    destination = None
    try:
        run.log({"logging_check/success": 1})
        if args.mode != "logging-check":
            processor = native_processor(manifest["tokenizer"])
            report["tokenizer_files"] = tokenizer_identity(manifest["tokenizer"])
            if manifest.get("tokenizer_files") and report["tokenizer_files"] != manifest["tokenizer_files"]:
                raise ValueError("Tokenizer bytes differ from the frozen manifest")
            report["processor"] = processor.metadata
            if args.mode == "self-check":
                report["checks"] = self_check(processor, manifest, run)
            elif args.mode == "probe":
                report["sources"] = []
                for task in manifest["tasks"]:
                    if task["shard"] != 0 or "url" not in task["sources"][0]:
                        continue
                    stats = dict(component=task["component"])
                    report["sources"].append(stats)
                    stream = public_records(task["sources"][0], processor, stats)
                    try:
                        sample = list(itertools.islice(stream, 2))
                    finally:
                        stream.close()
                    assert len(sample) == 2
                    stats["sample_tokens"] = [len(x["input_ids"]) for x in sample]
                    stats["sample_sha256"] = [digest_ids(x["input_ids"]) for x in sample]
                    print(json.dumps(stats), flush=True)
            elif args.mode == "build":
                if not manifest.get("tokenizer_files"):
                    raise ValueError("Production requires frozen tokenizer bytes")
                task = manifest["tasks"][args.index]
                destination = StoragePath(manifest["output_prefix"]) / task["component"] / "train" / identity
                report["task"] = task
                report["result"] = build_task(task, processor, str(destination), run)
            else:
                results = json.loads(args.results.read_text())
                assert len(results) == len(manifest["tasks"])
                grouped = {}
                for task, item in zip(manifest["tasks"], results, strict=True):
                    assert item["success"] and item["task"] == task
                    assert item["manifest_sha256"] == report["manifest_sha256"]
                    assert item["result"]["full_native_readback"]
                    assert item["canonical_success"] and item["wandb_finished"]
                    grouped.setdefault(task["component"], []).append(item["result"]["output"])
                report["components"] = {}
                for name, paths in grouped.items():
                    root = manifest["output_prefix"] + "/" + name + "/train"
                    ledger = assemble_component(paths, root, processor)
                    report["components"][name] = dict(cache_root=root.rsplit("/", 1)[0], ledger=ledger.to_dict())
        report["success"] = True
        run.log({"subset/success": 1})
        exit_code = 0
    except Exception as error:
        report.update(success=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if destination is not None:
            destination.mkdirs()
            (destination / "report.json").write_text(json.dumps(report, indent=2))
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
