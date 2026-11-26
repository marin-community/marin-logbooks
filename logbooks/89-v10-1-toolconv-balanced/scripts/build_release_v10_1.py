"""Build and upload the V10.1 release directory, then write its frozen publisher manifest.

V10.1 = the sealed V10 release with exactly two changes:
(1) the three math sources' stores are re-versioned at ``@toolconv-20261125``:
    4,383 verified tool-converted rows replace their originals (the apply
    receipt carries the per-shard map; unchanged shards keep their V10 shard
    paths re-expressed relative to the catalog root). The supervision stays
    V10's ``chat_assistant_only`` — the all-position loss for Exp7 is applied
    by the LLOM trainer at consumption.
(2) one new component ``sft/source/balanced_reasoning/v1`` (7,728 rows,
    4,070,781 tokens) enters the mixture at weight 0.005 per phase; the SFT
    weights renormalize by (S - 0.005)/S and the ``pretrain/`` replay share
    stays at V10's halved 0.10037232630869815.

Every other source, the supervision sidecar's inherited entries and the epoch
construction are byte-identical to V10. The manifest freezes before the
native-consumer receipt exists (training_ready false), as in V7/V8/V9/V10.
"""

import argparse
import csv
import hashlib
import io
import json

from rigging.filesystem.buckets import filesystem_for

V10_RELEASE_URI = "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.23-v10-overthinking/release-v10/release-complete-v10.json"
V10_RELEASE_SHA = "e0838c00f77bacb909faf9dfb62277a616c889c09bba0663ea7bc91b638c7411"
CATALOG_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer"
CONSUMER_REVISION = "6db3ce9022a414804142d02f6edd2fe47befc632"
REFERENCE_GEOMETRY = {"updates": 746, "global_batch": 256, "sequence_length": 262144,
                      "seed": 0, "packing_cap": 256, "lookahead": 128}
MATH_SOURCES = (
    "sft/source/nemotron_sft_v3/math_v2/high",
    "sft/source/nemotron_sft_v3/math_v3/train",
    "sft/source/nemotron_sft_v3/math_v4/train",
)
EXPECTED_REPLACED = {
    "sft/source/nemotron_sft_v3/math_v2/high": 1271,
    "sft/source/nemotron_sft_v3/math_v3/train": 2388,
    "sft/source/nemotron_sft_v3/math_v4/train": 724,
}
BALANCED_SOURCE = "sft/source/balanced_reasoning/v1"
BALANCED_WEIGHT = 0.005
BALANCED_ROWS_SHA = "6b5dc656130f4f946a793172ef4495d75f67c1d26e30ad813d0b42e3756ec9be"
DECLARED_REPLAY = 0.10037232630869815
TOOLCONV_COVERAGE = {
    "applied": 4383,
    "pre_applied_at_build": 46314,
    "unmappable_to_v10_lineage": 188469,
    "quarantined_tokenizer_mismatch": 10654,
    "verified_modified_2026_11_25_index_snapshot": 2378592,
}


def read_bytes(uri: str) -> bytes:
    fs, key = filesystem_for(uri)
    return fs.cat(key)


def write_bytes(uri: str, content: bytes) -> str:
    fs, key = filesystem_for(uri)
    fs.pipe(key, content)
    return hashlib.sha256(content).hexdigest()


def pinned_json(uri: str, sha: str, name: str) -> dict:
    content = read_bytes(uri)
    got = hashlib.sha256(content).hexdigest()
    if got != sha:
        raise ValueError(f"{name}: digest {got} differs from the pin {sha}")
    return json.loads(content)


def to_csv(rows: list[dict], columns: list[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def build_catalog(v10_catalog: dict, apply_receipt: dict, balanced_ledger: dict) -> dict:
    """V10's catalog with the three math entries re-keyed per the apply receipt and
    the balanced component added; every other entry byte-identical."""
    catalog = json.loads(json.dumps(v10_catalog))
    train = catalog["splits"]["train"]
    for source in MATH_SOURCES:
        old = train[source]
        data = apply_receipt["sources"][source]
        if data["base"]["rows"] != old["ledger"]["total_num_rows"]:
            raise ValueError(f"{source}: the apply receipt's base rows disagree with the V10 catalog")
        if data["replaced_rows"] != EXPECTED_REPLACED[source]:
            raise ValueError(f"{source}: the apply receipt replaced {data['replaced_rows']} rows")
        if data["output"]["rows"] != data["base"]["rows"] or data["output"]["row_delta"] != 0:
            raise ValueError(f"{source}: the applied population does not conserve")
        shard_rows = {}
        field_counts = {}
        for old_key, shard in data["shard_map"].items():
            new_key = shard["root_relative_key"]
            uri = f"{CATALOG_ROOT}/{new_key}"
            if uri != shard["cache_uri"]:
                raise ValueError(f"{source}: shard {old_key} does not resolve under the catalog root")
            replaced = shard.get("replaced_rows", 0)
            if replaced == 0:
                if uri != f"{old['cache_dir'].rstrip('/')}/{old_key}":
                    raise ValueError(f"{source}: unchanged shard {old_key} moved off its V10 path")
            elif "/stores-v2/" not in uri:
                raise ValueError(f"{source}: rewritten shard {old_key} is not in the stores-v2 tree")
            shard_rows[new_key] = shard["rows"]
            field_counts[new_key] = {"input_ids": shard["tokens"], "loss_weights": shard["tokens"]}
        if set(data["shard_map"].keys()) != set(old["ledger"]["finished_shards"]):
            raise ValueError(f"{source}: the shard map does not cover the V10 shard set")
        if len(shard_rows) != len(old["ledger"]["finished_shards"]):
            raise ValueError(f"{source}: the shard-key count changed ({len(shard_rows)} vs the V10 set)")
        tokens = sum(counts["input_ids"] for counts in field_counts.values())
        if tokens != data["output"]["tokens"]:
            raise ValueError(f"{source}: the per-shard token counts disagree with the receipt")
        train[source] = {
            "cache_dir": CATALOG_ROOT,
            "ledger": {
                "total_num_rows": data["output"]["rows"],
                "shard_rows": shard_rows,
                "is_finished": True,
                "finished_shards": sorted(shard_rows),
                "field_counts": {"input_ids": tokens, "loss_weights": tokens},
                "field_counts_by_shard": field_counts,
                "layout": old["ledger"].get("layout", "sharded"),
                "metadata": old["ledger"]["metadata"],
            },
        }
    balanced_key = "exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/balanced-cache-v1/cache"
    if BALANCED_SOURCE in train:
        raise ValueError("The balanced component already exists in V10")
    counts = balanced_ledger["field_counts"]
    if balanced_ledger["total_num_rows"] != 7728 or counts["input_ids"] != 4070781:
        raise ValueError("The staged balanced cache disagrees with its pinned build receipt")
    if balanced_ledger["metadata"]["preprocessor_metadata"]["loss_mode"] != "chat_assistant_only":
        raise ValueError("The staged balanced cache is not an assistant-only cache")
    if balanced_ledger["is_finished"] is not True:
        raise ValueError("The staged balanced cache is not finished")
    train[BALANCED_SOURCE] = {
        "cache_dir": CATALOG_ROOT,
        "ledger": {
            "total_num_rows": balanced_ledger["total_num_rows"],
            "shard_rows": {balanced_key: balanced_ledger["total_num_rows"]},
            "is_finished": True,
            "finished_shards": [balanced_key],
            "field_counts": dict(counts),
            "field_counts_by_shard": {balanced_key: dict(counts)},
            "layout": "sharded",
            "metadata": balanced_ledger["metadata"],
        },
    }
    return catalog


def build_weights(v10_weights: list) -> list:
    """Every phase: the balanced component at 0.005, SFT renormalized by
    (S - 0.005)/S, pretrain untouched."""
    weights = json.loads(json.dumps(v10_weights))
    for phase, phase_weights in weights:
        sft_sum = sum(value for name, value in phase_weights.items() if not name.startswith("pretrain/"))
        pretrain_sum = sum(value for name, value in phase_weights.items() if name.startswith("pretrain/"))
        if abs(pretrain_sum - DECLARED_REPLAY) > 1e-12:
            raise ValueError(f"phase {phase}: the V10 weights do not carry the halved replay share")
        scale = (sft_sum - BALANCED_WEIGHT) / sft_sum
        new_weights = {}
        for name, value in phase_weights.items():
            if name.startswith("pretrain/"):
                new_weights[name] = value
            else:
                new_weights[name] = value * scale
        if BALANCED_SOURCE in new_weights:
            raise ValueError(f"phase {phase}: the balanced component already carries weight")
        new_weights[BALANCED_SOURCE] = BALANCED_WEIGHT
        total = sum(new_weights.values())
        if abs(total - 1.0) > 1e-12:
            raise ValueError(f"phase {phase}: the V10.1 weights sum to {total}")
        new_pretrain = sum(value for name, value in new_weights.items() if name.startswith("pretrain/"))
        if abs(new_pretrain - pretrain_sum) > 1e-12:
            raise ValueError(f"phase {phase}: the replay share moved")
        phase_weights.clear()
        phase_weights.update(new_weights)
    return weights


def build_native_job(v10_job: dict, members: dict, catalog: dict) -> dict:
    """V10's reference-geometry job over the published V10.1 bytes."""
    job = json.loads(json.dumps(v10_job))
    geometry = {
        "batch_size": 256, "lookahead": 128, "packing_cap": 256, "sequence_length": 262144,
        "steps": 746, "trainer_seed": 0, "data_seed": None, "jax_threefry_partitionable": True,
    }
    for key, value in geometry.items():
        if job[key] != value:
            raise ValueError(f"the V10 job's {key} is not the reference geometry")
    if job["consumer_runtime_revision"] != CONSUMER_REVISION:
        raise ValueError("the V10 job pins another consumer revision")
    job["catalog_uri"] = members["cache_catalog.json"]["uri"]
    job["catalog_sha256"] = members["cache_catalog.json"]["sha256"]
    job["data_uri"] = members["data.json"]["uri"]
    job["data_sha256"] = members["data.json"]["sha256"]
    job["source_loss_uri"] = members["source_loss.json"]["uri"]
    job["source_loss_sha256"] = members["source_loss.json"]["sha256"]
    job["expected_source_populations"] = {
        name: {"documents": entry["ledger"]["total_num_rows"],
               "tokens": entry["ledger"]["field_counts"]["input_ids"]}
        for name, entry in catalog["splits"]["train"].items()
    }
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-receipt-uri", required=True, help="The v2 apply receipt on S3")
    parser.add_argument("--apply-receipt-sha256", required=True)
    parser.add_argument("--balanced-ledger-uri", required=True, help="The staged balanced shard_ledger.json on S3")
    parser.add_argument("--balanced-ledger-sha256", required=True)
    parser.add_argument("--output-prefix", required=True, help="Destination prefix, e.g. s3://.../release-v10.1")
    args = parser.parse_args()

    prefix = args.output_prefix.rstrip("/")
    v10 = pinned_json(V10_RELEASE_URI, V10_RELEASE_SHA, "release-complete-v10.json")
    if v10["training_ready"] is not True:
        raise ValueError("The V10 release is not the qualified seal")
    artifacts = {}
    for name in ("cache_catalog.json", "data.json", "mixture-rationale.json", "retention-by-source.csv",
                 "retention-by-transform.csv", "source_loss.json", "train_weights.json",
                 "native-measurement-job.json"):
        pin = v10["artifacts"][name]
        artifacts[name] = pinned_json(pin["uri"], pin["sha256"], name) if name.endswith(".json") else read_bytes(pin["uri"])
    apply_receipt = pinned_json(args.apply_receipt_uri, args.apply_receipt_sha256, "apply-receipt.json")
    if apply_receipt["status"] != "COMPLETE" or apply_receipt["pins"]["catalog_sha256"] != v10["artifacts"]["cache_catalog.json"]["sha256"]:
        raise ValueError("The apply receipt is not COMPLETE over the V10 catalog")
    balanced_ledger = pinned_json(args.balanced_ledger_uri, args.balanced_ledger_sha256, "shard_ledger.json")

    members = {}

    # 1. the V10.1 cache catalog: only the three math entries re-keyed, the balanced entry added
    catalog = build_catalog(artifacts["cache_catalog.json"], apply_receipt, balanced_ledger)
    train = catalog["splits"]["train"]
    previous_train = artifacts["cache_catalog.json"]["splits"]["train"]
    if len(train) != 434:
        raise ValueError(f"the V10.1 catalog carries {len(train)} sources, expected 434")
    for name, entry in train.items():
        if name in MATH_SOURCES or name == BALANCED_SOURCE:
            continue
        if entry != previous_train[name]:
            raise ValueError(f"{name}: an untouched entry differs from V10")
    catalog_document = json.dumps(catalog, sort_keys=True, indent=1).encode()
    members["cache_catalog.json"] = {
        "uri": f"{prefix}/cache_catalog.json",
        "sha256": write_bytes(f"{prefix}/cache_catalog.json", catalog_document),
    }

    # 2. train_weights.json (balanced at 0.005, SFT renormalized, replay untouched) and data.json
    weights = build_weights(artifacts["train_weights.json"])
    members["train_weights.json"] = {
        "uri": f"{prefix}/train_weights.json",
        "sha256": write_bytes(f"{prefix}/train_weights.json", json.dumps(weights, sort_keys=True, indent=1).encode()),
    }
    data = json.loads(json.dumps(artifacts["data.json"]))
    sft_component = data["components"]["sft/source/nemotron_sft_v3/math_v3/train"]
    data["components"][BALANCED_SOURCE] = json.loads(json.dumps(sft_component))
    data["train_weights"] = weights
    data["cache_catalog"] = members["cache_catalog.json"]["uri"]
    if data["num_validation_sequences"] != artifacts["data.json"]["num_validation_sequences"]:
        raise ValueError("The validation reservations moved")
    if sum(data["num_validation_sequences"].values()) != 56:
        raise ValueError("The validation reservations no longer total 56")
    if data["tokenizer"] != artifacts["data.json"]["tokenizer"]:
        raise ValueError("The tokenizer moved")
    if set(data["components"]) != set(train):
        raise ValueError("The data components and the catalog name different sources")
    data_document = json.dumps(data, sort_keys=True, indent=1).encode()
    members["data.json"] = {"uri": f"{prefix}/data.json", "sha256": write_bytes(f"{prefix}/data.json", data_document)}

    # 3. source_loss.json: V10 plus the balanced entry; every inherited entry byte-identical
    source_loss = json.loads(json.dumps(artifacts["source_loss.json"]))
    if BALANCED_SOURCE in source_loss["sources"]:
        raise ValueError("The balanced component already carries a supervision entry")
    source_loss["sources"][BALANCED_SOURCE] = {
        "loss_mode": "chat_assistant_only",
        "loss_weights_indexing": "prediction",
        "loss_weights_key": "loss_weights",
        "source_revision": BALANCED_ROWS_SHA,
        "teacher_only": False,
    }
    for name, entry in source_loss["sources"].items():
        if name != BALANCED_SOURCE and entry != artifacts["source_loss.json"]["sources"][name]:
            raise ValueError(f"{name}: an inherited supervision entry differs from V10")
    if set(source_loss["sources"]) != set(train):
        raise ValueError("The supervision sidecar and the catalog name different sources")
    source_loss_document = json.dumps(source_loss, sort_keys=True, indent=1).encode()
    members["source_loss.json"] = {
        "uri": f"{prefix}/source_loss.json",
        "sha256": write_bytes(f"{prefix}/source_loss.json", source_loss_document),
    }

    # 4. the mixture rationale with the V10.1 amendments recorded
    rationale = json.loads(json.dumps(artifacts["mixture-rationale.json"]))
    rationale["v10_1_amendments"] = {
        "user_decisions": "2026-11-25: add the balanced thinking and tool-conversion fixes to Mix V10 "
        "(no approval gate; the user removed it at 4:15 PM EST the same day). The SFT trainer config "
        "change (loss on all positions, as Exp1 did) is applied by the Exp7 LLOM consumption view "
        "(TextLmDatasetFormat; PR #12284) and does not rewrite the cached masks.",
        "toolconv": {
            "rule": "verified tool-converted rows REPLACE their unconverted originals in the same "
            "source; every other source stays byte-identical to V10",
            "coverage_boundary": {**TOOLCONV_COVERAGE, "note": "the 2026-11-25 index snapshot predates "
            "the RNO-era archive ingest, so the verified-modified count is a conservative floor; "
            "10,654 rows were quarantined by the tokenizer gate and are NOT silently substituted"},
            "applied_rows": 4383,
            "sources": sorted(MATH_SOURCES),
            "apply_receipt": {"uri": args.apply_receipt_uri, "sha256": args.apply_receipt_sha256},
            "supervision": "V10's chat_assistant_only cached masks are kept; replaced rows recompute "
            "the assistant-only mask with the pinned 2026.09.18 tokenizer; unchanged rows keep V10 "
            "bytes; unchanged shards keep their V10 shard paths",
        },
        "balanced_reasoning": {
            "rule": "one new component sft/source/balanced_reasoning/v1 (7,728 rows, 4,070,781 tokens, "
            "2,565,856 supervised positions) ADDED to the mixture",
            "weight": BALANCED_WEIGHT,
            "rationale": "a targeted thinking-style corrective store; the natural-token share (~4.07M "
            "of ~99.9B SFT tokens ≈ 0.004%) would be too small to influence behavior, while 0.005 "
            "gives ≈250M sampled positions per one-epoch ≈ 7 passes over the store — meaningful "
            "exposure without crowding out a family (STATUS.md §1c)",
            "renormalization": "every non-pretrain/ phase weight scales by (S - 0.005)/S; the pretrain/ "
            "replay share stays at V10's halved 0.10037232630869815",
            "source_revision": BALANCED_ROWS_SHA,
        },
    }
    rationale_document = json.dumps(rationale, sort_keys=True, indent=1).encode()
    members["mixture-rationale.json"] = {
        "uri": f"{prefix}/mixture-rationale.json",
        "sha256": write_bytes(f"{prefix}/mixture-rationale.json", rationale_document),
    }

    # 5. retention tables with the V10.1 populations, the toolconv stage and the new source
    by_source_rows = list(csv.DictReader(io.StringIO(artifacts["retention-by-source.csv"].decode())))
    columns = list(by_source_rows[0].keys())
    kept_rows = []
    for row in by_source_rows:
        if row["component"] in weights[0][1]:
            row["weight"] = repr(weights[0][1][row["component"]])
        if row["component"] in MATH_SOURCES:
            store = apply_receipt["sources"][row["component"]]
            row["output_documents"] = str(store["output"]["rows"])
            row["output_tokens"] = str(store["output"]["tokens"])
            row["retained_serialized_positions"] = str(store["output"]["tokens"])
        kept_rows.append(row)
    balanced_row = {column: "" for column in columns}
    balanced_row.update({
        "source": "balanced_reasoning/v1",
        "component": BALANCED_SOURCE,
        "loss_mode": "chat_assistant_only",
        "source_revision": BALANCED_ROWS_SHA,
        "teacher_only": "False",
        "weight": repr(weights[0][1][BALANCED_SOURCE]),
        "completed_input_documents": "7728",
        "completed_input_tokens": "4070781",
        "final_stage_input_documents": "7728",
        "final_stage_input_tokens": "4070781",
        "input_documents": "7728",
        "input_tokens": "4070781",
        "output_documents": "7728",
        "output_tokens": "4070781",
        "supervised_prediction_positions": "2565856",
        "retained_serialized_positions": "4070781",
    })
    kept_rows.append(balanced_row)
    members["retention-by-source.csv"] = {
        "uri": f"{prefix}/retention-by-source.csv",
        "sha256": write_bytes(f"{prefix}/retention-by-source.csv", to_csv(kept_rows, columns)),
    }

    by_transform_rows = list(csv.DictReader(io.StringIO(artifacts["retention-by-transform.csv"].decode())))
    transform_columns = list(by_transform_rows[0].keys())
    for source in MATH_SOURCES:
        store = apply_receipt["sources"][source]
        source_row = next(r for r in kept_rows if r["component"] == source)
        row = {column: "" for column in transform_columns}
        row.update({
            "source": source_row["source"], "component": source, "loss_mode": source_row["loss_mode"],
            "phase": "v10_1_toolconv", "transform": "toolconv@20261125",
            "input_documents": str(store["base"]["rows"]), "input_tokens": str(store["base"]["tokens"]),
            "output_documents": str(store["output"]["rows"]), "output_tokens": str(store["output"]["tokens"]),
            "changed_documents": str(store["replaced_rows"]),
            "accounting_note": (
                f"V10.1 toolconv: {store['replaced_rows']} verified tool-converted rows replace their "
                "originals in the sealed V10 store; assistant-only supervision preserved; unchanged "
                "rows and shards keep V10 bytes."
            ),
            "input_serialized_positions": str(store["base"]["tokens"]),
            "output_serialized_positions": str(store["output"]["tokens"]),
            "complete": "True",
            "input_population": str(store["base"]["rows"]), "token_measurement": "store_receipt",
        })
        for column, value in row.items():
            if column not in transform_columns and value:
                raise ValueError(f"retention-by-transform column absent: {column}")
        by_transform_rows.append({column: row.get(column, "") for column in transform_columns})
    members["retention-by-transform.csv"] = {
        "uri": f"{prefix}/retention-by-transform.csv",
        "sha256": write_bytes(f"{prefix}/retention-by-transform.csv", to_csv(by_transform_rows, transform_columns)),
    }

    # 6. the native consumer job spec over the published bytes (the reference geometry unchanged)
    job = build_native_job(artifacts["native-measurement-job.json"], members, catalog)
    job_document = json.dumps(job, sort_keys=True, indent=1).encode()
    members["native-measurement-job.json"] = {
        "uri": f"{prefix}/native-measurement-job.json",
        "sha256": write_bytes(f"{prefix}/native-measurement-job.json", job_document),
    }

    # 7. the publication recipe, then the frozen pre-consumer publisher manifest
    recipe = {
        "output_prefix": prefix,
        "v10_release": {"uri": V10_RELEASE_URI, "sha256": V10_RELEASE_SHA},
        "apply_receipt": {"uri": args.apply_receipt_uri, "sha256": args.apply_receipt_sha256},
        "balanced_ledger": {"uri": args.balanced_ledger_uri, "sha256": args.balanced_ledger_sha256},
        "geometry": REFERENCE_GEOMETRY,
        "schedule_updates": [0, 250, 500],
        "consumer_runtime_revision": CONSUMER_REVISION,
        "runtime_tokenizer_uri": data["tokenizer"],
        "inventory": v10["publisher_manifest"],
        "balanced_weight": BALANCED_WEIGHT,
        "note": "V10.1 versions the sealed V10 release with the tool-conversion replacement in the "
                "three math sources (assistant-only supervision preserved) and one new "
                "balanced-reasoning component at 0.005 per phase. Everything else is byte-identical V10.",
    }
    recipe_document = json.dumps(recipe, sort_keys=True, indent=1).encode()
    base = prefix.rsplit("/", 1)[-1]
    recipe_uri = f"{prefix.rsplit('/' + base, 1)[0]}/inputs/publication-recipe-{base}.json"
    recipe_sha = write_bytes(recipe_uri, recipe_document)
    manifest = {
        "artifacts": members,
        "geometry": REFERENCE_GEOMETRY,
        "inventory": v10["publisher_manifest"],
        "publication_recipe": {"uri": recipe_uri, "sha256": recipe_sha},
        "readiness": "Data artifacts frozen; native consumer resolve and padding receipt required",
        "training_ready": False,
        "v10_1_changes": {
            "toolconv_replaced_sources": sorted(MATH_SOURCES),
            "applied_rows": 4383,
            "balanced_source": BALANCED_SOURCE,
            "balanced_weight": BALANCED_WEIGHT,
            "apply_receipt": {"uri": args.apply_receipt_uri, "sha256": args.apply_receipt_sha256},
            "balanced_ledger": {"uri": args.balanced_ledger_uri, "sha256": args.balanced_ledger_sha256},
        },
    }
    manifest_document = json.dumps(manifest, sort_keys=True, indent=1).encode()
    manifest_sha = write_bytes(f"{prefix}/manifest.json", manifest_document)
    print(json.dumps({
        "prefix": prefix,
        "manifest_uri": f"{prefix}/manifest.json",
        "manifest_sha256": manifest_sha,
        "members": {name: pin["sha256"][:16] + "..." for name, pin in members.items()},
        "consumer_job": members["native-measurement-job.json"],
        "sources": len(train),
        "replaced_rows": sum(EXPECTED_REPLACED.values()),
    }, indent=1))


if __name__ == "__main__":
    main()
