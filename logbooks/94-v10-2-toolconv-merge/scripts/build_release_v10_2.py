"""Build and upload the V10.2 release directory, then write its frozen publisher manifest.

V10.2 = the sealed V10.1 release with exactly one change:
the eight toolconv-delta sources' stores are re-versioned at ``@toolconv-merge-20261127``
(stores-v4): 42,272 verified tool-converted rows replace their originals (the apply-v4
receipt carries the per-shard map; unchanged shards keep their V10.1 shard paths
re-expressed relative to the catalog root). The supervision stays V10's
``chat_assistant_only`` — the all-position loss for Exp7 is applied by the LLOM trainer
at consumption. 13,958 of the replaced rows were re-tokenized from text with the pinned
2026.09.18 tokenizer; the rest reuse store tokenizations digest-verified during the
payload extraction.

Named design exclusion (coordinator ruling 2026-11-27, option b): the 40,374
same-name-rewrite rows stay byte-identical V10 rows in V10.2 — no substitution touches
them — and are resolved in V10.3.

Every other source, the weights, the supervision sidecar and the epoch construction are
byte-identical to V10.1. The manifest freezes before the native-consumer receipt exists
(training_ready false), as in V7/V8/V9/V10/V10.1.
"""

import argparse
import csv
import hashlib
import io
import json

from rigging.filesystem.buckets import filesystem_for

V10_1_RELEASE_URI = "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.1/release-complete-v10.1.json"
V10_1_RELEASE_SHA = "81fab6977b6dc2acb904a9027f5c0e0adc062eee32a1d09e0dbf9a5a20a4759e"
CATALOG_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer"
CONSUMER_REVISION = "6db3ce9022a414804142d02f6edd2fe47befc632"
REFERENCE_GEOMETRY = {"updates": 746, "global_batch": 256, "sequence_length": 262144,
                      "seed": 0, "packing_cap": 256, "lookahead": 128}
REPLACED_SOURCES = (
    "sft/source/nemotron_sft_v3/math_v3/train",
    "sft/source/nemotron_sft_v3/math_v4/train",
    "sft/source/nemotron_sft_v3/opencode_v1/agent_skills",
    "sft/source/nemotron_sft_v3/opencode_v1/bash_only_tool",
    "sft/source/nemotron_sft_v3/opencode_v1/bash_only_tool_skills",
    "sft/source/nemotron_sft_v3/opencode_v1/question_tool",
    "sft/source/nemotron_sft_v3/swe_v1/r2e_gym",
    "sft/source/nemotron_sft_v3/swe_v2/openhands_swe",
)
EXPECTED_REPLACED = {
    "sft/source/nemotron_sft_v3/math_v3/train": 18680,
    "sft/source/nemotron_sft_v3/math_v4/train": 41,
    "sft/source/nemotron_sft_v3/opencode_v1/agent_skills": 17432,
    "sft/source/nemotron_sft_v3/opencode_v1/bash_only_tool": 314,
    "sft/source/nemotron_sft_v3/opencode_v1/bash_only_tool_skills": 2427,
    "sft/source/nemotron_sft_v3/opencode_v1/question_tool": 106,
    "sft/source/nemotron_sft_v3/swe_v1/r2e_gym": 1979,
    "sft/source/nemotron_sft_v3/swe_v2/openhands_swe": 1293,
}
EXPECTED_TOTAL_REPLACED = 42272
NAMED_EXCLUSION_ROWS = 40374
NAMED_EXCLUSION = {
    "rows": NAMED_EXCLUSION_ROWS,
    "ruling": "2026-11-27: the coordinator approved option (b) — proceed with the 42,272-row "
    "delta; the 40,374 same-name-rewrite rows stay byte-identical to V10 in V10.2 as a NAMED "
    "design exclusion, recorded in the seal notes with their row count and reason, and are "
    "resolved in V10.3.",
    "reason": "the rows are V10's same-name argument rewrites whose verified conversions share "
    "the source row name but whose before/after sha256 pins match NO rendered/converted-argument "
    "hash under raw-string or sorted-JSON canonicalizations (canonical-hash check "
    "/benfeuer/v10-2-canonical-hash-check-20261127, 3h20m, SUCCEEDED, all 40,374 mismatch, probed "
    "end-to-end); their converted forms cannot be re-derived from the store, so any substitution "
    "would be unverifiable",
    "resolution": "byte-identical V10 rows in V10.2; resolved in V10.3",
}
RETOKENIZED_FROM_TEXT = 13958
DELTA_COMPOSITION = {"lineage_not_applied": 40698, "args_check_not_applied": 1574}
APPLY_JOB = "/benfeuer/v10-2-apply-merge-20261127-r3"
TOOLCONV_COVERAGE = {
    "verified_modified_2026_11_27_index_snapshot": 2223483,
    "applied_baked_content_confirmed": 1253844,
    "not_applied_replaced_in_v10_2": EXPECTED_TOTAL_REPLACED,
    "same_name_design_exclusion_byte_identical": NAMED_EXCLUSION_ROWS,
    "transformed_no_tool_block": 0,
    "tracer_conflicts": 28787,
    "retention_csv_residual": "the retention CSV's 1,804,539 counts the campaign's inline "
    "substitutions (scripted + atlas real-tools paths, proven by raw-store/per-generation content "
    "probes); the row-attributable set = 746,682 lineage overlays + 1,253,844 baked; the residual "
    "= DUPLICATE VERIFICATIONS (the index carries multiple records per raw row: high +94,837, "
    "v3 +70,819, v4 +14,356 — distinct (part,row) counts reproduce the pre-RNO index table "
    "exactly) + RNO-era growth",
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


def build_catalog(v10_1_catalog: dict, apply_receipt: dict) -> dict:
    """V10.1's catalog with the eight entries re-keyed per the apply receipt; every other
    entry byte-identical."""
    catalog = json.loads(json.dumps(v10_1_catalog))
    train = catalog["splits"]["train"]
    for source in REPLACED_SOURCES:
        old = train[source]
        data = apply_receipt["sources"][source]
        if data["base"]["rows"] != old["ledger"]["total_num_rows"]:
            raise ValueError(f"{source}: the apply receipt's base rows disagree with the V10.1 catalog")
        if data["replaced_rows"] != EXPECTED_REPLACED[source]:
            raise ValueError(f"{source}: the apply receipt replaced {data['replaced_rows']} rows")
        if data["output"]["rows"] != data["base"]["rows"] or data["output"]["row_delta"] != 0:
            raise ValueError(f"{source}: the applied population does not conserve")
        if data["base"]["tokens"] + data["output"]["token_delta"] != data["output"]["tokens"]:
            raise ValueError(f"{source}: the applied token population does not conserve")
        shard_rows = {}
        field_counts = {}
        replaced_sum = 0
        for old_key, shard in data["shard_map"].items():
            new_key = shard["root_relative_key"]
            uri = f"{CATALOG_ROOT}/{new_key}"
            if uri != shard["cache_uri"]:
                raise ValueError(f"{source}: shard {old_key} does not resolve under the catalog root")
            replaced = shard.get("replaced_rows", 0)
            replaced_sum += replaced
            if replaced == 0:
                if uri != f"{old['cache_dir'].rstrip('/')}/{old_key}":
                    raise ValueError(f"{source}: unchanged shard {old_key} moved off its V10.1 path")
            elif "/stores-v4/" not in uri:
                raise ValueError(f"{source}: rewritten shard {old_key} is not in the stores-v4 tree")
            shard_rows[new_key] = shard["rows"]
            field_counts[new_key] = {"input_ids": shard["tokens"], "loss_weights": shard["tokens"]}
        if set(data["shard_map"].keys()) != set(old["ledger"]["finished_shards"]):
            raise ValueError(f"{source}: the shard map does not cover the V10.1 shard set")
        if len(shard_rows) != len(old["ledger"]["finished_shards"]):
            raise ValueError(f"{source}: the shard-key count changed ({len(shard_rows)} vs the V10.1 set)")
        if replaced_sum != data["replaced_rows"]:
            raise ValueError(f"{source}: per-shard replaced rows sum to {replaced_sum}")
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
    total = sum(EXPECTED_REPLACED.values())
    if total != EXPECTED_TOTAL_REPLACED:
        raise ValueError(f"the per-source expectation sums to {total}, not {EXPECTED_TOTAL_REPLACED}")
    return catalog


def build_native_job(v10_1_job: dict, members: dict, catalog: dict) -> dict:
    """V10.1's reference-geometry job over the published V10.2 bytes."""
    job = json.loads(json.dumps(v10_1_job))
    geometry = {
        "batch_size": 256, "lookahead": 128, "packing_cap": 256, "sequence_length": 262144,
        "steps": 746, "trainer_seed": 0, "data_seed": None, "jax_threefry_partitionable": True,
    }
    for key, value in geometry.items():
        if job[key] != value:
            raise ValueError(f"the V10.1 job's {key} is not the reference geometry")
    if job["consumer_runtime_revision"] != CONSUMER_REVISION:
        raise ValueError("the V10.1 job pins another consumer revision")
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
    parser.add_argument("--apply-receipt-uri", required=True, help="The apply-v4 receipt on S3")
    parser.add_argument("--apply-receipt-sha256", required=True)
    parser.add_argument("--output-prefix", required=True, help="Destination prefix, e.g. s3://.../release-v10.2")
    args = parser.parse_args()

    prefix = args.output_prefix.rstrip("/")
    v10_1 = pinned_json(V10_1_RELEASE_URI, V10_1_RELEASE_SHA, "release-complete-v10.1.json")
    if v10_1["training_ready"] is not True:
        raise ValueError("The V10.1 release is not the qualified seal")
    artifacts = {}
    for name in ("cache_catalog.json", "data.json", "mixture-rationale.json", "retention-by-source.csv",
                 "retention-by-transform.csv", "source_loss.json", "train_weights.json",
                 "native-measurement-job.json"):
        pin = v10_1["artifacts"][name]
        artifacts[name] = pinned_json(pin["uri"], pin["sha256"], name) if name.endswith(".json") else read_bytes(pin["uri"])
    apply_receipt = pinned_json(args.apply_receipt_uri, args.apply_receipt_sha256, "apply-receipt.json")
    if apply_receipt["status"] != "COMPLETE" or apply_receipt["pins"]["catalog_sha256"] != v10_1["artifacts"]["cache_catalog.json"]["sha256"]:
        raise ValueError("The apply receipt is not COMPLETE over the V10.1 catalog")
    replaced_total = sum(s["replaced_rows"] for s in apply_receipt["sources"].values())
    if replaced_total != EXPECTED_TOTAL_REPLACED:
        raise ValueError(f"The apply receipt replaced {replaced_total} rows in total")
    if sorted(apply_receipt["sources"].keys()) != sorted(REPLACED_SOURCES):
        raise ValueError("The apply receipt's sources are not the expected eight")

    members = {}

    # 1. the V10.2 cache catalog: only the eight entries re-keyed
    catalog = build_catalog(artifacts["cache_catalog.json"], apply_receipt)
    train = catalog["splits"]["train"]
    previous_train = artifacts["cache_catalog.json"]["splits"]["train"]
    if len(train) != len(previous_train):
        raise ValueError(f"the V10.2 catalog carries {len(train)} sources, expected {len(previous_train)}")
    for name, entry in train.items():
        if name in REPLACED_SOURCES:
            continue
        if entry != previous_train[name]:
            raise ValueError(f"{name}: an untouched entry differs from V10.1")
    catalog_document = json.dumps(catalog, sort_keys=True, indent=1).encode()
    members["cache_catalog.json"] = {
        "uri": f"{prefix}/cache_catalog.json",
        "sha256": write_bytes(f"{prefix}/cache_catalog.json", catalog_document),
    }

    # 2. train_weights.json: byte-identical V10.1 bytes (the mixture is unchanged)
    weights_bytes = read_bytes(v10_1["artifacts"]["train_weights.json"]["uri"])
    if hashlib.sha256(weights_bytes).hexdigest() != v10_1["artifacts"]["train_weights.json"]["sha256"]:
        raise ValueError("The V10.1 train weights no longer match their seal pin")
    members["train_weights.json"] = {
        "uri": f"{prefix}/train_weights.json",
        "sha256": write_bytes(f"{prefix}/train_weights.json", weights_bytes),
    }

    # 3. data.json: V10.1's with the cache catalog repointed; everything else identical
    data = json.loads(json.dumps(artifacts["data.json"]))
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

    # 4. source_loss.json: byte-identical V10.1 bytes (no supervision change)
    loss_bytes = read_bytes(v10_1["artifacts"]["source_loss.json"]["uri"])
    if hashlib.sha256(loss_bytes).hexdigest() != v10_1["artifacts"]["source_loss.json"]["sha256"]:
        raise ValueError("The V10.1 supervision sidecar no longer matches its seal pin")
    members["source_loss.json"] = {
        "uri": f"{prefix}/source_loss.json",
        "sha256": write_bytes(f"{prefix}/source_loss.json", loss_bytes),
    }

    # 5. the mixture rationale with the V10.2 amendments recorded
    rationale = json.loads(json.dumps(artifacts["mixture-rationale.json"]))
    rationale["v10_2_amendments"] = {
        "ruling": NAMED_EXCLUSION["ruling"],
        "toolconv_merge": {
            "rule": "verified tool-converted rows REPLACE their unconverted originals in the same "
            "source; every other source stays byte-identical to V10.1",
            "coverage_boundary": {**TOOLCONV_COVERAGE, "note": "the 2026-11-27 index snapshot covers the "
            "post-RNO archive; the tracer partitions every verified-modified row into named buckets "
            "with zero not_in_build for math; the 28,787 conflicts are index-vs-lineage disagreements "
            "resolved to named causes in reconcile/final/reconciliation.json"},
            "applied_rows": EXPECTED_TOTAL_REPLACED,
            "delta_composition": DELTA_COMPOSITION,
            "retokenized_from_text": RETOKENIZED_FROM_TEXT,
            "sources": sorted(REPLACED_SOURCES),
            "apply_receipt": {"uri": args.apply_receipt_uri, "sha256": args.apply_receipt_sha256},
            "apply_job": APPLY_JOB,
            "supervision": "V10's chat_assistant_only cached masks are kept; replaced rows recompute "
            "the assistant-only mask with the pinned 2026.09.18 tokenizer (13,958 re-tokenized from "
            "text, the rest digest-verified store tokenizations); unchanged rows keep V10 bytes; "
            "unchanged shards keep their V10/V10.1 shard paths",
            "token_delta": -132820164,
        },
        "named_exclusions": {"same_name_rewrites": NAMED_EXCLUSION},
    }
    rationale_document = json.dumps(rationale, sort_keys=True, indent=1).encode()
    members["mixture-rationale.json"] = {
        "uri": f"{prefix}/mixture-rationale.json",
        "sha256": write_bytes(f"{prefix}/mixture-rationale.json", rationale_document),
    }

    # 6. retention tables with the V10.2 populations and the merge stage
    by_source_rows = list(csv.DictReader(io.StringIO(artifacts["retention-by-source.csv"].decode())))
    columns = list(by_source_rows[0].keys())
    kept_rows = []
    for row in by_source_rows:
        if row["component"] in REPLACED_SOURCES:
            store = apply_receipt["sources"][row["component"]]
            row["output_documents"] = str(store["output"]["rows"])
            row["output_tokens"] = str(store["output"]["tokens"])
            row["retained_serialized_positions"] = str(store["output"]["tokens"])
        kept_rows.append(row)
    members["retention-by-source.csv"] = {
        "uri": f"{prefix}/retention-by-source.csv",
        "sha256": write_bytes(f"{prefix}/retention-by-source.csv", to_csv(kept_rows, columns)),
    }

    by_transform_rows = list(csv.DictReader(io.StringIO(artifacts["retention-by-transform.csv"].decode())))
    transform_columns = list(by_transform_rows[0].keys())
    for source in REPLACED_SOURCES:
        store = apply_receipt["sources"][source]
        source_row = next(r for r in kept_rows if r["component"] == source)
        row = {column: "" for column in transform_columns}
        row.update({
            "source": source_row["source"], "component": source, "loss_mode": source_row["loss_mode"],
            "phase": "v10_2_toolconv", "transform": "toolconv-merge@20261127",
            "input_documents": str(store["base"]["rows"]), "input_tokens": str(store["base"]["tokens"]),
            "output_documents": str(store["output"]["rows"]), "output_tokens": str(store["output"]["tokens"]),
            "changed_documents": str(store["replaced_rows"]),
            "accounting_note": (
                f"V10.2 toolconv merge: {store['replaced_rows']} verified tool-converted rows replace their "
                "originals in the sealed V10.1 store; assistant-only supervision preserved; unchanged "
                "rows and shards keep V10.1 bytes; the same-name-rewrite rows stay byte-identical V10 "
                "rows (named design exclusion, coordinator ruling 2026-11-27, resolved in V10.3)."
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

    # 7. the native consumer job spec over the published bytes (the reference geometry unchanged)
    job = build_native_job(artifacts["native-measurement-job.json"], members, catalog)
    job_document = json.dumps(job, sort_keys=True, indent=1).encode()
    members["native-measurement-job.json"] = {
        "uri": f"{prefix}/native-measurement-job.json",
        "sha256": write_bytes(f"{prefix}/native-measurement-job.json", job_document),
    }

    # 8. the publication recipe, then the frozen pre-consumer publisher manifest
    recipe = {
        "output_prefix": prefix,
        "v10_1_release": {"uri": V10_1_RELEASE_URI, "sha256": V10_1_RELEASE_SHA},
        "apply_receipt": {"uri": args.apply_receipt_uri, "sha256": args.apply_receipt_sha256},
        "geometry": REFERENCE_GEOMETRY,
        "schedule_updates": [0, 250, 500],
        "consumer_runtime_revision": CONSUMER_REVISION,
        "runtime_tokenizer_uri": data["tokenizer"],
        "inventory": v10_1["publisher_manifest"],
        "applied_rows": EXPECTED_TOTAL_REPLACED,
        "named_exclusions": {"same_name_rewrites": NAMED_EXCLUSION},
        "note": "V10.2 versions the sealed V10.1 release with the toolconv merge replacement in "
                "the eight delta sources (assistant-only supervision preserved, stores-v4). "
                "Everything else is byte-identical V10.1; the 40,374 same-name rows stay "
                "byte-identical V10 rows (named design exclusion, resolved in V10.3).",
    }
    recipe_document = json.dumps(recipe, sort_keys=True, indent=1).encode()
    base = prefix.rsplit("/", 1)[-1]
    recipe_uri = f"{prefix.rsplit('/' + base, 1)[0]}/inputs/publication-recipe-{base}.json"
    recipe_sha = write_bytes(recipe_uri, recipe_document)
    manifest = {
        "artifacts": members,
        "geometry": REFERENCE_GEOMETRY,
        "inventory": v10_1["publisher_manifest"],
        "publication_recipe": {"uri": recipe_uri, "sha256": recipe_sha},
        "readiness": "Data artifacts frozen; native consumer resolve and padding receipt required",
        "training_ready": False,
        "v10_2_changes": {
            "toolconv_replaced_sources": sorted(REPLACED_SOURCES),
            "applied_rows": EXPECTED_TOTAL_REPLACED,
            "delta_composition": DELTA_COMPOSITION,
            "retokenized_from_text": RETOKENIZED_FROM_TEXT,
            "token_delta": -132820164,
            "named_exclusions": {"same_name_rewrites": NAMED_EXCLUSION},
            "apply_receipt": {"uri": args.apply_receipt_uri, "sha256": args.apply_receipt_sha256},
            "apply_job": APPLY_JOB,
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
        "replaced_rows": EXPECTED_TOTAL_REPLACED,
        "named_exclusion_rows": NAMED_EXCLUSION_ROWS,
    }, indent=1))


if __name__ == "__main__":
    main()
