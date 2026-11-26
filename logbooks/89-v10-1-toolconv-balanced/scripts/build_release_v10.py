"""Build and upload the V10 release directory, then write its frozen publisher manifest.

V10 = the sealed V9 release with exactly the two user-approved subsampling changes:
(1) the pretraining replay share halved relative to V9 (declared 20.074465% ->
    10.037232630869815%: every ``pretrain/`` component weight x0.5, every SFT
    component renormalized so each phase sums to 1.0); (2) every source whose rows
    carry the Terminus-2 agent-harness marker re-versioned at
    ``@terminus2-keep20-20261123`` keeping the scan's deterministic seeded 20% of
    harness rows per source. Every other store, the supervision sidecar and the
    epoch construction are byte-identical to V9. The manifest freezes before the
    native-consumer receipt exists (training_ready false), as in V7/V8/V9.
"""

import argparse
import csv
import hashlib
import io
import json

from rigging.filesystem.buckets import filesystem_for

V9_RELEASE_URI = "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.22-v9-overthinking/release-v9/release-complete-v9.json"
V9_RELEASE_SHA = "2a7d231a3b79cc722929c76f6c03441c9d7d475ad79b46284eb35b75fd01c898"
CONSUMER_REVISION = "6db3ce9022a414804142d02f6edd2fe47befc632"
REFERENCE_GEOMETRY = {"updates": 746, "global_batch": 256, "sequence_length": 262144,
                      "seed": 0, "packing_cap": 256, "lookahead": 128}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-receipt-uri", required=True, help="The Terminus-2 scan receipt on S3")
    parser.add_argument("--scan-receipt-sha256", required=True)
    parser.add_argument("--reversion-receipt-uri", required=True, help="The keep-20 re-version receipt on S3")
    parser.add_argument("--reversion-receipt-sha256", required=True)
    parser.add_argument("--v10-weights-uri", required=True, help="The halved-replay V10 train weights on S3")
    parser.add_argument("--v10-weights-sha256", required=True)
    parser.add_argument("--output-prefix", required=True, help="Destination prefix, e.g. s3://.../release-v10")
    args = parser.parse_args()

    prefix = args.output_prefix.rstrip("/")
    v9 = pinned_json(V9_RELEASE_URI, V9_RELEASE_SHA, "release-complete-v9.json")
    if v9["training_ready"] is not True:
        raise ValueError("The V9 release is not the qualified seal")
    scan = pinned_json(args.scan_receipt_uri, args.scan_receipt_sha256, "terminus2-scan.json")
    reversion = pinned_json(args.reversion_receipt_uri, args.reversion_receipt_sha256, "terminus2-reversion.json")
    if reversion["store_version"] != "terminus2-keep20-20261123":
        raise ValueError("The re-version receipt is not the ruling-66 keep-20 store version")
    affected = sorted(reversion["sources"])
    if affected != scan["affected_sources"]:
        raise ValueError("The re-version receipt's affected sources disagree with the scan")
    v10_weights = pinned_json(args.v10_weights_uri, args.v10_weights_sha256, "v10-train-weights.json")
    # The declared replay share halves exactly: the 384 pretrain/ components' aggregate
    # phase weight goes 0.20074465261739632 -> 0.10037232630869815.
    share_halved = 0.10037232630869815
    for phase, phase_weights in v10_weights:
        pretrain_share = sum(value for name, value in phase_weights.items() if name.startswith("pretrain/"))
        if abs(pretrain_share - share_halved) > 1e-12:
            raise ValueError(f"phase {phase}: the V10 weights do not carry the halved replay share")

    members = {}

    # 1. byte-identical inherited members from the V9 release
    for name in ("source_loss.json",):
        content = read_bytes(v9["artifacts"][name]["uri"])
        members[name] = {"uri": f"{prefix}/{name}", "sha256": write_bytes(f"{prefix}/{name}", content)}

    # 2. the V10 cache catalog: only the affected sources' entries replaced
    catalog_bytes = read_bytes(v9["artifacts"]["cache_catalog.json"]["uri"])
    if hashlib.sha256(catalog_bytes).hexdigest() != v9["artifacts"]["cache_catalog.json"]["sha256"]:
        raise ValueError("The V9 cache catalog digest differs from its release pin")
    v9_catalog = json.loads(catalog_bytes)
    v10_catalog = json.loads(json.dumps(v9_catalog))
    train = v10_catalog["splits"]["train"]
    for source in affected:
        store = reversion["sources"][source]
        old_entry = train[source]
        if old_entry["ledger"]["total_num_rows"] != store["base"]["rows"]:
            raise ValueError(f"{source}: the re-version receipt's base rows disagree with the V9 catalog")
        if old_entry["ledger"]["field_counts"]["input_ids"] != store["base"]["tokens"]:
            raise ValueError(f"{source}: the re-version receipt's base tokens disagree with the V9 catalog")
        train[source] = {
            "cache_dir": store["new_cache_dir"],
            "ledger": {
                "total_num_rows": store["output"]["rows"],
                "shard_rows": store["shard_rows"],
                "is_finished": True,
                "finished_shards": store["finished_shards"],
                "field_counts": {"input_ids": store["output"]["tokens"], "loss_weights": store["output"]["tokens"]},
                "field_counts_by_shard": store["field_counts"],
                "layout": old_entry["ledger"].get("layout", "sharded"),
                "metadata": old_entry["ledger"]["metadata"],
            },
        }
    catalog_document = json.dumps(v10_catalog, sort_keys=True, indent=1).encode()
    members["cache_catalog.json"] = {
        "uri": f"{prefix}/cache_catalog.json",
        "sha256": write_bytes(f"{prefix}/cache_catalog.json", catalog_document),
    }

    # 3. train_weights.json (the halved replay) and data.json
    members["train_weights.json"] = {
        "uri": f"{prefix}/train_weights.json",
        "sha256": write_bytes(f"{prefix}/train_weights.json", json.dumps(v10_weights, sort_keys=True, indent=1).encode()),
    }
    data = json.loads(read_bytes(v9["artifacts"]["data.json"]["uri"]))
    data["train_weights"] = v10_weights
    data["cache_catalog"] = members["cache_catalog.json"]["uri"]
    data_document = json.dumps(data, sort_keys=True, indent=1).encode()
    members["data.json"] = {"uri": f"{prefix}/data.json", "sha256": write_bytes(f"{prefix}/data.json", data_document)}

    # 4. the mixture rationale with the V10 amendments recorded
    rationale = json.loads(read_bytes(v9["artifacts"]["mixture-rationale.json"]["uri"]))
    share_halved = 0.10037232630869815
    rationale["v10_amendments"] = {
        "user_decisions": "2026-11-23: cut the pretraining replay mix in half and cut 80% of the "
        "Terminus-2-format traces. Both are explicit user subsampling decisions (allowed exceptions "
        "to the fully-provisioned rule, recorded as such); no other change.",
        "replay_halving": {
            "rule": "half the declared replay share relative to V9 (the replay mix budget halves)",
            "v9_declared_share": 0.20074465261739632,
            "v10_declared_share": share_halved,
            "readings": {
                "note": "half the declared SHARE is the landed rule; both readings shown over the V9 "
                "epoch numbers (387,239 packed SFT sequences, 484,608 run sequences, 97,369 replay "
                "sequences); the precise V10 epoch comes from the capacity receipt over the frozen "
                "V10 catalog",
                "share_halved": {
                    "declared_fraction": share_halved,
                    "estimate_replay_sequences": 43204.63,
                    "estimate_total_sequences": 430443.63,
                    "estimate_replay_positions_vs_v9": 0.4437205714257076,
                },
                "positions_halved": {
                    "replay_sequences": 48684.5,
                    "declared_fraction": 0.11168129270388039,
                    "total_sequences": 435923.5,
                },
            },
            "weight_rule": "every pretrain/ component weight x0.5; every SFT component weight "
            "renormalized so each phase sums to 1.0",
            "note": "the precise V10 epoch plan (the packed SFT sequences over the keep-20 stores, "
            "the realized fraction, the update count) is the capacity receipt over the frozen V10 "
            "catalog, reported in MIX-COMPARISON-V10.md",
        },
        "terminus2_keep20": {
            "store_version": "terminus2-keep20-20261123",
            "keep": scan["keep"],
            "affected_sources": affected,
            "scan_receipt": {"uri": args.scan_receipt_uri, "sha256": args.scan_receipt_sha256},
            "reversion_receipt": {"uri": args.reversion_receipt_uri, "sha256": args.reversion_receipt_sha256},
        },
    }
    rationale_document = json.dumps(rationale, sort_keys=True, indent=1).encode()
    members["mixture-rationale.json"] = {
        "uri": f"{prefix}/mixture-rationale.json",
        "sha256": write_bytes(f"{prefix}/mixture-rationale.json", rationale_document),
    }

    # 5. retention tables with the V10 populations and the keep-20 stage
    by_source_bytes = read_bytes(v9["artifacts"]["retention-by-source.csv"]["uri"])
    by_source_rows = list(csv.DictReader(io.StringIO(by_source_bytes.decode())))
    columns = list(by_source_rows[0].keys())
    kept_rows = []
    for row in by_source_rows:
        store = reversion["sources"].get(row["component"])
        if row["component"] in v10_weights[0][1]:
            row["weight"] = repr(v10_weights[0][1][row["component"]])
        if store is not None:
            row["output_documents"] = str(store["output"]["rows"])
            row["output_tokens"] = str(store["output"]["tokens"])
            row["retained_serialized_positions"] = str(store["output"]["tokens"])
        kept_rows.append(row)
    members["retention-by-source.csv"] = {
        "uri": f"{prefix}/retention-by-source.csv",
        "sha256": write_bytes(f"{prefix}/retention-by-source.csv", to_csv(kept_rows, columns)),
    }

    by_transform_bytes = read_bytes(v9["artifacts"]["retention-by-transform.csv"]["uri"])
    by_transform_rows = list(csv.DictReader(io.StringIO(by_transform_bytes.decode())))
    transform_columns = list(by_transform_rows[0].keys())
    for source in affected:
        store = reversion["sources"][source]
        source_row = next(r for r in kept_rows if r["component"] == source)
        dropped = store["drops"]["terminus2_keep20"]
        row = {column: "" for column in transform_columns}
        row.update({
            "source": source_row["source"], "component": source, "loss_mode": source_row["loss_mode"],
            "phase": "v10_terminus2_keep20", "transform": "terminus2-keep20@20261123",
            "input_documents": str(store["base"]["rows"]), "input_tokens": str(store["base"]["tokens"]),
            "output_documents": str(store["output"]["rows"]), "output_tokens": str(store["output"]["tokens"]),
            "dropped_documents": str(dropped), "dropped_tokens": "",
            "drop_reason": "terminus2_keep20",
            "accounting_note": (
                f"V10 keep-20: {dropped} Terminus-2 harness rows dropped from the sealed V9 store; "
                "the kept 20% selection and the non-harness rows are byte-identical rows."
            ),
            "input_serialized_positions": str(store["base"]["tokens"]),
            "output_serialized_positions": str(store["output"]["tokens"]),
            "changed_documents": str(dropped), "complete": "True",
            "input_population": str(store["base"]["rows"]), "token_measurement": "store_receipt",
        })
        for column, value in row.items():
            if column not in transform_columns and value:
                raise ValueError(f"retention-by-transform column absent: {column}")
        row = {column: row.get(column, "") for column in transform_columns}
        by_transform_rows.append(row)
    members["retention-by-transform.csv"] = {
        "uri": f"{prefix}/retention-by-transform.csv",
        "sha256": write_bytes(
            f"{prefix}/retention-by-transform.csv", to_csv(by_transform_rows, transform_columns)
        ),
    }

    # 6. the native consumer job spec over the published bytes (the reference geometry unchanged)
    job = {
        "batch_size": 256,
        "body_probe_sequences_per_source": 1,
        "catalog_sha256": members["cache_catalog.json"]["sha256"],
        "catalog_uri": members["cache_catalog.json"]["uri"],
        "consumer_runtime_revision": CONSUMER_REVISION,
        "data_seed": None,
        "data_sha256": members["data.json"]["sha256"],
        "data_uri": members["data.json"]["uri"],
        "jax_threefry_partitionable": True,
        "lookahead": 128,
        "packing_cap": 256,
        "report_update_prefixes": [360, 377, 746],
        "sequence_length": 262144,
        "source_loss_sha256": members["source_loss.json"]["sha256"],
        "source_loss_uri": members["source_loss.json"]["uri"],
        "steps": 746,
        "trainer_seed": 0,
        "expected_source_populations": {
            name: {"documents": entry["ledger"]["total_num_rows"],
                   "tokens": entry["ledger"]["field_counts"]["input_ids"]}
            for name, entry in v10_catalog["splits"]["train"].items()
        },
    }
    job_document = json.dumps(job, sort_keys=True, indent=1).encode()
    members["native-measurement-job.json"] = {
        "uri": f"{prefix}/native-measurement-job.json",
        "sha256": write_bytes(f"{prefix}/native-measurement-job.json", job_document),
    }

    # 7. the publication recipe, then the frozen pre-consumer publisher manifest
    recipe = {
        "output_prefix": prefix,
        "v9_release": {"uri": V9_RELEASE_URI, "sha256": V9_RELEASE_SHA},
        "scan_receipt": {"uri": args.scan_receipt_uri, "sha256": args.scan_receipt_sha256},
        "reversion_receipt": {"uri": args.reversion_receipt_uri, "sha256": args.reversion_receipt_sha256},
        "v10_weights": {"uri": args.v10_weights_uri, "sha256": args.v10_weights_sha256},
        "geometry": REFERENCE_GEOMETRY,
        "schedule_updates": [0, 250, 500],
        "consumer_runtime_revision": CONSUMER_REVISION,
        "runtime_tokenizer_uri": data["tokenizer"],
        "inventory": pinned_json(v9["publisher_manifest"]["uri"], v9["publisher_manifest"]["sha256"], "v9-publisher.json")["inventory"],
        "replay_share_halving": {"v9": 0.20074465261739632, "v10": share_halved},
        "note": "V10 versions the sealed V9 release with the user-approved subsampling: the pretraining "
                "replay share halved, the Terminus-2-format traces cut to a seeded 20% per affected "
                "source. Everything else is byte-identical V9.",
    }
    recipe_document = json.dumps(recipe, sort_keys=True, indent=1).encode()
    base = prefix.rsplit("/", 1)[-1]
    recipe_uri = f"{prefix.rsplit('/' + base, 1)[0]}/inputs/publication-recipe-{base}.json"
    recipe_sha = write_bytes(recipe_uri, recipe_document)
    manifest = {
        "artifacts": members,
        "geometry": REFERENCE_GEOMETRY,
        "inventory": pinned_json(v9["publisher_manifest"]["uri"], v9["publisher_manifest"]["sha256"], "v9-publisher.json")["inventory"],
        "publication_recipe": {"uri": recipe_uri, "sha256": recipe_sha},
        "readiness": "Data artifacts frozen; native consumer resolve and padding receipt required",
        "training_ready": False,
        "v10_changes": {
            "replay_share": {"v9": 0.20074465261739632, "v10": share_halved},
            "versioned_sources": affected,
            "scan_receipt": {"uri": args.scan_receipt_uri, "sha256": args.scan_receipt_sha256},
            "reversion_receipt": {"uri": args.reversion_receipt_uri, "sha256": args.reversion_receipt_sha256},
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
        "affected_sources": affected,
        "replay_share": {"v9": 0.20074465261739632, "v10": share_halved},
    }, indent=1))


if __name__ == "__main__":
    main()
