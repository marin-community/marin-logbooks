"""Verify the V10 release receipts and prepare the qualified seal for final review.

V10 = the sealed V9 release with exactly the two user-approved subsampling changes
(the pretraining replay share halved, and the Terminus-2-format traces cut to a
seeded 20% per affected source). The seal verifies the inherited V9 qualification,
the V10 scan and re-version receipts, the halved-replay weights, and the V10
native-consumer receipt, then writes release-complete-v10.json. V10 supersedes
nothing: the V9 seal stays qualified for Exp6.
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import pathlib

from levanter.store.cache import CacheCatalog
from rigging.filesystem.buckets import filesystem_for

ROOT = pathlib.Path(__file__).parent
EVIDENCE = None

V9_RELEASE = {
    "uri": "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.22-v9-overthinking/release-v9/release-complete-v9.json",
    "sha256": "2a7d231a3b79cc722929c76f6c03441c9d7d475ad79b46284eb35b75fd01c898",
}
V10_DECLARED_REPLAY = 0.10037232630869815
V9_DECLARED_REPLAY = 0.20074465261739632
CONSUMER_TREE = pathlib.Path("/Users/benfeuer/Documents/marin-worktrees/exp3-data-consumer")
CONSUMER_REVISION = "6db3ce9022a414804142d02f6edd2fe47befc632"
REFERENCE_GEOMETRY = {"updates": 746, "global_batch": 256, "sequence_length": 262144,
                      "seed": 0, "packing_cap": 256, "lookahead": 128}


def download(uri, name, expected=None):
    fs, key = filesystem_for(uri)
    if fs.size(key) > 512 * 1024 * 1024:
        raise ValueError("Only small qualification metadata may be read locally: " + name)
    content = fs.cat(key)
    sha = hashlib.sha256(content).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError("Release artifact digest changed: " + name)
    path = ROOT / ("seal-input-v10-" + name)
    if path.exists() and path.read_bytes() != content:
        raise ValueError("Preserve prior qualification input; choose a fresh checker directory: " + name)
    path.write_bytes(content)
    return content, {"uri": uri, "sha256": sha}


def pinned_json(pin, name):
    content, measured = download(pin["uri"], name, pin["sha256"])
    return json.loads(content), measured


def verify_inherited_v9(previous):
    """V10 builds on exactly the qualified V9 seal and supersedes nothing."""
    if previous["training_ready"] is not True or previous["gpu_training_approved"] is not False:
        raise ValueError("The base V9 release is not the qualified data seal")
    if previous["geometry"] != REFERENCE_GEOMETRY:
        raise ValueError("The V9 release geometry is not the controlled geometry")


def verify_v10_changes(publisher, artifacts, scan, reversion, previous):
    """The re-version receipt drives the whole population change from V9."""
    if reversion["store_version"] != "terminus2-keep20-20261123":
        raise ValueError("The re-version receipt is not the ruling-66 keep-20 store version")
    affected = sorted(reversion["sources"])
    if affected != scan["affected_sources"]:
        raise ValueError("The re-version receipt's affected sources disagree with the scan")
    catalog = json.loads(artifacts["cache_catalog.json"])
    entries = catalog["splits"]["train"]
    previous_catalog = json.loads(download(previous["artifacts"]["cache_catalog.json"]["uri"], "v9-catalog.json",
                                           previous["artifacts"]["cache_catalog.json"]["sha256"])[0])
    previous_entries = previous_catalog["splits"]["train"]
    if set(entries) != set(previous_entries):
        raise ValueError("The V10 catalog renames or drops unexpected components")
    for name, entry in entries.items():
        if name in affected:
            continue
        if entry != previous_entries[name]:
            raise ValueError(f"{name}: an unchanged store's catalog entry moved")
    for source in affected:
        store = reversion["sources"][source]
        entry = entries[source]
        if entry["cache_dir"] != store["new_cache_dir"]:
            raise ValueError(f"{source}: the V10 catalog does not point at the versioned store")
        ledger = entry["ledger"]
        if ledger["total_num_rows"] != store["output"]["rows"]:
            raise ValueError(f"{source}: the versioned ledger row count differs from its receipt")
        if ledger["field_counts"] != {"input_ids": store["output"]["tokens"],
                                      "loss_weights": store["output"]["tokens"]}:
            raise ValueError(f"{source}: the versioned ledger token counts differ from its receipt")
        if set(ledger["finished_shards"]) != set(store["finished_shards"]):
            raise ValueError(f"{source}: the versioned store renamed shards")
        if ledger["shard_rows"] != store["shard_rows"]:
            raise ValueError(f"{source}: the versioned store moved rows between shards")
        if ledger["metadata"] != previous_entries[source]["ledger"]["metadata"]:
            raise ValueError(f"{source}: the versioned store dropped its metadata")
        if store["kept_harness_rows"] != scan["sources"][source]["keep_count"]:
            raise ValueError(f"{source}: the kept harness rows disagree with the scan selection")
        if store["dropped_harness_rows"] != scan["sources"][source]["marker_rows"] - scan["sources"][source]["keep_count"]:
            raise ValueError(f"{source}: the dropped harness rows do not match the 80% cut")
    return reversion, catalog


def verify_weights(artifacts, previous):
    weights = json.loads(artifacts["train_weights.json"])
    previous_weights = json.loads(download(previous["artifacts"]["train_weights.json"]["uri"], "v9-weights.json",
                                           previous["artifacts"]["train_weights.json"]["sha256"])[0])
    pretrain_share_previous = sum(value for name, value in previous_weights[0][1].items()
                                  if name.startswith("pretrain/"))
    scale_pretrain = 0.5
    scale_sft = (1.0 - V10_DECLARED_REPLAY) / (1.0 - V9_DECLARED_REPLAY)
    if len(weights) != len(previous_weights):
        raise ValueError("The V10 weight schedule changes the phase structure")
    for (phase, phase_weights), (previous_phase, previous_phase_weights) in zip(weights, previous_weights, strict=True):
        if phase != previous_phase:
            raise ValueError(f"Phase boundary changed: {phase} vs {previous_phase}")
        if abs(sum(phase_weights.values()) - 1.0) > 1e-9:
            raise ValueError(f"Phase {phase} weights do not sum to 1.0")
        if set(phase_weights) != set(previous_phase_weights):
            raise ValueError(f"Phase {phase} renames sources")
        pretrain_share = sum(value for name, value in phase_weights.items() if name.startswith("pretrain/"))
        if abs(pretrain_share - V10_DECLARED_REPLAY) > 1e-12:
            raise ValueError(f"Phase {phase} replay share is not the halved declared share")
        for name, value in phase_weights.items():
            expected = previous_phase_weights[name] * (scale_pretrain if name.startswith("pretrain/") else scale_sft)
            if abs(value - expected) > 1e-12:
                raise ValueError(f"Phase {phase} weight for {name} is not the halved-replay V9 weight")
    data = json.loads(artifacts["data.json"])
    if data["train_weights"] != weights:
        raise ValueError("The V10 data mixture weights differ from the pinned member")
    if set(data["components"]) != set(weights[0][1]):
        raise ValueError("The V10 data components and weights name different sources")
    return {"v9_declared_replay": V9_DECLARED_REPLAY, "v10_declared_replay": V10_DECLARED_REPLAY,
            "scale_pretrain": scale_pretrain, "scale_sft": scale_sft}


def verify_source_loss(artifacts, previous):
    source_loss = json.loads(artifacts["source_loss.json"])
    previous_loss = json.loads(download(previous["artifacts"]["source_loss.json"]["uri"], "v9-source-loss.json",
                                        previous["artifacts"]["source_loss.json"]["sha256"])[0])
    if source_loss != previous_loss:
        raise ValueError("The V10 supervision sidecar differs from V9")


def verify_consumer(job_pin, consumer_pin, publisher, artifacts, reversion):
    job, _ = pinned_json(job_pin, "native-job.json")
    consumer, _ = pinned_json(consumer_pin, "native-consumer.json")
    if consumer["complete"] is not True or consumer["job"] != job:
        raise ValueError("Incomplete or mismatched consumer receipt")
    for key, name in (("data", "data.json"), ("catalog", "cache_catalog.json"), ("source_loss", "source_loss.json")):
        if (job[key + "_uri"], job[key + "_sha256"]) != (
            publisher["artifacts"][name]["uri"], publisher["artifacts"][name]["sha256"]
        ):
            raise ValueError("Native consumer used different published bytes")
    expected_geometry = {"steps": 746, "batch_size": 256, "sequence_length": 262144,
                         "trainer_seed": 0, "data_seed": None, "packing_cap": 256, "lookahead": 128}
    if any(job[key] != value for key, value in expected_geometry.items()):
        raise ValueError("Production reference geometry changed")
    if job["consumer_runtime_revision"] != CONSUMER_REVISION:
        raise ValueError("Native consumer revision changed")
    for key, path in (
        ("implementation_sha256", CONSUMER_TREE / "experiments/post_training/long_loop/exp3_padding.py"),
        ("packing_implementation_sha256", CONSUMER_TREE / "lib/levanter/src/levanter/data/packing.py"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != consumer[key]:
            raise ValueError("Native consumer implementation does not match its pinned worktree")
    catalog = CacheCatalog.from_json(artifacts["cache_catalog.json"])
    versioned = {source: reversion["sources"][source] for source in reversion["sources"]}
    if not (
        set(catalog.splits["train"]) == set(consumer["corpus"]) == set(consumer["source_body_probes"])
        == set(job["expected_source_populations"])
    ) or not set(versioned) <= set(catalog.splits["train"]):
        raise ValueError("Final catalog, supervision, and native source coverage differ")
    for name, entry in catalog.splits["train"].items():
        expected = versioned.get(name)
        if expected is not None:
            rows, tokens = expected["output"]["rows"], expected["output"]["tokens"]
        else:
            rows, tokens = entry.ledger.total_num_rows, entry.ledger.field_counts["input_ids"]
        if (entry.ledger.total_num_rows, entry.ledger.field_counts["input_ids"]) != (rows, tokens):
            raise ValueError("Catalog ledger differs from the versioned population")
        if entry.ledger.field_counts["input_ids"] != entry.ledger.field_counts["loss_weights"]:
            raise ValueError("Catalog token and mask lengths differ")
        actual = consumer["corpus"][name]
        if (actual["documents"], actual["document_tokens"]) != (rows, tokens):
            raise ValueError("Native packing changed the normalized source population")
        if name in versioned and len(consumer["source_body_probes"][name]) != 1:
            raise ValueError("A source lacks its actual packed token/mask probe")
    if consumer["sampled_training_validation_document_overlap"] or consumer["dropped_token_fraction"]:
        raise ValueError("Native consumer found overlap or dropped tokens")
    data = json.loads(artifacts["data.json"])
    if consumer["num_validation_sequences"] != data["num_validation_sequences"] or not data["shuffle_before_trainval_split"]:
        raise ValueError("Native IID split changed")
    if set(consumer["update_prefixes"]) != {"360", "377", "746"}:
        raise ValueError("Missing approved duration/reference exposure views")
    for updates, view in consumer["update_prefixes"].items():
        totals = view["totals"]
        if (totals["sequences"], totals["context_positions"]) != (int(updates) * 256, int(updates) * 256 * 262144):
            raise ValueError("Exposure view uses a different update geometry")
        if totals["real_tokens"] + totals["padding_tokens"] != totals["context_positions"] or totals["dropped_tokens"]:
            raise ValueError("Exposure positions are not conserved")
        if view["per_source"].keys() != set(catalog.splits["train"]):
            raise ValueError("Exposure report omits active sources with zero draws")
    if consumer["update_prefixes"]["746"]["totals"] != consumer["totals"]:
        raise ValueError("Full reference and prefix exposure differ")
    return consumer


def main():
    previous, previous_pin = pinned_json(V9_RELEASE, "preserved-v9.json")
    verify_inherited_v9(previous)
    publisher, publisher_pin = pinned_json(EVIDENCE["publisher_manifest"], "publisher.json")
    if publisher.get("training_ready"):
        raise ValueError("Expected a frozen, pre-consumer publisher manifest")
    if publisher["geometry"] != REFERENCE_GEOMETRY:
        raise ValueError("Publisher reference geometry changed")
    artifacts = {}
    for name, pin in publisher["artifacts"].items():
        artifacts[name], _ = download(pin["uri"], name, pin["sha256"])
    data = json.loads(artifacts["data.json"])
    if data["cache_catalog"] != publisher["artifacts"]["cache_catalog.json"]["uri"]:
        raise ValueError("The V10 data names another cache artifact")
    scan, scan_pin = pinned_json(EVIDENCE["scan_receipt"], "terminus2-scan.json")
    reversion, reversion_pin = pinned_json(EVIDENCE["reversion_receipt"], "terminus2-reversion.json")
    weight_check = verify_weights(artifacts, previous)
    verify_source_loss(artifacts, previous)
    verify_v10_changes(publisher, artifacts, scan, reversion, previous)
    consumer = verify_consumer(EVIDENCE["native_measurement_job"], EVIDENCE["native_consumer_receipt"],
                               publisher, artifacts, reversion)
    checker_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    review = {
        "user_approval": "2026-11-23: the user APPROVED V10 ('yes') after the MIX-COMPARISON-V10 review; "
                         "the seal was prepared after that approval and carries the review evidence below.",
        "subsampling_exceptions": {
            "note": "Ruling-66 stage: the two changes are explicit user subsampling decisions — recorded "
                    "exceptions to the fully-provisioned rule. The pretraining replay share halves exactly "
                    "(20.074465% -> 10.037232630869815%), and the Terminus-2-format traces keep a seeded 20% "
                    "per affected source (2 of 433 sources affected; every other store byte-identical to V9). "
                    "No other reduction is introduced by V10; the V9 fully-provisioned audit "
                    "(361 of 434 stores smaller than their full build lineage, all recorded quality/correctness "
                    "gates) applies verbatim to every unchanged store.",
            "v9_fully_provisioned_audit": {
                "uri": "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.22-v9-overthinking/fully-provisioned-audit-v8.csv",
                "sha256": "ef70a1ddbbb37595c65e654d94e1dea8e6e2a8a0fddde2cdd674f060633d903c",
            },
        },
        "mix_comparison": {
            "note": "V10 vs the sealed V9 mix measured per source: the update count 1,893 -> 1,679 (-214) at "
                    "256x262,144; the declared replay share 20.074465% -> 10.0372% (realized 20.0923% -> "
                    "10.0697%); the Terminus-2 detector evidence and the keep-20 selection per affected "
                    "source; the store totals 134.12B -> 133.94B tokens / 44,725,307 -> 44,717,016 rows.",
            "markdown": {"uri": EVIDENCE["mix_comparison_md"]["uri"], "sha256": EVIDENCE["mix_comparison_md"]["sha256"]},
            "csv": {"uri": EVIDENCE["mix_comparison_csv"]["uri"], "sha256": EVIDENCE["mix_comparison_csv"]["sha256"]},
        },
        "capacity_epoch": {
            "note": "The V10 capacity receipt over the frozen catalog and the epoch plan (the update count "
                    "and the realized replay fraction).",
            "capacity": {"uri": EVIDENCE["capacity_receipt"]["uri"], "sha256": EVIDENCE["capacity_receipt"]["sha256"]},
            "epoch_plan": {"uri": EVIDENCE["epoch_plan"]["uri"], "sha256": EVIDENCE["epoch_plan"]["sha256"]},
        },
    }
    for pin_name in ("mix_comparison_md", "mix_comparison_csv", "capacity_receipt", "epoch_plan"):
        pin = EVIDENCE[pin_name]
        download(pin["uri"], pin_name.replace("_", "-") + "-v1", pin["sha256"])
    download(review["subsampling_exceptions"]["v9_fully_provisioned_audit"]["uri"],
             "fully-provisioned-audit-v9.csv",
             review["subsampling_exceptions"]["v9_fully_provisioned_audit"]["sha256"])
    native_catalog = CacheCatalog.from_json(artifacts["cache_catalog.json"])
    chat_tokens = sum(e.ledger.field_counts["input_ids"] for e in native_catalog.splits["train"].values()
                      if e.ledger.metadata.preprocessor_metadata.get("loss_mode") == "chat_assistant_only")
    chat_rows = sum(e.ledger.total_num_rows for e in native_catalog.splits["train"].values()
                    if e.ledger.metadata.preprocessor_metadata.get("loss_mode") == "chat_assistant_only")
    seal = {
        "schema_version": 1,
        "training_ready": True,
        "gpu_training_approved": False,
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
        "seal_implementation": {"uri": EVIDENCE["seal_implementation_uri"], "sha256": checker_sha},
        "supersedes": None,
        "publisher_manifest": publisher_pin,
        "artifacts": publisher["artifacts"],
        "scan_receipt": scan_pin,
        "reversion_receipt": reversion_pin,
        "native_consumer_receipt": EVIDENCE["native_consumer_receipt"],
        "native_measurement_job": EVIDENCE["native_measurement_job"],
        "geometry": publisher["geometry"],
        "num_validation_sequences": consumer["num_validation_sequences"],
        "natural_population": {
            "chat_assistant_only": {
                "output_documents": chat_rows,
                "output_tokens": chat_tokens,
            },
            "full_token": previous["natural_population"]["full_token"],
        },
        "exposure_views": {key: {"totals": value["totals"], "padding_fraction": value["padding_fraction"]}
                           for key, value in consumer["update_prefixes"].items()},
        "v10_changes": {
            "user_decisions": "2026-11-23: cut the pretraining replay mix in half and cut 80% of the "
                              "Terminus-2-format traces. Both are explicit user subsampling decisions "
                              "(exceptions to the fully-provisioned rule, recorded as such); no other change.",
            "note": "Exactly two sources carry the Terminus-2 harness marker (agenttrove and "
                    "calibforge-relay/terminus2, both at 100% of rows); their seeded 20% keep plus every "
                    "other source byte-identical to V9. The replay share halves exactly in all three phases.",
            "replay_halving": weight_check,
            "terminus2_keep20": {
                source: {
                    "base": reversion["sources"][source]["base"],
                    "output": reversion["sources"][source]["output"],
                    "harness_rows": reversion["sources"][source]["harness_rows"],
                    "kept_harness_rows": reversion["sources"][source]["kept_harness_rows"],
                    "dropped_harness_rows": reversion["sources"][source]["dropped_harness_rows"],
                    "keep_ids_sha256": reversion["sources"][source]["keep_ids_sha256"],
                    "drops": reversion["sources"][source]["drops"],
                }
                for source in reversion["sources"]
            },
            "keep_rule": scan["keep"],
            "epoch_plan": {"uri": EVIDENCE["epoch_plan"]["uri"], "sha256": EVIDENCE["epoch_plan"]["sha256"]},
        },
        "artifact_graph": EVIDENCE["artifact_graph"],
        "review_evidence": review,
        "inherited_qualification": "The replay population and every unchanged SFT store reuse the V9 qualified "
                                   "caches byte-identically; the V9 release record and its qualification apply "
                                   "verbatim to them. The two Terminus-2-format sources are re-versioned by the "
                                   "V10 keep-20 re-versioning with the native consumer receipt over the V10 release.",
        "readiness_scope": "Data qualification only; production launch requires user approval and the runtime graph resolve.",
        "consumer_limitation": consumer["limitation"],
    }
    path = ROOT / "release-complete-v10.json"
    if path.exists():
        raise ValueError("Preserve any prior local seal; do not overwrite it")
    path.write_text(json.dumps(seal, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"local_seal": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "affected_sources": sorted(reversion["sources"]),
                      "exposure_views": seal["exposure_views"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=pathlib.Path)
    args = parser.parse_args()
    EVIDENCE = json.loads(args.evidence.read_text())
    main()
