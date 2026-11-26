"""Verify the V10.1 release receipts and write the qualified seal.

V10.1 = the sealed V10 release with exactly two changes (2026-11-25, no
approval gate): the three math sources re-versioned @toolconv-20261125 (4,383
verified tool-converted rows replace their originals, V10's assistant-only
supervision preserved — the all-position loss for Exp7 comes from the LLOM
consumption view), and one new balanced-reasoning component at weight 0.005 per
phase. The seal verifies the inherited V10 qualification, the apply/repair
receipts, the balanced cache pins, the tokenizer pins, the weights, the
supervision sidecar and the V10.1 native-consumer receipt, then writes
release-complete-v10.1.json. V10.1 supersedes nothing: the V10 seal stays
qualified.
"""

import argparse
import datetime
import hashlib
import json
import pathlib

from levanter.store.cache import CacheCatalog
from rigging.filesystem.buckets import filesystem_for

ROOT = pathlib.Path(__file__).parent
EVIDENCE = None

V10_RELEASE = {
    "uri": "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.23-v10-overthinking/release-v10/release-complete-v10.json",
    "sha256": "e0838c00f77bacb909faf9dfb62277a616c889c09bba0663ea7bc91b638c7411",
}
CATALOG_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer"
CONSUMER_TREE = pathlib.Path("/Users/benfeuer/Documents/marin-worktrees/exp3-data-consumer")
CONSUMER_REVISION = "6db3ce9022a414804142d02f6edd2fe47befc632"
STORE_TOKENIZER_SHA = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"
BALANCED_ROWS_SHA = "6b5dc656130f4f946a793172ef4495d75f67c1d26e30ad813d0b42e3756ec9be"
BALANCED_SOURCE = "sft/source/balanced_reasoning/v1"
DECLARED_REPLAY = 0.10037232630869815
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
    path = ROOT / ("seal-input-v10-1-" + name)
    if path.exists() and path.read_bytes() != content:
        raise ValueError("Preserve prior qualification input; choose a fresh checker directory: " + name)
    path.write_bytes(content)
    return content, {"uri": uri, "sha256": sha}


def pinned_json(pin, name):
    content, measured = download(pin["uri"], name, pin["sha256"])
    return json.loads(content), measured


def verify_inherited_v10(previous):
    """V10.1 builds on exactly the qualified V10 seal and supersedes nothing."""
    if previous["training_ready"] is not True or previous["gpu_training_approved"] is not False:
        raise ValueError("The base V10 release is not the qualified data seal")
    if previous["geometry"] != REFERENCE_GEOMETRY:
        raise ValueError("The V10 release geometry is not the controlled geometry")


def verify_weights(artifacts, previous):
    """Balanced at 0.005, SFT renormalized by (S - 0.005)/S, pretrain untouched."""
    weights = json.loads(artifacts["train_weights.json"])
    previous_weights = json.loads(download(previous["artifacts"]["train_weights.json"]["uri"], "v10-weights.json",
                                           previous["artifacts"]["train_weights.json"]["sha256"])[0])
    if len(weights) != len(previous_weights):
        raise ValueError("The V10.1 weight schedule changes the phase structure")
    for (phase, phase_weights), (previous_phase, previous_phase_weights) in zip(weights, previous_weights, strict=True):
        if phase != previous_phase:
            raise ValueError(f"Phase boundary changed: {phase} vs {previous_phase}")
        if abs(sum(phase_weights.values()) - 1.0) > 1e-12:
            raise ValueError(f"Phase {phase} weights do not sum to 1.0")
        sft_sum = sum(value for name, value in previous_phase_weights.items() if not name.startswith("pretrain/"))
        scale = (sft_sum - 0.005) / sft_sum
        if set(phase_weights) != set(previous_phase_weights) | {BALANCED_SOURCE}:
            raise ValueError(f"Phase {phase} renames sources")
        for name, value in phase_weights.items():
            if name == BALANCED_SOURCE:
                if value != 0.005:
                    raise ValueError(f"Phase {phase} balanced weight is not 0.005")
            elif name.startswith("pretrain/"):
                if value != previous_phase_weights[name]:
                    raise ValueError(f"Phase {phase} moved a pretrain weight: {name}")
            elif abs(value - previous_phase_weights[name] * scale) > 1e-12:
                raise ValueError(f"Phase {phase} weight for {name} is not the renormalized V10 weight")
        pretrain_share = sum(value for name, value in phase_weights.items() if name.startswith("pretrain/"))
        if abs(pretrain_share - DECLARED_REPLAY) > 1e-12:
            raise ValueError(f"Phase {phase} replay share is not V10's halved declared share")
    data = json.loads(artifacts["data.json"])
    if data["train_weights"] != weights:
        raise ValueError("The V10.1 data mixture weights differ from the pinned member")
    if set(data["components"]) != set(weights[0][1]):
        raise ValueError("The V10.1 data components and weights name different sources")
    if data["tokenizer"] != previous_data_tokenizer(previous):
        raise ValueError("The V10.1 tokenizer moved")
    if data["num_validation_sequences"] != json.loads(download(
            previous["artifacts"]["data.json"]["uri"], "v10-data.json",
            previous["artifacts"]["data.json"]["sha256"])[0])["num_validation_sequences"]:
        raise ValueError("The V10.1 validation reservations moved")
    return {"scale_rule": "(S - 0.005)/S over the V10 SFT sum", "balanced_weight": 0.005,
            "pretrain_weight_rule": "byte-identical V10 pretrain weights"}


def previous_data_tokenizer(previous):
    return json.loads(download(previous["artifacts"]["data.json"]["uri"], "v10-data.json",
                               previous["artifacts"]["data.json"]["sha256"])[0])["tokenizer"]


def verify_source_loss(artifacts, previous):
    """Every inherited entry byte-identical; the balanced entry is the only addition."""
    source_loss = json.loads(artifacts["source_loss.json"])
    previous_loss = json.loads(download(previous["artifacts"]["source_loss.json"]["uri"], "v10-source-loss.json",
                                        previous["artifacts"]["source_loss.json"]["sha256"])[0])
    balanced = source_loss["sources"].get(BALANCED_SOURCE)
    if balanced is None:
        raise ValueError("The balanced component lacks a supervision entry")
    expected = {
        "loss_mode": "chat_assistant_only",
        "loss_weights_indexing": "prediction",
        "loss_weights_key": "loss_weights",
        "source_revision": BALANCED_ROWS_SHA,
        "teacher_only": False,
    }
    if balanced != expected:
        raise ValueError("The balanced supervision entry differs from the pinned store revision")
    for name, entry in source_loss["sources"].items():
        if name == BALANCED_SOURCE:
            continue
        if entry != previous_loss["sources"][name]:
            raise ValueError(f"{name}: an inherited supervision entry differs from V10")
    catalog = json.loads(artifacts["cache_catalog.json"])
    entries = catalog["splits"]["train"]
    if set(source_loss["sources"]) != set(entries):
        raise ValueError("The supervision sidecar and the catalog name different sources")
    for name, entry in entries.items():
        mode = entry["ledger"]["metadata"]["preprocessor_metadata"]["loss_mode"]
        if mode != source_loss["sources"][name]["loss_mode"]:
            raise ValueError(f"{name}: the supervision sidecar disagrees with the cached masks")
    counts = {"chat_assistant_only": 0, "full_token": 0}
    for entry in entries.values():
        counts[entry["ledger"]["metadata"]["preprocessor_metadata"]["loss_mode"]] += 1
    if counts != {"chat_assistant_only": 50, "full_token": 384}:
        raise ValueError(f"The loss-mode census changed: {counts}")


def verify_v10_1_changes(publisher, artifacts, apply_receipt, repair_receipt, balanced_receipt, previous):
    """The apply receipt drives the whole population change from V10."""
    if apply_receipt["status"] != "COMPLETE":
        raise ValueError("The apply receipt is not COMPLETE")
    if apply_receipt["pins"]["catalog_sha256"] != previous["artifacts"]["cache_catalog.json"]["sha256"]:
        raise ValueError("The apply receipt is not over the V10 catalog")
    if apply_receipt["pins"]["tokenizer_sha256"] != STORE_TOKENIZER_SHA:
        raise ValueError("The apply receipt does not pin the store tokenizer")
    if repair_receipt["status"] != "COMPLETE" or len(repair_receipt["repairs"]) != 1:
        raise ValueError("The repair receipt is not a single-row COMPLETE repair")
    repair = repair_receipt["repairs"][0]
    if repair["payload_sha256"] != "8d6c394b8a3107edfde4fd3658e12dcb43f26c68ebaa769c1ee51ca75db2323a":
        raise ValueError("The repaired payload digest differs from its export member pin")
    if repair_receipt["payload_population"] != {"payload_rows": 4383, "missing": 0}:
        raise ValueError("The repair receipt's payload population check failed")
    if balanced_receipt["pins"]["tokenizer_sha256"] != STORE_TOKENIZER_SHA:
        raise ValueError("The balanced cache does not pin the store tokenizer")
    if balanced_receipt["pins"]["rows_sha256"] != BALANCED_ROWS_SHA:
        raise ValueError("The balanced cache is not built from the pinned store rows")
    if balanced_receipt["output"] != {"cache_uri": balanced_receipt["output"]["cache_uri"], "rows": 7728,
                                      "supervised_prediction_positions": 2565856, "tokens": 4070781}:
        raise ValueError("The balanced cache receipt disagrees with the pinned build")

    catalog = json.loads(artifacts["cache_catalog.json"])
    entries = catalog["splits"]["train"]
    previous_catalog = json.loads(download(previous["artifacts"]["cache_catalog.json"]["uri"], "v10-catalog.json",
                                           previous["artifacts"]["cache_catalog.json"]["sha256"])[0])
    previous_entries = previous_catalog["splits"]["train"]
    changed = set(MATH_SOURCES) | {BALANCED_SOURCE}
    if set(entries) != set(previous_entries) | {BALANCED_SOURCE}:
        raise ValueError("The V10.1 catalog renames or drops unexpected components")
    for name, entry in entries.items():
        if name in changed:
            continue
        if entry != previous_entries[name]:
            raise ValueError(f"{name}: an unchanged store's catalog entry moved")
    for source in MATH_SOURCES:
        store = apply_receipt["sources"][source]
        if store["replaced_rows"] != EXPECTED_REPLACED[source]:
            raise ValueError(f"{source}: the applied delta count differs from the bound file")
        if store["output"]["row_delta"] != 0:
            raise ValueError(f"{source}: the applied population does not conserve")
        entry = entries[source]
        ledger = entry["ledger"]
        if ledger["total_num_rows"] != store["output"]["rows"]:
            raise ValueError(f"{source}: the re-keyed ledger row count differs from its receipt")
        if ledger["field_counts"] != {"input_ids": store["output"]["tokens"],
                                      "loss_weights": store["output"]["tokens"]}:
            raise ValueError(f"{source}: the re-keyed ledger token counts differ from its receipt")
        if entry["cache_dir"] != CATALOG_ROOT:
            raise ValueError(f"{source}: the entry does not use the common catalog root")
        if set(ledger["finished_shards"]) != set(store["shard_map"][key]["root_relative_key"]
                                                 for key in store["shard_map"]):
            raise ValueError(f"{source}: the finished shard keys differ from the receipt's map")
        for key, shard in store["shard_map"].items():
            uri = f"{CATALOG_ROOT}/{shard['root_relative_key']}"
            if uri != shard["cache_uri"]:
                raise ValueError(f"{source}: shard {key} does not resolve under the catalog root")
            if shard.get("replaced_rows", 0) == 0:
                if uri != f"{previous_entries[source]['cache_dir'].rstrip('/')}/{key}":
                    raise ValueError(f"{source}: unchanged shard {key} moved off its V10 path")
            elif "/stores-v2/" not in uri:
                raise ValueError(f"{source}: rewritten shard {key} is not in the stores-v2 tree")
        if ledger["metadata"] != previous_entries[source]["ledger"]["metadata"]:
            raise ValueError(f"{source}: the re-keyed store dropped its metadata")
    balanced = entries[BALANCED_SOURCE]
    if balanced["cache_dir"] != CATALOG_ROOT:
        raise ValueError("The balanced entry does not use the common catalog root")
    balanced_key = "exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/balanced-cache-v1/cache"
    if balanced["ledger"]["finished_shards"] != [balanced_key]:
        raise ValueError("The balanced entry does not point at the staged 08a cache")
    if balanced["ledger"]["total_num_rows"] != 7728 or balanced["ledger"]["field_counts"] != {
            "input_ids": 4070781, "loss_weights": 4070781}:
        raise ValueError("The balanced ledger disagrees with its build receipt")
    if len(publisher["artifacts"]) != 8:
        raise ValueError(f"The publisher manifest carries {len(publisher['artifacts'])} members")
    return apply_receipt, catalog


def verify_consumer(job_pin, consumer_pin, publisher, artifacts, apply_receipt):
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
    versioned = {source: apply_receipt["sources"][source] for source in MATH_SOURCES}
    versioned[BALANCED_SOURCE] = {"output": {"rows": 7728, "tokens": 4070781}}
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
    previous, previous_pin = pinned_json(V10_RELEASE, "preserved-v10.json")
    verify_inherited_v10(previous)
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
        raise ValueError("The V10.1 data names another cache artifact")
    apply_receipt, _ = pinned_json(EVIDENCE["apply_receipt"], "apply-receipt.json")
    repair_receipt, _ = pinned_json(EVIDENCE["repair_receipt"], "payload-repair-receipt.json")
    balanced_receipt, _ = pinned_json(EVIDENCE["balanced_cache_receipt"], "balanced-cache-receipt.json")
    weight_check = verify_weights(artifacts, previous)
    verify_source_loss(artifacts, previous)
    verify_v10_1_changes(publisher, artifacts, apply_receipt, repair_receipt, balanced_receipt, previous)
    consumer = verify_consumer(EVIDENCE["native_measurement_job"], EVIDENCE["native_consumer_receipt"],
                               publisher, artifacts, apply_receipt)
    checker_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    review = {
        "user_approval": "2026-11-25: the user REMOVED the approval gate (BRIEF.md 4:15 PM update: 'V10.1 "
                         "does NOT need user approval. Seal it yourself once the native-consumer receipt, "
                         "the capacity/epoch plan, the tokenizer check and the full-token loss_mode all "
                         "pass'). The full-token loss mode is satisfied by the Exp7 LLOM consumption view "
                         "(TextLmDatasetFormat; PR #12284), not by rewritten data; the cached masks keep "
                         "V10's assistant-only supervision.",
        "change_scope": {
            "note": "Exactly two changes from the sealed V10: the 4,383-row tool-conversion replacement in "
                    "the three math sources (rows conserved; assistant-only supervision preserved; only the "
                    "15 delta-bearing shards rewritten) and the new balanced-reasoning component at 0.005 "
                    "per phase (SFT renormalized by (S-0.005)/S; the pretrain replay share untouched). "
                    "Every other source, the supervision sidecar's inherited entries, the tokenizer, the "
                    "validation reservations and the reference geometry are byte-identical V10.",
            "toolconv_coverage_boundary": {
                "applied": 4383,
                "pre_applied_at_build": 46314,
                "unmappable_to_v10_lineage": 188469,
                "quarantined_tokenizer_mismatch": 10654,
                "verified_modified_2026_11_25_index_snapshot": 2378592,
            },
            "tokenizer_pins": {
                "store_tokenizer_sha256": STORE_TOKENIZER_SHA,
                "apply_receipt_pin": apply_receipt["pins"]["tokenizer_sha256"],
                "balanced_cache_receipt_pin": balanced_receipt["pins"]["tokenizer_sha256"],
                "repair_payload_export_member_sha256": "8d6c394b8a3107edfde4fd3658e12dcb43f26c68ebaa769c1ee51ca75db2323a",
            },
        },
        "mix_comparison": {
            "note": "V10.1 vs the sealed V10 mix measured per source: exactly the 3 math sources' tokens "
                    "move and exactly 1 source is new (asserted over every catalog row); the update count "
                    "1,679 -> 1,680 (+1) at 256x262,144; the declared replay share unchanged at "
                    "0.10037232630869815 (realized 0.100697 -> 0.100874); the store totals "
                    "133.943B -> 133.980B tokens / 44,717,016 -> 44,724,744 rows.",
            "markdown": {"uri": EVIDENCE["mix_comparison_md"]["uri"], "sha256": EVIDENCE["mix_comparison_md"]["sha256"]},
            "csv": {"uri": EVIDENCE["mix_comparison_csv"]["uri"], "sha256": EVIDENCE["mix_comparison_csv"]["sha256"]},
        },
        "capacity_epoch": {
            "note": "The V10.1 capacity receipt over the frozen catalog (offsets-only, token_bodies_read "
                    "false) and the epoch plan (1,680 updates; the realized replay fraction).",
            "capacity": {"uri": EVIDENCE["capacity_receipt"]["uri"], "sha256": EVIDENCE["capacity_receipt"]["sha256"]},
            "epoch_plan": {"uri": EVIDENCE["epoch_plan"]["uri"], "sha256": EVIDENCE["epoch_plan"]["sha256"]},
        },
    }
    for pin_name in ("mix_comparison_md", "mix_comparison_csv", "capacity_receipt", "epoch_plan",
                     "apply_receipt", "repair_receipt", "balanced_cache_receipt"):
        pin = EVIDENCE[pin_name]
        download(pin["uri"], pin_name.replace("_", "-") + "-v1", pin["sha256"])
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
        "toolconv_apply_receipt": EVIDENCE["apply_receipt"],
        "toolconv_repair_receipt": EVIDENCE["repair_receipt"],
        "balanced_cache_receipt": EVIDENCE["balanced_cache_receipt"],
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
        "v10_1_changes": {
            "user_decisions": "2026-11-25: add the balanced thinking and tool-conversion fixes to Mix V10; "
                              "no approval gate (removed by the user at 4:15 PM EST). The all-position SFT "
                              "loss is applied by the Exp7 LLOM trainer at consumption (PR #12284).",
            "note": "The three math sources re-version @toolconv-20261125 with 4,383 replaced rows and "
                    "V10's assistant-only cached masks; the balanced component enters at 0.005 per phase. "
                    "Everything else is byte-identical V10.",
            "weights": weight_check,
            "toolconv": {
                source: {
                    "base": apply_receipt["sources"][source]["base"],
                    "output": apply_receipt["sources"][source]["output"],
                    "replaced_rows": apply_receipt["sources"][source]["replaced_rows"],
                    "replaced_tokens": apply_receipt["sources"][source]["replaced_tokens"],
                    "replaced_original_tokens": apply_receipt["sources"][source]["replaced_original_tokens"],
                    "delta_files": apply_receipt["sources"][source]["delta_files"],
                }
                for source in MATH_SOURCES
            },
            "balanced_reasoning": {
                "source": BALANCED_SOURCE,
                "rows": 7728,
                "tokens": 4070781,
                "supervised_prediction_positions": 2565856,
                "source_revision": BALANCED_ROWS_SHA,
                "store_rows_sha256": BALANCED_ROWS_SHA,
                "output_tokens_and_weights_sha256": balanced_receipt["output_tokens_and_weights_sha256"],
                "weight": 0.005,
            },
            "epoch_plan": {"uri": EVIDENCE["epoch_plan"]["uri"], "sha256": EVIDENCE["epoch_plan"]["sha256"]},
        },
        "artifact_graph": EVIDENCE["artifact_graph"],
        "review_evidence": review,
        "inherited_qualification": "The replay population and every unchanged SFT store reuse the V10 qualified "
                                   "caches byte-identically (unchanged shards keep their V10 shard paths "
                                   "re-expressed relative to the catalog root); the V10 release record and its "
                                   "qualification apply verbatim to them. The three math sources are "
                                   "re-versioned by the V10.1 toolconv apply with the native consumer receipt "
                                   "over the V10.1 release.",
        "readiness_scope": "Data qualification only; production launch requires user approval and the runtime graph resolve.",
        "consumer_limitation": consumer["limitation"],
    }
    path = ROOT / "release-complete-v10.1.json"
    if path.exists():
        raise ValueError("Preserve any prior local seal; do not overwrite it")
    path.write_text(json.dumps(seal, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"local_seal": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "changed_sources": sorted(set(MATH_SOURCES) | {BALANCED_SOURCE}),
                      "exposure_views": seal["exposure_views"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=pathlib.Path)
    args = parser.parse_args()
    EVIDENCE = json.loads(args.evidence.read_text())
    main()
