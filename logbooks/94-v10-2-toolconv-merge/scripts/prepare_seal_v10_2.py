"""Verify the V10.2 release receipts and write the qualified seal.

V10.2 = the sealed V10.1 release with exactly one change (2026-11-27, coordinator
ruling option b): the eight toolconv-delta sources re-versioned
@toolconv-merge-20261127 (42,272 verified tool-converted rows replace their
originals, V10's assistant-only supervision preserved — the all-position loss for
Exp7 comes from the LLOM consumption view; 13,958 replaced rows re-tokenized from
text with the pinned 2026.09.18 tokenizer; unchanged shards keep their V10/V10.1
paths re-expressed relative to the catalog root). The 40,374 same-name-rewrite
rows stay byte-identical V10 rows as a NAMED design exclusion recorded in the seal
notes and resolved in V10.3. The seal verifies the inherited V10.1 qualification,
the apply receipt, the tokenizer pins, the byte-identical weights and supervision
sidecar and the V10.2 native-consumer receipt, then writes
release-complete-v10.2.json. V10.2 supersedes nothing: the V10.1 seal stays
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

V10_1_RELEASE = {
    "uri": "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.1/release-complete-v10.1.json",
    "sha256": "81fab6977b6dc2acb904a9027f5c0e0adc062eee32a1d09e0dbf9a5a20a4759e",
}
CATALOG_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer"
CONSUMER_TREE = pathlib.Path("/Users/benfeuer/Documents/marin-worktrees/exp3-data-consumer")
CONSUMER_REVISION = "6db3ce9022a414804142d02f6edd2fe47befc632"
STORE_TOKENIZER_SHA = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"
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
CANONICAL_HASH_JOB = "/benfeuer/v10-2-canonical-hash-check-20261127"
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
    path = ROOT / ("seal-input-v10-2-" + name)
    if path.exists() and path.read_bytes() != content:
        raise ValueError("Preserve prior qualification input; choose a fresh checker directory: " + name)
    path.write_bytes(content)
    return content, {"uri": uri, "sha256": sha}


def pinned_json(pin, name):
    content, measured = download(pin["uri"], name, pin["sha256"])
    return json.loads(content), measured


def verify_inherited_v10_1(previous):
    """V10.2 builds on exactly the qualified V10.1 seal and supersedes nothing."""
    if previous["training_ready"] is not True or previous["gpu_training_approved"] is not False:
        raise ValueError("The base V10.1 release is not the qualified data seal")
    if previous["geometry"] != REFERENCE_GEOMETRY:
        raise ValueError("The V10.1 release geometry is not the controlled geometry")


def verify_weights(artifacts, previous):
    """V10.2's schedule is byte-identical V10.1's (the mixture is unchanged)."""
    weights = json.loads(artifacts["train_weights.json"])
    previous_weights = json.loads(download(previous["artifacts"]["train_weights.json"]["uri"],
                                           "v10-1-weights.json",
                                           previous["artifacts"]["train_weights.json"]["sha256"])[0])
    if weights != previous_weights:
        raise ValueError("The V10.2 weight schedule differs from V10.1's")
    for (phase, phase_weights) in weights:
        if abs(sum(phase_weights.values()) - 1.0) > 1e-12:
            raise ValueError(f"Phase {phase} weights do not sum to 1.0")
        pretrain_share = sum(value for name, value in phase_weights.items() if name.startswith("pretrain/"))
        if abs(pretrain_share - 0.10037232630869815) > 1e-12:
            raise ValueError(f"Phase {phase} replay share is not the declared halved share")
    data = json.loads(artifacts["data.json"])
    if data["train_weights"] != weights:
        raise ValueError("The V10.2 data mixture weights differ from the pinned member")
    if set(data["components"]) != set(weights[0][1]):
        raise ValueError("The V10.2 data components and weights name different sources")
    if data["tokenizer"] != previous_data_tokenizer(previous):
        raise ValueError("The V10.2 tokenizer moved")
    if data["num_validation_sequences"] != json.loads(download(
            previous["artifacts"]["data.json"]["uri"], "v10-1-data.json",
            previous["artifacts"]["data.json"]["sha256"])[0])["num_validation_sequences"]:
        raise ValueError("The V10.2 validation reservations moved")
    return {"rule": "byte-identical V10.1 schedule (balanced 0.005, SFT renormalized, pretrain untouched)",
            "sha256": hashlib.sha256(artifacts["train_weights.json"]).hexdigest()}


def previous_data_tokenizer(previous):
    return json.loads(download(previous["artifacts"]["data.json"]["uri"], "v10-1-data.json",
                               previous["artifacts"]["data.json"]["sha256"])[0])["tokenizer"]


def verify_source_loss(artifacts, previous):
    """The supervision sidecar is byte-identical V10.1's; the mask census is unchanged."""
    source_loss = json.loads(artifacts["source_loss.json"])
    previous_loss = json.loads(download(previous["artifacts"]["source_loss.json"]["uri"],
                                        "v10-1-source-loss.json",
                                        previous["artifacts"]["source_loss.json"]["sha256"])[0])
    if source_loss != previous_loss:
        raise ValueError("The V10.2 supervision sidecar differs from V10.1's")
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


def verify_v10_2_changes(publisher, artifacts, apply_receipt, previous):
    """The apply receipt drives the whole population change from V10.1."""
    if apply_receipt["status"] != "COMPLETE":
        raise ValueError("The apply receipt is not COMPLETE")
    if apply_receipt["pins"]["catalog_sha256"] != previous["artifacts"]["cache_catalog.json"]["sha256"]:
        raise ValueError("The apply receipt is not over the V10.1 catalog")
    if apply_receipt["pins"]["tokenizer_sha256"] != STORE_TOKENIZER_SHA:
        raise ValueError("The apply receipt does not pin the store tokenizer")

    catalog = json.loads(artifacts["cache_catalog.json"])
    entries = catalog["splits"]["train"]
    previous_catalog = json.loads(download(previous["artifacts"]["cache_catalog.json"]["uri"],
                                           "v10-1-catalog.json",
                                           previous["artifacts"]["cache_catalog.json"]["sha256"])[0])
    previous_entries = previous_catalog["splits"]["train"]
    if set(entries) != set(previous_entries):
        raise ValueError("The V10.2 catalog renames or drops components")
    for name, entry in entries.items():
        if name in REPLACED_SOURCES:
            continue
        if entry != previous_entries[name]:
            raise ValueError(f"{name}: an unchanged store's catalog entry moved")
    for source in REPLACED_SOURCES:
        store = apply_receipt["sources"][source]
        if store["replaced_rows"] != EXPECTED_REPLACED[source]:
            raise ValueError(f"{source}: the applied delta count differs from the bound file")
        if store["output"]["row_delta"] != 0:
            raise ValueError(f"{source}: the applied population does not conserve")
        if store["loss_mode"] != "chat_assistant_only":
            raise ValueError(f"{source}: the applied store carries another loss mode")
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
                    raise ValueError(f"{source}: unchanged shard {key} moved off its V10.1 path")
            elif "/stores-v4/" not in uri:
                raise ValueError(f"{source}: rewritten shard {key} is not in the stores-v4 tree")
        if ledger["metadata"] != previous_entries[source]["ledger"]["metadata"]:
            raise ValueError(f"{source}: the re-keyed store dropped its metadata")
    replaced_total = sum(apply_receipt["sources"][s]["replaced_rows"] for s in REPLACED_SOURCES)
    if replaced_total != EXPECTED_TOTAL_REPLACED:
        raise ValueError(f"The apply receipt replaced {replaced_total} rows in total")
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
    versioned = {source: apply_receipt["sources"][source] for source in REPLACED_SOURCES}
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
    previous, previous_pin = pinned_json(V10_1_RELEASE, "preserved-v10-1.json")
    verify_inherited_v10_1(previous)
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
        raise ValueError("The V10.2 data names another cache artifact")
    apply_receipt, _ = pinned_json(EVIDENCE["apply_receipt"], "apply-receipt.json")
    weight_check = verify_weights(artifacts, previous)
    verify_source_loss(artifacts, previous)
    apply_receipt, catalog = verify_v10_2_changes(publisher, artifacts, apply_receipt, previous)
    consumer = verify_consumer(EVIDENCE["native_measurement_job"], EVIDENCE["native_consumer_receipt"],
                               publisher, artifacts, apply_receipt)
    checker_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    review = {
        "user_approval": "2026-11-27: the coordinator approved option (b) — seal V10.2 with the "
                         "42,272-row verified-toolconv delta; the 40,374 same-name-rewrite rows stay "
                         "byte-identical V10 rows as a NAMED design exclusion recorded in these seal "
                         "notes and resolved in V10.3. The all-position SFT loss is applied by the "
                         "Exp7 LLOM trainer at consumption (PR #12284); the cached masks keep V10's "
                         "assistant-only supervision.",
        "change_scope": {
            "note": "Exactly one change from the sealed V10.1: the 42,272-row toolconv merge "
                    "replacement in the eight delta sources (rows conserved; assistant-only "
                    "supervision preserved; only the 152 delta-bearing shards rewritten into "
                    "stores-v4; 13,958 rows re-tokenized from text with the pinned 2026.09.18 "
                    "tokenizer; the rest digest-verified store tokenizations). Every other source, "
                    "the weight schedule, the supervision sidecar, the tokenizer, the validation "
                    "reservations and the reference geometry are byte-identical V10.1.",
            "named_exclusions": {
                "same_name_rewrites": {
                    "rows": NAMED_EXCLUSION_ROWS,
                    "reason": "the rows are V10's same-name argument rewrites whose verified "
                              "conversions share the source row name but whose before/after sha256 "
                              "pins match NO rendered/converted-argument hash under raw-string or "
                              f"sorted-JSON canonicalizations (canonical-hash check {CANONICAL_HASH_JOB}, "
                              "3h20m, all 40,374 mismatch, probed end-to-end); their converted forms "
                              "cannot be re-derived from the store, so any substitution would be "
                              "unverifiable",
                    "resolution": "byte-identical V10 rows in V10.2; resolved in V10.3 (coordinator "
                                  "ruling 2026-11-27, option b)",
                },
            },
            "toolconv_coverage_boundary": {
                "applied_rows": EXPECTED_TOTAL_REPLACED,
                "delta_composition": {"lineage_not_applied": 40698, "args_check_not_applied": 1574},
                "verified_modified_2026_11_27_index_snapshot": 2223483,
                "applied_baked_content_confirmed": 1253844,
                "same_name_design_exclusion_byte_identical": NAMED_EXCLUSION_ROWS,
                "transformed_no_tool_block": 0,
                "tracer_conflicts": 28787,
            },
            "tokenizer_pins": {
                "store_tokenizer_sha256": STORE_TOKENIZER_SHA,
                "apply_receipt_pin": apply_receipt["pins"]["tokenizer_sha256"],
            },
        },
        "mix_comparison": {
            "note": "V10.2 vs the sealed V10.1 mix measured per source: rows conserve everywhere and "
                    "exactly the eight toolconv sources' tokens move (asserted over every catalog "
                    "row); the store totals shrink by 132,820,164 tokens (converted tool payloads are "
                    "far more compact than raw tool blocks).",
            "markdown": {"uri": EVIDENCE["mix_comparison_md"]["uri"], "sha256": EVIDENCE["mix_comparison_md"]["sha256"]},
            "csv": {"uri": EVIDENCE["mix_comparison_csv"]["uri"], "sha256": EVIDENCE["mix_comparison_csv"]["sha256"]},
        },
        "capacity_epoch": {
            "note": "The V10.2 capacity receipt over the frozen catalog (offsets-only, token_bodies_read "
                    "false) and the epoch plan (the update count and the realized replay fraction).",
            "capacity": {"uri": EVIDENCE["capacity_receipt"]["uri"], "sha256": EVIDENCE["capacity_receipt"]["sha256"]},
            "epoch_plan": {"uri": EVIDENCE["epoch_plan"]["uri"], "sha256": EVIDENCE["epoch_plan"]["sha256"]},
        },
    }
    for pin_name in ("mix_comparison_md", "mix_comparison_csv", "capacity_receipt", "epoch_plan",
                     "apply_receipt"):
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
        "v10_2_changes": {
            "user_decisions": "2026-11-27: the coordinator approved option (b) — the 40,374 same-name "
                              "rows stay byte-identical V10 rows in V10.2 as a NAMED design exclusion "
                              "and are resolved in V10.3. The all-position SFT loss is applied by the "
                              "Exp7 LLOM trainer at consumption (PR #12284).",
            "note": "The eight toolconv-delta sources re-version @toolconv-merge-20261127 with 42,272 "
                    "replaced rows and V10's assistant-only cached masks. Everything else is "
                    "byte-identical V10.1.",
            "weights": weight_check,
            "named_exclusions": {
                "same_name_rewrites": {
                    "rows": NAMED_EXCLUSION_ROWS,
                    "reason": "same-name argument rewrites whose converted forms cannot be re-derived "
                              f"from the store ({CANONICAL_HASH_JOB}: all 40,374 mismatch)",
                    "resolution": "byte-identical V10 rows in V10.2; resolved in V10.3 (coordinator "
                                  "ruling 2026-11-27, option b)",
                },
            },
            "toolconv": {
                source: {
                    "base": apply_receipt["sources"][source]["base"],
                    "output": apply_receipt["sources"][source]["output"],
                    "replaced_rows": apply_receipt["sources"][source]["replaced_rows"],
                    "replaced_tokens": apply_receipt["sources"][source]["replaced_tokens"],
                    "replaced_original_tokens": apply_receipt["sources"][source]["replaced_original_tokens"],
                    "delta_files": apply_receipt["sources"][source]["delta_files"],
                }
                for source in REPLACED_SOURCES
            },
            "epoch_plan": {"uri": EVIDENCE["epoch_plan"]["uri"], "sha256": EVIDENCE["epoch_plan"]["sha256"]},
        },
        "artifact_graph": EVIDENCE["artifact_graph"],
        "review_evidence": review,
        "inherited_qualification": "The replay population and every unchanged SFT store reuse the V10.1 "
                                   "qualified caches byte-identically (unchanged shards keep their "
                                   "V10/V10.1 shard paths re-expressed relative to the catalog root); "
                                   "the V10.1 release record and its qualification apply verbatim to "
                                   "them. The eight toolconv-delta sources are re-versioned by the "
                                   "V10.2 toolconv merge with the native consumer receipt over the "
                                   "V10.2 release.",
        "readiness_scope": "Data qualification only; production launch requires user approval and the runtime graph resolve.",
        "consumer_limitation": consumer["limitation"],
    }
    path = ROOT / "release-complete-v10.2.json"
    if path.exists():
        raise ValueError("Preserve any prior local seal; do not overwrite it")
    path.write_text(json.dumps(seal, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"local_seal": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "changed_sources": sorted(REPLACED_SOURCES),
                      "exposure_views": seal["exposure_views"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=pathlib.Path)
    args = parser.parse_args()
    EVIDENCE = json.loads(args.evidence.read_text())
    main()
