# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixture tests for the V10.1 toolconv apply: a 2-shard toy store with one delta row."""

import asyncio
import json

import numpy as np
from levanter.store.cache import TreeCache, write_levanter_cache
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from experiments.sft_new_mix.exp3_filters import BOS, END_HEADER, EOS, EOT, HEADER
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights
from experiments.sft_new_mix.v10_1_toolconv_apply import (
    apply_source,
    load_delta_bindings,
    root_relative_key,
)

SOURCE = "sft/source/nemotron_sft_v3/math_v4/train"
SHARD_A = "shards/aaaa/cache"
SHARD_B = "shards/bbbb/cache"
EXEMPLAR = {"input_ids": np.empty(0, dtype=np.int32), "loss_weights": np.empty(0, dtype=np.float32)}


def toy_tokenizer(tmp_path):
    vocabulary = {character: index for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([BOS, EOS, HEADER, END_HEADER, EOT])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return tokenizer, tokenizer_path


def chat_tokens(tokenizer, tool_name):
    text = (
        BOS
        + HEADER
        + "user"
        + END_HEADER
        + "\nSolve."
        + EOT
        + HEADER
        + "assistant"
        + END_HEADER
        + "\n<tool_call>"
        + json.dumps({"name": tool_name, "arguments": {"command": "pwd"}})
        + "</tool_call>"
        + EOT
        + EOS
    )
    return np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int32)


def assistant_weights(tokenizer, tokens):
    return prediction_weights(
        tokens,
        loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
        message_start_id=tokenizer.token_to_id(HEADER),
        assistant_prefix=tuple(tokenizer.encode(HEADER + "assistant" + END_HEADER + "\n", add_special_tokens=False).ids),
        trained_assistant_turns=None,
    )


def build_base_store(tokenizer, store_root):
    """A 2-shard store, 2 chat rows per shard, with assistant-only masks."""
    shards = {}
    for shard_key, tool_names in ((SHARD_A, ("bash", "bash")), (SHARD_B, ("bash", "bash"))):
        rows = [chat_tokens(tokenizer, name) for name in tool_names]
        path = f"{store_root}/{shard_key}"
        write_levanter_cache(
            [{"input_ids": ids, "loss_weights": assistant_weights(tokenizer, ids)} for ids in rows],
            path,
            metadata={"source": SOURCE, "loss_mode": "chat_assistant_only"},
        )
        shards[shard_key] = rows
    return shards


def build_catalog(shards, store_root):
    rows_by_shard = {key: len(rows) for key, rows in shards.items()}
    tokens_by_shard = {key: sum(len(row) for row in rows) for key, rows in shards.items()}
    return {
        SOURCE: {
            "cache_dir": str(store_root),
            "ledger": {
                "total_num_rows": sum(rows_by_shard.values()),
                "finished_shards": sorted(shards),
                "shard_rows": rows_by_shard,
                "field_counts_by_shard": {key: {"input_ids": tokens} for key, tokens in tokens_by_shard.items()},
                "field_counts": {"input_ids": sum(tokens_by_shard.values())},
                "metadata": {"preprocessor_metadata": {"loss_mode": "chat_assistant_only"}},
            },
        }
    }


def read_rows(cache_path, count):
    cache = TreeCache.load(str(cache_path), EXEMPLAR)
    return asyncio.run(cache.get_batch(list(range(count))))


def test_one_delta_replaces_one_row_and_conserves_the_population(tmp_path):
    tokenizer, tokenizer_path = toy_tokenizer(tmp_path)
    base_root = tmp_path / "base"
    shards = build_base_store(tokenizer, base_root)
    catalog = build_catalog(shards, base_root)

    payload = chat_tokens(tokenizer, "stateful_python_code_exec")
    payload_path = tmp_path / "payload.i32"
    payload_path.write_bytes(payload.astype("<i4").tobytes())
    weights = assistant_weights(tokenizer, payload)
    delta_path = tmp_path / "delta.jsonl"
    delta_path.write_text(
        json.dumps(
            {
                "id": "delta-1",
                "shard_key": SHARD_A,
                "output_row": 1,
                "payload_uri": str(payload_path),
                "tokens": len(payload),
                "supervised_tokens": int(weights.sum()),
                "retokenized": False,
            }
        )
        + "\n"
    )

    output_root = tmp_path / "stores-v2"
    bindings = load_delta_bindings([str(delta_path)], catalog)
    tokenizer_json = tokenizer_path.read_text()
    receipt = apply_source(SOURCE, catalog[SOURCE], bindings[SOURCE], str(output_root), tokenizer_json, 1, str(tmp_path))

    assert receipt["base"]["rows"] == 4
    assert receipt["output"]["rows"] == 4 and receipt["output"]["row_delta"] == 0
    assert receipt["replaced_rows"] == 1
    assert receipt["output"]["rewritten_shards"] == 1 and receipt["output"]["unchanged_shards"] == 1
    assert receipt["replaced_tokens"] == len(payload)
    assert receipt["replaced_original_tokens"] == len(shards[SHARD_A][1])

    # Only the delta-bearing shard is rewritten; the other keeps its V10 path.
    rewritten = receipt["shards"][0]
    assert rewritten["shard_key"] == SHARD_A and rewritten["replaced_rows"] == 1
    unchanged = receipt["unchanged_shards"][0]
    assert unchanged["shard_key"] == SHARD_B
    assert unchanged["cache_uri"] == f"{base_root}/{SHARD_B}"

    output = read_rows(rewritten["output_cache_uri"], 2)
    np.testing.assert_array_equal(output[1]["input_ids"], payload)
    np.testing.assert_array_equal(output[1]["loss_weights"], weights)
    # The unchanged row keeps its V10 bytes in both fields.
    np.testing.assert_array_equal(output[0]["input_ids"], shards[SHARD_A][0])
    np.testing.assert_array_equal(output[0]["loss_weights"], assistant_weights(tokenizer, shards[SHARD_A][0]))

    # The shard map covers every shard; the unchanged one points at the V10 shard.
    assert sorted(receipt["shard_map"]) == sorted(shards)
    assert receipt["shard_map"][SHARD_B]["cache_uri"] == f"{base_root}/{SHARD_B}"
    assert receipt["shard_map"][SHARD_A]["root_relative_key"] == root_relative_key(
        receipt["shard_map"][SHARD_A]["cache_uri"], str(tmp_path)
    )


def test_bindings_reject_records_outside_the_selected_sources(tmp_path):
    tokenizer, _ = toy_tokenizer(tmp_path)
    shards = build_base_store(tokenizer, tmp_path / "base")
    catalog = build_catalog(shards, tmp_path / "base")

    payload_path = tmp_path / "payload.i32"
    payload_path.write_bytes(chat_tokens(tokenizer, "bash").astype("<i4").tobytes())
    record = {
        "id": "delta-1",
        "shard_key": "shards/unknown/cache",
        "output_row": 0,
        "payload_uri": str(payload_path),
        "tokens": 10,
        "supervised_tokens": 5,
    }
    delta_path = tmp_path / "delta.jsonl"
    delta_path.write_text(json.dumps(record) + "\n")
    try:
        load_delta_bindings([str(delta_path)], catalog)
    except ValueError as error:
        assert "not in any selected source" in str(error)
    else:
        raise AssertionError("a record outside every selected source was accepted")

    record["shard_key"] = SHARD_A
    delta_path.write_text(json.dumps(record) + "\n" + json.dumps({**record, "id": "delta-2"}) + "\n")
    try:
        load_delta_bindings([str(delta_path)], catalog)
    except ValueError as error:
        assert "duplicate output_row" in str(error)
    else:
        raise AssertionError("two records for one output row were accepted")
