---
title: "Sealed mix V10.1: toolconv replacement + balanced reasoning"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/89"
related_issues: []
research_branch: "codex/89-v10-1-toolconv-balanced"
marin_base_revision: "b6f306dc6e2a9a58f211ce42cb2e09b5f1752497 (the lane's source tree: Marin 82cee32cfd5bd479aa0670ba1a92a9c0f96ec4bf + the atlas-split/sft-new-mix chain; this branch starts from that exact commit)"
archive_template_revision: "3846d697465e8629294936c6a29e2061bb1c296d"
status: active
---

# Sealed mix V10.1: toolconv replacement + balanced reasoning

## Question and decision criteria

- Goal and why it matters: version the sealed V10 SFT mix with (a) 4,383 verified tool-converted rows
  replacing their originals in the three math sources and (b) one new balanced-reasoning component,
  so the Exp7 LLOM lane (marin PR #12284) can train on all positions over a sealed release.
- Baseline and alternatives: V10 is the sealed, immutable base
  (`release-v10/release-complete-v10.json`, sha256 `e0838c00…`). Alternatives rejected: rewriting
  every cached mask to full-token (contradicts the LLOM consumption view and would desynchronize
  `source_loss.json`), rewriting all 237 math shards (25B+ tokens of redundant I/O), and waiting for
  the RNO-era index ingest (the verified floor is already buildable).
- Evidence needed to choose: rows conserved at the replaced rows, the recomputed assistant masks
  equal to the stage-E extract's own counts, a native-consumer receipt with zero dropped tokens and
  zero train/val overlap, and an epoch plan within V10's tolerance of the declared replay share.
- Constraints and unresolved questions: in-region data work only (CoreWeave `cw-us-east-08a`); no
  cross-region bulk GCS; the V10 caches stay immutable; the RNO-era verified-modified index remains
  a conservative floor (2,378,592 rows as of the 2026-11-25 snapshot).

## Baseline

- Date: 2026-11-26 05:20 EST.
- Published source revision: `b6f306dc6e2a9a58f211ce42cb2e09b5f1752497` (this branch's start).
- Configuration, command, and external inputs: see the rebuild procedure below; every input is a
  sha-pinned object in `s3://marin-us-east-08a/` (or the 02a balanced store).
- Environment and hardware when relevant: Iris CPU jobs on `cw-us-east-08a` at interactive priority;
  the offsets-only capacity measurement ran in-process on the operator's Mac (V10's known fallback
  for the federation self-pin).
- Result and evidence: sealed
  `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.1/release-complete-v10.1.json`
  sha256 `81fab6977b6dc2acb904a9027f5c0e0adc062eee32a1d09e0dbf9a5a20a4759e`.
- Limitations: the consumer receipt measures packing/exposure, not per-sampled-document
  assistant-loss position counts; the verified-modified population is a floor (RNO-era archives
  unindexed); 10,654 rows are quarantined by the tokenizer gate, never silently substituted.

## Entries

### 2026-11-26 04:29 UTC - apply v2 fails on one missing stage-E payload; repair re-fetches it

- Question or hypothesis and reason for trying it: the apply job must substitute 4,383 payload
  `.i32` files; any missing payload must stop the job (never silently drop a row).
- Published source revision: `b552a3e5b6` (apply fixes F1/F2/F5/F6; full history on
  `atlas-split/sft-new-mix`).
- Command or material configuration: `python -m experiments.sft_new_mix.v10_1_toolconv_apply`
  with the V10 catalog pin, the three merged delta JSONLs, `--output-root .../stores-v2`,
  `--destination .../apply-v2`, `--workers 8`, 16 CPU / 64 GB / 256 GB (job
  `/benfeuer/v10-1-toolconv-apply-20261126-v2-r2`, succeeded 6m53s, PEAK MEM 16.7 GiB; the first
  `/benfeuer/v10-1-toolconv-apply-20261126-v2` died 4m34s in).
- External inputs and artifact references: apply receipt
  `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/apply-v2/apply-receipt.json`
  sha256 `041fffcad2d74f2f268bf59a318d66302d788b767a58acdffe608a003d578934`; delta files
  `delta-high-merged-08a.jsonl` (1,271 rows, sha256 `b86149b3…`), `delta-v3-merged.jsonl` (2,388,
  `9910338c…`), `delta-v4-merged.jsonl` (724, `d32aef64…`) under `.../toolconv-extract-08a/`.
- Result, including failure or restart: v2 raised FileNotFoundError on payload
  `01ff97e0…` — the stage-E extract receipt records `delta_rows 2388, payloads_written 2387,
  fetch_errors 1` for math_v3/train: the extract caught one archive-fetch exception and left the row
  in the delta file without a payload. Restart: repair job
  `/benfeuer/v10-1-toolconv-payload-repair-20261126` (44s) re-fetched that one archive in-region,
  verified the archive + export-member digests, validated the store-tokenizer decode and the
  token/supervised counts (25,014 / 20,572), wrote the payload (sha `8d6c394b…` == its export member
  pin), and verified 0 missing of 4,383 payload URIs (receipt `.../repairs/payload-repair-receipt.json`,
  sha256 `7f7522ae…`).
- Evidence: rows conserved per source 901,810 / 3,060,671 / 538,654 (row_delta 0); replaced
  1,271 / 2,388 / 724; token deltas +9,816,366 / +13,984,963 / +8,960,599 == payload − original
  tokens; 15 rewritten shards (9/3/3) + 222 unchanged; every per-shard receipt carries
  `output_tokens_and_weights_sha256`.
- Interpretation and limitations: the loop's earlier failures (r1–r15) were F1 (a `source`-field
  filter that silently dropped every delta record — the files carry no such field), F2/F3 (an empty
  worker-initializer tokenizer), F4 (submits past lint failures), F5 (an unnecessary full-token mask
  rewrite) and F6 (rewriting all 237 shards); the v2 script fixes all of them and asserts the
  per-source applied counts and row bounds.
- Decision and next action: proceed to release assembly with unchanged shards re-keyed relative to
  the catalog root.

### 2026-11-26 05:20 UTC - V10.1 sealed: 434 sources, 1,680 updates

- Question or hypothesis and reason for trying it: assemble V10 = sealed V10 + the two changes and
  seal it without user approval (gate removed 2026-11-25 4:15 PM EST).
- Published source revision: `b6f306dc6e` (build/seal scripts in `logbooks/89-v10-1-toolconv-balanced/scripts/`).
- Command or material configuration: `build_release_v10_1.py` →
  `build_mix_comparison_v10_1.py` → in-process `measure_capacity` + `plan_mixture_epoch` →
  `exp3_padding` native consumer → `prepare_seal_v10_1.py` (exact commands in the rebuild procedure).
- External inputs and artifact references: the V10 seal
  (`.../release-v10/release-complete-v10.json`, sha `e0838c00…`); the staged balanced cache
  (`.../balanced-cache-v1/cache/shard_ledger.json` on 08a, sha `da1a992c…`; 7,728 rows / 4,070,781
  tokens / 2,565,856 supervised; `fsutil verified-copy` from 02a, 10 copied / 0 resumed); the apply
  receipt; the capacity receipt (sha `abb0056a…`); the epoch plan (sha `41b227fd…`); the native
  consumer receipt (job `/benfeuer/v10-1-native-consumer-20261126`, succeeded 14m40s, sha
  `29c2ff97…`).
- Result: release members at `.../release-v10.1/` (catalog sha `b415042c…`, data.json
  `3dc039e8…`, train_weights `7fd39c1f…`, source_loss `6bae0724…`, manifest `55ba9250…`, frozen
  pre-consumer); MIX-COMPARISON-V10.1 (population differs for exactly the 3 math sources + balanced;
  133.943B → 133.980B tokens, 44,717,016 → 44,724,744 rows); epoch plan 50 SFT stores, packed SFT
  sequences 386,696 (V10: 386,542), run_sequences 430,080, updates 1,680 (V10: 1,679), realized
  replay 0.10087425595238095 vs declared 0.10037232630869815 (V10 realized 0.100697; the +0.018pp
  shift is the discrete block packing of the +154 sequences); consumer COMPLETE over 434 sources,
  zero dropped tokens, zero train/val overlap; artifact record
  `sft-mixes/exp3-v10-1/release@2026.11.26` (fingerprint `f802e680`, record sha `774c8d20…`).
- Evidence: the seal sha256 `81fab6977b6dc2acb904a9027f5c0e0adc062eee32a1d09e0dbf9a5a20a4759e`,
  downloaded and re-verified after upload.
- Interpretation and limitations: the seal checker verifies the inherited V10 seal, the weights rule
  (balanced 0.005, SFT ×(S−0.005)/S, pretrain byte-identical, phases sum to 1.0 within 1e-12), the
  supervision sidecar against the cached masks (50 chat_assistant_only / 384 full_token), the
  unchanged-shard-path integrity (every unchanged shard composes to its exact V10 URI), the apply/
  repair/balanced receipts and the tokenizer pins (`6a61ca21…`). The consumer limitation and the
  verified-floor caveat above apply.
- Decision and next action: Exp7 consumes the receipt with `--sft-steps 1680`; one mixture epoch.

## Final decision

- Selected approach and evidence: V10.1 = sealed V10 + toolconv replacement (assistant-only
  supervision preserved) + balanced reasoning at w = 0.005; sealed and verified as above.
- Rejected alternatives and why: full-token mask rewrite (F5 — the LLOM consumption view supplies
  all-position loss; rewriting 3 sources would desynchronize `source_loss.json`); full 237-shard
  rewrite (F6 — ~25B redundant tokens); waiting on the RNO-era index ingest (the verified floor is
  buildable and the coverage boundary is recorded).
- Marin contribution PRs: none from this lane; the LLOM consumption view is marin PR #12284 (Exp7).
  The apply/extract scripts stay on the research branch (`experiments/sft_new_mix/`).
- Reproduction instructions: the rebuild procedure below.
- Remaining uncertainty: the RNO-era verified-modified index is a floor; a later V10.2 could widen
  the applied set. The 10,654 quarantined rows are excluded by design.

## Rebuild procedure (from scratch)

Pin the region: CoreWeave `cw-us-east-08a` S3 (`s3://marin-us-east-08a/...`); run data work as Iris
CPU jobs at `--priority interactive`; never move bulk data through a laptop. The source tree is this
branch (`codex/89-v10-1-toolconv-balanced`, start `b6f306dc6e`); the lane ran the same modules from
the uncommitted working tree of worktree `sft-new-mix-v7`, branch `atlas-split/sft-new-mix`.

1. **Inputs (all sha-pinned).**
   - V10 catalog: `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.23-v10-overthinking/release-v10/cache_catalog.json`
     sha256 `afd1b128e2ee7ad26c7027b347e1ef588138eecf3d678a4de242ba7fbed2f9a5` (433 sources; the 3
     math entries: 901,810 / 3,060,671 / 538,654 rows over 59/158/20 shards).
   - Stage-E delta bindings (bound by `shard_key` membership; the files carry no source field):
     `.../2026.11.25-v10-1-balanced-toolconv/toolconv-extract-08a/delta-high-merged-08a.jsonl`
     (1,271 rows, `b86149b3…`), `delta-v3-merged.jsonl` (2,388, `9910338c…`), `delta-v4-merged.jsonl`
     (724, `d32aef64…`); 4,383 unique ids; `retokenized: false` everywhere; payloads under
     `.../toolconv-extract-08a{,-02a-mirror}/payloads/`.
   - Store tokenizer: `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/scripted-v3/tokenizer.json`
     sha256 `6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45`.
   - Balanced store rows (02a): `s3://marin-us-east-02a/marin/users/benfeuer/balanced-reasoning/v1-store-2026.11.24.2/rows.parquet`
     sha256 `6b5dc656130f4f946a793172ef4495d75f67c1d26e30ad813d0b42e3756ec9be` (the source_revision
     pin); cache build receipt `.../balanced-cache-v1/receipt.json` (02a) sha256
     `a0e0a276e337a477a666bd83cf6cf1a909e60e607708d4398449e1d5bce6b28f`.

2. **Balanced cache build** (only if rebuilding the 08a copy): `v10_1_balanced_cache.py` produced
   the 7,728-row `chat_assistant_only` cache (job `/benfeuer/v10-1-balanced-cache-20261125`, 02a,
   46s); stage it into 08a with
   `uv run fsutil verified-copy s3://marin-us-east-02a/marin/users/benfeuer/nemotron-overthinking/2026.11.25-v10-1-balanced-toolconv/balanced-cache-v1/cache s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/balanced-cache-v1/cache`
   (verified: 10 copied / 0 resumed; ledger sha `da1a992c…`).

3. **Toolconv apply** (the only data-rewriting stage):
   ```bash
   uv run iris --cluster=cw-us-east-08a job run --no-wait --priority interactive --timeout 86400 \
     --job-name v10-1-toolconv-apply-<date> --cpu 16 --memory 64GB --disk 256GB --enable-extra-resources -- \
     python -m experiments.sft_new_mix.v10_1_toolconv_apply \
     --catalog-uri s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.23-v10-overthinking/release-v10/cache_catalog.json \
     --catalog-sha256 afd1b128e2ee7ad26c7027b347e1ef588138eecf3d678a4de242ba7fbed2f9a5 \
     --source sft/source/nemotron_sft_v3/math_v2/high --source sft/source/nemotron_sft_v3/math_v3/train \
     --source sft/source/nemotron_sft_v3/math_v4/train \
     --delta-bindings s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/toolconv-extract-08a/delta-high-merged-08a.jsonl \
     --delta-bindings s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/toolconv-extract-08a/delta-v3-merged.jsonl \
     --delta-bindings s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/toolconv-extract-08a/delta-v4-merged.jsonl \
     --output-root s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/stores-v2 \
     --destination s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/apply-v2 --workers 8
   ```
   Watch `rewriting N shards (K delta rows)` = K 1,271 / 2,388 / 724. If a payload is missing, use
   `v10_1_toolconv_payload_repair.py` (its `--delta-paths` re-verifies the full 4,383 population).
   Receipt check: rows conserved, replaced == 4,383, token delta == payload − original tokens,
   per-shard receipts with `output_tokens_and_weights_sha256`.

4. **Release assembly** (local metadata script; small pinned files only):
   ```bash
   uv run python logbooks/89-v10-1-toolconv-balanced/scripts/build_release_v10_1.py \
     --apply-receipt-uri .../apply-v2/apply-receipt.json --apply-receipt-sha256 041fffcad2d74f2f268bf59a318d66302d788b767a58acdffe608a003d578934 \
     --balanced-ledger-uri .../balanced-cache-v1/cache/shard_ledger.json --balanced-ledger-sha256 da1a992cfc7280bd2f361d5ded093558b68851c2d3783a416b0acc1a1f6540e0 \
     --output-prefix s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.1
   ```
   Asserts 434 sources, byte-identical untouched entries, unchanged-shard path integrity, the weight
   rule and the supervision additions; writes the frozen `manifest.json`.

5. **Native consumer receipt** (worktree `exp3-data-consumer` pinned at `6db3ce9022a4`):
   ```bash
   uv run --no-sync iris --cluster=marin job run --target-cluster=cw-us-east-08a --priority interactive \
     --cpu 32 --memory 128GB --disk 512GB --enable-extra-resources --timeout 259200 \
     --job-name v10-1-native-consumer-<date> --user benfeuer --no-wait -- \
     python -m experiments.post_training.long_loop.exp3_padding \
     --job <release-v10.1>/native-measurement-job.json --output <release-v10.1>/native-consumer-receipt.json
   ```
   Check: complete; 434 sources exact; zero dropped tokens; zero train/val overlap.

6. **Capacity + epoch plan** (worktree `east02a-infiniband-20261123`): the remote child hits the
   federation self-pin on 08a, so run `measure_capacity` in-process (offsets only, `token_bodies_read:
   false`) over the pinned `data.json` sha `3dc039e8…`, then `plan_mixture_epoch(..., batch_size=256,
   replay_fraction=0.10037232630869815)` → `epoch-plan.json` (updates **1,680**; realized replay
   0.10087425595238095).

7. **Comparison + seal**: `build_mix_comparison_v10_1.py` (asserts exactly the 3 math sources + the
   balanced component differ), then `prepare_seal_v10_1.py --evidence <json>` writes the local seal;
   upload to `<release-v10.1>/release-complete-v10.1.json` and re-verify the downloaded sha256 ==
   `81fab6977b6dc2acb904a9027f5c0e0adc062eee32a1d09e0dbf9a5a20a4759e`. Publish the artifact record
   with `publish_record_v10.py --name sft-mixes/exp3-v10-1/release --version 2026.11.26`.

Every receipt, manifest, capacity/epoch plan and comparison stays in object storage at the URIs
above with its sha256; nothing in this notebook embeds them.
