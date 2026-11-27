---
title: "Sealed mix V10.2: toolconv merge"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/94"
related_issues: ["https://github.com/marin-community/marin-logbooks/issues/89", "https://github.com/marin-community/marin/issues/12371"]
research_branch: "codex/94-v10-2-toolconv-merge"
marin_base_revision: "b6f306dc6e2a9a58f211ce42cb2e09b5f1752497 (the lane's source tree, the same commit the #89 branch starts from; this branch adds the V10.2 stage modules at their Marin locations in commit 4dc1d565b)"
archive_template_revision: "123b82a866495718a8b45f12b7e37c15cff5b939"
status: active
---

# Sealed mix V10.2: toolconv merge

## Question and decision criteria

- Goal and why it matters: version the sealed V10.1 SFT mix with the remaining verified
  tool-conversion delta — 42,272 verified tool-converted rows replacing their unconverted
  originals in the eight toolconv-delta sources — so the Exp7 LLOM lane trains on the fully
  applied conversion campaign without touching the sealed V10/V10.1 caches or the mixture
  weights.
- Baseline and alternatives: V10.1 is the sealed, immutable base
  (`release-v10.1/release-complete-v10.1.json`, sha256 `81fab697…`). Alternatives rejected:
  substituting the 40,374 same-name-rewrite rows (their converted forms cannot be re-derived
  from the store — see the named exclusion below), rewriting all shards (only the 152
  delta-bearing shards carry replaced rows), and changing the mixture weights (the merge
  changes row contents, not the mixture).
- Evidence needed to choose: rows conserved per source, per-shard receipts with digest pins,
  a native-consumer receipt with zero dropped tokens and zero train/val overlap, an epoch plan
  within the declared replay tolerance, and a comparison asserting token changes on exactly the
  eight sources.
- Constraints and unresolved questions: in-region data work only (CoreWeave `cw-us-east-08a`);
  no cross-region bulk transfers; the V10/V10.1 caches stay immutable; the 40,374 same-name
  rows are a named design exclusion resolved in V10.3.

## Named design exclusion (coordinator ruling 2026-11-27, option b)

The 40,374 same-name-rewrite rows (opencode_v1 sources) stay byte-identical V10 rows in
V10.2. The conversion campaign produced rewritten-arguments payloads for them, but the store's
before/after sha256 pins match no rendered/converted-argument hash under raw-string or
sorted-JSON canonicalizations — the canonical-hash check (`/benfeuer/v10-2-canonical-hash-check-20261127`,
3h20m, succeeded) probed the pins end-to-end and all 40,374 rows mismatch. Their converted
forms cannot be re-derived from the store, so any substitution would be unverifiable. The
coordinator approved option (b): record the exclusion in the seal notes with its row count and
reason, and resolve it in V10.3. The release builders, the seal checker, the comparison and
the mixture rationale all carry this block.

## Baseline

- Date: 2026-11-27 (seal ~17:45 EST).
- Published source revision: this branch, start `b6f306dc6e` + the stage scripts commit
  `4dc1d565b` (reconciliation, attribution, index-row export, payload extract, args check,
  canonical-hash check, delta-bindings export at their Marin locations; the V10.2
  release/seal/comparison builders under `logbooks/94-v10-2-toolconv-merge/scripts/`).
- Configuration, command, and external inputs: see the rebuild procedure below; every input is
  a sha-pinned object in `s3://marin-us-east-08a/`.
- Environment and hardware: Iris CPU jobs on `cw-us-east-08a` at interactive priority; the
  offsets-only capacity measurement ran in-process on the operator's Mac (the V10 fallback for
  the federation self-pin).
- Result and evidence: sealed
  `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.2/release-complete-v10.2.json`
  sha256 `228929e63f6f7c54cde9f2d5058261f120592184e7163d82e9731a5f6e9a4094`.
- Limitations: the consumer receipt measures packing/exposure, not per-sampled-document
  assistant-loss position counts; the same-name exclusion defers 40,374 rows to V10.3.

## Entries

### 2026-11-26 05:30 → 11-27 08:20 EST - reconciliation closes the census; the V10.2 delta is named

- Question or hypothesis: which verified-modified rows actually carry unconverted tool blocks
  in the sealed V10 stores, and can every one be attributed by named cause?
- Modules: `experiments/sft_new_mix/v10_1_reconcile_lineage.py` (tracer r10),
  `v10_1_content_attribution.py` (attribution r9), `v10_1_index_row_export.py`; receipts under
  `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/reconcile/final/`.
- Evidence: the tracer partitions all 2,223,483 verified-modified rows exactly
  (`reconcile/final/reconciliation.json`), conflicts 28,787, not_in_build ZERO for math; V10.1
  integrity verified (all 4,383 substitutions byte-exact in the stores-v2 shards). Content
  attribution (`reconcile/final/attribution.json`): applied_baked 1,253,844 (content-confirmed),
  not_applied 40,698 (the V10.2 delta), transformed_no_tool_block 0, mismatch 40,374 (the
  same-name rewrites). The retention CSV's 1,804,539 = the campaign's inline substitutions
  (scripted + atlas real-tools paths, proven by raw-store/per-generation content probes); the
  row-attributable set = 746,682 lineage overlays + 1,253,844 baked; the residual is DUPLICATE
  VERIFICATIONS (the index carries multiple records per raw row: high +94,837, v3 +70,819,
  v4 +14,356 — distinct (part,row) counts reproduce the pre-RNO index table exactly) plus
  RNO-era growth.
- Decision: seal V10.2 with the 40,698-row lineage delta plus the args-check not_applied subset;
  resolve the 40,374 same-name rows separately.

### 2026-11-26 → 11-27 EST - payload extraction and the args check build the 42,272-row delta

- Modules: `experiments/sft_new_mix/v10_2_payload_extract.py` (payload-extract r10, 6h11m job),
  `v10_2_args_check.py` (args-check r2), `v10_2_canonical_hash_check.py`
  (`/benfeuer/v10-2-canonical-hash-check-20261127`, 3h20m, succeeded).
- Evidence: 40,698 payloads extracted to `reconcile/v10-2/` (13,958 re-tokenized from text with
  the pinned 2026.09.18 tokenizer; 1,108 rescued via the range scan with digest verification;
  bindings sha `f7ca7a3c…`). Args-check receipt (`reconcile/final/args-check-receipt.json`):
  4,407 applied_baked, 1,574 not_applied (payloads extracted to `reconcile/v10-2-args/`, 286
  archives), 37,993 partial-hit. The canonical-hash check found no rendered/converted-argument
  hash for any of the 40,374 same-name rows under either canonicalization → the named exclusion.
- Decision: the coordinator approved option (b) 2026-11-27 — proceed with the 42,272-row delta;
  the 40,374 rows stay byte-identical V10 rows, resolved in V10.3.

### 2026-11-27 EST - merged delta bindings; apply-merge r3 writes stores-v4

- Module: `experiments/sft_new_mix/v10_2_delta_bindings_export.py` merged the lineage
  not_applied bindings (key-space fixed: the attribution's absolute store URIs → the V10.1
  ledger keys) with the args-check not_applied payloads into 8 per-source files at
  `reconcile/v10-2/merged-per-source/` (combined sha `29181cd8…`); 42,272/42,272 bound against
  the V10.1 catalog, output_row in range for every binding.
- Apply: `/benfeuer/v10-2-apply-merge-20261127-r3` (cw-us-east-08a, interactive; module
  `experiments.sft_new_mix.v10_1_toolconv_apply` with the V10.1 catalog pin and the 8
  merged-per-source bindings; the same invocation shape as the #89 notebook's step 3, with
  `--output-root .../stores-v4 --destination .../apply-v4`) — state succeeded, exit 0,
  failures 0, preemptions 0, 1/1 task, 29m25s. stores-v3 (the earlier 40,698-only apply) is
  superseded by stores-v4.
- Receipt verification (`apply-v4/apply-receipt.json`, sha
  `e16e3ff1a878e6e0402a50a568a92cc3320a9b50520ea18c07038f97b036a8f7`): status COMPLETE; rows
  conserve per source (row_delta 0 ×8); replaced_rows 42,272 = the per-shard sum; 270/270 base
  shards accounted (152 rewritten → stores-v4, 118 unchanged keep their V10/V10.1 paths); all
  152 rewritten shards carry per-shard receipts (`complete=true`, rows/tokens conserve,
  `output_tokens_and_weights_sha256` recorded); net token delta −132,820,164.

### 2026-11-27 ~16:35 EST - release assembled; the named exclusion enters the release metadata

- Script: `logbooks/94-v10-2-toolconv-merge/scripts/build_release_v10_2.py` over the pinned
  V10.1 seal (`81fab697…`) and the apply-v4 receipt.
- Command:
  ```bash
  uv run python build_release_v10_2.py \
    --apply-receipt-uri s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/apply-v4/apply-receipt.json \
    --apply-receipt-sha256 e16e3ff1a878e6e0402a50a568a92cc3320a9b50520ea18c07038f97b036a8f7 \
    --output-prefix s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.2
  ```
- Result: manifest sha `7eb2814708af185a9c76726d41e8a70dc8acf83d42a4c179ee3104b836ff7bd6`
  (frozen pre-consumer, training_ready false). Catalog `41a52e33…` (434 sources; ONLY the 8
  entries re-keyed — unchanged shards byte-identical V10.1 paths, rewritten shards under
  stores-v4); `train_weights.json` `7fd39c1f…` and `source_loss.json` `6bae0724…` byte-identical
  V10.1 members; `data.json` `b6c33bc5…` only repoints the catalog; the mixture rationale gains
  `v10_2_amendments` (the ruling, the 42,272-row delta, and the 40,374 named exclusion); the
  retention CSVs carry the 8 `v10_2_toolconv` rows. Publication recipe
  `inputs/publication-recipe-release-v10.2.json` sha `abfab5cc…`.

### 2026-11-27 ~16:35–16:50 EST - native consumer receipt

- Job: `/benfeuer/v10-2-native-consumer-20261127` (target-cluster cw-us-east-08a, interactive,
  32 CPU / 128 GB / 512 GB, timeout 3d) — module
  `experiments.post_training.long_loop.exp3_padding` in worktree `exp3-data-consumer` pinned at
  `6db3ce9022a414804142d02f6edd2fe47befc632`:
  ```bash
  uv run --no-sync iris --cluster=marin job run --target-cluster=cw-us-east-08a --priority interactive \
    --cpu 32 --memory 128GB --disk 512GB --enable-extra-resources --timeout 259200 \
    --job-name v10-2-native-consumer-20261127 --user benfeuer --no-wait -- \
    python -m experiments.post_training.long_loop.exp3_padding \
    --job s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.2/native-measurement-job.json \
    --output s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.2/native-consumer-receipt.json
  ```
- Result: succeeded in 14m53.8s (exit 0, failures 0, preemptions 0). Receipt
  `release-v10.2/native-consumer-receipt.json` sha
  `d457e381629adeb4c7b8acd6f7d95d51a7227249d700d8cb8e2f219816c253ee`: complete over 434 sources
  exact, zero dropped tokens, zero train/val overlap, 56 reserved validation sequences, exposure
  views 360/377/746.

### 2026-11-27 ~17:00–17:20 EST - capacity, epoch plan, comparison

- Capacity (in-process offsets-only `measure_capacity`, the V10 fallback — the federation
  self-pin still blocks the remote child; worktree `east02a-infiniband-20261123`, data pin =
  the V10.2 `data.json` `b6c33bc5…`):
  `.../nemotron-overthinking/2026.11.25-v10-1-balanced-toolconv/sft-epoch-capacity/2026.11.27.1/capacity.json`
  sha `3237f0024d14462a52962c81df8ef9ae345be80f925a046b20e25dce386daa6d` (`token_bodies_read: false`).
- Epoch plan (same prefix) sha `76ef8ac3f089bcad062b36fa7a9c88f3b913d28138a576bf769a829e13203731`:
  `plan_mixture_epoch` over the capacity's SFT sources, batch 256, replay 0.10037232630869815 —
  50 SFT stores, packed SFT sequences 386,148 (V10.1: 386,696; −548), run_sequences 429,312,
  updates 1,677 (V10.1: 1,680), realized replay 0.10054226296958856 (+0.017pp vs declared;
  V10.1's realization was +0.050pp — same direction from discrete block packing). The
  `v10_1_reference` block is read from the sealed V10.1 plan (sha `41b227fd…`). One mixture
  epoch; LLOM `--sft-steps` = 1,677.
- Comparison (`build_mix_comparison_v10_2.py`): rows conserve everywhere, token changes on
  exactly the 8 sources, weights byte-identical V10.1's (asserted equal); store totals
  133,979,903,214 → 133,847,083,050 tokens (−132,820,164); rows 44,724,744 → 44,724,744;
  updates 1,680 → 1,677. Uploaded to
  `s3://.../2026.11.25-v10-1-balanced-toolconv/mix-comparison-v2.{md,csv}` (md sha `d525f603…`,
  csv sha `b767c834…`).

### 2026-11-27 ~17:45 EST - SEALED

- Artifact record: `artifact-graph/sft-mixes/exp3-v10-2/release/2026.11.27/.artifact.json`
  (fingerprint `b562a076`, record sha `e2663408…`, payload_matches true), published from the
  frozen manifest per `scripts/publish_record_v10.py`.
- Seal: `scripts/prepare_seal_v10_2.py` with the evidence pins (publisher manifest
  `7eb28147…`, apply receipt `e16e3ff1…`, consumer receipt `d457e381…`, capacity `3237f002…`,
  epoch plan `76ef8ac3…`, comparison `d525f603…`/`b767c834…`, artifact graph `e2663408…`,
  checker at `inputs/prepare-seal-v10-2.py` sha `937647e9…`) verified: the inherited V10.1
  qualification, the byte-identical weights and supervision sidecar, the supervision-vs-masks
  census (50 chat_assistant_only / 384 full_token), the apply-v4 receipt over the V10.1 catalog
  with per-source shard-map integrity (unchanged shards keep their V10.1 paths; rewritten shards
  in stores-v4), the tokenizer pin `6a61ca21…`, and the consumer receipt.
- Seal:
  `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.2/release-complete-v10.2.json`
  **sha256 `228929e63f6f7c54cde9f2d5058261f120592184e7163d82e9731a5f6e9a4094`** (downloaded and
  re-verified after upload). V10.2 supersedes nothing: the V10 and V10.1 seals stay qualified.
- Registry: [marin#12371](https://github.com/marin-community/marin/issues/12371) carries the
  V10.2 row after V10.1.

## Rebuild procedure (from scratch)

Pin the region: CoreWeave `cw-us-east-08a` S3 (`s3://marin-us-east-08a/...`); run data work as
Iris CPU jobs at `--priority interactive`; never move bulk data through a laptop. The source
tree is this branch (`codex/94-v10-2-toolconv-merge`): start `b6f306dc6e`, stage scripts
`4dc1d565b`. The lane ran the same modules from the uncommitted working tree of worktree
`sft-new-mix-v7`, branch `atlas-split/sft-new-mix`.

1. **Inputs (all sha-pinned).**
   - V10.1 seal:
     `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.1/release-complete-v10.1.json`
     sha256 `81fab6977b6dc2acb904a9027f5c0e0adc062eee32a1d09e0dbf9a5a20a4759e` (its catalog
     `b415042c…`, weights `7fd39c1f…`, supervision `6bae0724…`).
   - Merged delta bindings (8 per-source files):
     `.../2026.11.25-v10-1-balanced-toolconv/reconcile/v10-2/merged-per-source/sft__source__nemotron_sft_v3__<source>.jsonl`
     — 42,272 rows total (lineage not_applied 40,698 + args-check not_applied 1,574), keys =
     V10.1 ledger keys, payloads under `reconcile/v10-2{,-args}/`.
   - Store tokenizer:
     `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/scripted-v3/tokenizer.json`
     sha256 `6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45`.
   - The reconciliation census:
     `reconcile/final/reconciliation.json` + `reconcile/final/attribution.json` +
     `reconcile/final/args-check-receipt.json` (named buckets; the named exclusion's evidence).

2. **Toolconv merge apply** (the only data-rewriting stage): the same invocation shape as the
   #89 notebook's step 3, with the V10.1 catalog pin, the 8 merged-per-source bindings,
   `--output-root .../stores-v4 --destination .../apply-v4` (job
   `/benfeuer/v10-2-apply-merge-20261127-r3`, succeeded 29m25s). Receipt check: rows conserved,
   replaced == 42,272, per-shard receipts with `output_tokens_and_weights_sha256`, unchanged
   shards keeping their V10.1 paths.

3. **Release assembly** (local metadata script; small pinned files only):
   ```bash
   uv run python logbooks/94-v10-2-toolconv-merge/scripts/build_release_v10_2.py \
     --apply-receipt-uri .../apply-v4/apply-receipt.json \
     --apply-receipt-sha256 e16e3ff1a878e6e0402a50a568a92cc3320a9b50520ea18c07038f97b036a8f7 \
     --output-prefix s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.2
   ```
   Asserts 434 sources, byte-identical untouched entries, unchanged-shard path integrity, the
   byte-identical weights and supervision members, and the named-exclusion metadata; writes the
   frozen `manifest.json`.

4. **Native consumer receipt** (worktree `exp3-data-consumer` pinned at `6db3ce9022a4`): the
   same invocation as the #89 notebook's step 5 with `--job-name v10-2-native-consumer-<date>`
   over `release-v10.2/native-measurement-job.json`. Check: complete; 434 sources exact; zero
   dropped tokens; zero train/val overlap.

5. **Capacity + epoch plan** (worktree `east02a-infiniband-20261123`): the remote child hits
   the federation self-pin, so run `measure_capacity` in-process (offsets only,
   `token_bodies_read: false`) over the pinned V10.2 `data.json`, then
   `plan_mixture_epoch` (batch 256, replay 0.10037232630869815) over the capacity receipt's SFT
   sources; write both to
   `.../nemotron-overthinking/2026.11.25-v10-1-balanced-toolconv/sft-epoch-capacity/<date>.1/`.

6. **Comparison**: `logbooks/94-v10-2-toolconv-merge/scripts/build_mix_comparison_v10_2.py`
   over the V10.1 seal, the frozen V10.2 manifest, the apply receipt, the V10.1 epoch plan and
   the V10.2 capacity/epoch receipts; it asserts row conservation everywhere, token changes on
   exactly the 8 sources, and byte-identical weights. Upload `mix-comparison-v2.{md,csv}`.

7. **Artifact record**: `scripts/publish_record_v10.py --name sft-mixes/exp3-v10-2/release
   --version <date> --manifest-uri <release-v10.2>/manifest.json --manifest-sha256 <sha>
   --members '<manifest members>' --graph-prefix .../artifact-graph/sft-mixes/exp3-v10-2/release/<date>`.

8. **Seal**: upload the checker to `inputs/prepare-seal-v10-2.py`, write the evidence JSON
   (publisher manifest, apply receipt, native job, consumer receipt, comparison md/csv,
   capacity, epoch plan, artifact graph, checker URI), then
   `uv run python scripts/prepare_seal_v10_2.py --evidence <evidence.json>`; upload
   `release-complete-v10.2.json` to the release prefix and re-verify the downloaded sha.
