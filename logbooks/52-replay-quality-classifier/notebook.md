---
title: Reusable low-value document classifier
source_repository: marin-community/marin
task_issue: https://github.com/marin-community/marin-logbooks/issues/52
related_issues: ["https://github.com/marin-community/marin/issues/11188", "https://github.com/marin-community/marin-logbooks/issues/40"]
research_branch: codex/52-replay-quality-classifier
marin_base_revision: 6e09d9043eb71d7bd487b5e3d79e275b95c454fd
archive_template_revision: 3744203cdaa19ab4b4943e9836e143e7a4c339c4
status: active
---

# Reusable low-value document classifier

## Question and decision criteria

Can a small model improve low-value recall over junkgate2 and rules at cutoffs chosen on separate calibration documents targeting 2% and 5% natural-document false-drop? Model choice uses development data only; a fresh grouped test split is frozen before annotation. A useful candidate must improve recall with a paired uncertainty interval and retain a low observed natural false-drop rate. Macro-stratum estimates do not establish corpus prevalence.

The initial sample contains 400 documents from 388 frozen replay caches. Published junkgate calibration reported held-out recall 1/12 and natural false-drop 2/67. Its 20-document independent check and head/tail-only annotation leave substantial uncertainty. All of these documents are development-only here.

## 2026-11-15 — freeze the larger frame

Expand to 3,600 distinct cache rows, balancing cache representation with small strata capped at their population. Decode up to 2,048 tokens each at head, middle and tail, merging overlapping intervals. Record actual intervals, tokenizer hash, cache coordinates and content fingerprints. Repeated visible content shares a split. Seed400 content stays in development. Fresh groups are deterministically assigned 50% training, 20% calibration, 30% test by content SHA256. This split precedes annotation.

Prior work: Harrier backfill and merge exist for 291 normalized sources (Marin #8162). Current `hero_data.harrier` pins source-to-path maps. Replay cache-row joins still require provenance verification. Existing `dml/lce-junk-filter` at `70acdcfd6f08996e855e5ab7c1e805dcaa1f5558` provides a stronger rules baseline than the review's rough thresholds; its source is preserved as `baseline_rules.py`.

Private input text and labeling request/response bodies stay outside Git. Public outputs contain code, rubric, label aggregates, model parameters and evaluation statistics. No text publication or Exp3 application is authorized before its respective gate.

The annotation rubric protects useful human-written code, math, multilingual prose and legible OCR. Low-value categories include dumps, machine junk, markup, whole tables, damaged OCR, trajectories/benchmark bundles, repetitive templates and explicit generated Q/A artifacts. Polished style alone never establishes model authorship; abstain when the visible windows cannot support a document-level label. GLM uses Together GLM-5.3-Flash at temperature 0.7, concurrency 8, no SDK retries, and exclusive durable request-admission receipts. Uncertain transport requests are never replayed.

Replay cache inspection found only `input_ids`, `.success` and `shard_ledger.json`; its consolidated ledger names itself and records only the tokenizer/length threshold. The original store's partitions also expose only token arrays and ledgers. No normalized document IDs or cache-row→source-ID locator has been verified. Reusing an embedding by row ordinal would risk silently joining unrelated documents. Therefore compare a bounded recomputation of Harrier on these 3,600 sampled documents, using at most 1,024 Harrier tokens per visible window and mean pooling. This differs from stored Datakit 8,192-token head embeddings and must be recorded in the model contract. Bulk replay embedding recomputation is not planned.

Annotation preflight found two issues in the early annotation prefix: the validation allowlist omitted the two newly added rubric categories, and some model rationales wrongly cited universal BOS/EOS wrappers as evidence of synthetic origin. Original response bodies are retained and can be revalidated locally without API replay. Stop the initial process after its early prefix; requests admitted without final receipts remain excluded. Add an explicit wrapper-artifact instruction before continuing on unattempted IDs. All prior attempts remain in the same receipt directory and are skipped. Independent Astra labeling uses the original rubric, so report this instruction difference and adjudicate wrapper-based rationales explicitly.

The first one-H100 embedding job `/benfeuer/replay-quality-harrier-20261115-v1` failed on its first forward pass: PyTorch cuDNN SDPA could not build a valid execution plan. No embeddings were produced. A separate experiment uses PyTorch's documented `sdpa_kernel` context to permit flash, memory-efficient and math SDPA while excluding only the failing cuDNN backend; see https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html. This is Transformers Harrier inference, with no vLLM settings changed. Record throughput before interpreting the compute cost.

Model comparison is fixed before test access: gradient boosting with 160 trees, learning rate .05, depth 2/3 and minimum leaf size 10/25, plus standardized logistic regression with C .1/1/10. Compare structural, Harrier, and concatenated features on an inner 20% content-grouped validation slice of the training split. Select by mean recall at 2%/5% validation natural false-drop, breaking ties by average precision. Prefer structural-only if within 2 percentage points of the best score, to avoid mandatory embedding compute. Refit on all training groups; choose cutoffs on the separate calibration split; open test metrics once the selected model and cutoffs are frozen. Abstentions and uncertain API admissions are excluded with explicit counts.

## 2026-11-15 — sampled views complete

Extraction source `5da022accd` completed 3,600 distinct cache rows from 388 strata, with 2,038 train, 655 calibration and 907 test rows. Exact fingerprints and input hashes are summarized in `sampling.json`; full private receipts remain outside Git. Command: run `extract.py --seed-frame <seed400-frame.json> --tokenizer <Marin a5ca45f tokenizer.json> --output <private sample directory>`. Features use the stripped tokenizer wrappers and bounded views, never source names, cluster/quality labels or document IDs. `prepare_features.py` completed all 3,600 rows. Junkgate scoring uses its pinned BME 512-token views of the same complete documents.

Independent labeling source `4a54548212` selects 150 fresh long and 150 fresh short documents with RNG seed 52611300. Thirty independent `codex exec -m gpt-6-astra` batches see only anonymized views and rubric. GLM original source `75f27bdaa2` was interrupted during preflight; continuation source `22dee98c13` preserves every started request and excludes uncertain admissions. The failed first embedding job used `879a7ad519`; the alternative backend experiment `/benfeuer/replay-quality-harrier-20261115-v2` uses `0274b5ae67`, batch priority, one H100, a three-day timeout and CoreWeave east storage. No training or inference service was modified.

## 2026-11-15 — development comparison and qualified embedding artifact

The alternative SDPA embedding job succeeded in 6m 9.66s task duration. Its 3,600 x 1,024 finite vectors match the input and output receipt digests; embedding artifact SHA256 is `b780dfdffe47a5e9b42bbca9780d2be38c6d956b2cd328bcff528b0b4ace0a73`. This qualifies the sampled feature computation, not a full-corpus embedding deployment.

A provisional run of the predeclared model grid used 1,504 available training labels and 450 calibration labels; it never loaded test labels. Structural GBDT (depth 3, minimum leaf 10) attained inner-validation mean recall .4737 at the two false-drop targets; best fusion was .4934 and embedding-only .3750. The predeclared 2-point cost rule selected structural. These provisional scores are development evidence only: wait for completed labels and adjudication before final fitting, calibration and test evaluation. The earlier structural-only probe used fewer available labels and is superseded for model comparison.

Concrete implementation qualification found and fixed a cached-bundle problem: every new source run must verify pinned model/calibration bytes before output admission, even if the same process already loaded that pin. Text scoring now reuses Datakit's exact text preprocessor and BOS/EOS policy rather than duplicating encoding. Cache tests exercise real jagged arrays with interior encoded content and requested row reordering; normalized-source tests preserve shard/ID order, skip completed outputs, and reject changed model bytes before output.
