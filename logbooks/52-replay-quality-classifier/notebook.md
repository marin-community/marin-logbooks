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
