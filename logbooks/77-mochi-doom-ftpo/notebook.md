---
title: Mochi doom-loop FTPO
source_repository: marin-community/marin
task_issue: https://github.com/marin-community/marin-logbooks/issues/77
related_issues:
  - https://github.com/marin-community/marin/issues/12240
  - https://github.com/marin-community/marin/issues/9691
research_branch: codex/77-mochi-doom-ftpo
marin_base_revision: f9fe837d6398cf27355b1f34ccba997fafd67ef1
archive_template_revision: fc1cb30da0967f6ab78ffd47444ed9144f50cba9
status: active
---

# Mochi doom-loop FTPO

## Question and decision criteria

Can the original online FTPO objective cut exact repetitive tails in Mochi SFT step 400407 by at least 50% in every measured target domain? Keep native correctness within max(3, 5% of N) lost passes per domain and 2% pooled. Compare each candidate's loop upper bound with the baseline's confirmed loops on identical prompt hashes. Extend sparse 32K domains to 256 prompts before demoting those with fewer than 10 confirmed loops. Non-target domains remain regression checks with at most three additional upper-bound loops.

Use up to 128 GB200 on cw-us-east-08a at production priority. Submit stages when ready and let Kueue schedule them. Serving remains TP1/BF16 with CUDA graphs. No training begins before baseline sealing. Executor start: 2026-11-24 17:46 EST (22:46 UTC); start no training after November 26 13:46 EST and seal by November 26 17:46 EST.

## Baseline

Pending the fixed-cohort greedy generation audit. Temperature 0.7 is reported separately. The detector is SkyRL's `tail_loop`, requiring eight exact suffix copies, with full-window search and period caps of 2,048 at 16K and 4,096 at 32K. Missing or damaged responses enter the upper bound and do not count as correctness passes.

## Entries

### 2026-11-24 — Recover original inputs and resolve source drift

The original DAPO split survives with its pinned revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`, 16,879 training rows and 512 validation rows. These 512 recovered prompts are the input cohort; the final eligible count remains pending the 2,048-token filter. Hash eligible token IDs under the pinned Mochi tokenizer. The original split used canonical message-JSON hashes; the new token hashes identify the Mochi-rendered prompts.

All 193 safetensors shard SHA-256 values agree between the 08a copy, its verified-copy receipt, and the 02a source export. The 08a export-manifest digest is `1a7886b05df25338a12642b25f25525c0eb6c5551384934a2f089d7ec5b4a88d`. The stale `grug/…-hf/2026.11.10` adoption path is absent; the surviving source is the export directory recorded in the verified-copy receipt. No weights were downloaded or converted. See the [model-identity receipt](model-identity.json).

The finished Snowball corpus contains schema-0.28 tasks, which the current runtime rejects. The November 19 Snowball selection records admitted copies of these same source populations at schema 0.30, supported by the current upgrader. Reuse those copies, retain the original corpus identity, and convert DAPO with the current source converter. Use the admitted TaskTrove instruction-following structured-output population as the design's programmatic IF fallback. Preserve its native checker and report the exact source.

The export has no tokenizer files. Bind `marin-community/marin-tokenizer-tools` at `3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09`, the pinned SFT tokenizer. Preserve assistant-prefill whitespace with the existing SkyRL renderer. Exclude EvalLite, Micro210, all evaluation prompts and all reserves from training before publication.

The design's 1-MiB NCCL buffer predates the current 4-MiB correctness default for NCCL 2.30.7. Keep that default; retain `NCCL_MAX_CTAS=4`, full recomputation, and CPU activation offload. Measure the hardest 32K case in S1.

The task explicitly forbids AGENTS and skill edits, so the archive does not add the workflow-required directory-local instruction copy. Marin's existing root instructions remain in force. Remove inherited workflows only on the archive branch.

Next: validate the CPU publication path and baseline presets locally, publish source, build cohorts on an 08a CPU worker, then submit S0. The full experiment is still pending; source recovery and passing detector tests do not establish a baseline loop rate.

## Final decision

Pending measured baseline and candidate results.

### 2026-11-24 18:05 EST — Submit the cohort build

Published source: `2678348b6b` on `benfeuer/mochi-doom-ftpo`. Submitted `/benfeuer/mochi-doom-cohorts-20261124a` to cw-us-east-08a with 16 CPUs, 96 GB memory, 64 GB disk, production priority, and a 72-hour runtime ceiling. The executor's 48-hour deadline still applies. Command: `uv run iris --cluster cw-us-east-08a job run --job-name mochi-doom-cohorts-20261124a --no-wait --enable-extra-resources --cpu 16 --memory 96GB --disk 64GB --priority production --timeout 259200 -e PYTHONPATH vendor/skyrl/skyrl-train:vendor/skyrl/skyrl-gym:vendor/skyrl -- python -m experiments.post_training.mochi_doom_ftpo.cohorts`. HF and storage credentials were forwarded through Iris secret delivery; no credentials are in source.

The 10 detector tests and four cohort/publication tests pass locally. All-file pre-commit and standalone Pyrefly pass. The affected safe suite reports 1,416 passes, five skips, and four failures in unchanged task-curation tests: two macOS Bash array failures, one code-runner reward failure, and a Linux-only bubblewrap backend call. Baseline replication of those failures is pending. No GPU job has been submitted, and no loop-incidence result is available.
