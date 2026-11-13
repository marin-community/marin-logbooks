---
title: "Filtering junk from long-context replay in cold-start SFT"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/40"
related_issues:
  - "https://github.com/marin-community/marin/issues/11188"
  - "https://github.com/marin-community/marin/issues/9519"
  - "https://github.com/marin-community/marin/issues/10799"
  - "https://github.com/marin-community/marin/issues/8701"
research_branch: "codex/40-lce-replay-junk-filter"
marin_base_revision: "cae189b2e6b3d08f0b83fbf95e3100a8c8806998"
archive_template_revision: "261c88369ab6bfe81d297a2b0f78710e139c99c3"
status: active
---

# Filtering junk from long-context replay in cold-start SFT

## Question and decision criteria

- Goal and why it matters: decide whether the cold-start SFT recipe should build its long replay bucket from filtered documents. About half of the long replay tokens are machine output or data dumps, and replay is the part of the mix meant to protect long-context ability.
- Baseline and alternatives: the current replay selection (control) against the same selection with flagged documents replaced by kept documents from the same cluster-and-quality cell.
- Evidence needed to choose: needle recovery at 262K on the four-condition MRCR probe (primary), held-out replay loss split by filter verdict, and BFCL-Parity and DS-1000 as guards against losing structured-data skill. The decision table is in the task issue.
- Constraints and unresolved questions: two arms only, so run-to-run noise on the primary metric is unmeasured. The probe has to be ported to the Snowball model on H100s before anything can be scored.

## Baseline

- Date: not yet run.
- Published source revision:
- Configuration, command, and external inputs:
- Environment and hardware when relevant:
- Result and evidence:
- Limitations:

## Entries

### 2026-11-12 23:40 UTC - Filter development on replay samples

- Question or hypothesis and reason for trying it: can a text-only rule set separate machine output and data dumps from documents worth a long-context training position? The quality tiers do not: the review in marin#11188 found low-value token shares of 41-60% in every tier.
- Published source revision: `15d37a3cec` (`lib/marin/src/marin/datakit/junk_filter.py`, `tests/datakit/test_junk_filter.py`).
- Command or material configuration: `scripts/replay_window_sample.py` draws documents from each of the 391 replay caches of the 10.05 mix, share-weighted with a floor of 8 per cache, and decodes eight 2,048-token windows per long document (head, tail, six stratified-random interior positions). `scripts/score2.py` applies the filter.
- External inputs and artifact references: replay caches of Datakit store `june-67b-a2b-length64k/2026.08.24`; tokenizer `grug_sft/tokenizer/2026.09.18`. Samples are in temporary object storage with a 30-day lifetime and are not archived here because they contain raw corpus text.
- Environment and hardware when relevant: one 8-CPU Iris job in the caches' region per sample, about 5.5 minutes each.
- Result, including failure or restart: six rounds of rules. On the 1,398-document review sample of marin#11188 the frozen filter flags 58 of 59 documents the review called machine junk and 326 of 374 it called data dumps, against 17 of 784 it called keep; most of those 17 were minified or generated files the review had mislabelled. On a 12,202-document development sample it flags 33% of long documents holding 53% of tokens. Interior windows roughly doubled the machine-junk token share against the review's head-and-tail estimate (22% against 9%). Only about 4% of short replay documents are flagged.
- Evidence: rule order and thresholds are in the module; `scripts/feat.py` and `scripts/quant.py` produced the per-chunk feature quantiles used to place them.
- Interpretation and limitations: the rules see surface shape only. They cannot tell generated code from hand-written code, or a CSV of sentences from a CSV of sensor readings.
- Decision and next action: freeze and check on a held-out slice.

### 2026-11-13 00:10 UTC - Held-out check, two fixes, second held-out check

- Question or hypothesis and reason for trying it: how much of what the frozen filter flags is really junk or data, and how much does it miss, on documents it was never tuned on?
- Published source revision: `15d37a3cec` for the first check; `70acdcfd6f` for the second.
- Command or material configuration: `scripts/val_prepare.py` scores a fresh-seed slice, drops documents already seen, and writes a shuffled sheet of excerpts with no verdicts. A reader labels each document machine junk, data dump or keep; `scripts/val_score.py` compares. First check: seed 3, 11,946 documents, 160 labels (40 junk verdicts, 40 data, 80 keep). Second check: seed 4, 11,776 documents, 100 labels (30, 30, 40).
- External inputs and artifact references: `validation/validation_labels.txt` and `validation/validation_key.jsonl`; `validation/validation2_labels.txt` and `validation/validation2_key.jsonl`. The sheets are not archived because they contain raw corpus text.
- Result, including failure or restart:

  | | First check | Second check |
  | --- | ---: | ---: |
  | Flagged documents labelled junk or data | 92.6% | 89.0% |
  | Unflagged documents labelled junk or data | 13.8% (11 of 80) | 7.5% (3 of 40) |
  | Estimated share of junk-or-data documents flagged | 76.4% | 85.2% |
  | Estimated share of machine-junk documents flagged | 89.6% | 100% (none among 40 unflagged) |
  | Flagged share of tokens | 52.0% | 52.8% |
  | Flagged share of training positions | 42.6% | 43.8% |

  The first check found one clear rule error (a scripture translation stored as verse number, tab, verse was called a table) and one clear miss (an SVG whose base64 payload is wrapped at 76 characters). Both were fixed in `70acdcfd6f` before the second check. The second check found a new false-positive pattern: two court judgments stored as JSON were called minified code because their text sits in long strings with escaped newlines. It is not fixed.
- Evidence: the label and key files above.
- Interpretation and limitations: the two checks agree within sampling error. The labels are one reader's judgement from four 150-character excerpts per document, and the same reader wrote the rules. Tier labels are loose: only 60% of machine-junk verdicts matched the junk label in both checks; most of the rest were data. Most misses are generated code, table-only documents and lists.
- Decision and next action: good enough to test the treatment at the document level. Do not tune further on these slices.

### 2026-11-13 00:50 UTC - Step 0 scoping: the probe cannot score the cold-start checkpoints yet

- Question or hypothesis and reason for trying it: the plan's first step was to score existing cold-start checkpoints with the needle-recovery probe, to learn where in the schedule recovery is lost. Is that runnable?
- Published source revision: none; read-only inspection.
- Command or material configuration: read `codex/paired-mrcr-base-eval` at `3f77157789`, `hero-mrcr-eval` at `1b113f404c`, and the export configs on `codex/sft-cold-start-reference-20261030`; listed the reproduction's run directories in object storage.
- Result, including failure or restart: not runnable as is.
  - The original probe is TPU-only and restores June-format checkpoints into `experiments.grug.moe.model.Transformer`.
  - The GPU port restores into the `moe_hero_ep` model but hardcodes the hero model config, GB200 resources and rack-based batch sizing.
  - The marin#10799 native checkpoints for steps 222, 471, 1024 and 2048, and the storage-side exports, are gone; the run directories hold only `best` and `provenance`. The checkpoints survive as Hugging Face exports. The reproduction's replay caches are still present.
- Interpretation and limitations: step 0 needs a Snowball variant of the GPU launcher, a re-import of the exports, and H100 time. Unknown: whether the exports restore into the current model code, how many H100 nodes a 262K probe needs, and the wall time.
- Decision and next action: port the probe to Snowball on H100s (decided 2026-11-12). Start with two arms; defer the seed replicate.

### 2026-11-13 01:10 UTC - Probe branch merged; the cold-start training code is not on main

- Question or hypothesis and reason for trying it: start the Snowball port of the GPU needle-recovery probe, and find the training entry point the two arms would use.
- Published source revision: this commit (merge of `hero-mrcr-eval` at `1b113f404c` into the research branch).
- Command or material configuration: `git merge origin/hero-mrcr-eval`. Conflicts in `moe_hero_ep/model.py`, its README and `levanter/data/text/datasets.py` were resolved in favour of main, which has since made the log-scaling slope and exponent required together with the floor and moved `TokenSeqDict` to `formats.py`. `valid_tokens` was added to that type. Two probe tests were adapted: one passes the slope and exponent explicitly, one carries a local copy of a reshard helper that main removed. `test_qk_mult.py` keeps main's version.
- Result, including failure or restart: the probe's tests, the MRCR dataset tests and the filter tests pass on the merged tree (47 tests, CPU). The probe core (`evaluate_checkpoint`, `CheckpointEvalConfig`, `evaluate_mrcr`) takes a model config and resources as inputs; only `build_checkpoint_eval` is tied to the hero model, GB200 nodes and rack-based batch sizing. A Snowball launcher is therefore a new builder, not a change to the core.
- Interpretation and limitations: the larger finding is about training, not the probe. The cold-start SFT pipeline used in marin#10799 (`experiments/grug/snowball_sft/`: launch, mixture, model, HF import, export) is not on Marin main. It survives only as `.source` snapshots on `codex/sft-cold-start-reference-20261030`, taken from a training worktree at base `8577d258` plus a patch, and that branch's README says it is not a tested port to current main. The archive PR was closed. Main's supported SFT paths are the hero SFT launcher (hero model on GB200) and the standard chat trainer for Snowball at 32K context without packing, neither of which is the 262K replay-mixing recipe.
- Decision and next action: the two arms need a training vehicle. Options are to restore the archived training worktree at its pinned base and apply the replay change there, to port `snowball_sft` to main first, or to run the comparison on the hero SFT launcher. This needs a decision before more probe work, because the probe launcher must match whichever model code writes the checkpoints.

### 2026-11-13 03:25 UTC - Smoke run: the probe scores the Snowball base at 262K on one H100 node

- Question or hypothesis and reason for trying it: decided 2026-11-13 to train the arms from the archived cold-start worktree (option A) and to start with a smoke run. Can the current probe code restore a Snowball checkpoint and score 262,144-token MRCR rows on H100s?
- Published source revision: `34fe7eb260970283624273974acfac07dc2606e1` (`experiments/grug/moe_hero_ep/snowball_recipe.py`, `snowball_hf_import.py`, `launch_snowball_mrcr_eval.py`, `test_snowball_hf_import.py`).
- Command or material configuration:
  - A read-through of the archived `snowball_sft/model.py` against the current `moe_hero_ep/model.py` found no added, removed or renamed parameter fields, so the probe runs from current code with a Snowball config. Two hero defaults do not fit H100s and are overridden: the dropless MoE backend (`sonic` instead of the Blackwell-only default) and the mixed-precision policy (bfloat16 parameters, as the reproduction stored them).
  - Import: `python -m experiments.grug.moe_hero_ep.snowball_hf_import --hf-id open-athena/snowball-67b-a2b-base-262k-qk175-skew8 --hf-revision 058ecaf27b9e4f37219df221a51e7d490d58ec3d --destination <tmp>/snowball-base-native-20261113`, one process on one 8xH100 node of `cw-rno2a` (64 CPU, 1500 GB RAM), interactive priority.
  - Probe: `python -m experiments.grug.moe_hero_ep.launch_snowball_mrcr_eval --run-id snowball-base-mrcr-smoke2 --checkpoint <tmp>/snowball-base-native-20261113/step-0 --mrcr-max-batches 4 --version 2026.11.13 --run`, a CPU coordinator on `cw-rno2a` dispatching one 8xH100 node with eight processes and eight context shards.
- External inputs and artifact references: MRCR dataset revision `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`, two-shot prompt, cap 262,144, tokenized with the Marin tokenizer. The imported checkpoint is in temporary object storage with a 30-day lifetime. Results: `eval/mrcr-snowball/snowball-base-mrcr-smoke2/2026.11.13/` under the cluster's Marin prefix (`mrcr_metrics.json`, `mrcr_example_losses.jsonl`).
- Environment and hardware when relevant: Iris jobs `/dml/lce40-snowball-base-import` (21.6 minutes) and `/dml/lce40-snowball-base-mrcr-smoke2` (8.4 minutes end to end).
- Result, including failure or restart:
  - Import: all 502 tensors equal the export exactly after a restore through `load_checkpoint_params`.
  - A first probe run with `--mrcr-max-batches 1` failed after four attempts. One node holds one row per batch, so one batch scored one of a source's four condition rows, and `pair_condition_losses` rejects an incomplete set. The log store timed out on every query, so this cause is inferred from the code, from seven of eight processes exiting cleanly, and from the four-batch rerun succeeding.
  - The four-batch run scored one complete source (8 needles, evidence 210,531 tokens back, 241 scored tokens):

    | Condition | Loss sum | Loss per token |
    | --- | ---: | ---: |
    | Query only | 400.88 | 1.663 |
    | Distractors only | 426.52 | 1.770 |
    | Needles only | 22.39 | 0.093 |
    | Full context | 66.99 | 0.278 |

    Isolated needle gain 1.571 nats per token, distracted gain 1.492, needle recovery 0.950.
- Interpretation and limitations: the port works end to end on H100s. One source says nothing about the aggregate, but the numbers are in the range marin#9519 reported for a 262K parent (isolated gain 1.39, recovery 0.91 over 299 sources), which is what a correctly loaded long-context model should show. Not tested: a checkpoint written by the archived trainer (none exists), agreement between this GPU path and the TPU path on the same checkpoint, and the per-row time, which the unavailable logs would have given. The imported checkpoint is not in the layout the archived trainer initializes from; the arms need the archived importer for that.
- Decision and next action: step 0 proper is now runnable: score the base on all 299 sources, then import and score the four marin#10799 exports (steps 222, 471, 1024, 2048).

## Final decision

- Selected approach and evidence:
- Rejected alternatives and why:
- Marin contribution PRs:
- Reproduction instructions:
- Remaining uncertainty:
