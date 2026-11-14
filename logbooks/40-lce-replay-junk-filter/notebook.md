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

### 2026-11-13 04:40 UTC - Filtered long replay built; both arms launched

- Question or hypothesis and reason for trying it: build the treatment and start the two arms. Directed 2026-11-13 to execute the experiment.
- Published source revision: research branch `bbb61d8c02` (`experiments/sft_cold_start/filtered_replay.py`, `logbooks/40-lce-replay-junk-filter/scripts/make_arm_configs.py`, `make_filtered_catalog.py`, `treatment_dose.py`). Training tree: branch `codex/40-lce-replay-junk-filter-training` at `00c033645b`, which is Marin `8577d258` plus the archived training patch and sources, plus three configs under `experiments/grug/snowball_sft/configs/`: `lce40-hf-import-20261113`, `lce40-arm-control-20261113`, `lce40-arm-filtered-20261113`.
- Command or material configuration:
  - Filtered caches: `python -m experiments.sft_cold_start.filtered_replay --cells logbooks/40-lce-replay-junk-filter/replay-cells-20261024a.yaml.gz --output <tmp>/replay-filtered-20261113 --tokenizer <grug_sft tokenizer 2026.09.18> --workers 12`, one 16-CPU job, 39 minutes. For each of the 189 training cells with a long budget it walks the cell's source shards in the control's order, judges each long document from eight 2,048-token windows, and keeps unflagged documents until the cell's long-token budget is crossed.
  - Arm configs: the production config `snowball-sft-production-128h100-20261024b` with run names, output paths, W&B project, retry budget, `initialize_from` (a fresh run of the archived importer), `stop_after_steps: 238` and `keep: every 119`. The filtered arm additionally points 1,336 long cache entries (167 cells, 8 copies each) and the cache catalog at the filtered caches. A diff of the two arm configs shows only those lines and the run names.
  - Launch: `python -m experiments.grug.snowball_sft.launch --config_path experiments/grug/snowball_sft/configs/<run>/config.yaml` from a CPU coordinator on `cw-rno2a`, interactive priority; each arm trains on 16 nodes of 8 H100s.
- External inputs and artifact references: Iris jobs `/dml/lce40-legacy-import` (10 minutes), `/dml/lce40-filtered-replay-full`, `/dml/lce40-arm-control-20261113`, `/dml/lce40-arm-filtered-20261113`. Caches, catalog, per-cell `verdicts.jsonl` and `summary.jsonl` are under `<tmp>/replay-filtered-20261113`; arm outputs under `<tmp>/arms/<run>`. All in temporary object storage with a 30-day lifetime.
- Result, including failure or restart:
  - The archived importer and the archived coordinator both ran from the archived tree on the current cluster.
  - Builder check: for a cell with nothing flagged (`c39q4-train`) the filtered cache holds the same 53 documents and 8,330,748 tokens as the control cache, in the same format.
  - All 189 cells met their budgets; no cell ran out of unflagged long documents. The builder judged 35,102 long documents and flagged 30.0% of them. The selection changed in 101 cells.
  - Dose. The control's long selection is 22,758 documents and 5.675B tokens. The filter flags 7.2% of those documents holding 30.1% of the tokens: structured data 23.7%, encoded blobs 2.3%, minified code 2.3%, repetition 1.0%, everything else under 1%. By quality tier the flagged token share is 58% (q0), 41% (q1), 32% (q2), 33% (q3) and 15% (q4).
  - Step 0: the base score was restarted once when its coordinator was preempted. The first import of the step-222 export failed because the SFT exports name the shared expert `shared_experts.0` where the base export says `shared_expert`; the importer now accepts both (`bbb61d8c02`) and the chain was restarted.
- Interpretation and limitations:
  - The dose is smaller than planned. The reproduction's replay store is cleaner than the 10.05 mix sampled earlier (30% of long tokens flagged against 52%), because it draws more from the higher quality tiers. The treatment changes roughly 5% of training positions, not 8%.
  - The filtered long caches hold 9% fewer tokens in total than the control's (5.158B against 5.672B over the swapped cells), because the control overshoots its budgets with very large flagged documents. Cell weights are fixed, so within a cell the long share of draws falls slightly; the arms are not exactly matched on long positions.
  - 22 of the 189 cells' long caches are not referenced by the production mixture and so are unchanged in both arms.
  - The two arms run at the same time on separate node sets, not one after the other.
- Decision and next action: when each arm reaches steps 119 and 238, score its checkpoints with the probe; then the guard evals.

### 2026-11-13 15:30 UTC - Results: step 0 and the two arms

- Question or hypothesis and reason for trying it: where in cold-start SFT is needle recovery lost, and does filtered long replay change it?
- Published source revision: research branch `260cf65632` (probe launcher, importer, `scripts/mrcr_summary.py`, `scripts/paired_mrcr.py`, `scripts/arm_losses.py`); training branch `00c033645b`.
- Command or material configuration: every checkpoint scored with `python -m experiments.grug.moe_hero_ep.launch_snowball_mrcr_eval --run-id <run> --checkpoint <path> --nodes 2 --version 2026.11.13 --run` on two 8xH100 nodes of `cw-rno2a`, all 299 sources, about 30 to 36 minutes each. The marin#10799 exports were first imported with `snowball_hf_import`. Arm comparison: `scripts/paired_mrcr.py <control run> <filtered run>`, 10,000 paired resamples over sources, seed 0.
- External inputs and artifact references: results under `eval/mrcr-snowball/<run>/2026.11.13/` in the cluster's Marin prefix, runs `snowball-{base,s222,s471,s1024,s2048}-mrcr-full` and `arm-{control,filtered}-s{119,238}-mrcr-full`. Arm checkpoints at `<tmp>/arms/lce40-arm-{control,filtered}-20261113/checkpoints/step-{119,238}`, 30-day lifetime. W&B project `marin-community/lce40-replay-filter`.
- Environment and hardware when relevant: each arm trained on 16 nodes of 8 H100s for 10 hours 10 minutes (238 updates, about 1,300 H100-hours each). Both training jobs started within a second of each other at 04:21 UTC after the control was restarted once at launch; neither was interrupted afterwards.
- Result, including failure or restart:
  - Step 0, needle recovery on 299 sources (token-weighted, 95% bootstrap interval):

    | Checkpoint | Recovery | Interval | Context gain, nats/token |
    | --- | ---: | --- | ---: |
    | Base (import of the pinned export) | 0.856 | 0.832 to 0.878 | 1.025 |
    | marin#10799 step 222 | 0.928 | 0.916 to 0.939 | 1.133 |
    | marin#10799 step 471 | 0.949 | 0.940 to 0.957 | 1.167 |
    | marin#10799 step 1024 | 0.943 | 0.934 to 0.951 | 1.170 |
    | marin#10799 step 2048 | 0.923 | 0.909 to 0.936 | 1.122 |

  - Arms:

    | Step | Metric | Control | Filtered | Filtered minus control (paired 95% interval) | Sources where filtered is higher |
    | --- | --- | ---: | ---: | --- | ---: |
    | 119 | Needle recovery | 0.880 | 0.909 | +0.028 (+0.023 to +0.034) | 241 of 299 |
    | 119 | Context gain, nats/token | 1.070 | 1.111 | +0.041 (+0.035 to +0.048) | |
    | 238 | Needle recovery | 0.935 | 0.948 | +0.013 (+0.008 to +0.018) | 177 of 299 |
    | 238 | Context gain, nats/token | 1.138 | 1.166 | +0.028 (+0.023 to +0.034) | |

  - Held-out losses logged during training are the same in both arms at the end (step 237, control/filtered): all validation 0.5239/0.5238, SFT sources 0.4059/0.4055, replay 0.8460/0.8464. On replay the filtered arm was about 0.003 lower at steps 49 to 149 and level from step 199.
  - A rough reference for run-to-run variation: this control at step 238 against the marin#10799 run at step 222, two runs of nearly the same recipe 16 updates apart, differ by +0.007 in recovery (+0.003 to +0.011) and +0.005 in context gain (-0.000 to +0.010).
  - Scoring the arm checkpoints confirmed that checkpoints written by the archived trainer restore into the current probe code.
  - Step 0 needed two restarts: one export import failed on a tensor-name difference (fixed), and one was preempted mid-write so its retry refused the partial destination (rerun to a fresh destination).
- Interpretation and limitations:
  - In this reproduction cold-start SFT raises needle recovery, with a peak near step 471 and a slow decline to step 2048 that stays above the base. The loss marin#9519 measured after a different 2,000-step SFT does not appear here, so the experiment could not test whether filtering prevents a loss. It tested whether filtering changes how fast recovery rises.
  - The filtered arm is ahead at both checkpoints by more than the source-sampling interval, and the gap halves between step 119 and step 238. That pattern fits a head start that the control closes, not a lasting difference; two checkpoints cannot say which.
  - There is one run per arm. The intervals cover which MRCR sources were drawn, not training noise. The only reference for the latter is the cross-run comparison above, which is confounded by 16 updates, a different import and different hardware batches; against it the recovery difference at step 238 is about twice as large and the context-gain difference about five times.
  - The filtered arm also saw slightly fewer long-document draws (9% fewer long tokens in its caches), which if anything works against it on a long-context measure.
  - Not done: the guard evals (BFCL-Parity, DS-1000), which need exported and served checkpoints; the held-out replay loss split by filter verdict; a seed replicate. The replay validation set is unfiltered, so the logged replay loss is not the split the plan called for.
- Decision and next action: see the final decision.

### 2026-11-14 00:45 UTC - Guard evals on the step-238 checkpoints

- Hypothesis or decision under test: filtering long replay does not lower tool-calling (BFCL-Parity) or code (DS-1000) scores at step 238.
- Published code revision: research branch `9a6c1ac027` for the eval configs; training branch `bd8fe6ed49` for `export_lce40_step238.py`. Pairing script: `logbooks/40-lce-replay-junk-filter/scripts/paired_guards.py`; export finalization: `scripts/finalize_export.py`.
- Run identities: exports `/dml/lce40-export4-{control,filtered}-step238`; smoke `/dml/eval-lce40-guards-control-step238-smoke2-246a1397`; full runs `/dml/eval-lce40-guards-control-step238-full1-0a14d372`, `/dml/eval-lce40-guards-filtered-step238-full1-c00d57f3`, and the filtered DS-1000 resume `/dml/eval-lce40-guards-filtered-step238-full1-44f98db4`.
- Inputs and outputs: each arm's step-238 checkpoint exported to `s3://marin-us-east-02a/tmp/ttl=30d/lce-junk-filter/exports/lce40-arm-<arm>-20261113-step238/hf` (39 shards, plus the 2026.09.18 tokenizer files and a generation config) and served from there without a Hugging Face repository. Records under `s3://marin-us-east-02a/tmp/ttl=30d/lce-junk-filter/evals/`.
- Environment and hardware when relevant: one 8xH100 node per run on `cw-us-east-02a`; BFCL-Parity under Terminus-2, DS-1000 with the main-branch local config, Daytona sandboxes.
- Result, including failure or restart:

    | Benchmark | Control | Filtered | Paired tasks | Solved by control only | Solved by filtered only | Exact McNemar p |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | BFCL-Parity | 94/122 (0.770) | 85/123 (0.691) | 122 | 17 | 7 | 0.064 |
    | DS-1000 | 47/200 (0.235) | 41/199 (0.206) | 199 | 20 | 14 | 0.39 |

  - One trial per benchmark was lost to a `DaytonaConnectionError` (control BFCL-Parity, filtered DS-1000); the launcher marks those two records `infra_failed` with no metrics although the other trials are stored. The counts above come from the stored per-task results.
  - The first filtered DS-1000 run died at 195 of 200 trials with `ValueError: Incomplete archived attempt ... for trial 167` from Harbor's result store; relaunching under the same run id resumed and finished.
  - The first smoke run sat at the Kueue scheduling gate on `cw-rno2a` for 24 minutes and was cancelled; all guard runs used `cw-us-east-02a`.
  - Every trial logs a non-fatal `TypeError` (`S3Path` passed to `Path`) when Harbor uploads agent logs back to the sandbox.
- Interpretation and limitations:
  - Both guards are lower in the filtered arm. Neither difference clears a 5% paired test, but BFCL-Parity is close (17 tasks lost against 7 gained) and both point the same way, so the guards do not support "no harm".
  - One training run per arm and one eval run per checkpoint: agent sampling noise and training noise are both unmeasured. The seed replicate, which would bound training noise, is still queued for H100s at step 23.
  - Step 238 is 5% of the schedule; these are early-training scores.
- Decision and next action: see the final decision.

## Final decision

- Selected approach and evidence: do not change the recipe on this evidence. The primary metric favours filtering (needle recovery +0.028 at step 119 and +0.013 at step 238, beyond the source-sampling interval, gap shrinking) and held-out SFT and replay loss are unchanged, but both guards are lower in the filtered arm (BFCL-Parity 0.770 to 0.691, paired p = 0.064; DS-1000 0.235 to 0.206, p = 0.39). The guards were set before launch to catch a loss of structured-data skill; a small, shrinking probe gain against a near-significant tool-calling drop is not enough to adopt. With one run per arm, neither effect is separated from training noise.
- Rejected alternatives and why: scoring on the TPU probe (cannot load these checkpoints); backporting the probe into the archived tree (unnecessary, the parameter layout is unchanged); porting the cold-start trainer to main first (deferred, larger than the experiment).
- Marin contribution PRs: none yet. Candidates: the filter (`dml/lce-junk-filter`), the Snowball probe launcher and importer, and the filtered replay builder.
- Reproduction instructions: the entries above give the published revisions, commands and inputs for the importer, the probe, the filtered cache build, the arm configs and the comparison scripts.
- Remaining uncertainty: training noise on the primary metric and on the guards (the second control seed is queued, not finished); whether either difference persists past step 238; and whether a recipe that does lose recovery under SFT, as in marin#9519, would respond differently.
