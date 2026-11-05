---
title: "Thoughtcrime: double-think prefixes from pi trajectories"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/25"
related_issues:
  - "https://github.com/marin-community/marin/issues/10400"
  - "https://github.com/marin-community/marin/pull/10894"
  - "https://github.com/marin-community/marin/pull/10907"
  - "https://github.com/marin-community/marin/pull/10908"
research_branch: "codex/25-thoughtcrime"
marin_base_revision: "79ecbccde83c7fd30f86a3a0d9e8c2da7fb9516d"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: concluded
---

# Thoughtcrime: double-think prefixes from pi trajectories

## Question and decision criteria

- Goal and why it matters: Marin models sometimes close a reasoning span twice (a stray `<|end_think|>`) or open a second span in agent turns (#10400). A fixed evaluation set would measure that rate per checkpoint, and could sit in the quick RL loop's evaluation panel.
- Baseline and alternatives: no dedicated measurement existed; the failure was visible only in pi transcripts of benchmark runs.
- Evidence needed to choose: a set of independent prompts drawn from real trajectories, a grader that scores reasoning markup alone, and a baseline score for the released model.
- Constraints: rows must be independent samples, not several prefixes of one trajectory; prompts must fit the quick loop's input window; the dataset republishes benchmark text, so licences need checking before release.

## Baseline

- Date: 2026-11-04.
- Model: `open-athena/Marin-Snowball-67B-A2B-v0.1` @ `adc0078a19a4383bc998fcdb43867b3360baf5f3` (byte-identical to Smartie step 14).
- Result: 0.796 on the final 328-row set (entry 2026-11-04 22:19 below).

## Source on this branch

This branch is the tip of the three stacked Marin drafts plus this directory and the archive-only removal of `.github/workflows`.

| Path | Purpose |
| --- | --- |
| `lib/verifyit/src/verifyit/modes/grade_reasoning_markup.py` | `reasoning_markup` grading mode: one closed span and no stray closer scores 1 (#10894). |
| `lib/taskcompendium/src/taskcompendium/convert/pi_trajectory.py` | Parses pi `pi.txt` event logs; builds one-turn tasks from a prefix row. |
| `experiments/post_training/task_curation/datasets/thoughtcrime.py` | Catalog entry `Marin:thoughtcrime`, pinned to the HF rows. |
| `experiments/post_training/thoughtcrime/mine_prefixes.py` | Miner over the evals root; writes `rows.parquet` and `report.json`. |
| `experiments/post_training/thoughtcrime/upload_prefixes.py` | Uploads rows, report and `dataset_card.md` to Hugging Face. |
| `experiments/post_training/thoughtcrime/eval_prefixes.py` | Single-node (H100x8) vLLM eval with pi's request settings. |
| `experiments/post_training/thoughtcrime/smoke.py` | One-update Qwen2.5-0.5B SkyRL smoke through the `taskcompendium` entrypoint. |
| `experiments/evaluation/serve/models/open-athena/Marin-Snowball-67B-A2B-v0.1.yaml` | v0.1 serve config, copied from the Smartie step-14 config. |
| `experiments/post_training/task_curation/inventory.py`, `recipes/snowball-100k.json` | `evaluation_sources` in selection recipes, adding thoughtcrime to the quick-loop panel (#10908). |

## Entries

### 2026-11-04 18:25 UTC - Draft rows with a matched control set

- Question: score v0.1 on prefixes cut before malformed replies ("hazard") against prefixes cut before well-formed replies in the same trajectories ("control").
- Command: `eval_prefixes.py` on the draft rows, Iris job `/dml/reasoning-span-eval-snowball-v0-1-20261104-r2`, one H100x8 on cw-rno2a.
- External inputs: 594 draft prefixes; output `s3://marin-us-east-02a/marin/rl-data/evals/reasoning-span-prefixes/marin-snowball-67b-a2b-v0-1/2026.11.04`.
- Result: 2,376 sampled replies, 0.800 well formed overall; 0.779 on hazard prefixes, 0.824 on control. Stray closes were 327 of 475 failures (69%).
- Interpretation: the control set added little, because both sets fail at similar rates. Several prefixes per trajectory also made rows correlated.
- Decision: drop the control set and the term "hazard"; select one first-failure prefix per trajectory and one row per task. Rename the dataset thoughtcrime.

### 2026-11-04 20:15 UTC - Mining the independent set

- Command: `uv run iris --config lib/iris/config/marin.yaml job run --enable-extra-resources --target-cluster cw-us-east-02a --cpu 16 --memory 64GB -- python -m experiments.post_training.thoughtcrime.mine_prefixes --version 2026.11.04 --run`, job `/dml/thoughtcrime-mine-20261104`, 51 minutes.
- Inputs: bixbench-pi, bfclparity-pi and tau3-pi runs of Marin models under `s3://marin-us-east-02a/marin/evals` started before `20261104-000000`.
- Output: `s3://marin-us-east-02a/marin/rl-data/inputs/thoughtcrime/2026.11.04/{rows.parquet,report.json}`.
- Result: 616 runs (191 without pi transcripts), 35,260 trajectories. First violations: none 24,364, stray_end 4,283, second_span 4,089, unclosed 2,524. 7,755 trajectories fail first with a double-think; 617 of those exceed the 12,288-token cap. One row per task leaves 328 rows: 205 BixBench (every task), 123 BFCL parity, 0 tau3; 212 stray closes, 116 second spans; 43 source models, the largest being `Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92` with 83 rows. Prompt-token quartiles 1,568 / 1,695 / 2,122 / 3,297; median reply position 4. Snowball-tokenizer prompt lengths match pi's within 1% for 221 of 328 rows, largest gap 869 tokens.
- Limitations: tau3 runs archive only `config.json` and `result.json`, so they contribute no rows. 991 trajectories whose pi transcripts carry structured `thinking` content are skipped. The miner is GIL-bound on JSON parsing; a process pool would shorten it.
- Upload: `open-athena/Marin-Thoughtcrime` @ `231751144c48b5553d5b921054b196e5e2d073f9`. All 328 rows convert, and the golden well-formed reply passes on every row.

### 2026-11-04 22:19 UTC - v0.1 on thoughtcrime

- Command: `uv run iris --config lib/iris/config/marin.yaml job run --enable-extra-resources --target-cluster cw-us-east-02a --gpu H100x8 --cpu 64 --memory 512GB --disk 600GB -e MARIN_TEMP_PREFIX s3://marin-us-east-02a -- python -m experiments.post_training.thoughtcrime.eval_prefixes --model-config experiments/evaluation/serve/models/open-athena/Marin-Snowball-67B-A2B-v0.1.yaml --output-prefix s3://marin-us-east-02a/marin/rl-data/evals/thoughtcrime/marin-snowball-67b-a2b-v0-1/2026.11.04`, job `/dml/thoughtcrime-eval-snowball-v0-1-20261104-e02a-local`, 6 minutes.
- Settings: 4 samples per prompt, 8,192 max tokens, tools advertised, `enable_thinking=True`, `skip_special_tokens=False`.
- Result: 0.796 over 1,312 replies (BixBench 0.807, BFCL 0.776; 0.796 on recorded stray closes, 0.795 on recorded second spans). 268 malformed replies: 140 stray closes, 99 second spans, 29 unclosed. Prompts by malformed replies out of 4: 0 for 167, 1 for 91, 2 for 40, 3 for 23, 4 for 7. 1,295 of 1,312 replies contain a reasoning marker, so the grader's gap (no markers scores 1) barely affects the score. By source model, prompts from `Marin-Snowball-67B-A2B-v0.1.1` transcripts score 0.733 (15 prompts).
- Failed attempts: `/dml/thoughtcrime-eval-snowball-v0-1-20261104` (cw-rno2a) and `/dml/thoughtcrime-eval-snowball-v0-1-20261104-e02a` (cw-us-east-02a) both failed with "Run:ai streamer read fault": the model cache resolved to the shared CoreWeave scratch bucket `s3://marin-us-east-08a/tmp/...`, and the cross-region stream hit the S3 client's throughput-minimum cutoff at 0/502 files. Setting `MARIN_TEMP_PREFIX=s3://marin-us-east-02a` serves from an in-region mirror.

### 2026-11-04 - Licence check and publication

- BixBench (Apache-2.0), BFCL (Apache-2.0), pi (MIT) and Harbor (Apache-2.0) permit redistribution with attribution.
- Gaps: `penfever/harbor-datasets` (corrected BixBench task files) and `laude-institute/harbor-datasets` declare no licence; BixBench's underlying capsule data carries no separate licence, and rows include excerpts in tool outputs (median 146 characters per BixBench row, largest 14,640).
- Added an Apache-2.0 dataset card with attribution (`experiments/post_training/thoughtcrime/dataset_card.md`, HF commit `0c516b6`) and made the dataset public.

### 2026-11-05 01:04 UTC - Thoughtcrime in a 3-step quick loop

- Question: does thoughtcrime work as a quick-loop evaluation source, and what does the Datakit SFT initializer score?
- Source: this branch after merging Marin main `ec047c8` (TaskSpec schema 0.30; evaluation sources curate with `--mode quick`).
- Selection: `build_panel_selection.py` appends the 328 QUICK-converted thoughtcrime tasks to the 100-task panel of `snowball-100k/2026.11.04`, published at `s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k-thoughtcrime/2026.11.04.1`. The `evaluation_sources` path needs an admitted FULL corpus, which main now refuses for evaluation sources.
- Command: `uv run python -m experiments.post_training.inventory.train_quick_loop --selection-uri <selection> --version 2026.11.04.73 --steps 3 --run`, coordinator `/dml/users-dml-checkpoints-snowball-inventory-2026.11.04.73-coordinator-4dc4ca30`, 40 H100 on cw-us-east-02a, 36 minutes for the training step.
- Failed attempts: `.71` used base selection `2026.10.23.2`, whose schema-0.28 tasks main no longer loads (main's launcher default still points there). `.72` scored thoughtcrime 0.0 because every request returned HTTP 400: the quick loop's vLLM engines had no tool parser, and thoughtcrime is the only panel source that sends `tools`. `.73` sets `enable_auto_tool_choice=True, tool_call_parser="hermes"` in `train_quick_loop.py`.
- Result (one sample per prompt, all 328 graded both times): thoughtcrime 0.820 before training (270 well formed, 57 malformed, 1 length) and 0.786 after 3 updates (258 well formed, 69 malformed, 1 length-penalized). Whole panel 0.561 before and 0.561 after.
- Evidence: `s3://marin-us-east-02a/marin/users/dml/checkpoints/snowball-inventory/2026.11.04.73/exports/dumped_evals/global_step_{0,3}_evals/open-athena_Marin-Thoughtcrime.jsonl`.
- Interpretation: the 0.034 drop is about 1.4 standard errors for 328 single samples; three updates do not establish a trend.
- Decision: any panel that includes thoughtcrime needs a vLLM tool parser on its rollout engines.

## Final decision

- Selected approach: the runnable catalog entry goes to Marin main: the `reasoning_markup` grading mode (#10894) and the pi-trajectory converter with the `Marin:thoughtcrime` catalog entry (#10907, trimmed).
- Kept here only: the miner, uploader, dataset card, single-node eval, training smoke, v0.1 serve config, and the selection `evaluation_sources` wiring (#10908, closed). The dataset is fixed at its pinned revision, so these are needed only to re-mine or re-score.
- Reproduction: check out this branch and run the commands in the entries above; Iris submissions source `.env`, then `unset MARIN_PREFIX`.
- Remaining uncertainty: tau3 coverage; replies without markers score 1; licence of the corrected BixBench task files; run-to-run variance of the score (one run of 4 samples per prompt so far).
