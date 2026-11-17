---
title: "Does filtering flawed documents from the SFT mix remove the behaviors they teach?"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/60"
related_issues:
  - "https://github.com/marin-community/marin/issues/11188"
research_branch: "codex/60-sft-mix-filtering"
marin_base_revision: "7e11f2ee6047a28ba7334f0d9db539d5ce535a5a"
archive_template_revision: "976dc0d1ff1acd04c6f1f8e1e3ff651c55eb92e4"
status: complete
---

# Does filtering flawed documents from the SFT mix remove the behaviors they teach?

This branch is the experiment branch `dml/sft-patch-11188-utf` archived as is. It carries the long-loop
prototype branch merged with Marin `main` at `7e11f2ee60`, plus the experiment code under
`experiments/sft_new_mix/`. The day-by-day working record of the three-arm experiment is
[`experiments/sft_new_mix/patch_experiment/NOTEBOOK.md`](../../experiments/sft_new_mix/patch_experiment/NOTEBOOK.md);
this notebook summarizes it and adds the later analyses and the typography arm. Analysis scripts and their
text outputs are in [`analysis/`](analysis/); paths to local checkouts and downloaded eval data are written as
`$MARIN_CHECKOUT`, `$ANALYSIS_DIR` and `$DATA_DIR`.

## Question and decision criteria

- Goal and why it matters: the 10.05 SFT model runs past 100,000 characters on 27-32% of AIME24 outputs and
  45-53% of OlympiadBench outputs, and most of those outputs are wrong or never box an answer. The audit in
  [marin#11188](https://github.com/marin-community/marin/issues/11188) listed flaws in the SFT mix that could
  teach this and other habits. The question is whether removing or repairing those documents removes the
  behaviors, measured on the model's own outputs.
- Baseline and alternatives: an unchanged control against arms that differ only in the cache catalog (the
  file that maps each mixture component to a tokenized store). Arms tested: a 32,000-token reasoning cap on
  three sources (`cap32k`), the cap plus removal of give-up endings (`cap32k-giveup`), and normalization of
  two gpt-oss typography characters across every SFT store (`utf-normalize`).
- Evidence needed to choose: for the length arms, the share of AIME24 and OlympiadBench outputs of at least
  100,000 characters at step 222 at most half the control's, with no decisive accuracy loss. For the
  typography arm, incidence of U+202F and U+2011 in outputs near zero, with no decisive accuracy loss. A
  difference is decisive when it exceeds 1.96 combined binomial standard errors.
- Constraints and unresolved questions: one seed per arm; 222 of 4,750 recipe steps; about 8.5 hours on 128
  H100s per arm.

## Baseline

- Date: 2026-11-13 to 2026-11-14.
- Published source revision: `ae0bc4fb54` (control training), branch `dml/sft-patch-11188-experiment`.
- Configuration, command, and external inputs: the historical 262,144-token-context cold-start recipe
  (`sft-262k-legacy.yaml`) for the Grug 67B-A2B mixture-of-experts model, initialized from
  `s3://marin-us-east-02a/marin/long-loop/base/2026.11.08.8`, catalog
  `s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/full-mixture/2026.10.24.2/cache_catalog.json`.
  Command: `python -m experiments.sft_new_mix.patch_arm train --arm control --catalog <catalog> --output
  s3://marin-us-east-02a/tmp/ttl=30d/sft-data-patch-11188/runs/control/sft --steps 222 --replicas 16`,
  submitted through `iris --cluster=marin job run --target-cluster=cw-rno2a --priority production`.
- Environment and hardware when relevant: cw-rno2a, 16 nodes of 8 H100. Iris job
  `/dml/sft-patch-11188-control-cw-rno2a-r4`; W&B run
  [sft-patch-11188-control](https://wandb.ai/kailua-ventures/sft-data-patch-11188/runs/sft-patch-11188-control).
- Evaluation: six benchmarks served with vLLM, data parallel 8 on one 8xH100 node, `max_model_len` 73,728,
  launched from Marin `main` at `12f0ceff06` with the campaign Evalchemy configs in
  `experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs/`. AIME24 and
  OlympiadBench run at temperature 0.7, seed 42, 10 samples per problem. MATH-500, GSM8K (0-shot), HumanEval+
  and IFBench run at server defaults; GSM8K, HumanEval+ and IFBench run with thinking off. Run groups
  `sft11188-control-s100b` and `sft11188-control-s222` under `s3://marin-us-east-02a/marin/evals/`.
- Result and evidence: step 222 AIME24 71.3% accuracy, 33.3% of outputs >= 100k characters, 24.3% with no
  boxed answer; OlympiadBench 22.3%, 48.3%, 24.7%; MATH-500 95.0%; GSM8K 81.2%; HumanEval+ 68.3%; IFBench
  strict prompt 19.7%. The fresh control reproduced the published step-222 checkpoint within 3.7 points
  (AIME24) and 5.0 points (OlympiadBench) on the long-output share. Published checkpoints of the full run
  show the habit at full strength by step 222 (37%, then 32-43% at steps 471, 1,024 and 2,048), so a 222-step
  run is a valid readout.
- Limitations: one seed; 30 problems per math benchmark; GSM8K moves 5.1 points within the control between
  steps 100 and 222.

## Entries

### 2026-11-13 - Length and give-up filters

- Question or hypothesis and reason for trying it: very long reasoning traces and traces that end by citing
  time pressure or guessing ("give-up endings") teach runaway reasoning.
- Published source revision: filter `c6a5b1d520`, experiment `5b48d94244` and `ae0bc4fb54`.
- Command or material configuration: `reasoning_store_filter.py` drops rows from existing tokenized stores and
  copies kept rows token for token. `cap32k` removes documents whose longest reasoning span exceeds 32,000
  tokens from `math_v3`, competitive coding (Python, C++) and OpenThoughts4. `cap32k-giveup` also removes
  give-up endings from `math_v3` and competitive coding. Filter jobs `/dml/sft-filter-<variant>-<source>` on
  cw-us-east-08a; training jobs `/dml/sft-patch-11188-<arm>-cw-rno2a-r4`.
- External inputs and artifact references: filtered stores under
  `s3://marin-us-east-02a/tmp/ttl=30d/sft-data-patch-11188/stores/<variant>/<source>/train`, each with a
  `selection.json` listing every dropped row and its reason.
- Result: the cap removes 84% of competitive-coding tokens, 95% of OpenThoughts4 tokens and 31% of
  `math_v3` tokens. Untouched sources (`science_v2/vendor`, `math_v2/high`, `math_proofs_v2`, `math_v4`,
  multilingual math) still put about 8% of training positions in long-reasoning documents, about a third of
  the control's exposure. Hand-read give-up hits: about 21 of 24 real in coding, about 17 of 24 unsupported
  guesses in math.
- Evidence: notebook entries of 2026-11-13 in `experiments/sft_new_mix/patch_experiment/NOTEBOOK.md`.
- Interpretation and limitations: a strong dose on three sources, partial on the mix as a whole.
- Decision and next action: train both arms for 222 steps beside the control.

### 2026-11-14 - Length and give-up arms at step 222

- Question or hypothesis and reason for trying it: primary readout under the pre-registered rule.
- Published source revision: `ae0bc4fb54` (training), `12f0ceff06` (eval launcher).
- Result:

  | Step 222 | AIME24 acc / >= 100k chars / no box | OlympiadBench acc / >= 100k chars / no box | MATH-500 | GSM8K | HumanEval+ | IFBench strict |
  |---|---|---|---|---|---|---|
  | control | 71.3% / 33.3% / 24.3% | 22.3% / 48.3% / 24.7% | 95.0% | 81.2% | 68.3% | 19.7% |
  | cap32k | 77.0% / 32.0% / 19.7% | 22.4% / 51.8% / 21.7% | 95.8% | 84.1% | 62.8% | 19.7% |
  | cap32k-giveup | 71.3% / 40.0% / 24.3% | 25.0% / 51.7% / 29.0% | 93.8% | 82.5% | 66.5% | 21.7% |
  | pass threshold | long share <= 16.7% | long share <= 24.2% | | | | |

- Evidence: W&B runs [cap32k](https://wandb.ai/kailua-ventures/sft-data-patch-11188/runs/sft-patch-11188-cap32k)
  and [cap32k-giveup](https://wandb.ai/kailua-ventures/sft-data-patch-11188/runs/sft-patch-11188-cap32k-giveup);
  run groups `sft11188-<arm>-s222`.
- Interpretation and limitations: neither arm passes. The only decisive difference is cap32k's AIME24
  accuracy, +5.7 points against a 3.9-point bar. No guard is decisively worse.
- Decision and next action: reanalyze the stored rollouts before designing another arm.

### 2026-11-15 - Reanalysis of the length and give-up arms

- Question or hypothesis and reason for trying it: the readout used only the long-output share and the
  no-box rate. Check other measures of runaway, the give-up rate itself, and what the guards can detect.
- Published source revision: no training; stored rollouts only.
- Command or material configuration: [`analysis/holistic/`](analysis/holistic/) (`load.py`, `math.py`,
  `summ.py`, `tail.py`, `anat.py`, `guards.py`, `he.py`), outputs in `summ.txt`, `tail.txt`, `anat.txt`,
  `guards.txt`.
- External inputs and artifact references: run groups `sft11188-{control,cap32k,cap32k-giveup}-s222`,
  `sft11188-control-s100b` and `sft11188-cap32k-giveup-s100`.
- Result:
  - The 100,000-character threshold is about 22,000-25,000 tokens, below the 32,000-token cap. Paired
    per-problem differences at longer thresholds show no reliable reduction: cap32k on AIME24 falls 4.0
    points over 150,000 characters (95% bootstrap interval -9 to +1) and 4.3 points over 200,000 characters
    (-9 to 0). cap32k-giveup outputs got longer by about 10,000-12,000 characters per problem.
  - The training filter's math give-up rule fires on 0-1.7% of outputs in every arm, the control included,
    against 3.7% of `math_v3` training rows. The model does not reproduce that phrasing.
  - The per-problem long-output share correlates 0.91-0.95 between arms. In the control, never-solved
    problems run long on 100% of AIME24 samples and 58% of OlympiadBench samples; always-solved problems run
    long on 0-5%.
  - cap32k's AIME24 gain comes almost entirely from long outputs that end correct (7.0% to 11.7% of
    outputs); short-output accuracy stays at 96%.
  - HumanEval+: cap32k loses 27 items and gains 18 (paired p = 0.23).
- Interpretation and limitations: the null on runaway stands. The give-up arm could not test whether
  give-up endings matter. The primary metric sat below the cap. Three of six guards ran with thinking off,
  and no reasoning-mode coding benchmark checked the cap.
- Decision and next action: look for pathologies in the baseline model's outputs that trace to specific data
  flaws, and test a filter whose target is known to appear in outputs.

### 2026-11-15 - Pathology scan of all eval trajectories

- Question or hypothesis and reason for trying it: which #11188 flaws show up as measurable behavior in the
  model's outputs, and how much accuracy would removing them buy?
- Published source revision: no training; stored rollouts only.
- Command or material configuration: [`analysis/pathologies/`](analysis/pathologies/). `turns.py` flattens
  runs into one table of outputs and agent steps; `detect.py`, `lang.py`, `agent_trials.py`, `falsedone.py`
  and `leak.py` compute the flags; `nnbsp_acc.py` and `lift.py` estimate accuracy effects. The run list is
  `runs_10.05.txt`.
- External inputs and artifact references: all stored evals of the 10.05 SFT model (about 72,000 outputs and
  agent steps across 22 benchmarks), the #11188 arms at steps 100 and 222, and the marin-logbooks#40 arms at
  step 238.
- Result: each detector is tied to one flaw, and hits were read by hand. Detectors whose hits came from the
  prompt were discarded (for example OlympiadBench prompts that contain the reference answer as context).

  | Pathology | 10.05 model | #11188 arms, steps 100-222 | Suspected training source |
  |---|---|---|---|
  | gpt-oss typography (U+202F, U+2011) | 43% of MMLU-Pro outputs | 3-12% of math, GSM8K and IFBench outputs | teacher style in `finance_v1`, `math_v2`, `math_proofs_v2`, `math_v3` |
  | Token soup: mixed-script gibberish in agent steps | 18% of TBLite and 22.5% of Terminal-Bench 2 trials; 0.3% of steps under 4k context tokens, 4.6% above 40k | not measured (long agent evals not run) | hypothesis: binary and pickle dumps in long-context replay |
  | Success claimed right after a visible error | 9.7% of TBLite and 3.7% of Terminal-Bench 2 trials | 3-7% of DS-1000 trials | agent traces ending in unverified success claims |
  | Wrong-language answer or stray foreign words | 0.1-2.5% of outputs | 0.3-2.3% of outputs | multilingual v1 |

  Quoted instructions absent from the prompt, "Perfect!" openers, summary files written into the repository,
  template tags and echoed replay junk each occur in at most about 1% of outputs. Responses to harness
  warnings could not be scored: the Terminus harness ran with inline thinking text and does not store raw
  responses. Hidden-test awareness could not be scored: nearly every SWE-bench trial failed in the harness
  before its first step.

  Estimated pass-rate lift if a pathology vanished, within the same task (TBLite and Terminal-Bench 2 run each
  task three times): token soup +1.7 points on TBLite (95% interval 0.2 to 3.4) and +1.9 on Terminal-Bench 2
  (0.4 to 4.1); success claims after an error +0.4 (-1.1 to 1.7) and +0.6 (0 to 1.5). Typography: of 116
  graded answers containing the characters, no wrong answer becomes right when they are removed.
- Interpretation and limitations: few flaws show up in outputs, and none is a large accuracy lever. The lift
  estimates are upper bounds, because a trial that avoids one pathology can fail another way. Token soup
  tracks context length, so it can only be tested with the long agent evals.
- Decision and next action: test typography normalization as a positive control. Its sources are exactly
  known and it appears in 3-12% of outputs, so if normalizing the data does not remove it within 222 steps, a
  null result on any other habit would carry no information.

### 2026-11-16 - Typography-normalized arm

- Question or hypothesis and reason for trying it: does replacing U+202F (narrow no-break space) with a space
  and U+2011 (non-breaking hyphen) with `-` in every SFT store remove both characters from outputs?
- Published source revision: normalizer `c6d7592824`, notebook records `589608f599`, `ff6a4a4a3e`,
  `52ca136176` (this branch).
- Command or material configuration: [`experiments/sft_new_mix/typography_store_normalize.py`](../../experiments/sft_new_mix/typography_store_normalize.py).
  The stores hold token IDs only, and each character is usually one token in the 2026.09.18 tokenizer
  (`110 miles` tokenizes as `110`, U+202F, `m`, `iles`). The normalizer finds each occurrence on the row's
  bytes, because U+2011 is sometimes split across two tokens; decodes the words around it, replaces the
  character and re-encodes those words; and checks that every rewritten row decodes to the source text with
  only those characters replaced, keeps its special tokens and contains neither character. Source stores are
  opened read-only. Eight partition jobs `/dml/sft-typography-normalize-p0`..`p7` (cw-us-east-08a, 32 CPUs
  each, about 30 minutes) and one assemble step that writes the catalog. Training: `patch_arm train --arm
  utf-normalize --catalog <normalized catalog> --steps 222 --replicas 16`, otherwise identical to the control.
- External inputs and artifact references: catalog
  `s3://marin-us-east-02a/tmp/ttl=30d/sft-data-patch-11188/typography/cache_catalog.json`; per-store counts in
  `typography/stores/catalog-report.json`; W&B run
  [sft-patch-11188-utf-normalize](https://wandb.ai/kailua-ventures/sft-data-patch-11188/runs/sft-patch-11188-utf-normalize).
- Result, including failure or restart: the normalizer rewrote 123 of 216 SFT stores, 7.1 million of 21.2
  million rows, replacing 32.5 million U+202F and 45.9 million U+2011. SFT tokens fell from 234.42 billion to
  234.36 billion. Replay and validation entries and every row count equal the control catalog's. A 900-row
  read-back found no mismatch. Most affected: `opencode_v1/agent_skills` 100% of rows, `finance_v1` 99.6%,
  `math_proofs_v2` 94%, `instruction_following_chat_v3/instruction_following` 87%, `math_v2` 71-76%, `math_v3`
  34%. The first training job, `/dml/sft-patch-11188-utf-normalize-cw-rno2a`, hung at step 48 (13:12 UTC) with
  every rank blocked inside the train step and no error, about when the next half-hourly temporary checkpoint
  was due. It held the 128 GPUs for 10.5 hours before it was noticed, was cancelled, and was resubmitted
  unchanged as `/dml/sft-patch-11188-utf-normalize-cw-rno2a-r2`, resuming from temporary step 36. Losses for
  steps 0-9 stayed within 0.003 of the control's; throughput matched at 505k tokens per second.
- Evidence: `experiments/sft_new_mix/patch_experiment/NOTEBOOK.md` entry of 2026-11-16.
- Interpretation and limitations: the cause of the hang is unknown.
- Decision and next action: export steps 100 and 222 and evaluate with the same six configs, using model
  configs rebuilt from the `sft11188-cap32k-giveup-s100` run record (they differ only in name and location;
  [`analysis/utf-eval/`](analysis/utf-eval/)).

### 2026-11-17 - Typography arm at steps 100 and 222

- Question or hypothesis and reason for trying it: readout of the typography arm.
- Published source revision: `52ca136176` (records), `12f0ceff06` (eval launcher).
- Command or material configuration: [`analysis/utf-eval/readout.py`](analysis/utf-eval/readout.py) on run
  groups `sft11188-utf-normalize-s100` and `-s222` against `sft11188-control-s100b` and `-s222`.
- Result:

  | Share of outputs with U+202F or U+2011 | Control s100 | utf-normalize s100 | Control s222 | utf-normalize s222 |
  |---|---|---|---|---|
  | MATH-500 | 12.4% | 0.2% | 9.0% | 0.0% |
  | GSM8K | 6.7% | 0.0% | 6.8% | 0.0% |
  | IFBench | 6.0% | 0.0% | 5.0% | 1.0% |
  | OlympiadBench | 6.7% | 0.3% | 3.3% | 0.0% |
  | AIME24 | 3.3% | 0.0% | 2.7% | 0.0% |
  | HumanEval+ | 0.0% | 0.0% | 0.0% | 0.0% |

  | Step 222 | AIME24 acc / >= 100k chars / no box | OlympiadBench acc / >= 100k chars / no box | MATH-500 | GSM8K | HumanEval+ | IFBench strict |
  |---|---|---|---|---|---|---|
  | control | 71.3% / 33.3% / 24.3% | 22.3% / 48.3% / 24.7% | 95.0% | 81.2% | 68.3% | 19.7% |
  | utf-normalize | 71.3% / 32.7% / 23.0% | 24.7% / 48.7% / 25.7% | 94.4% | 82.6% | 66.5% | 22.0% |

  No benchmark prompt contains either character. The remaining hits are two outputs at step 100 (`3n‑6` in a
  MATH-500 table, `⟂ B` in an OlympiadBench answer) and three IFBench outputs at step 222
  (`finite‑order`, `Nigeria‑style`, and a run of U+202F after an emoji). At step 100, GSM8K was 3.1 points
  lower (86.3% to 83.2%), the only decisive difference there, and AIME24's long-output share was 7.7 points
  higher (36.0% to 43.7%, not decisive); neither persisted to step 222.
- Interpretation and limitations: normalization removes the habit at both checkpoints with no measurable
  accuracy change. The benchmarks are English and mostly math; French number formatting uses U+202F
  correctly, and non-English typography after normalization was not evaluated.
- Decision and next action: see the final decision.

## Final decision

- Selected approach and evidence: normalize U+202F and U+2011 in the SFT mix. It cuts their incidence in
  outputs from 3-12% to 0-1% at steps 100 and 222 with no decisive accuracy, long-output or no-box change. It
  is a cleanup and does not change accuracy or runaway reasoning. It also shows that a data change can remove
  a copied habit within 222 steps, so the length arms' null reflects the length filter itself.
- Rejected alternatives and why: the 32,000-token cap on three sources and the added give-up filter did not
  reduce runaway reasoning on any measure tried. Runaway follows problem difficulty (per-problem correlation
  0.91-0.95 between arms). The give-up filter targeted phrasing the model does not produce.
- Marin contribution PRs: none yet. A PR adding the normalizer to the production SFT pipeline is the
  candidate contribution.
- Reproduction instructions: build the normalized stores with `python -m
  experiments.sft_new_mix.typography_store_normalize --catalog <catalog> --output-root <dir> --part <i>
  --parts 8 --tokenizer <tokenizer.json> --workers 28` for each part, then `--catalog <catalog> --output-root
  <dir> --catalog-output <new catalog>`. Train with `patch_arm train` and the new catalog; export with
  `patch_arm export`; evaluate with `analysis/utf-eval/export_and_eval.sh` and score with
  `analysis/utf-eval/readout.py`.
- Remaining uncertainty: one seed per arm and 30 problems per math benchmark, so effects of a few points
  cannot be separated from noise; the decisive-difference test treats the 10 samples of a problem as
  independent. A rerun of the length question needs a token-based threshold above the cap, a cap on every
  reasoning source, reasoning-mode guards including a coding benchmark, and two seeds. Token soup, the one
  pathology with a measurable lift, needs the long agent evals at the trained checkpoint. The normalized
  stores, catalog and model exports sit under a 30-day temporary prefix and expire in mid-December 2026.
