---
title: "Outcome-filtering Open-SWE traces for Qwen3-8B SFT"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/39"
related_issues: ["https://github.com/marin-community/marin/pull/10813"]
research_branch: "codex/39-open-swe-outcome"
marin_base_revision: "9cde3e5b69871ba733ccbe6ac84cd589d25b3003"
archive_template_revision: "ba8edcdfd730b451e18aed6245234566be5053af"
status: active
---

# Outcome-filtering Open-SWE traces for Qwen3-8B SFT

## Question and decision criteria

- Goal and why it matters: decide whether Marin should train only on
  [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) trajectories whose `resolved`
  field is 1, as [marin#10813](https://github.com/marin-community/marin/pull/10813) proposed.
- Baseline and alternatives: a random arm matched to the solved arm in partition mix and text volume per
  partition. Details and the full design are in issue #39.
- Evidence needed to choose: a task-by-task difference on SWE-bench Verified larger than evaluation noise,
  supported by measures that do not depend on the agent loop.
- Constraints and unresolved questions: training at 131,072 tokens ran out of memory, so the first rung
  trains at 65,536 tokens on trajectories that fit.

## Baseline

The noise pilot, the data stage and the training smoke are recorded in issue #39 (2026-11-12 to 2026-11-13).
Their code (`experiments/open_swe_outcome/arms.py`, `trial.py`, the Harbor policies and the pilot scripts) was
published on this branch on 2026-11-19 with the Terminus-2 round (merge `a064ecebdc`); the first round ran from
commit `d97646fe35` of the same history.

## Entries

### 2026-11-15 18:36 UTC - 10,000-row arms on SWE-bench Verified (200 tasks)

- Question: does the solved arm beat the matched random arm on SWE-bench Verified?
- Published source revision: training and evaluation ran from `d97646fe35` (unpublished; see Baseline).
- Training: Qwen3-8B, full fine-tune in Levanter, 65,536-token window, assistant-token loss, seeds 0 and 1.
  - Solved arm: 10,000 rows, all `resolved` = 1. Tokenized cache
    `s3://marin-us-east-08a/marin/tokenized/solved-10000-chat-f60c77/2026.07.17` (412M tokens).
  - Random arm: 9,435 rows, matched in partition mix and volume. Outcomes: 46.4% resolved, 40.4% failed,
    13.2% unknown (measured from the index and `membership_d3c29ebe/random-10000.parquet`).
  - Final checkpoints: step 786 (solved) and step 787 (random), under
    `s3://marin-us-east-08a/marin/users/dml/checkpoints/open-swe-outcome/qwen3-8b-{solved,random}-10000-seed{0,1}-ctx65536/2026.11.13/`.
    The first HF exports had wrongly sized tensors. `hf-cpu/` holds the corrected float32 re-exports.
- Evaluation: Harbor `swebench-verified-200-c9` (200 tasks, terminus-2 agent, 9 concurrent trials,
  50-minute agent limit), vLLM bf16 on 2×H100 TP2, `cw-rno2a`, submitted 2026-11-15 02:12 UTC.
  Jobs: `/dml/eval-open-swe-outcome-eval-{solved-s0-4d071c3c,solved-s1-681927f2,random-s0-e2e85902,random-s1-d0e55b63}`.
- Result: every run attempted all 200 tasks. Three runs failed at final archiving because one trial's
  archived attempt was incomplete, so their eval records hold no metrics. The fourth (random-s1) recorded
  `infra_failed` at 95% coverage. Per-trial grades were read from each run's FineStore `samples` table.

  | Run | Solved / graded | Agent timeouts | Generated tokens |
  |---|---|---|---|
  | solved-s0 | 6 / 191 | 133 | 35.6M |
  | solved-s1 | 7 / 190 | 63 | 25.4M |
  | random-s0 | 4 / 197 | 126 | 36.2M |
  | random-s1 | 9 / 190 | 78 | 27.5M |

  On the 168 tasks graded in all four runs: solved arm 12 solves, random arm 11. Paired difference
  +0.006 ± 0.027 solves per task (out of 2). Only 15 tasks were solved by any run.
- Evidence: results roots `s3://marin-us-east-02a/marin/evals/open-swe-outcome-eval-*-swebench-verified-200-c9-*/results`.
- Interpretation and limitations: the solve rate (2–5%) is too close to the floor to separate the arms. It is
  also far below the about 40% the 131k-context pilot checkpoint scored in the noise pilot. Two suspected
  causes, neither tested: the 65,536-token window, and the scaffold shift from OpenHands, SWE-agent and
  mini-SWE-agent in training to terminus-2 in evaluation. Most trials hit the agent limit. Both seed-0
  checkpoints generated about 40% more tokens than the seed-1 checkpoints at the same throughput (about
  620 tokens/s).
- Decision: measure the arms with held-out loss, which does not depend on the agent loop.

### 2026-11-16 19:13 UTC - Held-out validation set

- Question: build a fixed held-out set that neither arm saw, in the exact training format.
- Published source revision: `experiments/open_swe_outcome/val_loss.py` on this branch. The job ran from an
  uncommitted working tree at Marin `9cde3e5b69` (Iris bundles the working tree); the committed file is
  that version plus the later per-turn and paired additions.
- Command: `python experiments/open_swe_outcome/val_loss.py build --out-dir .../val-loss/v1 --design independent --per-class 300 --train-reference 100 --verify-rows 300 --seed 0`
  (the `--design` flag was added later; v1 used the independent design). Job `/dml/open-swe-val-build-r3`,
  `cw-us-east-08a`, CPU.
- Construction: exclude every trajectory in either arm and every task (`instance_id`) they cover (13,534 tasks).
  Keep `status` = kept and at most 65,536 tokens. One trajectory per task. Equal resolved and unresolved
  counts within each partition, with partitions weighted by their share of the training arms. Add 100
  training rows from each arm as a memorization reference.
- Result: 299 resolved, 299 unresolved, 100 + 100 training rows, written to
  `s3://marin-us-east-08a/marin/users/dml/open-swe-outcome/val-loss/v1/val.jsonl.gz`.
- Conversion check: converting raw traces to the arm format reproduced 300 sampled training rows exactly in all
  eight training partitions. Matching required two transformations the arm builder applies and the raw data
  lacks:
  - Tool-call IDs are renumbered `call_<message>_<call>`, and each tool result gets the next unanswered call ID.
  - A fixed 1,222-character "Solution Originality" instruction block is inserted between `</CRITICAL>` and
    `</instructions>` in the first user message of every v1_2 mini-SWE-agent trace. That is 6,929 of the 18,974
    training rows.

### 2026-11-16 22:50 UTC - Held-out loss

- Question: does the solved arm predict held-out successful trajectories better than the random arm?
- Command: `val_loss.py score --model <hf-cpu/step-*> --name <arm-seed> --val-dir .../val-loss/v1` for the four
  checkpoints and `Qwen/Qwen3-8B`. One H100 each, `cw-rno2a`, production priority. Jobs
  `/dml/open-swe-val-score-p2-{solved-s0,solved-s1,random-s0,random-s1,base}`. The loss uses Levanter's
  `ChatProcessor` with the training chat template, so the assistant mask matches training. Inference is in bf16
  with SDPA (cuDNN SDPA disabled; it has no execution plan at these lengths).
- Result: mean loss per assistant token (nats).

  | Model | Held-out resolved | Held-out unresolved | Train rows, random arm | Train rows, solved arm |
  |---|---|---|---|---|
  | base Qwen3-8B | 0.878 | 0.884 | 0.847 | 0.826 |
  | random, s0 / s1 | 0.4362 / 0.4363 | 0.4502 / 0.4502 | 0.387 / 0.390 | 0.376 / 0.376 |
  | solved, s0 / s1 | 0.4370 / 0.4366 | 0.4521 / 0.4526 | 0.409 / 0.409 | 0.359 / 0.359 |

  Paired per-trajectory differences, solved minus random: held-out resolved +0.0005 ± 0.0002 (s0) and
  −0.0000 ± 0.0002 (s1); held-out unresolved +0.0021 ± 0.0002 and +0.0016 ± 0.0002; each arm on its own
  training rows about −0.017.
- Evidence: `.../val-loss/v1/scores/*.parquet`; `val_loss.py summarize --val-dir .../val-loss/v1`.
- Interpretation: replicated across seeds. The arms predict held-out successes equally well. The solved arm is
  slightly worse on failures, by under 0.5% of the loss. Each arm's advantage on its own training rows is ten
  times the held-out gap and does not transfer.

### 2026-11-16 23:30 UTC - Classifying outcome with the arm likelihood ratio

- Question: can the models distinguish resolved from unresolved trajectories?
- Score: per-token log p_solved − log p_random, summed over both seeds.
- Evidence: `logbooks/39-open-swe-outcome/classify.py` on the v1 scores.
- Result: AUROC 0.66 [0.62, 0.70] for the ratio. Each model's own per-token loss gives 0.50 (base) to 0.52
  (fine-tuned). Trajectory length alone gives 0.57. The ratio holds at 0.65 within partition × length-quartile
  strata, and correlates at r = 0.76 between seeds.
- Decision: check where in a trajectory the signal sits, then control for the task.

### 2026-11-17 01:40 UTC - Per-turn prefix rescoring (exploratory)

- Question: is the signal the agent reporting its own test results at the end?
- Command: `val_loss.py score ... --scores-dir turn-scores --subset-prefix heldout` for the four checkpoints;
  jobs `/dml/open-swe-val-turns-*`. Each assistant turn's loss is stored separately. Per-turn totals match the
  earlier per-trajectory scores to a relative 8×10⁻⁸.
- Evidence: `logbooks/39-open-swe-outcome/prefix.py`; `.../val-loss/v1/turn-scores/`.
- Result: AUROC of the ratio by turns scored (median 52 assistant turns): first 10% 0.53, first 25% 0.61,
  first 50% 0.64, all 0.66, all but the last 5 turns 0.65, last turn alone 0.53.
- Interpretation: the signal is not the agent reporting its outcome at the end. Each task appears once, so task
  difficulty could explain it.

### 2026-11-17 04:40 UTC - Paired within-task control

- Question: on the same task and teacher, does the ratio prefer the resolved attempt?
- Command: `val_loss.py build --out-dir .../val-loss/v2-paired --design paired --per-class 300 --train-reference 0 --verify-rows 300 --seed 0`
  (job `/dml/open-swe-val-build-paired`), then per-turn scoring (jobs `/dml/open-swe-val-paired-*`).
- Construction: held-out tasks with at least one resolved and one unresolved trajectory in the same partition
  (same teacher model and scaffold). One of each per task. Every partition met its quota: 299 pairs.
- Evidence: `logbooks/39-open-swe-outcome/paired.py`; `.../val-loss/v2-paired/turn-scores/`.
- Result: the ratio ranks the resolved trajectory higher in 0.50 of pairs, 95% CI [0.44, 0.56]. Win rates are
  0.45–0.50 for every turn subset. The resolved trajectory is shorter in 0.47 of pairs.
- Interpretation: the cross-task AUROC of 0.66 came from task difficulty. On a fixed task neither arm prefers the
  successful attempt. Caveat: these tasks were both solved and failed by the same teacher, so outcomes may depend
  partly on luck, and an effect smaller than the CI is not excluded.

## Summary

Training Qwen3-8B on 10,000 solved trajectories gave no measurable gain over a matched random arm that was 46%
solved. Every measure agreed, at both seeds:

- SWE-bench Verified (200 tasks): 12 against 11 solves on 168 tasks graded in all runs.
- Held-out loss on successful trajectories: equal to within 0.0005 nats per token.
- Same task and teacher, one resolved and one unresolved attempt: within-pair win rate 0.50 [0.44, 0.56].

Caveats: one model size, one data size and the final checkpoint only. SWE-bench at 2–5% solve rates rules out
only large effects. Loss measures imitation of the teachers, which is a different thing from solving tasks.

Note on the cross-task classification signal: we measured an AUROC of 0.66 for separating resolved from
unresolved held-out trajectories by the arm likelihood ratio, with each task appearing once. Controlling for
length and source left 0.65. Pairing a resolved and an unresolved trajectory from the same task and teacher
brought it to 0.50, so the signal came from task difficulty.

## Next step: repeat on Qwen3.6-35B-A3B

Qwen3-8B at 65,536 tokens solves too few SWE-bench Verified tasks to compare arms. Qwen3.6-35B-A3B scores
54–59% in most of 12 full 500-task Verified runs on Marin's eval dashboard (range 50–64%), so a filtering effect
of a few points would be measurable task by task. Plan:

- Train the same two 10,000-row arms, two seeds each, from Qwen3.6-35B-A3B.
- Read out in order of cost: held-out loss and the paired within-task test on the existing `v1` and `v2-paired`
  sets, then the 200-task SWE-bench Verified run.
- Open questions to settle first:
  - Does Levanter support Qwen3.6's architecture for training? Marin serves it in vLLM with a gated-DeltaNet
    prefill backend.
  - What does a 35B-total, 3B-active MoE cost per billion tokens at 65,536 tokens?
  - Do the existing sets need rebuilding with Qwen3.6's chat template?
  - Should the training window or the evaluation scaffold change, given the 2–5% result at 65,536 tokens with
    terminus-2?

## Final decision

- Selected approach and evidence: no outcome filter at Qwen3-8B. At 10,000 rows per arm, outcome filtering
  showed no benefit on any measure. Issue #39 closed on 2026-11-17; the Qwen3.6-35B-A3B repeat is planned in
  [next-steps.md](next-steps.md) and will run under a new issue.
- Update 2026-11-19: the first round's 2–5% solve rate came largely from the training format. Trained in the
  Terminus-2 format, Qwen3-8B solves 12–17%, enough to compare arms task by task; see
  [Qwen3-8B trained in the Terminus-2 format](#qwen3-8b-trained-in-the-terminus-2-format).
- Rejected alternatives and why: none yet.
- Marin contribution PRs: none.
- Reproduction instructions: see the commands in each entry. The arm-building and training code is on this
  branch (see Baseline).
- Remaining uncertainty: effects at larger models, larger data or other stopping points; effects smaller than the
  paired test's CI.

## Qwen3.6-35B-A3B repeat

### 2026-11-17 23:30 UTC - Levanter model for Qwen3.6-35B-A3B (Phase 0, training path)

- Question: can Levanter train `Qwen/Qwen3.6-35B-A3B`? Phase 0 step 1 of [next-steps.md](next-steps.md).
- Prior work: an Echo search and a Marin PR search found no Qwen3.5 or Qwen3.6 training work. Levanter had the
  pieces: a `GatedDeltaNet` layer, a Qwen3 MoE block and a gated-attention layer, but nothing that combined them
  or read this checkpoint.
- Source: this branch, Marin base `18ae01ee42` (main on 2026-11-17) merged with the #39 archive at `a0c600138a`.
  - New: `lib/levanter/src/levanter/models/qwen3_5_moe.py` (`Qwen35MoeConfig`, registered as `qwen3_5_moe`;
    the defaults are Qwen3.6-35B-A3B's), and `lib/levanter/tests/test_qwen3_5_moe.py`.
  - Small changes: a `partial` RoPE config in `layers/rotary.py`, because Qwen3.6 rotates 64 of the 256
    dimensions in each head. `GatedRmsNorm.axis` in `layers/gated_deltanet.py` became a static field so the layer
    can be stacked and serialized.
- Architecture, checked against `transformers` 5.12.1 `modeling_qwen3_5_moe.py`:
  - 40 layers in 10 groups. Each group is 3 Gated DeltaNet layers followed by 1 gated softmax-attention layer.
    Levanter scans over the groups so the mixed layer pattern compiles as one block.
  - Every layer: 256 routed experts (top 8, renormalized) plus a shared expert scaled by a sigmoid gate.
  - RMSNorm scales are stored as offsets from 1, as in Gemma.
  - The attention `q_proj` emits a query and an output gate for each head. Levanter stores them as separate
    parameters and packs them into one tensor only when reading or writing a checkpoint.
  - The released checkpoint is a vision-language model. The loader keeps the text decoder under
    `model.language_model.*` and drops `model.visual.*` and the multi-token-prediction head `mtp.*`. Exports are
    therefore text-only `Qwen3_5MoeForCausalLM` checkpoints.
- Verification:
  - `uv run --package marin-levanter --group test --extra torch_test pytest lib/levanter/tests/test_qwen3_5_moe.py -m "torch or not torch"`:
    4 passed. These tests cover:
    - A tiny randomized HF model (6 layers, 8 experts). Levanter's logits, loss and every parameter
      gradient match HF's eager implementation within 1e-4.
    - A Levanter export reloads in HF with identical logits.
    - Loading from a vision-language checkpoint.
    - Causality, and independence from the delta-rule chunk size.
  - Mutation check: dropping the shared-expert gate fails both HF parity tests (largest logit error 4.6).
  - Neighbouring suites (`test_gdn_layer`, `test_gdn_kernels`, `test_rotary`, `test_qwen3_moe`): 54 passed.
  - Full-size layout: [qwen36_hub_shapes.py](qwen36_hub_shapes.py) reads the safetensors headers of the Hub
    checkpoint at revision `345d467b117d` without downloading weights.
    - Levanter's model initialized from that config has the same 693 text tensors, with identical names and
      shapes: 34,660,610,688 parameters, all BF16 on the Hub. `total_trainable_params` gives the same count.
    - The checkpoint's other 1.29B parameters are the vision tower and the MTP head.
- Not yet verified: logits from the real 35B weights against HF, any run on an accelerator, step time and
  memory at 65,536 tokens, and sharding of the 256 experts. The delta-rule kernel is the existing pure-JAX
  chunked implementation; no fused kernel is wired in.
- Interpretation: there is now a training path in Levanter, so the decision between Levanter and another trainer is
  settled in Levanter's favour unless cost (step 2) rules it out. Remaining Phase 0 work is in
  [next-steps.md](next-steps.md).

## Qwen3-8B trained in the Terminus-2 format

The first Qwen3-8B round solved 2–5% of the 200 tasks, too few to separate the arms. This round tests the
suspected cause: the arms were trained in their source scaffolds' native tool-call formats (OpenHands, SWE-agent,
mini-SWE-agent) and evaluated with Harbor's Terminus-2 agent.

### 2026-11-17 23:00 UTC - The native-format checkpoints write malformed Terminus-2 replies

- Question: is the 2–5% solve rate a scaffold mismatch?
- Evidence: trial `django__django-10914` of the solved-s1 evaluation
  (`s3://marin-us-east-02a/marin/evals/open-swe-outcome-eval-solved-s1-swebench-verified-200-c9-05aa6e7228235e508a13732f455a484d`).
  8 of 30 agent steps had JSON parse errors and 9 had parser warnings ("Extra text detected before JSON
  object"). Keystrokes often lacked a trailing newline.
- The four evaluations served the checkpoints with no vLLM reasoning parser, so each reply's reasoning reached
  Terminus-2's parser and stayed in the conversation history.
- No usable base-model reference existed. The only base Qwen3-8B run on this setup
  (`open-swe-noise-pilot-base`) lost 94 of 100 trials to `DaytonaValidationError`.
- Decision: rewrite the training trajectories into the Terminus-2 protocol, retrain both arms, and evaluate base
  Qwen3-8B on the same 200 tasks.

### 2026-11-17 23:35 UTC - Terminus-2-format arms

- Source: [`experiments/open_swe_outcome/terminus.py`](../../experiments/open_swe_outcome/terminus.py),
  `terminus_arms.py` and `terminus_chat_template.py`, commit `8295b2ac4c` (merged into this branch at
  `a064ecebdc`). Tests: `experiments/open_swe_outcome/test_terminus.py`, 12 passed with `test_arms.py`.
- Conversion of one arm row into the conversation Terminus-2 would have produced from the same shell actions:
  - The task message becomes Terminus-2's prompt (read from the vendored Harbor template) around the issue text,
    which is all a SWE-bench Verified task gives the agent.
  - Shell calls become keystrokes. Editor calls become `cat -n`, `find`, a heredoc or a Python replacement script.
  - Observations become the echoed command and its output, truncated to 10,000 bytes as Terminus-2 truncates.
  - `submit`, `finish` and mini-SWE-agent's submission command become `task_complete`, followed by Terminus-2's
    confirmation question and a confirming reply.
  - OpenHands `think` calls join the next turn's reasoning. `/workspace/<repo>` paths become `/testbed`.
  - Rows with operations that have no faithful shell equivalent (`undo_edit`, `insert`, failed `create`, commands
    sent while another runs) are rejected.
- Check on the 798-row held-out set `v1`: 778 rows convert, and all 42,725 converted replies parse with no error or
  warning under Harbor's own `TerminusJSONPlainParser`.
- Conversion lengthens rows by 13%, because each edit appears in both the reply and the echoed output. 22% of
  converted rows exceed 65,536 tokens (40% of SWE-agent rows), so the arms were re-selected by Terminus-2 token
  count rather than reused.
- Candidates: the 31,600-row arms built at `arms.py` version `2026.11.13.1` (131,072-token window), which are the
  leading rows of each hash order. Taking the first eligible rows in the same order selects what a rebuild from
  every trajectory would select. Command: `python -m experiments.open_swe_outcome.build_terminus_arms`, job
  `/dml/open-swe-terminus-arms`, `cw-us-east-08a`.
- Result (`s3://marin-us-east-08a/marin/open-swe-outcome/terminus/membership_954793f1/summary.json`):

  | Arm | Rows | Tokens | Resolved / failed / unknown |
  |---|---|---|---|
  | `terminus-solved-10000` | 10,000 | 425.8M | 100% resolved |
  | `terminus-random-10000` | 9,504 | 426.0M | 47.9% / 39.0% / 13.1% |

  Per-partition token volumes agree within 0.1%. 2,754 of 58,000 pool rows (4.7%) did not convert, mostly
  failed editor `create` calls (1,102) and failed `str_replace` results (740).
- Chat template: Qwen3's own template drops reasoning from assistant turns before the last user message, which in
  Terminus-2 is every turn but the last. `TERMINUS_CHAT_TEMPLATE` renders a think block on every assistant turn.

### 2026-11-18 00:15 UTC - Training, seed 0

- Command: `python -m experiments.open_swe_outcome.trial --arm terminus-{solved,random}-10000 --format terminus
  --seed 0 --accelerator 8xH100`, jobs `/dml/open-swe-terminus-train-{solved,random}-10000-s0`, `cw-rno2a`,
  production priority. Optimizer, batch size, window and one-epoch schedule match the native-format runs.
- Smoke (20 steps): 63.5 s per step, about 8,260 tokens/s, the same as the native-format smoke. Loss fell from
  0.69 to 0.39.
- Both runs finished at step 812 after 15 h 38 min. Checkpoints:
  `s3://marin-us-east-08a/marin/users/dml/checkpoints/open-swe-outcome/qwen3-8b-terminus-{solved,random}-10000-seed0-ctx65536/2026.11.13/`.
  The 8-GPU HF exports were redone on CPU (`reexport_hf.py --run ... --step 812`) to `hf-cpu/step-812`; hidden size
  4096 and 65,536 positions checked.

### 2026-11-18 18:45 UTC - First evaluation attempt: generation-prompt mismatch

- The first evaluations served the checkpoints with the `qwen3` reasoning parser. In-progress trials had parse
  errors on 20 of 30 steps (solved arm) and 9 of 24 (random arm). In most failures the reply was valid
  Terminus-2 JSON that the reasoning parser had filed as reasoning, leaving empty content.
- Cause: the template's generation prompt ended in `<think>\n`. With the Qwen3 tokenizer, a turn without reasoning
  trains as `<think>`, `\n\n`, `</think>`, and a turn with reasoning as `<think>`, `\n`, text. A prompt already
  ending in a lone `\n` token leaves no trained path to `</think>` on a turn without reasoning, so the model wrote
  the JSON inside the think span. The native-format arms used the same generation prompt.
- Fix (`e06b464dd3`): the generation prompt ends at `<think>`, which is a token prefix of training for both kinds
  of turn. Training rendering is unchanged; the evaluation configs serve the corrected template.
- The base configuration also needed two fixes: `hf_overrides` must be a JSON string, and vLLM 0.25 checks
  `max_model_len` against `max_position_embeddings` without the YaRN factor, so the override raises both.
- The two fine-tune evaluations were cancelled and relaunched; the failed base attempts had ended on their own.

### 2026-11-19 - SWE-bench Verified, 200 tasks

- Policy: [`swebench-verified-200-c9-declarative.yaml`](../../experiments/open_swe_outcome/harbor/swebench-verified-200-c9-declarative.yaml),
  the same 200 tasks and concurrency as the first round with Daytona declarative image builds, which create no
  account snapshots. Terminus-2, 65,536-token input, 50-minute agent limit, 2×H100, `cw-rno2a`.
- Model configs: [`eval/`](../../experiments/open_swe_outcome/eval/). Fine-tunes serve the corrected template with
  the `qwen3` reasoning parser. Base Qwen3-8B serves its own template with YaRN (factor 2 over 32,768 tokens).
- Runs: `open-swe-terminus-eval-{base-r3,solved-s0-r2,random-s0-r2}` under `s3://marin-us-east-02a/marin/evals/`.
  All three attempted every task. Each run record failed at Harbor's final archiving step, so grades come from
  per-trial `result.json` files: [terminus_swebench.py](terminus_swebench.py).

  | Model | Graded | Solved | Agent timeouts | Other exceptions |
  |---|---|---|---|---|
  | base Qwen3-8B | 182 | 5 (2.7%) | 16 | 18 |
  | Terminus solved arm, seed 0 | 162 | 27 (16.7%) | 63 | 39 |
  | Terminus random arm, seed 0 | 154 | 19 (12.3%) | 67 | 46 |

  On the 126 tasks graded in all three runs: base 4, solved arm 21, random arm 17.

  | Comparison | Common tasks | First only | Second only | Both | Sign test p |
  |---|---|---|---|---|---|
  | solved arm vs random arm | 134 | 14 | 10 | 9 | 0.54 |
  | solved arm vs base | 150 | 22 | 2 | 2 | 3.6e-05 |
  | random arm vs base | 144 | 17 | 4 | 0 | 0.0072 |

- Interpretation:
  - The format was the bottleneck. Terminus-format training lifts Qwen3-8B from 2–5% to 12–17%, and both arms beat
    the base model significantly.
  - The solved arm leads the random arm by 4 tasks on common tasks. That is about one standard deviation of the
    discordant count alone, before evaluation noise, and is consistent with no effect.
  - The decision rule needs `s`, the spread between two evaluations of one checkpoint, which is not yet measured.
  - Limits: one seed per arm. Other exceptions (mostly Daytona connection errors) left 154–162 tasks graded per
    fine-tune. A third of fine-tune trials hit the agent limit and were graded on the state they left.
