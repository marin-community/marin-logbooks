# Next steps: repeat the outcome-filtering test on Qwen3.6-35B-A3B

Handoff for [marin-logbooks#39](https://github.com/marin-community/marin-logbooks/issues/39). The
Qwen3-8B results are in [notebook.md](notebook.md); read its entries before starting. This file says
what to run next, what to measure, and what would change the decision.

## Where we are

[marin#10813](https://github.com/marin-community/marin/pull/10813) proposed training only on Open-SWE-Traces
trajectories whose `resolved` field is 1. We tested it on Qwen3-8B with two 10,000-row arms, each trained with
two seeds:

- **Solved arm:** 10,000 resolved trajectories.
- **Random arm:** 9,435 trajectories, matched to the solved arm in partition mix and text volume. It is
  46.4% resolved, 40.4% failed and 13.2% unknown.

Every measure came out the same for the two arms:

- **SWE-bench Verified, 200 tasks:** 12 against 11 solves on the 168 tasks graded in all four runs.
- **Held-out loss on resolved trajectories:** equal to within 0.0005 nats per token.
- **Paired control** (one resolved and one unresolved attempt per task, same teacher): the solved arm's
  preference for the resolved attempt had a win rate of 0.50, 95% CI [0.44, 0.56].

The Qwen3-8B result is weak evidence because SWE-bench solve rates were 2–5%. At that level only a large
effect would show. Qwen3.6-35B-A3B scores 50–64% on full SWE-bench Verified on Marin's eval dashboard (12 runs,
most at 54–59%). At that level a difference of a few points between arms would be measurable task by task.

## Question and decision

**Question:** at a model that solves about half of SWE-bench Verified, does fine-tuning on solved trajectories
beat fine-tuning on the matched random arm?

**Decision rule.** Fix this before training; it is issue #39's rule.

- Let `s` be the standard deviation of the solve-count difference between two evaluations of the same
  checkpoint, on the same 200 tasks.
- **Adopt the filter** if the solved arm beats the random arm by more than `3s`, pooled over seeds.
- **Keep all data** if the difference is within `1.5s`.
- **Anything in between:** run a larger rung (31,600 rows per arm, already built as `solved-31600` and
  `random-31600`).

## Phase 0: settle feasibility before training

1. **Training path.** Qwen3.6 is a hybrid model: gated-DeltaNet linear attention plus a mixture of experts.
   - Levanter has a `GatedDeltaNet` layer (`lib/levanter/src/levanter/layers/gated_deltanet.py`, with Pallas
     kernels) and a `Qwen3MoeLMHeadModel` (`lib/levanter/src/levanter/models/qwen3_moe.py`).
   - No Levanter model combines the two or loads the `Qwen/Qwen3.6-35B-A3B` checkpoint.
   - **Decide:** add a Levanter model with an HF round-trip test, or use another trainer. Search Echo and open
     PRs first for Qwen3.5/3.6 training work.
2. **Cost.** Measure step time and memory at a 65,536-token window on one node, as the Qwen3-8B smoke did
   (8,270 tokens/s on 8×H100). Then estimate the cost of training four runs at about 412M tokens each.
   - Full fine-tuning holds 35B parameters plus optimizer state, so plan for sharding across devices.
   - If four runs cost too much, run one seed per arm first and add the second only if the result lands in
     the in-between band.
3. **Base-model noise and reference.** Evaluate base `Qwen/Qwen3.6-35B-A3B` twice on the same 200-task set.
   - This measures `s`, and gives the score the fine-tunes should match or beat.
   - Use the same agent, limits and serving settings as the fine-tunes. The dashboard's 50–64% comes from
     500-task runs whose agent settings must be checked before they are compared with ours.
4. **Context.** The serve config `experiments/evaluation/serve/models/Qwen/Qwen3.6-35B-A3B.yaml` sets
   `max_model_len: 32768`. Training and evaluation need at least 65,536 so the evaluation can match the
   training window. Confirm that vLLM serves that length with compilation and CUDA graphs enabled. Do not use
   eager mode.

## Phase 1: training

- **Data:** reuse the existing arms unchanged: `solved-10000_5e1e1b9c` and `random-10000_d5a9e952` under
  `s3://marin-us-east-08a/marin/open-swe-outcome/`.
- **Tokenization:** retokenize with Qwen3.6's chat template.
  - The Qwen3-8B template carried `{% generation %}` tags for the assistant-token loss mask; the Qwen3.6
    template will need the same.
  - Check that assistant turns, tool calls and thinking blocks are masked as intended. Score a few rows with
    the mask printed before training.
- **Configuration:** match the Qwen3-8B runs except where the model forces a change: 65,536-token window,
  assistant-token loss, about one epoch, seeds 0 and 1. Record every difference in the notebook.
- **Watch the base model's skill.** The teacher models (MiniMax-M2.5, Qwen3.5-122B, Qwen3.6-27B,
  Qwen3.8-27B, DeepSeek-V4-Flash) are similar in strength to the base model. Fine-tuning on their traces could
  lower the base score. That outcome still answers the filtering question, but report it.

## Phase 2: what we evaluate

Run these in order of cost. Each uses fixed inputs that already exist, so results compare directly with
Qwen3-8B.

| Readout | Inputs | What it answers | Cost |
|---|---|---|---|
| Held-out loss | `val-loss/v1`: 299 resolved and 299 unresolved held-out trajectories, plus 100 training rows per arm | Does the solved arm predict held-out successes better? Does each arm memorize its own rows? | ~1 GPU-hour per checkpoint |
| Paired win rate | `val-loss/v2-paired`: 299 tasks, each with one resolved and one unresolved attempt from the same teacher | On a fixed task, does the solved arm prefer the successful attempt? | ~1 GPU-hour per checkpoint |
| SWE-bench Verified | Harbor `swebench-verified-200-c9`, 200 tasks, terminus-2, 50-minute agent limit | Does the solved arm solve more tasks? This is the decision metric. | ~14 h per checkpoint at 9–12 trials at a time |

The two loss readouts:

- **Scoring:** use `experiments/open_swe_outcome/val_loss.py score` with `--scores-dir turn-scores`, so
  per-turn losses are kept. Then run `logbooks/39-open-swe-outcome/paired.py` and `classify.py`.
- **Chat template:** `TOKENIZER` and `TOKENIZED_ARTIFACT` in `val_loss.py` point at Qwen3-8B. Point them at
  the Qwen3.6 tokenizer and its tokenized cache. The sets themselves are stored as messages and need no
  rebuild.
- **Precision:** `score` loads the model through HF transformers in bf16. Check that it runs Qwen3.6's linear
  attention, and record the GPU count.

SWE-bench:

- **Comparison:** compare task by task, over tasks graded in every run.
- **Grades:** read per-trial grades from the FineStore `samples` table. A single incomplete archived attempt
  fails Harbor's finalize step and leaves the run record without metrics. That happened to three of four
  Qwen3-8B runs.
- **Failure modes to report:** agent timeouts and infrastructure failures.

## Known traps

- **Arm format.** The arms differ from raw traces in two ways:
  - Tool-call IDs are renumbered.
  - A 1,222-character "Solution Originality" block is inserted into every v1_2 mini-SWE-agent trace.

  `val_loss.py` reproduces both and checks them against training rows. Any new data built from raw traces must
  apply them too.
- **Unpublished code.** The code that built the arms and trained the Qwen3-8B runs (`arms.py`, `trial.py`,
  the Harbor policy) is on the unpublished branch `worktree-open-swe-outcome-pilot` (`d97646fe35`). Ask the
  owner to push it before reusing it.
- **Iris:**
  - `--priority production` is allowed only on `cw-rno2a`.
  - Jobs at default priority can sit in the Kueue queue for hours.
  - The data lives in `cw-us-east-08a`.
- **Scoring with SDPA:** disable cuDNN SDPA (`torch.backends.cuda.enable_cudnn_sdp(False)`). It found no
  execution plan at these lengths on H100.
- **Daytona snapshot quota (45):** a full 500-task Verified run creates one snapshot per task image and hits
  the quota. Stay on the 200-task policy, or confirm snapshot cleanup works first.

## Deliverables

1. Notebook entries for Phase 0 (training path, cost, base noise), training, and each readout. Use the
   format of the existing entries: source revision, command, artifacts, result, interpretation.
2. A decision under the rule above, recorded in the notebook and in the issue #39 body.
3. If the filter is adopted, a Marin PR that adds outcome filtering to
   `lib/marin/src/marin/datakit/download/open_swe_traces.py`, linking the issue.
