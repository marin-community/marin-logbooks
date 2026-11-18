---
title: "Deletion Tuning on the Snowball 262K base under Micro210"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/63"
related_issues:
  - "https://github.com/marin-community/marin/issues/11663"
  - "https://github.com/marin-community/marin/issues/11658"
  - "https://github.com/marin-community/marin/issues/11337"
  - "https://github.com/marin-community/marin/issues/10804"
  - "https://github.com/marin-community/marin/issues/10933"
research_branch: "codex/63-snowball-deletion-tuning"
marin_base_revision: "b5d2db5f178263fac27d914cc4ecc1bc8eba31ab"
archive_template_revision: "c18f42e7294032cc9613903f61735d32d3394313"
status: active
---

# Deletion Tuning on the Snowball 262K base under Micro210

## Question and decision criteria

- Goal and why it matters: measure whether a learned deletion mask (Deletion
  Tuning, [arXiv 2611.10866](https://arxiv.org/abs/2611.10866)) on the Snowball
  67B-A2B long-context-extension base recovers most of the Micro210 gain of full
  fine-tuning from the same data and budget, with less forgetting and a mask
  artifact far smaller than a checkpoint. If it holds, post-training capability
  gains become a compact mask artifact instead of a full checkpoint.
- Baseline and alternatives: the unmodified base
  (`open-athena/snowball-67b-a2b-base-262k-qk175-skew8` @ step-157000) under
  Micro210; alternatives are the paper's baselines in spirit (full fine-tuning
  rows that exist on the same data) and the paper's own ablations (Wanda-only
  masks, random masks).
- Evidence needed to choose: Micro210 scores with uncertainty for base and every
  variant, paired per-task deltas against the base, learned sparsity, mask
  artifact size, and GPU-hours.
- Constraints and unresolved questions: identical Micro210 pins, seed, and
  task/attempt IDs across all rows; production jobs submit at production
  priority and let Kueue preempt; artifacts stay in-region (CoreWeave S3); any
  deviation from the paper's recipe needs a recorded reason here and in the
  marin issue Decision Log.

## Baseline

- Date: pending
- Published source revision: pending
- Configuration, command, and external inputs: pending
- Environment and hardware when relevant: pending
- Result and evidence: pending
- Limitations: pending

## Entries

### 2026-11-18 15:45 UTC - Task start: paper reading and plan of record

- Question or hypothesis and reason for trying it: the marin issue #11663 plan
  asks whether deleting fewer than 0.05 percent of pretrained weights matches
  full fine-tuning on a 67B-A2B production MoE under the project's own policy
  eval, extending the paper (max Qwen3-30B-A3B, 128 routed experts, gate
  projection only) to Snowball's 256-expert 67B/2B architecture.
- Published source revision: `b5d2db5f178263fac27d914cc4ecc1bc8eba31ab`
  (marin `main`, read-only). Working implementation tree:
  marin worktree branch `benfeuer/snowball-deletion-tuning` (same commit).
- Command or material configuration: no experiment run yet. Read the paper end
  to end, including appendices.
- External inputs and artifact references: paper
  [arXiv 2611.10866](https://arxiv.org/abs/2611.10866) (v1, 2026-11-07); marin
  issues #11663 (plan of record), #11658 (Exp3 V5 SFT mix), #11337 (long-loop
  campaign rows), #10804/#10933 (Micro210 runtime).
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: paper spec extracted (below). No
  failures.
- Evidence: paper sections 4, 5.1, appendices A-H.
- Interpretation and limitations: the paper's MoE result masks the gate
  projection of every routed expert and leaves the router unchanged; Snowball
  has 256 routed experts plus a shared expert, so the V3 wider scope (gate, up,
  down of routed and shared experts) is untested by the paper and is a genuine
  research question.
- Decision and next action: implement Deletion Tuning in the Snowball training
  path (Levanter SFT + Megatron/SkyRL GRPO), measure the base row first, then
  V1-V4.

## Paper spec extracted for implementation (arXiv 2611.10866 v1)

- Target weights: the gate projection of transformer MLPs. Dense models: one
  gate matrix per block. MoE (Qwen3-30B-A3B): the gate projection inside each of
  the 128 routed experts in every layer; the router stays unchanged.
- Trainable parameters: one logit `G` per masked weight, same shape as `W`.
  Everything else is frozen. AdamW, fp32 master weights, bf16 mixed precision,
  gradient checkpointing.
- Forward during training: `W~(t) = W ⊙ sigmoid(alpha(t) * G)`, with the linear
  ramp `alpha(t) = alpha_start + (t/T) (alpha_end - alpha_start)`.
  - GRPO: `alpha_start = 1e3`, `alpha_end = 1e4`, constant lr `3e-4`.
  - SFT: `alpha_start = 1e2`, `alpha_end = 1e3`, constant lr `5e-4`.
- Hard mask at evaluation and for rollouts: `W^ = W ⊙ 1[G > 0]`. Deploying the
  hard mask instead of the soft one changed accuracy by at most 0.08 points.
- Wanda prior initialization: score `s_ij = |W_ij| * ||X_j||_2` from one forward
  pass over calibration prompts from the task's training distribution (2048
  prompts for GRPO and SFT; 32 for SDFT). `G_ij ~ N(0.03, 0.01^2)` when `s_ij`
  is in the top `1 - eta` fraction, else `N(-0.03, 0.01^2)`, with
  `eta = 1/2048`, which zeros about 0.18 percent of gate weights at init.
- Objective: the regime's own loss plus the MaskLLM sparse-weight regularizer
  `-lambda * ||W~||_2^2` with `lambda = 1e-5` for SFT and GRPO (0 for SDFT). No
  fixed sparsity constraint; training decides how many weights to delete.
- GRPO recipe (SimpleRL-Zoo): prompt batch 256, n = 8 rollouts, response length
  4096, 150 steps, KL penalty `1e-4`. (GSM8K variant: batch 64, n = 4, 3
  epochs, no KL.)
- SFT recipe: 400 steps per task at effective batch 4, 75/25 train/eval split.
- Learned sparsity: every GRPO run deletes under 0.05 percent of all weights
  (0.034-0.046 percent across scales, appendix B).
- Adapter storage: bit-packed boolean hard mask per masked matrix, zstd
  compressed (1.0 MB at Llama1B, 8.4 MB at Llama8B, 29.6 MB at
  Qwen3-30B-A3B).
- Ablations: best Wanda-only mask stays within 2.4 points of the base while the
  trained mask gains up to 53 points (section 5.3); robust to `eta` in
  {1/2048, 1/4096, 1/8192}, lr in {1e-4, 3e-4}, lambda in {0, 1e-5} (appendix
  G).

## Planned variants (marin issue #11663)

- V1 SFT, paper scope: mask the gate projection of every routed expert; learn
  with SFT on a fixed sample of the Exp3 V5 SFT mix (#11658); paper SFT
  hyperparameters.
- V2 GRPO, paper scope: same mask scope, learned with GRPO on the Exp3 RL
  curriculum; paper GRPO hyperparameters.
- V3 wider scope: mask gate, up, and down projections of routed and shared
  experts, in the V1 regime (untested by the paper).
- V4 controls: Wanda-only mask and random mask at V1's learned sparsity, no
  training (the paper does not report a random-mask control).
