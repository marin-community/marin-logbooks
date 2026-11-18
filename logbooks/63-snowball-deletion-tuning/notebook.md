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
  67B-A2B long-context-extension base (67 billion total parameters, 2 billion
  active per token, 262,144-token context) recovers most of the Micro210 gain of
  full fine-tuning from the same data and budget, with a mask artifact far
  smaller than a checkpoint. If it holds, post-training capability gains become
  a compact mask artifact instead of a full checkpoint.
- Primary claim in this phase: each variant's improvement over the *base* model
  on Micro210, with paired per-task deltas and task-clustered uncertainty.
  The full-fine-tuning equivalence hypothesis is **untested in this phase**
  unless a matched comparator exists (see Comparator).
- Success criteria:
  - Each variant reports its Micro210 score with task-clustered uncertainty and
    paired per-task deltas against the base (same task/attempt IDs both sides).
  - "Most of the gain" is defined only against a *matched* full fine-tuning
    arm: a variant recovers most of the gain when it reaches at least 80
    percent of that arm's Micro210 gain over the base. If a comparator's gain
    over the base is not positive, its comparison is reported as inconclusive.
  - "Same budget" means identical training tokens and optimizer steps (and for
    V2, identical rollout samples per prompt and steps) as the variant arm.
- Known limitations of the record: artifacts under private storage URIs
  (s3://marin-us-east-*) are referenced but not publishable; run results link
  public issue/W&B records wherever they exist.

## Comparator

- The base row is the primary comparator for every variant.
- A full-fine-tuning comparator counts as matched only when it shares the
  variant's initial checkpoint, training data sample, token/step budget, and
  evaluation policy. Historical rows (round-1 SFT746 → RL16, Exp2 RL20) used
  different data and recipes, so they are context, not comparators.
- Where no matched arm exists, the record reports "missing comparator" instead
  of implying equivalence was tested. A matched arm can be added later on the
  same sample and budget.

## Retention (out of scope in this phase)

- The paper's forgetting claims (prior-task accuracy within about one point of
  the base) are background, not a measured outcome here.
- This phase's decision claim uses the Micro210 score and paired per-task
  deltas; per-benchmark paired deltas give a partial regression view within the
  evaluated cohort. A dedicated held-out retention panel (suggested in a
  research comment on [marin #11663](https://github.com/marin-community/marin/issues/11663))
  remains an open question and is not adopted in this phase.

## Evaluation contract

All rows use this contract. The runtime binding is being confirmed with the
Exp2 worker; fields marked **pending** are filled before the first launch and
recorded here.

- Cohort: the 200-task / 440-attempt Micro210 subset, excluding the tau3-pi
  benchmark ([marin #11083](https://github.com/marin-community/marin/issues/11083)).
  This is a subset of the 210-task / 450-attempt design in
  [marin #10804](https://github.com/marin-community/marin/issues/10804); rows
  are labeled with this cohort. The same task/attempt identity manifest (dataset
  row UID, task ID, repetition index) is reused for every row.
- Dataset: `open-athena/Marin-Micro-Eval-210` @
  `4564ed081c316bc3231ae5c3a53a46b39248c819`.
- Runtime wheels: `d5dd315d58f0f47514dd0a2208e90e2a48541328`.
- Tokenizer: `marin-tokenizer-tools` @ `3f6879ac`.
- Judge: MiniMaxAI/MiniMax-M3-MXFP8, 8x H100 (TP8), context 131,072.
- Policy serving: vLLM, TP1/DP8/EP8, `max_num_seqs` 8, context 65,536,
  16,384 tokens per turn, temperature 0.7, top-p 0.95.
- Seed: `20261030`.
- Prompt interface: the runtime's chat/tools serialization, tokenizer revision,
  thinking setting, and tool-parser settings come from the shared binding. The
  base model is a completion model (its model card states the bundled chat
  template does not indicate instruction tuning), so part of any measured gain
  may reflect instruction/tool-format ability rather than capability; this
  caveat applies to every row equally because the serialization is held fixed.
  A preflight confirms the binding's template and sampling settings before each
  launch. **Pending**: the exact binding values (they arrive with the Exp2
  binding confirmation).
- Aggregation: primary = pooled scored-attempt mean over scored attempts.
  Also reported: mean of benchmark means, native equal-20 summary, and coverage
  (scored/assigned attempts). Missing outcomes are reported, never silently
  dropped. Historical example with this contract: the round-1 SFT746 eval-only
  row ([marin #11337 comment](https://github.com/marin-community/marin/issues/11337#issuecomment-6965189889))
  scored pooled 0.4023 / mean-of-benchmark-means 0.4798 / native equal-20
  0.4703 with 420 of 440 attempts scored — these are available-only
  partial-credit reward means.
- Paired uncertainty: task-level cluster bootstrap (resample tasks, not
  attempts; repeated attempts of one task are not independent).

## Baseline

- Date: pending (base row not yet run)
- Published source revision: pending
- Configuration, command, and external inputs: pending
- Environment and hardware when relevant: pending
- Result and evidence: pending
- Limitations: pending

## Shorthand

- **Exp3**: the third long-loop experiment (training campaign).
- **V5**: the fifth data release of that campaign ([marin #11658](https://github.com/marin-community/marin/pull/11658)).
- **SFT746**: the round-1 supervised-fine-tuning checkpoint at optimizer step
  746 ([marin #11337](https://github.com/marin-community/marin/issues/11337)).
- **Micro210**: the 210-task micro evaluation cohort built for these long-loop
  runs ([marin #10804](https://github.com/marin-community/marin/issues/10804),
  [marin #10933](https://github.com/marin-community/marin/issues/10933)); this
  task's rows use its 200-task subset (see Evaluation contract).
- **67B-A2B / 262K**: 67 billion total parameters with 2 billion active per
  token; 262,144-token context.

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
  issues [#11663](https://github.com/marin-community/marin/issues/11663) (plan
  of record), [#11658](https://github.com/marin-community/marin/pull/11658)
  (Exp3 V5 SFT mix), [#11337](https://github.com/marin-community/marin/issues/11337)
  (long-loop campaign rows), [#10804](https://github.com/marin-community/marin/issues/10804)
  and [#10933](https://github.com/marin-community/marin/issues/10933) (Micro210
  runtime).
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

### 2026-11-18 16:00 UTC - Cold-read round 1 adopted; record corrections

- Question or hypothesis and reason for trying it: the logbook PR
  ([marin-logbooks #64](https://github.com/marin-community/marin-logbooks/pull/64))
  cold-read returned NOTES; this entry records the adopted fixes.
- Published source revision: unchanged (`b5d2db5f178263fac27d914cc4ecc1bc8eba31ab`).
- Command or material configuration: `~/.local/bin/logbook-cold-read.sh 64
  ses_e10b141a5ffemZjyE5PKMfjSsB 1`.
- External inputs and artifact references: report at
  `/Users/benfeuer/Documents/experiments/active/logbook-cold-reads/64-r1-20261118T154736Z.report.md`
  (local, not publishable); its findings are summarized here.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: verdict NOTES with 8 findings; all 8
  adopted:
  1. Comparator narrowed: the full-fine-tuning equivalence hypothesis is marked
     untested for this phase; matched-comparator definition, the 80-percent
     "most of the gain" rule, and the budget definition added (Comparator
     section).
  2. Retention marked out of scope for this phase's claim; the held-out
     retention panel is recorded as an open question (Retention section).
  3. Evaluation contract selected: 200-task/440-attempt subset with pins, seed,
     identity manifest, aggregation order, missing-outcome treatment, and
     task-clustered bootstrap (Evaluation contract section).
  4. Base-model prompt interface recorded as a contract item with the
     completion-model caveat and a preflight requirement (Evaluation contract
     section).
  5. Planned-run table added distinguishing resolved from pending settings,
     including the RLOO-N vs GRPO decision for V2 (Planned runs section).
  6. Random-mask attribution fixed: Wanda-only is the paper-derived ablation;
     the untrained random mask is a new control in this experiment, distinct
     from the paper's trained random sparse-weight baseline (MTH).
  7. Initial-deletion fraction corrected: eta = 1/2048 selects the low-Wanda
     delete group (about 0.049 percent of gate weights); the noisy Gaussian
     logit init then deletes about 0.18 percent of targeted gate weights under
     the hard mask at initialization. The plan-of-record description conflated
     the quantile with the deletion rate; a Decision Log note was posted on
     [marin #11663](https://github.com/marin-community/marin/issues/11663).
  8. Index heading replaced, shorthand defined, issue references made
     repository-qualified (this file and the index record).
- Evidence: this entry; the linked report (local).
- Interpretation and limitations: the evaluation contract's binding-dependent
  fields remain pending until the Exp2 binding is confirmed.
- Decision and next action: continue implementation; fill pending fields at
  launch; re-run the cold read after the next milestone.

### 2026-11-18 16:30 UTC - Training path decision: Exp3 SFT runtime + Megatron/SkyRL

- Question or hypothesis and reason for trying it: which training path hosts
  the Deletion Tuning implementation — the brief says "Megatron/SkyRL as Exp3
  uses", and the repo survey raised the question because Levanter's
  `snowball.py` snapshot is a different (main-branch) implementation of the
  same architecture.
- Published source revision: implementation run branch
  `benfeuer/snowball-deletion-tuning-run` = `benfeuer/exp3-rl-ib-restart`
  (`ba63397d23`) + the core commit `96b9e4a550` (cherry-picked from the
  main-derived branch `benfeuer/snowball-deletion-tuning` at `c550aa612f`).
- Command or material configuration: no experiment run yet.
- External inputs and artifact references: Exp3 branch survey
  (`experiments/post_training/long_loop/sft.py` imports
  `sft_runtime.train.run_grug`; `experiments/grug/moe/optimizer.py` registers
  the MuonH-family optimizer); base HF config
  (256 routed experts, hidden 2560, moe_intermediate 1280, 26 layers, 67B
  total / 2B active).
- Environment and hardware when relevant: target clusters are the Exp3 H100
  clusters (cw-us-east-02a / rno2a).
- Result, including failure or restart: decision recorded (also posted to the
  [plan of record's Decision Log](https://github.com/marin-community/marin/issues/11663)):
  1. V1/V3 (SFT) run on the Exp3 SFT runtime
     (`experiments/post_training/long_loop/sft_runtime/` — custom JAX/Equinox
     Hero-MoE model), not Levanter's `snowball.py`: that is the runtime Exp3's
     production SFT stage trains with, so the variants run on the same path
     round-1 SFT746 used.
  2. V2 (GRPO) runs on the Megatron/SkyRL path Exp3 uses (`specs/rl.yaml`,
     `exp3_rl_restart.py`).
  3. Framework-neutral core (alpha ramp, Wanda init, regularizer, packed mask
     artifact) lands on the main-derived branch as the PR payload; the
     sft_runtime and SkyRL integrations live on the run branch and land with
     that lineage when it merges.
  4. Geometry/memory check: V1 scope masks ≈ 21.8B routed gate weights; fp32
     logits plus Adam moments ≈ 262 GB ≈ 2.1 GB/GPU on 128 H100; V3 (gate, up,
     down of routed and shared) ≈ 65.6B masked weights ≈ 8.2 GB/GPU. Both fit.
- Evidence: this entry; the Decision Log comment.
- Interpretation and limitations: V2's Megatron design (logits as extra
  Megatron parameters on the vendored Grug layer, hard-mask conversion at the
  vLLM sync boundary) is designed but not yet implemented.
- Decision and next action: finish the SFT-runtime integration and its tiny-MoE
  behavior tests; then the V2 Megatron design; then the base row once the Exp2
  binding lands.

### 2026-11-18 17:10 UTC - SFT-runtime integration implemented and tested

- Question or hypothesis and reason for trying it: implement the paper's method
  on the Exp3 SFT runtime so V1/V3 can run on the production path.
- Published source revision: run branch `benfeuer/snowball-deletion-tuning-run`
  at `f621f82785` (= exp3-rl-ib-restart `ba63397d23` + core `96b9e4a550` + the
  integration commit). Files:
  `sft_runtime/deletion_masking.py` (new), `sft_runtime/model.py`,
  `sft_runtime/train.py`, `sft_runtime/export_writer.py`,
  `tests/post_training/test_sft_deletion_masking.py` (new).
- Command or material configuration: behavior tests
  `pytest tests/post_training/test_sft_deletion_masking.py` (7 tests) on the
  tiny MoE (4 experts, hidden 8, 2 layers, scatter MoE, reference attention)
  under `compact_grug_mesh(expert_axis_size=1)`; the neighboring suite
  (`tests/rl/test_deletion_tuning.py`, `tests/grug/test_grug_moe_pipeline.py`)
  still passes (26 passed, 1 pre-existing skip). Lint clean; pyrefly clean.
- External inputs and artifact references: paper method (this notebook's spec
  section); `marin.rl.deletion_tuning` core (the packed mask artifact gained an
  in-memory `build_mask_artifact` variant for storage-layer writers).
- Environment and hardware when relevant: single-device CPU mesh for the tests.
- Result, including failure or restart: integration complete:
  1. Model: optional per-weight logit fields on the routed experts
     (`gate_logits`/`up_logits`/`down_logits` on `MoEMLP`) and shared experts
     (`DenseMLP`); the forward reads `stop_gradient(W) * mask(G)` with the soft
     mask `sigmoid(alpha*G)` during training and the hard mask `1[G>0]` for
     evaluation and export; the hard/soft switch mixes arithmetically because a
     scalar `jnp.where` broadcasts a replicated condition against sharded
     cases, which explicit sharding rejects.
  2. Wanda calibration: a capture mode on `MoEMLP` emits shard-local per-expert
     sums of squared input activations (plus the all-token channel sums the
     shared experts read) through the routing-stats path; the driver reduces
     over the batch-shard axis and derives per-expert input norms with a
     layer-mean fallback for unrouted experts.
  3. Init: the global eta-quantile cannot sort a sharded tensor, so the delete
     threshold comes from a strided host-pooled score subsample
     (`wanda_score_samples` + `wanda_delete_threshold`, stride 64 over the
     output axis) and the device init compares `scores <= threshold`
     elementwise — sharding-safe, deterministic given seed, threshold, and
     mesh. The core module's positional `init_gate_logits` stays the reference
     implementation; the threshold rule matches it up to ties at the boundary
     value (test asserts the count difference is bounded by the tie count).
  4. Training: `_make_train_step` partitions gradients with a path-based filter
     tree so Adam state exists only for the gate logits (frozen weights get
     zero updates and no optimizer memory); the alpha ramp updates per step
     from the step counter; `_loss_and_grads` adds
     `-lambda * sum(||W~||^2)` after the per-token denominator and logs
     `train/deletion_regularizer` and `train/deletion_sparsity`.
     `deletion_hard` is a 0/1 float leaf because `jax.grad` rejects bools.
  5. Evaluation: `with_deletion_eval_mode` flips the hard-mask bit; the eval
     model getter and the dropless eval hook wrap it.
  6. Export: `apply_hard_mask` zeroes deleted weights before the bf16 HF
     state dict; the mask artifact rides the export as packed uint8 tensors
     (`np.packbits` bit order) under `deletion-mask/` with shape tensors, and
     the export manifest records it.
  Behavior-test results: soft forward equals a model with the soft-masked
  weights baked in; hard forward equals zeroed weights; a two-step train run
  moves only the gate logits (frozen weights bit-identical) and ramps alpha
  (1e2 -> 325 at step 1 of 4); capture + init produce the paper's expected
  init deletion rate; the regularizer and sparsity match hand computation;
  the exported state dict equals the manually masked weights and the packed
  masks unpack to the same keep mask.
- Evidence: the test file and this entry; commit `f621f82785`.
- Interpretation and limitations: the Wanda down-projection scores reuse the
  expert-input norms as a per-expert scale (the intermediate-width norms are
  not captured) — a recorded deviation for the V3 scope only; the threshold
  rule is quantile-exact up to subsampling noise and tie handling. The V2
  Megatron path is designed but not implemented. Nothing has run at 67B scale
  yet; a smoke through the production geometry is the next gate before any
  variant launch.
- Decision and next action: dry-run the SFT entry point end to end with a
  deletion config on a tiny/one-node spec; design and implement the V2
  Megatron path; launch the base row when the Exp2 binding lands.

## Planned runs (resolved vs pending)

Common settings for every row: the evaluation contract above; base checkpoint
`open-athena/snowball-67b-a2b-base-262k-qk175-skew8` @ step-157000 (HF revision
`058ecaf27b9e4f37219df221a51e7d490d58ec3d`); deletion-tuning defaults
eta = 1/2048, lambda = 1e-5, logit init N(+0.03, 0.01^2) keep group /
N(-0.03, 0.01^2) delete group; Wanda calibration on 2,048 prompts drawn from
the variant's own training sample; hard mask for evaluation, export, and
rollouts; mask artifact + standard checkpoint export per variant.

| Run | Regime | Mask scope | Data | Resolved | Pending |
| --- | --- | --- | --- | --- | --- |
| V0 base | none | none | none | checkpoint revision; eval contract | serving binding confirmation |
| V1 | SFT | routed-expert gate_proj (256 experts, all layers) | fixed sample of Exp3 V5 release | paper SFT recipe: AdamW on gate logits only, lr 5e-4 constant, alpha 1e2→1e3 | sample size + selection seed/manifest; step/token budget (the paper's 400 steps is per single task, not a mixture); packing/context policy; checkpoint selection rule |
| V2 | RL | routed-expert gate_proj | Exp3 RL curriculum | paper GRPO recipe: lr 3e-4 constant, alpha 1e3→1e4, KL 1e-4, hard-mask rollouts | curriculum pin; GRPO vs explicit RLOO-N adaptation label (the existing Snowball RL recipe is RLOO-N); batch/rollout geometry alignment |
| V3 | SFT (V1 regime) | gate, up, down of routed + shared experts | V1's sample | same recipe as V1 | same as V1 |
| V4a | none (no training) | Wanda-only mask at V1's learned sparsity | none | paper-derived ablation | V1's learned sparsity |
| V4b | none (no training) | random mask at V1's learned sparsity | none | new control in this experiment (the paper reports no random-mask control; distinct from its trained random sparse-weight baseline) | V1's learned sparsity |

Every deviation from the paper's recipe gets a reason recorded here and in the
[plan of record's Decision Log](https://github.com/marin-community/marin/issues/11663)
before the run launches.

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
  `eta = 1/2048`. The quantile selects the delete group (about 0.049 percent of
  gate weights); with the noisy logit init, the expected hard-mask deletion at
  initialization is `eta * Phi(3) + (1 - eta) * Phi(-3)` — about 0.18 percent
  of targeted gate weights.
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
