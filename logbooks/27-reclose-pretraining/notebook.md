---
title: "Where Grug's think re-close comes from: SFT replay and pretraining data audit"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/27"
related_issues:
  - "https://github.com/marin-community/marin/issues/10400"
  - "https://github.com/marin-community/marin/issues/10850"
  - "https://github.com/marin-community/marin-logbooks/issues/25"
research_branch: "codex/27-reclose-pretraining"
marin_base_revision: "6d7f94f852e7244dc42c4760b08ab4f66fd2de7a"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: active
---

# Where Grug's think re-close comes from: SFT replay and pretraining data audit

## Question and decision criteria

- Goal and why it matters: Grug SFT checkpoints sometimes close a thinking span
  with `<|end_think|>`, keep reasoning, and close again, so reasoning leaks into
  replies (marin-community/marin#10400). We want to know when the behavior is
  learned and where it comes from, so the fix lands on the right side of the
  pipeline. In particular: is it carried by identifiable pretraining rows that
  could be diffed out cheaply, by the `" answer"` row used to seed the new close
  token, or by SFT generalization that only SFT data can correct?
- Baseline and alternatives: the 10.05 SFT recipe (base step 157,000, 4,750
  steps, batch 256 x 262,144 tokens) and its final re-close probe values.
  Alternatives considered: (a) a masked counterfactual replay of the cooldown
  tail with the suspect rows' loss weight zeroed; (b) a dose-raising contrast
  (suspect source mixed at 10% versus the same documents repaired); (c) seeding
  the close row from the vocabulary mean instead of `" answer"`; (d) SFT data
  that shows a close followed by continuation without a second close.
- Evidence needed to choose: per-step probe curves from an exact SFT replay; the
  count and density of the suspect pattern in pretraining data; whether any
  cheap pretraining-side arm has enough dose to move the probe above its noise
  floor (about 0.01 on `p_first_close`, 0.003 on the hazard, measured from a
  preempted restart of the same run).
- Constraints and unresolved questions: no cross-region bulk data movement;
  16 GPU nodes on CoreWeave us-east-02a for replays; the pretraining caches and
  checkpoints live in GCS us-central2. The mean-seed run's result is recorded
  below; the SFT-side counterexample experiment has not been run.

## Baseline

- Date: 2026-10-24.
- Published source revision: the replay tooling on this branch under
  `experiments/grug_sft/` (`export_sft_batches.py`, `replay_data.py`,
  `reclose_eval.py`, `sft_start_replay.py`), copied from the working tree of the
  `dml/10400-sft10-continuation` worktree at Marin `6d7f94f852`.
- Configuration, command, and external inputs: the 10.05 SFT's first optimizer
  batches exported from its own data loader (seed 0, batch 256 at 262,144
  tokens) and replayed from the base checkpoint at step 157,000. A fixed probe
  set of 10.05 trajectories in three classes (SWE, tool-using, plain) is scored
  at every step by `reclose_eval.py`: `p_first_close` is the probability of
  `<|end_think|>` where 10.05 closed its first span, `p_reclose` the same where
  10.05 later re-closed, `hazard` the chance of closing anywhere in the
  continuation after the first close, split by whether the continuation starts
  with a lead-in such as "Wait,".
- Environment and hardware when relevant: 16 nodes on CoreWeave us-east-02a,
  context axis 4, weights sharded across the context axis (model change in
  `experiments/june_tpu_67b_a2b/moe/model.py` on this branch). Replay losses
  match 10.05 step for step (0.6997, 0.7631 for the first two steps).
- Result and evidence: 10.05's final checkpoint (step 161,750) scores
  `p_reclose` 0.39 / 0.45 / 0.36 and `p_first_close` 0.97 / 0.84 / 0.90 for
  SWE / tool-using / plain. Hazard with a lead-in 0.35 / 0.21 / 0.05, without
  one about 0. The vLLM prompt-logprob path on the Hugging Face export matches
  these to within 0.006
  ([`analysis/vllm_logprobs/reclose_logprobs.py`](analysis/vllm_logprobs/reclose_logprobs.py),
  run as an Iris job against a vLLM server). 09.21 (1,000 SFT steps) scores hazard with lead-in
  0.47 / 0.17 / 0.06 by the same vLLM path. Free-sampling eval leak rates:
  09.21 2.5% bixbench, 4.9% bfcl; 10.05 0.6%, 3.2%; the base model re-closed
  plain-text tags in 3.0% of few-shot samples.
- Limitations: probe values are conditional on 10.05's own trajectories; the
  eval leak rates are free-sampling per thinking message and are not on the same
  scale as the probe.

## Entries

### 2026-10-25 - Control replay from base, 100 steps (`replay-10400-base300e`)

- Question or hypothesis and reason for trying it: when during SFT does
  re-closing appear, and does it grow over training or arrive with the close
  token itself?
- Published source revision: this branch's `experiments/grug_sft/sft_start_replay.py`
  with `--start base --steps 300 --nodes 16 --context-axis-size 4`; cancelled at
  step 100 once the curves had settled. Earlier attempts on 16 nodes failed
  (cross-region batch download stall, then GPU OOM) until the batches were
  copied to the us-east-02a bucket and the weights were sharded across the
  context axis; one preemption restarted the run from step 0 and reproduced the
  first attempt's values within 0.012.
- External inputs and artifact references: batches at
  `s3://marin-us-east-02a/tmp/ttl=30d/dml/sft_replay_10400/batches/`; probe
  output per step at `.../reclose/replay-10400-base300e/step-*.jsonl` (30-day
  TTL from 2026-10-25). Per-step group means are preserved in
  [`analysis/probe_replay-10400-base300e.csv`](analysis/probe_replay-10400-base300e.csv).
- Result: mean `p_first_close` / `p_reclose` by class.

  | Step | SWE | Tool-using | Plain |
  | --- | --- | --- | --- |
  | 5 | 0.000 / 0.000 | 0.003 / 0.004 | 0.003 / 0.004 |
  | 10 | 0.030 / 0.065 | 0.201 / 0.134 | 0.105 / 0.163 |
  | 15 | 0.351 / 0.330 | 0.555 / 0.395 | 0.333 / 0.380 |
  | 25 | 0.674 / 0.406 | 0.549 / 0.384 | 0.285 / 0.354 |
  | 50 | 0.790 / 0.453 | 0.660 / 0.410 | 0.367 / 0.358 |
  | 100 | 0.849 / 0.546 | 0.825 / 0.515 | 0.567 / 0.376 |
  | 10.05 final | 0.969 / 0.392 | 0.837 / 0.447 | 0.902 / 0.360 |

  Hazard with a lead-in at step 100: 0.58 / 0.31 / 0.08 against 10.05's
  0.35 / 0.21 / 0.05. Hazard without a lead-in peaks at steps 13 to 17
  (SWE 0.14, tool-using 0.22, plain 0.07) and decays to 0.03 / 0.02 / 0.00 by
  step 78.
- Interpretation and limitations: re-closing arrives together with the first
  close, between steps 8 and 15, and `p_reclose` reaches 10.05's final level by
  step 15 and never falls. At step 10, `p_reclose` exceeds `p_first_close` for
  SWE and plain, so the model closes after reasoning-like continuation before it
  has learned the original close well. The remaining 4,650 steps of 10.05
  sharpen the first close and slowly lower the lead-in hazard but do not teach
  that a second close is wrong, consistent with the SFT data containing no
  example of one (0 of 1.38M turns). Steps 100 to 300 were not run.
- Decision and next action: stop the control at step 100; test the `" answer"`
  seed with a mean-seeded close row.

### 2026-10-25 - Mean-seeded close row (`replay-10400-meanseed`)

- Question or hypothesis and reason for trying it: the close token's LM-head row
  was seeded from `" answer"`, whose score inside reasoning tracks the token
  closely. If that seed carries the re-close prior in, a neutral seed should
  lower `p_reclose` relative to `p_first_close`.
- Published source revision: `sft_start_replay.py --close-seed mean`, 100 steps,
  16 nodes, context 4, rolling checkpoint under a `ttl=1d` prefix; the mean row
  is implemented by `mean_seeded_token_ids` in
  `experiments/grug_sft/special_token_train.py` with
  `tests/grug/test_special_token_reinit.py`.
- External inputs and artifact references: same batches; probe output at
  `.../reclose/replay-10400-meanseed/step-*.jsonl`; preserved means in
  [`analysis/probe_replay-10400-meanseed.csv`](analysis/probe_replay-10400-meanseed.csv).
- Result: `p_first_close` / `p_reclose`.

  | Step | SWE | Tool-using | Plain |
  | --- | --- | --- | --- |
  | 0 to 25 | < 0.001 / < 0.001 | same | same |
  | 50 | 0.251 / 0.229 | 0.305 / 0.180 | 0.121 / 0.128 |
  | 75 | 0.744 / 0.581 | 0.752 / 0.505 | 0.577 / 0.427 |
  | 100 | 0.802 / 0.617 | 0.854 / 0.546 | 0.616 / 0.443 |

  The step-0 tool-using hazard reads 0.000 where the `" answer"` seed gave 0.16,
  and the token starts at about 1e-8 everywhere.
- Interpretation and limitations: the neutral seed delays learning the close
  token by roughly 35 steps and then reproduces the control, with `p_reclose`
  rising in step with `p_first_close` in every class (at step 8, with both
  still below 1e-7, `p_reclose` was 4 to 6 times `p_first_close`). The
  `" answer"` anchor is not the channel; it only gives the token a head start. Single run per condition; the noise floor from the
  preempted control restart is about 0.01.
- Decision and next action: drop the seed hypothesis as the cause; look at the
  pretraining data and the SFT data.

### 2026-10-25 - Close row only (`replay-10400-closerow`)

- Question or hypothesis and reason for trying it: can the close token's
  LM-head row alone, with the rest of the model frozen, learn the first close?
- Published source revision: `sft_start_replay.py --trainable close-row`.
- Result: all probe values remain below 1e-5 through step 34, where the run
  ended ([`analysis/probe_replay-10400-closerow.csv`](analysis/probe_replay-10400-closerow.csv)).
- Interpretation: the close is not learnable through the output row alone on
  this horizon; body updates are required. No bearing on the re-close question
  beyond ruling out a row-only fix.

### 2026-10-25 - Can bad pretraining rows be diffed out cheaply?

- Question or hypothesis and reason for trying it: swe-zero-12m renders agent
  trajectories whose THOUGHT text sometimes contains stray `</thinking>` tags,
  including close, prose, close before the bash block. If that taught the base
  model to re-close, a counterfactual that removes those rows without redoing
  pretraining would show it.
- Proposed technique: masked counterfactual replay. Export the last N cooldown
  steps' batches with the existing replay format (per-token loss weights and
  per-document segment ids), zero the loss weight on the suspect rows (the
  mixture resolves every sequence to its source cell), replay from the
  checkpoint N steps before the base, then run the 30-step SFT replay head and
  compare probe curves with the control. Per-document segment masks mean a
  masked document never enters other documents' attention.
- Checkpoint cadence: the final 65K cooldown
  (`moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step141k-a30ef8`,
  launcher on Marin branch `june_tpu_67b_a2b`) kept permanent checkpoints every
  1,500 steps: 142,500 through 154,500, then 156,000. Hourly temporary saves did
  not survive. The smallest exact window is therefore N = 1,500, about 100B
  tokens.
- Decision: before costing a 1,500-step replay, count the suspect rows.

### 2026-10-25 - swe-zero's share of the cooldown tail

- Method: the cooldown samples 200 datakit cluster x quality cells from
  `gs://marin-us-central2/datakit/store_8ac06c74` with the phase-1 weights in
  `experiments/june_tpu_67b_a2b/moe/launch_datakit_moe_mix.py` (identical to the
  `june_tpu_67b_a2b` branch's copy; the `tail` concat component is spread over
  its cells by token count). swe-zero's per-document cluster and quality come
  from `datakit/cluster_assign/swe-zero-12m_e217479f` and
  `datakit/quality/swe-zero-12m_53ff59eb` (every fifth of 470 shards, 2.4M
  documents); token length per cell from one tokenized shard (mean 8,839
  tokens per trajectory). The store writes each cell as parts that stripe every
  source across all parts, so a cell's swe-zero share is uniform over its
  shuffle windows and the read position does not matter.
  Scripts: [`analysis/swe_cells.py`](analysis/swe_cells.py),
  [`analysis/swe_kept.py`](analysis/swe_kept.py),
  [`analysis/expected_rows.py`](analysis/expected_rows.py); outputs
  [`analysis/swe_cells.json`](analysis/swe_cells.json),
  [`analysis/swe_cells_kept.json`](analysis/swe_cells_kept.json).
- Result before dedup: swe-zero sits almost entirely in cluster 1 (c01q3 52B
  tokens, c01q4 31B, c01q2 19B, c01q1 3.4B); the reconstructed total is 106.8B
  against the registry's 106.91B. Share of phase-1 training tokens 3.96%, or
  about 30,000 trajectories in the last 100 steps.
- Result after dedup: the store's fuzzy-dedup attributes
  (`datakit/dedup_dabe67c2/outputs/source_112`, swe-zero is not on the exempt
  list) place 96% of swe-zero trajectories in near-duplicate clusters with 0.37%
  of those canonical, so the store kept about 4.5% of the source (kept fraction
  per cell 3.4% to 6.2%). Post-dedup: 4.8B swe-zero tokens in the store, 0.18%
  of training tokens, about 14 trajectories (120k tokens) per cooldown step.

  | Window | swe-zero trajectories | swe-zero tokens |
  | --- | --- | --- |
  | Last 100 steps | ~1,400 | ~12M (0.18%) |
  | Whole 15,000-step cooldown | ~206,000 | ~1.8B |

- Limitations: token lengths per cell from one shard; the dedup output files
  carry an October 2026 rewrite date while the store was built in May 2026,
  though the store's own counters record the 3.36B dedup drops.
- Decision: a 100-step masked replay removes under 0.1% of the model's lifetime
  swe-zero exposure at a learning rate near the cooldown floor; it cannot move
  the probe above the noise floor. Consider raising the dose instead
  (suspect source at 10% versus a repaired copy of the same documents).

### 2026-10-26 - How much of swe-zero actually carries the defect

- Method: decode one tokenized shard (25,743 trajectories), join to the dedup
  attributes, and count thinking-close tags per assistant turn
  ([`analysis/defect_rate.py`](analysis/defect_rate.py),
  [`analysis/defect_rate.out`](analysis/defect_rate.out)).
- Result: every close in swe-zero is stray (no opening tags), but only 0.8% of
  trajectories and 0.8% of turns contain one (kept subset 0.6% / 0.5%). Turns
  with two or more closes, or a close followed by prose, are 0.01% of turns.
- Interpretation: a repaired-copy arm against verbatim swe-zero at 10% mixing
  would differ on about 140 documents over 30 steps, two of them with the
  re-close shape. swe-zero is not a plausible carrier of the pattern.
- Decision: scan every store source for the pattern before designing any
  further pretraining-side arm.

### 2026-10-26 - Thinking-close tags across all 114 store sources

- Method: first 1,500 documents of one tokenized shard per source
  ([`analysis/scan_sources.py`](analysis/scan_sources.py),
  [`analysis/scan_sources.json`](analysis/scan_sources.json)), then a
  per-response rescan of the nine sources with any tag, counting a close
  followed by at least 20 characters of prose and a second close with no
  re-open inside one assistant response
  ([`analysis/scan_refined.py`](analysis/scan_refined.py),
  [`analysis/scan_refined.json`](analysis/scan_refined.json)).
- Result: 105 of 114 sources have no thinking-close tag in the sample.

  | Source | Registry tokens | Docs with any close | Opens / closes | Docs with stray close | Docs with close, prose, close |
  | --- | --- | --- | --- | --- | --- |
  | gpt-oss-rollouts | 3.2B | 100% | 1500 / 1500 | 0.07% | 0 |
  | nemotron-terminal | 6.1B | 100% | 12,757 / 12,758 | 4.9% | 0.07% |
  | nemotron_specialized/infinibyte_reasoning | 18.7B | 78% | 1,169 / 1,169 | 0.8% | 0 |
  | nemotron_sft/sft_math | 199.9B | 5.5% | 274 / 885 | 0.07% | 0.07% |
  | nemotron_sft/sft_code | 56.7B | 5.2% | 78 / 78 | 0 | 0 |
  | nemotron_sft/sft_general | 85.2B | 5.0% | 75 / 75 | 0 | 0 |
  | davinci-dev/env-native | 2.6B | 2.3% | 0 / 63 | 2.3% | 0 |
  | swe-zero-12m | 106.9B raw, 4.8B kept | 1.3% | 0 / 281 | 1.3% | 0.07% |
  | davinci-dev/ctx-native | 57.6B | 0.3% | 37 / 32 | 0.27% | 0.20% |

  Examples pulled by the scan: nemotron-terminal has the textbook case (a
  well-formed think block closes, the answer begins, "Wait, I need to see more
  context", pages of reasoning, a second `</think>`, then the JSON action);
  swe-zero's case is stray `</thinking>`, prose, `</thinking>`, bash;
  sft_math's stray closes are degenerate repeated fragments; ctx-native's hits
  are code that strips think tags.
- Interpretation and limitations: no source carries the close, continue, close
  shape above about one document in 500, a few thousand documents in a 23T-token
  store. The large well-formed sources teach that prose follows a close and that
  reasoning-like prose ends in a close, both correct. Samples are the first
  1,500 documents of one shard per source.
- Decision: no pretraining data intervention is dense enough to test or to
  apply; drop masked replay, repaired copies, and source-swap arms.

### 2026-11-04 - Interpretation and correction

- Correction: an earlier comparison in this investigation set the base model's
  3.0% few-shot re-close rate against the SFT probe hazard (0.35 to 0.47), which
  is conditional on a lead-in. The like-for-like free-sampling numbers are base
  3.0%, 09.21 2.5% / 4.9%, 10.05 0.6% / 3.2%. SFT does not amplify the
  behavior; it inherits a roughly 3% rate from the base and leaves it there.
- Interpretation: pretraining holds the tendency, but as a generalization, not
  as recitation: the base produces the shape at 20 to 30 times its frequency in
  the data. There are no removable rows. The SFT recipe shows 1.38M single-close
  turns and no counterexample, so it preserves the rate rather than reducing it.
  The SFT format does include `<|start_think|>`, so a depth cue exists; both
  special tokens are freshly seeded rows and the model has not learned to use
  the pair.
- Decision and next action: the lever is SFT data. Add turns where text follows
  `<|end_think|>` and the turn proceeds to the tool call or answer without a
  second close, with loss on everything after the close; mix them into the
  exported replay batches and run 30 to 50 steps against the control curve in
  this notebook. Marin #10850's later replay with the #10519 reasoning-span
  format reached the same conclusion from the renderer side.

### 2026-11-05 - Plan: does the base's plain-text close mark the SFT close positions?

- Question or hypothesis and reason for trying it: the base never saw
  `<|end_think|>`, but it saw plain `<think>`/`</think>` blocks. If the step 13
  to 19 re-close spike reuses a base "close tag here" feature, then (1) the base's
  p(`</think>`) at the probe positions should rank trajectories the way the
  replay's p(`<|end_think|>`) does at step 15, beyond a generic end-of-segment
  signal; and (2) seeding the close row from `</` should move the spike.
- Experiment 1 (no training):
  [`analysis/vllm_logprobs/base_tag_logprobs.py`](analysis/vllm_logprobs/base_tag_logprobs.py)
  re-renders the 3,255 probe trajectories as pretraining renders chat
  (`<role>\ncontent\n</role>`, reasoning in `<think>`) and has the base
  (`open-athena/snowball-67b-a2b-base-262k-qk175-skew8` at `058ecaf2`) score
  text-level p(`</think>`), p(`\n</think>`), p(`</thinking>`), p(`\n</thinking>`)
  and the control p(`\n</assistant>`) at the first close, 10.05's re-close, up to
  16 sentence or line ends and 4 other positions per continuation, stopping at
  the first tool-call token (13 of 604 re-closes fall after one and are dropped).
  The scores are compared with the `replay-10400-base300e` and
  `replay-10400-meanseed` per-trajectory dumps by
  [`analysis/base_tag_compare.py`](analysis/base_tag_compare.py)
  (within group x lead-in Spearman, plus a partial correlation with the control
  regressed out).
- Experiment 2: `sft_start_replay.py --close-seed lt-slash`, 30 steps, otherwise
  identical to `replay-10400-base300e`. `</` (id 524) is the first token of both
  plain tags. 2,850 of the 3,255 first closes follow a `.` token, and BPE merges a
  period with the tag into `.</`, so the `</` row should start low at most close
  positions. Step 0 of this run is a direct reading of the base's `</` logit in
  SFT format.
- Expected readings: experiment 1 supports the shared-feature reading if the
  partial correlation with step 15 `p_reclose` or `hazard` is clearly positive
  (above about 0.2). Experiment 2 supports it if the hazard peak rises or arrives
  earlier than with the `" answer"` seed (SWE 0.14 at step 19, tool-using 0.22 at
  step 15).

### 2026-11-05 - `</`-seeded close row, 20 steps (`replay-10400-ltslash`)

- Published source revision: `876ac0c3b9`, `sft_start_replay.py --close-seed
  lt-slash --steps 30 --nodes 16 --context-axis-size 4 --data-root
  s3://marin-rno2a/tmp/ttl=30d/dml/sft_replay_10400`, Iris job
  `/dml/replay-10400-ltslash-coord` on `cw-rno2a`. Cancelled after step 20 was
  scored, since the control's no-lead-in peak ends by step 19.
- External inputs and artifact references: probe output at
  `s3://marin-rno2a/tmp/ttl=30d/dml/sft_replay_10400/reclose/replay-10400-ltslash/step-*.jsonl`
  (30-day TTL). Per-step cell means for both runs, recomputed from the
  per-trajectory dumps by
  [`analysis/probe_compare_runs.py`](analysis/probe_compare_runs.py), are in
  [`analysis/probe_ltslash_vs_answer.csv`](analysis/probe_ltslash_vs_answer.csv).
- Result: `p_first_close` / `p_reclose`, `" answer"` seed versus `</` seed.

  | Step | SWE | Tool-using | Plain |
  | --- | --- | --- | --- |
  | 10 | 0.030 / 0.065 vs 0.226 / 0.064 | 0.201 / 0.134 vs 0.181 / 0.042 | 0.105 / 0.163 vs 0.091 / 0.048 |
  | 12 | 0.109 / 0.188 vs 0.479 / 0.206 | 0.396 / 0.266 vs 0.470 / 0.149 | 0.216 / 0.279 vs 0.251 / 0.146 |
  | 15 | 0.351 / 0.330 vs 0.564 / 0.306 | 0.555 / 0.395 vs 0.597 / 0.258 | 0.333 / 0.380 vs 0.380 / 0.223 |
  | 20 | 0.578 / 0.374 vs 0.566 / 0.289 | 0.562 / 0.411 vs 0.574 / 0.251 | 0.313 / 0.374 vs 0.348 / 0.187 |

  No-lead-in hazard at step 20: SWE 0.132 vs 0.130, tool-using 0.140 vs 0.080,
  plain 0.027 vs 0.007. The `</` arm's hazard starts high (SWE 0.43 at step 0)
  because it sums p over every continuation token, and SWE continuations contain
  literal closing tags where `</` is the next token. The hazard is not comparable
  across these seeds before about step 15.
- Interpretation and limitations: seeding the close row from the base's
  closing-tag row speeds up the first close (SWE 0.23 against 0.03 at step 10)
  and does not speed up the re-close. At step 20, with the first close at or
  above the control, `p_reclose` is 23% (SWE), 39% (tool-using) and 50% (plain)
  lower. This does not support the re-close being inherited through the base's
  closing-tag prediction. One run per arm; the differences are 8 to 19 times
  the 0.01 noise floor. Steps 21 onward were not run, so whether the gap
  persists to the 0.4 plateau the control reaches is untested.
- Decision and next action: if the gap matters for the recipe, extend this arm
  to 50 to 100 steps and compare plateaus. Experiment 1 (base plain-tag scoring)
  is still running.

## Final decision

- Selected approach and evidence: pending the SFT-side counterexample
  experiment. Evidence so far rules out the `" answer"` seed (mean-seed run) and
  any identifiable pretraining rows (share, dedup, defect-rate and source-scan
  entries above).
- Rejected alternatives and why: masked counterfactual replay of the cooldown
  tail (smallest exact window 1,500 steps; suspect rows are 0.18% of tokens and
  carry the shape in 0.01% of turns); repaired-copy and source-swap arms (no
  source dense enough); close-row-only training (does not learn the close).
- Marin contribution PRs: none yet. Candidates on this branch: the
  context-axis weight sharding in `experiments/june_tpu_67b_a2b/moe/model.py`
  and the replay/probe tooling under `experiments/grug_sft/`.
- Reproduction instructions: see the entries' scripts; replay launches use
  `experiments/grug_sft/sft_start_replay.py` on 16 us-east-02a nodes with the
  batches in the us-east-02a bucket; the datakit analyses read GCS us-central2
  metadata and sampled shards only.
- Remaining uncertainty: single runs per replay condition; probe set drawn from
  10.05's own trajectories; source scan samples one shard per source.
