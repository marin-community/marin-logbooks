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

### 2026-11-06 - Base plain-text close at the probe positions (`reclose27-base-tags`)

- Published source revision: `876ac0c3b9`,
  `analysis/vllm_logprobs/base_tag_logprobs.py --trajectories
  s3://marin-rno2a/tmp/ttl=30d/dml/sft_replay_10400/trajectories.jsonl --output
  .../logprobs/base_tags.json`, Iris job `/dml/reclose27-base-tags` on `cw-rno2a`
  (one coordinator retry, cause not logged). The forced-token shortcut disagreed
  with prompt logprobs by up to 0.99 nats on the validation sample, so the job
  scored every token with prompt logprobs, the path that matched Levanter
  earlier.
- External inputs and artifact references: output
  `s3://marin-rno2a/tmp/ttl=30d/dml/sft_replay_10400/logprobs/base_tags.json`
  (30-day TTL); correlations in
  [`analysis/base_tags.compare.json`](analysis/base_tags.compare.json) from
  [`analysis/base_tag_compare.py`](analysis/base_tag_compare.py).
- Result: within-cell Spearman between the base's p(`</think>`) and the
  replay's p(`<|end_think|>`), raw / with the `\n</assistant>` control regressed
  out (control alone in brackets).

  | Replay checkpoint | First close | Re-close (n = 588) | Hazard |
  | --- | --- | --- | --- |
  | `" answer"` step 10 | -0.02 / 0.06 (-0.10) | 0.41 / 0.38 (0.21) | 0.14 / 0.14 |
  | `" answer"` step 15 | 0.08 / 0.19 (-0.07) | 0.41 / 0.40 (0.18) | 0.14 / 0.18 |
  | `" answer"` step 20 | 0.39 / 0.30 (0.25) | 0.50 / 0.47 (0.25) | 0.15 / 0.18 |
  | `" answer"` step 100 | 0.48 / 0.35 (0.34) | 0.56 / 0.48 (0.34) | 0.05 / 0.07 |
  | mean seed step 60 | 0.54 / 0.35 (0.42) | 0.59 / 0.54 (0.34) | 0.18 / 0.19 |
  | `</` seed step 20 | 0.17 / -0.00 (0.14) | 0.35 / 0.40 (0.08) | 0.12 / 0.17 |

  Median base p(`</think>` or `\n</think>`) at 10.05's re-close versus its first
  close: SWE 0.084 vs 0.0069, tool-using 0.030 vs 5.0e-6, plain 0.033 vs 0.0054
  (lead-in trajectories, which hold 585 of the 588 re-closes).
- Interpretation and limitations: the base already marks the re-close
  positions. Its plain-text close ranks trajectories the way the replay's
  close token does at the re-close from step 10, before the replay has learned
  the first close, and the ranking survives regressing out a generic
  end-of-turn signal and appears under all three seeds. In plain text the base
  rates 10.05's re-close position as a likelier end of a think block than
  10.05's own first close. The `</`-seed result above shows this information
  does not travel through the base's closing-tag output row; it is read from
  the base's hidden states whichever way the row is seeded. Limitations: 0.8%
  of position estimates (349 of 42,255) exceed 1, up to 63, where the
  divided-out boundary token was itself improbable, so means are unreliable
  (medians and ranks are used above); rankings are across trajectories
  sampled by 10.05, so trajectory-level traits both models react to (lead-in
  type, continuation length) could contribute beyond the end-of-turn control.
- Decision and next action: pretraining primes where a re-close goes; SFT
  supplies the token that acts on it. That keeps SFT counterexample data as
  the lever. The 100-step `</`-seed run (`replay-10400-ltslash100`) tests
  whether that seed also lowers the re-close plateau.

### 2026-11-06 - When the replay learns to close, and to close twice

- Method: [`analysis/hazard_curves.py`](analysis/hazard_curves.py) turns the
  per-trajectory probe dumps of the three seeds into per-step probabilities,
  given `<|start_think|>`, on the fixed 10.05 trajectories: on time (the first
  close lands where 10.05 closed, `(1 - p_premature)^L * p_first_close`), early
  (`1 - (1 - p_premature)^L`), and double (on time, then again in 10.05's
  continuation: on time x hazard). The dumps keep only the mean close
  probability over the reasoning, so early is a lower bound and on time an
  upper bound. Output:
  [`analysis/hazard_curves.csv`](analysis/hazard_curves.csv),
  [`analysis/hazard_curves.png`](analysis/hazard_curves.png).
- Result: steps where each no-lead-in curve first reaches 10% / 50% of its own
  peak over 100 steps.

  | Seed | Group | On-time close | Double close | Double peak (step) |
  | --- | --- | --- | --- | --- |
  | `" answer"` | SWE | 12 / 17 | 13 / 16 | 0.101 (29) |
  | `" answer"` | Tool-using | 10 / 20 | 10 / 12 | 0.048 (17) |
  | `" answer"` | Plain | 11 / 32 | 9 / 12 | 0.011 (15) |
  | mean | SWE | 47 / 52 | 50 / 54 | 0.056 (61) |
  | mean | Tool-using | 45 / 51 | 50 / 54 | 0.010 (60) |
  | mean | Plain | 48 / 55 | 52 / 84 | 0.001 (100) |

  With the `</` seed (20 steps) SWE double close peaks at 0.183 at step 13,
  inflated by literal closing tags in SWE continuations where the `</`-seeded
  row fires; tool-using and plain peak at 0.041 and 0.007.
- Interpretation: the model learns to double-close at the same time it learns
  to close, within 0 to 5 steps at the 10% onset under both 100-step seeds,
  and the mean seed delays both together by about 37 steps. Without a lead-in,
  double close then falls (SWE 0.101 to 0.016 by step 100 with the `" answer"`
  seed) while the on-time close keeps rising. With a forced lead-in it
  plateaus (SWE about 0.5, tool-using 0.17, plain 0.05). Early closing with the
  `" answer"` seed starts at 0.43 (tool-using) and 0.72 (plain) at step 0,
  because the seed copies `" answer"`'s score at every reasoning token, so
  "P(at least one close)" is not a usable learning signal for that seed.

### 2026-11-06 - Correction: close counts at any position, not 10.05's close point

- Why: the previous entry's "on time" and "double" metrics assume 10.05's first
  close is the right place to close. That position is itself in question, so
  this entry counts closes at any position: P(thinking closed at least once |
  opened) and P(at least twice | opened), from the first reasoning token through
  the end of the scored text (10.05's re-close, or the trajectory end), with
  each position an independent chance to emit `<|end_think|>` along the fixed
  10.05 text. It supersedes the previous entry's metrics; that entry's own
  numbers are unchanged.
- Method: [`analysis/hazard_curves.py`](analysis/hazard_curves.py) at the commit
  recording this entry. The existing dumps keep only segment summaries, so each
  multi-position segment is estimated as a binomial at its mean. That
  overstates no-close, so P(at least one) is a lower bound, and the error on
  P(at least two) has no fixed sign. `reclose_eval.py` now writes exact
  `p_ge1` and `p_ge2` (commit `9893fb5798`, tested against enumeration in
  `tests/grug/test_reclose_close_counts.py`). The 100-step `</`-seed run was
  restarted as `replay-10400-ltslash100e` with that probe, so it gives exact
  curves and measures the estimate's error.
- Result, no lead-in, P(at least one) / P(at least two):

  | Step | Mean seed SWE | Mean seed tool-using | Mean seed plain | `" answer"` SWE | `" answer"` tool-using | `" answer"` plain |
  | --- | --- | --- | --- | --- | --- | --- |
  | 0 | 0.00 / 0.000 | 0.00 / 0.000 | 0.00 / 0.000 | 0.00 / 0.000 | 0.44 / 0.214 | 0.74 / 0.464 |
  | 15 | 0.00 / 0.000 | 0.00 / 0.000 | 0.00 / 0.000 | 0.44 / 0.061 | 0.87 / 0.605 | 0.68 / 0.328 |
  | 45 | 0.05 / 0.001 | 0.10 / 0.003 | 0.02 / 0.000 | 0.89 / 0.081 | 0.77 / 0.259 | 0.48 / 0.077 |
  | 50 | 0.28 / 0.017 | 0.36 / 0.036 | 0.13 / 0.002 | 0.80 / 0.058 | 0.73 / 0.209 | 0.43 / 0.058 |
  | 55 | 0.64 / 0.064 | 0.67 / 0.136 | 0.33 / 0.008 | 0.76 / 0.053 | 0.78 / 0.243 | 0.46 / 0.058 |
  | 100 | 0.81 / 0.042 | 0.89 / 0.302 | 0.62 / 0.028 | 0.85 / 0.036 | 0.87 / 0.294 | 0.58 / 0.041 |

  First step with P(at least one) >= 0.5 (k) and P(at least two) >= 0.05 (j),
  mean seed: SWE 53 / 54, tool-using 53 / 51, plain 67 / not reached.
- Interpretation and limitations: the mean seed is the clean timing test,
  because its close token starts near 1e-8 everywhere. There, closing twice
  appears in the same steps as closing at all (SWE and tool-using cross both
  lines within three steps of each other). The `" answer"` and `</` seeds start
  with closes already likely (tool-using 0.44 and plain 0.74 for `" answer"`;
  SWE 0.44 for `</`, from literal closing tags), so k and j are not meaningful
  for them. Counting early closes changes the plateau: by step 100, tool-using
  P(at least two) is about 0.3 under both 100-step seeds, mostly a close inside
  the reasoning followed by another, where the on-time metric gave 0.006 to
  0.012. These are teacher-forced along 10.05's text and estimated from
  segment means, so magnitudes wait on the exact run.

### 2026-11-07 - Exact 100-step close curves, `" answer"` vs `</` seed

- Runs: `replay-10400-answer100e` and `replay-10400-ltslash100e` on cw-rno2a at
  commit 9893fb5798. They use the same exported 10.05 batches in the same order,
  the same LR and the same special-token row reset; the only difference is the
  seed for the `<|end_think|>` LM-head row. Both training sub-jobs succeeded and
  wrote probes for steps 0 to 100. The parent jobs report `failed` only because
  the `" answer"` run's restarts (attempts 1 to 4, coordinator connection
  errors) used up the job's allowed task failures.
- Metrics: exact P(closed at least once | opened) and the hazard after 10.05's
  close (P(closes again | a close is in the text)), with no lead-in,
  teacher-forced on the 3,255 fixed 10.05 trajectories.
  [`analysis/hazard_curves.py`](analysis/hazard_curves.py) writes
  [`hazard_curves.png`](analysis/hazard_curves.png) and
  [`hazard_curves.csv`](analysis/hazard_curves.csv). The reference levels are
  the in-training probe on the 10.05 endpoint and vLLM top-20 lower bounds on
  09.21.
- Results at step 100, P(at least once) / hazard:

  | group | `" answer"` | `</` | 10.05 | 09.21 |
  |---|---|---|---|---|
  | SWE | 0.88 / 0.020 | 0.84 / 0.012 | 0.97 / 0.0008 | 0.99 / 0.0040 |
  | tool-using | 0.89 / 0.020 | 0.89 / 0.012 | 0.86 / 0.0065 | 0.84 / 0.0048 |
  | plain | 0.59 / 0.0019 | 0.64 / 0.0013 | 0.91 / 0.0001 | 0.84 / 0.0006 |

- When the replay learns to close: tool-using reaches both reference levels by
  step 80 under both seeds. SWE and plain do not reach either level in 100
  steps. The peak near step 13 is the early-close transient, not learned
  closing.
- When it learns to close twice: under the `" answer"` seed, the SWE hazard
  starts at 1.5e-4, below both references. It passes 10.05's level at step 4
  and 09.21's at step 7, peaks near 0.14 at step 18, then decays slowly. The `</`
  seed starts with the hazard already high (0.44 on SWE), so it gives no timing
  signal. By step 100 the two seeds differ by about 1.7x on SWE and tool-using;
  both remain 15 to 25x above 10.05 on SWE. The only reference crossing is
  `</` plain at step 57 (09.21's level), and it drifts back up to about 2x
  that level by step 100.
- Interpretation and limitations: double-closing appears within the first 15
  steps and SFT trains it back out slowly. That is consistent with the issue's
  observation at step 15. The early spike in the hazard is partly an artifact:
  the probe forces a close token into the text before the model has learned
  to produce one, and the model may then copy it. Treat "double-closing before
  closing" as unconfirmed. Single run per seed, teacher-forced on 10.05's text,
  no sampled check.

### 2026-11-07 - What the replay batches teach about closing

- Question: does the 10.05 SFT data contain double-close examples, and do
  batch-to-batch probe changes track batch content?
- Method: the 100 exported replay batches (copied locally from the cw-rno2a
  replay bucket, 5.7 GB). [`analysis/batch_scan/scan_batches.py`](analysis/batch_scan/scan_batches.py)
  walks every special-token assistant turn with loss and counts open/close
  patterns and plain-text tags.
  [`source_mix.py`](analysis/batch_scan/source_mix.py) labels each packed
  segment by source.
  [`batch_effects.py`](analysis/batch_scan/batch_effects.py) and
  [`mix_effects.py`](analysis/batch_scan/mix_effects.py) take Spearman
  correlations between per-batch features and the probe change from step k to
  k+1, averaged over both seeds. Outputs:
  [`batch_scan.jsonl`](analysis/batch_scan/batch_scan.jsonl) and
  [`source_mix.jsonl`](analysis/batch_scan/source_mix.jsonl).
- Results:
  - Across the 100 steps, 1,148,142 trained assistant turns think. 148 re-open
    thinking after a close and close again. None close twice without an open in
    between, none close without opening, and 20 never close. There are no
    plain-text `</think>` tags in special-token assistant turns; the 3,011
    plain-text tags all sit outside them, in raw-text reasoning.
  - Segments either think on every assistant turn or on none; mixed segments
    are rare (77 of about 26k at step 1).
  - Loss covers every token, including system, user and tool turns. Trained
    tokens by source: thinking chat 52%, thinking chat with tools 16%, no-think
    chat with tools 9%, no-think chat 2%, raw pretraining text 15%,
    continuation fragments 6%. Post-close assistant text is 6 to 9% of trained
    tokens (about 5M per step); thinking is 44 to 56%.
  - The seeds agree on per-step P(at least once) changes after step 30 (r =
    0.77 SWE, 0.94 tool-using, 0.92 plain), so specific batches drive them. They
    agree little on hazard changes (r = 0.03 to 0.37).
  - No source share or post-close share predicts the shared changes: |rho| <=
    0.28 over 90 steps, and <= 0.26 over steps 30 to 99, across 84 tests.
- Interpretation: the data contains no examples of a second close. The rise in
  the hazard comes from generalization (base priming plus copying the close
  token), not from imitating examples. The only pressure against it is
  ordinary loss on the 5M post-close tokens per step. That push is small where
  the model rarely wants to re-close, and it is strongest exactly where the
  model does want to re-close, positions this data rarely contains. That fits
  the slow decay. There are no batches or sources to remove.
- Next: test direct pressure in the replay harness, for example an
  unlikelihood term on `<|end_think|>` at every position after a turn's first
  close, or 10.05 re-close trajectories with the second close deleted. At
  inference, a decoding rule that blocks `<|end_think|>` after the first close
  removes the symptom without retraining.

### 2026-11-10 - Unlikelihood on a re-close, weights 10 and 100 (`replay-10400-answer-ulwide10`, `-ulwide100`)

- Question: does direct pressure against `<|end_think|>` after a close stop the
  double close without stopping the first close?
- Method: the `" answer"`-seeded 30-step replay with
  `--close-unlikelihood-weight W`, which adds `W * -log(1 - p(close))` at every
  trained position from a close until the next `<|start_think|>` or document
  start, across tool calls, tool responses and turn headers (commit
  b4bfa309d3). Weights 10 and 100; the plain `answer100e` replay is the
  control. All three use the in-loop teacher-forced probe, so the numbers
  compare directly (they are not on the scale of `hazard_curves.py`).
  [`analysis/unlikelihood/matched_reclose.py`](analysis/unlikelihood/matched_reclose.py)
  compares each penalized step to the control's hazard at the same P(at least
  one close), read from the control's rising phase. That separates fewer
  re-closes from learning to close later. Series:
  [`none_inloop.tsv`](analysis/unlikelihood/none_inloop.tsv),
  [`ulwide10_inloop.tsv`](analysis/unlikelihood/ulwide10_inloop.tsv),
  [`ulwide100_inloop.tsv`](analysis/unlikelihood/ulwide100_inloop.tsv); output
  [`matched_reclose.out`](analysis/unlikelihood/matched_reclose.out).
- Result at step 30:

  | run | SWE P(>=1) | SWE hazard | SWE hazard / control at same P(>=1) | tool P(>=1) | tool hazard | tool ratio | SWE p_first_close |
  |---|---|---|---|---|---|---|---|
  | control | 0.92 | 0.53 | 1 | 0.81 | 0.26 | 1 | 0.82 |
  | W = 10 | 0.86 | 0.41 | 0.84 | 0.78 | 0.19 | 0.71 | 0.75 |
  | W = 100 | 0.63 | 0.21 | 0.51 | 0.64 | 0.094 | 0.50 | 0.53 |

  - W = 10 tracks the control for 15 steps (ratio 0.93 to 1.01), then reaches
    about 0.8. Its gradient on the close logit is about `W * p`, and p is
    about 0.003 on the penalized span, so it barely acts until a re-close is
    already likely.
  - W = 100 halves the hazard at equal closing by step 25 (ratio 0.5 to 0.6,
    both groups). It also slows learning to close: SWE P(>=1) is 0.63 at step
    30, a level the control passed at step 12, and the probability of closing
    at the trained position falls from 0.82 to 0.53. The suppression is not
    confined to the post-close span; pushing the close logit down there also
    lowers it at the first close.
  - At W = 100 the SWE hazard held near 0.15 from step 15 to 26 while P(>=1)
    rose from 0.25 to 0.45. That is the target separation, but it began to
    rise with closing after step 26.
  - Both runs took about 3 minutes per step. The W = 100 run was preempted
    three times on cw-rno2a at interactive priority. Its restart from step 0
    reproduced steps 0 to 3 to within 0.0002, and it resumed the last attempt
    from the step-22 checkpoint.
- Interpretation: the unlikelihood term works in the right direction, but a
  single weight trades re-close suppression against learning the first close.
  The `-log(1 - p)` form is weakest where the model rarely re-closes and
  strongest where closing is the right answer one position away, so a larger
  weight mostly delays closing. Single seed per arm; 30 steps.
- Next: target the second close without touching the first. Options: a
  log-odds hinge on the post-close span (penalize only while the close logit
  is above a margin below the best alternative); a matching positive term on
  the first close to hold it in place; or a weight warm-up that starts after
  the model has learned to close (around step 15).

### 2026-11-13 - Trough-to-peak learning steps for closing and closing twice

- Question: at which step does each replay learn to close, and to close again,
  once each curve is measured over its own rise?
- Method: [`analysis/learning_steps/trough_peak.py`](analysis/learning_steps/trough_peak.py)
  rescales each per-group series from its trough (0) to its peak (1), over the
  pair of steps with the largest rise, and reports the steps where it first
  reaches 10%, 50% and 90%. Inputs: the old per-step probe CSVs
  (`p_first_close`, `p_reclose`), the segment-mean any-position estimate, and
  the exact 100-step curves, bundled in
  [`page_data.json`](analysis/learning_steps/page_data.json). Output:
  [`trough_peak.out`](analysis/learning_steps/trough_peak.out).
- Result, 50% step for close / close again:

  | start, metric | SWE | tool-using | plain |
  |---|---|---|---|
  | `" answer"`, at 10.05's close and re-close points | 18 / 14 | 13 / 12 | 14 / 11 |
  | mean row, same | 53 / 52 | 53 / 53 | 55 / 54 |
  | mean row, P(>=1) / P(>=2) | 52 / 53 | 51 / 75 | 55 / 61 |
  | `" answer"`, exact P(>=1) / hazard | 16 / 13 | 10 / 11 | not usable |

  The tool-using P(>=2) curve under the mean row keeps climbing slowly to step
  100; its 10% step (49) matches closing (45). Curves that start near their
  peak (`" answer"` plain P(>=1), the `</` start's hazard) have no usable rise.
- Interpretation: under every start, the re-close is half learned within about
  four steps of the close, often slightly before it. The start moves both
  together. That fits the base model already marking where a close belongs.
  Single run per start.

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
