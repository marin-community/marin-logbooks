---
title: "Mochi SFT comparison: base, full SFT, and V5 SFT on Micro210"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/83"
related_issues:
  - "https://github.com/marin-community/marin/issues/11337"
research_branch: "codex/83-mochi-sft-comparison"
marin_base_revision: "cc3706eb2cdb5e1fec695bc4bfb6582177af82ec"
archive_template_revision: "fc1cb30da0967f6ab78ffd47444ed9144f50cba9"
status: active
---

# Mochi SFT comparison: base, full SFT, and V5 SFT on Micro210

## Question and decision criteria

- Goal and why it matters: archive a like-for-like comparison of two supervised
  fine-tuning recipes that both start from the Mochi base checkpoint at step
  393491 — the production eleven-rack full-SFT run and the 128-GPU V5
  long-loop arm — with their configs, data mixes, and Micro210 eval results
  (Micro210 is Marin's 20-benchmark policy eval suite; the date in a policy
  name, such as 2026.11.23, identifies the sealed prompt and grading table).
  This record consolidates numbers that were previously spread across eval
  comments in [marin#11337](https://github.com/marin-community/marin/issues/11337).
- Baseline and alternatives: Mochi base step 393491. Alternatives compared:
  full SFT (`hero-sft-fullab-step393k-v2`, step 400407) and V5 SFT
  (`long-loop-mochi-sft-exp3-v5-bankadam-128`, step 2100).
- Evidence needed to choose: Micro210 policy evals with scored counts, plus the
  exact configs and per-component data mixes of both arms.
- Constraints and unresolved questions: record-only task — no new runs, no
  Marin PRs. Training data stays in object storage; only manifests, configs,
  and run metadata are copied here.

## Baseline

- Date: checkpoint used by both arms; evals run 2026-11-24/25.
- Published source revision: eval runtime `5ca48a7e`, frozen Micro210 policy
  `9d850b9f` (sealed table `2280f01a`) per each `record.json` provenance.
- Configuration, command, and external inputs: Mochi base step 393491, served
  from `s3://marin-us-east-08a/marin/users/benfeuer/mochi-bringup/models/mochi-base-step393491/2026.11.17.1/hf`.
- Environment and hardware when relevant: Micro210 evals on CoreWeave GB200s
  (16 task instances), Mochi GB200 serving profile.
- Result and evidence: Micro210 2026.11.23 policy pass@1 **0.2607** (110/422
  scored attempts; 422/434 rows scored, 12 infra trials absent) —
  [EvalDash](https://marina.oa.dev/evaldash/runs/mochi-base-step393491-micro210-2026.11.23),
  [record.json](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fevals%2Fmochi-base-step393491-micro210-2026.11.23%2Frecord.json),
  [issue post](https://github.com/marin-community/marin/issues/11337#issuecomment-7164731328).
  On the older 2026.11.20-together policy: **0.2477** pooled scored-attempt
  mean reward (429/440 valid) —
  [EvalDash](https://marina.oa.dev/evaldash/runs/mochi-base-step393491-micro210-2026.11.20-together),
  [record.json](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fevals%2Fmochi-base-step393491-micro210-2026.11.20-together%2Frecord.json),
  [issue post](https://github.com/marin-community/marin/issues/11337#issuecomment-7057267101).
- Limitations: single policy suite (Micro210, 20 benchmarks), single seed; the
  two policies aggregate differently (pooled mean reward vs pass@1) so their
  numbers are not directly comparable to each other.

## Entries

### 2026-11-25 18:45 UTC - Verified every eval number from record.json

- Question or hypothesis and reason for trying it: the comparison table came
  from a human-maintained CSV; before publishing, re-read each eval's
  `record.json` from object storage so every number in this notebook traces to
  the eval record.
- Published source revision: n/a (read-only verification of five
  `record.json` files under `s3://marin-us-east-02a/marin/evals/`).
- Command or material configuration: `uv run fsutil cat --raw
  s3://marin-us-east-02a/marin/evals/<run_id>/record.json` for each of the
  five runs listed below.
- External inputs and artifact references: the five `record.json` files, each
  linked per row in the results entry.
- Result (all five match the CSV exactly):

| Run | Policy | Score | Scored / valid | Verifier mean |
|---|---|---|---|---|
| base step 393491 | 2026.11.20-together | 0.2477 (pooled scored-attempt mean reward) | 429/440 valid | 0.24768 |
| SFT step 400407 | 2026.11.20-together | 0.5973 (pooled scored-attempt mean reward) | 413/440 valid | 0.59735 |
| base step 393491 | 2026.11.23 | 0.2607 pass@1 (110/422) | 422/434 scored | 0.2432 |
| SFT step 400407 | 2026.11.23 | 0.5971 pass@1 (249/417) | 417/434 scored | 0.5810 |
| V5 SFT step 2100 | 2026.11.23 | 0.5819 pass@1 (245/421) | 421/434 scored | 0.5693 |

- Interpretation and limitations: scored-row counts differ per run because
  unscored rows are infrastructure failures (12 absent for base, 17 for full
  SFT, 13 for V5 on the 2026.11.23 policy); the 2026.11.20-together base run
  combines the original .7 dump with an authorized 95-row .9 recovery. The
  policy description strings in `record.json` carry these details.
- Decision and next action: numbers verified; document the two arms' configs
  and data mixes.

### 2026-11-25 18:46 UTC - Full-SFT arm: config and data mix

- Question or hypothesis and reason for trying it: document exactly what
  `hero-sft-fullab-step393k-v2` was configured to train on, so its 0.5971 has
  a reproducible recipe behind it.
- Published source revision: Marin `main` commit
  `dca3b9c0eb486986f49d8407dfc09efe1590f486` ("[sft] Declare Snowball and
  Mochi training presets (#11712)") — the last commit touching all three
  source files, and byte-identical to Marin `main` at this branch's base
  `cc3706eb2cdb5e1fec695bc4bfb6582177af82ec` (verified by diff). Copies live
  in `configs/` of this directory.
- Command or material configuration: `MOCHI_SFT` in
  [`configs/mochi_sft.py`](configs/mochi_sft.py), launched via the
  `mixture-262k-176node` preset (W&B project `marin_moe_sft`, group
  `hero-sft`). Key settings:
  - initialize from `s3://marin-us-east-02a/marin/grug/hero-lce-log65536sq-step390k/2026.08.19.2/checkpoints/step-393491`;
  - Muon and Adam learning rates 3.2905e-5, max grad norm 1.0, weight decay 0,
    constant LR then linear decay to 0.1× over the final 10% of updates
    (692 steps), no warmup;
  - `replay_fraction=0.2035`, `dp_racks=11` (11 racks × 16 sequences/rack =
    176 packed 262,144-token sequences per update), routed-expert capacity
    factor 1.625, router bias frozen, chat special tokens reseeded at the SFT
    start with lr multiplier √32;
  - expert placement from `hero_sft_expert_placement_step393491.json` at the
    same commit; tokenizer `marin-community/marin-tokenizer-tools` pinned at
    `3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09`; recipe `version="2026.11.10"`,
    `source_revision="406dff47f217637ceca499b266989588cb0df5ab"`;
  - `eval_every=500`, periodic checkpoints kept every 500 steps.
- External inputs and artifact references: the SFT store manifest
  [`configs/hero_sft_stores_2026_09_29.json`](configs/hero_sft_stores_2026_09_29.json)
  (CoreWeave copy of `gs://marin-us-central2/grug_sft/stores/full-ab-packed-2026.09.29.json`),
  read by `load_sft_stores` in
  [`configs/launch_hero_sft.py`](configs/launch_hero_sft.py); replay draws from
  the base model's own pretraining data, the hero cooldown mixture over
  Harrier cluster×quality cells with 4× long-context skew.
- Data mix (from `plan_sft_mixture` + `one_epoch_weight_stages`):
  - 216 SFT stores, **968,596 packed sequences** at 262,144 tokens each
    (≈253.9B tokens; "packed-slot" figures count every sequence at its full
    262,144-token length, padding included), all packed with the 2026.09.18
    SFT tokenizer at the hero (262K Mochi pretraining ladder) context.
    Largest families: `nemotron_sft_v3` 64 stores
    / 781,096 sequences (math_v3/train alone 197,234; competitive programming
    115,015; math_v2+v4 130,056; proofs 62,723; finance 38,143; science_v2
    36,270; multilingual; SWE; OpenCode; agentic; ARC-AGI), `penfever-traces`
    130 stores / 16,484, `open_swe_traces` 13 stores / 131,741, plus
    `agenttrove` (19,988), `openthoughts4-code-glm-5.2-n4` (16,792), and ten
    smaller single-store families. The full per-store list is in the copied
    manifest.
  - One epoch spread over **19 mixture blocks of 64,064 sequences**
    (block = 176 × ceil(64,000/176) steps, `SFT_TARGET_BLOCK_SIZE=64,000`);
    19 × 64,064 = 1,217,216 sequences = **6,916 updates**. Each block draws an
    even share of every SFT component (33 stores with fewer than 19 sequences
    share one pooled component) and fills the rest with replay.
  - Replay: 248,620 sequences (≈65.2B packed-slot tokens) drawn from the hero
    cooldown cells by their pretraining weights. Realized replay fraction
    **20.43%** against the 0.2035 target (the plan rounds to whole blocks).
- Result: 6,916 updates, 393,491 → 400,407; ≈**319.1B packed-slot tokens**
  (1,217,216 × 262,144). These are packed-slot upper bounds: padding is not
  subtracted.
- Evidence: copied configs and manifest; run artifacts under
  `s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/`.
- Interpretation and limitations: the store manifest counts packed sequences,
  not measured token bodies, so the SFT-token figure inherits the same upper
  bound. The replay share realized (20.43%) differs slightly from the
  configured 0.2035 because blocks round to whole steps.
- Decision and next action: full-SFT arm documented; document the V5 arm.

### 2026-11-25 18:47 UTC - V5 arm: config and data mix

- Question or hypothesis and reason for trying it: document the Exp3 V5 arm —
  the long-loop prototype's Mochi SFT round — whose config exists only on a
  branch, so the record must name that branch and commit.
- Published source revision: Marin commit `a3d4a7c2b0` ("Add the V5 Mochi SFT
  profile from the BankAdam128 run") on branch `benfeuer/long-loop-prototype`;
  `experiments/post_training/long_loop/rounds/mochi-sft-exp3-v5-bankadam-128.yaml`
  does not exist on Marin `main`. Copy at
  [`configs/mochi-sft-exp3-v5-bankadam-128.yaml`](configs/mochi-sft-exp3-v5-bankadam-128.yaml).
- Command or material configuration: round `long-loop-mochi-sft-exp3-v5-bankadam-128`,
  version `2026.11.24.128`, cluster `cw-us-east-08a`:
  - 128 GB200 GPUs, batch 32 packed sequences × 262,144-token packing cap
    (8,388,608 tokens per update), packing lookahead none;
  - schedule 2,100 updates, warmup 5, decay 210; stopping condition
    `mixture_epochs: 1`; training budget 20-hour clock (40 min finalization
    reserve, 10 min update reserve);
  - initialized from `s3://marin-us-east-08a/tmp/ttl=30d/mochi-base-step393491/step-393491`
    (metadata SHA-256 `07df6c2f942b7f4a86b5d6fc42f1f5f0306a2bb3c3838a0b0616c8297d996267`);
  - hero reference `bankadam128-20261124/reference-dafc25df.json` (SHA-256
    `dafc25df2728ba470e2d82496c552e701d7d449cb7bdded8b2db3b73dfc6ab9e`); data
    seal `release-complete-v5.json` (SHA-256 `3e2161945e1adc60c9edcdb6b0d7395946966dc4e0d16b88660fbc38989603f5`).
- External inputs and artifact references: sealed V5 data release
  [`configs/v5-release-manifest.json`](configs/v5-release-manifest.json)
  (SHA-256 matches the config pin), component list `data.json` and weights
  [`configs/v5-train-weights.json`](configs/v5-train-weights.json) (SHA-256
  `e365fa1d...` matches the manifest pin), epoch capacity
  [`configs/v5-epoch-capacity.json`](configs/v5-epoch-capacity.json) (SHA-256
  matches the config pin), and run output `actual-exposure.json`
  ([`configs/v5-actual-exposure.json`](configs/v5-actual-exposure.json)).
- Data mix (release manifest + capacity, all read as metadata only):
  - **434 components**: 50 SFT sources and 384 pretrain-replay components
    (199 Harrier cluster×quality cells `c00q0-train`…`c39q4-train`, each split
    above/below 64k tokens: `gt_64k`/`lte_64k`). Mixture weights are **0.8
    SFT / 0.2 pretrain replay**, staged at steps 0, 250, and 500.
  - Proposal family shares of the SFT mass
    ([`configs/v5-mixture-rationale.json`](configs/v5-mixture-rationale.json)):
    instruction_chat 0.163, competitive_code 0.14, math_v2_v3 0.15,
    other_math 0.10, recorded_pass_agents 0.15, science 0.03,
    multilingual_v2 0.03, nemotron_swe 0.015, relay 0.01, safety 0.005,
    finance 0.0016, opencode_without_outcome 0.005, existing_identity 0.0002,
    miscellaneous 0.0001, and replay 0.2. The rationale: reduce math_v2+v3,
    remove multilingual_v1 mass, increase instruction/chat, safety, and
    recorded-pass agents, downweight unverified OpenCode.
  - Epoch capacity at the config's pinned path (one mixture epoch):
    **569,291 packed sequences** ≈ 142.87B tokens over 45.0M documents —
    SFT 435,504 sequences ≈ 108.84B tokens across the 50 sources (largest:
    math_v3/train 141,090; math_v2/high 52,694; finance 38,140; math_proofs
    33,805; competitive programming 37,399), replay 133,787 sequences ≈
    34.02B tokens. 56 sequences are reserved for validation across 7 sources.
- Result: the run trained **2,100 updates** (≈**17.6B packed-slot tokens** =
  67,200 × 262,144, upper bound) in the 20-hour window and stopped at the
  schedule bound (`stop_reason: maximum_updates`), not at the one-epoch
  stopping condition: 67,200 draws is ~11.8% of the epoch's 569,291 packed
  sequences, and the exposure receipt records `full_sft_coverage: false`. The
  20-hour clock also forced one restart (restored from step 1306). Realized
  draws: 53,774 SFT sequences (80.02%) + 13,426 replay sequences (19.98%).
- Evidence: `selected.json` and `actual-exposure.json` under
  `s3://marin-us-east-08a/marin/users/benfeuer/mochi-long-loop/long-loop-mochi-sft-exp3-v5-bankadam-128/sft/2026.11.24.128/`
  (copies of both in `configs/`); export
  `.../exports/step-2100` (the evaluated model).
- Interpretation and limitations: token figures are packed-slot upper bounds;
  the capacity manifest counts `token_bodies_read: false`, so per-source token
  totals come from catalog metadata rather than measured bodies. The score
  below is therefore an early-stop result, not an epoch-complete one, and the
  arm consumed each SFT source at most once.
- Decision and next action: V5 arm documented; assemble the results table.

### 2026-11-25 18:48 UTC - Results and reading

- Question or hypothesis and reason for trying it: compare the arms on the
  shared Micro210 policies and read what the difference does and does not
  establish.
- Published source revision: eval runtime `5ca48a7e` (2026.11.23 policy) and
  `ea21a69e` (2026.11.20-together policy), per record provenance.
- Command or material configuration: SkyRL mechanism, GB200 serving profiles,
  sealed 434-row 2026.11.23 table / 440-row 2026.11.20-together table.
- Result (all values re-verified from `record.json`):

| Arm | Checkpoint | Micro210 2026.11.23 pass@1 | Micro210 2026.11.20-together | EvalDash | record.json |
|---|---|---|---|---|---|
| Base | step 393491 | 0.2607 (422/434 scored) | 0.2477 (429/440 valid) | [base 11.23](https://marina.oa.dev/evaldash/runs/mochi-base-step393491-micro210-2026.11.23) | [link](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fevals%2Fmochi-base-step393491-micro210-2026.11.23%2Frecord.json) |
| Full SFT | step 400407 | 0.5971 (417/434 scored) | 0.5973 (413/440 valid) | [sft 11.23](https://marina.oa.dev/evaldash/runs/mochi-sft-step400407-micro210-2026.11.23) | [link](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fevals%2Fmochi-sft-step400407-micro210-2026.11.23%2Frecord.json) |
| V5 SFT (Exp3 arm) | step 2100 | 0.5819 (421/434 scored) | — | [v5 11.23](https://marina.oa.dev/evaldash/runs/mochi-v5-sft2100-micro210-2026.11.23) | [link](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fevals%2Fmochi-v5-sft2100-micro210-2026.11.23%2Frecord.json) |

  (2026.11.20-together record links: [base](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fevals%2Fmochi-base-step393491-micro210-2026.11.20-together%2Frecord.json),
  [SFT](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fevals%2Fmochi-sft-step400407-micro210-2026.11.20-together%2Frecord.json).)

- Interpretation and limitations:
  - Both SFT arms move the base by roughly +32 points (0.2607 → 0.5971 and
    0.2607 → 0.5819) on the 2026.11.23 policy, and the two 2026.11.20-together
    numbers (0.2477 → 0.5973) agree in magnitude.
  - The V5 arm reaches 0.5819 with an estimated ≈17.6B packed-slot tokens
    against the full arm's ≈319.1B — about 1/18 of the packed-slot budget —
    landing within 1.6 points of the full arm on the same policy and scored
    rows.
  - This is not a controlled ablation. The arms differ in mixture construction
    (216-store uniform-share epoch vs 50-source sealed 80/20 release),
    optimizer schedule (constant-then-decay over 6,916 updates vs
    warmup-5/decay-210 over 2,100), cluster and batch geometry, and stopping
    (full epoch vs 20-hour early stop at ~12% of the epoch). The gap between
    0.5819 and 0.5971 is within the spread one expects from those differences
    plus row-level infra failures; do not read it as a recipe ranking.
  - Single seed, single policy suite; per-benchmark numbers live in each
    `record.json` `metrics` field rather than here.
- Decision and next action: publish the branch and index record; no further
  experiments are in scope for this task.

## Known limits

- Token figures (≈319.1B and ≈17.6B) are packed-slot upper bounds: sequence
  counts × 262,144, padding not subtracted. Measured context positions and
  padding for the V5 arm live in its native consumer receipt (referenced by
  `position_accounting` in `v5-mixture-rationale.json`); the full-SFT arm has
  no equivalent receipt in this record.
- Both arms' data, eval rows, and run telemetry live in private Marin object
  storage. Filer and EvalDash links require Marin access. What is public-safe
  (configs, store/epoch manifests, weights, exposure receipt) is copied into
  `configs/` of this directory; no dataset contents are copied.
- The V5 round config exists only on `benfeuer/long-loop-prototype` at
  `a3d4a7c2b0`; per the task, no Marin PR is opened for this record.
- The V5 score is an early-stop result at 2,100 updates (~11.8% of one
  mixture epoch, `full_sft_coverage: false`), so it does not measure what a
  full epoch of the V5 mix would score.
- Micro210 scores come from one sealed policy per date with single-seed
  sampling; 2026.11.20-together numbers exist only for the base and full-SFT
  arms. Unscored rows (12–27 per run) are infrastructure failures counted as
  missing, not zero.
- The full-SFT replay fraction realized is 20.43% vs the 0.2035 target; the
  notebook shows the derivation.

## Final decision

- Selected approach and evidence: record-only comparison. Both SFT recipes
  improve the Mochi base by ~+32 points on Micro210 2026.11.23; the V5 arm
  (≈17.6B packed-slot tokens, early-stopped) is within 1.6 points of the
  production full-SFT arm (≈319.1B packed-slot tokens). Configs, per-component
  mixes, and verified scores are in this notebook and `configs/`.
- Rejected alternatives and why: no alternative presentation was tried; the
  numbers were re-verified from `record.json` before publication instead of
  trusting the CSV.
- Marin contribution PRs: none (per task).
- Reproduction instructions: full-SFT arm — run `MOCHI_SFT`'s
  `mixture-262k-176node` preset from Marin `main` (config copied at
  `configs/mochi_sft.py`, commit `dca3b9c0eb...`); V5 arm — apply
  `configs/mochi-sft-exp3-v5-bankadam-128.yaml` (commit `a3d4a7c2b0` on
  `benfeuer/long-loop-prototype`) with the long-loop round runner. Evals:
  Micro210 policies 2026.11.23 / 2026.11.20-together under #10197 scoring.
- Remaining uncertainty: whether the V5 arm's deficit to the full arm (if any)
  closes with a complete epoch; whether the two recipes differ on benchmarks
  where scored rows diverge; padding-adjusted token counts for both arms.
