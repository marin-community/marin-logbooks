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
- The model: Mochi is the 262K-context checkpoint of Marin's "hero" MoE
  pretraining ladder (run `hero-lce-log65536sq-step390k`, version
  `2026.08.19.2`, checkpoint step 393491). Architecture per `HERO_MODEL` in
  `experiments/grug/moe_hero_ep/heuristic.py` (Marin `main`): 48 layers,
  hidden 6,144, vocab 128,256, 384 routed experts with 8 active per token plus
  2 shared experts, 48 query heads (head_dim 128), 12 local-attention KV heads
  and 6 global-attention KV heads (stored maximum 12), sliding window 2,048
  with global attention every fourth layer, context 262,144. This record does
  not restate total or active parameter counts.
- Vocabulary: "hero" names that pretraining ladder and its recipe family;
  "Harrier cluster×quality cells" are the pretraining data store's
  cluster/quality buckets; "long-loop" is the Snowball SFT→RL→eval round
  prototype; "Exp3"/"V5" name the third long-loop experiment and its fifth
  sealed SFT data-mix candidate. The full-SFT arm trains on 11 racks of 64
  GB200s (704 GPUs, 176 packed sequences per update); the V5 arm on 128 GB200s
  (two racks, 32 sequences per update).
- Baseline and alternatives: Mochi base step 393491. Alternatives compared:
  full SFT (`hero-sft-fullab-step393k-v2`, step 400407) and V5 SFT
  (`long-loop-mochi-sft-exp3-v5-bankadam-128`, step 2100).
- Evidence needed to choose: Micro210 policy evals with scored counts, plus the
  exact configs and per-component data mixes of both arms.
- Constraints and unresolved questions: record-only task — no new runs, no
  Marin PRs. Training data stays in object storage; only manifests, configs,
  and run metadata are copied here.

## Baseline

- Date: checkpoint used by both arms. The 2026.11.23-policy evaluations ran
  2026-11-24/25 (record versions `2026.11.24.x`, records written 2026-11-25);
  the 2026.11.20-together evaluations ran earlier and were posted
  2026-11-21T04:27:28Z — their exact execution times are not in this record.
- Published source revision: for the 2026.11.23-policy runs, eval runtime git
  SHA `5ca48a7e`, frozen Micro210 policy digest `9d850b9f`, sealed table
  `2280f01a` — the `provenance` and `eval.source_digest` fields copied verbatim
  from each `record.json` (identifiers of the private eval runtime and frozen
  policy bundle; the public lineage lives in the linked issue comments). For
  the older 2026.11.20-together runs, the
  [posting comment](https://github.com/marin-community/marin/issues/11337#issuecomment-7057267101)
  gives policy SHA `8f7ef8170fbe...`, original runtime `ea21a69e55`, base
  recovery bundle `63a21236fb` with runtime corrections `e1fd3bcf3a`; ".7" and
  ".9" are run-version suffixes (record versions `2026.11.20.7` / `2026.11.20.9`),
  not sampling parameters.
- Policy and metric: each Micro210 policy seals one attempt per row of a fixed
  row table — 440 rows for 2026.11.20-together, 434 for 2026.11.23 — spread
  over 20 benchmarks (NUPA, MATH-500, GPQA, MMLU-Pro, IFBench, BFCL-Parity,
  BixBench, CRUXEval, DS-1000, HumanEval+, MBPP+, OlympiadBench, TriviaQA,
  GSM8K, MRCR, TBLite, SWE-bench Verified, Terminal-Bench 2, TruthfulQA,
  AIME24; 10–100 rows each per the linked comment table). Scoring follows the
  [Marin Eval Policy 2026-10-16 (#10197)](https://github.com/marin-community/marin/issues/10197):
  infrastructure errors stay absent, Harbor `AgentTimeoutError` and
  `NonZeroAgentExitCodeError` grades count as real grades, and timeouts get no
  automatic zero. "pass@1" is correct attempts over scored attempts on the
  sealed table; the 2026.11.20-together headline is the pooled verifier-score
  mean over valid attempts. How partial-credit verifier rewards (for example,
  MRCR's mean sequence similarity) map into the binary pass@1 counts
  (110/249/245) is not stated in the archived records, and the aggregation is
  not part of the public marin tree at the pinned runtime commit (`5ca48a7ee9`),
  so the pass predicate is unavailable here; per-benchmark verifier rewards
  are in each `record.json` `metrics` field, and the two headline numbers are
  quoted as recorded. The 2026.11.20-together judge was Together
  GLM-5.3-Flash (per its comment); the 2026.11.23 records leave the judge
  field null and its comment names none. Length behavior differs by run (e.g.
  92/434 base trials and 128/434 V5 trials stopped on length on the 2026.11.23
  policy). Whether any benchmark rows overlap training data was not checked
  for this record and remains unknown.
- Configuration, command, and external inputs: Mochi base step 393491, served
  from `s3://marin-us-east-08a/marin/users/benfeuer/mochi-bringup/models/mochi-base-step393491/2026.11.17.1/hf`
  (export manifest SHA-256 `fb5e8e8e0427...` per the posting comment).
- Environment and hardware when relevant: Micro210 evals on CoreWeave GB200s
  (16 task instances), Mochi GB200 serving profile (2026.11.20-together used
  the CW08 DP16/EP16 profile with compilation and PIECEWISE graphs).
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

Entry headings record the UTC commit time that published the entry. Corrections
supersede earlier statements without rewriting them.

### 2026-11-25 18:40 UTC - Verified every eval number from record.json

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
  unscored rows are absent, not zero: on the 2026.11.23 policy the base has
  12 infra trials absent, full SFT has 12 infra errors plus 5 unavailable
  trials (the record's own breakdown), and V5 has 13 infra trials absent; on
  the 2026.11.20-together policy the base run combines the original .7 dump
  with an authorized 95-row .9 recovery, and the absence breakdowns for both
  older runs are not stated in the archived records. The policy description
  strings in `record.json` carry these details.
- Decision and next action: numbers verified; document the two arms' configs
  and data mixes.

### 2026-11-25 18:40 UTC - Full-SFT arm: config and data mix

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

### 2026-11-25 18:40 UTC - V5 arm: config and data mix

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

### 2026-11-25 18:40 UTC - Results and reading

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

### 2026-11-25 18:50 UTC - Cold-read round 1: corrections to the V5 arm entries

- Question or hypothesis and reason for trying it: a fresh cold reader of PR
  #84 found material errors in the two V5 entries above and smaller slips
  elsewhere ([cold-read-responses.md](cold-read-responses.md) lists the notes
  verbatim in summary and their dispositions).
  Before writing any correction I re-derived every disputed number from the
  run receipts and the public round-runner code at
  `a3d4a7c2b0bb810939bb8967e5f54cb92c14f3f9`. This entry supersedes the
  conflicting statements above; the entries are kept as written.
- Published source revision: marin `a3d4a7c2b0bb810939bb8967e5f54cb92c14f3f9`
  (recorded as `source_revision` in the run's `inputs.json` receipt) for the
  long-loop code; marin `cc3706eb2` for the hero recipe files.
- Command or material configuration: reads of `inputs.json`,
  `actual-exposure.json`, and `selected.json` from the run directory; code
  reads of `experiments/post_training/long_loop/sft_epoch.py`,
  `experiments/post_training/long_loop/mochi_sft.py`,
  `lib/levanter/src/levanter/data/mixture_epoch.py`,
  `lib/levanter/src/levanter/training_budget.py`, and
  `experiments/grug/moe_hero_ep/train.py` at that revision.
- Corrections, each old claim → corrected claim:
  1. **The proposal mixture is not what the run trained on.** The round
     runner's `bind_sft_stopping` (`sft_epoch.py`) replaces the release
     `train_weights` with capacity-derived epoch weights: every included SFT
     source is drawn exactly its packed training-sequence count over one
     planned epoch, and replay fills its declared 0.2 share. The proposal
     family shares therefore only chose the source set and the replay share.
     The resolved plan is 17,010 updates over 9 mixture blocks (8 × 64,000
     draws plus a 32,320-draw final block, stage boundaries every 2,000
     updates), 544,320 draws total, realized replay fraction 0.200015. Actual
     family draws over the 2,100 updates run
     (from `actual-exposure.json`, aggregated by the rationale's
     `source_families`) were: math_v2_v3 25,123 (37.39%), replay 13,426
     (19.98%), other_math 7,458 (11.10%), competitive_code 5,031 (7.49%),
     finance 4,704 (7.00%), instruction_chat 3,991 (5.94%),
     opencode_without_outcome 2,847 (4.24%), nemotron_swe 1,658 (2.47%),
     multilingual_v2 993 (1.48%), recorded_pass_agents 961 (1.43%), science
     828 (1.23%), relay 78 (0.12%), safety 65 (0.10%), existing_identity 36
     (0.05%), miscellaneous 1 (0.00%). Per family these match the resolved
     schedule's inventory-proportional allocation to integer rounding; they
     differ materially from the proposal shares (e.g. recorded_pass_agents
     15% proposed vs 1.43% drawn; finance 0.16% vs 7.00%). The proposal shares
     also sum over the total mixture (replay included), not "of the SFT mass".
  2. **Epoch denominator corrected: 17,010 updates, not the 569,291-sequence
     inventory.** `plan_mixture_epoch` sizes the epoch from the SFT training
     sequences alone: ceil(435,448 / (0.8 × 32)) = 17,010 updates = 544,320
     draws. The 569,291 figure sums the whole capacity inventory (all replay
     stores plus 56 validation reservations) and is not the training epoch.
     The run's 2,100 updates are **12.35%** of the planned epoch (not 11.8%),
     and each selected packed SFT sequence was drawn at most once (53,774
     draws, 53,774 unique — no repeats).
  3. **`context_positions` in the exposure receipt is a planned-epoch field,
     not consumption.** It is carried over from the planned epoch exposure
     (`samples_planned × 262,144` per component; 544,320 × 262,144 =
     142,690,222,080 total), which is why it exceeds the 67,200 drawn slots
     (17,616,076,800 packed slots). The measured consumption fields in that
     receipt are `samples`, `unique_sequences`, and `coverage` (draws over the
     source's training-sequence requirement, summed per component).
  4. **Stop reason and restart corrected.** `selected.json` (now archived as
     [`configs/v5-selected.json`](configs/v5-selected.json)) records
     `stop_reason: maximum_updates` — the trainer reached its planned 2,100
     updates; the 20-hour clock was not the binding stop. `duration_seconds`
     72,000 is the configured budget, not measured wall-clock: the receipt's
     deadline minus its finalization time leaves 50,449 s unused, so the SFT
     stage finished about 5 h 59 m after its budget window opened
     (2026-11-24T20:30:04Z → 2026-11-25T02:29:15Z, deadline
     2026-11-25T16:30:04Z). `restored_step: 1306` records that the trainer
     session resumed from step 1306 — one restart during the run; the receipt
     does not record its cause.
  5. **Comparison wording corrected.** Full SFT scored 417 of 434 rows and V5
     421, so the arms were scored on the same policy but different scored
     subsets. The 1.52-point gap (0.5971 vs 0.5819) is descriptive; this
     record does not establish its cause or significance.
  6. **V5 optimizer provenance added.** The round pins a hash-verified
     BankAdam128 reference record; per the public long-loop README at
     `a3d4a7c2b0`, its `MochiWeightsInitializer` loads only base weights and
     pending router biases and starts update zero with fresh optimizer moments
     (`initializer_mode: weights_only` in the run receipt), whereas the
     full-SFT launcher restores the hero pretraining optimizer state. Stacked
     4D expert banks route to Adam (`muonh_param_ndims: [2, 3]`); other matrix
     groups keep MuonH. The run's warmup 5 and decay 210 scale the reference's
     fractions (round(3400/34000 × 2,100)); learning rates and clip are
     inherited from the reference record, not restated in the round yaml. The
     run receipt records logical batch 32 = physical batch 32, one microbatch,
     vocab 128,256, packing cap 262,144 with no lookahead, and the excluded
     NVLink domain `dh1-r136-us-east-08a`. (The warmup half of that sentence
     is corrected in the 19:19 entry below: warmup is retained, only decay is
     scaled.)
  7. **Micro210 policy definition added** to the Baseline section above (row
     table, judge, #10197 classification, pass@1 vs pooled mean, length stops,
     unknown overlap check). The 2026.11.20-together provenance is now scoped
     to its own lineage instead of borrowing the 2026.11.23 pins.
  8. **Legacy release metadata reconciled.** The release manifest's geometry
     (batch 256, 746 updates, lookahead 128, pack 256) and
     `training_ready: false` describe the publisher's proposal-time state
     before the native consumer receipt existed; the data seal certifies the
     completed release, and the round resolved its own geometry (batch 32,
     cap 262,144, no lookahead) against a fresh capacity receipt. The
     rationale's `position_accounting` note names a separate native consumer
     receipt without giving a locator; this record does not include a padding
     receipt, and the Known limits section no longer claims it does.
  9. **Inventory/unit slips.** The full-SFT manifest has nine single-store
     families (agenttrove and openthoughts4 the largest, seven smaller), not
     "agenttrove, openthoughts4, and ten smaller"; one mixture block is 64,064
     sequences = 364 updates, not "steps".
  10. **Model and fleet introduction added** to the Question section
      (architecture summary, hero/Harrier/long-loop/Exp3 vocabulary, 704 vs
      128 GPUs).
- Result: corrections applied to the summary sections and this entry;
  `configs/v5-selected.json` archived; `configs/README.md` updated.
- Evidence: run receipts (`inputs.json`, `actual-exposure.json`,
  `selected.json`), the round-runner code at `a3d4a7c2b0`, and the cold-read
  report linked above.
- Interpretation and limitations: the corrected reading strengthens the
  early-stop caveat — the V5 arm's realized mixture was inventory-proportional
  over the 50 sealed SFT sources, so its 2,100-update exposure is
  representative of the sealed mix's proportions but covers only ~12.35% of
  the planned epoch.
- Decision and next action: re-run the cold-read loop on the updated head.

### 2026-11-25 18:57 UTC - Cold-read round 2: refinements

- Question or hypothesis and reason for trying it: the second cold read
  confirmed the round-1 corrections and raised four refinements plus one
  publishing defect (the merged index PR still pinned the superseded
  notebook). This entry applies the refinements; the publishing defect is
  fixed by a follow-up index-record PR against logbooks `main`.
- Published source revision: unchanged — marin `a3d4a7c2b0bb810939bb8967e5f54cb92c14f3f9`
  (long-loop code), `cc3706eb2` (hero recipe files), runtime `5ca48a7ee9`.
- Command or material configuration: code reads of
  `lib/levanter/src/levanter/training_budget.py`,
  `experiments/post_training/long_loop/sft_epoch.py`,
  `experiments/grug/moe_hero_ep/sft_recipe.py`,
  `lib/marin/src/marin/execution/build_context.py`, and the Micro210 seal
  commit `5ca48a7ee9` ("[eval] Seal Micro210"); receipt re-aggregation.
- Refinements:
  1. **Partial-block sampling, not just integer rounding.** This refines
     correction item 1's phrase "match ... to integer rounding": the
     2,100-update run consumed one complete 64,000-draw block plus 3,200
     draws of the next shuffled block, so per-family realized shares are
     approximately inventory-proportional, with integer quota allocation plus
     seed-specific partial-block effects. Examples: finance proportional
     4,707.65 draws vs 4,704 actual; instruction_chat 3,971.98 vs 3,991. The
     exposure receipt is a deterministic sampler reconstruction (its `basis`
     field: `seed_specific_native_sampler_reconstruction`), not independently
     measured draw telemetry.
  2. **pass@1 predicate unavailable.** How partial-credit verifier rewards
     map into the binary pass@1 counts is not stated in the archived records
     and the aggregation is not part of the public marin tree at the pinned
     runtime commit; the Baseline section now says so explicitly and points
     to the per-benchmark verifier rewards in each `record.json`.
  3. **Reproduction version claim corrected.** A fresh `--prefix` isolates a
     rerun's outputs; the preset retains the historical version label
     `2026.11.10` because `resolve_version` returns a supplied version
     unchanged. Final decision updated.
  4. **Chronology scoped and entry timestamps fixed.** Baseline now scopes
     the 2026-11-24/25 dates to the 2026.11.23-policy evaluations and dates
     the 2026.11.20-together evaluations by their posting
     (2026-11-21T04:27:28Z). Entry headings now record the UTC commit time
     that published each entry (18:40 for the initial four, 18:50 for the
     round-1 corrections), matching the commit history.
- Result: notebook updated; follow-up index-record PR opened against
  logbooks `main` (the round-1 index PR was merged by a maintainer before the
  corrections landed, so `main` briefly pinned the superseded notebook).
- Evidence: same receipts and code reads as round 1; the round-2 cold-read
  report is summarized in [cold-read-responses.md](cold-read-responses.md).
- Interpretation and limitations: none of the refinements changes the
  headline comparison or the corrected V5 consumption picture.
- Decision and next action: cold-read round 3 against the new index-record PR.

### 2026-11-25 19:05 UTC - Cold-read round 3: two small clarifications

- Question or hypothesis and reason for trying it: the third cold read checked
  the arithmetic and hashes, found no public-safety exposures, and raised two
  small clarifications.
- Published source revision: unchanged.
- Command or material configuration: none (text-only fixes); architecture
  fields re-read from `experiments/grug/moe_hero_ep/heuristic.py` at
  `dca3b9c0eb` (`num_kv_heads=12`, `local_kv_heads=12`, `global_kv_heads=6`).
- Refinements:
  1. **KV-head qualification.** The architecture summary now reads "48 query
     heads (head_dim 128), 12 local-attention KV heads and 6
     global-attention KV heads (stored maximum 12)" instead of "12 KV heads".
  2. **Absence breakdown preserved.** The verification entry and Known limits
     now carry the archived records' own breakdown — 2026.11.23: base 12
     infra trials absent, full SFT 12 infra errors plus 5 unavailable trials,
     V5 13 infra trials absent — instead of collapsing full SFT's 17 absent
     rows into one label; the older runs' 11 and 27 absences are marked as
     not broken down in the archived records.
- Result: notebook updated.
- Evidence: [cold-read-responses.md](cold-read-responses.md) round 3.
- Interpretation and limitations: none; the comparison is unchanged.
- Decision and next action: cold-read round 4.

### 2026-11-25 19:19 UTC - Cold-read round 5: warmup provenance correction

- Question or hypothesis and reason for trying it: the fifth cold read
  verified the scores, coverage, source-copy fidelity, manifest hashes,
  exposure counts, the 12.35% epoch fraction, and the ~18.1× token ratio, and
  raised one provenance correction.
- Published source revision: unchanged — `a3d4a7c2b0bb810939bb8967e5f54cb92c14f3f9`.
- Command or material configuration: re-read of
  `experiments/post_training/long_loop/mochi_sft.py` lines 199-208 at that
  commit.
- Correction: round-1 correction item 6 says "warmup 5 and decay 210 scale
  the reference's fractions". For the fresh-optimizer arm the code retains
  the reference's warmup unchanged (`expected_warmup = hero_optimizer["warmup"]`)
  and scales only the decay tail (`round(hero_optimizer["decay"] ×
  schedule.updates / hero_updates)` = round(3400/34000 × 2,100) = 210). The
  corrected statement: **the five-update warmup is retained from the
  reference; the decay tail is scaled to 210 updates.** The round-1 entry
  above now carries a pointer note; its original wording is preserved.
- Result: notebook updated; no numeric settings or headline results change.
- Evidence: [cold-read-responses.md](cold-read-responses.md) round 5.
- Interpretation and limitations: none; the comparison is unchanged.
- Decision and next action: synchronize entry-point pins and run cold-read
  round 6.

## Known limits

- Token figures (≈319.1B and ≈17.6B) are packed-slot upper bounds: sequence
  counts × 262,144, padding not subtracted. No padding receipt is included in
  this record: the V5 rationale's `position_accounting` note names a separate
  native consumer receipt without a locator, and the archived exposure receipt
  measures sampled draws, not padding.
- Both arms' data, eval rows, and run telemetry live in private Marin object
  storage. Filer and EvalDash links require Marin access. What is public-safe
  (configs, store/epoch manifests, weights, run receipts) is copied into
  `configs/` of this directory; no dataset contents are copied.
- The V5 round config exists only on `benfeuer/long-loop-prototype` at
  `a3d4a7c2b0`; per the task, no Marin PR is opened for this record.
- The V5 score is an early-stop result: 2,100 updates of the planned
  17,010-update epoch (12.35% of its 544,320 draws), with every selected SFT
  sequence drawn at most once and no SFT sequence repeated. It does not
  measure what a full epoch of the V5 mix would score.
- Micro210 scores come from one sealed policy per date with single-seed
  sampling; 2026.11.20-together numbers exist only for the base and full-SFT
  arms, and the two policies are not comparable to each other. Unscored rows
  are absent trials counted as missing, not zero (2026.11.23: base 12 infra
  trials absent, full SFT 12 infra errors plus 5 unavailable, V5 13 infra
  trials absent; the older runs' 11 and 27 absences are not broken down in
  the archived records). Whether benchmark rows overlap training data was
  not checked.
- The full-SFT replay fraction realized is 20.43% vs the 0.2035 target; the
  notebook shows the derivation.
- The 2026.11.23-policy judge identity is not recorded in the archived eval
  records (null `judge` field); the 2026.11.20-together judge is named in its
  issue comment.

## Final decision

- Selected approach and evidence: record-only comparison. Both SFT recipes
  improve the Mochi base by ~+32 points on Micro210 2026.11.23; the V5 arm
  (≈17.6B packed-slot tokens, stopped at 12.35% of its planned epoch) is 1.52
  points below the production full-SFT arm (≈319.1B packed-slot tokens) on the
  same policy but different scored subsets. The cause and significance of the
  gap are undetermined in this record. Configs, per-component mixes, resolved
  schedules, and verified scores are in this notebook and `configs/`.
- Rejected alternatives and why: no alternative presentation was tried; the
  numbers were re-verified from `record.json` and the run receipts before
  publication instead of trusting the CSV, and the cold-read round-1
  corrections were re-derived from the run receipts and round-runner code
  before being applied.
- Marin contribution PRs: none (per task).
- Reproduction instructions:
  - Full-SFT arm: check out Marin `dca3b9c0eb486986f49d8407dfc09efe1590f486`
    (the file-identical preset lives there and at `main`); run
    `uv run python experiments/post_training/mochi_sft.py --preset mixture-262k-176node --export --run`
    with a fresh `--prefix` to isolate the rerun's outputs from the archived
    run's artifact path. The preset keeps the historical version label
    `2026.11.10` — `resolve_version` returns a supplied version unchanged — so
    the prefix, not the version, separates a rerun. This launches 704 GB200s
    for 6,916 updates. The archived run's own recipe recorded
    `source_revision 406dff47f217637ceca499b266989588cb0df5ab`; that
    identifies the historical execution, not the archive commit.
  - V5 arm: check out Marin `a3d4a7c2b0bb810939bb8967e5f54cb92c14f3f9`;
    preflight with
    `uv run python -m experiments.post_training.long_loop.round_runner --round experiments/post_training/long_loop/rounds/mochi-sft-exp3-v5-bankadam-128.yaml --version <new-version> --through sft --source-revision a3d4a7c2b0bb810939bb8967e5f54cb92c14f3f9 --validation-output /tmp/mochi-v5-repro`
    (substitute a fresh version and a writable validation output), then submit
    the same command with `--run` from an Iris CPU coordinator on 08a;
    submission requires explicit user approval per the long-loop README. The
    20-hour budget and the hash-pinned reference, data release, seal, capacity,
    and initializer are inside the round yaml.
  - Evals: Micro210 policies 2026.11.23 and 2026.11.20-together scored under
    the #10197 eval policy; per-run posts with EvalDash, record, and Filer
    links are in the linked marin#11337 comments.
- Remaining uncertainty: whether the V5 arm's deficit to the full arm (if any)
  closes with a complete epoch; whether the two recipes differ on benchmarks
  where scored rows diverge; padding-adjusted token counts for both arms; the
  2026.11.23-policy judge identity; train/eval overlap.
