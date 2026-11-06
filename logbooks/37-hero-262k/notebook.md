---
title: "Recipes and start checkpoint for the hero's 262K extension"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/37"
related_issues:
  - "https://github.com/marin-community/marin/issues/8435"
  - "https://github.com/marin-community/marin/issues/8506"
  - "https://github.com/marin-community/marin/issues/8701"
  - "https://github.com/marin-community/marin/issues/9615"
research_branch: "codex/37-hero-262k"
marin_base_revision: "92958f7a84886cacc88aae44431d001d37f0187d"
archive_template_revision: "fed27a740bb149773b4d429d9ef40e434cfceb63"
status: active
---

# Recipes and start checkpoint for the hero's 262K extension

## Source and evidence in this archive

The branch tree is arm code at `92958f7a84`. Its merge commit records, without changing the tree, every other Marin commit cited here, so each stays reachable:

| Commit | Contents |
|---|---|
| `1eed8fff84` | Six 11-rack arms |
| `92958f7a84` | Queue-64 arm, and memory logging |
| `5b22fd9c36` | Interim log-scaling power field used by the arms. The maintained field is in Marin #10986, head `05aa7d2a91`. |
| `ca1f3e159a`, `fcf4573757`, `1b113f404c` | Sweep and MRCR evals |
| `61a2a9e674`, `a5a59bdacd` | Heads of `hero-semifinal` and `hero-merge-eval` |
| `dff14a5419` | Merge tool used to save the checkpoint |
| `0fb0d0541f` | Memory diagnostics |
| `d0d4e15f45` | Deploy head used for the one-rack stop-path test |
| `70bf6bbc0a` | Head of the deploy PR, Marin #11041, when this record was written |
| `133d4e4d55` | Frozen-bias arm: `1eed8fff84` plus one diagnostic commit and a test-only import sort |

Files beside this notebook:

- `results/mrcr/<arm>/`: per-source MRCR losses and metrics at step 390351 for the seven arms and, in `frozen/`, the frozen-bias arm. `recovery_stat.py` tests the seven arms only.
- `results/overnight/`: tables behind the overnight results, and the arms' resolved configs in `plan_configs.json`.
- `results/merge/`: merge-study scores, with host topology fields removed.
- `analysis/`: `recovery_stat.py` is the corrected test; run it with `results/mrcr` as its argument. The other scripts generated the tables. Many read a private bucket or W&B and need project credentials.
- `commands/`: arm submission scripts, with local paths and credentials removed. Their `global184` and `qk184` branches were never submitted.
- `pages/`: result pages written before the final decision, with personal names and private links removed. Their proposals, including Global 1.88 as first finalist, are superseded by the decision entry below.

Other artifacts stay in their stores and need project credentials. Per-source eval outputs are in `s3://marin-us-east-02a/marin/eval/mrcr-hero/`. The arms' step-390351 checkpoints are under a one-day lifecycle prefix in `s3://marin-us-east-08a/tmp/ttl=1d/`, so they expire unless copied. W&B runs are public.

## Question and decision criteria

- Goal: choose query-scale recipes and a start checkpoint for the hero's last 200B tokens at 262,144-token context.
- Baseline and alternatives: `qk_mult` 1.66 unchanged (Keep 1.66) from raw step 390251, against raised global-layer scales, log-scaled global layers and a merged start.
- Evidence needed: held-out text loss, retrieval on long documents, stability of the first steps, memory margin at 11 racks, and whether the recipe can load in SFT and RL.
- Constraints: the run must fit two finalists in about one weekend of the full fleet and must not exhaust device memory.

## Baseline

- Date: 2026-11-05.
- Checkpoint: `s3://marin-us-east-02a/marin/grug/hero-ctx65k-step372k/2026.08.19.2/checkpoints/step-390251`, the final step of `hero-ctx65k-step372k`.
- Result: clean macro at 262K 1.64712, MRCR recovery 0.243 [0.211, 0.278] and full NLL 0.861 at `qk_mult` 1.66 before any 262K training.
- Limitations: this checkpoint had never trained at 262K, so baseline recovery is low on every measure that needs long-range attention.

Definitions, used throughout:

- `qk_mult` multiplies attention queries on every layer. `qk_mult_long_scale` also multiplies them on global layers, which are every fourth layer and the last.
- Log scaling multiplies a global-layer query that sees `n` keys in its document by `1 + slope * ln(max(n / floor, 1))`, raised to `power`. This has the form of Inkling's released log scaling, which pairs slope 0.1 with floor 128000. Neither source tested other floors at this length.
- Clean macro is the mean loss over 19 held-out subsets. Checkpoint jitter on it is 0.0002 to 0.0007.
- MRCR is the paired eval from Marin #8701: 299 long documents, four conditions, 115,834 scored response tokens per condition. Full context keeps the whole document. Distractor only removes the answer. Needle only keeps the repeated request's turns alone. Query only keeps nothing. Recovery is (distractor only - full) / (query only - needle only).
- MRCR is a teacher-forced copying task, and base models may lean on instruction-following that it does not measure. We weigh it with Paloma and Uncheatable.

The earlier rule for `qk_mult` is 1.3 × (1 + 0.1 ln(N/4096)), which gives 1.48 at 16K, 1.66 at 65K and 1.84 at 262K. It applied to all layers. Its factor follows [YaRN](https://arxiv.org/abs/2309.00071).

## Entries

### 2026-11-05 - No-training sweep over 19 recipes

- Question and reason: how far query scale alone moves retrieval and held-out loss on the final 65K checkpoint.
- Source: `hero-mrcr-eval`, at `ca1f3e159a`, and at `fcf4573757` and `1b113f404c` for the slope arms.
- Configuration: step 390251, 16 workers at CP4, 262K, no training.
- Environment: GB200.
- Evidence: `pages/sweep-results.html`.

| Recipe | Setting | Recovery [95%] | Full NLL | Distractor NLL | Clean macro, 262K, change from Keep 1.66 |
|---|---|---|---|---|---|
| Keep 1.66 | `qk_mult` 1.66 | 0.243 [0.211, 0.278] | 0.861 | 1.119 | reference, 1.64712 |
| Global 1.84 | long scale 1.108 | 0.933 [0.918, 0.947] | 0.090 | 1.081 | +0.00024 |
| All layers 1.84 | `qk_mult` 1.84 | 0.938 [0.924, 0.951] | 0.084 | 1.085 | +0.00266 |
| Global 1.88 | long scale 1.133 | 0.971 [0.965, 0.975] | 0.045 | 1.076 | +0.00091 |
| All layers 1.88 | `qk_mult` 1.88 | 0.969 [0.964, 0.974] | 0.046 | 1.083 | +0.00434 |
| Log floor 65536, squared | slope 0.1, power 2 | 0.992 [0.985, 0.999] | 0.057 | 1.109 | -0.00034 |

Interpretation: recovery rises quickly up to a global-layer scale of about 1.88. All-layers arms cost more clean macro than global-only arms at the same value, for about the same recovery. Log-arm recovery is not like-for-like, since their full and distractor rows run at a higher scale than their needle and query rows; slope 0.4 passes the rule below while being clearly worse on distractor NLL.

We proposed a rule: the lowest global scale whose recovery lower bound reaches 0.95, unless macro loss at 262K is more than 0.005 above Keep 1.66. Among global-layer arms only Global 1.88 passed. Global 1.84 missed on recovery, and Global 2.00 missed on cost. We wrote that threshold after seeing the first six arms. The next entry shows this rule does not describe trained checkpoints, and we did not use it for the final choice.

### 2026-11-05 - Six recipes, 100 steps, one rack

- Question and reason: do the sweep's gaps survive training?
- Source: commits on `hero-semifinal`, head `61a2a9e674`.
- Configuration: `launch_diagnostics` at 262K, CP4 × EP16, capacity factor 1.5, seed 0, long-document mix at 8x, 11-rack learning rate.
- Result:
  - Recovery reached 0.983 to 0.987 on every arm. Keep 1.66's recovery rose from 0.243 to 0.983 and its full NLL from 0.861 to 0.027. We have not explained why this takes so few steps.
  - Clean macro spread 0.0004 between arms at 262K.
  - Clean macro rose 0.005 to 0.007 on five arms. Controls below show this is mostly a one-rack effect.
  - 23.7 to 24.5 s/step, drop fraction 1.27e-3, peak memory 132.52 GiB, no failures.
- Evidence: `pages/sweep-results.html`, `results/overnight/semifinal_results.json`. Each arm is one seed, and intervals resample documents only.

Controls, all Keep 1.66 from raw step 390251:

| Control | Clean macro, 262K change | Recovery | Full NLL |
|---|---|---|---|
| 1 rack, 8x mix | +0.0062 | 0.983 | 0.0270 |
| 11 racks, 8x mix | +0.0006 | 0.982 | 0.0258 |
| 1 rack, 4x mix | +0.0068 | 0.983 | 0.0265 |
| 1 rack, no mix | +0.0061 | 0.980 | 0.0286 |

Eleven times the batch nearly removes the macro rise. The 11-rack run also differed in code and had in-loop evals, so batch size fits the change but is not isolated. The 100 steps do not separate 4x from 8x. The no-mix run may repeat tokens the hero already trained on.

Cross-scale check: weights trained 100 steps at Global 2.00 and evaluated at 1.66 have recovery 0.183 and full NLL 0.935, below the untrained checkpoint. Weights trained at 1.66 and evaluated at 2.00 keep recovery (0.993), at a cost of 0.019 in clean macro. So 100 steps are specific to the scale used. One pair of checkpoints from one seed.

Decision: one-rack results do not decide the recipe. Move to 11 racks.

### 2026-11-05 - Checkpoint merges

- Question and reason: does averaging the final checkpoints give a cheap head start?
- Source: merge evals on `hero-merge-eval`, head `a5a59bdacd`; saved checkpoint from `dff14a5419`. Evidence: `pages/merge-results.html`, `results/merge/`. Details and the exact command are in the [Marin #8435 comment](https://github.com/marin-community/marin/issues/8435#issuecomment-6676769570).
- Configuration: equal-weight hyperball average of 11 checkpoints, 385,500 through 390,000 every 500 steps plus 390,251. Step, optimizer state and non-parameter leaves come from step 390251.
- Result at qk 1.66, clean macro against raw step 390251: -0.00299 at 4K, -0.00293 at 65K, -0.00319 at 262K, with all 19 subsets improved.
- Negative results:
  - No merge reached the -0.005 bar we set in advance for a merged start.
  - A window that adds one 16K checkpoint costs almost nothing, but one that adds two loses most of the gain (-0.00075 at 4K), and three make things worse (+0.00352).
  - On a blend of raw 390251 and 100-step weights, the rule "an interior point beats both ends on loss" was not met. At weight 0.5 recovery keeps 99% of the gain at about raw's loss.
- Not tested: real extension checkpoints; a merge that spans a change of query scale, which the cross-scale check suggests may behave differently.

### 2026-11-05 and 11-06 - Out-of-memory error on the first train step after an eval

- Symptom: at 262K, CP4 × EP16, capacity factor 1.5, one rack, the train step after a dropless eval failed to allocate.
- Withdrawn hypothesis: missing `--gc-interval`. Pool use had returned to baseline, so garbage collection could not help.
- Cause: the move to the eval mesh ran a collective on a new 16-rank communicator, adding 5.63 GiB of buffers outside the pool for the life of the process.
- Fix: [Marin #10994](https://github.com/marin-community/marin/pull/10994). An 11-rack proof ran 100 steps with two handoffs, saves and a final eval; the move added nothing. The first eval forward adds one 176-rank communicator, about 1.03 GiB, once. The tightest device kept 2.08 GiB free.

### 2026-11-06 - Seven arms at launch scale

- Question and reason: which recipes hold at launch scale on the launch data, and is the merged start safe?
- Source: training at `1eed8fff84`, and `92958f7a84` for the queue-64 arm. Evals at `fcf4573757`, and `1b113f404c` for the squared arm.
- Command, with the arm's recipe flags:

```bash
python -m experiments.grug.moe_hero_ep.launch_diagnostics --run-id <run> --seq-len 262144 --batch-size 176 --dp-racks 11 \
  --context-axis-size 4 --expert-axis-size 16 --num-steps 390351 --schedule-steps 390251 \
  --extension-steps 4335 --extension-final-lr-ratio 0.01 --optimizer-racks 11 \
  --long-context-start-step 390251 --restore-from <checkpoint> --capacity-factor 1.5 <recipe flags> --save-checkpoints \
  --checkpoint-minutes 20 --checkpoint-path <temporary prefix> --eval-every 40 --gc-interval 100 --seed 0 --version dev --run
```

- Environment: GB200, 704 GPUs, 4x long-document mix (43.9% long tokens), 5% to 1% of peak learning rate over 4,335 steps.
- Result: all seven finished on their first attempt. 29.2 to 29.5 s/step, mean drop fraction 2.9e-4.

| Arm | W&B train run | Start | Recipe flags | Clean macro, 4K / 65K / 262K | Train loss, 100 steps | Steps clipped, first 20 / all 100 |
|---|---|---|---|---|---|---|
| Keep 1.66 | [`hero-lce11-merged-keep166`](https://wandb.ai/marin-community/marin_moe/runs/hero-lce11-merged-keep166) | merged | `--qk-mult 1.66 --qk-mult-long-scale 1` | 1.6869 / 1.6479 / 1.6455 | 0.8714 | 1 / 2 |
| Global 1.88 | [`hero-lce11-merged-global188`](https://wandb.ai/marin-community/marin_moe/runs/hero-lce11-merged-global188) | merged | `--qk-mult 1.66 --qk-mult-long-scale 1.133` | 1.6866 / 1.6475 / 1.6451 | 0.8708 | 5 / 7 |
| Global 2.00 | [`hero-lce11-merged-global200`](https://wandb.ai/marin-community/marin_moe/runs/hero-lce11-merged-global200) | merged | `--qk-mult 1.66 --qk-mult-long-scale 1.205` | 1.6867 / 1.6477 / 1.6452 | 0.8710 | 6 / 6 |
| Log floor 32768 | [`hero-lce11-merged-log32768`](https://wandb.ai/marin-community/marin_moe/runs/hero-lce11-merged-log32768) | merged | `--log-scaling-n-floor 32768 --log-scaling-alpha 0.1` | 1.6867 / 1.6478 / 1.6453 | 0.8709 | 4 / 4 |
| Log floor 65536, slope 0.2 | [`hero-lce11-merged-log65536a20`](https://wandb.ai/marin-community/marin_moe/runs/hero-lce11-merged-log65536a20) | merged | `--log-scaling-n-floor 65536 --log-scaling-alpha 0.2` | 1.6868 / 1.6478 / 1.6455 | 0.8712 | 2 / 3 |
| Log floor 65536, squared, queue 64 | [`hero-lce11-q64-log65536sq`](https://wandb.ai/marin-community/marin_moe/runs/hero-lce11-q64-log65536sq) | merged | `--log-scaling-n-floor 65536 --log-scaling-alpha 0.1 --log-scaling-power 2` | 1.6867 / 1.6477 / 1.6453 | 0.8712 | 7 / 8 |
| Keep 1.66, raw | [`hero-lce11-keep166-raw`](https://wandb.ai/marin-community/marin_moe/runs/hero-lce11-keep166-raw) | raw | `--qk-mult 1.66 --qk-mult-long-scale 1` | 1.6886 / 1.6496 / 1.6472 | 0.8730 | 3 / 3 |

Eval runs are `hero-mrcr-lce11-*`. Per-source MRCR losses are in `results/mrcr/`, with the test in `analysis/recovery_stat.py`. Earlier fuller tables: `pages/overnight-results.html`.

MRCR, 299 documents, paired against Keep 1.66 merged, Holm-corrected over the 15 pairs of merged arms:

| Arm | Full NLL | Change [95%] | Corrected p | Recovery [95%] | Distractor NLL |
|---|---|---|---|---|---|
| Keep 1.66 | 0.0265 | reference | | 0.983 [0.980, 0.985] | 1.0673 |
| Global 1.88 | 0.0263 | -0.0003 [-0.0011, +0.0006] | 1.00 | 0.983 [0.981, 0.985] | 1.0652 |
| Global 2.00 | 0.0254 | -0.0011 [-0.0021, -0.0003] | 0.13 | 0.984 [0.981, 0.986] | 1.0648 |
| Log floor 32768 | 0.0257 | -0.0009 [-0.0019, +0.0000] | 0.43 | 0.982 [0.980, 0.984] | 1.0652 |
| Log floor 65536, slope 0.2 | 0.0251 | -0.0014 [-0.0023, -0.0006] | 0.011 | 0.983 [0.981, 0.986] | 1.0662 |
| Log floor 65536, squared | 0.0249 | -0.0017 [-0.0026, -0.0009] | 0.002 | 0.984 [0.982, 0.986] | 1.0666 |
| Keep 1.66, raw | 0.0264 | -0.0001 [-0.0006, +0.0003] | not in the family | 0.982 [0.980, 0.985] | 1.0678 |

We re-ran the test from the per-source files, and the p values and changes reproduce. Other pairs that survive correction on full NLL: Global 1.88 against Global 2.00 (0.0009, p 0.001), against slope 0.2 (0.0012, p 0.011) and against squared (0.0014, p 0.001).

Interpretation:

- Clean macro does not separate the merged arms: spread 0.0004, jitter 0.0002 to 0.0007.
- Retrieval differences survive correction across documents only. No arm was repeated, and the tests exclude seed and checkpoint noise. At 11 racks we have no measure of that noise. Two runs at the same queue on one rack differ by about 1e-4 in train loss.
- Recovery separates no arm from Keep 1.66 after correction. Three pairs do separate, all involving log floor 32768: against Global 2.00, slope 0.2 and squared. Log-arm recovery is not like-for-like, so we do not read those three.
- Hard cases, chosen after seeing the data: across 42 comparisons of Global 1.88, Global 2.00 and log floor 32768 with Keep 1.66 by needle count and evidence distance, 2 recovery results survive correction. Global 2.00 at 8 needles and the farthest distance gains 0.0083 on 29 documents. Log floor 32768 at 32K to 65K distance loses 0.0043 on 48 documents. No full NLL result survives. At 2 needles, raised scales go the other way before correction.
- Early gradients: raised-scale arms reach 5.37 to 19.65 before clipping in the first two steps, against 0.62 for Keep 1.66. Clipping scales these to 1.0. The spikes at step 390252 land on the same batch. Keep 1.66 peaks at 2.48 over its 100 steps. 100 steps cannot show whether the spikes matter later.
- Train cross-entropy over the last 20 steps: Global 1.88, Global 2.00 and log floor 32768 are 3.0e-4 to 4.4e-4 below Keep 1.66. The one-rack repeat band is 1.5e-4. The gaps are shrinking.
- Levels between evals: the in-loop eval reads 0.0040 above the offline eval on the same 12 subsets, in all seven arms, because it drops each corpus's last partial window. So differences carry over, but levels do not.

Merged and raw, both Keep 1.66, one seed:

| Measure | Merged | Raw | Merged - raw |
|---|---|---|---|
| Clean macro at 4K, 65K, 262K | 1.6869, 1.6479, 1.6455 | 1.6886, 1.6496, 1.6472 | -0.0017 at each |
| Clean macro gap before training | | | -0.0029 to -0.0032 |
| Train cross-entropy, first 20 steps | 0.8830 | 0.8852 | -0.0022 |
| Train cross-entropy, all 100 | 0.8714 | 0.8730 | -0.0016 |
| Recovery | 0.983 | 0.982 | +0.001, interval includes 0 |
| Full NLL | 0.0265 | 0.0264 | +0.0001 [-0.0003, +0.0006] |
| Grad norm, max, first 20 | 1.04 | 1.23 | |
| Grad norm, max, all 100 | 2.48 | 1.23 | |
| Drop fraction, mean | 2.93e-4 | 2.89e-4 | |

Raw is higher on all 23 subsets at 262K, and merged train cross-entropy is lower on all 100 steps. The gap fell by about half, so it may not last the run. Spike rates are not separated.

### 2026-11-06 - Memory growth at 262K

- Question and reason: pool reserved memory grew about 0.87 MiB per step per device. At that rate the tightest device would run out in about 1,800 to 2,000 steps, and exhausting a device is fatal.
- Source: `hero-mem-growth` at `0fb0d0541f` and `hero-arm3-q32` at `92958f7a84`, both diagnostic-only.
- Result: the train loader's 512-batch queue lives in device memory, about 2 GiB per device at 262K (4 MiB per batch). Growth largely ends when the queue first wraps.
  - Queues of 16 and 48 on one rack reduced used memory by 1.94 and 1.81 GiB, as predicted, and growth largely stopped at the wrap.
  - Eleven racks at queue 64, in `hero-lce11-q64-log65536sq`: the tightest of 704 devices had 3.856 GiB free, against 2.08 GiB at queue 512. Pool slope fell from 1.045 to 0.083 MiB/step past the wrap at step 390315. The run stopped before the second wrap.
  - One rack at queue 64, soak: mean slope 0.915 MiB/step before the first wrap, then 0.187, 0.109, 0.016, 0.016 and 0.000. No device grew over the last 104 steps. Three evals changed nothing.
- Failed or withdrawn:
  - Trimming the pool with `cuMemPoolTrimTo` resets reserved memory, but its floor kept growing at the same rate.
  - Linear growth to out-of-memory: withdrawn, since growth ends at the wrap.
  - "65K keeps a 16 GiB slack": false. The 65K pool is equally compact.
- Unexplained and not adopted: setting the release threshold to peak + 0.06 GiB held reserved memory flat through four evals and saves. We do not know why.
- Limitations: queue 64 and the squared recipe changed together in the one 11-rack run. Queue depth does not change which batch a step gets. All six arms at queue 512 peaked within 0.004 GiB of each other, so recipe does not explain the gap. Neither finalist's exact configuration has run past step 390351 at 11 racks. Non-pool memory rises about 1.03 GiB once, at the first eval, at 11 racks only; read free memory after the first eval.

### 2026-11-06 - Decision

- First finalist's start: the merged checkpoint. Second finalist's start point: undecided.
- First finalist: Global 1.84, `qk_mult` 1.66, long scale 1.108. It follows the value of the earlier rule, on global layers only, where earlier switches changed all layers. It lies between two trained arms, Keep 1.66 and Global 1.88, which do not separate on clean macro, full NLL or recovery. That is an interpolation and has not been measured. No rehearsal ran; it was skipped. Local layers use a 2048-token window, and in the no-training sweep All layers 1.84 cost 0.00266 on clean macro against 0.00024 for Global 1.84, at about the same recovery.
- Second finalist: log floor 65536, slope 0.1, power 2. Needs [Marin #10986](https://github.com/marin-community/marin/pull/10986) (open, head `05aa7d2a91`) for its slope and power. The arm ran an interim version, `5b22fd9c36`. Export is not required before training. A capped export to 65,536 tokens is sized at 0.5 to 1 day and full vLLM support at 2 to 4 days elapsed. Both are estimates, and that work has not started.
- Loader queue: 64.
- Learning rate: 5% to 1% of peak over 4,335 steps. [PENDING: confirmation.]

Not tested at decision time:

- Global 1.84 in training, or any recipe at 11 racks past step 390351. The deploy PR is [Marin #11041](https://github.com/marin-community/marin/pull/11041).
- A repeat of any arm at 11 racks.
- The 11-rack stop path.
- SFT and RL loading of the merged checkpoint.
- Export of the second finalist. Hero exports refuse log scaling until serving support lands.
- Commits on the deploy head that no arm ran, including Marin #10556, which skips short-convolution weight gradients for frozen weights. Reading `api.py:92`, the trained weight takes the old backward path, so we expect no effect on hero training, and a CPU check with stand-in kernels matched values, gradients and HLO op count. We did not check compiled GB200 code. The launcher path with the deploy constants has also not run at 262K.

### 2026-11-06 - Second finalist's start and landing

This entry updates the decision above, whose "undecided" start and "needs Marin #10986" no longer hold.

- Start: the second finalist is a sibling of the first. It starts from the merged step-390251 and does not continue from the first finalist. Data, learning-rate schedule and shape match the first, and only the recipe changes.
- Landing: it launches from a branch that carries Marin #10986 and its constants. That branch merges to main only if this recipe wins, at the comparison on Monday 2026-11-09. Marin #11041 stays the first finalist's deploy PR.
- Check on the first finalist's head: W&B shows that no training run has used it. The only Global 1.84 run is the no-training eval.
- Limit: the finalists share a start and differ in recipe, with one seed each, so training noise remains unmeasured in their comparison.

### 2026-11-06 - Why the first finalist changes global layers only

The first finalist changes global layers only because, in the no-training sweep, global-only had lower clean macro than all layers at the same value, with about the same retrieval. Both columns use the same checkpoint, raw step 390251.

| Measure | Global 1.84 | All layers 1.84 | Global 1.88 | All layers 1.88 |
|---|---|---|---|---|
| Clean macro, 4K | 1.68855 | 1.69105 | 1.68921 | 1.69265 |
| Clean macro, 65K | 1.64963 | 1.65208 | 1.65034 | 1.65376 |
| Clean macro, 262K | 1.64736 | 1.64978 | 1.64803 | 1.65146 |
| Recovery | 0.933 | 0.938 | 0.971 | 0.969 |
| Full NLL | 0.090 | 0.084 | 0.045 | 0.046 |

Global-only is lower on all 19 subsets at both values: by a mean of 0.0024 at 1.84 and 0.0034 at 1.88. Recovery intervals overlap at both values. This is no-training evidence only. All-layers recipes were never trained at 262K, and the sweep's recipe gaps mostly closed within 100 training steps in the arms that did train.

### 2026-11-06 - Run length

Each finalist trains 3,240 steps, from step 390251 to stop step 393491, or 149.5B tokens. This replaces the 200B and 4,335 steps planned earlier. About 27 h per finalist at 29.2 s/step with saves (estimate). The 100-step arms ran on the 4,335-step schedule. [PENDING: confirmation of the learning-rate schedule for this length.]

### 2026-11-06 - Frozen router bias, not adopted

The router bias is a per-layer, per-expert offset on routing scores. Each step, the hero replaces it with the values solved on the previous batch, so it follows one batch and is not a running average. In the Global 1.88 arm its norm rose from 67.23 to 73.52 over 100 steps. This arm asks whether holding the checkpoint's bias changes the results. The main risk is that a bias averaged over 65K training does not balance the 262K mix, so drops rise.

- Rule, fixed before submission: launch as planned by default. Reject if drops rise over 2x. Adopting needs real-text loss better than the control by more than 0.0007, with drops and retrieval no worse. We could still decline, because 100 steps cannot show long-run drops.
- Decision: not adopted. Both finalists stay unfrozen, Global 1.84 and log floor 65536 squared. Drops did not reach the reject line, and clean macro did not reach the adopt line.
- Run: [`hero-lce11-frozen-global188`](https://wandb.ai/marin-community/marin_moe/runs/hero-lce11-frozen-global188), eval run `hero-mrcr-lce11-frozen-global188-step390351`. Control: `hero-lce11-merged-global188`.
- Source: training at `133d4e4d55`. Evals at `fcf4573757` with the recipe flags of the control, whose no-training baseline also serves here.
- Command: the Global 1.88 arm's command above, with its run ID and temporary prefix changed, and the environment variable that turns on the diagnostic:

```bash
HERO_FROZEN_ROUTER_BIAS=390251 python -m experiments.grug.moe_hero_ep.launch_diagnostics --run-id hero-lce11-frozen-global188 \
  --qk-mult 1.66 --qk-mult-long-scale 1.133 <other flags as in the launch-scale arms>
```

- Environment: as the control, including merged step-390251, seed, data and queue 512. A token-by-token comparison of the dry-built specs found differences only in run ID, checkpoint path, output path and step fingerprint.
- Freeze: after the restore, the diagnostic sets the pending bias to the negated stored `params.router_bias`, the average over 11 checkpoints, and keeps it. It does not use the checkpoint's own pending bias, which comes from one batch at norm 69.18. So the two runs apply different biases from the first step. The applied norm was 67.23333 on all 100 steps, and in the final checkpoint the largest element change from merged was 3e-8. That checkpoint is under the one-day prefix `s3://marin-us-east-08a/tmp/ttl=1d/hero-lce11-frozen/global188/`, so it expires unless copied.
- Files: `results/mrcr/frozen/`, and `results/overnight/frozen_results.txt`, the output of `analysis/overnight/frozen_res.py` with `results` as its argument. It reads W&B.

Results at step 390351, frozen and control, one seed each. Intervals resample documents only.

| Measure | Frozen | Control | Change [95%] | Rule |
|---|---|---|---|---|
| Drop fraction, mean, all 100 steps | 3.083e-4 | 2.927e-4 | 1.05x | Reject over 2x |
| Drop fraction, mean, first 20 | 3.193e-4 | 3.075e-4 | 1.04x | |
| Drop fraction, mean, last 20 | 3.327e-4 | 3.115e-4 | 1.07x | |
| Drop fraction, max | 6.235e-4 | 5.864e-4 | 1.06x | |
| Clean macro, 4K | 1.6864 | 1.6866 | -0.00014 | Adopt needs over 0.0007 better |
| Clean macro, 65K | 1.6474 | 1.6475 | -0.00014 | Adopt needs over 0.0007 better |
| Clean macro, 262K | 1.6449 | 1.6451 | -0.00015 | Adopt needs over 0.0007 better |
| MRCR full NLL | 0.0253 | 0.0263 | -0.0010 [-0.0015, -0.0006] | No threshold |
| MRCR recovery | 0.9827 | 0.9831 | -0.0004 [-0.0012, +0.0004] | No threshold |
| MRCR distractor NLL | 1.0653 | 1.0652 | +0.0001 [-0.0005, +0.0007] | |
| Train cross-entropy, 100 steps | 0.8705 | 0.8708 | -3.20e-4 | |
| Grad norm, median | 0.207 | 0.190 | | |
| Grad norm, largest | 83.56 | 6.31 | | |
| Steps clipped, all 100 | 5 | 7 | | |
| Step time, median | 29.4 s | 29.2 s | | |
| Peak memory | 132.52 GiB | 132.52 GiB | | |

Unexplained gradient spike: the grad norm reached 83.56 at step 390318, against 0.186 in the control on the same step. It is the largest value in all eight 11-rack arms; the next largest is 19.65. The frozen arm's next-largest value is 5.74. Clipping scaled the step to norm 1.0. Train cross-entropy on that step is 0.9266 against 0.9269, about 0.1 above the steps next to it in both runs. Drops are 2.12e-4 against 2.25e-4. The next step's grad norm is 0.317. We have not found the cause, and one run cannot show whether freezing caused it. Because that is open, we cannot rule out a spike like this in an unfrozen run.

Interpretation:

- Clean macro is lower on 13 of 19 subsets at 262K, but the change is within checkpoint jitter of 0.0002 to 0.0007. `macro_loss` moves more at 65K and 262K (-0.0011, -0.0008), mostly from ptb (-0.0162, -0.0099), which we leave out of clean macro and which goes the other way at 4K (+0.0107).
- Full NLL is lower, with an interval that excludes zero, and the rule set no threshold for it. As in the launch-scale arms, the interval excludes seed and checkpoint noise. Recovery does not separate.
- Train cross-entropy is below the control on 99 of 100 steps. The mean gap of -3.20e-4 is larger than the one-rack repeat band of 1.2e-4 to 1.5e-4. It is -3.63e-4 over the first 20 steps and -2.55e-4 over the last 20.
- Drops are higher on 63 of 100 steps, and the largest 10-step ratio is 1.19. The fitted change over 100 steps is +9.3e-6 frozen and -8.9e-6 control, small against step-to-step scatter. 100 steps cannot show long-run drops. W&B logs only total drops, so we have no per-layer view.
- The spike at step 390252 remains with the bias held: 5.74 against 6.31. So a change of bias at that step does not explain it.
- The frozen arm also logs the norm the solver would have applied: mean 73.94, range 69.62 to 80.05, with no trend across 10-step means. The solve depends on the current bias through each token's threshold, so this is the response to the frozen bias on these weights. It does not stand in for the unfrozen trajectory.

### [PENDING: first finalist run ID, links, launch time and results]

### [PENDING: second finalist results and comparison]
