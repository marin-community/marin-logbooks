# Notebook: do reasoning-length and give-up filters reduce runaway reasoning?

Working record for the data-patch experiment that follows marin-community/marin#11188.
The approved plan and decision rule are summarised in the first entry; later entries are appended.

## 2026-11-13: question, baseline and design

**Question.** The 10.05 SFT model runs past 100k characters on 27-32% of AIME24 outputs and 45-53% of
OlympiadBench outputs, and most of those outputs are wrong. Does removing long-reasoning and give-up
documents from the SFT mix reduce this?

**Baseline evidence** (existing eval rollouts, scanned with the scripts in gist
`5ce8d65b840d6a7c273977fde8b9a1cf`; settings temperature 0.7, 73,664-token cap):

| Checkpoint | AIME24 outputs >= 100k chars | AIME24 no boxed answer | OlympiadBench outputs >= 100k chars |
|---|---|---|---|
| Repro step 222 | 37.0% | 25.3% | 53.3% |
| Repro step 471 | 43.0% | 29.7% | 50.7% |
| Repro step 1024 | 34.3% | 20.7% | 47.2% |
| Repro step 2048 | 32.3% | 18.7% | 51.3% |
| 10.05 (seven runs) | 27-32% | 12-17% | 45-53% |

The habit is at full strength by step 222 of 4,750, so a 222-step run is a valid readout.

**Design.** Three arms of the historical 262,144-context cold-start recipe, 222 updates each, same seed
and weights, differing only in the cache catalog:

| Arm | Catalog |
|---|---|
| `control` | `full-mixture/2026.10.24.2/cache_catalog.json`, unchanged |
| `cap32k` | `math_v3`, competitive coding (Python, C++) and OpenThoughts4 with documents over 32,000 reasoning tokens removed |
| `cap32k-giveup` | as `cap32k`, plus give-up endings removed from `math_v3` and competitive coding |

**Decision rule.** Primary: the share of outputs >= 100k characters on AIME24 and OlympiadBench at step 222
is at most half the control's, and the gap exceeds the gap between the fresh control and the published
step-222 checkpoint. Guards: AIME24, OlympiadBench, MATH-500, GSM8K, HumanEval+ and IFBench accuracy show
no decisive loss under the #10197 pairwise rule.

**Changes from the approved plan.**

- Filtering is done on the tokenized stores (`reasoning_store_filter.py`), not by rebuilding sources. Kept
  documents are byte-identical to the control's.
- The third arm does not include the Terminus or multilingual hygiene patches; they are not built.
- Replay junk removal is out of scope; it is tested under marin-logbooks#40.
- Permanent checkpoints land at steps 100, 200 and 222 (the recipe keeps every 100), not 50, 100 and 222.

## 2026-11-13: filter audit on sample shards

Source commit `c6a5b1d520` (filter), dry runs from a workstation on shards `math_v3` 7 and 101,
`competitive_coding_python` 13, `competitive_coding_cpp` 29, OpenThoughts4 0.

Reasoning length (longest think span, tokens):

| Source | Rows over 32k | Tokens in those rows |
|---|---|---|
| `math_v3` | 8% | 31% |
| competitive coding | 51% | 84% |
| OpenThoughts4 | 78% | 95% |

Give-up rules, read by hand (12 hits per shard):

- Coding: about 21 of 24 real ("I cannot produce a correct solution", placeholder code). Two false positives
  came from `# placeholder` in ordinary code and "hope it's okay"; both patterns were narrowed afterwards.
- Math: the rule mostly matches "Given the time, I'll ...". About 17 of 24 are unsupported guesses; the rest
  stop verification early after a real derivation. Hit rate 3.7-3.9% of rows and 8.0-8.6% of tokens. The rule
  is therefore defined as "reasoning ends by citing time pressure or guessing", not "answer is wrong".
- Math give-up hits are short (median 4.3k reasoning tokens); coding hits are long (median 86k). Give-up and
  length are separable in math and not in coding.

**Interpretation.** A 32k cap is a large intervention on coding and OpenThoughts4. It is kept for this
experiment because a strong dose gives the clearest test. For a production mix it implies reweighting.

## 2026-11-13: filtered stores

Experiment commit `5b48d94244`. Jobs `/dml/sft-filter-<variant>-<source>` on `cw-us-east-08a`; outputs under
`s3://marin-us-east-02a/tmp/ttl=30d/sft-data-patch-11188/stores/<variant>/<source>/train`, each with a
`selection.json` listing every dropped row and its reason.

| Store | Rows kept / total | Tokens kept / total |
|---|---|---|
| `cap32k/ot4` | 8,264 / 36,666 | 0.169B / 3.354B |
| `cap32k/cp_python` | 163,637 / 333,399 | 2.091B / 13.056B |
| `cap32k/cp_cpp` | 164,092 / 331,927 | 2.133B / 12.983B |
| `cap32k-giveup/cp_python` | 162,806 / 333,399 (831 give-up) | 2.079B / 13.056B |
| `cap32k-giveup/cp_cpp` | 162,889 / 331,927 (1,203 give-up) | 2.120B / 12.983B |

With weights pinned to the control's, 222 steps draw about 0.21B positions from OpenThoughts4, more than
the 0.169B the filtered store holds, so that source wraps about 1.2 times in the patched arms.

## 2026-11-13: cluster and priority

The long-loop runner pins `cw-rno2a` at production priority. Reno had 3 of 504 H100 free with 376 held by
production work, so no band could place the gang there. `cw-us-east-02a` had 22 free and 224 held by batch
jobs. The arms run there on 16 nodes (128 H100, the geometry of the original step 0-471 run) at the default
interactive priority, through `patch_arm.py`, which calls the long-loop's `sft_run` directly.

## 2026-11-13: launch attempts for the control arm

| Attempt | Cluster, priority | Outcome |
|---|---|---|
| `/dml/sft-patch-11188-control` | `cw-us-east-02a`, interactive | Never accepted. The peer accepted no handoff submitted after 18:04 UTC (Echo wiki 984). Cancelled at 19:41 UTC. |
| `...-control-cw-rno2a` | `cw-rno2a`, interactive | Gang started 19:44 UTC, evicted 19:57:58 UTC by the 256-H100 production gang `/benfeuer/long-loop-exp2-full-2026.11.13.11` before the first step. |
| `...-control-cw-rno2a-r2` | `cw-rno2a`, interactive, resumable | Gang never admitted behind production work. Cancelled when production priority on Reno was approved. |
| `...-control-cw-rno2a-r3` | `cw-rno2a`, production | Step 1 committed at 20:33 UTC, then `TypeError: wait_at_barrier(): incompatible function arguments` in the first-checkpoint restore check. Cancelled at 20:54 UTC. |
| `...-control-cw-rno2a-r4` | `cw-rno2a`, production | Running from commit `ae0bc4fb54`, which has the checkpoint fix; resumes from the step-1 checkpoint. |

**Changes made because of these attempts.**

- `patch_arm.py train` restores from the arm's own checkpoints and carries a preemption retry budget, so an
  eviction costs at most the time since the last half-hourly checkpoint.
- `lib/levanter/src/levanter/checkpoint.py` rounds the async checkpoint timeout to an integer. The long-loop
  branch made it a float config field; JAX multiplies it by 1000 and passes it to `wait_at_barrier`, which
  rejects a float. Main passes an integer constant and does not have the bug.

**Evaluation recipe.** The eval launcher on this branch caps OlympiadBench output at 32,768 tokens, below the
baseline's average of 38.9k generated tokens. Evaluation therefore uses the baseline launcher at `e0cd563`
with the campaign's six Evalchemy configs (AIME24 and OlympiadBench at temperature 0.7, seed 42, 10 repeats;
MATH-500, GSM8K 0-shot, HumanEval+, IFBench at the server default), served DP8 on one 8xH100 node with
`max_model_len` 73,728. The published step-222 checkpoint was evaluated without a `generation_config.json`;
these exports write one with stop tokens, so comparison with the published numbers is looser than between arms.

## 2026-11-14: control arm finished and evaluated

`/dml/sft-patch-11188-control-cw-rno2a-r4` committed permanent step 222 at 05:55 UTC (8 h 49 min of training,
no failures or preemptions; final logged loss 0.583). Exported in the stacked-expert layout to
`runs/control/serve-step222/hf` and evaluated as run group `sft11188-control-s222` on Reno.

| Benchmark | Fresh control, step 222 | Published repro, step 222 |
|---|---|---|
| AIME24 accuracy / outputs >= 100k chars / no boxed | 71.3% / 33.3% / 24.3% | 72.3% / 37.0% / 25.3% |
| OlympiadBench accuracy / outputs >= 100k chars / no boxed | 22.3% / 48.3% / 24.7% | 25.0% / 53.3% / 22.3% |
| MATH-500 | 95.0% | |
| GSM8K 0-shot | 81.2% | |
| HumanEval+ pass@1 | 68.3% | |
| IFBench strict prompt | 19.7% | |

The fresh control reproduces the published checkpoint within 3.7 points (AIME24) and 5.0 points
(OlympiadBench) on the long-output share. Passing the primary rule therefore needs at most 16.7% (AIME24)
and 24.2% (OlympiadBench) long outputs in a patched arm.

**Rehearsal on control step 100** (run group `sft11188-control-s100b`): AIME24 69.7% / 36.0% / 26.3%,
OlympiadBench 22.0% / 49.0% / 23.7%, MATH-500 92.8%, GSM8K 86.3%, HumanEval+ 65.2%, IFBench 19.7%. The habit
is at full strength by step 100.

**Pipeline fixes found by the rehearsal.**

- Export memory peaks at about 420 GB; 64 GB and 400 GB were OOM-killed. CPU nodes on `cw-us-east-02a` are
  full of other jobs' file cache and stall allocation (40+ minutes); a Reno GPU node restores in 5-12 minutes.
- The baseline launcher at `e0cd563` is rejected by the controller as too old; evals run from main at
  `12f0ceff06`, the last commit with the Evalchemy route. The six benchmark configs are unchanged.
- The eval stack's vLLM release rejects per-expert tensors (`Unexpected GrugMoE weight
  'model.layers.2.mlp.experts.72.down_proj.weight'`); `patch_arm.py export` now writes stacked tensors.

## 2026-11-14: all arms trained to step 222 and evaluated

| Arm | Training job | Step 222 committed | Wall time | Failures / preemptions |
|---|---|---|---|---|
| `control` | `/dml/sft-patch-11188-control-cw-rno2a-r4` | 05:55 UTC | 8 h 49 min | 0 / 0 |
| `cap32k-giveup` | `/dml/sft-patch-11188-cap32k-giveup-cw-rno2a-r4` | 14:40 UTC | 8 h 33 min | 0 / 0 |
| `cap32k` | `/dml/sft-patch-11188-cap32k-cw-rno2a-r4` | 16:26 UTC | 8 h 27 min | 0 / 0 |

Exports: `runs/<arm>/serve-step222/hf` (stacked-expert layout, Reno GPU node, about 15 minutes each).
Evaluation run groups: `sft11188-<arm>-s222` on Reno, launcher at main `12f0ceff06`, campaign Evalchemy configs.

**Primary readout** (temperature 0.7, 73,664-token cap; scan script from gist `5ce8d65b840d6a7c273977fde8b9a1cf`):

| Step 222 | AIME24 acc | AIME24 >= 100k chars | AIME24 no box | OlympiadBench acc | OlympiadBench >= 100k chars | OlympiadBench no box |
|---|---|---|---|---|---|---|
| control | 71.3% | 33.3% | 24.3% | 22.3% | 48.3% | 24.7% |
| cap32k-giveup | 71.3% | 40.0% | 24.3% | 25.0% | 51.7% | 29.0% |
| cap32k | 77.0% | 32.0% | 19.7% | 22.4% | 51.8% | 21.7% |
| pass threshold | | <= 16.7% | lower | | <= 24.2% | lower |

**Guards:**

| Step 222 | MATH-500 | GSM8K 0-shot | HumanEval+ | IFBench strict prompt |
|---|---|---|---|---|
| control | 95.0% | 81.2% | 68.3% | 19.7% |
| cap32k-giveup | 93.8% | 82.5% | 66.5% | 21.7% |
| cap32k | 95.8% | 84.1% | 62.8% | 19.7% |

**Pairwise test** (|d| > 1.96 x combined standard error, `compare.py`): no long-output or no-answer
difference is decisive for either arm, and no guard is decisively worse. The only decisive difference is
cap32k AIME24 accuracy, +5.7 points against a 3.9-point bar; OlympiadBench accuracy for the same arm is
+0.3.

**Conclusion.** Neither patch passes the primary rule. Removing documents whose reasoning exceeds 32,000
tokens from `math_v3`, competitive coding and OpenThoughts4, with or without give-up endings, does not reduce
runaway reasoning in a 222-step cold-start SFT; the long-output share stays at the control's level on both
benchmarks. The patches cost nothing measurable on the guards.

**Limits.** One seed per arm, so run-to-run noise is estimated only from the fresh-versus-published control
pair (3.7 and 5.0 points on the long-output share). 222 of 4,750 steps; the habit is flat across the
published repro steps 222 to 2048, but a late effect cannot be excluded. The filter left about 8% of
training positions in long-reasoning documents from untouched sources (`science_v2/vendor`, `math_v2/high`,
`math_proofs_v2`, `math_v4`, multilingual math), about a third of the control's exposure. Generation used
the eval stack's served defaults for temperature on MATH-500 and the guards.

**Next decision.** A null at two-thirds removal does not show that long reasoning is irrelevant. The
cheapest discriminating follow-up is a cap on every reasoning source, or a decoding-side length penalty,
before changing the production mix. The cap32k AIME24 gain needs a second seed before it is treated as real.

## 2026-11-16: typography-normalized arm (`utf-normalize`)

**Question.** The gpt-oss teacher writes U+202F (narrow no-break space) and U+2011 (non-breaking hyphen)
where other text uses a space or a hyphen, and the step-222 control reproduces both in its eval outputs.
Each is usually one token (`âĢ¯`, `ĠâĢ¯`, `âĢĳ`), and the word after a U+202F is split non-canonically
(`110 miles` -> `110`, U+202F, `m`, `iles`). Does replacing them in the SFT data remove the habit from
outputs, and does it change accuracy?

**Arm.** Every SFT store of the control catalog, every turn, U+202F -> space and U+2011 -> hyphen; replay
untouched. Same recipe, initializer, 222 steps and 16 Reno nodes as the control; only the catalog differs.
Compared with `control-s100b` and `control-s222` (the control is reused, not retrained).

**Normalizer.** `experiments/sft_new_mix/typography_store_normalize.py`. Occurrences are found on the row's
bytes because U+2011 is sometimes split across tokens; the words around each occurrence are decoded,
replaced and re-encoded, so replacement text gets the tokenization a fresh encode gives. Each rewritten row
must decode to its source text with only those characters replaced, keep its special tokens, and hold no
target character. A 900-row read-back of a written store found no mismatch.

| | Value |
|---|---|
| SFT stores rewritten | 123 of 216 |
| Rows rewritten | 7,143,608 of 21,245,136 |
| U+202F / U+2011 replaced | 32,459,383 / 45,939,268 |
| SFT tokens | 234.42B -> 234.36B |
| Most affected | `instruction_following_chat_v3/instruction_following` 87% of rows; `opencode_v1/agent_skills` 100%; `finance_v1` 99.6%; `math_v2/*` 71-76%; `math_proofs_v2` 94%; `math_v3` 34% |

Normalization jobs `/dml/sft-typography-normalize-p0`..`p7` (cw-us-east-08a, 32 CPUs each, about 30 min).
Catalog `s3://marin-us-east-02a/tmp/ttl=30d/sft-data-patch-11188/typography/cache_catalog.json`, per-store
counts in `typography/stores/catalog-report.json`. Replay and validation entries and every row count equal
the control catalog's.

**Training.** `/dml/sft-patch-11188-utf-normalize-cw-rno2a`, launched 10:54 UTC, output
`runs/utf-normalize/sft`. Losses at steps 0-9 are within 0.003 of the control's; throughput 505k tokens/s,
as the control.

**Evaluation plan.** Export steps 100 and 222 with `patch_arm.py export`; run groups
`sft11188-utf-normalize-s100` and `-s222` with the launcher at `12f0ceff06` and the same six campaign
Evalchemy configs. The model configs are rebuilt from the `cap32k-giveup-s100` run record and differ from
it only in name and location. Readout: share of outputs containing U+202F or U+2011 (prompt-free), plus the
primary and guard metrics above.

**Hang and restart.** The first training job hung at step 48 (13:12 UTC) with every rank blocked in the
train step and no error, about when the next half-hourly temporary checkpoint was due; it held the gang for
10.5 hours before it was noticed. Cancelled and resubmitted unchanged as
`/dml/sft-patch-11188-utf-normalize-cw-rno2a-r2`, resuming from temporary step 36.

### Step 100 readout (`sft11188-utf-normalize-s100` vs `sft11188-control-s100b`)

Share of outputs containing U+202F or U+2011 (no prompt contains either):

| Benchmark | control | utf-normalize |
|---|---|---|
| MATH-500 | 12.4% | 0.2% |
| GSM8K 0-shot | 6.7% | 0.0% |
| IFBench | 6.0% | 0.0% |
| OlympiadBench | 6.7% | 0.3% |
| AIME24 | 3.3% | 0.0% |
| HumanEval+ | 0.0% | 0.0% |

The residue is one MATH-500 output (`3n‑6`, `n‑1` in a table) and one OlympiadBench output (`⟂ B`).

| Step 100 | AIME24 acc / >= 100k chars / no box | OlympiadBench acc / >= 100k chars / no box | MATH-500 | GSM8K | HumanEval+ | IFBench strict prompt |
|---|---|---|---|---|---|---|
| control | 69.7% / 36.0% / 26.3% | 22.0% / 49.0% / 23.7% | 92.8% | 86.3% | 65.2% | 19.7% |
| utf-normalize | 67.3% / 43.7% / 27.0% | 21.0% / 50.3% / 28.0% | 94.2% | 83.2% | 71.3% | 20.0% |

Only the GSM8K drop (-3.1) clears 1.96 binomial standard errors among the accuracy and length metrics; the
control itself moves -5.1 on GSM8K between steps 100 and 222, so a single-seed difference of this size is
within run-to-run variation. AIME24's long-output share (+7.7) is not decisive (300 samples of 30 prompts).

### Step 222 readout (`sft11188-utf-normalize-s222` vs `sft11188-control-s222`)

Training `-r2` committed step 222 at 07:27 UTC on 2026-11-17; export and eval followed unchanged.

| Benchmark | control | utf-normalize |
|---|---|---|
| MATH-500 | 9.0% | 0.0% |
| GSM8K 0-shot | 6.8% | 0.0% |
| IFBench | 5.0% | 1.0% |
| OlympiadBench | 3.3% | 0.0% |
| AIME24 | 2.7% | 0.0% |
| HumanEval+ | 0.0% | 0.0% |

The IFBench residue is three outputs: two U+2011 compounds (`finite‑order`, `Nigeria‑style`) and a run of
U+202F after emoji.

| Step 222 | AIME24 acc / >= 100k chars / no box | OlympiadBench acc / >= 100k chars / no box | MATH-500 | GSM8K | HumanEval+ | IFBench strict prompt |
|---|---|---|---|---|---|---|
| control | 71.3% / 33.3% / 24.3% | 22.3% / 48.3% / 24.7% | 95.0% | 81.2% | 68.3% | 19.7% |
| utf-normalize | 71.3% / 32.7% / 23.0% | 24.7% / 48.7% / 25.7% | 94.4% | 82.6% | 66.5% | 22.0% |

No accuracy, long-output or no-box difference clears 1.96 binomial standard errors. The step-100 GSM8K drop
and AIME24 long-output rise do not persist.

**Conclusion.** Replacing U+202F and U+2011 in the SFT data removes the habit from outputs: incidence falls
from 3-12% of outputs to 0-1% at both checkpoints. It costs nothing measurable on any of the six benchmarks
and does not change runaway reasoning. The normalization is safe to adopt in the production mix; it is a
cleanliness fix, not an accuracy lever. Limits: one seed per arm; benchmarks are English and math-heavy, so
non-English typography (French number spacing uses U+202F legitimately) was not evaluated.
