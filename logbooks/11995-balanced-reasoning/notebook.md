---
title: "Difficulty-banded balanced reasoning traces for SFT"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin/issues/11995"
related_issues: ["https://github.com/marin-community/marin/pull/11997"]
research_branch: "codex/11995-balanced-reasoning"
marin_base_revision: "f3b4338d2fcfaa0515a0e251754c0d6ac1915ed4"
archive_template_revision: "fc1cb30da0967f6ab78ffd47444ed9144f50cba9"
status: active
---

# Difficulty-banded balanced reasoning traces for SFT

This branch preserves the source tree of draft marin#11997 as a durable experiment branch. The lane's output reaches a mix as a versioned source repair through the Atlas queue, not as merged code, so the PR is closed on main. Further lane commits land here.

## Question and decision criteria

- Goal and why it matters: the Mochi policy overthinks on Nemotron-style prompts; its responses are bimodal (finish near 2k tokens or run to the 16k cap with an open think block). SFT data whose reasoning length scales with task difficulty, and that ends with a committed answer or stated uncertainty, should fix that at the source.
- Baseline and alternatives: current SFT mixes, whose competitive-programming stores carry p50 18.6k-22.3k assistant tokens; the Exp6 stopgap of dropping traces over 16k.
- Evidence needed to choose: per-band solve rates against the task's own grader, graceful-form pass rate, token cost per band, and an eval of a mix that includes the store.
- Constraints and unresolved questions: generation runs on the Ortet GLM-5.3 relay; 4,701 census tasks cannot be lowered because their graders need container images the specs lack.

## Baseline

- Date: 2026-11-24
- Published source revision: f3b4338d2fcfaa0515a0e251754c0d6ac1915ed4
- Configuration, command, and external inputs: `experiments/post_training/task_curation/balanced_reasoning/` on the taskforge effect-driven stack (#11109-#11112, #11947); GLM-5.3 on the Ortet relay; difficulty evidence ledger with a source-pooled fallback.
- Environment and hardware when relevant: Iris driver job on cw-us-east-08a.
- Result and evidence: see the pilot entry.
- Limitations: the hosted difficulty ledger has about one attempt per task, so banding pools all four pinned solvers.

## Entries

### 2026-11-24 13:30 UTC - 100-task pilot

- Question or hypothesis and reason for trying it: does the lane produce verified, band-appropriate traces at an acceptable cost before the full run?
- Published source revision: f3b4338d2fcfaa0515a0e251754c0d6ac1915ed4
- Command or material configuration: `/benfeuer/balanced-reasoning-pilot`, 100 tasks (40 easy / 35 medium / 20 hard / 5 unsolved), 16 streams.
- External inputs and artifact references: `s3://marin-us-east-02a/marin/users/benfeuer/balanced-reasoning/`
- Environment and hardware when relevant: cw-us-east-08a; Ortet relay bulk pool.
- Result, including failure or restart: 100/100 tasks completed with zero relay retries. Solve rates: easy 1.00, medium 0.971, hard 0.90, unsolved band 0.80. Graceful-form pass rate 1.00 (21 best-effort and 2 uncertain resolutions). Output 758.8k tokens. Two defects fixed during the run: taskforge's default author model was deepseek-v4.1-flash (pinned to glm-5.3; affected attempts discarded), and the relay rejected empty tools arrays.
- Evidence: PILOT-REPORT in the lane's working directory; event logs under the pilot run root.

### 2026-11-24 13:45 UTC - v1 store submitted

- Question or hypothesis and reason for trying it: build the first store at scale.
- Published source revision: f3b4338d2fcfaa0515a0e251754c0d6ac1915ed4
- Command or material configuration: `/benfeuer/balanced-reasoning-v1`, 32 streams, selection digest 294186df.
- External inputs and artifact references: `s3://marin-us-east-02a/marin/users/benfeuer/balanced-reasoning/v1/`
- Environment and hardware when relevant: cw-us-east-08a.
- Result, including failure or restart: running. Selection 8,131 tasks: easy 4,000/4,000, medium 4,000/4,000, hard 67/2,500, unsolved 64/1,500. The hard and unsolved shortfall comes from small arc and boolq pools and from tasks whose specs lack grader images.
- Evidence: the selection manifest above.

### 2026-11-24 17:10 UTC - v1 store verified and published

- Question or hypothesis and reason for trying it: the run finished in about one hour against a 13-15 h estimate; verify nothing was skipped before building the store.
- Published source revision: 3e14dded35 (this branch; carries the publish-prefix fix the store build used)
- Command or material configuration: `/benfeuer/balanced-reasoning-v1` (driver), `/benfeuer/balanced-reasoning-v1-publish-2` (store build + publish), `/benfeuer/balanced-reasoning-v1-verify-2` (verification).
- External inputs and artifact references: selection `294186df`; store `s3://marin-us-east-02a/marin/users/benfeuer/balanced-reasoning/v1-store-2026.11.24.1/` (rows.parquet sha256 6b5dc656…, receipts.jsonl, manifest.json); private dataset version [open-athena/balanced-reasoning-v1](https://huggingface.co/datasets/open-athena/balanced-reasoning-v1).
- Environment and hardware when relevant: cw-us-east-08a; Ortet relay bulk pool at 32 streams.
- Result, including failure or restart: the run attempted all 8,131 selected tasks and recorded an attempt.json for each - nothing was skipped or deferred. The speed is explained by the mix, not by skips: easy and medium bands are 7,989 of 8,131 tasks and their selected traces are short (easy p50 172 assistant tokens, medium p50 238), so per-task cost is far below the pilot's hard-band extrapolation the estimate was based on. Output 33.7M tokens. Rows: 7,728 (easy 3,917 / medium 3,695 / hard 64 / unsolved 52); the 403 tasks with no passing attempt contribute no row. Resolutions: 6,131 solved_shortest, 1,575 graceful_best_effort, 22 graceful_uncertain, 403 unsolved. Every selected trace finished with finish_reason=stop: zero cap-hits. Grader receipts exist for all 7,728 rows (60 spot-checked receipt roots, none missing); 6,448 selected traces also passed the grader outright, including 317 graceful best-effort answers that were in fact correct. The word-13-gram blend scan dropped zero rows. The store publishes as the private open-athena dataset version above; its Atlas-queue registration convention is the one handoff still open (the queue mechanism is owned by the Atlas operators; the store URI and manifest digest are ready for it).
- Evidence: verification job log (/benfeuer/balanced-reasoning-v1-verify-2); manifest.json in the store prefix; the dataset card carries the manifest summary.

### 2026-11-24 17:40 UTC - store routed as an SFT-mix candidate; licensed and re-published

- Question or hypothesis and reason for trying it: the coordinator ruled the store is an SFT store, not an RL source - it does not go through the Atlas RL queue or the difficulty queue (the brief's "Atlas queue" wording was wrong); its route is the next SFT mix.
- Published source revision: a66d02073d (this branch; separates unsolved tasks from decontamination drops in the manifest)
- Command or material configuration: `/benfeuer/balanced-reasoning-v1-publish-3` rebuilt and republished the store; the HF dataset received the license commit.
- External inputs and artifact references: store `s3://marin-us-east-02a/marin/users/benfeuer/balanced-reasoning/v1-store-2026.11.24.2/` (rows and receipts byte-identical to .1, manifest corrected: dropped_by_decontamination 0, unsolved_no_row 403); dataset [open-athena/balanced-reasoning-v1](https://huggingface.co/datasets/open-athena/balanced-reasoning-v1) commit c16c3bf8 adds the first-party OpenMDW 1.1 LICENSE (official text, sha256 2ab44b68…) and the license card metadata; the dataset stays private.
- Result, including failure or restart: the store is ready as an SFT-mix candidate; the coordinator takes it to the user for the next mix. No queue filing is needed.
- Evidence: manifest.json at the .2 prefix; the dataset card and LICENSE on the Hub commit above.

### 2026-11-24 18:00 UTC - v1 cohort analysis: token counts, selection basis, and why it misses the target

- Question or hypothesis and reason for trying it: how large is the v1 store, how was its cohort chosen, and does it address the overthinking it was built for?
- Published source revision: 199b518b2d (this branch)
- Command or material configuration: sums and counts over `rows.parquet` from the published store (`assistant_tokens`, `prompt_tokens`, `reasoning_tokens`, `band`, `source_id`, `difficulty_basis`).
- External inputs and artifact references: `s3://marin-us-east-02a/marin/users/benfeuer/balanced-reasoning/v1-store-2026.11.24.2/rows.parquet` (sha256 6b5dc656130f4f946a793172ef4495d75f67c1d26e30ad813d0b42e3756ec9be); selection digest 294186df.
- Environment and hardware when relevant: local read of the published store.
- Result, including failure or restart:
  - Size: 7,728 rows; 5,827,840 prompt tokens and 2,584,496 assistant tokens (905,677 of them reasoning), about 8.4M tokens in total. The run generated 33.7M output tokens, about 13x the retained assistant tokens, because only the shortest correct trace per task is kept. Against an SFT mix of about 134B tokens (V9), the store is about 0.002% of trainable tokens.

    | Band | Rows | Prompt tokens | Assistant tokens | Reasoning tokens | Mean assistant tokens |
    |---|---:|---:|---:|---:|---:|
    | easy | 3,917 | 1,763,602 | 897,855 | 254,265 | 229 |
    | medium | 3,695 | 3,668,217 | 1,644,879 | 637,855 | 445 |
    | hard | 64 | 130,455 | 25,370 | 9,774 | 396 |
    | unsolved | 52 | 265,566 | 16,392 | 3,783 | 315 |

  - Selection: candidates were Atlas census TaskSpecs whose graders run on the lane's machines (4,701 tasks were excluded because their graders need container images the specs lack). Each candidate was joined by task id to the difficulty ledger. With about one attempt per task, the lane pooled the four pinned solvers, then fell back to the source's average solve rate (DESIGN S0 ladder, level 3). 539 rows (7%) carry per-task difficulty (`ledger:pooled`); 7,189 rows (93%) carry the source average (`source:pooled`). Bands were filled against quotas of 4,000 / 4,000 / 2,500 / 1,500 in digest order, with the shortfall reported, not rebalanced.
  - Composition: every row is a math word problem. Easy is mostly hendrycks_math (1,558), rlvr_math (1,526) and gsm8k (647); medium is numina_math (1,387), dapo_math (1,357), svamp (613) and asdiv (331); hard (64) and unsolved (52) come from the same math sources through their per-task ledger rows.
- Evidence: the counts above, from the store's rows.parquet.
- Conclusion: v1 does not address the motivating failure. The overthinking it targets appears on competitive-programming and Nemotron-style prompts, and the cohort contains no code, instruction-following or agentic tasks. Because 93% of rows use a source-average difficulty, the bands mostly reflect which source a task came from, which leaves the hard and unsolved bands nearly empty. A useful v2 would need graders that run code, instruction-following and agentic tasks (the specs missing images fixed), per-task difficulty from denser difficulty waves or the lane's own k-attempt solve rates, and quotas filled only from per-task difficulty. No v2 is planned at this point.
