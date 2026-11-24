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
