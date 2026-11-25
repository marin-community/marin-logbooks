---
title: "Value-guided resampling for SWE-bench rollouts"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/79"
related_issues: []
research_branch: "codex/79-value-resampling"
marin_base_revision: "2fa51cb69fe524239edb59660b4386b8394cc69c"
archive_template_revision: "fc1cb30da0967f6ab78ffd47444ed9144f50cba9"
status: active
---

# Value-guided resampling for SWE-bench rollouts

## Question and decision criteria

Does a frozen goal-conditioned value head improve pass rate by choosing which partial agent rollout receives the continuation budget? Compare the head's choice with a uniform choice from the same two branches on each held-out SWE-bench Verified issue. Use the same actor and paired seeds, with equal action and model-token limits. Grade fresh patches and report paired gains, losses, and actual cost.

## Baseline

The scalar head reads a frozen Qwen3-4B-Instruct-2507 state before an agent action. The input contains the issue, the public success criterion, and observed agent actions and tool results. Training groups splits by issue and uses binary terminal verdicts. The long-context experiment preserves the goal and recent history within 65,536 tokens.

The offline diagnostic compares goal-only difficulty, terminal states, and same-issue prefixes at fixed action counts. Useful checks include a tiny balanced memorization fit and within-issue pass/fail concordance. The controlled experiment uses a cut after four completed actions and changes how branches are selected.

## Entries

### 2026-11-25 UTC — Define the continuation trial

A branch must restore both its agent history and the workspace state visible at the cutoff. Prefer replay from an identical task image, with explicit checks for deterministic observations and file state. Run a feasibility trial through the patch verifier before expanding the paired comparison. Record the actor checkpoint, benchmark version, replay coverage, action and token allocations, and actual consumption with the experiment source revision.
