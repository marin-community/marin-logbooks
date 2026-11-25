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

### 2026-11-25 UTC — Implement the paired continuation pilot

The runner replays four archived Bash actions in a private workspace at the verified instance base commit. It checks the recorded exit codes, scores the live observations with the frozen value head, and restores a chosen branch by replaying it again. The selected continuation must reproduce the reference workspace diff hash and action exit codes. The actor is frozen Qwen3-14B at `40c069824f4251a91eefaf281ebe4c544efd3e18`. The pilot gives each arm one continuation, paired seeds, up to 20 model calls, and up to 6,144 generated tokens. If both arms select the same branch, they share that branch's seeded continuation. Account for this shared continuation as equal allocated budget.

The verifier runs in a separate fresh workspace with SWE-bench 4.1.0 and the Verified dataset at `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. The actor receives the issue and observed history. Test fixtures enter only the verifier workspace. Feasibility runs check image availability, replay, action generation, patch extraction, and grading before the paired sample expands.

Run `python -m experiments.post_training.goal_value.resampling MANIFEST HEAD OUTPUT --actions 20 --tokens 6144 --replicates 1` with Python 3.12, PyTorch 2.9.1, Transformers 4.57.6, and Daytona SDK 0.200.2. The uncommitted manifest and checkpoint are inputs to the run. The manifest contains issue IDs, two four-action prefixes per issue, and the corresponding recorded exit codes. The checkpoint is the saved 65k scalar readout.

The pilot manifest selects at most six issues by a fixed issue hash from the eligible archive groups. Eligibility requires two prefixes with four Bash actions, a recorded exit code for each action, and matching goal text. A replay mismatch excludes the issue and records its reason. The launcher is [run-pilot.sh](run-pilot.sh).
