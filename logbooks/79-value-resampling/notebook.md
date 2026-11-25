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

The research tree includes the [trajectory converter](../../experiments/post_training/goal_value/data.py), [frozen-backbone trainer](../../experiments/post_training/goal_value/train.py), and [typed-decision evaluator](../../experiments/post_training/goal_value/typed_decisions.py). The converter keeps one copy of each clean trajectory and indexes decision boundaries. Its goal combines the task with a public description of what counts as success. The trainer learns a linear head from binary outcomes and evaluates calibration and discrimination on task-disjoint splits.

The offline comparisons include released Clef and released and adapted Kev, which return yes/no probabilities, and frozen generative verifiers at 4B and 14B, which return PASS/FAIL probabilities after a short analysis. The paired terminal analysis uses the same held-out attempts and measures predictions before the last validation action, after its visible output, and around submission. The environment-selection analysis scores the goal before any agent action. The particle-selection analysis compares two attempts from the same task and checkpoint at the same completed-action count.

## Entries

### 2026-11-25 UTC — Define the continuation trial

A branch must restore both its agent history and the workspace state visible at the cutoff. Prefer replay from an identical task image, with explicit checks for deterministic observations and file state. Run a feasibility trial through the patch verifier before expanding the paired comparison. Record the actor checkpoint, benchmark version, replay coverage, action and token allocations, and actual consumption with the experiment source revision.

### 2026-11-25 UTC — Implement the paired continuation pilot

The runner replays four archived Bash actions in a private workspace at the verified instance base commit. It checks the recorded exit codes, scores the live observations with the frozen value head, and restores a chosen branch by replaying it again. The selected continuation must reproduce the reference workspace diff hash and action exit codes. The actor is frozen Qwen3-14B at `40c069824f4251a91eefaf281ebe4c544efd3e18`. The pilot gives each arm one continuation, paired seeds, up to 20 model calls, and up to 6,144 generated tokens. If both arms select the same branch, they share that branch's seeded continuation. Account for this shared continuation as equal allocated budget.

The verifier runs in a separate fresh workspace with SWE-bench 4.1.0 and the Verified dataset at `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. The actor receives the issue and observed history. Test fixtures enter only the verifier workspace. Feasibility runs check image availability, replay, action generation, patch extraction, and grading before the paired sample expands.

Run `python -m experiments.post_training.goal_value.resampling MANIFEST HEAD OUTPUT --actions 20 --tokens 6144 --replicates 1` with Python 3.12, PyTorch 2.9.1, Transformers 4.57.6, and Daytona SDK 0.200.2. The uncommitted manifest and checkpoint are inputs to the run. The manifest contains issue IDs, two four-action prefixes per issue, and the corresponding recorded exit codes. The checkpoint is the saved 65k scalar readout.

The pilot manifest selects at most six issues by a fixed issue hash from the eligible archive groups. Eligibility requires two prefixes with four Bash actions, a recorded exit code for each action, and matching goal text. A replay mismatch excludes the issue and records its reason. The launcher is [run-pilot.sh](run-pilot.sh).

### 2026-11-25 UTC — Restore the benchmark commit explicitly

The first setup run reached the workspace revision check. The public SWE-bench image starts at a packaging commit titled `SWE-bench`. A fresh sandbox probe found the benchmark commit in that image and verified that it has the same tracked file tree. The detached checkout succeeded. The runner now verifies tree equivalence and checks out the benchmark commit before replay. It also records prompt tokens and time spent scoring prefixes. Budgets and case selection stay fixed for the retry.

### 2026-11-25 UTC — Check grading and evaluate both branches

The initial comparison exercised the patch grader. Public reference fixes passed as positive controls in two disposable verifier workspaces. Some image packaging commits included executable-bit differences. A detached checkout restored the exact benchmark revision with a clean working tree. The runner checks cleanliness after that checkout.

The second pilot runs two seeds on each branch with the same frozen actor and caps. It enables the actor's reasoning mode and activates the benchmark Python environment in its shell. It records parse failures, command failures, submission status, and patch size alongside token consumption. The head selects one branch per issue. The uniform baseline is the mean outcome across the two branches for each seed. The guided arm uses the same one-branch continuation allocation per seed. Evaluating both branches measures the uniform policy's expectation directly and also exposes the frequency of mixed outcomes across seeds.

Run the current launcher with the same manifest and head. The initial one-seed, non-reasoning sample remains reproducible at the source revision of that run.
