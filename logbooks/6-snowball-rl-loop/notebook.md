---
title: "Snowball reward and iteration-time screening"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/6"
related_issues: ["https://github.com/marin-community/marin/pull/10605"]
research_branch: "codex/6-snowball-rl-loop"
marin_base_revision: "6d7f94f852e7244dc42c4760b08ab4f66fd2de7a"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: active
---

# Snowball reward and iteration-time screening

## Question and decision criteria

Which small training or scheduling change offers a useful tradeoff between final evaluation reward and end-to-end iteration time on a fixed mixed-task selection?

Screen four recipes from fresh SFT initialization with one seed and 25 optimizer updates each. A candidate is interesting if it improves the reward/time frontier or signed evaluation reward gain per hour within a matching evaluation cohort. This screen does not establish general improvement or async non-inferiority. Confirmation requires a separate design; it is outside the current finite program.

Time each measured run from submission through terminal artifact success, including scheduling, startup, initial/final evaluation, training, checkpoint and export. Report qualification time and resource cost separately. Incomplete or unknown scoring coverage remains visible and excludes trusted gain ranking.

## Baseline and planned treatments

Source inspection uses Marin `6d7f94f852e7244dc42c4760b08ab4f66fd2de7a`. The shared binding is `experiments/post_training/inventory/train_quick_loop.py`; the four-arm selector is not implemented yet.

Once materialization passes, hold fixed the Snowball 100k selection, SFT model and matching tokenizer (`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`, revision `fe666478454e79fe6027a0bab0388d9abd68f51a`), seed 20261020, 128 prompt groups/update, four answers/group, 24576-token request window and 8192-token response limit. Use 32 learner and 8 inference H100 GPUs. Run one experiment at a time. The screening panel has 100 fixed tasks drawn from the preserved 910-task validation population; both are excluded from training. The remaining 810 tasks are not evaluated or used to select an arm in this screen. Any later confirmation on the full 910 needs a separate design.

| Arm | Change from control | Reason |
| --- | --- | --- |
| A | Synchronous reward leave-one-out (RLOO-N), learning rate 1e-6, token-mean loss, rollout age 0, concurrency 128, full-batch cohorts | Establish a fresh control on the actual selection. |
| B | Only learning rate 2e-6 | Test whether the short update budget benefits from larger steps. |
| C | Only sequence-mean loss, weighting generated sequences equally instead of tokens | Test an objective with different length weighting; its gradient scale also changes. |
| D | Rollout age cap 2 and concurrency 192, retaining full-batch cohorts | Test overlapping generation and updates while bounding policy age. |

Concurrency counts simultaneous rollout requests. Rollout age is the learner update number minus the generation-policy update number. D is a scheduling package; it does not isolate the causal effect of age alone. Require observed nonzero consumed ages and overlap, age compliance, and accounting for stale discarded/regenerated work.

Freeze actual task-content and canonical generation/scoring protocol hashes before launch. Preserve complete initial/final task outputs, coverage, family-level reward and response-cap completion. The 20-task-per-family screen gives noisy family estimates. Prior frozen-data training runs are not a control for this new selection.

## Entries

### 2026-10-24 00:20 UTC - Choose the finite screen before launching

The authorized program has four approaches, including async. Source inspection supports the four candidates above. The existing `async_rl.py` CLI changes the data pool, entrypoint, objective and token budgets, so it would confound this comparison. Reuse the current quick-loop binding with narrow arm overrides instead.

Prior async work in [Marin #8955](https://github.com/marin-community/marin/issues/8955) motivates explicit policy-age and wasted-work accounting. Its math workloads do not qualify the current mixed-task path. [Response-cap pilots](https://github.com/marin-community/marin/issues/8936#issuecomment-5558461338) also show why shortening only the async response budget would confound a speed comparison.

No experimental GPU run has launched. First require the exact selection's 100000/100/910 rows, manifest provenance, quotas and heldout/dedup exclusions. Its CPU materialization encountered a native Parquet-reader credential handoff failure; a focused maintained fix is being validated separately. This operational prerequisite is not a training result. Private execution receipts remain on the authorized task surface.

Before the four measured screens, qualify A and D for three updates at the actual geometry and full token budgets. Validate B/C's complete rendered configuration against A, allowing only declared treatment and run-identity differences. Require finite metrics, exact update counts, completed checkpoint/export and terminal success. A failed qualification ends that attempt for diagnosis; no blind retries or larger automatic confirmation program.

## Final decision

Pending materialization, reviewed executable configurations, qualification and measurements. No approach is selected for promotion yet. Publish exact source/configuration before executing each experiment, then append results and commit-pinned evidence. Preserve rejected alternatives and meaningful failed attempts. Extract only maintainable selected changes into ordinary Marin PRs.
