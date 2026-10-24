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

Source inspection uses Marin `6d7f94f852e7244dc42c4760b08ab4f66fd2de7a`. The shared binding is `experiments/post_training/inventory/train_quick_loop.py`; its research-only `--arm` selector keeps one artifact entry point for all four treatments.

Once materialization passes, hold fixed the Snowball 100k selection, SFT model and matching tokenizer (`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`, revision `fe666478454e79fe6027a0bab0388d9abd68f51a`), seed 20261020, 128 prompt groups/update, four answers/group, 24576-token request window and 8192-token response limit. Use 32 learner and 8 inference H100 GPUs. Run one experiment at a time. The screening panel has 100 fixed tasks drawn from the preserved 910-task validation population; both are excluded from training. The remaining 810 tasks are not evaluated or used to select an arm in this screen. Any later confirmation on the full 910 needs a separate design.

| Arm | Change from control | Reason |
| --- | --- | --- |
| A | Synchronous reward leave-one-out (RLOO-N), learning rate 1e-6, token-mean loss, rollout age 0, concurrency 128, full-batch cohorts | Establish a fresh control on the actual selection. |
| B | Only learning rate 2e-6 | Test whether the short update budget benefits from larger steps. |
| C | Only sequence-mean loss, weighting generated sequences equally instead of tokens | Test an objective with different length weighting; its gradient scale also changes. |
| D | Rollout age cap 2 and concurrency 192, retaining full-batch cohorts | Test overlapping generation and updates while bounding policy age. |

Concurrency counts in-flight prompt groups, with four sampled answers per group. Rollout age is the learner update number minus the generation-policy update number. D is a scheduling package; it does not isolate the causal effect of age alone. Require observed nonzero consumed ages and overlap, age compliance, and accounting for stale discarded/regenerated work.

Freeze actual task-content and canonical generation/scoring protocol hashes before launch. Preserve complete initial/final task outputs, coverage, family-level reward and response-cap completion. The 20-task-per-family screen gives noisy family estimates. Prior frozen-data training runs are not a control for this new selection.

## Entries

### 2026-10-24 00:20 UTC - Choose the finite screen before launching

The authorized program has four approaches, including async. Source inspection supports the four candidates above. The existing `async_rl.py` CLI changes the data pool, entrypoint, objective and token budgets, so it would confound this comparison. Reuse the current quick-loop binding with narrow arm overrides instead.

Prior async work in [Marin #8955](https://github.com/marin-community/marin/issues/8955) motivates explicit policy-age and wasted-work accounting. Its math workloads do not qualify the current mixed-task path. [Response-cap pilots](https://github.com/marin-community/marin/issues/8936#issuecomment-5558461338) also show why shortening only the async response budget would confound a speed comparison.

No experimental GPU run has launched. First require the exact selection's 100000/100/910 rows, manifest provenance, quotas and heldout/dedup exclusions. Its CPU materialization encountered a native Parquet-reader credential handoff failure; a focused maintained fix is being validated separately. This operational prerequisite is not a training result. Private execution receipts remain on the authorized task surface.

Before the four measured screens, qualify A and D for three updates at the actual geometry and full token budgets. Validate B/C's complete rendered configuration against A, allowing only declared treatment and run-identity differences. Require finite metrics, exact update counts, completed checkpoint/export and terminal success. A failed qualification ends that attempt for diagnosis; no blind retries or larger automatic confirmation program.

### 2026-10-24 - Implement and validate the four recipe choices locally

The selector changes only the declared recipe fields: B doubles the learning rate, C uses sequence-mean loss, and D permits age two with 192 in-flight groups. D retains full-batch cohorts. The model, tokenizer, selection, seed, task-session backends, evaluation, context budgets and physical geometry remain shared. A retains the existing artifact name; B/C/D receive distinct names to prevent collisions. Each launch still requires a verified fresh immutable version, and automatic retries remain disabled.

Complete rendered-recipe comparisons pass for all four arms at both three and 25 updates. All eight source plans also compose through the production frozen SkyRL CPU bootstrap: the resolved entrypoint is TaskCompendium, A/B/C select synchronous execution, D selects asynchronous execution, and every arm retains full-batch cohorts and initial/final evaluation. This is local configuration validation, not GPU qualification, measured training or a materialized-data audit. Neither the source nor dependency lock was repinned.

For local plan review, the shared command is `uv run python -m experiments.post_training.inventory.train_quick_loop --arm <a|b|c|d> --steps <3|25> --selection-uri <verified-selection-root> --version <fresh-calendar-version> --max-concurrent 1`, without `--run`. Concrete commands and rendered plans are retained on the task's authorized artifact surface. They use proposed plan versions; no remote freshness claim follows. Publish the reviewed source/configuration and record its exact commit before execution. No experimental run was submitted during this work.

The scoring/generation configuration can be fingerprinted, but that descriptor does not establish an immutable served judge revision. The configured judge is a model alias; its served revision remains unknown unless independently established. Actual selection content, task grader bytes and environment/image identities must come from the materialization gate. Leave these evidence limits explicit instead of treating a configured fingerprint as complete runtime equivalence.

### 2026-10-24 - Include the maintained native reader prerequisite

The research source includes the mounted-credential Parquet reader correction from [Marin #10629](https://github.com/marin-community/marin/pull/10629), source `480fa5bd650dce1394d6c2892d9576e20213786f`. All 59 affected tests, including the native-read regression, passed with the frozen Polars 1.42.1 dependency. The merge preserves the launcher and test from [the reviewed research configuration](https://github.com/marin-community/marin-logbooks/tree/bb2026330214b6d0d8863fef5f8e87a1083bdbf2/experiments/post_training/inventory) and keeps inherited archive CI removed. The correction is a shared prerequisite, with no treatment-specific data or scoring change. Live materialization and GPU qualification remain separate gates.

### 2026-10-24 02:00 UTC - Qualify named-file credential delivery before GPU submission

A local preflight reproduced a missing-credential error with synthetic named secret files: the training launcher looked only in process environment variables. The [credential-reader correction](https://github.com/marin-community/marin-logbooks/commit/4189afacbde8a13e5dd8b10298ecd245d0404d33) resolves configured sandbox and judge keys through Rigging and forwards them through Iris named secrets. SDK consumers read the mounted files, with environment fallback. Checkpoint export skips credential resolution and excludes configured API keys from ordinary task environment settings. Recipes, treatment knobs, model, tokenizer, data and geometry are unchanged.

Independent review rejected an initial callback edit that removed an already inherited credential variable, breaking direct environment-only callers. The corrected callback preserves that fallback and passes the existing credential name into Harbor. It adds no renamed API-value variable or temporary credential transport.

Frozen CPU validation passed 83 focused SkyRL cases and 131 focused Harbor cases. After the final export-filter change, 21 affected backend cases passed, including training/export with file/environment delivery. The maintained Harbor lane passed all 988 offline runtime and three acquisition cases. Both package locks validate; their package versions and sources are unchanged. These checks used synthetic keys and made no grading, sandbox or model calls. They qualify the local credential path, not a GPU execution.

The Snowball task-curation producer has passed the native S3 authentication failure described in [Marin #10629](https://github.com/marin-community/marin/pull/10629) and is processing its 13 source graphs. It must still produce the versioned selection of 100000 training, 100 screening and 910 full-validation tasks in the regional artifact store. Bounded readback will check admission provenance, counts and schema, and hash the actual 100-task screening cohort. Whole-training-set guarantees rely on the maintained producer. No experimental GPU run has launched; this data gate still blocks all six planned qualification/screening submissions.

### 2026-10-24 03:30 UTC - Cover shared credentials in legacy task consumers

Review of [Marin #10633](https://github.com/marin-community/marin/pull/10633) found that Harbor task factories and Nemotron judge adapters still read configured credential names only from environment variables. Removing shared names from ordinary task environment settings exposed these missed readers. The correction also covers both Verifyit checker subprocesses, whose serialized judge configuration bypasses the parent judge resolver.

Harbor now resolves explicit train and eval credential names through the existing named-secret mechanism. When both names are nonempty, the launcher skips its otherwise unused canonical environment/Secret Manager lookup. Partial and default configurations retain that fallback. Checkpoint export and inactive task paths retain their existing behavior. The redundant launcher file-state matrix remains removed because Rigging already tests those rules.

All 104 focused frozen CPU cases pass, including actual launch-request construction, synthetic child-file delivery and both bounded Verifyit wrappers using a loopback HTTP endpoint. A fresh advisory review reported no findings. The tests use synthetic credentials; no experimental GPU run, grading service or sandbox was started. The initial credential-path checks missed these active legacy consumers, so the earlier local validation did not establish coverage of the full launch path.

The research branch preserves the [selected source correction](https://github.com/marin-community/marin-logbooks/commit/5c0fb4d0829f41be6ea5995f1bf2aa53f4081872) and test cleanup from the maintained contribution branch. Model, tokenizer, data recipe, treatment parameters and geometry are unchanged. Materialization and bounded cohort readback remain pending. Rebind the launch request to the published corrected source before any qualification submission.

## Final decision

Pending materialization, reviewed executable configurations, qualification and measurements. No approach is selected for promotion yet. Publish exact source/configuration before executing each experiment, then append results and commit-pinned evidence. Preserve rejected alternatives and meaningful failed attempts. Extract only maintainable selected changes into ordinary Marin PRs.
