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

Time each measured run from submission through terminal artifact success, including scheduling, startup, initial/final evaluation, training and native checkpointing. HF export is disabled for all four measured screens; explicit export adds a separate job outside this shared protocol. Report qualification time and resource cost separately. Incomplete or unknown scoring coverage remains visible and excludes trusted gain ranking.

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

### 2026-10-24 05:10 UTC - Complete the new selection and freeze the screening cohort

The new selection materialized successfully. Bounded readback confirmed 100000 training rows, 100 screening rows and 910 full-validation rows in the canonical TaskSpec Parquet format. All 13 source manifests record completed FULL processing with no unprocessed rows, and their admitted counts reconcile with selection exclusions and the unchanged source and family quotas. Quality remains unreviewed: optional model assessment was disabled. Eleven sources passed their declared sampled controls; two declare no controls.

The complete 100-task screening panel parsed and round-tripped without dropping TaskSpec fields. It contains 20 tasks from each of the five families. Content, membership, grader, environment and private-resource declaration hashes are retained in the private execution record for comparison cohorts. The full training and 910-row bodies were not downloaded. Their deduplication, heldout exclusion and field-preservation guarantees rely on the maintained producer and provenance checks. External asset bytes and immutable served judge revisions remain unattested.

Validation initially stopped because its checker expected resolved dependency paths in an artifact field the writer leaves empty. The reviewed correction checks the 14 portable dependency identities, complete materialized input configuration and retained fingerprint instead. Original evidence was preserved. The successful check used the same bounded metadata/footer/panel scope and cumulative read budget; no data rebuild or policy change was needed.

No experimental GPU run has launched. Next bind the recorded cohort and resolved generation/scoring protocol to the pinned runnable source, check fresh run/checkpoint identities, then qualify the synchronous control for three updates before the matched async qualification.

### 2026-10-24 05:55 UTC - Start the synchronous qualification

The completed selection and recorded 100-task screening cohort are bound to the [frozen executable source](https://github.com/marin-community/marin-logbooks/tree/fb4d5d7ee8ea6314a379d35d148767b1046e852a/experiments/post_training/inventory). The resolved generation and scoring configuration is retained with the launch record. A separate comparison descriptor removes only the run-specific trajectory destination, run identifier and declared synchronous/asynchronous mode; it retains sampling, evaluation, harness, retention policy and budgets, and includes physical allocation and parallelism. Future runs still require their own configuration and cohort comparison. Configured equality does not establish the immutable served judge revision, external asset bytes or actual scoring coverage.

The synchronous A qualification was submitted once for three updates. At the first startup check, the coordinator and five eight-H100 replicas were running on attempt zero with no controller failures. The coordinator reused the completed selection, and the SkyRL entrypoint and experiment logger initialized. Worker bundle identity, loaded model identity, learner/inference readiness, optimizer updates, evaluation coverage and checkpoint/export completion remain unverified. This attempt is qualification cost; no measured-arm reward or improvement result is available yet.

### 2026-10-24 06:15 UTC - Hold qualification for launch-lock provenance

The outer `uv run iris` command resolved dependencies after the clean-source check and before Iris bundled the working tree. The root lockfile changed, including Torch package sources and dependency markers. The saved resolver output and file timestamps identify this mutation window. Iris bundles working-tree bytes, so the committed source revision alone does not identify the coordinator's dependency inputs. The original and changed locks, patch and semantic comparison are preserved. No equivalence is assumed. The separate SkyRL project lock and pinned frozen bootstrap remain unchanged locally; the actual submitted bundle and installed runtime still need attestation. Qualification acceptance and subsequent launches are held while that provenance is checked.

At the latest bounded observation, the coordinator and all five GPU replicas remained running on attempt zero with no controller failures. Runtime logs reported initial evaluation on 100 attempted and 100 scored tasks, mean reward 0.2201 and a 0.41 length-stop fraction, followed by PPO training dispatch. These are runtime counters; complete task-level verdict review, optimizer-update completion and terminal checkpoint/export checks remain pending. The attempt is not eligible for a trusted comparison or dashboard result. No retry or restart was performed.

### 2026-10-24 06:50 UTC - Finish the synchronous attempt; retain the qualification hold

The training coordinator, five training replicas and four checkpoint-export replicas all reached terminal success on attempt zero with no controller failures. Logs reported training progress of three out of three steps and published the final HF policy, configuration and tokenizer. The artifact step also reported success. Acknowledgement-to-controller-terminal time was 61.23 minutes, including export; this remains qualification cost.

Runtime evaluation counters reported 100 attempted and 100 scored tasks before and after training, with mean reward moving from 0.2201 to 0.16. The reported length-stop fraction increased from 0.41 to 0.46. These provisional counters do not establish complete task membership, per-task verdicts, checkpoint resumability or a matched comparison. No improvement claim or dashboard ranking follows.

Launch-lock provenance remains unresolved. The coordinator's retained provenance explicitly marks its source as dirty. The original source and changed root lock are preserved; the exact submitted bundle and installed GPU environment have not been independently attested. All jobs are terminal and will not be probed again. Narrow durable result, checkpoint and provenance metadata checks remain pending. The async qualification and all four measured screens stay on hold; no retry, restart or new experiment has been submitted.

### 2026-10-24 08:00 UTC - Make HF export opt-in for future loops

[Marin #10644](https://github.com/marin-community/marin/pull/10644) makes HF export opt-in in the general RL launch configuration and quick-loop CLI. Native recovery checkpoints, in-training evaluation and terminal result records remain enabled. Explicit `--export-hf` retains the separate export job for downstream HF model consumers. The maintained contribution is published and awaiting review; it is not merged.

The research branch includes the [selected source change](https://github.com/marin-community/marin-logbooks/commit/812cd7f4ad0d2aa34f77c22dda65cdc0ab037459), preserving all four arm treatments, artifact names and dependency locks. It also retains the five test deletions requested in review of [Marin #10633](https://github.com/marin-community/marin/pull/10633). The maintained export change passed focused and affected CPU checks. The archive adaptation passed actual CLI and callback-construction checks for default and explicit export, plus lint and independent source review. These checks do not qualify GPU execution.

The completed A3 attempt used export-on and retains its original outcome and 61.23-minute qualification cost. Future D3 and all four measured screens will use export-off with the same submission-to-terminal timing definition. Each future request and resolved protocol must bind the updated source and retention behavior before launch. A3 result and provenance review remain pending; no new experiment has launched.

### 2026-10-24 10:19 UTC - Review retained synchronous qualification results

Retained-result readback confirms the same 100 screening tasks before and after the three-step synchronous qualification, with 20 tasks per family. Each phase has 100 usable numeric verifier scores marked `verified`, including zero and partial credit, and no serialized exception or server-error rows. The mean normalized verifier score fell from 0.2201 to 0.16, a change of −0.0601. This is a descriptive observation on this cohort; it establishes neither improved quality nor a trusted cross-arm ranking.

| Family | Initial mean normalized verifier score | Final mean |
| --- | ---: | ---: |
| Calendar | 0.20 | 0.05 |
| Competitive coding | 0.00 | 0.00 |
| Reasoning gym | 0.4505 | 0.35 |
| STEM | 0.10 | 0.05 |
| Structural | 0.35 | 0.35 |
| All 100 | 0.2201 | 0.16 |

All 100 `passed` fields in each phase are null. A usable numeric score does not establish boolean success, so no pass rate is validated. Aggregate pass-like metrics can fall back to positive reward when no boolean pass verdict is recorded; they are not independent pass verdicts. The cohort linkage uses the sole validation input and source-defined row order, rather than a separate TaskSpec signature recorded with each verdict.

Checkpoint and export publication metadata are consistent: the latest completion marker identifies step 3, the export request and successful export child agree, and the HF manifest lists 45 files with 39 weight shards referenced by the index. Recorded sizes and three fetched small metadata-file hashes agree with the manifest. Weight bodies, the full tokenizer and checkpoint tensor bodies were not read. These checks establish publication and metadata consistency, not full weight integrity, initializer/export architecture parity or optimizer resume.

The reported dirty tracked source variant differs locally from the frozen executable source only in `uv.lock`. The exact submitted bundle, installed GPU environment and loaded modules and weights remain unattested. Retained coordinator and post-staging configurations match the reviewed transformations at their pre-validator writing phases; they do not attest the later active GPU validator state. Immutable served judge revision and external asset bytes also remain unknown.

Saved metrics show applied-update markers at steps 2 and 3. The step-1 marker was not recovered; logged three-out-of-three progress and the step-3 checkpoint do not independently establish all three applied optimizer updates. The retained evidence is accepted with these limits, while scientific qualification remains unaccepted. All A3 jobs and retained readers are terminal. The original export-on attempt retains its 61.23-minute qualification cost; no new experiment has launched. Future D3 and the four 25-update screens remain planned with export off, and all launches stay on hold. No dashboard result is appended.

### 2026-10-24 21:53 UTC - Accept the evidence-limited A3 prerequisite

The user explicitly chose “Accept evidence-limited A3 qualification.” This accepts A3 as the prerequisite for preparing D3, the three-update asynchronous qualification. It changes the acceptance policy, not the scientific evidence. The observed score decline, missing boolean pass verdicts, unattested submitted and loaded runtime identity, absent step-1 applied-update marker and untested optimizer recovery remain as recorded above. No quality improvement, runtime equivalence or trusted arm ranking is established.

D3 preparation may proceed against a clean, frozen source and dependency snapshot with a pinned `uv` executable and outer `uv run --locked` invocation. The preparation must bind the export-off source, resolved configuration, cohort/comparison protocol, request and packaged payload. Submission still requires exact fresh version, artifact, checkpoint and job identities, with lookup errors kept distinct from confirmed absence, no overlapping run, and a durable one-submit record with acknowledgement capture. Preparation is not launch authority; a separate exact submission approval is still required. D3 and all four 25-update screens remain unlaunched, and no dashboard result is appended.

### 2026-10-25 02:09 UTC - Record bounded asynchronous qualification results

D3 requested three asynchronous updates with HF export disabled. Its coordinator reached terminal success, and the retained artifact records success and reports child success. Final child-task states were not directly observed. Invocation-to-coordinator-terminal time was 39.737657 minutes; acknowledgement-to-coordinator-terminal time was 39.622353 minutes. This export-off completion contract is not directly comparable with A3's historical 61.23-minute export-on qualification cost. The original A3 result is unchanged.

Each D3 evaluation endpoint contains 100 usable numeric verifier scores marked `verified`, with no serialized exception or server-error rows. Dataset UIDs link to the cached cohort mapping, which attributes 20 rows to each family. This is cached attribution, not fresh attestation of input bytes, row order, filtering loss or a TaskSpec attached to each verdict. Mean normalized verifier score fell from 0.28928048 to 0.22114198, a change of −0.06813850.

| Cached family attribution | Initial mean normalized verifier score | Final mean |
| --- | ---: | ---: |
| Calendar | 0.35 | 0.10 |
| Competitive coding | 0.05 | 0.00 |
| Reasoning gym | 0.49640241 | 0.40570990 |
| STEM | 0.20 | 0.25 |
| Structural | 0.35 | 0.35 |
| All 100 | 0.28928048 | 0.22114198 |

All 200 `passed` fields are null. Numeric score coverage establishes neither boolean pass rates nor statistical significance, quality improvement or a trusted cross-arm ranking. These are descriptive before/after objective scores only.

The native checkpoint publication marker identifies step 3, and four checkpoint control objects have nonempty recorded sizes. Their bodies and checkpoint tensors were not read. This is publication evidence, not proof of all three finite applied optimizer updates, tensor integrity or optimizer/scheduler/RNG resume. The no-export result's null HF model and global-step fields are expected for this path; no HF export was requested.

The artifact's `dep_paths` is empty. Declared dependency identities, fingerprints and retained configuration/source declarations do not independently attest resolved input bytes. Submitted ZIP source binding is recorded, but it does not establish the installed or loaded GPU runtime, package or weight identities. Configuration agreement at the reviewed pre-validator phases does not attest later active GPU validator state. Immutable served judge revision and external asset bytes remain unknown.

Complete applied-update and consumed-rollout-age evidence, observed nonzero consumed age, age-bound compliance, generation/update overlap and closed accounting of stale discarded and regenerated work remain missing. Missing accounting is not zero waste. D3's bounded operational and numeric observations are accepted; its qualification is not. The A3 user exception does not extend to D3. Qualification, dashboard/ranking use and all four 25-update screens remain held, with no D3 policy exception granted. All job and retained-result read scopes are permanently retired; this result authorizes no further reads or execution.

## Final decision

The strict hold remains. D3 has accepted bounded operational and descriptive evidence, including numeric-score decline and native checkpoint publication, but it has not met qualification. All-step finite applied-update evidence, consumed ages and overlap, complete stale-work accounting, input/runtime attestation and optimizer recovery remain incomplete. A3 retains its explicit evidence-limited user acceptance; that exception does not apply to D3. No trusted arm ranking, dashboard result, statistical improvement or boolean pass rate is established. All four 25-update export-off screens remain unlaunched and held. Preserve both qualification outcomes and their different timing contracts; no new reader, retry or experiment follows from this record.
