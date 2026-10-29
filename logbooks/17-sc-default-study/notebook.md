---
issue: https://github.com/marin-community/marin-logbooks/issues/17
source_repository: marin-community/marin
source_revision: 821bb46059beaea06cf00607b8a49f0187cb3ea2
archive_template_revision: f54b3aa0e6ee533e988d2b97e9db0d1523f934dd
status: active
---
# Score-centering defaults on Snowball

All three arms use clipped token loss with tokenwise truncated importance sampling (TIS). Score centering (SC) subtracts the sampling policy expectation of that training score. Its approximation retains the most probable 128 or 32 token candidates at each response position and models the remaining tail. The primary comparison is SC32 versus TIS. Report separate recommendations on enabling score centering and its candidate count. Product defaults remain disabled and 128.

The acceptable margins are one absolute percentage point of held-out task success and 5% training-cycle elapsed time. Start with three matched seeds per arm; one finite, justified extension is allowed. Learning curves, per-task grades, response lengths and allocated compute must support the recommendation. A nonsignificant difference cannot establish an acceptable trade-off.

## Discovery and protocol

### 2026-10-28 03:19 UTC: Scope and retained baseline

This phase starts from the [completed performance study](https://github.com/marin-community/marin-logbooks/issues/13) at `821bb46059beaea06cf00607b8a49f0187cb3ea2`. That source preserves the qualified local-vLLM/Megatron runtime, pinned wheel, compact chat adapter and executable research driver. Its fixed-weight costs and eight locally graded prompts do not establish learning effects. This phase will use normal optimizer and sampler updates, no synthetic routes, and no hypothetical probes.

The [recent full-inventory curriculum evidence](https://github.com/marin-community/marin/issues/10314#issuecomment-6308654380) has complete paired development evaluations at updates 0, 25 and 50. It supplies a candidate training duration and mixture, while the reused 910-task panel is insufficient as an untouched test set. The exact recipe, grader readiness and a separate held-out split are being checked before the protocol is frozen. No study GPU jobs have been submitted and no SC learning outcomes have been examined.

Core draft [#10429](https://github.com/marin-community/marin/pull/10429) remains at `a165e48ab209f71a512b5e138466662e47e53576`; probes draft [#10658](https://github.com/marin-community/marin/pull/10658) remains at `eb1a3bf6a29341fdc777f4490b8609b7ea2dc03e`. Experimental drivers and bulk results stay on this research branch.

### Predeclared initial study and extension

The [protocol](protocol.json) fixes three matched seeds (1701, 2903, 4517), 50 policy updates per run, and a new 10,000-task holdout with equal weight across calendar, coding, Reasoning Gym, STEM and structural compliance. Full-task success at update 50 is the primary quality endpoint. A fixed balanced 1,000-task subset runs at updates 0 and 25. The old 910-task development panel is excluded. Split identities use a fixed hash order. A public-task fingerprint hashes public content and resources; a contract hash also preserves private grading inputs. Matching content is excluded from training. These are exact content checks, not a test for similar questions. Prior SFT/pretraining exposure remains unknown.

The recipe uses the healthy full-mixture optimizer and 40-H100 geometry, the existing October-5 Grug initializer, ordinary clipped token loss with tokenwise TIS, 128 prompts with four answers each, AdamW at `1e-6`, 8,192-token replies within 24,576 tokens, and no KL or entropy auxiliary loss. Normal rolling staleness-2 sampling overlaps training, using the production loop. These changes from the synchronous reference are explicit; its learning curves select the bounded window, not an expected result. The first TIS run must complete before expanding submissions. The previous packed-CP5/100K qualification remains available, while this shorter recipe uses CP1 with TP/SP1 and packing.

Match the initial checkpoint and the trainer, optimizer, sampling and data-shuffle seeds within each triplet. Generated answers and subsequent weights may diverge naturally. Report 95% intervals that include paired-seed and paired-task uncertainty. Use intervals adjusted for the two possible looks for each metric in each comparison; this is not a joint coverage claim across every result. The quality margin is minus one absolute percentage point; the cost margin is plus 5%. One extension may add three matched seeds, at the same 50-update endpoint, after publishing a precision justification and updated cost estimate. It cannot change the metric or select a checkpoint. Initial endpoint results remain separately visible. The planning estimate is 1,500–2,500 H100-hours for the initial matrix and the same maximum range for the possible extension; measured first-run time will refine this. Ten thousand tasks can resolve a small paired task difference, but three seeds may leave the one-point decision inconclusive.

The source pool has 100,000 canonical tasks in 328,741,175 bytes, SHA256 `bc9326bc87e9fd992748147aaa3f8261e649819456027645cb00cbc23dc7e94d`. All thirteen sources have declared graders. The package-lock object and documented Together credential resolve; actual grading is checked before GPU submission. This pool uses local package-backed graders and in-process grading, so Daytona is not required. Qwen startup additionally enables reference scoring and checks live serving weights after sleep; its outcomes are excluded from study inference.

Before execution, keep the healthy uniform sampler over all retained canonical tasks. The canonical loader drops arbitrary routing metadata, so the study does not add a loader patch or custom sampler. The fixed retained family and source counts define the training mixture and will be published before the first Grug run. The primary time is the complete training-cycle span through final weight publication, with exclusive evaluation sessions subtracted. Startup, evaluation, checkpoint/export and teardown are reported separately.

### CPU qualification and frozen split complete

The [frozen data summary](frozen-data.json) records 89,557 training contracts and 10,000 held-out tasks, with zero exact public-content or grading-contract overlap. The exclusion removes 443 additional training rows that share selected public content. The uniform training mixture is calendar 7.945%, coding 22.815%, Reasoning Gym 22.989%, STEM 23.137%, and structural 23.115%; exact source counts and file/ordered-ID hashes are retained. Each evaluation family has 2,000 tasks. No SC learning outcomes have been examined.

The real CPU [grader gate](grader-qualification.json), source `70f9b2586d55a858e44dfdf6e50e1cdc70b09b2d`, completed as `/romain/sc-default-graders-20261028-01a16b97-v1`. All thirteen sources produced grades, including the Together-backed judge. Two in-process reference answers scored one; the other sources used invalid-answer controls. The structured-output control's JSON parse error is an expected zero for invalid submitted text, not an unavailable grader. This one-task CPU job used no GPUs. The exact canonical grader lock and contracts remain unchanged.

All nine initial configurations and the Qwen startup pass the real SkyRL recipe composer and objective checks. The actual Grug tokenizer and trainer dataset preserve all 10,000 ordered held-out IDs within the 16,384-token prompt budget. The [dataset qualification](dataset-qualification.json) also checks that 993 native structural tasks retain the existing assistant-prefix continuation harness; the remaining 9,007 use Shellbox. An initial CPU check found that a dotted release name became nested configuration keys and incorrectly selected Shellbox. A research-only Hydra config group preserves that literal source name. This repair and the qualification driver remain outside both product PRs.

The upcoming Qwen startup uses two nonzero-learning-rate updates, reference scoring, disjoint held-out grading, live serving-weight hashes after sleep, and checkpoint/result export. Complete it before the first 40-H100 Grug allocation. The main recipe has no reference auxiliary loss, so its reference-scoring cost is zero.

### Qwen training passed; recover the separate export

The [Qwen evidence](qwen-qualification-progress.json) from source `6aae057cf351e0819946e38f33709092abe5b1b5` records two actual optimizer updates at `1e-6`, finite positive gradient norms, no skipped steps, and nonzero policy/reference KL. All ten disjoint held-out tasks were graded at updates 0, 1 and 2. All eight serving replicas agreed after each publication; the weights changed twice, and the final hash survived checkpoint restoration. This startup supplies runtime evidence only and is excluded from Grug inference.

The GPU training child succeeded. Its separate checkpoint-export child failed before conversion because the research dispatch selected the training measurement function for the export entrypoint too. The coordinator therefore failed. Restrict that dispatch to the normal training module, then run the existing request-driven converter against the saved update-2 checkpoint. Do not regenerate training or answers. The failed export remains part of compute accounting, and Grug waits for verified export completion. The repair changes only this research branch.

The paired seed/task [analysis implementation](inference.py) now follows the predeclared intervals. Its [numerical verification](inference-verification.json) checks known seed and task variance components, geometric cost ratios, decision boundaries, and rejection of missing grades. These are synthetic checks, not study measurements.

### Qwen qualification complete; first Grug baseline

The normal converter recovered the saved checkpoint as `/romain/sc-default-qwen-export-01a16b97-v2`, using published source `ea0a72af4956612553e4cd47a9e54b84748799e4`. Iris reports success, the export request is complete, and the portable Hugging Face model inventory verifies. A direct read of the exported first-layer attention output matrix exactly matches the final live serving-weight hash. The [completed qualification](qwen-qualification.json) preserves the original coordinator/export failure as well as this recovery. The three GPU attempts total 1.881 H100-hours by controller attempt durations; final accounting will also inspect pod allocation timestamps.

The [collector](collect.py) reads saved verifier verdicts, preserves missing grades, applies their declared normalization, and keeps the same balanced 1,000-task panel for every learning-curve point. It collected all thirty saved Qwen grades and their response lengths. The full 10,000-task update-50 endpoint remains separate from that curve panel.

Use `cw-rno2a` for the entire Grug matrix. At the pre-submission check it had 16 unrequested H100s; East had seven. These snapshots do not reserve capacity, so the five-node 40-H100 baseline uses batch and may wait. Submit only TIS seed 1701, version `2026.10.28.1`, through `python -m experiments.post_training.sc_default_study --version 2026.10.28.1 --arm tis --seed 1701 --full --cluster cw-rno2a --run`. Complete its planned 50 updates, held-out grading and exports before expanding submissions. Its settings are unchanged from the frozen protocol, and it will count toward the initial matrix if valid.

### Baseline queued; analysis path checked

The [run manifest](runs.json) records the first Grug coordinator and actual five-task GPU child, both using source `56e9d2eac9b327be8e392780b8ced8d737705cc9`. The data-preparation child succeeded. The [remote data check](main-remote-data-validation.json) confirms the exact training/evaluation file hashes and all 10,000 evaluation-index entries from the independent CPU freeze. The gang is waiting for batch quota; no Grug optimizer update has run yet.

The [reproduction guide](README.md) connects collection, matrix assembly, inference and plotting. The [reporting check](reporting-verification.json) exercised that path on explicitly synthetic data and rejected a changed task identity. It does not supply study results. Final curves will use each run's actual consumed loss tokens and measured cycle span.

The [startup allocation accounting](startup-allocations.json) supplements controller durations with pod scheduling and final-container timestamps. Conservatively taking the larger duration for each attempt gives 1.904 H100-hours across Qwen training, the failed export, and its successful recovery. Unbound queue time consumes no GPUs. All startup jobs are terminal.

### Correction: reject the first Grug startup and repair dispatch

The export repair restricted the measurement dispatch to `main_base`. The real Qwen and Grug launch documents register `taskcompendium`, which also calls `BasePPOExp`. Grug therefore entered the normal loop without the study hooks and began a 10,000-task initial evaluation. The declared initial curve panel has 1,000 tasks. The missing measurement objects and the unexpected evaluation denominator exposed this before any optimizer update. The earlier CPU dispatch check mocked the module name and missed the mismatch.

The [rejected attempt](rejected-grug-v1.json) is excluded from every learning and cycle-cost comparison. Its coordinator and all five GPU tasks are authoritatively killed. Its [allocation record](rejected-grug-v1-allocations.json) contributes 11.303 H100-hours, including startup and the interrupted initial evaluation. Saved attempt records remain at their original paths; other jobs were untouched. The actual loader retained all 89,557 training and 10,000 evaluation rows, so the frozen data and mixture remain unchanged.

The research dispatch now recognizes both BasePPOExp training modules, keeps checkpoint export native, and rejects unsupported study entrypoints. The [replacement CPU check](actual-dispatch-verification.json) loads the actual Qwen, Grug and export documents through the real registry. The measured training code, recipe, input writer and continuation config are byte-for-byte unchanged from the successful original Qwen qualification. Requalify the complete Qwen lifecycle at fresh version `2026.10.28.2`, then retry TIS seed 1701 at the same fresh version. The scientific protocol and both product PRs are unchanged; no additional Grug seed or arm has been submitted.

### Corrected lifecycle passed; fresh baseline submitted

The [Qwen version-2 qualification](qwen-v2-qualification.json), using published source `66bb62a9d8584647e4f7777057a014af45d5c3b6`, completed two actual updates, reference scoring, all thirty held-out grades, consistent changed weights on all eight serving replicas, checkpoint restoration, and the ordinary Hugging Face export. Training, native export, and the coordinator all succeeded without a recovery job. The exported representative tensor again matches the final live serving tensor exactly. Its [two GPU allocations](qwen-v2-allocations.json) total 1.591 H100-hours. Startup and rejected-attempt allocations so far total 14.798 H100-hours; none supplies a Grug inference observation.

The fresh TIS seed-1701 baseline is submitted as `/romain/users-romain-checkpoints-sc-default-study-tis-seed1701-2026.10.28.2-coordinator-2c7e6c52`, using the same qualified source. The frozen inputs, 50-update endpoint, 40-H100 batch geometry and evaluation schedule are unchanged. Both pools were checked again: RNO had sixteen unrequested GPUs and East had zero, so queue wait remains possible. The [manifest](runs.json) keeps the rejected version-1 attempt separately and will admit version 2 only after all planned training, grading, measurement and export checks pass. No other Grug study arm or seed has been submitted.

### Keep judge failures recoverable at the analysis boundary

Version 2 used the correct measurement driver and the declared 1,000-task initial panel. One Together-backed judge reply remained unreadable after its two normal attempts, leaving 999 valid verdicts. The harness stopped before the first optimizer update. The [zero-update attempt](rejected-grug-v2.json) and its [10.630 H100-hours](rejected-grug-v2-allocations.json) remain excluded from learning/cycle inference and included in total allocation accounting.

The [saved-answer recovery](saved-answer-regrade.json) used the same response tokens, task contract, judge model and endpoint. Its first valid verdict scored one. No answer was regenerated for that recovery. The historical dump contains decoded text with an EOS marker; reconstructing the serving text uses the documented `decode(response_token_ids, skip_special_tokens=True)` invariant. Future failed verdicts also retain their exact original grading messages.

The live harness now records missing verdicts as pending and lets normal training and export finish. The collector still refuses incomplete coverage. A bounded [grade-only recovery tool](../../experiments/post_training/sc_default_regrade.py) adds a supplemental verdict only when the saved answer and original contract hashes agree, and never replaces an existing valid grade. The [CPU boundary replay](pending-grades-verification.json) checks that the real 999/1,000 case stays explicitly missing while the job continues, and that failed grading conversations survive serialization. The model, optimizer, TIS, serving, seeds, panel and decision rules are unchanged; the completed Qwen checks for those paths are reused.

Restart TIS seed 1701 as a fresh full training attempt at version `2026.10.28.3`. Preserve the recovered version-2 initial panel alongside that zero-update failure; it is not a substitute primary endpoint or an independent seed. Any grade gaps in the new run must be repaired from that run's own saved answers before admission. No additional Grug arm or seed is submitted until this first valid full baseline completes.

The published recovery path was then exercised on the saved version-2 panel. The collector rejected 999/1,000 before the supplemental verdict and accepted all 1,000 afterward. All original answer-object hashes remain unchanged. The [end-to-end check](saved-panel-recovery-verification.json) preserves the missing original verdict beside its first valid recovery. Full success is 46.9% on this zero-update panel, with family rates from 34.0% to 72.5%; the selected tasks avoid an overall floor or ceiling. This startup observation does not enter the primary update-50 comparison.

### 2026-10-28 14:28 UTC: First Grug update and weight publication

TIS seed 1701, version `2026.10.28.3`, uses published source `4642b64a9654eda85a347044faacc13cc3d49d1a`. Its five batch workers are running. The [actual configuration and first-update evidence](grug-tis1701-v3-first-update.json) confirms the frozen data, 40-H100 geometry, ordinary rolling sampling, disabled SC/capture/probes, and pinned local vLLM V2 path. The real loader retains all 89,557 training and 10,000 held-out tasks. The first optimizer update used learning rate `1e-6`, gradient norm 0.303, zero skipped steps, and 236,067 consumed loss tokens. Weight publication completed; its 119.1 seconds are included in the 227.8-second complete first cycle. This single cycle does not establish the full-run cost.

The declared initial panel generated 1,000 saved answers. One missing judge verdict was recovered from its exact original grading messages, using the same contract, model and endpoint; the first valid retry scored 0.75. The collector now verifies all 1,000 grades, with 45.4% full-task success. This is the new run's zero-update curve point, not a primary endpoint or an independent seed. Training continues toward the fixed 50-update endpoint. No SC run or additional seed has been submitted; full baseline training, grading and native export still gate expansion.

### 2026-10-28 18:10 UTC: Training succeeded; original task contracts block complete grading

The version-3 TIS baseline completed all 50 updates and final training weight publication. Its measured complete-cycle span is 9,513.162 seconds, excluding the 345.604-second midpoint evaluation. On the fixed 1,000-task curve panel, full success rose from 45.4% at update 0 to 61.1% at update 25, with complete grades at both points. All 10,000 final answers are saved. The training child succeeded and used 147.836 H100-hours; its native export is queued separately on four nodes and 32 H100s. No other study arm or seed has been submitted.

The final panel has 9,990 valid original verdicts. One missing verdict is an unreadable Together judge reply. Nine others expose broken local grading contracts: two JSON schemas contain unresolved references, and seven Reasoning Gym time-interval tasks fail the original generator's equality guard. A [static check of every held-out contract](heldout-static-grader-audit.json) finds exactly those two malformed schemas. The [original pinned generator replay](time-intervals-replay-qualification.json) checks all nineteen time-interval tasks. For the seven failures, configuration, question and reference answer still match; only the metadata date changes from October 24 to October 28. These audits do not inspect model answers or call a scorer. One control per source established grader capability but did not establish that every frozen task contract was usable.

The collector rejects the incomplete final panel. The [proposed amendment](heldout-grader-amendment-proposal.json) would omit those same nine objectively identified contracts from every arm's quality analysis, retaining 9,991 tasks with the existing equal family weights. The 1,000-task curve panel, all training data and settings, all raw answers, and the full evaluation/cycle-cost measurement would remain unchanged. This is a proposed data-scope change requiring Romain's approval; no exclusion is applied. The original coverage gate continues to block admission and expanded submissions. Independently, the recovery tool can now select particular saved UIDs, allowing the one judge reply to be retried without changing any task contract or accepting the nine local failures.

### 2026-10-28 19:40 UTC: Common quality mask approved; first full baseline admitted

Romain approved excluding the same nine contracts from every quality comparison. No SC run has been submitted or outcome examined. The [approved eligibility file](eligibility.json) binds those original UIDs to the frozen index, public task fingerprints and grading contracts. The quality panel retains 9,991 tasks: 1,993 Reasoning Gym, 1,998 structural, and 2,000 in each other family. Each family still has weight 0.2. Training, all 10,000 generated endpoint answers, the fixed 1,000-task curves, seeds, duration, hardware, cost definition and decision margins remain unchanged. This amendment follows baseline inspection and is not part of the initial predeclaration.

The one missing judge verdict was recovered from its saved original conversation with the same task, model and endpoint; the second grade-only call supplied the first valid verdict. The original objects are unchanged. The [admission check](grug-tis1701-v3-admission-verification.json) verifies all 10,000 original answers, all 9,991 retained grades, and preservation of the nine excluded failures. The [amended reporting check](amended-reporting-verification.json) exercises assembly, inference and plots with the unequal family counts. It verifies equal family weighting and rejects a missing retained grade, changed task identity or wrong eligibility hash. The paired interval and decision calculations are unchanged.

The TIS seed-1701 endpoint is **66.137% equal-family full-task success**. The same fixed curve panel scores 45.4%, 61.1% and 65.2% at updates 0, 25 and 50; its last point is separate from the larger primary panel. The complete-cycle span is **9,513.162 seconds**, or 158.553 minutes, for 50 updates and 65,845,266 consumed loss tokens. The exclusive midpoint evaluation contributes another 345.604 seconds. Startup and initial/final evaluations remain separate; the startup measurement overlaps the initial evaluation and must not be added without subtracting that overlap.

The [ordinary health checks](grug-tis1701-v3-training-health.json) verify all 50 nonzero-learning-rate updates, finite positive gradient norms, no skipped updates, finite mismatch telemetry, and accepted staleness at most two. PPO clipping is zero in this configured single policy update, where the old/current ratio is effectively one; ordinary tokenwise TIS remains active. The [portable export check](grug-tis1701-v3-export-verification.json) verifies the successful native export, 39 weight shards totaling 134,157,838,824 bytes, and actual changes to sampled attention and router weights. It does not check every tensor or every live serving replica; the Qwen lifecycle separately qualified restoration.

The training, export and coordinator jobs all succeeded. Their [allocation accounting](allocation-accounting-through-first-baseline.json) totals 157.194 H100-hours for this baseline, including 9.358 for the native export. Startup and rejected attempts add 25.428, giving 182.622 H100-hours so far. At baseline cost, the remaining eight initial runs require about 1,258 H100-hours; SC overhead and differing generated lengths remain to be measured. One TIS run alone supports no default recommendation. Product defaults and both PRs remain unchanged.

The main run used source `4642b64a9654eda85a347044faacc13cc3d49d1a`. Subsequent changes recover saved grades and apply the approved offline eligibility rule; they do not change training or serving. Next submit SC128/1701, SC128/2903 and SC32/4517 as three 40-H100 batch gangs in RNO, preserving the declared order within each seed. Complete the remaining eight runs before the initial comparison. Native resumable state and raw records remain at their original paths. A trial archive copy did not pass full verification, so no temporary bulk has been removed; preservation and cleanup remain pending after consumers stop.

### 2026-10-28 20:00 UTC: Three further predeclared runs submitted

Published source `4fe021097407ed8f1385343ea353b590fa184bd9` launches SC128/1701, SC128/2903 and SC32/4517 at artifact version `2026.10.28.3`. The [submission record](wave1-submissions.json) contains exact coordinator and GPU-child IDs, request hashes and the initial task states. All three use five-node 40-H100 RNO batch gangs and the unchanged frozen recipe. The first SC128 gang is admitted and initializing; the other two are waiting for Kueue admission. Controller RUNNING alone does not mean GPU allocation. This is the next step of the original nine-run matrix, not an extension.

The command for each arm/seed is `uv run --offline --frozen python -m experiments.post_training.sc_default_study --version 2026.10.28.3 --arm <arm> --seed <seed> --full --cluster cw-rno2a --run`, with the normal Marin environment and existing Together credential delivery. The shared resource board records this batch expansion after checking both pools. No learning result from these runs is available yet. Apply the same approved quality mask only during collection; generate all original endpoint tasks.

### 2026-10-28 22:01 UTC: All three next runs training; temporary judge outage recovered

The [progress snapshot](wave1-training-progress.json) records ordinary training at updates 31, 3 and 4 for SC128/1701, SC128/2903 and SC32/4517 respectively. Both later gangs were physically admitted around 21:30 UTC. All three actual launch configuration audits pass, with nonzero learning rates, finite positive gradient norms, no skipped updates at the captured steps, and candidate mass telemetry from the normal SC batches. Their initial fixed 1,000-task panels have complete grades, with full success 44.9%, 45.2% and 45.6% respectively. These are curve points, not independent endpoints or SC effect estimates. All three remain ineligible for primary inference while training, final grading and export are pending.

A brief Together service outage returned HTTP 503 for 56 coding-family grades in the first SC128 midpoint evaluation. Its [original failure record](grug-sc1281701-v3-midpoint-judge-outage.json) preserves the missing UIDs and answer-object hashes. Training continued under the existing pending-grade rule. The [grade-only recovery](grug-sc1281701-v3-midpoint-recovery-verification.json) obtained the first valid verdict on one canonical call per missing answer, using the same task, judge model and endpoint. All original objects are unchanged. The repaired fixed panel has all 1,000 grades and 61.9% full-task success at update 25. The [interim training check](grug-sc1281701-v3-health-during-judge-outage.json) through update 28 shows mean training grade coverage 99.902%, a minimum of 98.828%, finite positive gradients and no skipped updates; the normal recipe preserves its usual handling of missing training grades. No training data or rule was changed.

The SC32 initial panel separately needed [one unreadable judge reply recovered](grug-sc324517-v3-initial-recovery-verification.json). All 1,000 grades now validate against that run's original saved answers. No answer was regenerated, no valid verdict was replaced, and no additional task was excluded. The approved common nine-contract mask remains fixed for the larger final quality panel. Continue all three to the declared 50-update endpoint.

The separate logbooks-main [index PR #18](https://github.com/marin-community/marin-logbooks/pull/18) adds only this study's index row. It is a draft at `d24bc99cb81d89909df9500472620b54d880bf2b`; its final status and commit-pinned notebook link will be updated after the bounded study. It contains no research source or product code.

### 2026-10-28 23:52 UTC: First SC128 final panel complete; native export queued

SC128 seed 1701 completed all 50 training cycles and final evaluation. Its [collected verification](grug-sc1281701-v3-collected-verification.json) checks every planned update, all 10,000 generated endpoint answers, and all 9,991 admitted grades under the unchanged approved mask. Four unreadable final judge replies were recovered from the saved original conversations; all original answer-object hashes are unchanged. The nine excluded original contracts remain separately preserved. No further task was omitted.

The equal-family primary success is **66.017%**. Its fixed 1,000-task curve is 44.9%, 61.9% and 64.5% at updates 0, 25 and 50. The primary complete-cycle span is **9237.205 seconds** for 63,394,185 consumed loss tokens and 63,431,103 response tokens. All fifty updates have finite positive gradients, no skipped steps, finite chosen-token mismatch telemetry and accepted staleness at most two. Existing sampled candidate-mass telemetry is preserved; it uses up to 4,096 evenly spaced valid tokens per update and adds no model or gradient pass. These are descriptive results for one seed. Formal comparisons wait for all nine initial runs.

The cycle-50 record is written immediately after final training weight publication. The ordinary train metric at update 50 and the completed-training manifest appear later, after step-end checkpoint/evaluation callbacks. A latest logged train update of 49 during those callbacks is expected and does not mean the final optimizer cycle is stuck. The measured primary span excludes those final callbacks and the 331.908-second exclusive midpoint evaluation; phase timers retain their normal overlap.

The training child and all five tasks succeeded. Its [allocation record](allocations-grug-sc1281701-v3.json) totals **145.350 H100-hours**, including startup, all evaluations, native checkpoint and teardown. The separate four-node 32-H100 export is queued at the latest capture. This run remains ineligible for primary inference until native export verification succeeds. SC128/2903 and SC32/4517 continue normal training; both midpoint 1,000-task panels have complete grades, with fixed-panel success 60.5% and 64.7% respectively.

The small [export verifier](verify_export.py) makes the existing operator check repeatable. Its [baseline replay](export-verifier-replay.json) reproduces the prior independent 39-shard inventory, manifest hash, byte count and both sampled router/attention weight checks exactly. It uses byte ranges, not another model run. Training and serving source remains `4fe021097407ed8f1385343ea353b590fa184bd9`; all new code in this entry is offline verification outside the product PRs. Wait for native export before advancing the seed-1701 order to SC32.

### 2026-10-29 00:27 UTC: SC128 seed 1701 admitted; advance its fixed order

The native [SC128 export verification](grug-sc1281701-v3-export-verification.json) passes: the request and export terminal record succeeded, all 39 indexed BF16 weight shards are present, and sampled attention/router weights differ from the exact initializer. The training child, all five training tasks, export child, all four export tasks and CPU coordinator are authoritatively succeeded. The [export allocation](allocations-grug-sc1281701-v3-export.json) adds 8.745 H100-hours, giving **154.095 H100-hours** for this complete run. Unbound queue time is excluded.

All training, common quality-panel, measurement and export gates now pass. The [manifest](runs.json) admits this run, bringing the initial matrix to two of nine eligible runs: TIS/1701 and SC128/1701. No formal comparison or default decision is made from a single matched seed. Next submit SC32/1701, the final arm in that seed's predeclared order. SC32/4517 is in its final evaluation/checkpoint callbacks and SC128/2903 continues training. Each seed advances only after its preceding run passes the same gates.

Both H100 pools were checked again before the next submission: RNO has zero unrequested GPUs and East has two at the snapshot, so the next five-node RNO batch gang may queue. Keep the selected region, hardware, source recipe, seeds, duration, original generated panel and approved common nine-task quality mask fixed. Native checkpoints remain at their original paths; bulk preservation and cleanup still wait until consumers stop.

### 2026-10-29 00:31 UTC: SC32 seed 1701 submitted

The [next submission record](wave2-submissions.json) identifies SC32/1701 at version `2026.10.28.3`, launched from published clean source `22cc8860aa05dfa5ef4dbcb6a79bfd5712ac9d69`. The source changes since the first three SC submissions are offline export verification and research records; training, serving, data, weights and the approved quality rule are unchanged. This is the last arm in seed 1701's original TIS → SC128 → SC32 order. Both preceding runs now pass every gate. The new five-node 40-H100 RNO batch gang is recorded as: gpu gang queued for batch admission. Queued controller state is not physical GPU use. The shared resource board announces this staged continuation of the initial matrix; it is not an extension.

### 2026-10-29 02:24 UTC: Storage quota blocks two exports and the next startup

SC32/4517 completed all fifty updates and all original 10,000 endpoint answers. Its [collected verification](grug-sc324517-v3-collected-verification.json) passes all ordinary training-health and common-quality gates, with **68.711% equal-family task success**, **8217.812 seconds** of primary training-cycle time and 52,010,661 consumed loss tokens. The fixed 1,000-task curve is 45.6%, 64.7% and 67.0%. Three unreadable final judge replies were recovered from the unchanged saved conversations. SC128/2903 also completed fifty updates and all 10,000 answers; two retained judge verdicts, UIDs 1806 and 6226, remain missing. No additional task is excluded. These are incomplete, unmatched descriptive results; formal inference still waits for the initial matrix.

Both separate native exporters failed before conversion because rank zero could not write its Ray rendezvous object. SC32/1701 also failed at that same startup stage, before training. The [failure record](storage-hold-20261029.json) contains exact job identities, terminal states and a bounded operator probe that returned **HTTP 405: account write-suspended for the availability zone**. Existing storage telemetry at 01:51 UTC reports **100.384 TiB used against a 100 TiB RNO2A quota**. This is an external account-capacity failure. The diagnostic exposed the server rejection by removing the Expect header on its private client only; no transport or shared configuration was changed.

All owned GPU tasks are terminal. After preserving their evidence, the three CPU coordinators stuck in error-bookkeeping writes were stopped; succeeded training children and failed export children retain their original states. Resource-board rows 1575 and 1577 are closed, and no further GPU work is submitted. Two of nine initial runs remain fully admitted. The [native inventory](native-checkpoints-at-storage-hold.json) preserves all four completed update-50 checkpoints, totaling 3,224,664,148,972 bytes. Each has the same 32-shard distributed-checkpoint layout and payload sizes as the successfully exported baseline, including its ten empty distributed-rank files. This is an inventory check, not a full checksum or restore test. The two failed export requests remain marked in progress because failure bookkeeping also could not write.

Resume after a verified small write/readback succeeds. Recover only the two missing original SC128/2903 grades, then use export-only jobs for its and SC32/4517's existing checkpoints. Retry SC32/1701 at a fresh artifact version with the same seed and initializer. Advance each remaining predeclared seed order after its normal gates pass. An estimated 6 TB of headroom covers the remaining initial matrix and one-checkpoint archive overlap; account-capacity recovery has been requested from Romain. No native state or completed result was deleted, the failed archive trial remains unverified, and preservation/cleanup remains outstanding. Product code and defaults remain unchanged.

### 2026-10-29 02:25 UTC: Correction to allocated-compute accounting

The earlier allocation records called the larger of controller duration and available pod duration conservative. Most main-run pods had already been garbage-collected, so those values were controller-only. The controller starts its clock after the pod reaches Running, which omits some bound setup time. The global conservative claim is withdrawn; the original records remain visible. Primary training-cycle times, evaluation results and decision rules are unaffected.

The [reconstruction](allocation-reconstruction-20261029.json) uses retained live PodScheduled timestamps or Kubernetes Scheduled events where available, and observed Kueue admission as a start proxy otherwise. It ends at the later of controller terminal observation and any retained final-container exit. This gives **621.000 H100-hours across executed tasks**, including startup, rejected runs and failed exports. A preempted SC32/1701 gang adds **2.602 quota H100-hours** from observed admission to preemption, reported separately because no binding or running observation survives. The later requeue wait is excluded. These are reconstructed resource spans, not exact provider billing or certified bounds; admission polling, quota-to-binding delay and final pod release leave some uncertainty.

Under this method, the complete TIS/1701 run is **157.312 H100-hours**, SC128/1701 is **154.309**, and earlier qualification/rejected work is **25.697**. The [current manifest](runs.json) labels reconstructed values explicitly and retains the previously reported values. Future jobs will capture scheduling and terminal evidence while pods are live. No additional learning run or statistical look was performed for this correction.

### 2026-10-29 02:37 UTC: Stage missing verdicts locally during the storage hold

The research-only grade recovery command now accepts `--local-manifest`. It atomically saves and flushes each first valid verdict to local disk instead of attempting a remote write. Repeating the command reuses those exact verdicts. After storage recovers, `--publish-staged` checks every saved-answer hash, task contract and judge configuration before publishing; it cannot replace an existing valid verdict or make a new judge call. Normal recovery without these options is unchanged apart from stricter validation of previously recovered rows. This adds no training, scoring or product-PR code.

The [bounded verification](local-regrade-staging-verification.json) replays four actual saved SC128/1701 verdicts with judge calls forbidden and publication intercepted locally. Exact verdict reuse and publication validation pass; conflicting manifests and a different run are rejected. It performs no remote write and establishes no storage readiness. Lint, formatting and syntax checks pass for the research helper.

Next recover only SC128/2903's missing update-50 UIDs 1806 and 6226 with the same saved conversations, task contracts, judge model and endpoint. Use the usual grade-only command with `--uids 1806 6226 --local-manifest <private-local-manifest>` and preserve that file and its hash. Stop at the first valid verdict per answer under the existing limit of three infrastructure-recovery calls. This CPU work can finish while GPU execution remains held. The ordinary collector still waits for publication to the existing remote recovery path; native export remains a separate gate.

### 2026-10-29 03:40 UTC: Recover a missing score line; storage writes return

The first bounded local recovery pass saved UID 1806's first valid verdict on one canonical grading call. UID 6226 exhausted three calls without a complete verdict. One additional diagnostic call kept the identical task, prompt, judge and parser and recorded its replies privately. Its first checklist score was valid. A later check twice returned the same explicit trailing score on the explanation line, which the strict reader rejected because it expected a separate final line. This is a reply-format failure, not an unavailable task or a new exclusion. Partial checks do not supply a complete task verdict.

The research command's explicit `--recover-score-line` fallback now reads one unambiguous trailing score field by passing that field to the original parser. Allowed numeric values and ordinary checklist aggregation are unchanged. Strict valid replies retain their values; duplicate fields, trailing text, embedded non-field names and disallowed values remain rejected. The [qualification](score-line-recovery-verification.json) replays all three actual diagnostic replies and nine controls without judge calls. This small recovery reader applies consistently to missing verdicts in every arm. It does not replace any already valid grade, change a task contract or modify the live training grader. Use it for the pending UID 6226, retaining the first complete valid verdict and accounting for every failed and diagnostic call. The original nine-task mask and primary endpoint stay fixed.

A bounded object write, readback and exact probe-object removal now succeed at 03:31 UTC. Storage telemetry at 03:01 reports about 0.65 TB of regional headroom, enough for the two approximately 134-GB portable exports but insufficient for the remaining native training checkpoints. Prepare export-only recoveries with fresh attempt directories, recheck both GPU pools and announce the work before submission. New training remains held for additional capacity. No completed checkpoint has been deleted and no default-setting inference has been made.

### 2026-10-29 03:47 UTC: Export-only recoveries prepared and checked

The [two recovery plans](export-recovery-plans.json) compose the normal request-driven converter against the saved SC32/4517 and SC128/2903 update-50 checkpoints. Both use four RNO batch nodes and 32 H100s, the original model/export destinations and unchanged saved policy geometry. Their published input and derived configs use fresh rendezvous, log and terminal directories, preserving failed attempts. The small [operator entrypoint](export_checkpoint.py) reuses the previously qualified Qwen recovery wrapper and normal converter, including its export-request completion bookkeeping. Composition and launch validation pass without any cloud submission or remote write.

The [export-verifier qualification](recovery-export-verifier-qualification.json) replays the actual TIS baseline with an explicit terminal URI and expected job identity. Its manifest, all 39 shard sizes and both sampled tensor checks exactly match the prior independent result; a wrong job identity is rejected. The optional terminal URI lets a recovery prove its own success without overwriting or relabeling the failed attempt. All original export gates remain required.

Both H100 pools were checked at 03:43 UTC: RNO had 104 unrequested GPUs and East nine, as snapshots without reservation. A read-only storage-metrics workflow was also requested to refresh headroom before submission; it runs no GPU work and writes no telemetry. Submit the two 32-H100 batch exports only after a fresh storage write/readback and sufficient headroom, with a resource-board announcement and the newly published source revision. Training remains held while its larger checkpoint headroom is unavailable. The initial matrix and both product PRs retain their existing scope.

### 2026-10-29 03:51 UTC: Fresh capacity permits the original matrix to resume

The [read-only metrics refresh](storage-headroom-recovered.json), observed at 03:44:35 UTC, reports **6,667,110,998,854 free bytes** under the existing 100-TiB quota. Together with the successful write/readback, this resolves the earlier capacity request. This is sufficient for the remaining initial-matrix estimate, subject to other workloads' use; recheck it as work advances. No account setting, other owner's data or shared service was changed.

After publishing this source and checking the resource board and both pools again, resume the two prepared native exports and retry SC32/1701 at fresh artifact version `2026.10.28.4`. The latter uses the original initializer, seed and frozen fifty-update recipe through `python -m experiments.post_training.sc_default_study --version 2026.10.28.4 --arm sc32 --seed 1701 --full --cluster cw-rno2a --run`. Its preceding TIS/1701 and SC128/1701 already pass every gate. The failed version-3 startup and preemption remain in accounting. These are staged continuations of the original matrix, requiring at most 32 + 32 + 40 batch H100s, not a study extension. Record each actual submission source and external job identity next.

### 2026-10-29 04:21 UTC: Two recovered exports admitted; four of nine runs qualify

Published clean source `df8833df0c66823d2a4e85d0aecf31dc02ccc82f` submitted both export-only recoveries and SC32/1701 version 4. The [submission record](wave3-submissions.json) binds their exact job IDs and controller configurations. The exports use the direct RNO controller, so they must be inspected with `lib/iris/config/cw-rno2a.yaml`; the training coordinator uses the Marin hub. All thirteen GPU tasks were running at the first capture, and their [live scheduling clocks](wave3-live-allocation-start.json) preserve setup time. The [source comparison](wave3-runtime-source-comparison.json) confirms the study launcher, measurement hooks, canonical verifier and learner worker/trainer paths are unchanged from the earlier SC training source. SC32/1701's actual resolved launch audit passes; no completed training was repeated.

SC128/2903 now has all 9,991 valid quality grades. UID 1806 was retained from its first staged valid verdict. The score-line repair completed UID 6226's ordinary checklist, returning reward 0.75 and therefore a full-task failure. No task was omitted or zero-imputed. Its [collected verification](grug-sc1282903-v3-collected-verification.json) passes all fifty update-health checks and unchanged raw-answer hashes. Equal-family final success is **64.967%**, with **9552.350 seconds** of primary cycle time. Six canonical grade-only calls cover both recovered tasks, including the three failed calls and one diagnostic call outside the finally adopted manifests. This is a descriptive single-seed endpoint; no formal comparison has been run.

The [SC32/4517 export](grug-sc324517-v3-export-r2-verification.json) and [SC128/2903 export](grug-sc1282903-v3-export-r2-verification.json) both pass their exact recovery terminal/job identity, completed native request, 39-shard inventory and sampled attention/router weight checks. Every exporter task succeeded and both local operators exited zero. Their original coordinators remain recorded as stopped after the provider failure; their failed export jobs and costs remain in history. Successful recovery supplies the missing artifact gate without relabeling those earlier failures. The [manifest](runs.json) now admits four initial runs: TIS/1701, SC128/1701, SC128/2903 and SC32/4517.

The recoveries add **8.510 and 8.958 H100-hours**, using live PodScheduled-to-controller-terminal spans with release-time uncertainty stated. All eight exporter pods were absent at the [04:16 capture](wave3-live-allocation-after-exports.json), and their resource-board row is closed. SC32/1701 version 4 continues on its original five-node batch recipe. Next advance the fixed seed orders to TIS/4517 and SC32/2903 at version `2026.10.28.3`, using the ordinary full-study launcher after fresh resource checks and a board announcement. These remain original initial-matrix runs. Formal inference, extension decisions, preservation/cleanup and default recommendations still wait for the required completed work.

### 2026-10-29 04:40 UTC: Next two original runs submitted; SC32 retry training

Published clean source `4b4d4febc8df9af64bf301bdd96b00f2fe68ae09` submitted TIS/4517 and SC32/2903 at version `2026.10.28.3`. Their [submission record](wave4-submissions.json) contains exact coordinator/GPU-child IDs and request hashes. Both preceding seed-order endpoints are fully admitted. These are two original fifty-update runs, each using five-node 40-H100 RNO batch gangs followed by normal native export. The source changes since the last submissions are evidence records; the training recipe and implementation are unchanged.

Before submission, both pools were checked: RNO had 24 unrequested GPUs and East one at 04:25 UTC. A fresh [direct storage check](storage-headroom-pre-wave4.json) at 04:26:54 reports **6.429 TB free** under the unchanged quota. Resource-board row 1587 announces the two gangs, which may queue. At the first 04:38 capture all ten new GPU tasks were building and unbound; controller RUNNING does not establish GPU use. Their actual resolved-launch audits will run once the workers write those documents. The missing documents during this unbound phase are expected.

SC32/1701 version 4 has [four measured updates](grug-sc321701-v4-first-updates.json), with nonzero learning rate, finite positive gradient norm, no skipped updates and complete initial 1,000-task grading. Its existing sampled captured-mass mean at update 4 is 0.99166. These are ordinary progress diagnostics, not a completed endpoint or formal comparison. Four initial runs remain admitted and three are in progress or queued. TIS/2903 and SC128/4517 remain unsubmitted until their preceding arms pass every gate. No extension or default change has been made.

### 2026-10-29 07:25 UTC: SC32 seed 1701 admitted; both next gangs received GPUs

SC32/1701 version 4 completed all fifty planned updates and all original 10,000 endpoint answers. The [collected verification](grug-sc321701-v4-collected-verification.json) passes every update-health and common quality gate, with **67.588% equal-family full-task success**, **7447.345 seconds** of primary cycle time, **61,697,769 consumed loss tokens** and **61,722,730 response tokens**. Its fixed 1,000-task curve is 46.9%, 62.1% and 65.5% at updates 0, 25 and 50. The 347.106-second exclusive midpoint evaluation remains outside the primary cycle span. These are descriptive results; formal comparisons still wait for all nine initial runs.

One unreadable midpoint judge result (UID 306) and three final results (UIDs 756, 6641 and 6961) were recovered from the saved original conversations using the same canonical grader and common recovery-only score-line reader. Each took one canonical grade-only call, for four total. The final rewards were 0.75, 1 and 1 respectively; the first counts as a full-task failure. All original final answer-object hashes are unchanged, and the [midpoint proof](grug-sc321701-v4-midpoint-recovery-verification.json) checks its complete 1,000-task panel. No valid verdict was replaced and no task beyond the nine approved exclusions was omitted.

The normal native exporter succeeded. Its [verification](grug-sc321701-v4-export-verification.json) checks the exact job and terminal record, completed native request, all 39 BF16 shards totaling 134,157,838,824 bytes, and changed sampled router/attention weights against the exact initializer. The training child, five training tasks, export child, four export tasks and CPU coordinator all succeeded. Live PodScheduled-based [training](grug-sc321701-v4-allocation.json) and [export](grug-sc321701-v4-export-allocation.json) reconstruction gives **120.422 + 8.629 = 129.050 H100-hours**. The release capture at 07:24 UTC confirms all nine task pods absent. These are reconstructed spans with controller-terminal uncertainty, not exact billing or certified bounds. The rejected version-3 startup remains separately recorded.

The [manifest](runs.json) now admits five of nine runs and completes the original seed-1701 triplet. TIS/4517 and SC32/2903 have now received all ten requested batch GPU tasks; their [captured state](wave4-training-start.json) distinguishes startup from measured updates. Both actual resolved-launch audits pass ([TIS/4517](grug-tis4517-v3-config-audit.json), [SC32/2903](grug-sc322903-v3-config-audit.json)). Their training source remains `4b4d4febc8df9af64bf301bdd96b00f2fe68ae09`. Keep monitoring those same jobs. TIS/2903 and SC128/4517 remain unsubmitted until their preceding arms pass every gate. No extension or default change has been made; preservation and cleanup remain outstanding.

### 2026-10-29 08:13 UTC: One original machine-grade replay prepared

TIS/4517's initial 1,000-answer panel has 999 valid verdicts. UID 472 reports a machine grader-command failure. The judge-only recovery tool correctly rejected it before making any grading call. Two devbox setup checks also stopped before any canonical grade: first at the ambient signing region for the original East package lock, then because this host cannot start Bubblewrap. Neither produced a replacement verdict or changed a task. Both active training jobs continue normally.

The small [replay driver](replay_machine_grade.py) calls the ordinary `grade_task` with the exact saved conversation, frozen TaskSpec, original local Bubblewrap backend and pinned package lock. It routes the small lock read to its declared signing region within that operator process, saves its one canonical attempt durably before remote I/O, and refuses to replace an existing diagnostic output. It publishes no supplemental verdict. The [plan](machine-grade-replay-plan.json) specifies one four-CPU, zero-GPU RNO batch task and exact input/output hashes and paths. Publish this driver before submitting it; record the actual source and job identity next. If the grader still fails, keep this grade missing while investigating. The common nine-task mask does not authorize another exclusion.

### 2026-10-29 08:24 UTC: Machine replay identifies a format-parser crash; evaluation decision pending

The [one-call CPU replay](machine-grade-replay-result.json), submitted from published source `30144a28f145ecc0e4fe1b19d7a7c89d27a68a9a` as `/romain/sc-default-tis4517-grade0-472-01a16b97-r1`, reproduced the original failure. Frozen validation and package-lock hashes match, and runtime identity `33753eb20463820c` matches the earlier grader qualification. The original prime-factorization task asks for factors separated by the multiplication character. Its pinned scorer rejects LaTeX notation while parsing factors, raising `ValueError` instead of returning a score. This is an answer-format edge case, not the previously identified date-dependent contract problem. The diagnostic used one canonical grading call and zero GPUs. Its first result is preserved; no supplemental grade was published.

The proposed narrow rule would count this specific parser rejection as an incorrect format, reward zero, consistently in every arm while preserving valid existing grades and all raw evidence. It would change handling of a previously undefined grading result after some outcomes were observed, so it needs an explicit evaluation decision. The alternative common exclusion would change the fixed data scope and also needs consultation. Keep UID 472 missing until Romain chooses; the existing nine-task mask remains the only approved exclusion set. Continue the two healthy training jobs and independent checks.

### 2026-10-29 09:56 UTC: TIS seed 4517 preempted after 39 updates; retry at a fresh version

Kueue evicted the TIS/4517 batch gang at 09:41 UTC due to ClusterQueue prioritization. The [preemption proof](tis4517-v3-preemption-verification.json) and [retained events](tis4517-v3-preemption-events.json) identify all five original attempt-0 pods and the subsequent automatic attempt-1 requeue. All 39 recorded updates have finite positive gradients, finite losses/mismatch telemetry, the fixed learning rate and no skipped update. The run cannot enter inference because it did not complete the planned fifty-update endpoint. This is a scheduler interruption; its partial results are retained rather than mixed with another attempt.

The automatic requeue had acquired a new five-node gang during setup. After capturing its state, the owned coordinator and child were stopped to prevent reuse of the same artifact root. Both are authoritatively killed; all ten old/new attempt pods are now absent. A bounded capture preserves 100 original telemetry/evaluation objects totaling **78,176,267 bytes**, and all initial answer-object hashes still match the earlier pre-recovery capture. The original root remains intact. Live scheduling clocks plus retained controller requeue/kill events reconstruct **96.190 + 0.849 = 97.039 H100-hours** for the interrupted training and stopped setup. Queue gaps are excluded; exact release times remain uncertain. These costs remain part of total study overhead.

The format-parser failure discussed in the preceding entry belongs only to this infrastructure-rejected run's initial curve panel. Its original grade remains missing, and no new scoring rule or exclusion was applied. The question can be set aside for the current valid matrix; the approved nine-task mask and frozen grader remain in force. A matching failure in a future valid run would still need resolution.

Retry the original TIS/4517 seed from the same initializer at fresh artifact version `2026.10.28.4`, with all fifty updates and the same normal batch recipe. The [fresh-root check](tis4517-v4-fresh-root-verification.json) confirms that root is unused and no update-50 native checkpoint exists for the interrupted run. The ordinary dry plan passes. The [runtime comparison](tis4517-v4-runtime-source-comparison.json) shows no change to the study launcher, measurement hooks, SkyRL or Verifyit since its training source `4b4d4febc8df9af64bf301bdd96b00f2fe68ae09`.

Before this retry, [both pools](tis4517-v4-capacity-precheck.json) were checked at 09:49 UTC: RNO had 272 unrequested H100s and East two, as snapshots without reservation. A fresh [read-only storage refresh](storage-headroom-pre-retry-tis4517-v4.json) reports **4.646 TB free**, enough for the remaining initial-matrix estimate. Resource-board row 1593 announces the retry. Publish this record before submission, then record its actual source and job identity. This replaces an interrupted attempt within the original matrix; it is not the optional scientific extension. SC32/2903 continues its final checkpoint/evaluation callbacks. Five initial runs remain admitted.

### 2026-10-29 10:07 UTC: Fresh TIS retry admitted to GPUs; SC32 final collection in recovery

The TIS/4517 [version-4 submission](tis4517-v4-submission.json) uses published clean source `ec841c6b8311cbedc949daf4002319b49dbbff88`. All five batch GPU tasks are running and its [actual resolved-launch audit](grug-tis4517-v4-config-audit.json) passes. The original frozen recipe, initializer, seed and fifty-update endpoint are retained. Its coordinator is `/romain/users-romain-checkpoints-sc-default-study-tis-seed4517-2026.10.28.4-coordinator-3cbc13bb`; the submission record binds the exact child and request hashes. Resource-board row 1593 is running. The local submission operator exited zero; the remote run continues.

SC32/2903 has completed fifty training updates, the native checkpoint and all original 10,000 endpoint answers. Both fixed 1,000-task interim panels have complete grades. The first ordinary final collection had 9,989 of 9,991 required common quality verdicts: UID 5076 had an unreadable judge reply, and UID 2046 exceeded the live verifier deadline. UID 5076 now has its first valid judge verdict, reward 0.75 and therefore a full-task failure, published from its durable stage after one canonical grade call. The normal four-node native exporter is running. This run remains ineligible until both final verdicts and native export pass.

The already published [machine replay driver](replay_machine_grade.py) will replay UID 2046 in one small CPU task using its exact saved conversation, original TaskSpec/local backend/package lock and original 330-second script timeout. The [new replay plan](sc322903-final-machine-replay-plan.json) specifies all paths and hashes before submission. It produces one durable diagnostic result and never publishes a supplemental grade by itself. A timeout stays missing until a valid original verdict is recovered; no answer is regenerated and no additional task is excluded. Formal inference still waits for all nine initial runs.

### 2026-10-29 10:44 UTC: Format-rule approval recorded; bounded machine-verifier recovery prepared

Romain approved counting the specific prime-factor integer-parser rejection as an incorrect required answer format, consistently in every arm, while keeping the task. The [approved rule](approved-prime-factor-format-rule.json) records his exact answer, observed receipt at 10:16:35 UTC, narrow traceback/task/package conditions and post-outcome timing. It covers no arbitrary runtime or judge failure. The first observed case remains the initial panel of the rejected 39-update TIS/4517 attempt; scoring that format failure never readmits the interrupted run. The original nine-task exclusion mask remains fixed. Apply the rule to matching missing evaluation grades in any arm, preserving all already valid verdicts and all raw evidence.

SC32/2903's normal [native export verification](grug-sc322903-v3-export-verification.json) passes the exact terminal/job, completed request, 39 indexed shards and sampled weight changes. Its coordinator and all nine training/export tasks succeeded; their pods were absent at 10:28 UTC. Live-clock reconstruction gives **119.901 + 9.366 = 129.268 H100-hours**. GPU resource-board row 1587 is closed. The run remains ineligible because UID 2046 still lacks a final grade. TIS/4517 version 4 is training normally and has complete initial 1,000-task grading.

UID 2046's [one-call original machine replay](sc322903-machine-replay-r1-result.json) hit the same process resource cap after 301.512 seconds. Its frozen scorer gives each of 101 tests ten seconds, but an independent child watchdog caps the whole suite at 300 seconds; its binary adapter marks that expiry as infrastructure error. No zero was inferred. The [new bounded replay plan](sc322903-final-machine-replay-r2-plan.json) uses the scorer's already defined finite whole-suite window, `(10 + 1) * 101 + 5 = 1116` seconds, with a 1176-second enclosing script window. Original tests, candidate code, test order, first-failure short circuit, individual ten-second budgets, four-GiB child limit and package lock remain fixed. Only the separate overall watchdog and enclosing script timeout change in the research-only recovery copy. Both original and operational TaskSpec identities are recorded. Apply the same recovery to an equivalent missing grade in any arm; a timeout itself still supplies no score.

The [local qualification](machine-recovery-local-qualification.json) checks actual frozen resources and the original parser/timeout diagnostics without any grading call. A [separate CPU control](lcb-window-control-plan.json) will reuse a captured successful program from SC128/2903 against these same 101 tests; it must still score one and can never replace its original valid grade. Publish the changed [replay driver](replay_machine_grade.py), these plans and the small [publication checker](publish_machine_grade.py) before the CPU jobs or supplemental adoption. The checker validates saved-answer, task, package, source and operational-window identities, stages the first result durably, preserves other valid recoveries and makes no grading call. All product code, learner configuration and default settings remain unchanged.

### 2026-10-29 11:12 UTC: SC32 seed 2903 admitted; six of nine initial runs qualify

The [bounded missing-grade replay and control](lcb-window-recovery-verification.json) both pass all 101 original tests. UID 2046's saved SC32/2903 program scores one after **374.099 seconds**; the captured successful SC128/2903 control retains score one after **243.601 seconds**. The former was a correct answer cut off by the independent 300-second resource watchdog. The individual ten-second test limits, original tests, ordering, short circuit, candidate code and four-GiB memory cap are preserved. The first failed original-budget call remains recorded; a timeout supplied no score. The [submission record](lcb-window-recovery-submissions.json) binds both successful zero-GPU CPU jobs to published source `a5e3e173a8320d7ef29a4c0e54aa3fbf2e87b9c1`.

The publication checker validates all frozen inputs and the published operational-window copy, stages the recovered result and adds it without another grading call, preserving UID 5076's earlier reward 0.75. The ordinary collector now accepts all original 10,000 answers and all 9,991 common quality verdicts. Its [verification](grug-sc322903-v3-collected-verification.json) confirms unchanged original answer-object hashes and every one of the fifty update-health checks. Two accepted supplemental verdicts used two calls; the prior failed machine call makes **three actual missing-grade recovery calls**, plus **one separate qualification call**. These scopes are explicit in the verification.

The descriptive endpoint is **68.040% equal-family full-task success**, with **7348.736 seconds** of primary complete-cycle time, **61,744,411 consumed loss tokens** and **61,776,921 response tokens**. The fixed 1,000-task curve at updates 0, 25 and 50 is 45.3%, 62.7%, 67.1%. All training, quality, measurement and [native export](grug-sc322903-v3-export-verification.json) gates pass. The [manifest](runs.json) admits this sixth initial run. These single-seed endpoints are descriptive; formal comparisons still wait for the original nine-run matrix.

The user-approved specific format rule is also applied to the preserved original prime-factor diagnostic in the rejected TIS/4517 version-3 initial panel, yielding reward zero with **no additional grading call**. The [normal interim collection](grug-tis4517-v3-initial-recovery-verification.json) has all 1,000 grades and identical original answer-object hashes. The 39-update run stays ineligible. No valid initial-matrix endpoint has needed that format rule so far, and the same approved rule remains available to every arm; the nine-task mask is unchanged.

Next submit TIS/2903 at unused version `2026.10.28.3`, the last arm in its original SC128 → SC32 → TIS seed order. Both preceding arms now pass every gate. The [fresh root](tis2903-v3-fresh-root-verification.json) is unused. [Both pools](tis2903-v3-capacity-precheck.json) were checked at 11:00 UTC: RNO had zero unrequested H100s and East seven. Keep the RNO five-node batch recipe and allow it to queue. Resource-board row 1594 announces the submission. A fresh [read-only storage check](storage-headroom-pre-tis2903-v3.json) reports **21.530 TB free** at 11:01 UTC under the unchanged quota; no other owner's data or account setting was changed by this study. Publish this record before launch and record its actual source and IDs afterward. TIS/4517 version 4 continues healthy training; SC128/4517 waits for its gates. No scientific extension or default change has been made.

### 2026-10-29 11:35 UTC: Last seed-2903 arm submitted and admitted to GPUs

The [TIS/2903 submission](tis2903-v3-submission.json) uses published clean source `819c48c79d35f7fe2c81228fbf6df6a9630ef65d` and version `2026.10.28.3`. All five tasks in its 40-H100 batch gang have received GPUs despite the earlier full-pool snapshot, and its [actual resolved launch audit](grug-tis2903-v3-config-audit.json) passes. Resource-board row 1594 is running. The local submitter exited zero; the exact coordinator, child and request hashes are preserved in the submission record. The [latest state](tis-baseline-training-progress.json) also shows TIS/4517 version 4 continuing healthy training with a complete midpoint evaluation. Six initial runs are admitted, two are running, and only SC128/4517 remains unsubmitted behind its original seed-order gates. No scientific extension is declared.

A bounded [interim evidence snapshot](tis4517-v4-interim-evidence-inventory.json) preserves TIS/4517's existing update-0/update-25 answer objects and ordinary telemetry: 73 objects totaling 73,488,424 bytes in the owned native session artifacts. This protects the original interim evidence if a future scheduler requeue attempts the same artifact root. It adds no model, grader or gradient call and changes no source object.

### 2026-10-29 11:51 UTC: Terminal resource records consolidated without duplicate attempts

The small [allocation combiner](allocation_accounting.py) replays the historical sixteen-job reconstruction, the two standalone recovered exports, both later completed SC32 training/export pairs and the interrupted TIS/4517 attempt. Its [qualification](allocation-accounting-qualification.json) recomputes all retained clocks and rejects a duplicated real allocation proof. The supplied terminal subset contains **23 GPU jobs and 92 executed task attempts**, reconstructing **993.825 H100-hours**. The earlier **2.602 H100-hours of observed preexecution quota reservation** remain separate because physical GPU use was not established. Both currently training TIS runs are excluded from this subtotal and will be added only with their own terminal proofs. Failed attempts, stopped automatic retries and exports remain counted once under their actual identities. This is resource accounting, with the original clock limitations retained; no model/grader call or scientific comparison was added.

### 2026-10-29 12:02 UTC: Exact-byte preservation prepared; bulk I/O remains deferred

The [preservation driver](preserve.py) generalizes the deferred first-baseline helper to the owned stopped-run scratch prefixes. It makes a content-addressed inventory, preserves exact original paths and identities, copies with two CPU threads, and verifies full SHA256 equality for every source/archive object. A resumable checkpoint is bound to that exact input inventory. An existing unverified derived copy is streamed again if its bytes differ; no source object is changed or deleted. Completion requires the entire original source tree to remain unchanged and publishes a content-addressed restore manifest. Object names retain the earlier trial's metadata identity so its roughly 49-GB copy can be fully checked and reused; matching object size or multipart ETags alone never establishes preservation.

The [bounded qualification plan](preservation-qualification-plan.json) uses three tiny owned control files, including an empty object and a deliberately wrong same-size derived archive member. Publish this driver and plan before the real storage round trip. Verify full hashes, unchanged originals, repair and identical-manifest resume, with zero GPU/model/grader calls. This small control does not qualify large multipart copying by itself. Full native preservation remains deferred until measured GPU consumers stop, followed by a fresh resource-board/storage check and a published finite bulk plan. Temporary checkpoint sources remain intact.

### 2026-10-29 12:05 UTC: Small real-storage preservation control passes

The [real control](preservation-control-verification.json), run from published clean source `25e3994632bde471f5815fcfefe7b6b0f9497a55`, passes on three owned RNO objects totaling **257,079 bytes**. Full source/archive SHA256 digests match the declared payloads, including the empty object. The deliberately wrong same-size derived member is detected and repaired by streaming, while every original payload stays unchanged. A second invocation reuses the verified members and returns the exact same content-addressed complete manifest, `429656730db36234de42f9ea197b8b385c3542bf91f4b6f23623d3be74960736`. No source object was deleted; no GPU, model or grader call was made. The proof binds the plan, actual tool, operator and execution source hashes. This qualifies small-object hash/repair/resume behavior; full-size multipart verification and all bulk copying remain deferred until measured GPU consumers stop. Both TIS baselines continue healthy training.

### 2026-10-29 12:28 UTC: Reporting shows elapsed clocks and both interval levels explicitly

The [plotter](plots.py) now uses the already retained elapsed clock for the learning curve's time axis, so the exclusive midpoint evaluation pause appears in elapsed time from first rollout. The cost curve continues to show the primary complete-cycle clock with exclusive evaluations removed. The endpoint chart displays black nominal 95% intervals inside the gray predeclared 97.5% decision intervals and names the latter as the decision gate. This exposes both existing protocol quantities directly in the figures.

The [presentation qualification](presentation-qualification.json) renders all three SVG/PNG pairs from nine clearly synthetic run summaries. A 120-second evaluation pause gives 620 seconds of elapsed time versus 500 seconds of primary cycle time. The synthetic outer quality interval crosses the fixed margin while its inner interval does not, making the different decision role visible. The learning and endpoint figures were visually inspected and are legible. All six currently admitted real summaries already retain the needed ordinary elapsed field. No real paired analysis, model/grader call, new measurement or inferential change was made. Formal comparisons still wait for all nine initial runs.

### 2026-10-29 13:17 UTC: TIS seed 4517 admitted; final initial arm ready to submit

TIS/4517 version 4 completes all fifty planned updates and all original 10,000 endpoint answers. Its [collected verification](grug-tis4517-v4-collected-verification.json) accepts every common quality grade and all update-health gates, with **67.388% equal-family full-task success**, **6824.343 seconds** of primary cycle time, **60,656,830 consumed loss tokens** and **60,670,477 response tokens**. The fixed 1,000-task full-success curve is 0.455/0.621/0.670 at updates 0, 25 and 50. The run's actual training source remains `ec841c6b8311cbedc949daf4002319b49dbbff88`. These are descriptive single-seed results; formal comparisons still wait for all nine initial runs.

Eight final judge verdicts were unreadable. The first recovery for UID 881 made three unsuccessful canonical calls during a Together HTTP503 outage; its original pending diagnostics are preserved separately. After a later valid reply, all eight saved answers obtained their first valid canonical verdicts in one call each, using the common recovery-only score-line reader. Thus accepted manifests contain eight calls and the actual recovery total is **eleven**, including the three failed calls. Publication and collection add no calls. All original final answer-object hashes match their pre-recovery capture. No task was omitted or timeout scored zero. TIS/2903's midpoint also hit the service outage: its [recovery verification](grug-tis2903-v3-midpoint-recovery-verification.json) restores 32 missing verdicts in 32 calls, retaining all 1,000 original answers and identical source-object hashes. Its training continues with healthy measured updates.

The normal [TIS/4517 native export](grug-tis4517-v4-export-verification.json) passes the exact job/terminal identity, completed request, all 39 indexed BF16 shards totaling 134,157,838,824 bytes and changed sampled router/attention weights. The five training tasks, four export tasks and coordinator all succeeded. All nine GPU task pods are absent. The live-clock [training](grug-tis4517-v4-allocation.json) and [export](grug-tis4517-v4-export-allocation.json) proofs reconstruct **116.101 + 8.546 = 124.646 H100-hours**, with the original release-time limitations retained. Resource-board row 1593 is closed. The [manifest](runs.json) now admits seven initial runs. The rejected 39-update version-3 attempt remains separately preserved and excluded.

Next submit **SC128/4517 at unused version `2026.10.28.3`**, the last arm in its original SC32 → TIS → SC128 order. Both preceding arms pass every gate. Its [fresh permanent and temporary roots](sc1284517-v3-fresh-root-verification.json) are unused, and its [runtime source comparison](sc1284517-v3-runtime-source-comparison.json) matches the earlier SC128 training source on the launcher, measurement hooks, SkyRL and verifier paths. The frozen fifty-update recipe uses five RNO batch nodes/40 H100s and the normal 32-H100 native exporter. Resource-board row **1604** announces it. [Both pools](sc1284517-v3-capacity-precheck.json) were checked at 13:15 UTC: RNO had 88 unrequested H100s and East zero, as snapshots without reservation. The latest [read-only storage check](storage-headroom-pre-sc1284517-v3.json), observed at 12:55 UTC, reports **18.013 TB free** under the unchanged quota. Publish this source before launch and record the actual source and external identities afterward. That submission will complete the original initial matrix; no scientific extension or default change is declared.

### 2026-10-29 13:22 UTC: All nine initial runs submitted; last arm has received GPUs

The [SC128/4517 submission](sc1284517-v3-submission.json) uses published clean source `25f7f13932293ab4ec00584bef8a431504b2d4db` and fresh version `2026.10.28.3`. Its coordinator is `/romain/users-romain-checkpoints-sc-default-study-sc128-seed4517-2026.10.28.3-coordinator-97f9d527`, with the exact GPU child and request hashes bound in the record. The local operator exited zero. All five tasks have received GPUs, and the [actual resolved-launch audit](grug-sc1284517-v3-config-audit.json) passes. Resource-board row 1604 is running. The frozen fifty-update recipe, initializer, seed, normal weight updates, batch priority and RNO layout are retained.

All **nine original initial-matrix runs are now submitted**. Seven are fully admitted; TIS/2903 and SC128/4517 continue on their normal 40-H100 batch gangs. The former has complete initial and midpoint grading after its saved-answer recovery. The last arm is starting and has no measured update yet. Formal paired inference and the single-extension decision wait for both final endpoints and native exports. No scientific extension has been declared and product defaults remain unchanged.

### 2026-10-29 14:54 UTC: Last baseline finishes; apply the common saved-answer recovery

TIS/2903 version 3 completes all fifty updates and all original 10,000 endpoint answers. Every training task succeeded and its five pods are absent. Its [training allocation](grug-tis2903-v3-allocation.json) reconstructs **136.218 H100-hours**; the normal four-node native exporter is running. The first final collection has 9,964 of the required 9,991 common quality verdicts. Twenty-six missing judge results include unreadable score replies and a transient HTTP503 interval; their canonical saved-answer recovery is running with the existing common score-line reader. The initial and midpoint panels remain complete.

The remaining missing result is the same frozen code task, UID 2046, whose live verifier deadline expired. The [one-call CPU replay plan](tis2903-final-machine-replay-plan.json) applies the already qualified finite whole-suite window to this TIS answer. Its [identity preflight](tis2903-machine-replay-identity-verification.json) verifies the exact same original task, 101 tests, package lock, operational TaskSpec and unchanged replay driver used by the earlier recovery and successful control. Preserve all individual ten-second limits, test order, binary scoring and four-GiB child memory cap; allow the existing 1116-second child window inside a 1176-second script window. No score is inferred from the timeout. Publish this plan before the single zero-GPU, four-CPU batch task. The first valid machine verdict will be checked against the saved answer and all frozen identities before supplemental adoption. Judge publication must finish before the machine row is added to that panel.

Seven initial runs are admitted. This baseline remains pending its missing grades and native export; SC128/4517 continues healthy training. All nine initial endpoints are still required before formal paired inference or the extension decision. No new workload, exclusion, backend or product change is introduced.

### 2026-10-29 15:21 UTC: Last baseline admitted; eight of nine initial runs qualify

The [single CPU recovery](tis2903-machine-replay-verification.json), submitted from published source `2dfd00f64af7af1f28d80931a64fcc5634d76ff4`, returns reward one after **581.907 seconds**. Its original verifier report records **101 executed tests out of 101, all true**. The TIS answer was correct and exceeded the independent process watchdog. The qualified finite whole-suite recovery preserves its original tests, ten-second individual limits, order, binary score, local backend and four-GiB child cap. The [submission record](tis2903-machine-replay-submission.json) binds the exact zero-GPU job, source, command and request hash. No answer is regenerated and no timeout supplies a score.

The publication checker validates every frozen identity and original saved-answer hash, stages the machine result separately and merges it while preserving the 26 previously published final judge rows. It makes zero additional calls. Normal collection now accepts all original 10,000 endpoint answers and all 9,991 common quality verdicts. Its [verification](grug-tis2903-v3-collected-verification.json) passes all fifty update-health checks and unchanged original final-answer hashes. Grade-only recovery used **32 midpoint judge calls + 26 final judge calls + one final machine call = 59 canonical calls**. No new control or unsuccessful extra replay was needed.

The descriptive endpoint is **64.495% equal-family full-task success**, **8636.544 seconds** of primary cycle time, **70,501,978 consumed loss tokens** and **70,538,025 response tokens**. The fixed 1,000-task full-success curve is 0.448/0.591/0.628 at updates 0, 25 and 50. The normal [native export](grug-tis2903-v3-export-verification.json) passes its exact job/terminal identity, completed request, all 39 indexed BF16 shards totaling 134,157,838,824 bytes and changed sampled router/attention weights. Training, exporter and coordinator all succeeded; all nine GPU task pods are absent. The live-clock [training](grug-tis2903-v3-allocation.json) and [export](grug-tis2903-v3-export-allocation.json) proofs reconstruct **136.218 + 9.097 = 145.314 H100-hours**, retaining the documented clock limits. Resource-board row 1594 is closed.

The [manifest](runs.json) now admits eight initial runs. Only SC128/4517 remains in progress, with healthy updates and complete initial and midpoint 1,000-task grading; neither panel needed recovery. All three TIS seeds are complete, but formal paired inference and the extension decision still wait for the ninth endpoint and native export. No scientific extension, additional exclusion or default change is declared.

### 2026-10-29 16:35 UTC: Shared storage quota blocks reporting after the last arm's update 40

SC128/4517 publishes update-40 model weights at **15:41:13 UTC**, then stops advancing its durable metrics beyond update 39. The [incident verification](sc1284517-step40-storage-hold.json) preserves the completed update's optimizer diagnostics and log identity. A bounded thread capture of the confirmed study entrypoint shows the main thread waiting in S3 commit beneath `write_json` at the cycle-report write. The entrypoint remains alive and sleeping; all five original task attempts remain running. No numerical failure is observed, and no restart has been launched.

A fresh official read-only [storage workflow](https://github.com/marin-community/marin/actions/runs/40906349444), observed at 15:55 UTC, reports the shared RNO account **636,959,327,416 bytes over its existing 100-TiB quota**. A small owned write/readback probe exceeds its 25-second bound without a completed result. This is a recurrence after the earlier 18.013-TB headroom observation; no other owner's data or account setting was changed by this study. The devbox has only about 160 GiB free, below one native checkpoint, so a local full-state archive cannot solve this hold.

The eight completed native update-50 checkpoints total **6,445,345,917,876 bytes** and have no verified full-state archive. Their portable final models, all original evaluation answers/results and ordinary telemetry remain stored separately. Romain has been asked whether the study may retain those verified final models/results while discarding completed native restart/optimizer state. That would change the preservation plan; **the decision is pending and no native state has been deleted**.

The [read-only disposal planner](discard_native_state.py) freshly rechecks all eight final model exports and their recorded manifest/sample hashes, verifies the exact native inventories, and selects only their 256 `.distcp` tensor shards, totaling **6,440,504,852,224 bytes**. It retains all other native metadata, including completed export requests, and excludes the active run. Its disposal command requires a matching explicit approval record and preserves per-run receipts. Source lint and the actual read-only preflight pass; no disposal command has been run. Resource-board rows 1586 and 1604 record the quota hold and current 40-H100 allocation. Eight initial runs remain admitted; formal inference, any extension decision and native cleanup remain subject to their existing gates.

### 2026-10-29 17:05 UTC: A bounded private probe confirms provider write suspension

A fresh [study-owned small-object probe](storage-write-denial-20261029-1705.json)
returns HTTP **405 MethodNotAllowed** in 0.187 seconds with the exact provider
message `Account is write suspended for this availability zone. Please check
capacity quota and contact support.` It uses one request attempt with five-second
connect and ten-second read bounds, removing `Expect` only on that private
probe client. No running trainer or shared client configuration is changed.
This provides a direct provider response in addition to the earlier 25-second
probe's inconclusive timeout. The [16:58 UTC collector](https://github.com/marin-community/marin/actions/runs/40916651419)
still reports the same 636,959,327,416-byte quota excess.

The existing [shared October 29 incident](https://marina.oa.dev/echo/wiki/893)
now includes this study's update-40 reporting hold; no duplicate incident was
created. All five live task attempts remain running with durable telemetry at
39. The retention decision is still pending, and no native restart state has
been deleted. Formal inference continues to wait for the ninth valid endpoint.

### 2026-10-29 17:45 UTC: Bound the idle allocation while preserving the interrupted run's evidence

The provider still rejects the private study write at 17:40 UTC. With no
capacity remedy yet authorized, keeping forty H100s idle costs another forty
H100-hours per hour. The [operational stop plan](sc1284517-storage-hold-stop-plan.json)
sets 18:15 UTC (11:15 a.m. Pacific) to release this exact owned coordinator and
its descendants if writes remain blocked and no authorized recovery is underway.
Romain has been told the deadline. This avoids an indefinite idle allocation;
it does not end the study or authorize native checkpoint deletion.

A bounded local snapshot now preserves **99 existing objects totaling
75,584,683 bytes**, including all original initial/midpoint answers and ordinary
telemetry. Every captured object's SHA256 is rechecked locally. The plan records
the exact source, capture-manifest identity and restore directory. It does not
preserve the live optimizer; no step-40 native checkpoint exists. If stopped,
the forty-update attempt remains ineligible under the existing fifty-update
admission gate, with all consumed GPU-hours retained. Complete the same arm
later from the frozen initializer at a fresh artifact version after verified
storage recovery. Apply the same operational response to any arm with this
external hold. No valid endpoint or cycle clock is replaced or adjusted, and
no scientific extension is declared.

### 2026-10-29 18:20 UTC: Storage-stalled SC128 seed 4517 stopped; all forty GPUs released

The private write probe still returns HTTP 405 at 18:15 UTC, with no authorized
capacity remedy underway. The [published stop plan](sc1284517-storage-hold-stop-plan.json)
is executed against only this run's exact coordinator at 18:16 UTC. The
[verification](sc1284517-v3-storage-stop-verification.json) records coordinator
and all five GPU tasks killed, their original attempt identities, and all task
pods absent by 18:17 UTC. The live-clock [allocation proof](grug-sc1284517-v3-allocation.json)
retains **197.524 H100-hours**, including startup, useful work and the storage
hold, with the same clock limitations as the other proofs. Resource-board row
1604 is closed; this study now has no active GPU allocation or grade-replay job.

Forty optimizer updates and sampler publications completed, but durable metrics
end at 39. The blocked reporting write never completed. This partial attempt
has no update-50 quality endpoint or native export and remains ineligible under
the frozen gates. Its in-memory optimizer state was lost at termination; the
verified 99-object original-evidence snapshot and unchanged remote answers and
telemetry remain preserved. No completed native restart checkpoint was deleted,
and the retention decision remains pending. The [manifest](runs.json) retains
the stopped attempt and its cost rather than replacing its history.

Eight initial runs remain admitted. After verified storage recovery, complete
the same SC128/4517 arm from the fixed initializer at a fresh artifact version.
The failed infrastructure attempt is separate from that replacement's valid
fifty-update cycle comparison; total study allocation includes both. No partial
endpoint, shortened training duration, adjusted valid-run clock, extra task
exclusion, default change or scientific extension is adopted. Formal inference
and any justified finite extension still wait for a ninth admitted endpoint.

### 2026-10-29 18:38 UTC: Qualify full-state backup in East before concluding the storage hold is an impasse

The previous goal turn made progress by releasing the stalled allocation and
publishing its evidence. Fresh checks still show provider write suspension in
RNO and no surviving study GPU pods. The [18:25 UTC official collector](storage-headroom-east-preservation.json),
however, reports **270,519,747,299,050 bytes of East-02A headroom**. A full native
backup there may restore RNO capacity while preserving every optimizer byte,
under the goal's existing verified-preservation cleanup rule. This is an
alternative to discarding unique restart state; that separate retention-change
question remains unanswered and cannot authorize such a discard.

The [archive driver](preserve.py) now resolves source and archive filesystems
separately and streams between the two explicitly owned bucket namespaces,
hashing the source during copying and reading the archive back in full. It
retains the same immutable input, unchanged-source, exact-restore and resume
checks. It may inventory a run's `checkpoints/` subtree, which avoids copying
14,536 temporary attempt-log objects merely to preserve one native checkpoint.
The first TIS run's checkpoint tree is 45 objects and **805,115,247,605 bytes**,
including the two-byte latest-step marker. All source objects remain read-only.

Publish the [tiny control plan](cross-region-preservation-control-plan.json) and
[control operator](qualify_preservation.py) before execution. The control reuses
the three original 257,079-byte RNO payloads, deliberately corrupts only a new
East-derived copy, and checks full bytes, repair and identical-manifest resume.
It uses no GPU, model, grader or native checkpoint transfer. Only after it
passes, publish and coordinate one bounded real-checkpoint pilot. No source
cleanup is permitted without that pilot's verified full-content archive and
stopped-consumer proof; all ordinary native metadata must remain available for
the final-model verifier. No study recipe, endpoint or inference rule changes.

### 2026-10-29 18:43 UTC: East tiny backup control passes; qualify the real CPU launch before the native pilot

The [local cross-region control](cross-region-preservation-control-verification.json)
passes under published clean source `b521b5958fea807fa1e364ec2fc00471a8dbfb6d`.
All three original RNO objects remain byte-for-byte unchanged. The new East
copy has matching full SHA256s, the deliberately wrong same-size member is
repaired, and a second invocation returns the identical complete manifest.
The immutable complete-manifest SHA is
`02339074f67fb83a532e628cd8fa55aed6c9899d1ec7579d6053acaaadbc510e`.
No native checkpoint was copied or deleted.

The [next tiny control](cross-region-preservation-iris-control-plan.json) repeats
this path inside the actual RNO Iris CPU environment, including source packaging
and dependency setup, before any large transfer. Its portable operator checks
its own and the archive tool's published hashes; the client records clean Git
source and the actual resolved job identity. Iris result capture uses its
ordinary output directory. The finite [native pilot](native-preservation-pilot-plan.json)
then covers only the first completed TIS checkpoint tree: 45 objects,
805,115,247,605 bytes, two streaming threads, zero GPUs. Source inventory and
archive readback must both pass in full. Only a later separately guarded cleanup
may use that verified full-state archive under the existing preservation rule.
The unresolved unique-state-discard question does not grant cleanup authority
and is unnecessary if this full-preservation route succeeds.

### 2026-10-29 18:58 UTC: Actual CPU backup control succeeds; one full native checkpoint is ready to copy

The [actual Iris control](cross-region-preservation-iris-control-verification.json)
succeeds under clean published source `c108fbaf4f079d9f056ca38796b5228009f9a0dc`
in job `/romain/sc-default-preserve-control-east-01a16b97-r1`. Its actual request
has four CPUs, sixteen GiB RAM and no GPU. The unchanged source/target hashes,
wrong-copy repair and identical-manifest resume all pass. The complete manifest
`3ee5b7791a74d5947fd47393b53e5ff5fdbb99ea021da141c6530db27113e6bc`
is independently read back and its digest checked. The task then spends its
bounded finalization period on local-result capture before reporting success;
large-copy receipts go directly to East, and that driver emits no local output
file. The control and prior stopped GPU task pods are now absent.

The [full native pilot's immutable input](native-preservation-pilot-input.json)
contains exactly **45 objects and 805,115,247,605 bytes**. Its SHA is
`ff0161d8ab60c6d0787441c6c50d4f7d6bf0c94f9c1213cc70ea9809ef723a35`.
Only inventory metadata has been written to East; no real checkpoint tensor
has been transferred or removed yet. The [finite plan](native-preservation-pilot-plan.json)
uses the same qualified archive code, two streaming threads and one CPU job,
with a twelve-hour queue-plus-run ceiling and no GPUs. Shared storage rows
1586 and 1624 announce the RNO read and East archive. Publish this evidence
before launching the real copy, record its actual source and job, and verify
the complete full-byte archive before considering any restorable source cleanup.
All measured GPU consumers remain stopped.

### 2026-10-29 19:05 UTC: One full native archive is running on a CPU worker

The [actual pilot submission](native-preservation-pilot-submission.json) uses
published clean source `e867ae0b2a777b1dc0267d8d4336f5ec7c9a2efe` in
`/romain/sc-default-preserve-tis1701-01a16b97-r1`. The resolved request SHA is
`a2d3579b95cb9c064444b6e01e09018d95a29bebca45abc6aae8fecbc7b5ff6a`.
The controller reports running on its original attempt with four CPUs, sixteen
GiB RAM, 128 GiB disk and **zero GPUs**. The submitted command binds the exact
immutable 45-object input. Source files remain read-only.

The archive progress receipt is live in East and reports copying/verifying
with two threads, with no source deletion. Large members count as verified
only after their full archive readback. Shared storage row 1624 is running;
row 1586 retains the RNO source/cleanup context. All measured study GPU runs
remain terminal. Next verify this complete 805.115-GB native archive, then use
its full-state restore proof to assess source cleanup under the original
preservation policy. The optional unique-state discard question is still
unanswered; this backup route retains that state and does not require it.
No scientific inference, new training arm, extension or default change is made.

### 2026-10-29 19:24 UTC: Prepare cleanup that retains the complete native restart state

The [preserved-state cleanup helper](cleanup_preserved_native.py) is prepared
while the native archive runs. Its read-only plan requires a complete immutable
full-content manifest, the original input digest and scope, matching source
and archive object identities, the recorded archive job request, and fresh
controller checks that the original coordinator, learner, exporters and archive
worker are all terminal. It also reuses the current portable-model verifier.
Only the exact 32 covered native tensor shards may be removed; all checkpoint
metadata, including the export request and latest-step marker, stays in RNO.
The complete native restart state and exact restore paths remain in East.

This route implements the goal's original verified-preservation cleanup rule.
It does not weaken or invoke the separate [unique-state disposal](discard_native_state.py)
approval path, whose retention-change question remains pending. A bound local
intent/receipt supports a partial removal retry without accepting changed
source, archive or metadata. Source lint passes, but the real read-only
preflight still awaits the full native archive; **no cleanup command has been
executed**. The active 805.115-GB pilot retains its published source and all
measured study GPU consumers remain stopped.

### 2026-10-29 19:40 UTC: Full-size multipart members verify; the complete checkpoint is still pending

The [per-member checkpoint](native-preservation-large-members-check.json) now
records 11 verified objects totaling **117,970,591,549 bytes**,
including two 58,967,880,123-byte native tensor shards. Both large files pass
streaming source SHA256 and full archive readback. This provides actual
full-size multipart evidence beyond the tiny controls; it does not complete
the 45-object native archive or its final unchanged-source check. The original
checkpoint remains intact and no source cleanup is allowed on this partial
evidence. The CPU worker is healthy and uses no GPU.

The prepared cleanup helper's [live consumer check](preserved-cleanup-consumer-check.json)
also verifies the completed control job and its recorded request digest, while
refusing the still-running archive worker. It makes no model, grader or source
delete call. The full archive preflight and removal remain unexecuted. Continue
monitoring the same archive job; extend quiet checks to roughly fifteen minutes
while large objects finish. RNO still explicitly rejects writes at 19:33 UTC.

### 2026-10-29 21:00 UTC: Writes recover, but free capacity is still below one run's checkpoint/export requirement

A [small write/readback and a five-MiB multipart round trip](storage-write-recovery-insufficient-headroom.json)
pass at 20:47–20:50 UTC and remove their synthetic objects. The fresh official
collector still shows the unchanged 100-TiB quota, with only **69,932,525,133
bytes free**. This study has deleted no native checkpoint state; the external
recovery action is unknown. Successful tiny writes do not supply the roughly
one-TB native-checkpoint plus portable-export capacity needed for the last arm.
Do not allocate another GPU run on this evidence. Continue the full-state
archive route and recheck capacity before retrying.

The [preparatory runtime comparison](sc1284517-retry-runtime-source-comparison.json)
confirms that complete `experiments`, `vendor`, `lib` and `infra` trees, plus
`uv.lock` and `pyproject.toml`, are unchanged from the last SC128/4517 training
source. Archive-only helpers do not change its recipe. Fresh-root, resolved
launch, both pools and resource coordination checks remain required at launch.

Only metadata has been prepared for the [remaining seven completed checkpoints](native-preservation-remaining-plan.json):
315 objects totaling **5,640,230,670,287 bytes** in immutable per-run input
manifests. No additional native transfer has begun. After the entire first
805-GB pilot passes and its worker stops, the finite batch may use at most seven
CPU copy jobs with two streams each, under refreshed headroom and shared-resource
checks. All measured GPU consumers must stay stopped during that bulk I/O.
This completes the original full-state preservation scope; no unique optimizer
state is discarded and any later source cleanup keeps exact restore proofs.

### 2026-10-29 21:41 UTC: Eleven TB of headroom permits preparing the last initial arm's fresh attempt

A new [official headroom observation](storage-headroom-pre-sc1284517-v4.json)
reports **11,170,336,264,899 bytes free at 21:26 UTC** under the same quota.
The external capacity action remains unknown; this study still deleted no
native state. This now exceeds the checkpoint/export requirement. Finish and
verify the nearly complete CPU pilot before the measured GPU run; defer the
other seven large archives until all measured consumers stop again.

The [fresh SC128/4517 version-4 plan](sc1284517-v4-retry-plan.json) retains the
same initializer, frozen data, fifty updates and forty-H100 RNO batch layout.
Its [dependency graph](sc1284517-v4-dry-graph-verification.json) builds with exit
zero without executing training. Both the permanent and scratch roots are
[unused](sc1284517-v4-fresh-root-verification.json). The [pool snapshot](sc1284517-v4-capacity-precheck.json)
shows 48 unrequested RNO H100s and none in East; this is not a reservation.
Resource-board row 1629 announces the same-region batch
request. The whole runtime/dependency tree comparison already matches the
prior source. Publish this plan before launch, recheck readiness, then capture
the actual source, IDs and resolved configuration. No GPU has been submitted
and no scientific extension is declared. The stopped version-3 attempt and
all its allocation cost remain recorded.

### 2026-10-29 21:52 UTC: The full native pilot and restorable-cleanup preflight pass

The [full pilot verification](native-preservation-pilot-verification.json)
passes for all **45 objects and 805,115,247,605 bytes**. Its immutable complete
manifest SHA is
`167ffb077b0c41b34df51f36f8557a6c1689dfe4f61adb1d8935cba8ef00bc70`.
Every source/archive SHA256 matches and the full original source inventory
remains unchanged. The worker succeeds; Kubernetes reports its task and output
uploader both terminated with exit zero. The completed Pod object remains
visible for retention, but no copy command is running. This is the first
fully preserved native optimizer/restart checkpoint, not just sampled weights.

The [real read-only cleanup preflight](native-preservation-pilot-cleanup-preflight.json)
also passes. It rechecks the immutable input and complete digests, all source
and archive object identities, the exact archive request, the current portable
model, and live terminal state for all four original/copy consumers. It selects
only **32 redundant source tensor objects totaling 805,063,106,528 bytes** and
retains thirteen metadata objects, including the export request and latest-step
marker. Full native restart state remains in the exact-byte East archive.
Publish this concrete result before running the separately guarded removal.
No source removal has occurred yet, and no unique-state-discard permission is
assumed. The original full-preservation cleanup rule authorizes this restorable
removal; the optional question about losing optimizer state remains separate.

Headroom now permits the ninth initial arm's version-4 retry. After recording
the completed archive and any covered-source cleanup, defer the other seven
large copies while that measured GPU run executes. The published retry plan,
unchanged runtime trees and same-recipe gates remain in force.
