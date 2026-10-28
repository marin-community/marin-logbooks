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
