---
title: "Manual inference-perfmax workflow and UX"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/57"
related_issues: []
research_branch: "codex/57-inference-perfmax"
marin_base_revision: "5a439b9c00f56ad6d20f1ef62bffb3d867b58ece"
archive_template_revision: "ba8edcdfd730b451e18aed6245234566be5053af"
status: active
---

# Manual inference-perfmax workflow and UX

Test whether one serving preset and explicit overrides can run, compare and
recover AgentX trials through Iris. The skill chooses experiments. The runner
executes one trial and preserves its evidence. Actions and dashboards are out
of scope. Use at most eight H100s and sixteen GB200s concurrently, at interactive
priority. This task requires no tuning sweep, new numerical investigation,
infrastructure deployment or fork adoption. PRs stay draft and unmerged.

## Starting points

Marin base is `5a439b9c00f56ad6d20f1ef62bffb3d867b58ece`.
Runner [#10645](https://github.com/marin-community/marin/pull/10645), source
`d669e9e4d9d1417b661671bc73e1baa4bbb9f2a2`, supplies preparation, cache,
process cleanup and native serving helpers. Its fixed grids, Actions and importer
are outside this path. Skill [#11320](https://github.com/marin-community/marin/pull/11320),
earlier source `3f8f5da43dad693cab618d33abaf70c17364ffee`, supplies the campaign
method. Its overlapping human guides are consolidated into the skill and README.

## 2026-11-17 — Implementation decisions and negative findings

Resolve model, tokenizer, workload and runtime identities before GPU submission.
Keep one model/device preset per YAML and one shared AgentX recipe. A new trial
ID preserves each invocation; native Iris IDs preserve retries. Recovery cannot
submit GPUs. Normal runs retain upstream warmup, replay and validity.

Keep native AgentX validity, numerical correctness and recommendation acceptance
separate. Romain approved labeled workflow diagnostics with precise qualification
gaps. A pass for another checkpoint/runtime remains regression evidence; pending
or failed required checks block a recommendation. Snowball v0.1.1 does not inherit
the step-42150 gate's scoped 0.125 bound. The shared 0.075 bound is unchanged.
Hero also lacks matching shared numerical evidence. Add no numerical evaluation.

Iris's Kubernetes RUNNING timestamp omits earlier setup. Capture PodScheduled
through container finish instead. Retain all native attempts and costs. When a
pod expires, its saved scheduling time and Iris's terminal observation give a
conservative end time. Missing timestamps stay unknown. A live read-only
kubectl probe caught a string-escaping error before allocation; the corrected
[projection](analysis/allocation-projection-probe.json) yields eight fields.

A repeated real upstream fixture caught a dependency leak that unit tests in
Marin could not detect: profile validation imported Marin into AgentX's isolated
process. Split Marin's serving configuration from the pure harness commands.
The corrected fixture passes in 17.64 seconds under a 2-GiB process memory cap.
It prepares/restores native data and replays a tiny local Weka trace over HTTP.
Unmodified InferenceX normalizes the same exports to
[eight](analysis/cpu-fixture-h100.json) and [sixteen](analysis/cpu-fixture-gb200.json)
devices. Those are normalization inputs; GPU allocation is zero. The summaries
retain hashes of the original exports and upstream aggregates.

Forty-eight safe local cases and seven real released-router cases passed.
Six trial/report cases cover scoped check acceptance, frozen runtime selection,
separate diagnostic comparison groups and failed-attempt recovery. Synthetic
costs in those tests are fixtures. The affected selector chooses 77 Marin files;
with about 3 GiB free and full swap on devbox, broad CPU validation uses CI.

A fresh contribution-scope reader found no historical notebooks, captured
results, abandoned sweeps, scheduling or dashboard code in the proposed Marin
diff. It identified one unnecessary constraint: a mandatory node receipt checked
hosts only after allocation. Make it optional. Iris selects placement; explicit
allowed-host checks may still reject an admitted gang. Preserve Hero's existing
approved ShortConv compiler guard from the earlier runner handoff.

A prose reader requested concrete placement format, build references, job
identities and override precedence. The README now supplies them.
[Historical Snowball observations](analysis/snowball-historical-calibration.csv)
support starting near C128 but use an older runtime. They are not a current baseline.

## Live execution

GPU validation has not started. Record each published source revision, command,
Iris/native attempt identity, checks, measurements, allocation cost and cleanup
with the focused execution results.

Contribution: https://github.com/marin-community/marin/pull/11320

### Direct-controller launch rejection

Source `4846f4781a6083f58c45a480efb15799bd15b644` attempted Snowball
`20261117T015312Z-6ca63b0cc654` and Hero `20261117T015313Z-f85a8d257772`.
Both controllers rejected the root launch with `INVALID_ARGUMENT`: a local
`cluster` constraint was interpreted as a federation-peer name. No root job
or GPU child was admitted, so allocated GPU cost was zero. Omit the federation
constraint when entry and target controller are the same; retain it for a
cross-controller launch. Retain the failed local plans and logs as diagnostics.

### Live diagnostics: native execution and delivery failures

Published source `5a3b94f821` admitted Snowball
`20261117T015550Z-e7ef55077e67` and Hero `20261117T015550Z-b09439ebe1fc`.
CPU preparation finished before GPU admission. Snowball loaded the pinned wheel,
completed native warmup and the 90-second replay, and exported nine requests
with no errors. Hero's four-host gang stopped during startup: the pinned task
image lacks `numactl`, required by its copied `--numa-bind` flag. Remove that
optional binding flag for this image; keep the approved ShortConv compiler
guard. This changes a placement optimization, not model numerics.

Both coordinator collections rejected their own worker receipts because launch
provenance held `command_line` as a tuple before JSON and as a list afterward.
Normalize provenance at planning. A boundary regression test now checks complete
plan equality after JSON delivery. Manual recovery reads the retained JSON and
can collect these original attempts without another GPU allocation. Hero's first
failed gang costs 0.485 GB200-hours; all four native attempts and archives are
retained. A recovered report must show job state separately from pending checks.

Snowball recovery then found that upstream omits `metric_duration_coverage`
for a short unsafe replay. Treat missing coverage as incomplete instead of
crashing. Full runs still require complete coverage; diagnostics cannot pass.
The real isolated fixture now exercises this exported format and passes in
19.42 seconds. Snowball's recovered report records 0.996 H100-hours, nine
successful profiling requests, pending model numerics and no recommendation.

### Fresh Snowball completion

A clean published source `1ba1bb209d` ran trial
`20261117T021916Z-32b1ee916242` with the same C8 and 90-second diagnostic
arguments. CPU preparation, eight-H100 serving, native AgentX, collection,
reporting and release completed normally; the attached CLI exited zero.
The serving pod reached `Succeeded`, with its final container finishing at
02:30:20 UTC. This trial costs 0.842 H100-hours; the two Snowball trials total
1.838 H100-hours. See [the normal completion record](analysis/snowball-normal-completion.json)
for exact source, commands, native identities and allocation timing. The first
trial remains a recovered diagnostic. Neither is a comparable AgentX result.

Hero's retry uses published source `67ec452698`. Its launch provenance marks
the worktree dirty because the earlier compact result summaries were generated
before submission and committed afterward in `1ba1bb209d`. The source and
configuration diff between those revisions is empty. Retain that dirty flag
and the exact Iris bundle identity; do not present the launch as a clean tree.
The fresh Snowball launch from `1ba1bb209d` is clean.

### Fresh Hero completion and cleanup

Hero trial `20261117T021522Z-21bf94f72725` completed CPU preparation,
four-host/sixteen-GB200 native serving, AgentX, collection and reporting. The
attached CLI exited zero. The same wheel and approved compiler guard loaded;
NUMA binding was off. The four GPU pods were gone at the final provider check.
See [the normal completion record](analysis/hero-normal-completion.json).

Hero costs 4.548 GB200-hours for the completed trial plus 0.485 for the earlier
startup failure: 5.033 GB200-hours total. Snowball totals 1.838 H100-hours.
These costs include setup, warmup, drain and failed attempts. No GPUs remain
allocated by this goal. Both fresh terminal tables match the retained summary
bytes exactly. A live recommendation request for the fresh Snowball trial
exits one, and neither campaign contains `recommended.yaml`. The unit cases
separately prove blocking for mismatched or failed checks under a valid synthetic
measurement. All real measurements remain explicit short diagnostics.

Current CPU CI is green on contribution source
`8f664fc5921bb12e409ba19744606bc1f2a8630d`. The 48-case targeted suite, seven
trial/identity/check cases, seven released-router cases and corrected real
AgentX/InferenceX fixture pass. No new numerical or RL investigation ran.
[UX findings and proposed RFC clarifications](ux.md) preserve the remaining gaps.
The final High GOAL review is pending; PRs remain draft and unmerged.

### Final High GOAL review

One High review batch ran serially through the native Claude, Codex and AGY
CLIs. Claude Opus 5.5 and Codex 6.1 Sol were confirmed in native records; both
used the requested models. Codex's recorded effort is xhigh. Claude's effort was
requested as xhigh but is absent from its result metadata. AGY completed the
request for Gemini 3.8 Flash High; its actual model and tool trace were not
exposed. Keep those fields unknown. No second panel was requested.

All readers identified partial-export recovery as a material gap. Claude and
AGY also found that an ambient storage prefix could override the target and a
serving-only check could replace numerical qualification. The contribution
fixes all three. Trials without results now remain visible with unknown cost.
Checksums and native identity mismatches remain hard errors. Numerical checks
must be present, passed and scoped to the exact checkpoint/runtime pair.
This enforces Romain's existing acceptance decision; no waiver was introduced.

Smaller fixes add runtime code links, explain whole-list overrides and reject
unknown CUDA toolchains before allocation. The experimental JSON loader keeps
its separate role from the promoted TOML updater; broad parser consolidation is
deferred. Fifty-one focused CPU cases pass after these changes, including
interrupted native archives, missing results, check removal/relabeling and wrong
scope. No further GPU run or numerical investigation is needed: serving, native
replay and the validated hardware commands are unchanged. The final contribution
source is `02fcd6469902c93c02c3806c7487d332c62dc463`; its CI is pending.

### Handoff

Final CI is terminal and green on `02fcd6469902c93c02c3806c7487d332c62dc463`.
All current PR comments and reviews were read; none require action. The final
code successfully reprocessed the saved real Snowball exports through recovery
without resubmitting GPUs. The refreshed terminal reports match their saved
summaries and include both confirmed pre-admission launch rejections at zero
GPU cost. Those records use the original INVALID_ARGUMENT responses and fresh
controller NOT_FOUND checks; no allocation was inferred from a missing file.

[Snowball summary](analysis/snowball-summary.md), [Hero summary](analysis/hero-summary.md),
[CI](analysis/ci-final.json), [review disposition](analysis/final-review.json)
and [submission evidence](analysis/submission-rejections.json) preserve the final
state. Both campaigns contain no recommended.yaml. Matching numerical checks
and a complete normal AgentX window remain qualification work for a separate
scope. The manual workflow implementation and UX pressure test are complete;
contribution and index PRs remain draft and unmerged. All owned resource-board
rows are closed. No post-green monitor was started.

## 2026-11-17 — Shared correctness checks

A new delegated goal supersedes the earlier diagnostics-only scope. Keep
Snowball v0.1.1 at `59852d6f07261722bbec1edb6ab148317156b58e`. Replace Hero
step144000 with finalized SFT step400407. Work until 17:00 UTC, without asking
questions. Provisional tolerance choices are delegated. The aggregate interactive
ceilings are 48 H100s and 128 GB200s, including children and retries.

The initial implementation is contribution source `8dbbc3a13885f8067969928534bc1b9876585473`.
The shared checker separates saved-reference agreement, engine batching, cache
state, and decode versus recomputation. Deployment and score generation stay
outside its HTTP functions. This is a working checkpoint; real HTTP tests and
GPU qualification are still required.

[Hero #11430](https://github.com/marin-community/marin/issues/11430) already has
matching Levanter reference scores for step400407, including independent
context-parallel layouts through 262K. Preserve its raw files beyond their TTL
and verify each token-history hash. Its reviewed wheel is still `0ac9dcb157ee`.
Use its graph-enabled PIECEWISE settings as the released starting point; the
faster full-graph overlay is experimental and is not silently adopted.

Snowball's supplied tokenizer reproduces all 64 representative prompt token
arrays exactly. Generate missing scores with the Snowball class and explicit
model config recorded by that checkpoint's identity SFT. Its public HF metadata
says `production`, while that record specifies the Snowball snapshot. Retain
this discrepancy and use the recorded training config explicitly.

### Matching references and first policy

Both selected checkpoints now have durable references. Snowball's eight-H100
job `/romain/inference-reference-snowball-20261117-a1` succeeded from published
source `ccd2610878156b747005408b0e5f9f05a4c4470e`. It scored 1,770 positions
across the exact 64 representative token arrays. BF16 weights and projection,
FP32 log-softmax, JAX 0.11.1. The saved observations passed manifest, case and
history-hash validation. Allocated cost was 14,296 GPU-seconds, or 3.971
H100-hours, through pod container finish. Loading dominates the 1,748-second
worker runtime; the eight scoring batches total about 33 seconds. Routine
checks reuse these scores and do not rerun this model load.

Hero reuses 15,855 positions across 37 histories. Its raw `ref-a4`, CP16 and
CP32 files are preserved under the durable golden root. The normalized suite
uses `ref-a4` through 65K and CP32 at 131K/262K. It omits the final 262K position
so an output token fits the serving context. Its reference source is
`d8a9890ceae7c1a0e6c343aaa818db2b327b7060` for the short run and
`95e705902ae3713a51aecbb69014dce1e9133666` for context-parallel runs. Every
retained row's original int32 token-history hash matches the frozen inputs.
Checkpoint metadata, export request and expert placement match step400407.

Initial bounds are explicitly provisional. Hero's short-reference limits
(maximum .65, p99 .125, p90 .04, fraction above .075 at most .035) leave a margin
around the five retained BF16 controls in #11430. Its long limits are .35, .09,
.04 and .02. Snowball starts at .25, .125, .05 and .05. Its old step42150 gate
provides nearby evidence; its pass and bound do not transfer. Preserve selected-
checkpoint baseline failures and use repeatability evidence before revising bounds.
Matched cache splits require exact agreement. Other path limits are visible in
the two checked-in policies and await validation on the selected deployments.
These bounds do not qualify RL trainer probability ratios.

Contribution source `66e734c616a98f9417c75fe9bf42a57d515b403d` wires the same
HTTP functions into pytest and the CLI. Perfmax retains the report and checks
its outcome. It reads reference artifacts on CPU, runs endpoint checks before
AgentX and resets the cache before the unchanged native warmup. Missing or
failed required checks block ordinary measurements and accepted recommendations;
explicit diagnostics may still run. The existing adoption gate remains intact.
Keep this integration in draft PR #11320 so the bindings and callers can be
reviewed together; the reference-generation evidence stays in this archive.

The focused CPU checks pass: 31 cases, including nested vLLM prompt-score
responses, cache/batch coverage failures, wrong reference identity, CLI report
agreement, scope and recommendation checks. The independent selected-position
oracle also passed. Remote dependency preflight and real endpoint comparisons
are next; neither model is qualified by this entry.

### Launch failure and recommendation blocking

The latest completion boundary permits clearly labeled workflow diagnostics
with a precise list of missing qualification evidence. Keep measurement validity,
numerical correctness and recommendation acceptance separate. Run applicable
existing checks and reuse matching evidence; start no new numerical investigation.
A different checkpoint's pass remains regression evidence.

The remote CPU preflight passed seven HTTP/CLI cases in the pinned task image
with no GPU allocation. Hero attempt
`/romain/knotmeter-20261117t092450z-884fd63135d9` failed before model loading:
the compilation-cache identity treated the literal chat template as a filename
and raised `ENAMETOOLONG`. A local reproduction confirms the error. Write the
template to a file before native serving. Retain the
[failed execution](analysis/hero-correctness-a1-execution.json) and its 0.210
GB200-hours; all four pods were absent at cleanup. Snowball's first endpoint
attempt remains queued at this checkpoint. Neither attempt supplies a pass.

Contribution source `9cd50ea15915929e05fdd53a19afc71de04e2341` fixes that launch
path, retains the full numerical policy hash and labels provisional decisions.
Thirty-eight focused CPU cases pass, including real HTTP and CLI reports fed
into recommendation selection. Missing references, cache failures, unmatched
checkpoint/runtime evidence, omitted required checks, invalid AgentX submission
validity and explicit diagnostics all block recommendations. Reports remain
available for review. These fixtures establish workflow behavior, not model
numerical qualification.

### Streamed weight identity and admission retry

Hero's second attempt loaded the selected export and exposed a checker bug.
RunAI sets `model` to a local metadata directory and preserves the original
weight URI in `model_weights`. The
[pinned runtime source](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/vllm/config/model.py#L131)
defines that contract. Verify the preserved source strictly and retain both
fields. The [report](analysis/hero-correctness-a2-report.json) failed deployment
identity before any numerical request. Its [execution](analysis/hero-correctness-a2-execution.json)
used 2.321 GB200-hours; all four pods were gone at cleanup. Reference, batch,
cache and decode comparisons remain pending for this deployment.

Contribution source `2ace0b2eeefd1ad90e74194f22e67ca00d132349` fixes that
normalization. Thirty-nine focused CPU cases pass, including valid streamed
weights and rejection of the wrong preserved weight source before scoring.
The mandatory advisory batch completed once; its
[resolutions](analysis/shared-checks-advisory-resolution.json) cover typed
comparison records, storage joins, test value and fixture documentation.

Snowball's first endpoint attempt never passed its scheduling gate. Cancel it
after 30 minutes and retry the existing suite with a 384-GB host request. Its
8-H100 allocation is now running. The numerical settings and policy are the
same. The [canceled receipt](analysis/snowball-correctness-a1-execution.json)
retains incomplete allocation timing; no allocation was observed before
cancellation. The local broad safe-test run recorded five failures before its
own 4-GiB cap ended it. Identify the first failure and use CI for full coverage.
Do not count that partial run as a pass.

### Required checks block both selected profiles

Snowball a2 and Hero a3 passed deployment identity, genuine four-request engine
batching and decode versus fresh recomputation under the provisional policies.
The batch maximum probability errors were .011999 and .011306, respectively.
The 64 emitted-token comparisons had maxima .041100 and .045121. These limited
passes do not qualify either profile.

Both reference stages stopped with `ServerDisconnectedError`. Snowball retained
13 completed reference responses; its last 23,113-token response took 15.96
seconds. Raw responses remain in each durable run archive. Neither reference
stage completed its required suite. Both cache stages stopped because a
127-token seed followed by 257 generated tokens yielded only a 112-token cache
hit. That hit did not reach generated tokens. The required coverage remains
missing; no numerical bounds are changed to bypass it.

The saved [Snowball report](analysis/snowball-correctness-a2-report.json) and
[Hero report](analysis/hero-correctness-a3-report.json) preserve those failures.
Their allocated costs were .831 H100-hours and 2.809 GB200-hours. Snowball's pod
was Failed with every container terminated; Hero's four pods were absent.

Contribution source `7db04440dcd9873c517d034a585d67e69e1e8f8c` opens fresh HTTP
connections for checks after large prompt-score processing. This tests the
expired-keepalive hypothesis; the cause is not yet proved. Rerun each existing
suite once, with the same numerical settings and policies, to validate the
transport fix. No cache or numerical investigation follows this retry.

The report also reads the current leader's recorded check outcome when no
AgentX window exists. A failed precheck must remain visible instead of being
shown as pending. The 39 focused CPU cases pass, including recovery through
retained archives with an earlier pass and a later failure.

The broad CPU run stopped after 1,074 passes at two unchanged task-curation
fixtures. Their hard-coded paths lack `python3` and `head` on this Nix host;
[narrow rechecks and source identities](analysis/affected-host-limitations.json)
confirm the local limitation. The [partial summary](analysis/affected-cpu-summary.json)
is not a full-suite pass. Use contribution CI for broad completion.

### Comparator correction and diagnostic handoff

The High-tier final review inspected source
`7db04440dcd9873c517d034a585d67e69e1e8f8c`. It found that a returned observed
prompt token below the top 50 could lower the missing-token cutoff and
understate the probability-error bound. Contribution source
`2cd493de0bcedf598a61bd2974b44bd65f15857a` restricts distribution comparisons to
the true top 50 while retaining the observed token for decode rescoring. An
independent normalized 80-token example reproduces the old under-bound.
Forty-two focused CPU cases pass, including the corrected bound, partial
reference retention, cache coverage failures and HTTP/CLI recommendation
blocking. Required lint and type checks pass. CI on that exact source is
terminal: 17 successful checks and 20 skipped. The
[resolutions](analysis/goal-review-resolution.json) record targeted fixes.

Hero a4 completed all 37 reference histories before the comparator correction.
CPU replay of its saved observations now passes all 15,855 positions under the
unchanged provisional limits. Short/long maxima are 0.470978/0.278653; p99 values
are 0.093855/0.054485. Snowball a2 yielded 13 of 64 complete histories. Replaying
its 194 observed positions gives maximum 0.078634 and p99 0.064165, within the
provisional limits. Its remaining 51 histories and 1,576 positions stay pending.
The [replay script](analysis/recompute_reference.py) is published at
`50c2cc5fcc70f56ac6f294a4a5dc62601b4ae098`. It verifies matching artifact,
checkpoint, runtime and policy hashes. Read-back hashes match both uploaded
reports. Original reports and observations remain intact; overall cache
failures still block both profiles.

Both final live cache probes reported 112 cached tokens after a 127-token seed
and 257 generated tokens. That covers no generated tokens. The old checker
aborted before eight remaining boundaries and the final matched-repeat control.
Source `2cd493de0b` keeps the failed decode-reuse subcheck while collecting the
remaining prefill groups. Real HTTP fixtures verify this behavior; the new full
suite has not run on either GPU profile. Prefix retention interval zero may
explain the missing decode reuse, but the cause is uninvestigated. No setting,
limit or required check was changed to force a pass.

Snowball's final transport retry remained unadmitted and was canceled after
more than half an hour. All owned GPU jobs are terminal and resource-board rows
are closed. This goal recorded 4.802 H100-hours and 9.338 GB200-hours, including
reference generation, setup and failed attempts. Two unadmitted Snowball
receipts have incomplete allocation timing and no observed allocation. The
[cost ledger](analysis/correctness-costs.json) retains that uncertainty.

The agreed completion boundary permits labeled workflow diagnostics and precise
qualification gaps, with no new numerical investigations. Neither preset is
qualified. No normal AgentX window, accepted recommendation or tuning child was
produced. The integrated perfmax GPU path has component evidence but no new
end-to-end GPU run. The merged step42150 gate at the same reviewed runtime is
regression evidence only. RL trainer probability ratios and learning quality
remain unqualified. The [handoff](handoff.md) records commands, identities,
provisional limits, measured differences and the remaining evidence.


## 2026-11-17 — Complete validation and tuning

The new goal replaces diagnostic-only completion and the overnight deadline.
It requires complete shared-check evidence on both models, supported cache
reuse, a valid one-hour AgentX window and an inspected fresh-context campaign.
The aggregate ceilings are 48 H100s and 128 GB200s at interactive priority.
The initial parent allocation is eight H100s plus sixteen GB200s. One tuning
child may use eight additional H100s.

Contribution source `12055994a685e6fdff27c6ece463f4998c247e3e` keeps the reviewed
vLLM runtime and fixed Levanter scores. Both presets now retain periodic cache
checkpoints every 128 tokens. The pinned
[cache configuration](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/vllm/config/cache.py#L148)
and [retention masks](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/vllm/v1/core/single_type_kv_cache_manager.py#L1576)
retain selected replay/shared-prefix states at interval zero and periodic
states at positive intervals. The old probe required an arbitrary generated
history to be reusable, which zero does not promise. The prior 112-token hits
show missing coverage; they do not establish corrupt state. Test the supported
128-token setting without removing decode-created reuse or changing bounds.

The checker now retains each failed reference/cache case and runs later cases.
Missing prefill hits and unmatched-repeat coverage are explicit failed
requirements. They cannot supply numerical cache comparisons. Forty-three
focused CPU/HTTP/CLI cases pass, including an interrupted cache boundary followed
by successful later boundaries and an interrupted reference followed by later
histories. Required lint and type checks pass. Snowball uses the previously
admitted 384-GB host request. Neither new profile is qualified before live tests.

The pytest launchers will run the complete shared suite using the checked-in
profiles and saved goldens. Their execution receipts record exact source,
settings, run URI and all native attempts. No reference regeneration, fork
refresh or RL trainer-alignment investigation is part of this phase.


### Live validation and campaign dispatch

Both pytest validation gangs are running from contribution source
`12055994a685e6fdff27c6ece463f4998c247e3e`. Snowball's API is responding; Hero
has reached compilation. The [launch record](analysis/complete-validation-start.json)
contains the exact commands, live job IDs, pinned runtime, plans and receipt
locations. A running job supplies no qualification pass.

A fresh-context child, native session `01a1eabe-cfa1-7f52-bd95-a69658df86e3`,
has its own isolated research worktree and an eight-H100 share. Its dispatch
supplies the skill, profile, objective and scope. It must choose an alternative,
establish a baseline and produce at least one valid normal one-hour AgentX
window. The parent will inspect its native commands and retained artifacts.


### Full Snowball pass and Hero repeat-control failure

Snowball's [live report](analysis/snowball-retention128-a1-summary.json) passes
all five stages under the unchanged provisional policy. It covers 64/64
histories and 1,770/1,770 reference positions. Maximum reference error is
0.196002, p99 0.088755 and p90 0.030446. All nine prefill and decode-created
cache boundaries were exercised. Boundary 127 reused 368 tokens after decoding,
compared with the earlier 112-token miss. Changed-split maximum error is
0.080051; the matched-repeat control is exact. The job succeeded, its provider
pod is terminal with every container completed, and board row 2201 is closed.
Its [receipt](analysis/snowball-retention128-a1-execution.json) records 1.255556
H100-hours.

Hero's [live report](analysis/hero-retention128-a1-summary.json) completes all
37 histories and 15,855 reference positions within policy. Deployment, batch,
decode, all nine prefill/decode-reuse boundaries and the matched-repeat coverage
pass. Its overall result remains failed: equal cached counts at boundaries
127/255/2047 yielded probability differences 0.000004765, 0.000189799 and
0.002857573 against a zero bound. The final matched-repeat comparison is exact.
Changed-split maximum error is 0.032986. Its [receipt](analysis/hero-retention128-a1-execution.json)
records 4.773333 GB200-hours. All four provider pods are absent and board row
2202 is closed. No limit changed to accept these observations.

[CPU replay of the retained interval-zero observations](analysis/hero-retained-equal-split-replay.json)
also finds nonzero warm-versus-repeat errors at boundary 127: 0.000019060 and
0.000012151 in two earlier runs. The new interval is therefore not necessary
for this symptom. This does not establish its cause or justify a pass.

The next focused Hero experiment compares three fresh recomputations, immediate
cached repeats, repeats after an alternate branch and changes in the seed's
shape at the three failing boundaries. If uncached repeats vary, they measure a
numerical control independent of cache reuse. If changes occur only after a
branch at equal hit counts, investigate cached-state or checkpoint replacement
before changing policy. Preserve both patterns and their exact cached counts.
The experiment remains diagnostic and cannot qualify the profile.

The historical [control test](../../tests/cluster/vllm/test_repeat_controls.py)
uses the existing deployment fixture and shared HTTP client. Its test selector
is an archive-only launcher option; the contribution's default gate is unchanged.
A real local HTTP smoke completed all 27 control comparisons in 2.02 seconds.
Run the focused GPU control from its published source:

```sh
uv run python -m experiments.evaluation.inference_correctness.launch \
  --profile hero-gb200 --name hero-repeat-controls-a1 \
  --pytest-test tests/cluster/vllm/test_repeat_controls.py \
  --output hero-repeat-controls-execution.json
```

Contribution CI at `12055994a6` is terminal: 17 successful checks and 20 skipped.
Interactive priority remains requested. Iris's effective Batch fallback is
allowed by current task steering; record it and leave shared scheduling alone.
The child's normal Snowball baseline has passed its actual endpoint CLI checks
and entered native warmup using a verified prepared-workload cache hit. No
performance result or recommendation is claimed yet.


### Equal hit counts can conceal a changed prefill split

The [focused repeat report](analysis/hero-repeat-controls-a1-summary.json) shows
exact fresh recomputations at all three failing boundaries. Initial equal-count
cached repeats differ by up to 0.001137. Hit counts then advance from
112/240/2032 to 128/256/2048. Once prepared, repeats remain exact, including
after another branch. The [execution](analysis/hero-repeat-controls-a1-execution.json)
cost 2.596667 GB200-hours. All four provider pods are absent.

The checker assumed equal hit counts meant equal prefill splitting. The pinned
scheduler can invalidate that assumption. With 112 cached tokens and a 144-token
prompt, its first suffix can run as 32 tokens. After dense attention discovers
coverage at 128, the same 112-token joint hit can run as 16 plus 16. The new
junction creates a Mamba128 checkpoint. This predicts the observed hit sequence
112,112,128; it also predicts 240,240,256 and 2032,2032,2048. The
[CPU reproduction](analysis/hero-scheduler-split-reproduction.json) executes the
actual pinned scheduler method and confirms these chunk choices. It does not
observe live chunks or GPU state bytes.

A read-only scheduler hook will confirm actual starts, requested/returned chunks
and junctions on the dedicated Hero control deployment. The hook checks the
original method's AST hash and returns its result unchanged. Fresh-process and
spawned-child CPU smoke tests pass. The AST fingerprint uses the serving
Python3.12 environment. The hook is private to this archived diagnostic through
its explicit environment flag; no wheel, shared service or release changes.

```sh
uv run python -m experiments.evaluation.inference_correctness.launch \
  --profile hero-gb200 --name hero-scheduler-trace-a1 \
  --pytest-test tests/cluster/vllm/test_repeat_controls.py \
  --set 'serving.environment.HERO_SCHEDULER_TRACE="1"' \
  --output hero-scheduler-trace-execution.json
```

Keep the initial seed/branch comparisons and all coverage requirements. Correct
the unsupported classification after the trace; reserve exact comparisons for
fully prepared branch prefixes with the same suffix plan. No numerical limit
changes. The original failed report remains available.

### Stronger branch control and first normal AgentX result

The optional scheduler trace failed during startup. Its retained host archive
contains hook startup records and peer-disconnect errors, but no scheduler
split records. The original cause remains unresolved. The
[failure observation](analysis/hero-scheduler-trace-a1-observation.json),
[receipt](analysis/hero-scheduler-trace-a1-host3-receipt.json) and
[cleanup](analysis/hero-scheduler-trace-a1-cleanup.json) preserve this attempt
and its 1.214646 GB200-hours. No trace retry is planned.

The pinned source counterexample establishes that equal hit counts alone are
insufficient for the old exact comparison. It does not prove the live cause.
[Contribution 64bfff4bc5](https://github.com/marin-community/marin/commit/64bfff4bc5612a4bf34a17719d7e7c7a14f7595e)
keeps every original comparison and bound. It adds required exact branch
controls at all nine boundaries after the full usable prefix is cached. A
sibling differs inside that cached region; the original branch must still hit
the full prefix and return identical scores. Preparation stops on coverage,
never on a numerical pass. Initial seed/branch pairs use the existing changed
split bound.

All [44 focused CPU cases](analysis/complete-validation-cpu-44.txt) and required
lint pass. A disposable [HTTP regression](analysis/branch-checker-regression.txt)
shows that the old checker accepts an injected branch-state alias under a
permissive synthetic split bound, while the new exact branch check rejects it.
The [replay source](analysis/branch_checker_regression.py) uses the checked-in
HTTP engine. Production numerical policies and saved goldens are unchanged.
Hero's ordinary full pytest run now uses this published contribution without
the private hook. Snowball's C64 tuning point will exercise the same updated
checks through the endpoint CLI.

The fresh-context agent completed its C128 baseline. Root independently read
the native exports, commands, check report and allocation, verified archive and
report hashes, and recomputed phase and token totals. The
[audit](analysis/snowball-baseline-root-audit.json) confirms a valid normal-hour
window and 11.368889 H100-hours including startup and warmup. Its 124.39607
tokens/s/GPU counts completion tokens reported for successful profiling
requests. The p90 TTFT is 103.33113 seconds; p90 per-request completion is
297.61639 seconds. These are not session-tree or RL-step measurements.

Keep the phase errors visible: 989 warmup requests had content and 417 had
empty-content validation errors; profiling had 5,165 successes and one
disconnect. Native submission validity and both duration-coverage checks pass.
The agent chose C64 next to test whether lower cache pressure improves this
throughput/latency tradeoff. Prepared workload reuse avoids repeated data work.

### Hero completes the corrected shared suite

Hero's ordinary [full pytest report](analysis/hero-qualified-a2-summary.json)
passes all five groups on published contribution `64bfff4bc5`. It completes
37 saved histories and 15,855 positions. Reference maxima are 0.560200 for
short histories and 0.256977 for long histories, within the unchanged
provisional policy. Batch overlap reaches four requests on engine 0; all four
batch and 64 decode comparisons pass.

All nine prefill, fully prepared branch and decode-created reuse boundaries
pass. The nine branch comparisons and final repeat have exactly zero error.
Changed prefill comparisons retain a maximum error of 0.032986. These results
resolve the required cache coverage and exact branch checks for this deployment;
they do not establish the cause of the earlier equal-count differences.

The [execution](analysis/hero-qualified-a2-execution.json) costs 4.992222
GB200-hours. Root verified the retained report and native archive hashes and
[all four provider pods](analysis/hero-qualified-a2-cleanup.json) have terminated
containers. The resource-board entry is closed. Snowball's C64 candidate is
still queued; it will run the corrected CLI checks before its normal window.

### Both corrected suites pass; C64 continues after preemption

The first C64 allocation passed every updated shared check on source
`06b9a63682`: 64 histories, 1,770 reference positions, all nine prefill,
prepared-branch and decode-created reuse boundaries, and ten exact controls
with zero error. Root independently verified the report and native archive
hashes in the [audit](analysis/snowball-c64-attempt0-root-audit.json). The
[submitted-plan audit](analysis/snowball-c64-plan-root-audit.json) confirms
that client concurrency, 128 to 64, is the only profile difference from baseline.

Kueue preempted this allocation during native warmup. No profiling window or
final export completed. The native archive preserves the original failed
worker record. A [derived receipt](analysis/snowball-c64-attempt0-derived-receipt.json)
and separate [recovery record](analysis/snowball-c64-attempt0-receipt-recovery.json)
state that the original receipt was missing and identify the verified inputs.
This recovers evidence; it does not turn the interrupted attempt into a pass
for measurement or recommendation. Iris queued its configured retry. No manual
restart or scheduling change was made.

The reissued goal also asks for the evidence and limits behind each numerical
policy, using retained data without a new calibration campaign. The
[policy explanation](numerical-policies.md) covers each comparison and the
[observed statistics](analysis/numerical-policy-observations.json). Hero's
reference margins have prior BF16 control evidence, with a different top-k
comparison and launch scope. Snowball's reference cutoffs and the nonzero cache,
batch and decode cutoffs remain initial screens. Passing checking code and
passing those provisional limits are separate claims. RL trainer ratios
remain unqualified.

### Keep the qualification scope after selecting a point

The report previously showed the RL qualification limit only when no point was
selected. Contribution `e929e4528e` keeps it visible after selection and in the
saved YAML comments. This is a reporting change: acceptance rules, numerical
limits and measurement code are unchanged. The
[20 affected report cases](analysis/report-scope-tests.txt) pass, including
saved-profile parsing and retained runtime selection; required lint and types
pass. A local render using the completed baseline confirmed the selected path's
footer without writing a campaign decision. C64 keeps its submitted source
`06b9a63682`. After the tuning agent finishes, the coordinating agent will
refresh the final report wording without another GPU allocation or a changed
point selection.

### Completed fresh-context comparison and report refresh

Both normal windows pass protocol validity, duration coverage, unchanged-host
checks and the required numerical screen. C64 measures 92.30190 reported
completion tokens/s/GPU, versus C128's 124.39607. Its p90 TTFT falls from
103.33113 to 60.39601 seconds and per-request completion from 297.61639 to
154.10088 seconds. The tuning agent retains C128 for throughput. Its 5%
similarity criterion was a predeclared heuristic, not Romain's numerical target.
This is a useful negative result for throughput; two points do not locate a
global optimum or settle interactivity preferences.

The coordinating agent independently verified the
[C64 native archive and report](analysis/snowball-c64-root-audit.json), phase
counts, token totals and all shared-check coverage. The final attempt has
3,392 successful profiling requests, zero profiling errors, 510 nonempty
warmup responses, 192 empty-content warmup errors and 25 deadline credit
cancellations. The interrupted attempt remains separately retained and
performance-ineligible. The [cost ledger](analysis/complete-validation-costs.json)
includes it; the two-point campaign costs 23.964314–24.107522 H100-hours under
the retained end-time bounds.

The final report was refreshed from contribution `e929e4528e`, preserving the
agent's C128 selection. The [audit](analysis/report-refresh-audit.json) verifies
that terminal and stored summary bytes match and that `recommended.yaml`
reloads with the tested profile and exact runtime. Both outputs state the RL
qualification limit. Independent provider checks find both GPU attempts and
the CPU coordinator terminal. The board is closed. CI on `e929e4528e` is
terminal: 17 successful checks, 20 skipped. The final High GOAL review remains
pending.

### Final High review: correct the metric scope and retain the limits

One final High GOAL batch completed with Claude, Codex and AGY. The
[resolution](analysis/complete-validation-goal-review-resolution.json) records
requested and observed models, native identities, action inspection, findings
and deferred improvements. No second panel or GPU experiment was needed.

The review found a material error in the current metric descriptions. Reference
checks compare fixed top-64 scores with serving top-50 scores. Cache and batch
checks compare two serving top-50 lists. Decode checks measure only the emitted
token's absolute probability error, not a bound over all returned tokens. Its
raw log-probability error has no acceptance limit. The
[format verification](analysis/metric-scope-verification.json) confirms all saved
score rows contain 64 entries. Contribution `2471a015ac` and the current handoff
correct these descriptions. Earlier blanket descriptions must be read with this
correction. No score, bound, checking code or measured pass changed.

The review also exposed an important scope choice. Retention 128 enables the
required decode-created coverage and narrows eligible settings for these fixed
probes. Zero-to-128 throughput cost is unmeasured on both models; no Hero AgentX
window was run. This remains a focused setting choice authorized by the goal.
The exact cache control checks persistence after a sibling; it cannot detect a
wrong state that remains unchanged. Policy values stayed fixed, but equal-hit
initial branch comparisons moved to the changed-step limit. The injected fault
escapes the old checker only under the permissive synthetic limit; production
limits would reject it in either version.

The [retained live samples](analysis/campaign-cache-telemetry.json) show C64
reduced KV use to 34–54%, while actual prefix hits remained 24–28% against
about 96% theoretical reuse. Cache capacity is an incomplete explanation.
Routing across Snowball's eight data-parallel engines is an untested lead.
Two skill sentences now ask agents to compare those rates, KV use and routing.
Older-runtime results remain a lead rather than a controlled regression claim.
The exact [44-case command](analysis/complete-validation-cpu-command.json) was
recovered from this session's native record; its passing implementation inputs
remain unchanged. Final lint, types and skill validation pass.

Final [CI](analysis/ci-complete-validation-final.json) on `2471a015ac` is
terminal: 17 successful checks and 20 skipped. No implementation inputs changed
after the passing focused checks and live runs.

A separate full-archive lint attempt hit 60 missing-import diagnostics in the
archive's incomplete local environment and 15 older raw receipts without trailing
newlines. Their bytes were preserved. All nine changed notes and evidence files
pass the required entry point with explicit file arguments. Contribution lint,
types and final CI pass; the archive-wide attempt is not a contribution failure.

## Hero FP8 campaign and retention-aware coverage — 2026-11-18 UTC

The new goal pursues a real Hero BF16/online-FP8 comparison and useful targeted
fixes. Model, tokenizer, goldens and initial runtime stay pinned. Romain removed
the ownership prerequisite: LA1 is idle and its extraction is planned. DA5 owns
the correction and will leave a precise source/handoff without contacting LA1.

The [cache handoff](cache-retention-handoff.md) records source `4f38ff6b3d` and
the pinned runtime contract. Zero can qualify through supported prompt replay
and branches. Positive/dense probes target a reachable periodic endpoint.
Exact controls now require a sibling hit and also cover replayed generated
history. No goldens or numerical limits changed. All 56 combined focused
CPU/HTTP/report cases pass.

The [fresh-context Hero campaign](hero-fp8-01a1ec55.md) has a share of at most
32 GB200 GPUs. DA5 retains at most 16 GB200 and eight H100 for affected checks.
The child stopped its local preparation after memory pressure and moved that
step to the existing CPU coordinator before allocating GPUs. Its small report
fix compares equal numerical criteria across weight precision while preserving
full policy identity, applicability and provisional status. No new grading
framework or duplicate experiment was introduced.
