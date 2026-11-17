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
