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
