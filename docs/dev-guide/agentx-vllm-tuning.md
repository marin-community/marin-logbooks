# Inference performance campaigns

Use the [inference-perfmax skill][skill] to improve RL rollout and evaluation
serving. AgentX supplies a repeatable proxy workload; check application gains
on representative rollout or evaluation traffic before claiming them.

This guide defines the campaign method. Read the [technical reference](inference-perfmax-reference.md)
when choosing AgentX controls or correctness checks, and the
[record convention](inference-perfmax-records.md) before recording attempts.
Live performance validation of this workflow remains deferred.

## Establish the decision

Search prior Marin work through [use-echo][echo]. Inspect current primary runs,
source, and upstream docs, recipes, releases, PRs, issues and reviews. Check
model, GPU and topology support and whether a proposed feature is merged,
released and adopted in Marin. Discussion and other-model wins suggest
hypotheses; inspect the relevant code and evidence. The reference links
[existing upstream agent guidance](inference-perfmax-reference.md#upstream-starting-points).

Propose a compact kickoff covering the model/export/tokenizer pins, workload
and protocol, baseline, throughput and latency decision, allowed changes,
correctness checks, GPU budget and retention. Resolve consequential unknowns
together with the user. Search before asking for facts already available.
Choose numerical thresholds from the campaign's needs and evidence.

Identify the model team's designated reference serving run through its benchmark
issue, report or dashboard. Find the latest run passing the agreed measurement
and correctness criteria and the best comparable run; pin both when they differ. Treat their
latency as a starting reference. If a usable baseline is missing, explain the
gap and propose establishing one within budget. Hero/Mochi on B200/GB200 and
Snowball on H100 are starting examples, not current hardware or qualification
claims.

## Test explanations and spend proportionally

Write the observation, suspected cause, proposed change, predicted outcome and
rejection condition. Separate client/router limits, CPU gaps, KV-cache pressure,
compute, memory traffic and communication. Correlate request, queue, cache and
device trends across workers; account for counter resets with
[query-finelog][finelog]. High GPU utilization alone cannot identify the cause.

When metrics leave competing explanations, capture a representative trace.
Inspect prefill, decode, cache reuse and relevant ranks. Follow the execution
path that determines completion: summed kernel times can count overlap twice.
Use a controlled change or ablation to test causality. Measure throughput again
without profiler overhead.

Choose experiments by plausible end-to-end gain, supporting evidence and total
GPU cost. Label estimates; a small or overlapped kernel limits possible gains.
Reuse relevant checks and stop attempts that no longer justify their cost,
within agreed authority. Choose parallelism for GPU efficiency even when it
extends wall time.

Use this ladder when it helps; skip stages already supported by evidence and
change the order when justified:

| Evidence needed | Possible experiment | Reason to spend more |
| --- | --- | --- |
| A testable cause | Inspect prior results and metrics | Plausible gain remains |
| An effect on the suspected path | Correctness check, trace or microbenchmark | Prediction survives |
| A useful operating region | Targeted calibration and repeats | Gain exceeds observed variation |
| A supported recommendation | Repeated baseline/candidate target-protocol runs | Tradeoffs and qualification hold |

Count allocated GPU-seconds across every attempt, including startup,
compilation, warmup, draining and failed launches. Before a costly expansion,
estimate its cost and the result that would justify it. Routine choices stay
autonomous after kickoff. Consult for material scope or cost changes, changed
correctness criteria, or consequential designs that are hard to reverse.

## Compare and hand off

Keep the selected model and measurement controls fixed within a comparison,
except the declared change. Separate diagnostic, calibration and confirmation
evidence. A short calibration cannot establish a full-hour result. AgentX
reports metrics and curves. This optional ladder is a campaign strategy;
AgentX does not assign it a score.

Compare completed output tokens/s at fixed GPU allocation and output tokens/s
per allocated GPU across allocations. Show repeat variation, latency and tails,
errors and requests canceled at the profiling cutoff alongside throughput.
Recommend a measured point in the high-throughput region before small gains incur a large latency
cost. Prefer lower latency among similar-throughput choices; explain the
tradeoff against the baseline and agreed criteria.

Report benchmark validity, numerical/state correctness, task quality and RL
qualification separately. Matching baseline output or small average log-probability
error alone does not establish RL safety. Use the reference's
[RL checks](inference-perfmax-reference.md#correctness-and-rl-qualification)
for the actual trainer and sampling convention. Preserve existing failures and
bounds; changed acceptance criteria require user agreement.

Use [research engineering](research-engineering.md) to preserve runnable code,
analysis and notebook on a Marin-derived research branch without merging to
main. The record convention connects that archive to regional CoreWeave
artifacts, including failed hypotheses and repeated measurements.

Deliver the serving recipe with exact commands, pins and hardware; measured
comparisons and latency tradeoffs; total cost; qualification; artifact links;
and remaining uncertainty. State which commands are on main, which helpers are
unmerged, and which examples or capabilities need later work. Stop when the
evidence supports the decision, the budget is spent, or further work requires
consultation.

[skill]: https://github.com/marin-community/marin/blob/main/.agents/skills/inference-perfmax/SKILL.md
[echo]: https://github.com/marin-community/marin/blob/main/.agents/skills/use-echo/SKILL.md
[finelog]: https://github.com/marin-community/marin/blob/main/.agents/skills/query-finelog/SKILL.md
