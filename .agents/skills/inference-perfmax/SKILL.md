---
name: inference-perfmax
description: Improve inference throughput and GPU efficiency for RL rollouts and evaluations through evidence-led tuning campaigns measured with AgentX. Use to choose experiments, qualify serving changes, compare latency tradeoffs, and preserve reusable results.
---

# Improve inference for RL rollouts

Read the [campaign guide](../../../docs/dev-guide/agentx-vllm-tuning.md) at
kickoff. Initial examples are Hero/Mochi on B200/GB200 and Snowball on H100;
Hero and Mochi name the same model. Discover the actual checkpoint and hardware.

## Agree on the campaign

Resolve available context and propose defaults for the model, checkpoint pin,
workload, baseline, success criteria, allowed changes, correctness requirements
and resource budget. Present consequential uncertainties together for user
review. Propose the latest qualified reference run and the best comparable run
as baseline candidates.

Optimize completed output throughput and GPU efficiency for RL serving. Find a
high-throughput region with acceptable latency against the baseline; prefer
lower latency when throughput is similar. Agree on any numerical decision
criteria for this campaign.

After agreement, choose hypotheses, techniques, experiment order, parallelism
and stopping strategy autonomously within that authority. Consult again for a
material scope or cost change, changed correctness criteria, or a consequential
design that is hard to reverse.

## Run an economical search

Reuse prior findings and upstream guidance. Identify a plausible bottleneck,
predict the effect of a change, and seek evidence that could reject it. Choose
the cheapest experiment that could change the decision. The guide's compute
ladder is optional; it does not prescribe a grid or exhaustive checklist.
Prioritize GPU efficiency over elapsed time. Count startup, compilation,
warmup and failed allocations in cost.

Read the [measurement and correctness reference](../../../docs/dev-guide/inference-perfmax-reference.md)
when selecting a protocol or qualification check. Pin comparison controls and
reuse upstream AgentX definitions. Keep diagnosis, calibration and confirmation
separate. Report benchmark validity, numerical/state correctness, task quality
and RL qualification separately; preserve known failures and numerical bounds.

## Preserve the result

Before the first attempt, use the [record convention](../../../docs/dev-guide/inference-perfmax-records.md)
with existing Marin artifacts and the research archive. Preserve upstream
exports, failures and repeats under their actual attempt identities.

Deliver a reproducible serving recipe, measured baseline comparisons and
latency tradeoffs, total GPU cost, evidence links and remaining uncertainty.
Label inference-only results and deferred RL checks. Distinguish commands on
main, unmerged helpers, illustrative examples and capabilities requiring later
work. Follow existing publication, numerical and resource policies.
