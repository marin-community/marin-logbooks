---
title: "GLM tool-conversion production fleet for retained SFT shards"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin/pull/11496"
related_issues: ["https://github.com/marin-community/marin/pull/11496"]
research_branch: "codex/11496-glm-tool-conversion-fleet"
marin_base_revision: "7ca4909d20dfb7a003909922f6cf992852f117bf"
archive_template_revision: "fc1cb30da0967f6ab78ffd47444ed9144f50cba9"
status: active
---

# GLM tool-conversion production fleet for retained SFT shards

This branch preserves the source tree of marin#11496 as a durable experiment branch. The PR is closed on main; the fleet stays a research and production pipeline, not a library change.

## Question and decision criteria

- Goal and why it matters: convert faux tool calls in retained SFT shards into real harness tools, so SFT mixes teach the tool format the policy is served with.
- Baseline and alternatives: unconverted shards; deterministic rule conversion only; rule conversion plus GLM-5.3 conversion for the residual rows.
- Evidence needed to choose: per-row adapter verification of converted rows, review and audit holds counted separately, and a semantic QA sample on the serving endpoint in use.
- Constraints and unresolved questions: the converter runs on shared GLM serving capacity, so throughput depends on that pool.

## Baseline

- Date: 2026-11-24
- Published source revision: 7ca4909d20dfb7a003909922f6cf992852f117bf
- Configuration, command, and external inputs: the fleet entry points in this tree (ortet and scale drivers, one batch loop with durable claims, archives and cursors); GLM-5.3 through the shared serving pool manifest.
- Environment and hardware when relevant: driver pods on CoreWeave cw-us-east-08a; GLM serving pool of 32 TP4 replicas on 128 GB200.
- Result and evidence: 19,037,413 of 20,749,152 catalog rows processed (91.78%) before the fleet moved to the new serving pool; about 544k rows verified by the indexer.
- Limitations: 340,798 rows deferred by an earlier serving outage wait for replay on the new pool.

## Entries

### 2026-11-24 14:52 UTC - Serving-pool QA gate and fleet restart

- Question or hypothesis and reason for trying it: after the previous relay lost its upstream credential, does conversion on the new GLM serving pool produce rows that pass verification?
- Published source revision: 7ca4909d20dfb7a003909922f6cf992852f117bf
- Command or material configuration: replay of 289 deferred rows through the full pipeline against one qualified pool endpoint, then the offline adapter verifier over the converted sample.
- External inputs and artifact references: s3://marin-us-east-02a/marin/users/benfeuer/tool-normalization/tool-conversion-index/qa/qa-gate-replay.json
- Environment and hardware when relevant: cw-us-east-08a; pool qualified at 200/200 real requests and 38,139/38,139 sustained requests.
- Result, including failure or restart: 248 of 289 rows converted (85.8%), 41 held for review or audit, zero infrastructure deferrals; the offline verifier passed 230 of 230 sampled converted rows. All 32 drivers relaunched on the pool at 14:02 UTC.
- Evidence: the QA receipt above; the pool manifest at s3://marin-us-east-08a/marin/users/benfeuer/glm-synthesis-serving/v3/full-32-r3/manifest.json
