---
title: BiomniBench-DA public-50 ID evaluation proposal
source_repository: marin-community/marin
task_issue: https://github.com/marin-community/marin-logbooks/issues/45
related_issues:
  - https://github.com/marin-community/marin/issues/11385
research_branch: codex/45-biomnibench-da-id
marin_base_revision: 246f795749166285a864e891e56facfbcf9127e7
archive_template_revision: 603234653e8490b7e4f74195406f8c490c690789
status: active
---

# BiomniBench-DA public-50

## Question and decision criteria

Can the public 50-task subset supply a useful in-distribution (ID) development evaluation for Marin after five disclosed input modifications? The [protocol](protocol.md) records resource, harness, judge and scorer differences from the authors' implementation. Adoption requires feasible tasks, auditable grading, external-model baselines and evidence of value beyond BixBench. No Marin or Snowball model runs are authorized before the policy decision. Full baselines and repetitions need a separate budget agreement.

[Marin #11385](https://github.com/marin-community/marin/issues/11385) owns the protocol/adoption discussion. [Logbooks #45](https://github.com/marin-community/marin-logbooks/issues/45) owns this research record.

## Source and reproduction boundary

This is a retrospective archive. The control and three-task continuation were submitted from uncommitted files on Marin `246f795749166285a864e891e56facfbcf9127e7`. Their execution bundles were supplied to Iris. This later research commit must not be reported as the commit that executed those runs. [source-provenance.json](source-provenance.json) records original workspace file hashes before publication edits.

The public copy replaces private registry addresses with `registry.example.invalid`, removes VM-specific credential loading and parameterizes submission names and secret sources. The bundle inclusion paths and launcher workspace path now point to this logbook. Formatting is normalized. These publication edits have not been validated by another paid evaluation. Original files and unsanitized records remain in the operator's local evidence directory; restricted object-store locations and runtime access details are omitted here.

Small evidence files and protocol source are included. Input datasets, raw trajectories, serving logs, credentials and organization-wide snapshot inventories are excluded. The input manifest preserves public dataset URLs, pinned revisions, sizes and checksums. Published benchmark text is attributed to Phylo's CC-BY-4.0 dataset in [NOTICE.md](NOTICE.md).

## Baseline and protocol choices

- Source: `phylobio/BiomniBench-DA` revision `e1c8ca5e11a620087bc48d97888eb69176a1f235`; public 50 tasks, compared with 100 tasks and three repetitions in the paper.
- Harness: Marin's selected BixBench Pi configuration, Pi 0.87.0; Biomni's 3,600-second agent and 900-second verifier limits. The pilot uses two concurrent trials and zero trial retries.
- Candidate: Qwen3.6-35B-A3B, pinned revision in [pilot/qwen.yaml](pilot/qwen.yaml), four H100s. Judge: MiniMax-M3-MXFP8, pinned revision in [pilot/judge.yaml](pilot/judge.yaml), eight H100s. The judge's rubric request is 8,192 output tokens, temperature zero and a 120-second request timeout. BixBench supplies the serving configuration; Biomni supplies the rubric.
- Sandbox: preserve native CPU/GPU requests and cap RAM/disk at verified account limits of 8 GiB/10 GiB. The authors' 16 GiB/20 GiB requests were rejected before allocation.
- Five derived-input variants remain in the denominator: two CRC tasks omit the oversized expression matrix; three lupus tasks use the pinned raw-count derivative while retaining cells and annotations. The pilot covers one example of each plus a task whose source Docker COPY needs normalization.
- Signed rubric penalties are retained. The released authors' parser omits negative contributions. Report the signed score clamped to 0–100 and the unsigned comparison from the same criterion judgments; this is an explicit scorer deviation.

## Entries

### 2026-11-13 — Graded control

Question: does the standard launcher produce a durable, graded task with the chosen harness and judge? Task `da-4-6` used unchanged input bytes and the selected resource cap.

The control returned 28/100 signed and 38/100 unsigned, with both requested reports present and six criterion judgments. [The retained evaluation](pilot/control-completed-evaluation.json) records the decisions, including a −10 penalty. Judge claims about fabricated references have not been independently audited. This is one diagnostic task, not a benchmark score.

The durable run record and reward agreed, and EvalDash API ingestion and continuous-score metadata were verified. Dashboard visual rendering remains unchecked. The trial took approximately 4m46s, including about 4m17s of agent execution and seven seconds of grading. Total job elapsed time included serving startup and queues. GPUs have not been independently profiled for utilization.

The initial control image lacked the expected data; input-layer repair and remote checksum validation preceded the successful run. A local command-output collection probe also failed; remote validation succeeded. These preparation failures are distinct from solver outcomes.

### 2026-11-13 — Scoring failure controls

The [empty-submission control](pilot/empty-submission-control.json) returns zero without calling the judge. The [rubric replay](pilot/offline-replay.json) exercises signed penalties from the actual rubric.

An [isolated replay of the production reward reader](pilot/reward-normalization-replay.json) found that a present verifier result with a missing or nonnumeric reward can be counted as scored zero; a nonfinite reward can remain NaN. This is not an end-to-end collection-failure test. Resolving scoring coverage semantics is a gate for full baselines. It does not invalidate the control's explicit numeric reward.

### 2026-11-13 — Three remaining environments

The remaining tasks are `da-1-3` (CRC matrix omission), `da-17-3` (lupus raw counts), and `da-20-3` (COPY normalization). Their remote CPU image job initially waited for federation acknowledgement during the incident tracked in [Marin #11389](https://github.com/marin-community/marin/issues/11389), then completed in 229 seconds. No claim is made that the whole incident was resolved.

The first runtime check was rejected at the shared account's 45-snapshot quota before sandbox creation. The supported cleanup tool identified 45 snapshots unused for more than one hour with no sandbox references, revalidated them before deletion, and confirmed their removal. Fourteen provider-owned snapshots were retained. The second check passed CRC hashes but exposed a validator mistake: its metadata header is quoted whitespace, not tab-separated. The check was corrected without changing input bytes.

The third attempt passed all three tasks. [The archived runtime report](pilot/remaining-runtime-report.json) confirms 78 input checksums, Pi 0.87.0, the judge dependency, CRC metadata columns, and the lupus matrix's 1,263,676 cells by 30,172 genes with `X` referencing `raw/X`. Every validation sandbox closed. These checks establish packaging and access, not biological task feasibility or solver success.

### 2026-11-13 22:31 UTC — Qwen continuation submitted

Standard-launcher preflight passed. One batch containing exactly the three remaining tasks was submitted with two concurrent trials, zero trial retries, a 30-minute scheduling deadline and a four-hour parent timeout. The successful control is excluded. The latest observed state at initial submission was worker initialization; results remain pending verification. A 22:46 UTC check showed the parent running with no recorded failures. No additional full run or model cohort was launched.

Source command in the original workspace: `PYTHONPATH=lib/rolloutengine/src:lib/taskforge/src:lib/taskcompendium/src:lib/shellbox/src .venv/bin/python artifacts/biomnibench-qwen-pilot/launch_remaining.py --submit`. The public relocated entrypoint is [pilot/launch_remaining.py](pilot/launch_remaining.py).

Execution identity: `20261113-223120-Qwen-Qwen3.6-35B-A3B-eb2719471b9f4c029731104f4efa016a-biomnibench-remaining-pilot-c6401b5785495ea3907f8575bdbc7b94`. Runtime access requires team credentials; private endpoint and storage URLs are intentionally absent.

### 2026-11-13 — Judge lifetime decision remains open

The current co-hosted launcher holds the judge for the evaluation batch. A shared judge could amortize startup and intermittent requests, but an always-on service could reserve idle GPUs. [Ben's earlier RFC](https://github.com/marin-community/marin/issues/7236) proposed multi-purpose shared serving; Russell and Romain raised utilization and maintenance concerns. No maintained shared MiniMax service was verified. This continuation retains the selected BixBench serving pattern. Future reuse must preserve model and grading provenance.

## Next decision

Verify the three new solver outcomes, durable grades, dashboard ingestion and resource cleanup. Audit rubric judgments and modified-input feasibility before agreeing on full-50 baselines. Research code has not been promoted into a Marin contribution PR. See [README.md](README.md) for reproduction prerequisites and commands.
