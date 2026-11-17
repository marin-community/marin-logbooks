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

## 2026-11-16: continuation results and judge-only recovery

The three-task continuation ended with two scores: da-1-3 scored 67/100
(72 without signed penalties); da-17-3 reached the 3600-second agent limit
and received 0/100. Task da-20-3 has no grade. Its saved MiniMax response
contains exactly `{}`, with finish_reason=stop and two completion tokens.
The strict parser correctly rejected this as missing criterion ratings.
The native run remains infrastructure-failed with 2/3 grading coverage;
its 33.5 mean among graded tasks is not a benchmark score.

Both da-20-3 reports were retrieved from durable storage. Their SHA-256
values are pinned in `pilot/regrade_saved.py`. That script prepares one
judge-only retry using the existing rubric, prompt, MiniMax revision,
8192-token output cap, temperature zero, and 120-second request timeout.
The original failed response and native run remain unchanged. The retry
uses a separate storage prefix and no solver or Daytona execution.
At preparation time no compatible judge was identified in the running-job
inventory. The proposed execution starts the original eight-H100 judge
under a CPU parent capped at 5400 seconds, then cleans it up after one
request. Source and configuration are committed before submission.

Recovery has not yet executed. It must produce a complete valid grade;
another empty object remains an ungraded failure. The four-task pilot,
including the earlier 28/100 control, is not a full baseline.

The recovery was submitted at 17:09 UTC from
`6aebb77b0a47ec9bb33165da715749ae1555e538`, after lint and a storage-read
preflight passed. The preflight matched both report hashes and the original
failed reply. The 5400-second parent runs `pilot/regrade_saved.py` with
`--source-commit` set to that revision and distinct source/output prefixes.
Its preparation child reported a shared-cache hit for the pinned MiniMax
weights. Grading is still pending at this checkpoint. Submission receipts,
private storage paths, and monitoring state are retained under the local
`artifacts/biomnibench-qwen-pilot/recovery-da-20-3/` directory.

At 17:21 UTC, the judge worker was `building` with `SchedulingGated`:
Kueue had not admitted the GPU workload. No grading request had run.
The single recovery remains queued under its 5400-second parent cap;
there is no additional launch. The local monitoring record names the job
and next checks. No host scheduling tool is available for an automatic
follow-up after this session yields; a later status check must verify the
job's terminal state, saved grade/error, and descendant cleanup.

## 2026-11-16: recovered grade and full-pass authorization

The judge-only retry succeeded: da-20-3 scored 74/100, signed and unsigned,
with both reports present. The parent succeeded after 55m16s including queue
and startup; its judge worker and broker are terminal. The original failure
is preserved. This establishes saved-submission recovery without a solver
rerun, but does not establish the judge's error rate.

The user authorized completing the grading safeguards and a full public-50
Qwen pass, with speed prioritized. Plan one fresh solver attempt per task,
16 concurrent trials, unchanged 3600-second agent and 900-second verifier
limits, one Qwen H100x4 service and one shared MiniMax H100x8 service, with
an eight-hour parent cap. No Marin models or additional repetitions are
included. Allow at most one immediate retry for a malformed judge response,
preserving both responses, while the shared judge remains warm.

Metadata inventory contains 50 tasks and 44 distinct input sets after the
five disclosed input changes. Reuse four validated pilot images and prepare
40 missing images on remote CPU resources; no dataset payloads are downloaded
to the shared VM. Native LFS SHA-256 or Git blob identifiers validate source
bytes before publishing layers. All original instructions and rubrics remain
pinned; the existing five input-disclosure notes are retained.

Full-pass safeguards now include a strict finite-numeric reward check in the
research branch's Harbor aggregation path. An invalid present verifier reward
is unscored, including a missing value, string, NaN, infinity, or boolean;
a genuine numeric zero remains scored. Existing agent-error taxonomy is
unchanged. The focused runner suite passed 51 tests.

The full-pass verifier retries only a malformed judge reply once, retaining
both raw responses and error metadata under `judge-attempts/`. Frozen reports
are copied to the verifier root even if both attempts fail. It inspects the
submission directory before assigning an empty-submission zero. Five offline
OpenAI-transport tests passed for valid, retry-success, twice-invalid, empty,
and missing-directory cases. The original prompt and rubric are unchanged.

The current US Daytona quota supports 16 sandboxes at the requested resources.
There are 45 required snapshots: 44 input images plus a second resource size
for the shared da-14 image. The documented cleanup tool identified and
requested deletion of two stale unreferenced snapshots. Provider-owned
snapshots are excluded. The image job prepares the remaining 40 images in
one CPU job; its source is pinned at `1de749bc17912734264533cf6b514019ab428c3c`.

The image job succeeded on 2026-11-16. Its output archive was read back and
hashed as `69b19d6b28318daee3da36c28f595e2d5253b98fffc51c9f57632cc4b63440a9`.
It contains all 40 new image manifests. Combining these with the four pilot
images produced `full/task-manifest.json` and 50 task recipes. The local
control input also matches its pinned Git blob identifier. All 50 original
rubrics parse successfully and each has a maximum of 100 points.

Daytona inventory confirms the two stale snapshots have been deleted; the
remaining 14 entries are provider-owned. Runtime validation will use the
actual Harbor-emitted snapshot specifications, with `PYTHONPATH=lib/marin/src`
and `uv run --isolated --project vendor/harbor --frozen --extra runtime python
logbooks/45-biomnibench-da-id/full/validate_runtime.py`. This CPU-only check
prepares 45 snapshots and verifies all five modified tasks plus the two
largest distinct unchanged inputs before starting the GPU services.

The full launcher preflight passed on source `9db13af9eba8c4bb2e3bd47e11b3eea74e5f3efa`.
It peaked at 555,708 KiB RSS, exceeding the shared VM's 500 MiB limit; future
launch preparation will run remotely. The remote mode uses the standard
`launch_group` and keeps its CPU parent alive until the evaluation finishes.
The first CPU runtime check exited before snapshot creation: the snapshot
driver consumes relative dataset paths from its working directory. Correct
the helper to run from the policy directory with an absolute driver path and
module path, and expose subprocess stderr for any further diagnostics. Keep
the failed job `/exedev/biomnibench-public50-runtime-20261116` separate from
its corrected successor. Neither attempt starts model GPUs.

The corrected runtime job `/exedev/biomnibench-public50-runtime-v2-20261116`
succeeded on source `e082006b24a47c18f185fbfcdf7359a4b6e40525`. Readback
confirmed 45 active snapshots and seven passing input checks, with every
test sandbox closed. The last two snapshots to finish were the 2.64 GB
lupus image and 3.63 GB da-19-4 input image. Daytona exposed their pulling
state but not registry-image download logs. The validation archive SHA-256
is `2c52849eb7f97ae5f6e0e4582e3549e972fa3a22240dc52402ed44a48d12f5cc`.
Both image and validation archives were copied to durable preparation
storage and their remote checksums matched. See `full/validation-summary.json`.

Submitted the full public-50 launcher as
`/exedev/biomnibench-public50-qwen-20261116` on source
`c4e8f2a29c13fb3a3f3d399481c2f49d0f3d198f`, after reading back the passing
validation. The launcher uses one CPU, 2 GiB RAM, and a 16 GiB disk, with
failure and preemption retries disabled and an eight-hour cap. It calls the
standard evaluation submission path and remains alive until its child
evaluation finishes. The fresh 50-task pass retains 16 concurrent trials,
one Qwen H100x4 service and one shared MiniMax H100x8 judge. Submission is
not evidence of model readiness, grading, or EvalDash ingestion; those
checks remain pending. See `full/launch-summary.json`.

The first full launcher failed before creating an evaluation child or model
service. `launch_group` attempted a second federation hop from the already
federated CPU launcher, and the regional controller rejected that destination.
Correct the remote launch mode's submission-cluster identity to its existing
destination so children stay on the same controller. Preserve the failed
launcher receipt and reuse the completed input validation for its successor;
no solver attempt or GPU service was started by the rejected submission.

The corrected launcher `/exedev/biomnibench-public50-qwen-v2-20261116`, on
source `90cc6ad17d243a60135937ed8403a9a892c3760b`, successfully submitted its
evaluation child at 19:25 UTC. At 19:26 UTC the evaluation and its Qwen model
preparation job were running. Its run ID is
`20261116-192525-Qwen-Qwen3.6-35B-A3B-50cdfb029c2c4394a664f6c57b20fdf1-biomnibench-public50-qwen-37e4ab136e2755f08bc074ebe26abcf1`.
Model readiness, first graded trials, final coverage, cleanup and EvalDash
ingestion still require verification. The passing input validation is reused;
the failed first launcher did not create any solver attempts to combine with
this fresh public-50 pass.

At 20:27 UTC, Qwen model preparation had succeeded but its H100x4 worker
remained scheduling-gated after about one hour in the queue. Kueue reported
insufficient unused GPU and RDMA quota. No solver task or judge call had
started. A read-only comparison of the other H100 region found some free
GPU quota but many CPU/GPU placement waits; it did not establish a faster
placement. Retain the validated hardware and current queue rather than
claiming that submitted means evaluating. The run advances automatically
when admitted, within its existing eight-hour parent cap.

## November 17: reconcile the interrupted public-50 pass

The evaluation driver was preempted by Kueue after 3h26m; its candidate and judge services terminated with the parent. The last durable Harbor summary, updated November 16 at 22:49:58 UTC, has 38 terminal trials: 29 recorded rewards and nine errors. Twelve trials were unfinished. This is not a complete benchmark score, and dashboard ingestion has not been verified.

Three errors (da-1-4, da-13-5, da-13-6) are malformed judge responses after both configured attempts. Their answers and traces are archived. The first task returned criterion entries without the required outer schema; the other two returned empty objects. Preserve those failures rather than accepting a different schema retrospectively.

Task da-15-1 finished its solver but failed Daytona command-output collection. Its trajectory contains successful `write` calls for `/app/trace.md` and `/app/answer.txt` at steps 63 and 64. Recovery inputs were extracted verbatim from those payloads, with hashes and provenance, without executing trajectory commands. They are reconstructed submissions, not sandbox filesystem readbacks.

Five tasks (da-8-1, da-8-2, da-8-3, da-9-1, da-9-7) failed before solver execution because the account's 45-snapshot quota was reached. On November 17, none of the 45 preflight snapshots remained; the account held 25 non-provider snapshots. The deletion timing and actor are unverified. No sandboxes matching the prepared snapshot names or benchmark name remained. Preflight readiness before a long GPU queue did not guarantee runtime readiness.

The [recovery selection](full/recovery-plan-nov17.json) preserves 29 scored tasks, selects four saved submissions for separate grading recovery, and selects only 17 tasks for solver continuation. No replacement run has been launched. A fresh bounded compute window is requested because the original eight-hour allocation has ended. Before launch, recheck snapshot capacity and ensure the selected subset close to task execution. Keep original results and recovery identities separate.

Raw trial metadata, saved judge replies, answer/trace inputs, hashes and private account inventory are retained under the task's ignored `artifacts/biomnibench-qwen-full/reconciliation/` directory. The original durable object prefix is recorded in the existing submission evidence; no private endpoint or credential is copied into this notebook.

The continuation was approved with a fresh eight-hour cap, the same four-H100 Qwen and eight-H100 MiniMax services, 17 solver tasks, and four saved submissions. The recovery wrapper uses the standard evaluation executor and its canonical result path for the 17-task subset. It first makes one new grading request per frozen submission with the original rubric, prompt and settings, recording those grades separately, then ensures the subset's 16 snapshots immediately before solver execution. It does not delete account snapshots or overwrite prior scores. Frozen inputs were uploaded to a separate durable prefix and are checked against committed SHA-256 values at execution. Static lint and syntax checks passed; full-project type checking was not repeated locally because its measured memory exceeds this VM's limit. End-to-end validation remains the bounded recovery itself.

Monitoring caught a startup failure in the first recovery launcher: `daytona` is an optional dependency absent from the base `marin-core` environment. No model services or solver attempts were created. The wrapper now invokes the supported snapshot CLI in its declared Shellbox environment instead of importing the optional SDK into the base launcher. Static checks passed. The corrected launcher was submitted as `/exedev/biomnibench-qwen-recovery17-v2-20261117` from `c29382e5b2ce0b2f850be0c93ffcffabe17614d2`, retaining the first submission's deadline of November 18 at 02:25:28 UTC. At 18:38 UTC it had submitted its evaluation child and passed the previous startup failure; model readiness and results remained unverified.

### November 17 — First-pass coverage and recovery split

Reconciliation of first-pass Harbor trial records against all 50 task directories gives 26/45 unchanged-input tasks and 3/5 derived-input tasks with numeric signed and unsigned scores. Three recorded zeros are included. The other 21 comprise four saved submissions, five sandbox setup failures and twelve interrupted trials. Coverage denominators include every task. A full-population mean remains unavailable.

The combined recovery ended before any additional solver trials started. Its four grading subprocesses exited with code 2 and produced no grade artifacts. A subsequent live Daytona preflight found all 16 required snapshots missing and hit the account limit of 45 on creation. On the logbooks branch, commit `8004b2dac3e932e5437ab0dbeba4d85d4f5383cb` switches frozen-submission grading to the explicitly bundled runtime task tree. All twelve files used by those four graders (script, rubric, configuration) matched the originals byte for byte. The judge-only submission `/exedev/biomnibench-four-frozen-20261117` uses the remaining approved window.

The earlier seven sandbox checks exercised all five derived-input environments. For the two colorectal tasks, they verified hashes, absence of the omitted matrix and presence of patient, cell subtype, treatment and tissue fields. For the three lupus tasks, they verified hashes, the raw-count matrix dimensions, the shared raw-count storage and presence of donor, disease, cell type and ethnicity fields. The monocyte task also includes the checksummed interferon signature. These checks establish the named inputs are accessible. They do not establish correct patient joins, ancestry sample sizes, gene alignment, or completion of every rubric analysis within the resource limits.

First-pass trial accounting is in [first-pass-resource-accounting.json](full/first-pass-resource-accounting.json). Of 38 terminal records, 33 contain solver usage: 26,451,485 input tokens and 886,903 output tokens. Median solver duration among these 33 was 498.94 seconds. The 32 trials with verifier timestamps had a median verifier duration of 7.41 seconds. Phase totals sum individual trial durations across concurrent runs. Queueing, serving startup, the 12 incomplete trials, judge token usage and dollar cost are outside these measurements. None of the 38 records reported dollar cost.

### November 17 — Audit of the graded Qwen control

For pilot trial `da-4-6__ZPjiKBr`, the archived answer, trace and trajectory support the judge's finding that the submission used Fisher and Wilcoxon tests. The report provided neither the requested log-rank comparisons nor Cox hazard ratios with confidence intervals. Those two criteria received zero. Tool observation 14 records the within-group median of 8.052753644 and two groups of 32, supporting criterion 2. Its judge explanation incorrectly describes using the scaled value directly as implicit division by ten. The rubric also permits stating the scale, and the answer explicitly names the scaled column. This wording discrepancy does not establish a change in points.

There is direct evidence for a sourcing error underlying the ten-point penalty. The solver's first citation supplies a DOI and a PMID for an asserted lung cancer study. Those identifiers resolve to different papers on unrelated subjects: [DOI record](https://pubmed.ncbi.nlm.nih.gov/39059362/) and [PMID record](https://pubmed.ncbi.nlm.nih.gov/39029679/). This verifies a concrete example independently of the examples MiniMax supplied. The six recorded criterion scores sum to 28 signed and 38 before penalties. This retrospective check covers one known control and cannot estimate judge disagreement across the public 50.

For each of the three proposed external baselines, [external model pins](full/external-model-pins.json) identifies a successful BixBench run whose EvalDash model location and revision match the selected campaign configuration. Both Gemma candidates are retained because their coverage differs. The hash and agent kwargs describe the configuration file. Verifying actual inference requests and comparing judge, timeout and trial coverage across runs are separate checks.

### November 17 — Three frozen grades recovered

The judge-only Iris job succeeded and its model worker and broker stopped. All eight archived report files matched their source checksums. Replaying the saved MiniMax decisions with the original rubrics reproduced the signed and unsigned scores for da-1-4 (76/76), da-13-6 (75/80), and da-15-1 (50/55). The da-15-1 reports were reconstructed from successful file-write observations in the original trajectory. The single recovery call for da-13-5 returned a malformed response missing the expected criteria and reasoning structure. That task remains unscored under the bounded recovery rule.

Qwen now has 32/50 unique tasks with verified grades: 28/45 unchanged-input tasks and 4/5 modified-input tasks. Seventeen tasks still need solver execution and one saved submission has exhausted its agreed grading retry. None of the four judge-only calls is a new solver repetition. The earlier combined recovery's worker logs confirm four missing grader-script paths and the Daytona snapshot quota error.
