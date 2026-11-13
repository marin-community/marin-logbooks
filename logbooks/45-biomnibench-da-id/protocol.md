# [evals] Validate BiomniBench-DA public-50 for ID evaluation

## TL;DR

Propose **BiomniBench-DA public-50 with five derived-input variants** as a candidate in-distribution (ID) development evaluation for Marin. Review the protocol and collect external-model evidence before deciding whether to add it to the ID policy. No Marin models, including Snowball, will be evaluated until that policy decision.

Use Marin's standard evaluation launcher and the selected BixBench harness/judge settings, while retaining Biomni's tasks, grading rubrics and 60-minute agent budget. Our Daytona eval account rejects the authors' 16 GiB RAM / 20 GiB disk requests, so the proposal caps resources at 8/10 GiB and documents the affected inputs. This is a modified public-subset protocol; it does not reproduce the paper's full-benchmark scores.

The Qwen control task (`da-4-6`) completed with a signed score of 28/100 and unsigned comparison of 38/100. Both required reports and the graded verifier result were read back from the archive. EvalDash's API reports continuous reward 0.28 with 1/1 scored coverage; visual rendering and independent audit of the judge's decisions remain pending. The remaining three pilot cases are in image preparation; full baselines remain pending.

## Description

[BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA/tree/e1c8ca5e11a620087bc48d97888eb69176a1f235) contains biomedical data-analysis tasks requiring executable analysis and written reports. Use all 50 public tasks at revision `e1c8ca5e11a620087bc48d97888eb69176a1f235`, retaining the five modified-input tasks in the denominator. The [paper](https://www.biorxiv.org/content/10.64898/2026.05.12.724604v2.full) reports aggregates over 100 tasks and three repetitions. We have not found author scores for the public 50 or saved per-task judgments that establish scoring parity.

The evaluation would measure multi-step analysis and reporting for ID development. Its incremental value beyond BixBench still needs evidence: task-content comparison alone does not establish different model rankings or useful checkpoint-selection signals. This proposal follows the external-baseline and evidence pattern in [SimpleQA Verified #11063](https://github.com/marin-community/marin/pull/11063).

The existing [task-content comparison with BixBench public 205](https://github.com/marin-community/marin/issues/9258#issuecomment-6781595235) reviews all 50 BiomniBench prompts/input descriptions, all 205 BixBench questions and 12 selected BiomniBench rubrics. It identifies candidate additional coverage in censored survival analysis, TCR repertoire interpretation, cell–cell communication, gene co-expression networks and chromatin analysis. Its manual inventory labels 27 tasks as substantially overlapping, 11 as extensions and 12 as distinctive workflows; these are granularity-dependent judgments, not a measured redundancy rate. The comparison motivates this proposal but does not establish different model rankings, reliable grading or feasibility of the five modified-input tasks.

## Hypothesis or Goal

Determine whether the public tasks remain valid under the disclosed resource and input changes, whether scoring is reproducible and auditable, and whether the benchmark adds useful coverage beyond BixBench.

Review requested:

- Are the resource/input changes and signed-penalty scoring acceptable for an ID proposal, subject to validation below?
- What additional evidence is needed to establish value beyond BixBench?
- Confirm implementation placement before policy adoption. The pilot uses the existing Harbor path; TaskCompendium/TaskSpec migration is deferred during protocol investigation. That does not establish an exception to the guidance that new evaluation implementations belong in TaskCompendium.

## Proposed protocol

| Component | Setting |
| --- | --- |
| Population | All 50 public tasks; no exclusions based on model performance. Report the aggregate and separate 45 unchanged-input / five derived-input summaries. |
| Sandbox | Daytona; preserve each task's native CPU/GPU allocation (2 CPUs, no GPU); RAM = min(native, 8 GiB), disk = min(native, 10 GiB). The release requests 16/20 GiB for 48 tasks and 4/10 GiB for two. |
| Time limits | 3,600 seconds for the agent; 900 seconds for the verifier. Setup and serving startup are separate. |
| Harness | Pi through Marin's standard launcher and vendored Harbor. Pilot pin: Pi 0.87.0. The selected BixBench YAML does not pin Pi; the final baseline version still needs agreement. |
| Context/output | BixBench policy's 65,536 context / 16,384 per-response output settings, with its compaction behavior and model-specific reasoning settings. Verify effective requests. |
| Judge | MiniMaxAI/MiniMax-M3-MXFP8, revision `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`, using the selected BixBench campaign's dedicated-server configuration. Preserve Biomni's rubric and grading prompt. |
| Judge request | Pilot candidate: temperature 0, 8,192 output tokens, 120-second request timeout, no SDK retries; outer verifier limit 900 seconds. These are explicit Biomni pilot settings, not a claim of identical BixBench request settings. |
| Baselines | Qwen3.6-35B-A3B first, then Gemma-4-26B-A4B-it and GPT-OSS-20B. Use pinned campaign configurations and record effective reasoning settings. No Marin/Snowball models. |
| Attempts | Pilot: one attempt, no automatic trial retries. Proposed full baseline: three independent solver repetitions per task, subject to protocol and compute-budget agreement. Agree on uncertainty estimation and invalid-output/timeout/recovery rules before launch. Operational retries and judge-only recovery are not independent repetitions; do not take a best-of score. |
| Concurrency | One trial for the initial control; two proposed for the four-case pilot. Concurrency is an operational setting. |
| Evidence | Preserve per-task submissions, trajectories, judge responses, rubric decisions, failures and provenance; validate continuous-score reporting through EvalDash. |

The pilot uses Marin commit `246f795749166285a864e891e56facfbcf9127e7`, with local adapter/preparation files and a bundle-inclusion change. It is not an unchanged main-only execution. Freeze the complete source, image and input manifests for baseline runs. The copied Qwen configuration pins model revision `995ad96eacd98c81ed38be0c5b274b04031597b0`; remaining baseline revisions must be recorded before their runs.

## Changes from the release

Live create requests against the Daytona eval account rejected 16 GiB RAM and 20 GiB disk with explicit per-sandbox maxima of 8 GiB and 10 GiB. These are verified account constraints, not universal Daytona limits. The current choice is the closest supported allocation on this account without requesting a limit increase.

| IDs | Change | Reason and validation required |
| --- | --- | --- |
| M01–M02 | For `da-1-3` and `da-1-4`, omit the approximately 19.18 GB expression matrix `GSE236581_counts.mtx`; retain clinical and cell metadata. | The matrix exceeds the disk limit. Verify that retained data supports every rubric requirement; expression-based annotations can no longer be rederived. |
| M03–M05 | For `da-17-1`, `da-17-3` and `da-17-5`, replace the approximately 12.22 GB lupus H5AD with the pinned 2.64 GB raw-count derivative. | Preserve all cells and required metadata within the disk limit. Validate gene/cell/donor alignment and task-specific feasibility; normalized expression and embeddings are absent. |
| M06–M07 | Disclose the five input changes in task instructions and remove copies of deliberately omitted files from image recipes. | Prevent assumptions about unavailable data; retain exact prompt/recipe diffs and file hashes. |
| M08 | Existing adapter normalizes `da-20-3`'s 72 JSON-form Docker `COPY` entries to a directory copy. | Historical workaround for Daytona Dockerfile parsing. Apply only if needed by the prepared-image path; verify identical paths and bytes with no extra files. |
| M09 | Cap RAM/disk at 8/10 GiB; preserve smaller native requests and native CPU/GPU. | The confirmed account ceilings reduce the computational budget. Record requested/enforced resources and OOM/disk failures; input fit alone does not prove analysis feasibility. |
| M10 | Replace the release's judge provider implementation with the selected MiniMax judge through Marin's verifier connection. | Align judge model/serving with the selected BixBench campaign. This can change grading decisions and requires auditing. |
| M11 | Apply signed rubric penalties; retain an unsigned comparison score from the same decisions. | The written rubric includes negative points that the released parser drops. This is an intentional scoring change. |
| M12 | Require complete, unique criterion judgments with valid levels/reasons; reject malformed or truncated judge responses. | Treat grader failures as unscored, rather than silently accepting incomplete grades. Error and recovery rates can differ from the release. |
| M13 | Prepare digest-pinned images with Pi/verifier dependencies; adapt verifier connection variables and export continuous `reward`. | Required by the current Marin path. The control image needed a separate data layer to restore its original input at its original path; runtime checksum validation passed. Preserve build recipes, package versions and hashes. |

If any modified task cannot support a required rubric criterion, that blocks the full-50 protocol. Restore sufficient inputs/resources or seek review of a revised protocol before running full baselines; do not count input-induced impossibility as model failure or silently exclude the task.

<details>
<summary>Lupus derivative construction and provenance</summary>

The lupus derivative retains `obs`, `raw` and `uns`, rebuilds `var` from `raw.var`, and links root `X` to `raw.X`. It retains all cells; it is not cell subsampling. Both `X` and `raw.X` contain raw counts. Other root groups, including the original normalized matrix and embeddings, are omitted. The original container filename is retained.

- [Pinned derivative](https://huggingface.co/datasets/open-athena/biomnibench-da-lupus-raw-counts/tree/60044fab4ca09e55041729670deb55b4e6205c83): revision `60044fab4ca09e55041729670deb55b4e6205c83`, file `lupus-raw-counts.h5ad`, 2,642,705,030 bytes.
- Derived SHA-256: `33179d1cf993091dc0bd41372629534f6b7da1ce1114d74c36110ffbee375d53`.
- Original SHA-256: `7ba85edbdc033a9aeeecabe7a1cfcb0660282cc24b04f49fb52dd3825052c620`.
- These values come from the existing adapter. Rehash downloaded bytes before use; preserve the conversion recipe, source attribution and license. Record the CRC matrix's source hash and all prepared-file hashes in the final manifest.

</details>

## Scoring and failure handling

Preserve `/app/trace.md`, `/app/answer.txt` and the source rubrics. The judge reads agent-authored reports; it does not independently verify every reported action. Audit sampled judgments against the captured tool trajectory.

For each task, sum signed rubric points, retain the raw total, and clamp to `[0, 100]`. Average the clamped per-task scores. Thus a raw −5 is reported as zero, and a reliability penalty of −10 reduces otherwise perfect work from 100 to 90. The released parser's unsigned calculation would yield 100 in the latter case. Neither calculation is established as the one used for the paper's results.

Export `reward = score / 100` for Marin's continuous metric path, preserving the 0–100 score, raw signed total and unsigned comparison in the evidence. Verify EvalDash's displayed scale; report mean rubric score, not accuracy.

The selected runner does not yet guarantee this failure contract: with no exception and a present `verifier_result`, `_read_trial` marks the trial scored even when `rewards.reward` is missing, defaulting its value to zero ([source](https://github.com/marin-community/marin/blob/246f795749166285a864e891e56facfbcf9127e7/lib/marin/src/marin/evaluation/harbor/runner.py#L207-L233)). This is confirmed by code inspection, not an end-to-end control. Add missing, nonnumeric and nonfinite reward controls alongside a valid zero; verify scored coverage as well as the aggregate. Resolve any demonstrated adapter/shared-runner defect in a scoped PR before full baselines.

Proposed failure handling:

- Confirmed empty submissions receive zero; grade partial submissions and flag them. Confirm emptiness only after successful inspection of the stopped agent's output location and successful collection. Missing files in an incomplete archive do not establish emptiness. Distinguish confirmed empty output from collection failure in result metadata before full baselines.
- Setup, collection and judge failures remain unscored. Failure to collect a report is not evidence that the report was empty.
- On agent timeout, establish that the agent stopped before capturing and grading its submission. This behavior still needs live validation for the selected runtime.
- Judge-only recovery reuses the same saved submission, rubric and judge settings; retain both original and recovery records. Record any solver rerun separately.
- Report graded coverage and all failures. An incomplete scored-task mean is not a complete 50-task score. Resolve retry/invalid-output rules before full baselines: the paper specifies a retry for invalid/missing output and then zero, which differs from the current pilot's zero retries and proposed unscored infrastructure failures.

Preserve the source instructions' paper-lookup restrictions. Record network access and installed packages; the current environment permits runtime installation and does not fully pin transitive dependencies. The unchanged-input group still shares resource, harness, judge and scorer changes; its score is not native-protocol parity.

## Status

Evidence collected on 2026-11-13:

| Check | Result and limit |
| --- | --- |
| Resource limits | Live requests confirmed the eval account's 8 GiB RAM / 10 GiB disk ceilings. |
| Scorer replay | Local controls reproduce signed 100/95/90 versus unsigned 100/100/100 for reliability levels A/B/C. These are synthetic judgments, not model scores. |
| Empty submission | An intentionally empty local submission directory emits zero and makes no judge call. The current `missing_submission` label does not itself distinguish confirmed emptiness from failed collection; this control does not validate remote collection-failure handling. |
| Control environment (`da-4-6`) | Remote validation passed for Pi, Python/R, verifier dependencies, the original input checksum and sandbox closure. This does not validate the other 49 tasks or live judging. |
| Standard launcher | Qwen control completed: 28/100 signed, 38/100 unsigned, both reports preserved, verifier status `graded`, 1/1 scored and no recorded trial errors. EvalDash API readback agrees (reward 0.28, continuous metric); this is one control task, not a benchmark-wide score. |
| Remaining execution | Four-case validation (`da-4-6`, `da-1-3`, `da-17-3`, `da-20-3`) and full-50 baselines remain pending. All five derived-input tasks require checks before a full baseline. |

<details>
<summary>Preparation failures and repairs</summary>

Early image preparation failed on an interactive package-install prompt and then on a missing input file; both were repaired before the control evaluation was submitted. A local transport check timed out; the remote worker check subsequently passed. Preserve these negative records in the evidence package. They are not solver failures or zero scores. Historical Luna/GLM diagnostic scores use different settings and will not be pooled into the proposed baselines.

</details>

Before requesting ID adoption:

- [ ] Freeze the final protocol and a 50-row task manifest recording original/derived input hashes, image digests, rubric feasibility, requested/enforced resources and model/Pi/judge revisions. Agree on the proposed three independent solver repetitions and compute budget separately from infrastructure retries and judge recovery. Proposed reporting: mean each task across repetitions, then average tasks equally; report per-repetition means and a task-cluster bootstrap interval, preserving repetitions within each resampled task. Label this as task-sampling uncertainty, not a complete estimate of solver or judge variability.
- [ ] Verify transformed inputs and rubric feasibility for all five affected tasks; validate packaging, enforced resources and task-specific disk/memory use.
- [ ] Exercise good, wrong, partial, confirmed-empty, malformed-judge, missing/invalid-reward and collection-failure controls, plus timeout capture and judge-only recovery; verify coverage and continuous-score display, and audit live judgments against tool traces.
- [ ] Complete external-model public-50 baselines with per-task scores, coverage, 45/five subgroup summaries, failure rates and the agreed uncertainty analysis.
- [ ] Attach the exact pilot patch, preparation recipes and compact validation records; the linked base commit alone does not reproduce the locally modified pilot.
- [ ] Verify archived results and EvalDash ingestion/display; attach reviewable evidence links. Local records are prepared, but a published evidence package is pending.
- [ ] Report wall time, startup/queue time, tokens, sandbox/GPU use, retries, cleanup and measured cost where available. Separate estimates from measurements.
- [ ] Assess incremental coverage beyond BixBench using matched external-model results and documented protocol differences. Reuse suitable existing BixBench evidence where available; additional runs require a separate agreed plan. Three model rankings alone are limited evidence and do not establish checkpoint-selection value. Resolve implementation placement and ownership with evaluation maintainers before the policy decision.

## Links

- [Biology evaluation tracking #9258](https://github.com/marin-community/marin/issues/9258).
- [Existing adapter PR #184](https://github.com/marin-community/harbor/pull/184) and [adapter source at inspected revision](https://github.com/marin-community/harbor/tree/95a4b88b54cae602ade6e79d9536a2c766657301/adapters/biomnibench-da/src/biomnibench_da).
- [Selected BixBench policy](https://github.com/marin-community/marin/blob/246f795749166285a864e891e56facfbcf9127e7/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/harbor-configs/bixbench-pi.yaml) and [MiniMax judge configuration](https://github.com/marin-community/marin/blob/246f795749166285a864e891e56facfbcf9127e7/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/judge-model-configs/MiniMaxAI-MiniMax-M3-MXFP8.yaml).
- [External-baseline evidence precedent #11063](https://github.com/marin-community/marin/pull/11063).
- Graded Qwen control in EvalDash (access-controlled link omitted). Scorer investigation and complete pilot provenance package: pending attachment before an adoption request.

## Decision Log

Agreed for this investigation: public-50 scope, disclosed resource/input changes, signed scoring with an unsigned comparison, and external baselines starting with Qwen. No Marin-model evaluation before the ID policy decision.

Still open before full baselines: final runtime pins, independent repetitions, uncertainty estimation and failure/recovery rules. Implementation placement, ownership and adoption remain maintainer decisions.

## Conclusion

One control task has completed end-to-end grading and EvalDash API ingestion. There is not yet evidence sufficient to recommend ID adoption. The next decision is whether the proposed deviations and validation plan are acceptable for that assessment.
