---
title: Mochi doom-loop FTPO
source_repository: marin-community/marin
task_issue: https://github.com/marin-community/marin-logbooks/issues/77
related_issues:
  - https://github.com/marin-community/marin/issues/12240
  - https://github.com/marin-community/marin/issues/9691
research_branch: codex/77-mochi-doom-ftpo
marin_base_revision: f9fe837d6398cf27355b1f34ccba997fafd67ef1
archive_template_revision: fc1cb30da0967f6ab78ffd47444ed9144f50cba9
status: active
---

# Mochi doom-loop FTPO

## Question and decision criteria

Can the original online FTPO objective cut exact repetitive tails in Mochi SFT step 400407 by at least 50% in every measured target domain? Keep native correctness within max(3, 5% of N) lost passes per domain and 2% pooled. Compare each candidate's loop upper bound with the baseline's confirmed loops on identical prompt hashes. Extend sparse 32K domains to 256 prompts before demoting those with fewer than 10 confirmed loops. Non-target domains remain regression checks with at most three additional upper-bound loops.

Use up to 128 GB200 on cw-us-east-08a at production priority. Submit stages when ready and let Kueue schedule them. Serving remains TP1/BF16 with CUDA graphs. No training begins before baseline sealing. Executor start: 2026-11-24 17:46 EST (22:46 UTC); start no training after November 26 13:46 EST and seal by November 26 17:46 EST.

## Baseline

The valid math16k baseline has 12 confirmed loops / 512 (2.34375%), zero unassessed responses and 390 native full-credit passes / 512 (76.171875%). The 32K initial and reserve panels are being rerun after correcting wall-clock censoring; no training has started. The required math target is candidate UB ≤ 6 with at least 365 passes.

V2 cohorts use the complete native-rendered manifest. The first S0 attempt was cancelled at 19:25 EST because its preparer omitted native answer-format instructions from the prompt hashes; its outputs and v1 hashes are excluded. Corrected S0 was submitted at 19:45:04 EST on November 24, runtime 2026.11.24.4. Its first live prompt matches the v2 hash exactly.

The detector remains SkyRL's `tail_loop`, requiring eight exact suffix copies, with full-window search and period caps of 2048 at 16K and 4096 at 32K. Missing or damaged responses enter the loop upper bound. Correctness uses explicit native full-credit pass flags; the metadata repair and immutable input hashes are recorded below.

## Entries

### 2026-11-24 — Recover original inputs and resolve source drift

The original DAPO split survives with its pinned revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`, 16,879 training rows and 512 validation rows. These 512 recovered prompts are the input cohort; the final eligible count remains pending the 2,048-token filter. Hash eligible token IDs under the pinned Mochi tokenizer. The original split used canonical message-JSON hashes; the new token hashes identify the Mochi-rendered prompts.

All 193 safetensors shard SHA-256 values agree between the 08a copy, its verified-copy receipt, and the 02a source export. The 08a export-manifest digest is `1a7886b05df25338a12642b25f25525c0eb6c5551384934a2f089d7ec5b4a88d`. The stale `grug/…-hf/2026.11.10` adoption path is absent; the surviving source is the export directory recorded in the verified-copy receipt. No weights were downloaded or converted. See the [model-identity receipt](model-identity.json).

The finished Snowball corpus contains schema-0.28 tasks, which the current runtime rejects. The November 19 Snowball selection records admitted copies of these same source populations at schema 0.30, supported by the current upgrader. Reuse those copies, retain the original corpus identity, and convert DAPO with the current source converter. Use the admitted TaskTrove instruction-following structured-output population as the design's programmatic IF fallback. Preserve its native checker and report the exact source.

The export has no tokenizer files. Bind `marin-community/marin-tokenizer-tools` at `3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09`, the pinned SFT tokenizer. Preserve assistant-prefill whitespace with the existing SkyRL renderer. Exclude EvalLite, Micro210, all evaluation prompts and all reserves from training before publication.

The design's 1-MiB NCCL buffer predates the current 4-MiB correctness default for NCCL 2.30.7. Keep that default; retain `NCCL_MAX_CTAS=4`, full recomputation, and CPU activation offload. Measure the hardest 32K case in S1.

The task explicitly forbids AGENTS and skill edits, so the archive does not add the workflow-required directory-local instruction copy. Marin's existing root instructions remain in force. Remove inherited workflows only on the archive branch.

Next: validate the CPU publication path and baseline presets locally, publish source, build cohorts on an 08a CPU worker, then submit S0. The full experiment is still pending; source recovery and passing detector tests do not establish a baseline loop rate.

### 2026-11-24 18:05 EST — Submit the cohort build

Published source: `2678348b6b` on `benfeuer/mochi-doom-ftpo`. Submitted `/benfeuer/mochi-doom-cohorts-20261124a` to cw-us-east-08a with 16 CPUs, 96 GB memory, 64 GB disk, production priority, and a 72-hour runtime ceiling. The executor's 48-hour deadline still applies. Command: `uv run iris --cluster cw-us-east-08a job run --job-name mochi-doom-cohorts-20261124a --no-wait --enable-extra-resources --cpu 16 --memory 96GB --disk 64GB --priority production --timeout 259200 -e PYTHONPATH vendor/skyrl/skyrl-train:vendor/skyrl/skyrl-gym:vendor/skyrl -- python -m experiments.post_training.mochi_doom_ftpo.cohorts`. HF and storage credentials were forwarded through Iris secret delivery; no credentials are in source.

The 10 detector tests and four cohort/publication tests pass locally. All-file pre-commit and standalone Pyrefly pass. The affected safe suite reports 1,416 passes, five skips, and four failures in unchanged task-curation tests: two macOS Bash array failures, one code-runner reward failure, and a Linux-only bubblewrap backend call. Baseline replication of those failures is pending. No GPU job has been submitted, and no loop-incidence result is available.

### 2026-11-24 18:12 EST — Fill the calendar reserve without changing the initial cohort

The first CPU build reached cohort freezing and failed because only 79 admitted Ultra calendar prompts survived the 2,048-token cap and exclusions. Keep the initial 64 Ultra calendar prompts and 64 closure prompts. Give Ultra calendar priority in hash ordering; fill the remaining calendar reserve with the admitted TaskTrove instruction-following calendar-v3 source from the same corpus. A regression test checks that the initial calendar half remains entirely from the primary source while excluding the complete reserve from training.

Published fix: `7c4069cce5`. Replacement CPU job `/benfeuer/mochi-doom-cohorts-20261124b` was submitted on 08a at 18:12 EST with the same resources. No previous GPU work or model checkpoint is involved. The detector suite now has 12 passing cases, including open thinking spans begun in the rendered prompt; the cohort/publication suite has four passing cases. All four broad-suite failures also reproduce on the clean base `f9fe837d6398cf27355b1f34ccba997fafd67ef1`.

W&B credentials are absent from this shell's usual stores. Baseline generation will retain console and native records so measurement can proceed while the credential-location question is pending. Training retains its W&B backend pending resolution. S0 evaluates the math16k cohort, greedy 32K panel, and temperature-0.7 panel in one saved-model workload to share the model startup; this differs from the design's three engine partitions but preserves each sampling and budget contract.

### 2026-11-24 18:51 EST — Freeze cohorts and submit S0

The cohort build succeeded at 18:21 EST. All 512 recovered DAPO validation prompts survived the prompt cap. The six 32K domains each contain 128 initial rows and 128 disjoint reserve rows. The manifest records the source versions, dropped prompts, row hashes, and cohort hashes; its SHA-256 is `839713ba2fd14bd755f6600782ed1fac7ca60258d7a7416ddc590de1015b345e`. See [the frozen manifest](cohorts-manifest.json). The published selection contains 183,167 eligible training candidates; these are raw source pools, not the final S3 sampling mix. The mixed round will select only baseline-qualified target domains and set their weights explicitly.

The following v1 hash table is superseded by the native-rendering correction recorded below.

| Cohort | Initial N | Cohort SHA-256 |
| --- | ---: | --- |
| math16k | 512 | `44ac92e7cc75a29872a47fd2a8e9544192548badef31088ea2773d6eb6700253` |
| math32k | 128 | `b5f9113c9d7518b5faa91a6d8e22b30966fcec376c2bc8040ecf9f4de2bd48cf` |
| if32k | 128 | `9fb017a9e7505e9e207ec94b91d4a64f6648eb9afb1570c1c8bc0f480973a3e7` |
| code32k | 128 | `2740f1718525d4a8d1286c9079429c9c3554a07b84e4519602f572d902d7004e` |
| rgym32k | 128 | `4d542fd4d284a4c90894395b984439216282df8c3b01f633ccd2fbdd3a3b52d4` |
| mcq32k | 128 | `4aa9ec654df306747ef0838df0fd1105efef153dfcdc369793bf3ae59b99f152` |
| struct32k | 128 | `d53d6935d682ed937dd98fc4dcf607c7cede9bf06ae7a7216e0cc5036e196bbb` |

Published source: `a620d5ce2e`. All 11 presets pass local preflight, including explicit in-region durable outputs and temporary checkpoint roots. Generation uses TP1/BF16 and CUDA graphs. The baseline disables routed-expert capture because it does not train; S1 will test R3 together with top-32 logprobs. Native FTPO pilots select the lowest loop upper bound, break ties by correctness, and stop after two non-improving pilots or a correctness loss above five percentage points. These controls do not change the FTPO loss or detector.

S0 was submitted at 18:51 EST, version `2026.11.24.2`, 128 GB200 at production priority. The hub coordinator is `/benfeuer/users-benfeuer-checkpoints-mochi-doom-ftpo-doom-baseline-64eac6dcb0-in-5c5ccf7a-2026.11.24.2-coordinator-56b5c94e`. Its GPU child, submitted at 18:52:30 EST, appends `/users-benfeuer-checkpoints-mochi-doom-ftpo-doom-baseline-64eac6dcb0-in-5c5ccf7a-2026.11.24.2-a023f5fa9de9`. Parent and child are running; no incidence is measured yet and no training has started.

The W&B credential is now available for training. S0 retains console and native records. An advisory review and a refreshed affected-test run are in progress against this committed source. Previously observed macOS-only test failures reproduce on the clean base; new failures will be investigated separately.

Next: audit exact saved token IDs, extend any 32K domain with fewer than ten confirmed loops to its fixed reserve, and seal the target set before S1.

### 2026-11-24 19:17 EST — Validate the audit and prepare target-qualified pools

S0 began generating the 512 math16k responses at 19:10 EST, about 19 minutes after coordinator submission. Its immutable runtime remains source `a620d5ce2e`, version `2026.11.24.2`. No loop-incidence count is available yet. The cohort builder ran source `7c4069cce5`; the frozen v1 manifest and Parquet files remain unchanged.

Source `31ca3c1b2a` adds the fixed-reserve decision to the audit. A sparse initial 32K domain requires its complete 128-row reserve before demotion. Candidate audits use `--baseline` to keep the sealed baseline's prompt hashes, cohort size, and target decision. An improved candidate cannot shrink its cohort or demote a domain because it now has fewer loops. Native audit tests report 15 passes; a synthetic CLI baseline/candidate file round-trip also passes. These fixture results are software checks, not Mochi measurements. All seven frozen cohort hashes match the shared hashing implementation.

The training-selection builder reads the frozen raw pools after sealing. It retains math and qualified target domains, balances calendar against closure, and records explicit per-batch quotas approximating math 2 : each other target 1 in batches of 64. It caps each other-domain pool at 4,000 eligible rows. Three publication and selection tests pass, including the native task-family identity consumed by the sampler. Before S3, bind these recorded quotas and the filtered pilot into the launch config and inspect its preflight. Mixed and long stages now require an explicit candidate selection. No training has been submitted.

Two advisory reviews completed. Follow-up changes reuse the existing Parquet reader and one shared selection publisher, remove ignored recipe fields, share prompt/cohort hashing, clarify upper-bound and callback behavior, and count unavailable grading as non-pass. The original frozen receipt retains its original requested recipe; its actual counts describe raw candidate pools. Orchestration remains in the experiment entrypoints, and the canonical storage environment overrides remain documented and tested. Native pilot classification consumes typed batches; offline classification additionally checks saved prompt identities and damaged rows. Both call the unchanged native detector.

The refreshed affected suite reports 1,547 passes, five skips, and six failures. Four unchanged failures reproduce on clean main; the two additional sandbox timeouts pass a focused rerun (three tests). The focused cohort/geometry/locality suite reports 84 passes. Required lint, formatting, and type checks pass.

The runner takes its explicit RL artifact version from `Preset.revision`; the CLI `--version` does not replace that pin. Upcoming runtime revision `2026.11.24.3` passes the S1 preflight with 48 policy/reference GPUs and 64 rollout GPUs, permanent outputs on 08a, and native checkpoints under the 08a three-day temporary root. S1 still waits for baseline sealing.

### 2026-11-24 19:33 EST — Reject the first S0 attempt and refreeze native prompts

A live retained prompt did not match its frozen v1 hash. The cohort builder rendered the TaskSpec's source conversation, while the native shell harness also appended the answer-format instruction. For DAPO that instruction is “Give your answer as plain text.” Reproducing the complete native request yielded exactly the saved 157 prompt token IDs. The model, tokenizer revision, and detector were unchanged. See the [hash mismatch receipt](prompt-rendering-mismatch.json).

Cancelled only the owned S0 coordinator with `iris-cancel-own` at 19:25 EST. Iris confirmed both the parent and GPU child killed; the child reason was parent termination. The partial outputs do not establish loop incidence. V1's hash table and prompt eligibility are superseded, and no training ran against those pools. The original 48-hour deadline remains November 26 at 17:46 EST.

Source `5f2b2f931c` now calls the native shell prompt renderer, including its answer-format instruction and advertised tools, or the native assistant-prefix renderer with its whitespace-preserving tokenization. All 512 original math validation tasks remain eligible; the longest complete prompt is 1,051 tokens. Four focused regression/publication tests pass, as do required formatting, lint, and type checks. The corrected preparer exactly reproduces the live prompt IDs.

Submitted `/benfeuer/mochi-doom-cohorts-20261124c` at 19:32:59 EST on cw-us-east-08a, CPU16 / 96 GB / 64 GB disk, production priority. It publishes new cohort version `2026.11.24.2`, records the rendering contract, and preserves the invalid v1 receipt as superseded. Upcoming S0 uses fresh preset runtime revision `2026.11.24.4`. No GPU job is currently active for this task.

Next: verify CPU publication, preflight the new baseline, submit immediately, and compare a completed live prompt with v2 before accepting the audit. Seal all required domains and reserves before S1.

### 2026-11-24 19:45 EST — Publish native-rendered v2 and resubmit S0

CPU job `/benfeuer/mochi-doom-cohorts-20261124c` succeeded. The v2 manifest was published at 19:42:56 EST; its SHA-256 is `1c2f2b83a3b895b518e7745d3a1fe05af857f56273449c6a3aad9087f9354af3`. Its expected prompt hash exactly matches the saved native prompt used to diagnose v1. All seven cohort hashes verify. All 512 original DAPO validation task IDs remain eligible. See [the manifest](cohorts-manifest-v2.json) and [selection summary](cohorts-summary-v2.json).

| Cohort | Initial N | Reserve N | Cohort SHA-256 |
| --- | ---: | ---: | --- |
| math16k | 512 | 0 | `baca68d0a62f8ab6823694ac09bab00889660a63645f5e7c94e3f3fe84c1d6df` |
| math32k | 128 | 128 | `3645a35331040ad003782ee8938236782e4581bc8ae2e3c3384cc9e40d4e205c` |
| if32k | 128 | 128 | `8a7b7acf2c737045e880783e00ea6e59a1ab03d14894f5a2219acc31dcec9790` |
| code32k | 128 | 128 | `b5312e27017fa4896e3e6b19c104331fcf814a949e7fb807dba00c84bcdeeae5` |
| rgym32k | 128 | 128 | `382773c43bc215645f189afd1c81a6e595e2ca95d093e6da44ebe1f9c48b5a17` |
| mcq32k | 128 | 128 | `026750d3c3999c46819c0a8741b449b9da8c7ab576cabe39687bd4b1820f7dcb` |
| struct32k | 128 | 128 | `e0119f50e982582c27f912949530adf114101be4d9f25fc69b87758328cea52f` |

The eligible training pools contain 183,111 rows, with 16,879 math rows. Complete native rendering drops 56 additional over-cap rows relative to v1: seven calendar, five reasoning_gym, and 44 IF. The heldout union contains 1,792 distinct task IDs; the provisional pilot has 144 rows. All evaluation and reserve identities are excluded from training. Target qualification and the final pilot remain pending baseline counts.

Published source: `1661a15e9f`; native-renderer implementation: `5f2b2f931c`. The new baseline preflight resolves runtime version `2026.11.24.4`, cohort version `2026.11.24.2`, 32 four-GPU hosts, TP1/BF16, CUDA graphs, production priority, and in-region outputs. Submission was accepted at 19:45:04 EST.

Coordinator (marin hub): `/benfeuer/users-benfeuer-checkpoints-mochi-doom-ftpo-doom-baseline-cc75b435e5-in-7985a1b9-2026.11.24.4-coordinator-6bbb3801`. GPU child appends `/users-benfeuer-checkpoints-mochi-doom-ftpo-doom-baseline-cc75b435e5-in-7985a1b9-2026.11.24.4-a8f0fc0dbf9a`, submitted at 19:46:03 EST, 32 replicas with NVLink-domain placement. Parent and child are running. The watchdog is registered and validated.

Next: check a live completed prompt against v2, audit the saved full sessions, extend sparse domains, and seal the baseline before S1. No training has started.

### 2026-11-24 21:14 EST — Measure the math baseline and repair missing pass metadata

The valid S0 math16k session contains **12 confirmed loops / 512 prompts (2.34375%)**, zero unassessed responses, UB12, and **390 native full-credit passes (76.171875%)**. Correctness unavailable/error counts are both zero after the native adapter repair. Length stops: 108/ 512; open thinking at cap: 107/ 512. Response lengths p50/p90/p99: 3142.5/16384/16384. The paired candidate math gates are UB ≤ 6 and at least 365 passes.

The original native checkers emitted scores but left `verification.passed` null, including successful score 1 checks. Source `4ff39edb2e` makes the Verifyit adapter preserve full-credit passes and adds an explicit inclusive `StdoutReward.pass_at_least=1` contract for campaign script checkers. Partial credit is a non-pass. Missing flags count as correctness-unavailable in the auditor. The original checker outputs and generation dumps remain immutable. The CPU replay ran the repaired native adapters on those exact saved scores, changing only the absent pass flag. It did not regenerate responses or rerun checkers. Future campaign tasks declare the same threshold, so baseline and candidates use one correctness contract.

Completed audit: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/audits/s0-v4/math16k-native-pass.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Faudits%2Fs0-v4%2Fmath16k-native-pass.json)). SHA-256: `ca7eb185f6efd51a0465913a9b4af710cc3dfcd23e05f06294e1a382b0f6082a`. The [frozen v2 manifest](cohorts-manifest-v2.json) digest and all 512 prompt hashes match. Source-to-derived dump hashes and the precise replay operation are in `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/audits/s0-v4/math-native-grades-v1/receipt.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Faudits%2Fs0-v4%2Fmath-native-grades-v1%2Freceipt.json)).

Math generation completed at 20:52:53 EST; the greedy 32K panel began immediately afterward. GPU coordinator: `/benfeuer/users-benfeuer-checkpoints-mochi-doom-ftpo-doom-baseline-cc75b435e5-in-7985a1b9-2026.11.24.4-coordinator-6bbb3801`; child suffix `a8f0fc0dbf9a`. CPU audit `/benfeuer/mochi-doom-audit-math16k-20261124a` succeeded (55.73s). CPU metadata replay/audit `/benfeuer/mochi-doom-replay-math16k-20261124a` submitted at 21:12:22 EST and succeeded (31.71s), zero failures/preemptions. Native grade tests: 61 passed; audit tests: 16 passed; all 768 frozen panel tasks preserve their native prompts under the new pass declaration, and 2304 local adapter replays preserve every field except `passed`. Required lint/type checks passed.

The 32K greedy and sampled panels remain in progress. This is a partial baseline milestone; sparse-domain reserve decisions and training wait for the full measurement.

### 2026-11-24 22:24 EST — Remove wall-clock censoring from the 32K measurement

The first complete 32K greedy panel has **221 unassessed responses / 768**. These are model-operation `AgentTimeoutError` failures. The inherited canonical TaskSpec session sets a 1,800-second model-turn deadline; long or queued requests hit it before the 32K token limit. This is a censored attempt, excluded from domain qualification.

| Excluded 32K cohort | Unassessed / 128 |
| --- | ---: |
| math | 26 |
| IF | 2 |
| coding | 100 |
| reasoning_gym | 38 |
| structured outputs | 11 |
| MCQ | 44 |

Diagnostic audit: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/audits/s0-v4/initial-greedy.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Faudits%2Fs0-v4%2Finitial-greedy.json)), SHA-256 `a5cc8e8e4b1d21744cce98daa82252d73022d64a44f4ec5384a47d954ad2199a`. The original outputs remain intact. The completed 16K math cohort has zero unassessed responses, so its 12 confirmed loops and 390 passes / 512 remain the valid math baseline.

Source `1043557f45` explicitly removes model-turn, total-turn and attempt wall-clock cutoffs for these single-turn measurements. The generated-token limit remains 32,768, prompt allowance 2,048 and serving window 34,816. Verifier and cleanup deadlines remain 300 and 60 seconds. Two local rendering/runtime regression tests pass; native lowering and the actual cloud resolved configuration confirm these limits. The current native audit/pilot suite has 17 passes. Broader validation and unchanged host/dependency failures are recorded in the private execution status.

The owned v4 coordinator and child were cancelled with `iris-cancel-own` at 22:05 EST and confirmed killed. Its unfinished sampled panel is excluded. The new **128-GB200 production** `doom-baseline-32k` run was accepted at **22:20:44 EST**, runtime `2026.11.24.5`, with unchanged weights, tokenizer and cohort-v2 identities. It evaluates initial greedy, fixed-reserve greedy, initial temperature-0.7 and reserve temperature-0.7 profiles in one workload. Only domains with fewer than ten confirmed loops on the valid initial panel will include reserve rows in their final cohort.

New coordinator: `/benfeuer/users-benfeuer-checkpoints-mochi-doom-ftpo-doom-baseline-32k-757bfc9804-in-ac3b27d4-2026.11.24.5-coordinator-57b4baeb`. Child suffix `ae3b4421b775`, submitted at 22:21:30 EST. All 32 four-GPU replicas are running with zero failures/preemptions at the first probe. Cloud runtime receipt: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/diagnostics/2026.11.24/baseline32-v5-runtime-receipt.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Fdiagnostics%2F2026.11.24%2Fbaseline32-v5-runtime-receipt.json)).

No training has started. Domain qualification waits for this fixed-token rerun. The 48-hour clock remains anchored at November 24 17:46 EST.

### 2026-11-24 23:09 EST — Verify the fixed-token runtime and pilot safeguards

The first completed v5 trajectory reproduces its frozen native prompt hash exactly (`00083ab5d418cd06098d20c7d36cdb8d8e016583e3acc73a8e3e8211eec76e78`). The retained request confirms temperature 0, top-p 1, top-k -1, the 32,768-token cap, both Mochi stop IDs and unstripped special tokens. Receipt: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/diagnostics/2026.11.24/native-prompt-v5-check.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Fdiagnostics%2F2026.11.24%2Fnative-prompt-v5-check.json)). This bounded integrity check is not an incidence sample. At 23:03:53 EST the initial greedy profile had completed 281/768 prompts; the coordinator remained running without failures or preemptions.

Source through `9de9bd7ebb` includes the targeted advisory-review fixes. Shared protocol constants and separate generation/training builders preserve all six complete configuration snapshots exactly. The pilot and checkpoint selector now admit a loss of exactly five percentage points without binary-subtraction rounding, and missing stage state raises through the callback handler. Pilot state is persisted as a typed stage record. The native guard/audit suite passed 28 tests, the rendering/publication/selection suite passed five, and required lint/type checks passed. Qualified selections now declare their held-out exclusion URIs. Their actual family quotas, filtered pilot and effective saved-model panels will be bound only after the baseline is sealed.

### 2026-11-24 23:48 EST — Initial 32K greedy incidence, before reserve qualification

The valid initial 32K greedy audit completed at **23:42 EST on November 24**. All 768 responses are assessed; generation/grade errors and unavailable correctness flags are zero.

| Cohort | N | Confirmed loops / UB | Full-credit passes | Length stops | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| math16k (previous valid pass) | 512 | 12 | 390 | 108 | target |
| math32k | 128 | 6 | 99 | 25 | extend to 256 |
| IF | 128 | 2 | 123 | 2 | extend to 256 |
| coding | 128 | 30 | 26 | 97 | target |
| reasoning_gym | 128 | 23 | 61 | 29 | target |
| structured outputs | 128 | 11 | 59 | 11 | target |
| STEM MCQ | 128 | 37 | 48 | 41 | target |

The fixed reserve profile began at **23:42:10 EST**. Only math and IF will add reserve rows; unused reserves cannot change the other domains. Training waits for the sealed baseline and baseline temperature-0.7 measurement.

Audit: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/audits/s0-v5/initial-greedy32k.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Faudits%2Fs0-v5%2Finitial-greedy32k.json)); SHA-256 `e27eb2a63a3b7e5aed08ad6e363fdda52ce18f6ec49aca0e2512a484be561a4a`. Native metadata replay changes only pass flags on saved checker scores, with original/derived SHA receipts in `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/audits/s0-v5/initial-native-grades-v1/receipt.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Faudits%2Fs0-v5%2Finitial-native-grades-v1%2Freceipt.json)).

GPU coordinator: `/benfeuer/users-benfeuer-checkpoints-mochi-doom-ftpo-doom-baseline-32k-757bfc9804-in-ac3b27d4-2026.11.24.5-coordinator-57b4baeb`, child suffix `ae3b4421b775`. Initial 32K generation took **3,797 seconds** after startup. CPU audit `/benfeuer/mochi-doom-audit-initial32k-20261124b` submitted **23:45 EST**, succeeded in **41.34 seconds**, zero failures/preemptions. Source `9de9bd7ebb`; full repository lint and focused native guards/audit tests pass.

### 2026-11-25 00:54 EST — Sealed greedy baseline and target curriculum

The **greedy baseline is sealed**. All seven cohorts have zero unassessed responses, unavailable correctness flags and errors. The fixed reserve changes math32K to 13 loops/256 and IF to 6/256; IF is non-target under the declared rule.

| Cohort | N | Baseline confirmed / UB | Baseline passes | Candidate UB max | Candidate passes min |
| --- | ---: | ---: | ---: | ---: | ---: |
| code32k | 128 | 30 | 26 | 15 | 20 |
| if32k (non-target) | 256 | 6 | 242 | 9 | 230 |
| math16k | 512 | 12 | 390 | 6 | 365 |
| math32k | 256 | 13 | 202 | 6 | 190 |
| mcq32k | 128 | 37 | 48 | 18 | 42 |
| rgym32k | 128 | 23 | 61 | 11 | 55 |
| struct32k | 128 | 11 | 59 | 5 | 53 |

The pooled 32K panel is **1,024 prompts**, with **638 baseline passes** and a candidate minimum of **618**. Every domain must satisfy both its loop and correctness limits on identical prompt hashes; math16K is evaluated separately. The temperature-0.7 baseline profiles are still running, and no training has started.

Baseline audit: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/audits/s0-v5/baseline-greedy.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Faudits%2Fs0-v5%2Fbaseline-greedy.json)), SHA-256 `b0a9f978726d48bcade59c3ab852965b0733b2b99b8bc777a7bbeb4d81f15892`.

Qualified selection version **2026.11.25.1** publishes 16,879 math and 29,334 mixed training prompts, a filtered pilot of 128, and the effective 1,024-prompt 32K panel. All 1,792 held-out and reserve identities remain excluded from training. Mixed batches have 64 prompts: math21, coding11, reasoning_gym11, MCQ11, structured10; structured training balances calendar/closure at 1,307 each. IF remains a saved-model regression check.

The fallback calendar population uses numeric FileReward graders. Source `18cdab53c1` declares exact full credit through the native exclusive threshold and shares its native parser with replay. All 1,792 prompt hashes are unchanged; original checker scores and generation evidence remain intact. The first seal job failed on the unsupported adapter before audit/publication; its partial replay outputs are excluded. The corrected receipt is under `audits/s0-v5/reserve-native-grades-v2/receipt.json`.

CPU `/benfeuer/mochi-doom-seal-baseline-20261125b`, submitted **00:42:25 EST November 25**, succeeded in **3m24s**, zero failures/preemptions. The greedy reserve profile finished **00:28:18 EST** (2,768s), and sampled initial began **00:28:22 EST**. GPU coordinator remains `/benfeuer/users-benfeuer-checkpoints-mochi-doom-ftpo-doom-baseline-32k-757bfc9804-in-ac3b27d4-2026.11.24.5-coordinator-57b4baeb`, child `ae3b4421b775`.

Source `311dc5c211` binds the qualified selections, exact quotas, filtered pilot and effective panels. All 16 presets pass native composer/driver preflight; focused suites pass 32 native guard/audit/replay, 61 script-grader, and six cohort/gate tests. Planning receipt: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/diagnostics/2026.11.25/qualified-config-receipt.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Fdiagnostics%2F2026.11.25%2Fqualified-config-receipt.json)). The two-update 32K S1 smoke is ready with 112 GPUs and temporary output storage; submit after the owned S0 allocation finishes.

### 2026-11-25 01:30 EST — Finalize quota provenance and smoke observability

The qualified data version is now **2026.11.25.2**. The earlier math-only summary recorded the campaign's mixed quotas even though the resolved math preset correctly used 64 math prompts. The new immutable version records the math-only quota as 64, preserves the mixed 21/11/11/11/10 quotas, and distinguishes its own recipe version from parent cohort version 2026.11.24.2. Training prompts, native grader declarations, held-out identities, effective cohort hashes and counts are unchanged. The old publication remains intact and is superseded before any training.

CPU `/benfeuer/mochi-doom-qualify-targets-20261125c`, submitted 01:03:58 EST, succeeded in 2m34.8s. Canonical summary SHA-256: targets `f5f655d7acaff0801e89f72dbb362219388a3ca6bc9f2412ac2fb4c8feeab357`; math `8ca742f5833c606e25fba28238b25f3f54afb71593f94ba152e4ca12f59ca981`.

Source `20ee97b685` explicitly enables console, durable records and W&B for training. All twelve qualified presets pass native composer/driver preflight at version 2026.11.25.2; the four unchanged baseline presets retain their passes. The resolved run specifications match each published selection's exact quotas. Planning receipt: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-doom-ftpo/diagnostics/2026.11.25/qualified-config-receipt-v3.json` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-doom-ftpo%2Fdiagnostics%2F2026.11.25%2Fqualified-config-receipt-v3.json)). S1 preflight confirms 28 four-GPU hosts, 112 GB200 at production priority, and an explicit ttl=3d output root. No training has started.

The initial temperature-0.7 baseline profile finished at 01:11:29 EST in 2,587 seconds. Its fixed-reserve profile began at 01:11:34 EST and has completed 585/768 prompts by 01:29:32 EST, without job failures or preemptions. S1 submission waits for this final owned baseline profile and its output verification.

## Final decision

The greedy baseline and target set are sealed. Baseline sampling is still running; no intervention or final candidate has been measured.
