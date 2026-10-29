# Reproduce the Snowball comparison

The [protocol](protocol.json) fixes the study. The [notebook](notebook.md) records
decisions and failures. [runs.json](runs.json) identifies actual attempts and
whether each qualifies for inference. Queued or incomplete runs are not results.
The experimental code lives on this Marin-derived research branch; the product
PRs contain none of these drivers.

Launch from the repository root with the normal Marin credentials and a valid
`TOGETHER_API_KEY` in the environment. Use the pinned repository environments.
The driver sends required credentials through the existing Iris mechanism;
published configuration and results contain no secret values.

```sh
uv run --frozen python -m experiments.post_training.sc_default_study \
  --version 2026.10.28.3 --arm tis --seed 1701 --full --cluster cw-rno2a --run
```

Omit `--run` to print the artifact graph. `--smoke --arm sc32 --seed 1701`
selects the small Qwen qualification. Each new training attempt needs a fresh
artifact version. An export-only recovery reuses its pending request with a new
Iris job name. Complete the first Grug baseline before submitting the other
predeclared arms and seeds. Check the shared resource board and both H100 pools
before submissions or expansions. All Grug gangs use batch and the same region.

The canonical source, generated split, tokenizer, model revision, graders and
package lock are pinned in the protocol and frozen-data evidence. The data
artifact contains `evaluation-index.json`. Preserve its ordering: evaluation
UIDs are loader row indices. Its `source` names the curriculum import; evaluation
dumps also record the upstream dataset, which several imports can share.

Collect raw saved verdicts into private scratch outside Git. Set `study_results`
to that scratch directory and put a copy of the published run manifest there.
Obtain the exact evaluation index from the recorded data artifact. For each
complete run, use its recorded export URI and manifest `collected` subdirectory:

```sh
PYTHONPATH=vendor/skyrl/skyrl-gym:. uv run --frozen python \
  logbooks/17-sc-default-study/collect.py \
  "$study_export" "$study_index" "$study_results/collected/tis-seed1701" \
  --eligibility logbooks/17-sc-default-study/eligibility.json
```

The collector requires complete admitted graded coverage and keeps missing grades
explicit. A live unreadable verdict writes a pending marker while training and
export continue. Before admitting the run, retry only its missing judge verdicts
with the recorded model, endpoint and original task contract:

```sh
uv run --frozen python -m experiments.post_training.sc_default_regrade \
  "$study_export" "$study_tasks" --step 50
```

Here `study_tasks` is the frozen validation Parquet. Repeat for a missing grade
at update 0 or 25, then collect again. Recovery preserves the original failure
and stops at the first valid verdict. The collector verifies the answer and
contract hashes; it cannot replace a valid grade. Non-judge failures require
their original machine recovery. The saved answers are never regenerated for
grading. The collector calculates full-task success from
the existing normalized verifier score, not from shaped optimization rewards.
Every curve uses the same 1,000-task subset, including at update 50. The complete
9,991-task quality endpoint supplies primary inference. All 10,000 original
answers must still be generated. Timings retain normal overlap;
phase totals must not be summed.

Use `--uids` to retry particular saved answers while investigating a different
grader failure. This selects recovery work only; it does not relax the analysis
coverage requirement or exclude tasks from the panel.

If object-store writes are unavailable, add `--local-manifest /absolute/path/to/recovery.json`
to the same grade-only command. Each first valid verdict is atomically saved and
flushed to local disk; no remote manifest is written. Keep this private file and
its hash. Repeating the command validates and reuses recorded verdicts.

After a storage write/readback probe passes, repeat with the same local manifest
and `--publish-staged`. This validates every staged saved-answer hash, contract
and judge configuration before uploading the manifest. It makes no additional
judge calls. Existing valid verdicts cannot be replaced. Then run the ordinary
collector against the remote export as usual.

Use `--recover-score-line` for a missing judge verdict whose reply puts one
trailing `SCORE` field on the same line as its explanation. This recovery-only
reader passes that field to the original parser and retains its allowed numeric
scores. It rejects ambiguous fields, extra trailing text and out-of-range
scores. Use the same fallback for missing verdicts in any arm. Already valid
grades, prompts, judge configuration, aggregation and live training graders
remain unchanged.

The contract audit checks local schema references and replays the original
date-sensitive Reasoning Gym generator without calling its scorer or using
model answers:

```sh
PYTHONHASHSEED=0 uv run --frozen python logbooks/17-sc-default-study/grader_audit.py \
  "$study_tasks" "$study_results/grader-audit.json"
```

Romain approved the common nine-task exclusion after the first TIS baseline and
before any SC run or outcome. The fixed [eligibility file](eligibility.json)
identifies two unusable schemas and seven date-sensitive generator contracts.
It applies to every arm and seed, including any excluded task that happens to
receive a valid grade. The collector verifies the original index and contract
identities, preserves excluded rows separately, and requires a valid verdict for
every retained task. Reasoning Gym retains 1,993 tasks and structural retains
1,998; the other families retain 2,000 each. Each family still has weight 0.2.
The fixed 1,000-task curves, training inputs and all cost measurements are
unchanged. Omitting `--eligibility` retains the original strict coverage gate;
such a collection cannot enter the amended comparison.

After auditing all nine initial runs, assemble and analyze their compact matrix:

```sh
uv run --frozen python logbooks/17-sc-default-study/assemble.py \
  "$study_results/runs.json" "$study_results/initial.npz"
uv run --frozen python logbooks/17-sc-default-study/inference.py \
  "$study_results/initial.npz" "$study_results/initial-analysis.json"
uv run --frozen python logbooks/17-sc-default-study/plots.py \
  "$study_results/runs.json" "$study_results/initial-analysis.json" "$study_results/figures"
```

Retain that initial matrix and analysis separately if a justified extension is
declared. Pass the explicitly chosen seed prefix with `assemble.py --seeds` for
the second and final analysis. Publish the finite extension and its estimated
cost before launching it. Nominal 95% intervals are reported; the two-look
adjusted 97.5% intervals determine the predeclared decision gates. Coverage is
per metric and comparison, not joint across every result. A remaining
inconclusive outcome is valid after the bounded work.

Keep generated answers, ordinary metrics, resumable checkpoints and exports in
their recorded object stores. Publish compact matrices, summaries, plots and
checksums with exact source/configuration and restore references. Preserve
failed attempts in resource accounting. Clean only owned temporary bulk after
all consumers stop and preservation has been verified.

Verify each native update-50 export with the run root and source revision from
the run manifest and the exact initializer URI recorded in
[export-verifier-replay.json](export-verifier-replay.json):

```sh
uv run --frozen python logbooks/17-sc-default-study/verify_export.py \
  "$study_run_root" "$study_initial_model" "$study_run_source" \
  "$study_results/export-verification.json"
```

This requires a completed native export and all 39 indexed weight shards. It
compares one router tensor and one attention tensor with the initializer using
small byte-range reads. The replay check reproduces the original baseline
manifest, byte count and both tensor checks. It does not verify every parameter
or replace the Qwen check of live serving weights after sleep.

For an export-only recovery, use [the prepared plans](export-recovery-plans.json)
and the ordinary request-driven converter through `export_checkpoint.py`. The
input and derived configs are in `recovery-configs/`; each uses its existing
update-50 checkpoint and a fresh attempt directory. Recheck storage headroom
and both GPU pools, coordinate the batch allocation, and publish the exact
launch source before submission. Run with the normal SkyRL CPU environment
and `PYTHONPATH=vendor/skyrl:vendor/skyrl/skyrl-train:vendor/skyrl/skyrl-gym:.`.

Verify the successful recovery with `verify_export.py --terminal-uri <its-terminal-uri>
--expected-job-id <its-exact-job-id>`. Keep the failed attempt and its resource
cost. A new terminal record must identify the actual successful recovery;
completed training is not repeated.

Combine the published terminal allocation proofs with `allocation_accounting.py
<output.json> <proofs...>`. Include the historical reconstruction and each later
per-job proof exactly once. The combiner recomputes GPU-hours from the retained
clocks, rejects repeated jobs and attempts, and keeps preexecution quota
reservation separate. Active jobs enter only after their terminal proofs exist.
These reconstructed spans have the timing limits documented in their sources;
they are not exact provider billing or certified bounds.

After all measured consumers stop, inventory each owned scratch run and preserve
it on a CPU worker in RNO with `preserve.py inventory <source-prefix/>
<archive-prefix> --run-source-revision <training-commit>`, then
`preserve.py copy <immutable-manifest-uri> --execution-source-revision
<archive-tool-commit>`. Check the resource board and current storage headroom
before bulk I/O. The complete manifest records full content hashes and exact
restore paths. Cleanup requires verified preservation of the unchanged source
inventory and stopped consumers; this driver never deletes source objects.

Learning curves show elapsed time from the first rollout, including the
exclusive midpoint evaluation pause. The cost curves use the primary
training-cycle time with that pause excluded. Endpoint charts show nominal
95% intervals in black and the predeclared 97.5% decision intervals in gray.
Use the outer decision intervals when comparing with the fixed margins.

Give `publish_machine_grade.py` a separate local staging path for that machine
verdict. It stages the singleton machine result, then merges it into the remote
recovery manifest while preserving existing judge rows. Finish all judge
publication for the panel before adding its machine verdict. Keep the judge
and machine staging files separately for provenance.

The archive destination may be this session's owned namespace in RNO or East-02A.
For the other region, `preserve.py` streams through separate source and target
filesystems and verifies the full archived content. A source ending in
`<run>/<version>/checkpoints/` preserves the native checkpoint tree without
copying temporary attempt logs. Qualify the published tiny control with
`qualify_preservation.py <plan.json> <output.json> --execution-source-revision
<commit>` before the finite full-size pilot. The driver never deletes source
objects; any later cleanup still needs stopped consumers, unchanged inventory,
verified full-content preservation and exact restore references.

For restorable cleanup, use `cleanup_preserved_native.py plan` with the complete
archive URI, reviewed native inventory, current run manifest and actual archive
submission record. It freshly verifies the full archive identities, portable
model and terminal consumers, then writes a read-only selection. The separate
`remove` command rechecks those facts and removes only the covered 32 native
tensor shards, retaining metadata and the full archived restart state. Preserve
the intent/receipt and immutable complete manifest as exact restore references.
This is separate from disposing of unarchived optimizer state, which still
requires the explicit retention approval enforced by `discard_native_state.py`.
