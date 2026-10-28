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
  "$study_export" "$study_index" "$study_results/collected/tis-seed1701"
```

The collector requires complete graded coverage and keeps missing grades
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
10,000-task endpoint supplies primary inference. Timings retain normal overlap;
phase totals must not be summed.

Use `--uids` to retry particular saved answers while investigating a different
grader failure. This selects recovery work only; it does not relax the analysis
coverage requirement or exclude tasks from the panel.

The contract audit checks local schema references and replays the original
date-sensitive Reasoning Gym generator without calling its scorer or using
model answers:

```sh
PYTHONHASHSEED=0 uv run --frozen python logbooks/17-sc-default-study/grader_audit.py \
  "$study_tasks" "$study_results/grader-audit.json"
```

The proposed nine-task exclusion requires approval and has not been applied.
The original complete-coverage gate remains in force while that decision is
pending.

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
