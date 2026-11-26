---
title: "Mochi 50-step RL data-mix campaign"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/91"
related_issues: ["https://github.com/marin-community/marin/pull/12329"]
research_branch: "codex/91-mochi-rl-data-campaign"
marin_base_revision: "46e0c1b502bb0467b98bae922552ece67d6d984e"
archive_template_revision: "3846d697465e8629294936c6a29e2061bb1c296d"
status: active
---

# Mochi 50-step RL data-mix campaign

## Question and decision criteria

- **Goal and why it matters:** find which RL data sources and mixtures improve Mochi most per RL step. The answer chooses the RL data mix for the next long-loop runs.
- **Baseline and alternatives:**
  - The baseline is the config and data of Mochi RL run `.131` (`long-loop-mochi-exp4-dspark-rl-2026.11.25.131`); its Micro210 seal comes from the Mochi lane.
  - The alternatives are 100 candidate mixes of 4–8 sources at equal weight, from the holiday Atlas burndown (marin #12329): v2 and v3, 50 each.
- **Evidence needed to choose:**
  - Micro210 score of each run's best-by-validation checkpoint (canonical policy, #10197 scoring).
  - Train reward over steps 1–10 vs 41–50.
  - Best validation step.
  - Signs of broken data, such as all-zero or all-one rewards, grader errors or infrastructure-absent rollouts.
- **Constraints:**
  - The RL config is fixed. Only the curriculum changes.
  - 50 optimizer steps per run, with no time cap.
  - 3 runs going at once, at production priority on the cw-us-east-08a GB200s, with 128 GPUs each (48 trainer + 80 rollout).
  - Zone free space on the cw-us-east-08a object storage stays above 15%. Below that, campaign checkpoints are pruned, always keeping each run's best and final checkpoints.
- **Rules for choosing mixes:**
  - Each mix has at least 2 SWE/agentic families and at least 2 competition-math families.
  - Each source has at least 500 tasks.
  - Sources saturated at the start of RL are left out: nemotron_if (0.97) and openscience (0.92).
  - The queue follows human ratings: High, then Medium, then Low; Skip never runs.
  - Mixes with a source whose stored grader predates fixes on main (#12329 review) run after all clean mixes.
  - Mixes run as listed, including sources Pending in the served R40 catalog, at their declared pins. Each run records which of its sources diverge from the catalog.

## Baseline

- Date: 2026-11-25
- Run: Mochi RL `.131` (Exp4 DSpark); step-34 best-by-val export; Micro210 eval child succeeded 2026-11-26. The sealed score will be added here when the Mochi lane publishes it.
- Config: DSpark (4 draft tokens), TP1/BF16 engines with CUDA graphs, 48 trainer + 80 rollout GB200, the .131 initializer, drain and rolling defaults.

## Entries

### 2026-11-26 13:30 UTC - Campaign start and launcher validation

- Question or hypothesis: set up a fixed-config, data-only RL ablation that is cheap enough to run about 50 mixes in a week.
- Published source revision: the campaign launcher landed at `a97c1a1198`. Setup fixes followed at `219b720287` (inherit the parent cluster for nested CPU stages) and `390c5e318b` (verify task_sha256 against the exact published JSON bytes, not canonicalized JSON).
- Configuration: the .131 recipe with a `--mix-id` argument that resolves sources through the Atlas catalog, `--max-steps 50` and no learning-clock callback.
- External inputs:
  - candidate mixes `mix_candidates_v2.csv` and `mix_candidates_v3.csv`, plus the source table `rl_sources_on_main.csv` (holiday burndown, #12329);
  - the saturation evidence table from the .129 and .131 per-route rewards.
- Result:
  - The launcher validation passed: native recipe and graph dry runs, plus 58 landing tests.
  - Six first-wave roots (.502–.507) failed before any GPU allocation on the two setup defects above. No training happened, and they are not counted as runs.
  - The first wave is relaunching with fresh versions.
- Evidence: `s3://marin-us-east-08a/marin/users/benfeuer/mochi-rl-data-campaign/preparation/2026.11.26.1/launcher-validation-final.json`.
- Interpretation: none yet.
- Next decision: when the first three runs reach 50 steps, compare their reward curves and Micro210 against `.131`.

## Run table

Updated as runs start, finish and get evaluated. The Grafana links use the stall-safe form, pinned to the job and execution, until marin #12292 merges.

| Mix | Priority | Status | Steps | Reward 1–10 | Reward 41–50 | Best val step | Micro210 | Catalog divergence | Links |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v2-mix-033 | High | relaunching | 0 | | | | | swesmith-v5 (Excluded in served R40; sealed retained release @758b28bf) | |
| v2-mix-009 | Medium | relaunching | 0 | | | | | | |
| v2-mix-011 | Medium | relaunching | 0 | | | | | | |
