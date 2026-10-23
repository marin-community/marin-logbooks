# Marin Logbooks

Marin Logbooks preserves the experiments, analysis, and decisions behind Marin
research engineering work. Each task has an issue here, a runnable research
branch, and a notebook in that branch. Selected implementations become pull
requests in [Marin](https://github.com/marin-community/marin).

The task issue is the entry point: its body summarizes the current conclusion,
decisions, open questions, and links. Milestone comments explain important
attempts and results. The notebook records why an approach was tried, the code
and evidence used, what happened, and what was learned.

Marin's
[`research-engineering`](https://github.com/marin-community/marin/blob/main/.agents/skills/research-engineering/SKILL.md)
workflow owns the research process. This repository defines storage and Git
mechanics. Background literature research can use Marin's separate research
workflow.

## Repository layout

- `main` holds this guide, templates, and the [task index](INDEX.md). Its history
  stays separate from Marin source history.
- `codex/<issue-number>-<topic>` holds a research task, starting at an exact Marin
  commit. It contains the full working source tree and supporting work under
  `logbooks/<issue-number>-<topic>/`, including `notebook.md` and `AGENTS.md`.
- Each task issue lives in
  [marin-community/marin-logbooks](https://github.com/marin-community/marin-logbooks/issues).
  Related Marin issues and contribution PRs are linked from it.

Research branches share ancestry with Marin. This repository is not maintained
with `git push --mirror`: archived branches remain available as Marin advances.
Keep published research history; merge upstream changes when needed. Never
merge a research branch into logbooks `main` or replace `main` with Marin's
history. Add or update its index row through an ordinary PR against `main`.

Large datasets, checkpoints, and raw telemetry stay in their existing stores.
Record exact artifact references and access limitations in the notebook. All
published material must be suitable for public development.

Read [AGENTS.md](AGENTS.md) and follow [the branch procedure](WORKFLOW.md).
Existing `marin/*.md` logbooks remain historical records at their current paths;
new tasks use research branches.
