# Logbook Guidelines

Marin's
[`research-engineering`](https://github.com/marin-community/marin/blob/main/.agents/skills/research-engineering/SKILL.md)
workflow defines the research process. Follow [WORKFLOW.md](WORKFLOW.md) for
repository-local storage and Git mechanics.

All content is public. Do not publish secrets, credentials, personal data,
private results, or links that disclose restricted resources.

## Default branch

Keep `main` as an index, templates, and repository guidance. Research source
trees live on Marin-derived branches. Add index entries through PRs against
`main`; do not merge research branches into it or rewrite its existing history.
Existing `marin/*.md` records remain at their current paths.

## Research branches

Start from an exact Marin commit and use `codex/<issue-number>-<topic>`. Store the
notebook and supporting analysis in `logbooks/<issue-number>-<topic>/`. Copy
[templates/research-AGENTS.md](templates/research-AGENTS.md) into that directory
as `AGENTS.md` before publishing. This root file is absent on Marin-derived
branches; Marin's root `AGENTS.md` still governs their source tree.

Preserve published commits. Fetch and merge remote changes before pushing;
never force-push, rebase published history, or overwrite another writer's work.
Use explicit fetch and push refspecs. Follow the concurrent-writer procedure
when a push is rejected.

Remove inherited workflows before the first research push and after upstream
merges. Research CI requires explicit opt-in; do not change repository settings.
Archive-only workflow removals must stay out of Marin contribution PRs.

Publish notebook and supporting evidence commits before posting issue updates.
Use commit-pinned links. Change only the task assigned to you unless a broader
edit is requested. Correct material factual errors with new notebook entries;
small formatting and broken-link fixes may edit an existing entry.

Use concise commit subjects such as `Record #123 curriculum comparison`. Do
not include attribution or user-identifying information. Agent-authored issues
and PRs need the `agent-generated` label; agent comments begin with `🤖` unless
their exact text was approved by the user. Never merge a PR without explicit
authorization.
