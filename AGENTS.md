# Logbook Guidelines

The Marin
[`research`](https://github.com/marin-community/marin/blob/main/.agents/skills/research/SKILL.md)
skill is authoritative for when to create a logbook, what evidence belongs in
it, and when to update the coordinating issue. Follow this file for
repository-local write mechanics.

All content is public. Do not publish secrets, private results, credentials,
personal data, or links that disclose restricted resources.

## Files

Store each Marin logbook at `marin/<issue-number>-<topic>.md`. Change only the
logbook assigned to the current task unless the user requests broader edits.

Keep the entry history append-only. Correct a material factual error with a new
entry that identifies the correction. Small formatting and broken-link fixes
may edit an existing entry.

## Git

Fetch and rebase before pushing. Never force-push or rewrite published history.
Stop and resolve unexpected changes to the same logbook instead of overwriting
them.

Use concise commit subjects such as `Update #1234 BF16 logbook`. Do not include
agent attribution or user-identifying information.
