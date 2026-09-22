# Logbook Guidelines

Update this repository only when the user explicitly requests a public logbook
or provides an existing logbook here. A multi-session task alone does not
authorize creating one.

All content is public. Do not publish secrets, private results, credentials,
personal data, or links that disclose restricted resources.

## Record Structure

Use three levels of detail:

1. Keep decision-relevant attempts in the logbook.
2. Promote material milestones and status changes to comments on the
   coordinating issue.
3. Maintain the issue body as the current big-picture summary.

Store each Marin logbook at `marin/<issue-number>-<topic>.md`. Change only the
logbook assigned to the current task unless the user requests broader edits.

## Logbook Entries

Append an entry when an attempt tests a real hypothesis, produces a meaningful
result or negative result, changes the baseline or next action, or invalidates
earlier evidence. Include the source revision, exact command or material
configuration, hardware when relevant, result, interpretation, and next action.

Omit command mistakes, ordinary environment repair, transient operational
noise, evidence-free retries, and debugging whose only outcome is that the code
runs. Link raw logs, dense telemetry, and large tables in their source systems.

Keep the entry history append-only. Correct a material factual error with a new
entry that identifies the correction. Small formatting and broken-link fixes
may edit an existing entry.

## Publication Boundaries

Logbook entries may accumulate locally during active work. Before posting a
coordinating issue comment, commit and push every logbook entry that supports
the comment, then link the exact logbook commit from the comment. Also commit
and push before a cross-session handoff or when stopping work, even when no
issue comment is warranted.

One commit may contain several entries. Fetch and rebase before pushing. Never
force-push or rewrite published history. Stop and resolve unexpected changes to
the same logbook instead of overwriting them.

Use concise commit subjects such as `Update #1234 BF16 logbook`. Do not include
agent attribution or user-identifying information.
