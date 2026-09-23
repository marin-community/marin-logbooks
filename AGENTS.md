# Logbook Guidelines

Use this repository for experimental programs expected to require repeated
launches or controlled comparisons and produce evidence worth reusing. Default
to a logbook for kernel tuning across configurations or shapes, evaluation of a
new modeling or training idea, and systems or evaluation experiments with
several meaningful runs. Also use it when the user explicitly requests a public
logbook or provides an existing one here.

Do not start a logbook for ordinary implementation, debugging, incident
response, a single reproduction or smoke test, or one-off validation. A task
lasting multiple sessions does not by itself need a logbook. Start one if the
work later becomes a qualifying experimental program.

Create a logbook by default when qualifying work has a public coordinating
issue. If the work is not already public, suggest the logbook and confirm the
publication surface before writing here. Do not create a coordinating issue
without explicit authorization.

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
