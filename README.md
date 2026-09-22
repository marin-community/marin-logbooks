# Marin Logbooks

This repository holds public, high-fidelity records for Marin research programs
that explicitly request a logbook. Logbooks record decision-relevant attempts,
including negative results, with enough detail to reproduce the evidence.

The coordinating GitHub issue remains the main public entry point:

| Surface | Content |
| --- | --- |
| Logbook | Decision-relevant attempts, commands, configurations, results, and interpretation |
| Issue comments | Milestones, material status changes, blockers, and decisions |
| Issue body | Current conclusion, baseline, confidence, and overall status |

Routine debugging, raw logs, and dense telemetry stay out of the logbook. Link
Finelog, W&B, commits, and other source artifacts when they support a result.

## Layout

Store Marin logbooks at `marin/<issue-number>-<topic>.md`. Start from
[`templates/logbook.md`](templates/logbook.md), and link the coordinating issue
in the frontmatter.

Read [`AGENTS.md`](AGENTS.md) before adding or updating a logbook.
