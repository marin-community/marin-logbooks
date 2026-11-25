---
title: "Comparing Claude Code and Pi on SWE-bench Science"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin/issues/12108"
related_issues: []
research_branch: "codex/12108-swebench-science-claude-code"
marin_base_revision: "d1e8497b47274a7c69dd0fe483983a6d07fc6468"
archive_template_revision: "fc1cb30da0967f6ab78ffd47444ed9144f50cba9"
status: complete
---

# Comparing Claude Code and Pi on SWE-bench Science

With Qwen3.8-27B on the same 118 tasks, Marin Claude Code solved 32, Marin Pi solved 23, and the authors’ Claude Code solved 35. Marin and the authors’ Claude Code agreed on 95 individual grades. The benchmark catalog uses Pi. The experiment also changed the available context and reply limits, so the nine additional solves over Pi cannot be attributed to the harness alone.

The experiment record follows the six-task pilot, the frozen 32-task checkpoint, and the remaining 80 tasks through the final grade and token audit. It includes the model revision, launch instructions, per-task disagreements, and recovery history. [Experiment record](README.md) · [Paired outcomes and token audit](README.md#all-eligible-tasks)
