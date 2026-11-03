---
title: "Agentic SFT shootout on Snowball"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin/issues/10213"
related_issues: ["https://github.com/marin-community/marin/issues/10197", "https://github.com/marin-community/marin/issues/9920"]
research_branch: "codex/10213-sft-shootout"
marin_base_revision: "fca974b90bea247cf640e9cb17b315ed3a89e89d"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: complete
---

# Agentic SFT shootout on Snowball

This archive holds the final write-up of the experiment. The experiment ran from
2026-10-16 to 2026-10-24. Its code is in the Marin draft PRs listed below, not on
this branch; the branch only stores the paper.

## Question and decision criteria

- Goal: find which SFT method, if any, improves Snowball v0 and v0.1 on
  SWE-bench Verified and TBLite over plain SFT at equal supervised tokens,
  without losing GSM8K or BFCL-Parity.
- Alternatives: plain SFT, rejection-sampled SFT (RS-multi), relay, recovery
  and plateau dropping.
- Evidence: paired per-task bootstrap intervals on SWE-bench Verified random-100
  and TBLite-100.

## Final decision

- Result: no method beat plain SFT by enough to matter. Every arm trained on
  GLM 5.3 traces scored below its parent. On the same 1,432 tasks, the
  contributor traces (Kimi-2.5, Qwen3.6-27B) beat the GLM LOW traces by
  SWE-bench +26 [+17, +36] and TBLite +9 [+1, +17].
- Decision: stop and write up. The cross-format test that would separate the
  teacher from the harness format (about 125 H100-h) was not run.
- Paper: [agentic-sft-shootout.pdf](agentic-sft-shootout.pdf) (67 pages).
- Marin contribution PRs (drafts): #10251, #10252, #10253, #10255, #10258,
  #10324, #10325, #10327.
- Results by round: comments on marin-community/marin#10213.
- Remaining uncertainty: single runs on 100 tasks per screen; teacher and
  harness format are not separated.
