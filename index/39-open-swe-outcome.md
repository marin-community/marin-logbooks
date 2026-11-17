# Does training only on solved Open-SWE-Traces trajectories beat a matched random subset for Qwen3-8B SFT?

- Task issue — [#39](https://github.com/marin-community/marin-logbooks/issues/39)
- Status — Closed 2026-11-17. No benefit at Qwen3-8B with 10,000 rows per arm and two seeds: SWE-bench Verified (200 tasks) 12 against 11 solves on 168 common tasks; held-out loss on resolved trajectories equal within 0.0005 nats per token; within-task win rate 0.50 [0.44, 0.56]. SWE-bench solve rates of 2–5% rule out only large effects. A Qwen3.6-35B-A3B repeat is planned in next-steps.md under a new issue.
- Research branch — [codex/39-open-swe-outcome](https://github.com/marin-community/marin-logbooks/tree/codex/39-open-swe-outcome)
- Notebook at commit — [Notebook](https://github.com/marin-community/marin-logbooks/blob/a0c600138a3f5412012d37ba0aef089367df32f4/logbooks/39-open-swe-outcome/notebook.md)
- Marin contributions — None
