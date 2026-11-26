# What is Snowball's maximum decode throughput at the 65,536-token request window, and what is the smallest Snowball RL job that keeps training saturated?

- Task issue — [#71](https://github.com/marin-community/marin-logbooks/issues/71)
- Status — Complete: the right-sized Snowball RL layout is chosen and measured — 64 GB200 (trainer 8×4 PP2/CP4/EP4 + 4 engines DP8/EP8 HBM 0.90, seqs 128; 16 rollout workers; gates open; staleness 2; quotas 22/42; cadence-3) at ~0.96 GPU-hour/update, 195.0 s/update with 13.4% rollout wait; both 32-GPU floors probed (3 engines regress; trainer-shrink does not fit). CAVEAT: GB200-only measurements; H100 confirming run open
- Research branch — [codex/71-snowball-resize](https://github.com/marin-community/marin-logbooks/tree/codex/71-snowball-resize)
- Notebook at commit — [Notebook](https://github.com/marin-community/marin-logbooks/blob/be4e5409bd34c8d20b21a2f5e29e2e00a3590d0a1/logbooks/71-snowball-resize/notebook.md)
- Marin contributions — Tracking issue [marin#11969](https://github.com/marin-community/marin/issues/11969); no code PR
