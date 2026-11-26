# Which engine and admission configuration should the first real Mochi long-loop RL training run use, and what decode/rollout throughput does each candidate achieve on GB200s?

- Task issue — [marin-community/marin-logbooks#69](https://github.com/marin-community/marin-logbooks/issues/69)
- Status — Complete: the training config was selected (A10 geometry — EP16/DP16/TP1, seq64, HBM 0.85, batch 8192; 48 trainer + 16 rollout = 64 GB200) and the REAL Mochi RL training run is launched and verified past its first policy update (step 1 completed 2026-11-22 00:50:23.475 UTC; full-update record 6,890,762 tokens at 73.378 tok/s/rollout-GPU time-weighted, 0 preemptions/resets; correctness measured). Open questions for a future campaign: A12 offload layout, A13 Harbor-512 occupancy, the consumed-batch agentic share
- Research branch — [codex/69-mochi-throughput](https://github.com/marin-community/marin-logbooks/tree/codex/69-mochi-throughput)
- Notebook at commit — [Notebook](https://github.com/marin-community/marin-logbooks/blob/84c6681784280b596f6000dbc0daf5c9ed5a3951/logbooks/69-mochi-throughput/notebook.md)
- Marin contributions — Tracking issue [marin#11954](https://github.com/marin-community/marin/issues/11954); no code PR
