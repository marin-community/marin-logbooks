# Which engine and admission configuration should the first real Mochi long-loop RL training run use, and what decode/rollout throughput does each candidate achieve on GB200s?

- Task issue — [marin-community/marin-logbooks#69](https://github.com/marin-community/marin-logbooks/issues/69)
- Status — Active: sprint in flight; best real-rollout arm A3 (inflight128 + staleness11 + seqs32 + HBM 0.85) at 127.22 tok/s/GPU first-25min (2.28x the v103 control); best pure-decode arm EP16/seqs64/shared8 at 552.15 tok/s/GPU; real RL launch planned 2026-11-21 ~19:45 UTC
- Research branch — [codex/69-mochi-throughput](https://github.com/marin-community/marin-logbooks/tree/codex/69-mochi-throughput)
- Notebook at commit — [Notebook](https://github.com/marin-community/marin-logbooks/blob/9302ae7272487479924d8eb634ce6e76ed488db4/logbooks/69-mochi-throughput/notebook.md)
- Marin contributions — Tracking issue [marin#11954](https://github.com/marin-community/marin/issues/11954); no code PR
