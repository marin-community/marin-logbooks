# Archive of the Snowball long-loop prototype (marin#11120)

Task issue: [marin-logbooks#49](https://github.com/marin-community/marin-logbooks/issues/49).
Campaign notebook: [codex/49-snowball-long-loop](https://github.com/marin-community/marin-logbooks/tree/codex/49-snowball-long-loop).

## 2026-11-19: Archive the prototype source tree

This branch preserves the complete source tree of the long-loop prototype PR so it stays available permanently, whatever happens to the PR.

- Source: [marin#11120](https://github.com/marin-community/marin/pull/11120), branch `benfeuer/long-loop-prototype`.
- Exact Marin commit: [`0dc178c550`](https://github.com/marin-community/marin/commit/0dc178c5504dc5dd475962e326cc3068837e4497). This head includes the merged SFT checkpoint-retention change ([marin#11719](https://github.com/marin-community/marin/pull/11719)).
- Archive-only change: inherited `.github/workflows` files are removed, as the logbooks workflow requires. No other file differs from that commit.
- The long-loop code lives under `experiments/post_training/long_loop/`. The campaign results, decisions and run records are in the campaign notebook above, not here.

This branch is a snapshot. Record later prototype revisions as new entries with their exact commits, merged in with the procedure in [WORKFLOW.md](https://github.com/marin-community/marin-logbooks/blob/main/WORKFLOW.md). Do not rebase or force-push.
