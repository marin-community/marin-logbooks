# What does Mix V10 look like with the full verified tool-conversion merge applied (V10.2)?

- Task issue — [marin-logbooks#94](https://github.com/marin-community/marin-logbooks/issues/94)
- Status — Complete: sealed `release-complete-v10.2.json` (sha256 `228929e63f6f7c54cde9f2d5058261f120592184e7163d82e9731a5f6e9a4094`) over the sealed V10.1 base (#89); 42,272 verified tool-converted rows replace their originals in the eight toolconv-delta sources with V10's assistant-only supervision preserved (rewritten shards in stores-v4, unchanged shards byte-identical V10/V10.1); the 40,374 same-name-rewrite rows stay byte-identical V10 rows as a NAMED design exclusion resolved in V10.3; 1,677 updates per mixture epoch @ 256×262,144
- Research branch — [`codex/94-v10-2-toolconv-merge`](https://github.com/marin-community/marin-logbooks/tree/codex/94-v10-2-toolconv-merge)
- Notebook at commit — [`a8a722842`](https://github.com/marin-community/marin-logbooks/blob/a8a722842fca294ce0c263a38f53de49b4899efb/logbooks/94-v10-2-toolconv-merge/notebook.md)
- Marin contributions — [marin#12284](https://github.com/marin-community/marin/pull/12284) (the Exp7 LLOM consumption view that applies all-position loss at training time)
