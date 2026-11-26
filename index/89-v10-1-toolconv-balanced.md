# What does Mix V10 look like with the tool-conversion fixes and balanced reasoning applied (V10.1)?

- Task issue — [marin-logbooks#89](https://github.com/marin-community/marin-logbooks/issues/89)
- Status — Complete: sealed `release-complete-v10.1.json` (sha256 `81fab6977b6dc2acb904a9027f5c0e0adc062eee32a1d09e0dbf9a5a20a4759e`); 4,383 tool-converted rows replace their originals in the three math sources with V10's assistant-only supervision preserved, and the balanced-reasoning component enters at w = 0.005 per phase; 1,680 updates per mixture epoch @ 256×262,144
- Research branch — [`codex/89-v10-1-toolconv-balanced`](https://github.com/marin-community/marin-logbooks/tree/codex/89-v10-1-toolconv-balanced)
- Notebook at commit — [`3b06cf9b44`](https://github.com/marin-community/marin-logbooks/blob/3b06cf9b4465108fc610aab62aef9dd116bcd157/logbooks/89-v10-1-toolconv-balanced/notebook.md)
- Marin contributions — [marin#12284](https://github.com/marin-community/marin/pull/12284) (the Exp7 LLOM consumption view that applies all-position loss at training time)
