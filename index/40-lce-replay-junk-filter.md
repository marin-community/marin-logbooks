# Does filtering junk from long-context replay help cold-start SFT?

- Task issue — [#40](https://github.com/marin-community/marin-logbooks/issues/40)
- Status — Decision: adopt filtered long replay for doing no harm, with no benefit found. At 238 steps against two control seeds, tool calling, code, held-out long-document loss and generative MRCR-65k (filtered 0.203 against 0.219 and 0.195) all match the controls. A teacher-forced needle-recovery lead for the filtered run did not carry over to generated answers.
- Research branch — [codex/40-lce-replay-junk-filter](https://github.com/marin-community/marin-logbooks/tree/codex/40-lce-replay-junk-filter)
- Notebook at commit — [Notebook](https://github.com/marin-community/marin-logbooks/blob/028da3f4b04b34ddb5ac85f7d8033263c3407001/logbooks/40-lce-replay-junk-filter/notebook.md)
- Marin contributions — None yet; candidates: junk filter, Snowball probe launcher, filtered replay builder
