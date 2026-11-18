# Does a learned deletion mask on Snowball 67B-A2B expert projections recover full fine-tuning's Micro210 gain under the same data and budget?

- Task issue — [#63](https://github.com/marin-community/marin-logbooks/issues/63)
- Status — Active. **Base Micro210 row measured** (pooled 0.1327 / mean-of-benchmark-means 0.1596, 440/440 slots scored). Paper spec, evaluation contract, and planned-run table recorded; Deletion Tuning core + tiny-MoE tests public; the production SFT path was corrected to Levanter (Snowball model + shared SFT launcher) and the V1/V3 integration targets it; V2 targets Megatron/SkyRL. The full-fine-tuning equivalence hypothesis is marked untested for this phase unless a matched comparator exists.
- Research branch — [codex/63-snowball-deletion-tuning](https://github.com/marin-community/marin-logbooks/tree/codex/63-snowball-deletion-tuning)
- Notebook at commit — [Notebook](https://github.com/marin-community/marin-logbooks/blob/916541cf85962e9a9023199103150231bc59b4f6/logbooks/63-snowball-deletion-tuning/notebook.md)
- Marin contributions — None yet
