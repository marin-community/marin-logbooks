# Cold-read responses

Disposition of cold-read notes for this record. The loop runs
`logbook-cold-read.sh` per the task brief; notes arrive as a report per round.

## Round 1 — PR #84, 2026-11-25, verdict NOTES

Reader report: `84-r1-20261125T184108Z` (private working directory; notes
summarized below). Scope: PR head `da357b1db4`, notebook `90b062782`. No
secrets or private dataset contents flagged.

| # | Note (summary) | Disposition |
|---|---|---|
| 1 | Proposal mixture presented without explaining the planner replaces it; report proposal vs resolved vs actual family shares; "shares of the SFT mass" mislabeled | Accepted. Correction entry item 1 (actual family-draw table, resolved 17,010-update plan, corrected label). |
| 2 | Epoch denominator is inventory (569,291) not the configured epoch; 2,100 updates = 12.35% not 11.8%; "each SFT source at most once" wording | Accepted. Verified ceil(435,448/(0.8×32)) = 17,010 updates, 544,320 draws; correction entry item 2; Known limits and Final decision updated. |
| 3 | `context_positions` incompatible with sample counts | Accepted. It is the planned-epoch field carried into the receipt (544,320 × 262,144 total); correction entry item 3. |
| 4 | `selected.json` claimed archived but absent; stop_reason/restored_step sourced nowhere; 20-hour budget vs measured time | Accepted. `configs/v5-selected.json` archived; stop reason `maximum_updates` (not the clock); budget window and finalization times stated; correction entry item 4. |
| 5 | "Same policy and scored rows" contradicts 417 vs 421 scored subsets; unsupported "spread one expects" claim | Accepted. Corrected to same policy, different scored subsets; the 1.52-point gap is descriptive with cause/significance undetermined. |
| 6 | V5 optimizer differences hidden behind unexplained reference (fresh moments, bank→Adam routing, inherited LRs) | Accepted. Verified against the long-loop README at `a3d4a7c2b0` and `inputs.json` (`initializer_mode: weights_only`); correction entry item 6. |
| 7 | Micro210 metric underdefined (sampling, pass criteria, limits, judge, dev/held-out, overlap) | Partially accepted. Baseline section now defines the row table, #10197 classification, pass@1 vs pooled mean, judges, and length stops from the public record. Declined in part: the 2026.11.23 judge identity is not present in the archived records (`judge: null`), and no train/eval overlap check exists for this record — both are now stated as unknown rather than inferred. Exact per-trial generation parameters beyond the sealed one-attempt-per-row table are not restated here; the linked issue comments carry the available detail. |
| 8 | Older-policy provenance oversimplified; .7/.9 unexplained | Accepted. Baseline now scopes 2026.11.20-together lineage separately (runtime `ea21a69e55`, recovery bundle `63a21236fb`, corrections `e1fd3bcf3a`, policy SHA `8f7ef8170fbe...`) and explains version suffixes. |
| 9 | Legacy release geometry/readiness unreconciled; position_accounting locator claim | Accepted. Correction entry item 8; Known limits rewritten (no padding receipt in this record). |
| 10 | "Ten smaller single-store families" miscount; block unit labeled steps | Accepted. Nine single-store families; one block = 64,064 sequences = 364 updates. Correction entry item 9. |
| 11 | Model identity and project vocabulary unintroduced; rack/GPU accounting | Accepted. Question section now summarizes the Mochi/hero architecture, defines hero/Harrier/long-loop/Exp3-V5, and states 704 vs 128 GPUs. Total/active parameter counts are not restated (none published in the sources consulted); noted in the record. |
| 12 | Reproduction commands missing; archive vs execution commit conflated; status inconsistent | Accepted. Final decision now has pinned invocations with fresh-version placeholders and separates the archive commit from the historical `source_revision 406dff47f2...`. Issue and index status updated with the corrected state. |

No notes were declined outright; note 7 is partially declined with reasons
above (missing facts are stated as unknown instead of inferred).
