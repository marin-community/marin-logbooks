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

## Round 2 — PR #84, 2026-11-25, verdict NOTES

Reader report: `84-r2-20261125T185033Z` (private working directory; notes
summarized below). Scope: merged PR head `da357b1db`, notebook `8b77d27d`. No
public-safety exposures identified.

| # | Note (summary) | Disposition |
|---|---|---|
| 1 | PR review surfaces still lead to the superseded notebook: PR #84 body/diff pin `90b062782`; REST head `da357b1db` | Accepted — root cause: a maintainer merged PR #84 at 18:50:26Z, before the corrected index pin (`5ceaf7ad8`) landed on its branch, so `main` pinned the superseded notebook. Fixed by a follow-up index-record PR against `main` pinning the corrected notebook; PR #84 carries a pointer comment. The merged PR body itself is immutable. |
| 2 | pass@1 numerator underdefined for partial-credit benchmarks (MRCR mean-similarity etc.); state the predicate or mark it unavailable | Accepted — investigated: the aggregation is not in the public marin tree at the pinned runtime commit (`5ca48a7ee9`) or on `main`, and the archived records do not state it. Baseline now explicitly marks the pass predicate unavailable and points to per-benchmark verifier rewards in `record.json`. |
| 3 | Reproduction claims a fresh `--prefix` creates a fresh recipe version; `MOCHI_SFT` fixes `version="2026.11.10"` and `resolve_version` returns a supplied version unchanged | Accepted — Final decision corrected: the prefix isolates outputs, the version label stays `2026.11.10`. |
| 4 | Chronology unqualified (2026-11-24/25 vs older policy posted 2026-11-21); correction-entry timestamp 19:05 vs commit 18:50:09Z | Accepted — Baseline dates scoped per policy; entry headings now record the UTC commit time that published each entry (convention stated under Entries). |
| 5 | "Match to integer rounding" understates partial-block sampling; receipt is a deterministic sampler reconstruction | Accepted — round-2 entry item 1 refines the phrasing with the reader's examples and names the `basis` field. |

No notes declined in round 2.

## Round 3 — PR #85, 2026-11-25, verdict NOTES

Reader report: `85-r3-20261125T185744` (private working directory). The reader
checked the arithmetic and hashes and found no public-safety exposures.

| # | Note (summary) | Disposition |
|---|---|---|
| 1 | Qualify KV heads: 12 local-attention KV heads, 6 global-attention KV heads, 12 is the stored maximum | Accepted — Question section updated with the heuristic.py fields. |
| 2 | Preserve the full-SFT absence breakdown (12 infra errors + 5 unavailable) instead of asserting all 17 absent rows are failures | Accepted — verification entry and Known limits now carry each record's own breakdown; older runs' absences marked as not broken down. |

No notes declined in round 3.

## Round 4 — PR #85, 2026-11-25, verdict NOTES

Reader report: `85-r4-20261125T190516` (private working directory). The reader
verified the public scores against the eval posts, the receipt arithmetic
(968,596 SFT packs; 53,774 + 13,426 draws; 319,085,871,104 vs
17,616,076,800 packed-slot tokens, 18.11×; 2,100/17,010 = 12.35%) and the four
published artifact hashes.

| # | Note (summary) | Disposition |
|---|---|---|
| 1 | Entry-point pins lag: PR #85 body and issue #83 Status/Links/Conclusion still point to older notebook commits | Accepted — PR #85 body, issue #83 Status/Links/Conclusion, and the index record all re-pinned to the then-current head; dated milestone comments preserved as history. |

## Round 5 — PR #85, 2026-11-25, verdict NOTES

Reader report: `85-r5-20261125T191234` (private working directory). Scores,
coverage, source-copy checks, manifest hashes, exposure counts, the 12.35%
epoch fraction, and the ~18.1× token ratio verified; no other actionable
notes.

| # | Note (summary) | Disposition |
|---|---|---|
| 1 | Round-1 correction item 6 says warmup and decay are scaled; the fresh-optimizer code retains warmup at 5 and scales only decay to 210 | Accepted — dated correction entry added (warmup retained, decay scaled); item 6 carries a pointer note, original wording preserved. |



## Round 6 — PR #85, 2026-11-25, verdict NOTES

Reader report: `85-r6-20261125T192019` (private working directory). Scores,
source copies, hashes, exposure arithmetic, and the warmup correction verified;
no public-safety exposures; no settled limitations reopened.

| # | Note (summary) | Disposition |
|---|---|---|
| 1 | PR #85 body's notebook-link label reads `b4b8dc5ea` while its URL targets `001cb667c` | Accepted — label updated to `001cb667c`, URL unchanged (a replace-ordering slip in the round-5 body edit). |

## Round 7 — PR #85, 2026-11-25, verdict SATISFIED

Reader report: `85-r7-20261125T192729` (private working directory). "No
actionable notes." The reader independently verified all five headline scores
and coverage counts against the public eval posts, the four archived source
copies against their named commits, the four published metadata hashes, the
exposure arithmetic (319,085,871,104 vs 17,616,076,800 packed-slot tokens,
18.1133×; 12.3457% of the planned epoch), the family-draw totals, no-repeat
SFT accounting, the maximum-updates stop, and the warmup/decay provenance.
No dispositions required.
