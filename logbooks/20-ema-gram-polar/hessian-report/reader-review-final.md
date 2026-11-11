# Final-candidate reader review

- Reviewer: independent reader agent `/root/report_reader`.
- Candidate: `output/pdf/hessian-experiment-report.md`.
- Final SHA-256: `feaafa1cab3160f0cde2830d4111fbbc2df3cdbdce54738593cd80a35e98a4e4`.
- Scope: manuscript read in order, including tables, figure caption, and appendices. No code, result archive, or later definition was used to supply missing earlier meaning. This is a comprehension and claim-connection check, not independent numerical or rendered-layout validation. No manuscript edits were made.

## Verdict

The central account is coherent and substantially self-contained: motivation → restricted measurements → geometric results and their acceptance limits → wider/fresh/local-step controls → a deliberately limited interpretation. Previous material ambiguities about attention acceptance, the reading route, and multiple meanings of alpha are repaired. The expanded results preserve the distinction between accepted, rejected, and numerically unresolved estimates. No central-result contradiction or missing definition that blocks the main interpretation remains.

**PASS for reader comprehension and claim-to-evidence exposition. No remaining material blocker was found.** The final revision supplies the missing token-to-target comparison and resolves the affected local comprehension. Remaining suggestions below are optional accessibility improvements, not a request to remove scientific content. The full-candidate review was followed by a check of the revised passages and their dependent interpretation against the exact final SHA above.

## Verified final repairs

- **Final whitespace-only check:** Restoring one trailing space after “The observed difference was 0.008272.” in memory exactly recovers previously approved SHA `e3deb3de83d77068a98f57e1fcb00cc4b931c8c568ae0e8c19c2b272629bbe00`. No other manuscript bytes changed. The comprehension PASS therefore applies to the final SHA above. The manuscript was not modified by this verification; PDF table layout remains outside this reader review's scope.
- **The token advantage now has an explicit comparison.** Section 1 states, “The first recorded evaluation at or below C4 loss 4.0 occurred after 33,000 updates for α = 1/8 and 34,000 for α = 0.” It gives 1.081 versus 1.114 billion training tokens, defines a token, and says the comparison does not interpolate between evaluations. The reader can now distinguish the observed target-specific token advantage from the final-loss difference. The separate hardware-layout caveat correctly limits interpretation of elapsed time. This resolves the material gap in the preceding candidate.
- **The leakage appendix now specifies the wider comparison.** “For the wider-output comparison, Q still contains the original twelve input directions, while P projects onto all 48 output directions.” It explicitly says the wider P is built from the expanded basis rather than Q alone. This matches the fixed-input comparison in Sections 7–8 and makes the formula's adaptation recoverable locally.
- **The quarter-power MLP table now identifies its first column as “Directions.”** The preceding paragraph identifies the model block; the column and rows now communicate the changed quantity directly.
- **Fresh-history sampling is described without implying statistical independence.** “The two fresh entries correspond to separately randomized, disjoint sets of complete sequences. They are not independent training runs.” This is consistent with the earlier whole-sequence disjointness explanation and the stated limits on inference from two histories.

## Evidence that the central story is recoverable

| Reader question | Exact evidence and interpretation |
|---|---|
| What is the question, and is it answered? | The first sentence states, “These measurements do not yet establish why one-eighth power gave lower final evaluation loss.” Section 9 ends by identifying the missing causal training comparison. The title is therefore qualified immediately. |
| How do the three alpha meanings differ? | “Training alpha is the setting used to produce a saved model.” “Fit-derived alpha is a value calculated afterward from a curvature measurement.” Later, “Candidate alpha is the value tried for one hypothetical optimizer step.” These definitions now precede the relevant comparisons. |
| What was measured? | Section 2 names the two matrices, the three original checkpoints, the selected Gram directions, and the full 12 × 12 projected interaction matrix. “Only the named matrix was allowed to change when its Hessian was differentiated.” This prevents a whole-model-Hessian interpretation. |
| Why would the proposed patterns suggest one-eighth? | Section 1 supplies α = ρ/(2r) before the prediction table. Section 3's example with curvatures 1 and 4 and proxy values 1 and 256 explains the fourth-power case. The direct grid is explicitly a separate geometric calculation. |
| What does acceptance establish? | Section 7 calls the precision test an “empirical agreement screen” and says it is not a rigorous shared-error bound. Section 8 distinguishes an unresolved negative sign from a certified negative-curvature conclusion. Bootstrap acceptance is explicitly conditional. |
| What changed in the wider experiment? | “Leakage is compared using the same original twelve input directions and the same full-response denominator.” The later leakage subsection repeats that only the output representation expands. This makes the comparison interpretable. |
| What do fresh histories show? | The distance is defined before its table; the fixed weights and common coordinates are explicit. “Two fresh histories do not assign a variance to each cause or prove that drift is the sole cause.” This keeps the descriptive comparison separate from causal attribution. |
| Were fresh histories used for the local optimizer tests? | “These update comparisons use the saved optimizer history, updated with that one gradient.” The final fresh-fit paragraph repeats the distinction when it becomes consequential. This repetition is useful. |
| What do the local loss numbers mean? | Loss change, its sign, the ×1,000 scale, native size, norm matching, and conditional bootstrap scope are introduced before interpretation. The report distinguishes less deterioration from improvement and an undistinguished difference from equivalence. |
| What survives in the conclusion? | Section 9's five numbered answers correspond to actual result subsections. It identifies selected-direction geometry, leakage, history dependence, and update-size dependence without claiming a new training ranking. |

## Optional accessibility refinements

1. **Add a short Section 8 navigation sentence.** This is now a long section with trajectory, local-update, wider-space, and fresh-history results. A one-line map at its start would reduce navigation effort without shortening any result: trajectory checks first, then local steps, wider measurements, and fresh histories.
2. **Keep displayed result precision proportional to its job.** The six-decimal leakage entries help show very small changes and should remain. In contrast, the two eight-decimal norm-matched mean loss changes make the reader compare tiny digits immediately before the interval says the effects are not distinguished. A rounded display plus the paired interval, with full precision retained in the linked CSV, may be easier to read. This is optional; the current statement is not misleading.
3. **Split the fresh-history data paragraph.** The paragraph beginning “The two histories use separately randomized, disjoint text windows” covers disjointness, hash exclusions, recording, common coordinates, and off-diagonal meaning. Separate the sampling/checking explanation from the coordinate/interactions explanation. Preserve both.
4. **Gloss AdamW at its first prose use in Section 8.** The sentence about norm matching introduces this optimizer name before Appendix A defines it. The relevant point is that the rescaling includes parameters using a different optimizer rule. A brief local gloss or that ordinary-language phrase avoids needing the optional appendix for the caveat.
5. **Separate provenance accounting from findings.** The 8,392 unique sequences and 25 rejected batches, and the final total HVP/history/branch count, are legitimate reproduction details. A distinct note or short paragraph would let a reader follow the scientific conclusion without mentally mixing counts of sequences, gradients, HVPs, and experiments. This is a layout preference, not a recommendation to delete them.

## Content to preserve

Preserve the complete motivating comparison; the explanation of r, rho, and fit-derived alpha; the restrictions to two matrices and selected directions; the distinction between geometric and actual optimizer operations; the negative-sign precision caveat; the failed bootstrap counts; the fixed-input leakage comparison; the native versus matched-norm results; and the separation of fresh-history geometry from saved-state optimizer steps. These are essential to understanding the conclusions. Blanket shortening would weaken this report.

## Review boundary

The seven completed follow-up experiments are treated as reported results, not independently verified here. The figure's caption is comprehensible; its rendering and PDF pagination were not part of this manuscript reader check. This verdict applies only to the SHA above.
