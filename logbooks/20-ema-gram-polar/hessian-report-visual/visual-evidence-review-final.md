# Final scientific audit of the visual revision

Final inspected candidate: `output/pdf/visual_revision/visual-report.md`

Candidate SHA-256: `1a2f04e9d879a30dd01447fa82bd62a37e1ea81f4bce29d364f7f2b96076fee9`

Reviewed generator SHA-256: `9034f5afa2c6e3f2ac86be5f9065689b0f43236b619370fca558e8302f23bd3a`

Reviewed figure-data SHA-256: `12b124c7f63578261d9ef707e4a038b7226354085586d6e36fdf3fb010d71654`

Verdict: **Pass. No altered measurements, numerical results, acceptance thresholds, or unsupported scientific conclusions found. No outstanding scientific correction.**

This final audit includes the complete initial numerical audit recorded in `evidence-review.md` and a fresh reading of the entire final manuscript and generator. The unchanged figure-data and source checksums bind the initial verification to the final figures. This is an audit of existing evidence, not a rerun of the TPU experiments. No manuscript or scientific source data was edited.

## Initial verification retained

Verified all acceptance cells against strict accepted-positive-power flags and empirical-resolution status; all five accepted raw fitted powers; all four 12-to-48 leakage comparisons with the original twelve input directions and denominator fixed; all twelve raw 48-direction proxy-distance markers; and all optimizer and paired-contrast plotted rows against detailed aligned records or the CSV.

Independently regenerated all sixty loss-change means and all 120 conditional percentile intervals from raw sixteen-batch losses with 10,000 paired bootstrap resamples and the declared seed. Verified whole-update norm matching, including the complete optimizer output and step fraction. The distinction between fresh diagnostic histories and saved history used in optimizer branches is retained.

The exact scientific-results and optimizer-step CSV hashes remain:

- `scientific-results.json`: `a9418468d9c0bfb14f19f10ebf66c8c1e743bff6171448c4a1c2eb6dd0e0de3c`
- `optimizer-step-results.csv`: `99c9fe330f66bc3f5224e47f7003646c8f9f9920a7de22a7583834ab16e5f058`

## Final delta review

The added precision-margin explanation faithfully describes `experiments/ema_gram/curvature_resolution.py`: average the eight projected Hessians before fitting; symmetrize the mean; use its positive diagonal entries to define the same diagonal congruence scaling for the mean and two finite-difference estimates; take the spectral norms of all three pairwise discrepancies; compare their maximum with the minimum eigenvalue of the scaled mean. The required inequality is strict: minimum scaled eigenvalue greater than ten times the maximum discrepancy. The prose's “largest possible response to an input with norm one” correctly describes the induced spectral norm. The explanation remains an empirical agreement screen and explicitly does not exclude common systematic errors.

The revised leakage explanation correctly aggregates the squared entries across all averaged HVP responses, giving a Frobenius norm; it does not average separate per-direction ratios. The mean-before-fit clarification and deferred direct-grid discussion preserve the original method and conclusions. Basic definitions, MLP/block naming, and the explicit paired-contrast dot/interval caption introduce no changed result. The conditional scopes, lack of equivalence proof, fixed-state limits, single-seed origin, and uncertainty about the final training advantage remain intact.

Read the final plotting generator in full. Formatting, strict zip calls, source-header cleanup, SVG whitespace normalization, and figure layout changes do not alter scientific selection or numeric conversion. The complete figure-data file is byte-identical to the initially audited file.

Visually re-inspected final Figure 1 and Figure 3. Figure 1 now has separate legible cards with no text overlap; Figure 3 hypothesis labels clear the top border and plotted dashed lines. Both initial layout findings are resolved. Other figure-specific interpretations and labeling qualifications are unchanged from the initial audit.

This verdict binds the exact candidate and generator hashes above. PDF and HTML pagination, packaging, links, and interactive behavior are separate presentation validation tasks.

Final terminology delta verified: “optimizer setting 1/8,” “alpha derived from the fit,” and plain phrases replacing hyphenated compounds retain the same scientific objects and claims. The raw proxy definition now correctly applies to either saved or fresh Gram factors. The plot labels “Equal update norms” and “norm outside / norm of full response” remain precise when read with the manuscript definitions of complete updates and combined response norms. Re-inspected the changed leakage and optimizer images. Figure-data remains byte-identical; no changed result or acceptance rule was introduced. The final grammar correction “an input with norm one” is included in the bound candidate hash.

Final metadata-only deletion verified: restoring the deleted final sentence “This visual revision changes presentation, not the measurements or their acceptance thresholds.” with its original line separation reconstructs the preceding approved candidate SHA-256 `1a2f04e9d879a30dd01447fa82bd62a37e1ea81f4bce29d364f7f2b96076fee9` exactly. No other manuscript content changed. The scientific pass therefore binds final manuscript SHA-256 `8b24439aa25459611f19d4c8a039b273feba758889e01f6c95657426309f51d7`. Generator SHA-256 remains `9034f5afa2c6e3f2ac86be5f9065689b0f43236b619370fca558e8302f23bd3a`.
