# Independent final terminology review

Reviewer: `/root/visual_terms`, dedicated read-only terminology auditor.

Final candidate: `visual-report.md`.

SHA-256: `1a2f04e9d879a30dd01447fa82bd62a37e1ea81f4bce29d364f7f2b96076fee9`.

**Verdict: PASS for first-use terminology, notation, scientific naming, affected definition dependencies, and the explicit hyphen-preference check.** No unresolved substantive terminology blockers remain in this guide. The initial REVISE verdict in `terminology-review.md` applies only to the earlier SHA and is superseded for this final candidate.

Read-only review procedure: traversed the guide in source order, including all nine figures, their labels, alt text, and captions. Rechecked the initial first-use ledger against the final manuscript; later text was not used to excuse missing meaning at substantive earlier use. Actual technical-reference destinations were consulted only to verify explicit deferrals, not to fill the guide's local definition gaps. The figures below were visually inspected. Numerical accuracy and final browser/PDF packaging are separate validation dimensions.

## Final first-use and dependency ledger

The unchanged background entries in the initial ledger were rechecked and retain their successful status. The following ordered ledger records all main scientific objects and all repaired dependencies in the final guide.

| Object | First substantive use / definition | Evidence available at that point | Status |
|---|---|---|---|
| Hessian, gradient statistics, fit | Introductory previews, lines 1–23 | Reader is directed to Section 2; actual substantive analysis follows its definitions | FORWARD-REFERENCED; destination verified |
| optimizer setting 1/8 | opening, line 9 | Plain-language setting; named alpha at 35 before comparison table | DEFINED HERE / DEFINED EARLIER in substantive use |
| weights, training, loss, optimizer | 33 | Adjustable model numbers, changes to improve predictions, error score, chooser of changes | DEFINED HERE |
| alpha / α | 35 | Controls relative rescaling using stored gradient statistics | DEFINED HERE |
| C4 | 37 before table | Reserved text evaluation in earlier comparison | DEFINED HERE |
| token and seed | 48–50 | Unit of text; controller of random choices | DEFINED HERE |
| learning rate | 50 | “multipliers controlling the size of updates” | DEFINED HERE |
| gradient | 60 | Loss response to small changes in weights | DEFINED HERE |
| Hessian / H, curvature | 64 | Change of gradient as weights move; local bending | DEFINED HERE |
| Gram matrix | 68 | Products of gradient entries, with left and right matrices | DEFINED HERE |
| Gram proxy / K | 70 | Curvature statistic acting by multiplying weight changes on both sides | DEFINED HERE |
| EMA | 72 | 99% prior average plus 1% new gradient product | DEFINED HERE |
| fit | 76, before acceptance analysis | “approximates the Gram proxy using a power of the Hessian, within the selected directions” | DEFINED HERE |
| raw proxy | 78 | Gram factors without the additional stabilizing modification; does not conflate raw with saved | DEFINED HERE; modification forward reference verified at 177 |
| training alpha / alpha derived from fit / candidate alpha | 82–86 | Model-producing setting / derived from now-defined fit / hypothetical next-step setting | DEFINED HERE |
| checkpoint | 94 | Saved model and training state | DEFINED HERE |
| attention / MLP / block | 102 | Roles of both components; multilayer perceptron expanded; block is each selected matrix | DEFINED HERE |
| exact measured matrices | 102 | Explicitly deferred to technical reference | FORWARD-REFERENCED; actual query/gate matrix descriptions verified in reference |
| subspace | 104 | Changes formed by combining selected directions | DEFINED HERE |
| HVP | 106 | Gradient response for each direction, including interactions | DEFINED HERE |
| batch and fresh history | 112 | Group of text sequences processed together; histories formed at fixed weights | DEFINED HERE |
| eigenvalue / matrix logarithm | 122 | Scaling along special direction; eigenvalue logarithms retaining directions | DEFINED HERE |
| acceptance exclusions | 126–128 | Negative curvature, insufficient numerical separation, missing/failed check | DEFINED HERE |
| automatic differentiation / finite difference | 132 | Code derivative rules / estimates using nearby weights | DEFINED HERE |
| mean Hessian | 134 | Average eight group Hessians first; then fit; not an average of fitted powers | DEFINED HERE |
| coordinate scaling / discrepancy size | 136 | Common diagonal scale applied to all estimates, pairwise comparison, maximal response to an input of size one | DEFINED HERE |
| precision margin | 138 | Weakest curvature on that scale divided by largest pairwise discrepancy; exceeds ten | DEFINED HERE |
| exact precision matrix formula | 138 | Explicit companion reference; local operation sufficient for guide | FORWARD-REFERENCED; destination verified |
| bootstrap | 148 | Repeatedly resample eight groups and recompute fit | DEFINED HERE |
| conditional fit intervals | 150 | Only accepted resamples contribute; limitations stated | DEFINED HERE |
| fitted power r | 160 | Power relating K to H; symbols H/K already defined | DEFINED HERE |
| rho / ρ | 162 | Variation in log proxy explained after removing scale | DEFINED HERE |
| alpha derived from fit = ρ/(2r) | 164 | Both symbols and positivity condition established before use | DEFINED EARLIER |
| ridged proxy | 177 | Normalize each factor by mean eigenvalue, add 0.01 to diagonal | DEFINED HERE |
| separate geometric check and additional optimizer transformations | 179 | Full details deliberately deferred; no local undefined optimum claim remains | FORWARD-REFERENCED; grid and optimizer sections verified |
| leakage | 185–191 | Outside curvature response; combined outside norm / combined full norm | DEFINED HERE |
| norm | 189 | Square root of sum of squared entries | DEFINED HERE |
| combined norm / aggregation order | 189–191 | Average each direction's response across groups; collect responses and sum all squared entries before square root; not mean of separate ratios | DEFINED HERE |
| input / output directions | 187–203 | Input weight movements versus gradient-response components | DEFINED EARLIER |
| clipping | 217 | Shrink gradient if norm exceeds 0.1 | DEFINED HERE |
| sequence hashes | 219 | Full-sequence duplicate exclusion; explicit limits on overlap | DEFINED HERE by operational role |
| centered logarithm and distance | 221 | Remove mean log eigenvalue; root-sum-squared entry differences normalized by sqrt dimension | DEFINED HERE |
| historical drift | 217–231 | Changing weights/history already explained | DEFINED EARLIER |
| weight decay | 237 | Separate shrinkage of weights | DEFINED HERE |
| original update size and matched norms | 243–246 | Native optimizer magnitude versus scaling complete update to alpha 0 norm, including decay | DEFINED HERE |
| loss change | 250–257 | After-step minus unchanged loss; sign and ×1000 scale explained | DEFINED HERE |
| paired batch intervals | 254–261 | Shared evaluation batches; bootstrap; fixed state and gradient condition in caption | DEFINED HERE |
| paired contrast | 269–273 and Figure 8 | Same evaluation batches; 1/4 minus 1/8 explicit on axis; dot/interval semantics in caption | DEFINED HERE |
| full step and starting checkpoint labels | Figures 7–9 | Defined candidate sizes and checkpoint training alphas before plotting | DEFINED EARLIER |

## Cross-references and figure labels

Section references 2, 4, 5, 6, and 7 and the Figure 7 reference resolve to the described manuscript objects. The named technical reference, evidence index, and optimizer step table now have explicit links. Their final relative-file resolution is a packaging check assigned to the author; this review verified the underlying technical-reference text destinations for the deferred scientific details.

Figure 1 now keeps titles and subtitles within their boxes. Figure 3 now has sufficient top label clearance. The updated leakage axis and optimizer panel titles use the same operations as the guide. Scientific component labels use defined terms rather than repository aliases. No introduced symbol appears in a mathematical expression before its needed meaning. No circular definition remains among the repaired concepts.

The explicit hyphen check passes: avoidable authored compounds were rephrased. The remaining “Hessian-vector product” and “finite-difference” terminology is established technical usage. Hyphens in filenames, Markdown syntax, and source identifiers are not rendered compound prose.

## Figure versions inspected

- experiment-map.png: `5b68ed8ddf83b5932fec81fd1a4d72087788b637921512ab445a754f33335e3d`
- acceptance-map.png: `4b77b9fdf995acb57804331ae18d8df8305dcc51f297582207ff5d7908e3670b`
- fitted-powers.png: `a554b5e43b09f884d663d3568a4b1ed1daf710f13149351cdcc04dfec85c5d2b`
- leakage-illustration.png: `a57e8326f77812c0b267975342ccc20541c1735781ad9c3e8f4a6358080cad8c`
- leakage-expansion.png: `e474db9f9df19a85f0ea09f67f8140cfc57c6302e2f4f887d478301ed2a52430`
- history-distances.png: `e55e51796486ee8013fa2a171ccdfd90c23e5ff6ec991d0863122c0d51f343f5`
- optimizer-eighth.png: `c61acdee633c284790711822906cce86a613a5e738da6a23de09a13147fea94c`
- paired-contrast.png: `229dc45370ef1e306c18c6247dad2c214f68c25bd81f00db5aad4d476f6ba9a5`
- optimizer-quarter.png: `769025cb073d400a95e91a311a3c055c1a7fb1370994ba16c33bdeea1a615563`

## Hash-binding delta addendum

Current final guide SHA-256: `8b24439aa25459611f19d4c8a039b273feba758889e01f6c95657426309f51d7`.

Verified the exact delta by appending one newline, the deleted final sentence, and one terminal newline to the current bytes. The reconstructed bytes hash to the previously approved SHA `1a2f04e9d879a30dd01447fa82bd62a37e1ea81f4bce29d364f7f2b96076fee9`. Thus the only change is removal of the last presentation-metadata sentence and its separating blank line: “This visual revision changes presentation, not the measurements or their acceptance thresholds.”

This deletion removes no scientific definition, claim, qualification, or cross-reference dependency. **PASS remains valid for current SHA `8b24439aa25459611f19d4c8a039b273feba758889e01f6c95657426309f51d7`.**
