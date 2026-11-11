# Final independent terminology review

Reviewer: `/root/report_terminology`.

Final reviewed manuscript SHA-256: `feaafa1cab3160f0cde2830d4111fbbc2df3cdbdce54738593cd80a35e98a4e4`.

Whitespace-only final verification: adding one trailing space after “The observed difference was 0.008272.” in memory recovers exactly the prior passing SHA `e3deb3de83d77068a98f57e1fcb00cc4b931c8c568ae0e8c19c2b272629bbe00`. No other byte differs. The PASS verdict therefore applies unchanged to this final SHA. The manuscript was not modified by this verification.

Pre-repair full-review SHA-256: `0644c1438322a83786d8b319ac2186d5bd1d2dd3a4352190cc9ce78fd0ba2f82`.

Reviewed illustration SHA-256: `a57e8326f77812c0b267975342ccc20541c1735781ad9c3e8f4a6358080cad8c`.

Reviewed generator SHA-256: `13463f21c654b740d8c0b9e82a5e83e94553cbb1c64e1d52ea8209276ac5a852`.

Scope: the full candidate, in exact reading order, including the opening overview, Sections 1–9, every table, both appendices, and the leakage illustration and caption. All new completed-result sections were read. The image was visually inspected. No manuscript or figure edits were made.

## Verdict

**PASS. No material terminology, stale-definition, code-jargon, or mathematical-notation blocker remains in the final candidate.** First-use explanations support the report's main reasoning. The earlier centering-order issue is fixed. The final new metrics—precision margin, residual score, conditional loss intervals, and fresh-proxy distance—have usable definitions and scope qualifications.

The complete pre-repair candidate was reviewed in manuscript order. The final exact SHA was then verified, and all changed passages and affected dependencies were read directly from the revised file. The following fixes pass:

- Final lines 644–652 explicitly retain twelve input columns in Q while P projects onto the expanded 48 output directions. The added statement that P is constructed from the expanded basis removes the earlier ambiguity with P=QQ^T.
- Final lines 47–49 define token locally and distinguish first recorded evaluation at the loss target, observed token counts, and elapsed execution time. The lack of interpolation and differing hardware layouts are explicit. These additions introduce no terminology dependency or claim of intrinsic optimizer speed.
- The final line 502 table header “Directions” describes its entries, while the preceding paragraph supplies the quarter-power MLP context.
- Final line 549 uses “separately randomized, disjoint sets of complete sequences,” consistent with the complete-sequence disjointness definition and the limitations on statistical independence earlier in the report.

No manuscript or figure edits were made by this reviewer. The final verdict covers the complete candidate through the full review plus this dependency recheck.

### Resolved finding: distinguish leakage input directions from output projection

In the pre-repair candidate, Appendix A lines 642–646 defined `P = QQ^T` and called the resulting ratio “the measured leakage.” This covered measurements using the same input and output directions but did not state the wider leakage comparison in Section 7 line 277 and Section 8 lines 511–518. The final candidate resolves this at line 650.

The requested repair, now implemented and verified, was:

> For the wider-output comparison, Q still contains the original twelve input directions, while P projects onto all 48 output directions.

The final version also explicitly says P is built from the expanded basis rather than from Q alone. The exact-definition appendix now agrees with the main text.

## First-use and definition-dependency ledger

The ledger below records the full pre-repair review; its line numbers refer to SHA `0644c1438322a83786d8b319ac2186d5bd1d2dd3a4352190cc9ce78fd0ba2f82`. Changed definitions and their final line numbers are recorded in the passing recheck above. Definitions are credited only when available in the reading prefix, except for explicit verified forward references. Standard arithmetic, sums, ratios, squared values, and ordinary matrix indexing are treated as background; specialized scientific operations are audited below.

| Term / operation | First use or definition | Available evidence and later dependency | Verdict |
|---|---:|---|---|
| Weights, loss, optimizer | 27–29 | Adjustable model numbers; prediction-error score; rule selecting changes | DEFINED HERE; used throughout |
| Gradient, Hessian, curvature | 31 | Local loss change and change of gradient; small movement against nonzero gradient | DEFINED HERE; used in the measured response and fit |
| Preconditioning | 33 | Rescaling directions for curvature differences | DEFINED HERE; subsequent inverse-power mechanism uses this concept |
| Gram matrices, transpose | 35 | Gradient times transpose and reversed right product; row/column exchange | DEFINED HERE; later raw/ridged actions depend on them |
| EMA | 37 | Retain 99% and add 1% new product | DEFINED HERE; fresh/saved-history comparisons consume it |
| Alpha and relative inverse powers | 39 | Relative direction rescaling, including possible amplification | DEFINED HERE; whole-update magnitude is not incorrectly promised to decrease |
| Training alpha / fit-derived alpha | 41 | Training setting versus value calculated from measured curvature | DEFINED HERE; distinguished in tables and conclusions |
| Seed and held-out evaluation | 45–49 | Evaluation text reserved from training; randomness explanation before repetition claim | DEFINED HERE within introductory comparison |
| Ridge | 51 | Gram stabilization, explicit Section 3 reference | FORWARD-REFERENCED; operation verified at 157 |
| Learning rate | 51 | Overall update size | DEFINED HERE; table's matrix-specific scope explained at 62 |
| H, K, proxy | 68–70 | Hessian coordinates; raw left/right action and selected components | DEFINED HERE before hypothesis |
| r | 72 | Fitted power relationship | DEFINED HERE |
| Eigenvalue, eigenvector, symmetric matrix | 74 | Scaling value/direction; row/column symmetry | DEFINED HERE before logarithm and acceptance conditions |
| Matrix logarithm | 76 | Log positive eigenvalues, preserve eigenvectors | DEFINED HERE |
| rho | 78 | Explained centered log-proxy variation | DEFINED HERE; precise formula later at 624 |
| Centering / approximate objective | 80–84 | Mean log eigenvalue removed before comparison; squared difference minimized | DEFINED HERE; exact-grid distinction explicit |
| Checkpoint, trajectory | 100 | Saved state and training-state sequence | DEFINED HERE |
| Parameters, architecture, layer, token | 106 | Functional explanations and weight synonym | DEFINED HERE |
| Attention, MLP, query/gate matrices | 110 | Roles given before matrix table | DEFINED HERE |
| Selected directions, subspace | 119–123 | Outer-product construction; selection scales; all combinations | DEFINED HERE |
| HVP, projection | 127–129 | Gradient response; component along each selected direction | DEFINED HERE |
| Loss components, floating/mixed precision, norm, clipping | 131–135 | Loss roles, approximate arithmetic, entrywise size, threshold shrinkage | DEFINED HERE |
| Hashes, caches, valid prediction positions | 139–145 | Matching identifiers, stored training text, scored positions | DEFINED HERE |
| Gram factors, raw/ridged proxy | 153–159 | Factors synonymous with left/right matrices; action; normalization and diagonal addition | DEFINED HERE / EARLIER |
| Log-derived exponent, direct grid minimum | 169–175 | Formula versus tested rescaled-curvature objective | DEFINED HERE / EARLIER; training optimum explicitly distinguished |
| Fit validity and bootstrap | 197–207 | Eigenvalue ratio gate; symmetry reference; replacement resampling and conditional interval | FORWARD-REFERENCED symmetry check verified at 219; remaining operations DEFINED HERE |
| Positive definite / indefinite | 215–217 | Eigenvalue-sign criteria | DEFINED HERE; later rejection tables use them |
| Symmetry discrepancy / spectrum | 219–229 | Norm of transpose difference divided by matrix norm; set of eigenvalues | DEFINED HERE |
| Leakage | 239–245 | Outside-response norm divided by full-response norm; groups averaged first | DEFINED HERE |
| Illustration labels/caption | 255–259 | Inside/outside/full response and norm already explained; full norm visibly normalized to one | DEFINED EARLIER; label/caption check PASS |
| Wider measurement / fixed leakage inputs | 277 | Original twelve inputs retained; output projection enlarged to 48 | DEFINED HERE; consumed by wider leakage table |
| Automatic differentiation, central finite difference | 283–285 | Derivative rules versus nearby-weight evaluations; relative error denominator explicit | DEFINED HERE |
| Empirical precision screen | 293–302 | Symmetrization, common rescaling, three discrepancies, spectral norm, tenfold criterion | DEFINED HERE; no missing coordinate-rescaling dependency |
| Fresh histories / effective samples / disjointness | 306–312 | Fixed weights, 512 batches, EMA, weight concentration, complete-sequence overlap check | DEFINED HERE; limitations on shared fragments and independence retained |
| Momentum, singular values, polar operation | 316–320 | Running average; paired directions and strengths; positive strengths replaced by ones | DEFINED HERE |
| Candidate alpha, native version, norm matching | 322–328 | One hypothetical step; original magnitude; whole-update rescaling | DEFINED HERE; full-model scope explicit |
| Paired evaluation / conditional variation | 330–334 | Same text; fixed state and one update gradient | DEFINED HERE |
| Precision margin | 342 | Smallest rescaled eigenvalue divided by largest discrepancy | DEFINED HERE before table |
| Accepted / rejected / unresolved results | 344–367 | Numerical positivity and empirical-screen reasons attached; no certification of tiny negative signs | DEFINED EARLIER; applied with explicit scope |
| Residual score | 371 | Squared centered log-proxy error divided by squared proxy size | DEFINED HERE before comparison table |
| Bootstrap interval conditioning | 385–401 | Both screens reapplied; middle 95% of accepted repetitions | DEFINED HERE / EARLIER |
| Loss change and displayed scale | 429–431 | After minus before; negative improvement; values multiplied by 1,000 | DEFINED HERE before numeric table |
| Conditional local interval | 433–441 | Pairing already defined; 10,000 resamples and middle 95% specified immediately after table | DEFINED HERE across table explanation; no material ambiguity |
| Paired candidate differences | 449–473 | Candidate order, common starting state/text, and norm convention supplied | DEFINED EARLIER / HERE |
| AdamW in result explanation | 451 | Names optimizer for outside-transform parameters; its mechanics defined at 660 | DEFINED LATER; nonblocking because the claim concerns already-defined whole-update rescaling, not AdamW mechanics |
| Wider fits and wider leakage | 483–522 | Input directions and output projection explicitly separated | DEFINED EARLIER / HERE; appendix generalization remains required |
| Fresh-proxy distance | 528 | Centered log matrices; entrywise difference norm divided by square root of dimension | DEFINED HERE before table |
| Fresh 1 / Fresh 2 | 530 | Two separately sampled histories at unchanged weights | DEFINED HERE; useful experiment labels rather than internal run aliases |
| Fresh-fit uncertainty scope | 545–562 | Hessian/coordinates fixed; curvature resampling conditional on proxy; no substitution into full steps | DEFINED HERE / EARLIER |
| Final interpretation | 566–582 | Uses established distinctions; future tests visibly future | DEFINED EARLIER; no new unexplained criterion |
| W, ell, e_i, d | 588 | Weight matrix, group loss, selected unit directions, count | DEFINED HERE before equations |
| Inner product / Hessian operator | 590–592 | Entrywise-product sum and within-block operator | DEFINED HERE before equation |
| L, R, E, K entries | 594–596 | Factors, weight change, action | DEFINED HERE |
| Trace, identity, m/n, A/B | 600–606 | All constituents defined before ridge formulas | DEFINED HERE |
| X/Z, Frobenius product/norm, r/rho/alpha_log | 610–630 | Centered logs and norm precede regression/objective formulas | DEFINED HERE / EARLIER |
| Congruence transformation | 632–638 | Two-sided multiplication; exact grid differs from log approximation | DEFINED HERE; used again at 662 |
| Q, P, H_full, leakage formula | 642–648 | Q input columns and P=QQ^T; all symbols defined before ratio | Earlier scope ambiguity RESOLVED by final line 650; input and output sets now explicitly distinguished |
| Momentum matrix and optimizer formula | 652–654 | Momentum and factors already defined; negative learning-rate factor distinguished | DEFINED HERE / EARLIER |
| C/U/Sigma/V, polar decomposition | 656–658 | Directions and strengths precede decomposition; positive singular-value condition explicit | DEFINED HERE |
| Newton–Schulz / AdamW roles | 658–660 | Approximation iteration and coordinate rescaling roles | DEFINED HERE |
| Appendix B evidence terms | 666–672 | Provenance and established checks; numerical thresholds repeat definitions | DEFINED EARLIER / ordinary reproducibility terms |

## Stale-definition audit

No material stale definition requires removal. Semantic consumers, rather than word counts, were checked:

- H/K, Gram factors, EMA, alpha/r/rho, logarithms, centering, and raw/ridged variants feed the later fits, comparisons, and formulas.
- The training/fit/candidate alpha distinction is consumed by the separate geometric and optimizer-step results and by Section 9.
- HVP, projection, subspace, and norm are needed for leakage, empirical validation, and wider measurements.
- Symmetry, positivity, empirical margin, and bootstrap validity directly explain accepted and rejected results.
- Momentum, singular values, and the polar operation explain why geometric rescaling is not the whole optimizer; the appendix implements that distinction mathematically.
- Fresh-history weighting, effective sample count, disjointness, and conditional uncertainty constrain the later sampling and historical-change interpretation.
- The temporary appendix symbols have immediate computational consumers. They are not unused named concepts.
- “Spectrum” is a local explanatory alias in one diagnostic paragraph. It introduces no dependent machinery and is not a material clarity problem.

## Code-derived terminology and figure review

No material code-jargon blocker was found. “Fresh 1” and “Fresh 2” identify two defined scientific replicates. Public model/optimizer names identify actual experimental objects. Package and evidence references occur in provenance context rather than substituting for scientific definitions. The manuscript does not expose internal version aliases or configuration fields as unexplained result concepts.

The unchanged leakage illustration is legible and coherent with its caption. It shows the combined norm decomposition for original middle attention, labels the scaling to full norm one, and avoids claiming the model itself has two dimensions. Labels do not introduce new scientific objects. The surrounding explanation distinguishes norm ratio from squared-norm fraction.

## Review limits

This is an independent terminology, notation, and internal-consistency review of the complete exact final candidate. It does not independently rerun experiments or certify all source measurements. The required leakage repair and all other revised passages have now been checked against the final SHA. No material clarity or notation blocker remains.
