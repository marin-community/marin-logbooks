# Final scientific evidence audit

Candidate: `output/pdf/hessian-experiment-report.md`

Final inspected candidate SHA-256: `feaafa1cab3160f0cde2830d4111fbbc2df3cdbdce54738593cd80a35e98a4e4`

Initial fully audited candidate SHA-256: `0644c1438322a83786d8b319ac2186d5bd1d2dd3a4352190cc9ce78fd0ba2f82`. The final changes were separately inspected and approved.

Verdict: **Pass for scientific evidence and numerical consistency. No material correction required.** This is a local evidence audit, not an independent rerun of the TPU experiments or a statistical certification of the conclusions. No report prose or experiment files were edited.

## Scope and verification

Read the entire candidate, original early/middle/late result records, all seven follow-up directories, the supplied fitting implementation, the aligned analysis, the empirical resolution implementation, and the compact scientific results. Inspected the sole scientific illustration (`leakage-illustration.png`) visually. Confirmed the seven saved job verifications each show succeeded with four tasks, and each saved W&B verification has eight finished runs. Controls are the completed attempt-02 artifacts.

Independently recomputed empirical gate statuses and margins from every follow-up NPZ and every 12/48 restriction. Floating-point eigensolver differences are at insignificant last digits. Verified strict acceptance uses `accepted_positive_power`, not merely the embedded package `valid` status. Verified the wider first-12 curvature arrays equal the original middle arrays exactly for both blocks and trajectories.

Independently recomputed all raw/ridged fresh-proxy centered-log distances at both 12 and 48 dimensions. All sixteen combinations satisfy fresh-to-fresh distance less than either saved-to-fresh distance. Verified the report's fresh fit tables and raw-r/rho and ridged-alpha ranges against accepted fits only.

Independently checked all sixty optimizer branches against the raw sixteen-batch losses and CSV; regenerated all 120 baseline/versus-eighth percentile intervals using the declared seed and 10,000 paired resamples. All agree. The two control sequence manifests are identical: each has 1,049 accepted batches, 8,392 unique full-sequence hashes, and 25 rejected batches. Each control has two 512-batch histories and 30 branches.

Counted follow-up main HVPs directly from saved curvature array shapes: 3 × 2 × 8 × 12 + 2 × 2 × 8 × 48 = **2,112**. Original main HVPs are a separate 576. Derivative checks and optimizer gradients are not included.

## Verified key results

- One-eighth accepted mean margins: early attention 10.9051; middle attention 29.2034; middle MLP 36.4185. Strict bootstrap acceptance counts: 511, 619, 585 out of 1,000. Rejections respectively: 47/442, 178/203, 281/134 indefinite/unresolved. Reported raw r/alpha conditional intervals agree.
- One-eighth early MLP minimum: −4.78835e−8. Late attention minimum: −4.78888e−7, maximum 2.66281. These are measured nonpositive means; the declared screen returns before sign-error calibration. The report correctly avoids sign certification.
- Late one-eighth MLP has margin 4.78378 and is excluded from accepted estimates despite the embedded numerical fit being valid. Wide one-eighth MLP similarly has margin 7.10004 and remains excluded, including when fresh proxies are substituted.
- Quarter-power MLP margins: 193.1333 at 12 directions and 42.0113 at 48. Wide raw r/rho/alpha: 1.14744 / 0.882772 / 0.384671. Wide bootstrap: 663 accepted, 247 indefinite, 90 unresolved; accepted alpha interval [0.383379, 0.387031].
- Same-input nested leakage 12→48: quarter attention 0.927400579→0.926992802; quarter MLP 0.296787465→0.296574549; eighth attention 0.387614337→0.387058764; eighth MLP 0.513948503→0.513878260. All reductions are below 0.001. Quarter attention retains incomplete projected derivative references and is properly qualified.
- Forty-eight-direction raw fresh distances (fresh pair; saved to each): quarter attention 0.177759; 0.838489, 0.869063; quarter MLP 0.264393; 1.741462, 1.753943; eighth attention 0.151022; 0.627163, 0.665427; eighth MLP 0.156415; 1.143900, 1.155072.
- Accepted fresh raw fit alpha: quarter MLP12 0.472698/0.476740; quarter MLP48 0.469355/0.474186; eighth attention12 0.430562/0.431952; eighth MLP12 0.460524/0.466244. Accepted fresh raw r range 1.042134–1.141952; rho range 0.983362–0.996985; ridged alpha range 0.442240–0.489842.
- Quarter control baseline 3.673040584. All five full native mean changes are positive. Native quarter-minus-eighth difference −0.000519663, interval [−0.000819708, −0.000230771]; norm-matched difference −0.000241131, interval [−0.000523593, 0.000032908].
- Eighth control baseline 3.620126367. All ten smaller native branches have negative means and individual conditional upper bounds below zero. Full native quarter-minus-eighth difference −0.000276312, interval [−0.000360997, −0.000182686]. Norm-matched means for eighth/quarter are −0.000504360/−0.000504583; paired interval [−0.000164562, 0.000168999]. Native half-power mean changes from +0.000450715 to −0.000211090 after norm matching.
- Original tables, original bootstrap counts/intervals, and all six original leakage values match their saved records. The triangle's inside norm 0.374 and outside norm 0.927 agree with the original middle-attention norm decomposition.

## Interpretation audit

Section 9 stays within the evidence: selected-subspace geometry, conditional fixed-state loss effects, and observed fresh-history differences. It does not promote unresolved or indefinite fits, infer a full training optimum from geometric alpha, claim a causal drift/noise decomposition, or claim equivalence from an interval containing zero. The direct-grid minima are correctly distinguished from the centered-log formula; the reported values agree with the saved grid.

The middle MLP half-sample caveat is accurate: its even-indexed four-group mean fails numerical positivity (minimum approximately −1.11679e−7), despite acceptance of the eight-group mean. It is helpful but not indispensable beyond the 41.5% strict bootstrap failures. Its current short presentation is appropriate and makes no false split-validation claim. Package split results are not independently precision-calibrated.

The final candidate implements the recommended wording refinement: “separately randomized, disjoint sets of complete sequences.” This resolves the independence ambiguity.

Final-delta verification: the added C4≤4.0 statement agrees with `single-seed-findings.md` lines 105–117: 33,000 versus 34,000 first observed crossing updates; 1,081,344,000 versus 1,114,112,000 tokens, correctly rounded to 1.081 and 1.114 billion. The no-interpolation and unmatched-hardware qualifications are correct. The appendix now correctly distinguishes the original twelve HVP inputs Q from the expanded 48-direction output projector P. The quarter-MLP table heading now correctly says Directions. No outstanding correction remains for the final SHA.

## Inspected artifact hashes

- `experiments/ema_gram/curvature_resolution.py`: `26debf632f11b6bfcb2ab510b53885b7a7e3f9150e6e2bab8cbb149a0bed8abe`
- `logbooks/20-ema-gram-polar/hessian_aligned_analysis.py`: `de86cedfb964d8e181865c08daf5771a25f5d2170371e65b1b4a508ff7783ed0`
- `output/pdf/scientific-results.json`: `a9418468d9c0bfb14f19f10ebf66c8c1e743bff6171448c4a1c2eb6dd0e0de3c`
- `output/pdf/optimizer-step-results.csv`: `99c9fe330f66bc3f5224e47f7003646c8f9f9920a7de22a7583834ab16e5f058`
- `output/pdf/leakage-illustration.png`: `a57e8326f77812c0b267975342ccc20541c1735781ad9c3e8f4a6358080cad8c`

Additional inspected source SHA-256: `logbooks/20-ema-gram-polar/single-seed-findings.md`: `14f78f50e0b302433cc61f6a614641966b26b8c060d3589da514a364a667b21e`.

Final whitespace-only delta verified: restoring exactly one trailing ASCII space after “The observed difference was 0.008272.” recovers the previously approved report SHA-256 `e3deb3de83d77068a98f57e1fcb00cc4b931c8c568ae0e8c19c2b272629bbe00`. No other manuscript content changed. The passing scientific verdict therefore binds the final SHA above. PDF layout adjustments are outside this scientific-content audit.
