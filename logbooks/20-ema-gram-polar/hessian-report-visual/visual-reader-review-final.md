# Independent reader review — final candidate

Date: 11 November 2026.

## Exact candidate

Manuscript SHA-256: `1a2f04e9d879a30dd01447fa82bd62a37e1ea81f4bce29d364f7f2b96076fee9`.

Ordered manuscript-plus-nine-PNG manifest SHA-256: `147112e3489435a63f11afccfc4126aa3b53930be8ae91a458e5d2bf0bdf55b7`.

## Scope and verdict

**PASS: no material comprehension barrier remains in the reviewed prose and figures.**

I reread the complete final `visual-report.md` and visually inspected all nine referenced PNGs in reading order. I did not consult source records, implementation code, the previous technical report, or explanatory material outside the candidate to fill missing meaning. No manuscript edits were made. This verdict covers reader comprehension and the PNG figures, not correctness of the underlying experiments or HTML/PDF pagination.

The report tells me what was measured, how to read each result, what it shows, and what it cannot establish. Its principal conclusion remains restrained: the measurements do not explain the earlier final-loss advantage of training alpha 1/8. The three distinct uses of alpha, selected-block versus full-model scope, numerical acceptance, conditional uncertainty, response leakage, history distance, and update-norm comparison are distinguishable without reconstructing the experiment from external materials.

## Recheck of earlier findings

- **R1 resolved:** Figure 1's headings and body text now fit inside their cards. The three additions are visually separate, counts remain explicit, and the fixed-checkpoint scope is clear.
- Figure 3's proposed-power labels now sit below the upper boundary with clear spacing.
- The mean Hessian and numerical precision screen now explain the averaging order, common coordinate scaling, and comparison against discrepancies. The empirical-check limitation remains explicit.
- “Fit,” “raw proxy,” “block,” “MLP,” and “batch” now receive definitions before they carry important interpretation.
- The leakage definition now explicitly combines responses after averaging by data group rather than averaging individual leakage ratios.
- Figure 8's caption identifies paired mean differences and conditional 95% bootstrap intervals directly, and preserves the warning that including zero does not prove equivalence.
- The final detail pointers are Markdown links. Their target documents and final rendered navigation are outside this reader pass.

## Figure-level final check

| Figure | Reader conclusion and retained limitation |
|---|---|
| 1 | The added work addresses trajectory mismatch, coverage, and optimizer controls; these are fixed-checkpoint experiments. |
| 2 | Acceptance depends on positivity and numerical checks; negative, unresolved, and missing-check outcomes remain separate. |
| 3 | Accepted mean powers differ from the proposed powers; plotted dots are not uncertainty intervals or training-optimal alphas. |
| 4 | Leakage is a combined response-norm ratio; squared norms add, and this is not a two-dimensional model or a fraction of loss/examples. |
| 5 | Extra output directions barely reduce leakage for fixed inputs; the finding is limited to these chosen directions and retains the attention-check caveat. |
| 6 | Fresh histories are closer to each other under the stated distance; individual distances are not uncertainty intervals or an identified drift/noise decomposition. |
| 7 | Candidate local losses depend on update magnitude; intervals condition on one state and gradient batch and do not rank long-term training. |
| 8 | The paired contrast favors 1/4 at original size and includes zero after norm matching; this does not establish equivalence. |
| 9 | Positive mean changes remain loss increases, even when smaller; different starting checkpoints do not isolate training alpha causally. |

No further required edits. The previously suggested additional intuition for history distance remains optional; its definition and “smaller = closer” axis are sufficient here.

## Manifest

```text
1a2f04e9d879a30dd01447fa82bd62a37e1ea81f4bce29d364f7f2b96076fee9  visual-report.md
5b68ed8ddf83b5932fec81fd1a4d72087788b637921512ab445a754f33335e3d  experiment-map.png
4b77b9fdf995acb57804331ae18d8df8305dcc51f297582207ff5d7908e3670b  acceptance-map.png
a554b5e43b09f884d663d3568a4b1ed1daf710f13149351cdcc04dfec85c5d2b  fitted-powers.png
a57e8326f77812c0b267975342ccc20541c1735781ad9c3e8f4a6358080cad8c  leakage-illustration.png
e474db9f9df19a85f0ea09f67f8140cfc57c6302e2f4f887d478301ed2a52430  leakage-expansion.png
e55e51796486ee8013fa2a171ccdfd90c23e5ff6ec991d0863122c0d51f343f5  history-distances.png
c61acdee633c284790711822906cce86a613a5e738da6a23de09a13147fea94c  optimizer-eighth.png
229dc45370ef1e306c18c6247dad2c214f68c25bd81f00db5aad4d476f6ba9a5  paired-contrast.png
769025cb073d400a95e91a311a3c055c1a7fb1370994ba16c33bdeea1a615563  optimizer-quarter.png
```


## Final style-delta recheck

Reread the complete manuscript after the final style sweep. Visually inspected all four changed PNGs (experiment map, leakage expansion, and both optimizer panels); SHA checks confirm the other five PNGs are unchanged from the preceding full review. The distinction between optimizer setting alpha and fitted power r is clearer. The raw-proxy definition now covers fresh as well as saved statistics. “Equal update norms” remains unambiguous because the text explicitly defines complete updates, including all parameters and weight decay. The leakage axis remains consistent with the combined-norm definition. No overflow or new comprehension barrier was introduced. Confirmed the final wording “an input of size one,” which avoids using “norm” before its definition. The exact hashes and manifest above identify this latest candidate. **Final verdict remains PASS.**


## Final metadata-deletion binding

**PASS retained.** Verified cryptographically that restoring only the deleted closing sentence, “This visual revision changes presentation, not the measurements or their acceptance thresholds.”, with its original newline spacing reproduces the previously reviewed manuscript SHA-256 `1a2f04e9d879a30dd01447fa82bd62a37e1ea81f4bce29d364f7f2b96076fee9` exactly. All nine PNG hashes remain unchanged. Thus the scientific prose and figures are identical to the reviewed candidate; this deletion removes only the presentation-change note. No additional comprehension review is necessary for this deletion.

This addendum supersedes the candidate identity above for the delivered manuscript.

Final manuscript SHA-256: `8b24439aa25459611f19d4c8a039b273feba758889e01f6c95657426309f51d7`.

Final ordered manuscript-plus-nine-PNG manifest SHA-256: `9113acd7ea851fa04253edfe1ef4e927312da09b3b7552c7bfe5a8488cfb2ec6`.

```text
8b24439aa25459611f19d4c8a039b273feba758889e01f6c95657426309f51d7  visual-report.md
5b68ed8ddf83b5932fec81fd1a4d72087788b637921512ab445a754f33335e3d  experiment-map.png
4b77b9fdf995acb57804331ae18d8df8305dcc51f297582207ff5d7908e3670b  acceptance-map.png
a554b5e43b09f884d663d3568a4b1ed1daf710f13149351cdcc04dfec85c5d2b  fitted-powers.png
a57e8326f77812c0b267975342ccc20541c1735781ad9c3e8f4a6358080cad8c  leakage-illustration.png
e474db9f9df19a85f0ea09f67f8140cfc57c6302e2f4f887d478301ed2a52430  leakage-expansion.png
e55e51796486ee8013fa2a171ccdfd90c23e5ff6ec991d0863122c0d51f343f5  history-distances.png
c61acdee633c284790711822906cce86a613a5e738da6a23de09a13147fea94c  optimizer-eighth.png
229dc45370ef1e306c18c6247dad2c214f68c25bd81f00db5aad4d476f6ba9a5  paired-contrast.png
769025cb073d400a95e91a311a3c055c1a7fb1370994ba16c33bdeea1a615563  optimizer-quarter.png
```
