# Large BF16 Splash separate-context VJP analysis

The separate capture passes every exact source guard and all33full/composed comparisons. Full named/native forward outputs are exact. The MLP tail is also exact on the common named residual/cotangent, including input VJP and its4active parameter gradients. The persistent difference is already present in the attention prefix.

| Common-operand named/native result | Max absolute error | Relative L2 | Different elements |
| --- | ---: | ---: | ---: |
| Full and prefix input VJP |3.72529e-9 |0.0588515% |1,650,899/33,554,432 |
| Full and prefix Q-master VJP |4.76837e-7 |0.457671% |2,456,792/4,194,304 |
| Full inputnorm-master VJP |9.53674e-7 |0.536624% |1,253/2,048 |
| Prefix inputnorm-master VJP |9.53674e-7 |0.565345% |1,223/2,048 |

Within each BF16 implementation, full and prefix input/Q VJP **hashes** match exactly. The inputnorm hashes do not match across these contexts. Fixed named r/ct_r substitution remains a distinct control: it must not be described as a pure fusion test. Full replay also produces tiny down/K/V differences where the normal full-Trainer record was exact, while its up/O gradients are exact where normal training differed. The replay has not reproduced every normal training cotangent/fusion detail.

## Concrete optimized rounding-order candidate

Both retained StableHLO graphs multiply the raw Splash dQ by the same BF16 scale(1/sqrt128 rounded to0.08837890625), then apply inverse RoPE. The shared private Splash forward and backward functions are byte-identical between named/native. Runtime kernel operands and dQ outputs were not materialized, so kernel identity does not prove equal internal values.

The optimized prefix HLO differs:

- Named `last-named-prefix.hlo.txt:793–800` rounds `dQraw*scale` toBF16 first. Its inverse RoPE then multiplies that rounded result by ordinary sin/cos(`803–847`; entry1340).
- Native `last-native-prefix.hlo.txt:244–265` precomputes and rounds `sin*scale` and `cos*scale` toBF16. The raw Splash dQ then multiplies these already-scaled coefficients(`555–599`; entry978,1050–1078).

The same distinction appears in full optimized HLO(named1214–1221; native280–301). These sequences agree over real arithmetic but change BF16 double-rounding sites. The earlier107M attention replay used head64 and an exact power-of-two scale1/8; it did not test the new head128 non-power-of-two scale. No source scalar mismatch or precision-enum mismatch was found.

This is a concrete post-StableHLO localization hypothesis. The artifacts do not yet prove it explains the observed error magnitude, locate which optimizer pass changed the sequence, or classify a compiler defect. A production patch is premature.

The matched FP32-highest Splash control is a different-compute-dtype diagnostic reference. Q-master relative error against it is0.9254% for named and0.9277% for native; neither path is uniformly closer across all outputs. This is not exact-real-arithmetic evidence or a new accepted training result.

## Smallest next boundary

Hold actual large-prefix raw BF16 Splash dQ and BF16 cos/sin fixed. Compare the two retained optimized rounding sequences in separate JITs, checking post-RoPE dQ before common-operand Q-projection input/master VJPs. Retain exact isolated-prefix/capture hashes and source controls, and inspect HLO to verify rounding boundaries survived compilation. This tests a specific new head128 rounding-order hypothesis; it does not repeat the rejected full-model attention-axis patch. No new control has been implemented or launched.

Evidence: `native-jax-bf16-splash-15b-separate-vjp-packed-numerical-analysis.json` retains all numerical rows, source guards, HLO pointers/hashes, reference comparisons and artifact manifest. No source/math/threshold/LR change, experiment, remote query or download was performed for this audit. No full-Trainer identity, performance or convergence claim follows.
