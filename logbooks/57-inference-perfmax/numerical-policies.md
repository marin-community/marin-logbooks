# Evidence for the numerical policies

The two BF16 policies are provisional engineering screens for Snowball v0.1.1
on H100 and Hero/Mochi step400407 on GB200. Their fixed Levanter scores and
acceptance limits are separate. No limit changed during this validation.
A pass means the recorded deployment meets these limits on the listed cases.
It does not establish an acceptable error budget for RL training.

The checks use three metrics:

- Reference: a conservative bound on the largest absolute probability error
  over the union of saved top-64 and serving top-50 scores.
- Cache and batch: the same bound over two serving top-50 lists. Missing tokens
  can have probability between zero and that list's cutoff. Raw results retain
  lower and upper bounds; acceptance uses the upper bound.
- Decode: the absolute probability error of the emitted token during generation
  versus fresh scoring. Other returned tokens are not compared here. Its raw
  log-probability error is recorded without an acceptance limit.

Maximum, p99, p90 and the fraction exceeding 0.075 aggregate these per-position
values. The [format check](analysis/metric-scope-verification.json) verifies the
saved list sizes. None of these metrics measures full-vocabulary KL or a trainer
log-probability ratio. Probabilities 0.01 and 0.02 differ by only 0.01 but have a
ratio of two.

The limits below are maximum / p99 / p90 / fraction above 0.075. Fractions are
shown as percentages. The policy files are the authority for exact values:
[Hero](../../lib/marin/src/marin/testing/inference/resources/endpoint/hero-sft-step400407.json)
and [Snowball](../../lib/marin/src/marin/testing/inference/resources/endpoint/snowball-v0.1.1.json).

Hero's `short` and `long` are fixed manifest labels, not request-length rules.
The [short group](analysis/hero-reference-groups.json) has 32 histories of
40–1,835 tokens. The five long histories have 4,096, 32,768, 65,536, 131,072 and
262,143 tokens; the last leaves room for one output token. The manifest labels
select the applicable reference limits.

| Comparison | Limits | Evidence behind the choice | What remains weak |
| --- | --- | --- | --- |
| Hero short reference | 0.65 / 0.125 / 0.04 / 3.5% | Margins around the five BF16 controls in [#11430](https://github.com/marin-community/marin/issues/11430). Its short-history control band reaches p90 0.034, p99 0.114 and 2.5% above 0.075; isolated maxima reach about 0.50. | These handpicked margins are not that study's bootstrap intervals. Its top-5 versus top-64 statistic and launch settings differ from this top-50 checker and cache-enabled deployment. |
| Hero long reference | 0.35 / 0.09 / 0.04 / 2% | The same control study and its [65K–262K comparison](https://github.com/marin-community/marin/issues/11430#issuecomment-6882101992). Independent JAX layouts show their own numerical spread; vLLM's retained maximum reaches 0.289. | Limited books, layouts and runtimes. The margins do not certify every long history or quantify downstream quality. |
| Snowball reference | 0.25 / 0.125 / 0.05 / 5% | An initial selected-checkpoint screen. [#7354](https://github.com/marin-community/marin/issues/7354) and the [step42150 release gate](https://github.com/marin-community/marin/pull/11241) supply nearby regression evidence. | Different checkpoints and statistics cannot justify these exact cutoffs. The current 64-history pass shows this deployment fits them; it does not calibrate their rejection power. |
| Fully prepared cache branch and repeat | 0 / 0 / 0 / 0% | Exact agreement is required after the complete usable prefix is cached and survives a sibling branch. Both live deployments have nine such controls plus a repeat with zero error. An [injected branch-state alias](analysis/branch-checker-regression.txt) fails the corrected checker. | This checks persistence of returned next-token distributions. A wrong state that remains unchanged can escape it. The injected fault validates the check, not a nonzero tolerance. |
| Changed prefill steps | 0.20 / 0.12 / 0.06 / 15% | Initial engineering margins for fresh/warm, initial seed/branch and decode-created reuse versus fresh computation. Each deployment contributes 27 comparisons across nine boundaries. | The cutoffs were not derived from repeated healthy and faulty runtimes. Equal hit counts can conceal different prefill steps, so they do not establish an exact-comparison case. |
| Batch dependence | 0.20 / 0.18 / 0.15 / 50% | Initial engineering margins. Four requests must overlap on engine 0, with next-token scores compared against isolated execution. | Four comparisons are a small screen. They do not qualify maximum batch sizes or every distribution of load across engines. |
| Decode versus recomputation | 0.25 / 0.15 / 0.075 / 15% | Initial engineering margins. The shortest and longest saved histories below 4,096 tokens each produce 32 emitted-token comparisons against fresh scoring. | Only the emitted token's probability is checked. This does not calibrate long rollouts, sampling transformations or trainer probabilities. |

The [Hero report](analysis/hero-qualified-a2-summary.json) and
[Snowball audit](analysis/snowball-c64-root-audit.json) identify the
actual checked settings and immutable artifacts. Current observed statistics
are below; these observations demonstrate passes, not justified safety margins.

| Reference group | Positions | Maximum | p99 | p90 | Fraction above 0.075 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hero short | 11,370 | 0.560200 | 0.089248 | 0.027405 | 1.4600% |
| Hero long | 4,485 | 0.256977 | 0.053160 | 0.024455 | 0.3790% |
| Snowball | 1,770 | 0.196002 | 0.088755 | 0.030446 | 1.5254% |

The [retained observations](analysis/numerical-policy-observations.json) include
the earlier [full Snowball report](analysis/snowball-retention128-a1-summary.json).
Hero/Snowball observed maxima are 0.032986/0.080051 for changed prefill steps and
0.105459/0.041100 for emitted-token decode error. Hero's batch maximum is 0.007571; Snowball's
ranges from 0.042133 to 0.115348 across these passes. Each run contains only four
batch comparisons, so these observations provide limited evidence about
repeatability. All fully prepared branch and repeat
comparisons are zero. Raw observations, failed reports and policy hashes remain
available. Snowball's interrupted C64 attempt completed its checks but supplies
no completed performance window.

CPU and real HTTP tests validate the metric, identity checks, retained failures
and recommendation blocking. With a permissive synthetic changed-step bound,
the new exact control also catches an injected branch-state error that the older
checker misses. Production bounds reject that injected error in both versions.
That evidence validates checking behavior. It does not
establish that the nonzero limits separate acceptable numerical drift from a
harmful change. Hero's reference margins have the strongest prior control
evidence; Snowball's reference limits and all three nonzero path limits remain
initial screens. All nonzero limits retain provisional status.

Checkpoint or tokenizer mismatches, non-finite scores and missing required
coverage still fail. A provisional engineering recommendation must retain its
policy and exact runtime. Trainer/rollout log-probability alignment and RL
learning qualification remain deferred. No new calibration campaign was run
to write this explanation.
