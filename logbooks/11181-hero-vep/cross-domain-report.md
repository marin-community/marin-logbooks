# Cross-domain review controls and instruction order

Completed November 16, 2026. This probe asks whether the VEP label dependence is DNA-specific and whether it follows the last label mentioned in the response instruction.

## Findings

1. **Clear sentiment is robust on these authored examples:** 64/64 correct across four clear reviews, both mappings, both list orders, both instruction mention orders and short/structured context. No factor changes the semantic prediction on a clear review. This is four reviews, not 64 independent examples or a broad sentiment benchmark.
2. **Ambiguous sentiment reproduces mapping sensitivity outside DNA:** class-to-letter swaps change semantic preference in 6/16 matched ambiguous-review pairs; list order and mention order also each change it in 6/16 pairs. Adding structured context changes it in 2/16. These are sensitivity findings, not accuracy failures: the two ambiguous reviews have no answer key.
3. **VEP bias does not simply follow the last-mentioned letter:** B wins all 32 VEP contexts, including the instruction “B or A.” Mention-order reversal changes confidence but no preferred letters. Class-to-letter swaps change semantic preference in all 16/16 VEP pairs.

## Intuitive example

Review (item 5):

> Some features are good and some are bad. Overall I have mixed feelings about it.

Short-context prompts have the same question, the same instruction “Respond with exactly one letter: A or B.” and the same B-then-A list order. Only the class-to-letter assignment changes:

| Displayed option list | Preferred letter | Preferred sentiment | Conditional P(positive given A/B) |
|---|---|---|---:|
| B: negative; A: positive | B | Negative | 21.44% |
| B: positive; A: negative | B | Positive | 53.22% |

The review text does not change. This is the same type of semantic mapping dependence seen in VEP, but it is not proof that either sentiment is objectively wrong. No renaming-induced error occurred on the clear positive/negative controls.

The mixed review also flips with the ordinary A-then-B list order in the structured-context condition: conditional P(positive) moves from 19.93% to 55.83%, again preferring B both times. Exact complete prompts and scores are archived; metadata is identical within each pair and explicitly irrelevant to sentiment.

A second example is “The item arrived on Tuesday. I have not opened or used it yet.” Its short A-then-B, A-or-B condition changes P(positive) from 47.90% to 68.00% when the mapping swaps, preferring B both times. This input contains no direct evaluation of the product.

## Matched comparisons

| Domain / intervention | Semantic flips | Median absolute change in conditional target probability |
|---|---:|---:|
| All reviews: class-to-letter mapping | 6/48 | 3.27 percentage points |
| All reviews: option-list order | 6/48 | 4.22 points |
| All reviews: instruction mention order | 6/48 | 3.49 points |
| All reviews: structured context | 2/48 | 2.84 points |
| VEP: class-to-letter mapping | 16/16 | 27.88 points |
| VEP: option-list order | 0/16 | 5.47 points |
| VEP: instruction mention order | 0/16 | 3.02 points |

All review flips are among the 16 ambiguous pairs per intervention; none among the 32 clear pairs. “Target” is positive for reviews, pathogenic for VEP. Probabilities are renormalized over A/B, not calibrated probabilities of sentiment or pathogenicity. Shared items make these contrasts dependent; no significance claim is made.

Short review prompts have 64–67 tokens; adding 12 neutral metadata rows produces 343–346 tokens, exceeding these VEP prompts' 270–278 tokens. Clear-review accuracy stays perfect. This argues against prompt length or this particular structured distraction alone being sufficient to cause the VEP behavior; it does not rule out other long-context or structured-input failures. Length and presentation changed together.

Mean A/B mass falls with added metadata: clear positive 94.73% to 90.36%, clear negative 94.42% to 88.79%, ambiguous 90.26% to 83.25%. Successful classification does not imply unchanged confidence. VEP mean A/B mass is 94.64%.

## Interpretation

DNA is not necessary for semantic mapping dependence: it appears in ordinary ambiguous reviews. The pattern is consistent with answer priors having more influence when task evidence is weak, but no internal uncertainty measure was collected and this does not prove that mechanism in VEP. We have not shown that the model possesses the biological information needed to classify these variants from a short sequence window.

The earlier “last-mentioned label” hypothesis is not a sufficient explanation of VEP: reversing the instruction does not change any of its 32 preferred B outputs. Prior C/D results also show the behavior is not unique to token B. A model-wide rule, training-data imbalance or a specific training bug cannot be inferred from this small set.

This is now suitable to communicate as an **evaluation robustness issue with an intuitive non-DNA example**, while retaining the clear-review successes and ambiguity caveat. Do not describe it as general inability to follow A/B mappings. More VEP examples alone will not repair the confound. No Slack message has been sent.

## Protocol and execution

Frozen source/protocol/inputs: `b6b2c1ae2c095df1acecc574a97270c883de6e95`. Input SHA256 `5b8881335dc7a8667b3ec8e35f58f3ab791cb3f956e795983444809ab3b47262`. Exact six authored review strings and complete rendered prompts are in `cross-domain-inputs.json`. VEP reuses the prior four autosomal variants, real DNA, forward strand only. No chromosome X or reserved validation data.

Run 21:11:09–21:15:28 UTC, November 16: 259.52 seconds. 128 scores plus four repeats, 132 generated tokens, 384 matching live tokenization checks and one model discovery = 517 successful HTTP requests. Maximum completion latency 0.546 seconds. Four repeat checks have zero drift. Same advertised checkpoint and serving fingerprint as the prior VEP-factor run; no independent weight checksum. Sequential fixed rank, one-second gaps; no guard fired. No inference process remains active.

Preparation: 0.73 seconds, peak RSS 222,008 KiB, exit 0 under shared lock, single-thread/nice/ionice limits. Lightweight standard-library analysis validates all indices, completion marker and repeat checks. The runner imports the previously frozen `vep_factor_diagnostic.run`; the wrapper replaces its fixed completion counts with the current 128/4 counts. Scores are unmasked candidate logprobs, not sampled-letter counts. This probe uses Answer: prefill, not free reasoning.

Public artifacts: `cross_domain_probe.py`, `cross-domain-protocol.md`, `cross-domain-inputs.json`, `cross-domain-scores.jsonl`, `cross-domain-analysis.json`, `cross-domain-summary.json`. Replay: `python cross_domain_probe.py analyze cross-domain-inputs.json cross-domain-scores.jsonl OUTPUT.json` from the logbook directory. Private wire/access details remain outside Git.

## Research state

Research branch only; no PR, main merge, training change, new deployment or benchmark rerun. Reserved validation untouched. This bounded probe is complete. Next work should focus on a scoring protocol robust to answer-label changes, with held-out checks before any VEP scale-up; task-specific prompt tuning on these examples would not establish general robustness.
