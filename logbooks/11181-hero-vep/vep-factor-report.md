# VEP label, option-order and DNA sensitivity: November 16, 2026

## Result

On four previously selected variants, **B is preferred in 32/32 A/B contexts and D in 31/32 C/D contexts**. Both displayed option orders, semantic assignments, and real/shuffled DNA are included. This is not specific to token B and is not simply choosing the last option in the displayed list. It remains specific to this VEP prompt family; the earlier easy-question prefill succeeds under swapped labels.

The instruction always says “A or B” or “C or D,” so alphabetically later symbol and last-mentioned symbol in that instruction remain confounded. The experiment does not identify the underlying training or prompt mechanism. No new training defect is established.

| Matched intervention | Semantic preference changes | Median absolute change in conditional pathogenic probability | Median absolute pathogenic-margin change |
|---|---:|---:|---:|
| Swap class-to-letter assignment | 31/32 | 24.05 percentage points | 0.9836 logprob units |
| Reverse displayed option-list order | 1/32 | 7.30 points | 0.3112 |
| Replace A/B with C/D | 1/32 | 3.65 points | 0.1577 |
| Shuffle DNA flanks | 1/32 | 2.10 points | 0.0908 |

Each comparison holds all other factors fixed. Semantic margins always use pathogenic-minus-benign under the saved mapping. These are repeated conditions on **four variants**, not 32 independent biological examples. Comparisons are descriptive, with no accuracy, AUPRC or significance claim. Conditional probabilities renormalize only the candidate pair and are not calibrated pathogenicity confidence.

## Letter, position and semantic preferences

| Labels / DNA | Later alphabetic symbol preferred | Last-listed option preferred | Pathogenic preferred |
|---|---:|---:|---:|
| A/B, real | 16/16 | 8/16 | 8/16 |
| A/B, shuffled | 16/16 | 8/16 | 8/16 |
| C/D, real | 16/16 | 8/16 | 8/16 |
| C/D, shuffled | 15/16 | 9/16 | 7/16 |

Last-listed wins 33/64, while the later symbol wins 63/64. Mapping swaps change semantic preference in 16/16 real-DNA pairs and 15/16 shuffled pairs. The sole first-symbol preference chooses C narrowly: `13:107866379:T:C`, shuffled DNA, C=benign/D=pathogenic, D listed before C; conditional pathogenic probability 49.12%.

List order changes confidence even when the preferred letter stays fixed. Neither list reversal nor C/D labels fixes mapping dependence here.

## DNA sensitivity

The 255-base sequences retain their center reference base and VCF substitution. Each 127-base flank is independently shuffled once per variant, preserving its length and base counts; all other prompt text is identical within a DNA contrast. Local sequence structure is destroyed and the result may be biologically implausible. Shuffled inputs have no known pathogenicity label.

DNA shuffle changes conditional pathogenic probability by median 2.10 points, mean 2.29, maximum 7.82. This is smaller than mapping and list-order effects on these inputs. It establishes string sensitivity, not functional biological understanding. The central allele/composition remain unchanged; one shuffle per variant cannot prove DNA is ignored.

## Execution and provenance

Frozen source/protocol/inputs: `c98d68adea8e1a9966b4f5fd599316a08569b547`. Input SHA256 `b091cf8db007a698a78b5b9950a5440dfa57605469fe191b28e6d1706bc79787`. Reuse the first control and positive from each of the first two frozen pilot match groups, forward strand, from the prior chat-replication manifest: `13:107866379:T:C`, `13:113118831:T:G`, `11:533875:G:T`, `11:112025527:G:C`. No score-based selection, chromosome X or reserved validation inputs.

64 fully crossed contexts plus four repeats, execution order randomized with seed 11181. Leading-space candidates: A/B=362/426, C/D=356/423. All 192 live tokenization checks pass. Advertised checkpoint and serving fingerprint match the chat pilot; no independent weight checksum.

Run: November 16, 21:01:20–21:03:33 UTC, 133.40 seconds. All 261 HTTP requests succeeded; 68 generated tokens. Fixed rank, sequential requests, one-second completion gaps. Maximum completion latency 0.612 seconds; no guard fired. Four repeats have zero logprob drift; all four matching prior pilot baseline contexts also match exactly. No inference process remains active.

Offline preparation: 0.61 seconds, peak RSS 222,840 KiB, exit 0, shared lock and single-thread/nice/ionice limits. Baseline-input equivalence and every flank-composition check passed. Analysis is bounded standard-library processing.

Artifacts: `vep_factor_diagnostic.py`, `vep-factor-protocol.md`, `vep-factor-inputs.json`, `vep-factor-scores.jsonl`, `vep-factor-analysis.json`, `vep-factor-summary.json`, `vep-factor-baseline-comparison.json`. Replay: `python vep_factor_diagnostic.py analyze vep-factor-inputs.json vep-factor-scores.jsonl OUTPUT.json`. Raw wire responses, endpoint and authentication metadata remain private.

## Decision

Report **strong mapping dependence in these direct-answer VEP prompts**, despite stable scoring and successful easy-question controls. More DNA examples alone would not remove this confound. This is useful evaluation evidence, not proof of universal B preference or a demonstrated need for training.

A next bounded control could vary response-instruction mention order (A-or-B versus B-or-A) independently of list order; direct semantic answer scoring is another possibility but requires explicit variable-length token handling. Neither has been run. Keep VEP scores exploratory and validation untouched. Research branch only; no PR or main merge.
