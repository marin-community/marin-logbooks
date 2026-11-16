# VEP label, order and DNA sensitivity diagnostic

Frozen before execution, November 16, 2026. User approved follow-up to the easy-question diagnostic. Research branch only; no training, deployment, PR, main merge or reserved validation use.

## Design

Reuse the same four variants from the prior chat replication: first control and positive in each of the first two frozen pilot match groups, forward strand only. No selection based on scores; no chromosome X. Fully cross within each variant:

- Labels: A/B versus C/D.
- Semantic assignment: first symbol pathogenic / second benign, or reversed.
- Option-list order: first symbol listed first, or second symbol listed first.
- DNA: original versus shuffled left and right flanks. Shuffle the 127 bases on each side independently with seed 11181 + source variant index, preserve the center reference base and VCF REF/ALT substitution. Preserve each flank's base counts and total length, but not motifs, dinucleotides or biological validity. One shuffled realization per variant, shared across every format condition.

This is 4 x 2 x 2 x 2 x 2 = 64 contexts. Zero-shot only, pinned chat template, assistant `Answer:` prefill. The original text and compact FASTA/VCF representation are unchanged except the declared interventions. The instruction's symbol mention order stays canonical (`A or B` / `C or D`) while the option-list order varies: estimates concern list order, not every occurrence of the symbols. Candidate single-token boundaries are verified offline and live. All exact prompts, IDs and mappings are in `vep-factor-inputs.json`; source `vep_factor_diagnostic.py`.

Randomize execution order with seed 11181. After all 64 contexts, repeat the four original-DNA, A/B, primary-mapping, forward-list contexts. Use unconstrained next-token logprobs of both candidates from one response; sampled output is evidence only, not the score. Convert the margin to pathogenic-minus-benign according to each saved mapping, without outcome-based flips.

## Budget and stop criteria

68 one-token completion requests (64 + 4 repeats), 192 live tokenization requests, one model discovery: at most 261 HTTP requests, 68 generated tokens, 600 seconds. Fixed rank, one request in flight, one-second completion gap. Stop on error, tokenizer mismatch, serving fingerprint change, completion latency above 10 seconds, or repeated logprob delta above 1e-5. Confirm advertised checkpoint matches the preceding pilot. No automatic expansion or retry. Access details and wire metadata remain private.

## Analysis and interpretation

For each intervention, compare the two levels while holding all other factors fixed: 32 paired contrasts each for list order, symbol set, DNA shuffle and semantic assignment. Report semantic preference changes, signed/absolute pathogenic-margin differences and conditional A/B or C/D probability differences. Report literal-symbol, list-position and semantic preferences separately. No accuracy or AUPRC estimate: four cases are diagnostic, and shuffled inputs have no known pathogenicity labels.

A shuffle response establishes string sensitivity, not functional biological understanding. A small response to one shuffle is not proof that DNA is ignored. List order, symbol changes and instruction text may interact; compare matched contrasts rather than attributing all effects to a single universal bias. Retain individual outputs and failures. Compare matching baseline contexts with previous pilot scores only as secondary reproducibility evidence, checking serving fingerprint separately.
