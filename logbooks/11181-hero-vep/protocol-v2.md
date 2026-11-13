# VEP protocol v2

This is the revised branch-only experiment protocol. The v1 pilot and its
freeze remain historical evidence at commit
`f7592176b0bb20b25509ef9019ecf4f6163894b6`; they do not evaluate this revision.
The original v1 validation freeze is superseded. No v2 benchmark result or
300-variant validation result is claimed.

## Answer conditioning

Every classification prompt ends in `Answer: `, including one trailing ASCII
space. Encode that full prefix first. Encode the bare answer `A` or `B`
separately, remove the same BOS prefix returned for the empty string, verify
that exactly one answer token remains, and append that token ID to the prefix
IDs. Score only its observed log probability. Do not re-encode the concatenated
text, which can merge space and letter into another token. Verify the native
returned answer decodes to exactly the requested bare letter. The shared space
is conditioning context and is excluded from the score.

## One mapping per benchmark variant

For each of the fixed pilot and validation panels separately, rank variant IDs
by SHA-256 using seed 11181 and namespace `answer-mapping-v2`. Assign the first
half A=pathogenic/B=benign and the second half the reversed mapping. Selection
uses variant identity only, never labels or model scores. Store `answer_mapping`
on each query row, plus protocol version, seed and assignment rule in the
manifest. Both methods and strands reuse that assignment. Demonstration answer
letters follow the query mapping. The benchmark reports one pathogenic-minus-
benign score per method, averaged over strands; there is no mapping-average
benchmark metric. Exact balance is across each panel, not conditioned on class.

The main matrix is 16 scores per variant: eight DNA and eight classification.
The 100-variant pilot has 1,600 targets; validation has 4,800. Both-mapping
classification checks are available through `mapping_diagnostic_targets` on a
small separately selected diagnostic subset, outside benchmark aggregation.
The previous explicit-pair pilot diagnostic is not part of the v2 benchmark.

## Shared zero-/two-shot template

`templates/classification.txt` defines the identical instructions, options,
query and answer suffix. `templates/variant.txt` defines the common FASTA/VCF
block. Zero-shot supplies an empty examples section; two-shot inserts the two
fixed labeled examples in their original order. Removing that section from a
two-shot prompt must produce the zero-shot prompt byte-for-byte, including the
trailing space. No sequence, locus, chromosome selection or demonstration
identity changes.

## Validation and history

Fifteen offline tests pass, including a tokenizer that would merge ` A`/` B`
if the whole input were re-encoded, both semantic mapping directions, balance
and label/order-independent assignment, matched query/template bytes, complete
matrix coverage and final-guard failure handling. The tests use one worker,
retain excluded live-test markers and timeout guards, and omit unrelated root
fixtures. Required lint passes. Live token-path verification and v2 manifest
materialization are recorded in subsequent notebook entries.

All work remains on the research branch. No PR, main-branch merge or deployment
change is part of this protocol revision.
