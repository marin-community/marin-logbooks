# Protocol v3: canonical single-token answers

Status: implementation and offline checks complete; full rerun unapproved and
unrun. Supersedes v2 answer-boundary and request semantics. Historical v1/v2
artifacts must not be relabeled or pooled with v3 classifications.

- Query prompt ends exactly at `Answer:` (no trailing whitespace).
- Candidate strings are ` A` and ` B`, observed token IDs 362 and 426.
- Tokenize each complete prompt-plus-candidate and require an unchanged prefix
  followed by exactly one candidate token. Discover IDs from those encodings.
- One completion request per context: `max_tokens=1`, `echo=false`,
  `logprobs=1`, and `logprob_token_ids=[A_id,B_id]`. No candidate masking or
  logit bias. The generated token is ignored; score requested probabilities.
- Score pathogenic minus benign log probability using the saved per-variant
  mapping. Average the two strand scores. Never flip signs based on results.
- Preserve seed 11181 and the balanced, label-independent mapping assignment.
  Zero/two-shot prompts differ only by the two demonstrations.
- DNA likelihood definitions and null/healthy prefixes are unchanged.
- Keep one client, fixed serving rank, one-second completion gap, and existing
  latency/drift guards. Companion results may be consumed once; repeated
  sentinel pairs must make fresh calls.
- Materialize protocol-v3 manifests and reference-summary hashes before any
  run. The runner and aggregator reject v2 manifests.
- Before a full classification-only rerun, verify the new scorer against a
  small teacher-forced leading-space baseline and confirm serving identity.
  Reusing old DNA scores requires explicit provenance and identity checks.

For 100 variants: 2 methods × 2 strands = 400 classification responses,
producing 800 candidate-score records. Tokenization and sentinel calls are
additional. Current generic runner also scores DNA; a classification-only
launcher remains to be prepared before the approved scope can be executed.
