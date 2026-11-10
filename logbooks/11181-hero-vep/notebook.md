---
marin_base: 6f62a629e157b91cb170d6733e959f48b5d88906
archive_template_revision: b6d087a1bfa92ee89acbb57f8122e3cddd81c452
issue: https://github.com/marin-community/marin/issues/11181
---

# Marin Hero on variant effect prediction

Coordinating issue: https://github.com/marin-community/marin/issues/11181

Marin base: `6f62a629e157b91cb170d6733e959f48b5d88906`.
Research branch: `codex/11181-hero-vep`.

## 2026-11-10: Development inputs and serving availability

Compare unconditioned DNA likelihood, healthy-individual-conditioned likelihood,
and zero-shot/two-shot A/B classification. The first panel has ten complete
missense match groups (100 variants). Freeze the approach before a larger fresh
development comparison; do not use the dataset's held-out test split.

The original inference service is unavailable. Discovery has not identified a
replacement serving the intended model. No experiment inference has run, and no
new deployment has been launched. Exact service details remain in a private
local note. Model identity and tokenization must be verified before execution.

Downloaded the development variant table and its paired, materialized 255-base
sequence table. The input revisions and file hashes are fixed in `prepare.py`.
The materialized sequences permit center-allele and reverse-complement checks;
independent reference-genome verification is still required.

An initial proposal to reserve chromosome X for demonstrations was rejected:
chromosome X is outside the intended distribution. All selection pools now use
chromosomes 1–22 only. Reserve chromosome 1 for two demonstrations, one positive
and one control in separate groups and non-overlapping windows. Draw pilot and
fresh validation groups from the remaining autosomes. Selection uses SHA-256
ordering with seed 11181, independent of model scores. Reserve 30 groups (300
variants) for fresh validation, excluding groups with windows overlapping the
pilot. This is a preparation target, not a claim that validation has run.

`prepare.py` verifies input checksums, complete groups, source/harness labels,
strand pairing, center-only substitutions, and sequence lengths. It writes
variant identities, metadata, sequences, and per-sequence checksums.
`templates/` contains the exact approved prompt text. `prompts.py` renders the
classification variants and rejects candidate tokenizations that cross the
conditioning boundary or require more than one token.

A synthetic selection test passed (one test, 0.05 seconds; peak process RSS
80,836 KiB). It checks autosomal exclusion, class balance, complete groups,
pilot/validation separation, and reproducibility under input reordering. This
is implementation validation, not a dataset preparation or model result.

Pending: finish and validate scoring transport; archive source before running
preparation; independently validate genome windows; identify the intended live
model; verify BOS and continuation boundaries against its tokenizer; execute the
pilot with a bounded request/time budget; freeze methods and run fresh
validation; publish uncertainty estimates and the resulting decision.

The three offline tests subsequently passed (0.09 seconds; peak RSS 81,188 KiB),
including reversed option/answer mapping, reverse-complement alleles, query-label
independence, and scored-token selection. The shared interpreter lacks pytest-xdist;
the successful invocation retained the full default marker exclusions and session
timeout guard while removing parallel-worker flags.
