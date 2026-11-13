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

## 2026-11-10: Prepared autosomal panels

Source commit: `e9c61b9c43883f966a57afbf20abba8f9eb0c869`, published before execution.
Run `python logbooks/11181-hero-vep/prepare.py --inputs INPUT_DIRECTORY
--output panels.json --seed 11181 --validation-groups 30` with the two pinned
parquet files named in `prepare.py`. Python 3.12 with pyarrow, single-threaded
execution. Start/end: 20:41:51 UTC; elapsed 0.59 seconds; exit 0; peak RSS
82,132 KiB. No remote model calls.

Found 493 complete autosomal missense groups. Prepared two chromosome-1
examples, ten pilot groups (100 variants, ten positives), and 30 disjoint fresh
development groups (300 variants, 30 positives). Source/harness identity,
label, allele, length, and reverse-complement checks passed for every selected
variant. Pilot and fresh validation windows do not overlap. Independent genome
verification remains pending.

`selected-variants.json` records identities, labels, groups, and sequence hashes;
`preparation-result.json` records counts and the reproducible full-panel hash.
The full sequence manifest remains a local derived artifact and can be
regenerated from the pinned inputs. The validation panel is reserved: no scores
have been computed, and its outcomes must not guide pilot prompt selection.

## 2026-11-10: Independent reference verification and scoring transport

Source commit `79f9823cc567a74b901f539ee47d5f8b176ff479` was published before
running `verify_reference.py --panels panels.json --output ensembl-reference.jsonl`.
The checker used nine sequential requests, at most 50 regions each, to the
[documented Ensembl sequence API](https://rest.ensembl.org/documentation/info/sequence_region_post),
explicitly requesting GRCh38. All 402 full 255-base reference windows matched.
The raw public responses and summary are archived alongside this notebook.
This completes the independent genome check left open in the previous entry.

The scoring client uses `/v1/completions` with echoed prompt log probabilities,
zero generated tokens, and token-ID output. It verifies echoed IDs and the full
continuation boundary, scores only observed target tokens, and stops on a model
or serving-fingerprint mismatch. A/B continuations must each be one contextual
token. Empty-prefix likelihood requires the tokenizer to supply a valid leading
context token; it must not silently omit the first DNA token. Exact requests and
responses belong in a private stream because model metadata may identify the
internal service. No credentials or endpoint URL are included in this archive.

`run_panel` prepares all four approaches, both strands, both answer mappings,
and an explicit-sequence format check on three positive and three control pilot
variants. It accepts an authenticated client supplied by the private launcher.
Each score is flushed immediately; only a completed matrix gets a completion
record. The client enforces caller-specified request and time limits and bounds
its tokenizer cache. It makes no assumption that prompt likelihood requests
reuse server KV cache.

Scoring transport is not live-validated. The intended endpoint remains
unavailable. No likelihoods, rankings, or model-quality metrics exist yet.

Five offline tests passed after adding the OpenAI completion parser and full
scoring target matrix (0.18 seconds; peak RSS 85,436 KiB). The repository lint
entry point passed for all changed source and evidence files. This validates
local parsing and prompt construction, not the live serving protocol.

## 2026-11-10: Score aggregation and native metric reporting

`aggregate.py` requires a complete score matrix, matches each record to its
exact prompt hash, and refuses duplicate or missing targets. It computes
reference-minus-alternate likelihoods and pathogenic-minus-benign answer
log probabilities, preserves strand differences, and averages reversed answer
mappings with the correct sign.

`report_metrics.py` imports the existing MarinDNA metric implementation instead
of reimplementing AUPRC or bootstrap sampling. The source file is checked against
SHA-256 `bee720324d61ea8b88a4644b6bd036569e9bab6f8e931d6335d165fb2e2b2395`.
Use the `snakemake/analysis/evals_v2/src` directory from the MarinDNA repository
at commit `922e41149c8f9bdb3130f780f66163a0241ff045` on `PYTHONPATH`; the hash check is authoritative if checking
out a newer revision. Execute the script using `uv run --locked --script` and
its adjacent dependency lock. Bootstrap sampling resamples whole match groups,
and method deltas use paired resamples. Ten-group pilot metrics are explicitly
exploratory. Standard subset/global/macro reporting retains the existing
30-group minimum. Within-group ranking counts ties as half a win.

Synthetic fixtures with 10 and 30 matched groups verified perfect-ranking AP=1,
zero paired deltas for identical methods, and the reporting threshold. These
are implementation checks, not model results. Both fixture runs exited 0;
peak RSS was below 161 MiB. Details are in `metrics-validation.json`.
Seven offline tests passed in 0.11 seconds (peak RSS 85,252 KiB).

The original successful service probe reported one internally accounted
completion token despite returning only prompt tokens. The parser now permits
that usage value while requiring the echoed token IDs to equal the input
exactly; a regression test covers both zero and one internally counted token.

The serving registry was checked again and currently reports no serving
endpoints. Public Slack searches have not identified a replacement for the
intended checkpoint. The experiment awaits a live endpoint. No deployment was
provisioned, no dataset inference was launched, and no model quality or caching
claim can be made. Once access is restored, confirm the execution budget, run
the small live protocol check, then the pilot; freeze methods before using the
reserved 300-variant panel. The archive index entry and final issue conclusion
remain pending with the experiment.

## 2026-11-13: Live SFT service and protocol correction

The user authorized proceeding on the newly available final SFT step 400407.
This replaces the earlier quick-SFT checkpoint; comparisons will identify the
new checkpoint explicitly. A live authenticated model-discovery request passed.
Exact service and access details remain private.

The initial tokenizer probe returned BOS token 128000 followed by eight tokens
for `ACGTACGTACGTACGT`. The service does not expose `/detokenize` (HTTP 404).
A direct completion request with those token IDs succeeded and returned finite
observed-token log probabilities. The response includes both OpenAI echoed
logprobs and native prompt logprobs with decoded observed-token strings.
The client now checks the target by joining those decoded strings, while still
requiring exact echoed token IDs and an unchanged contextual token boundary.
This removes the unavailable route without relaxing the scored-text boundary.
The raw probe responses remain in the private local execution record.

Initial authorized execution limits: one request at a time; a 10-variant smoke
with at most 2,000 HTTP requests and 15 minutes. A successful smoke precedes the
100-variant pilot; methods must be frozen before fresh validation.

The corrected protocol check passed at source
`540e5c48c13f897f7e85df5c2fd234a052006627`: 12 HTTP requests in 4.27 seconds.
The unconditioned 16-base DNA log likelihood was -25.168287590146065;
healthy-prefix DNA likelihood was -8.622630901634693. In an actual two-shot
query, candidate ` A` is token 362 and ` B` is token 426, with log probabilities
-2.3077545166015625 and -3.238219738006592. These values establish protocol
behavior, not VEP quality. Sanitized evidence is in `protocol-check.json`.
A 10-variant smoke on the fixed pilot manifest is now running sequentially under
the limits above. The metrics route is accessible; before the smoke, summed
prefix-cache query and hit counters were both zero, with no running requests.

The 10-variant sequential smoke completed: 272 scored continuations, 647 HTTP
requests, all HTTP 200, in 224.20 seconds. Full-matrix aggregation passed. One
serving fingerprint was observed. Completion-call latency was 0.400 seconds
median and 0.689 seconds p95. The service processed 91,512 prompt tokens; its
prompt-token counter rose by exactly that amount. Prefix-cache queries and hits
both remained zero. No cache savings are established for this prompt-logprob
path. See `smoke-10-accounting.json` and `smoke-cache-counters.json`.

The full pilot will use four independent clients, each with its own HTTP session,
tokenizer cache, and private wire log, under a combined cap of 12,000 HTTP
requests and 45 minutes. Scores are flushed under a write lock; failure stops
new targets and leaves the run without a completion marker. A fixture verifies
that one and four clients preserve the same complete matrix without duplicates.
All eight offline tests and lint passed. The 10-variant inputs will first be
repeated with four clients to measure live repeat variation before the full
100-variant pilot. Prompts and scoring formulas remain unchanged.
