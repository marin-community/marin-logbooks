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


## 2026-11-13: Repeatability gate fails before full pilot

Source `0c32deba687ce5149360effca7b9471db7ab16d6` produced a complete
four-client repeat of the same ten variants: 272 scores, 652 HTTP requests,
75.63 seconds. An earlier four-client attempt ended with SIGTERM after 223
scores and 538 successful HTTP responses; its cause is unknown and its partial
records are preserved privately. The retry was bounded to 800 requests and
300 seconds. Neither partial results nor the retry replace the sequential run.

Complete-matrix aggregation passed for both completed runs. Input hashes,
target token IDs, prefix boundaries, and input lengths match exactly. Across
544 completion responses, one serving fingerprint was observed and native
observed-token logprobs agreed exactly with echoed logprobs. Nevertheless,
target likelihoods differed by up to 3.8038 natural-log units (mean absolute
difference 0.2374). Maximum strand-averaged variant-score differences were
2.8781 (null), 0.5401 (healthy), 0.8898 (zero-shot), and 0.2611 (two-shot).
Each method's ordering of the ten variants changed. See
`smoke-repeat-comparison.json`. These are repeatability diagnostics, not
estimates of VEP quality.

A bounded follow-up replayed the first eight completion payloads from the
sequential smoke, each three times sequentially (24 requests, 120-second cap),
using the same published ScoringClient and private authentication. The eight
cases are forward-strand null/healthy reference/alternate plus zero-shot A/B
under both mappings for one autosomal variant. Target-only likelihood ranges
were 0.0445–0.6876; primary A ranged by 0.5450 and primary B by 0.4433.
`sequential-stability.json` records every target score. This replay used an
ad-hoc diagnostic command rather than a separately published experiment
configuration; original wire requests and responses remain private and replayable.
Do not confuse whole-input likelihood variation with target-only variation.

The endpoint is available and accepts the required protocol, but the current
blocker is unexplained score variation under identical requests. Sequential
requests alone do not remove it. This does not establish a specific serving
bug or its cause: global batching, numerical behavior, and routing remain
unverified. No serving configuration was changed, no deployment was launched,
and the full 100-variant pilot and reserved 300-variant panel remain unscored.
Next gate: establish a stable scoring route or quantify a justified repeat
protocol before comparing prompt methods. Preserve compilation and CUDA graphs.


## 2026-11-13: Isolate rank and batch dependence

The follow-up request asked for the cause and a fix. Live read-only startup
inspection identifies BF16, no quantization, TP=1, DP=16 with expert parallelism,
chunked prefill, prefix caching disabled, and compilation plus PIECEWISE CUDA
graphs enabled. All four task nodes report vLLM source
`3004cd8f163299116b96d1b324cfff418c387388`. No job or serving settings changed.

The deployed completion handler supports `X-data-parallel-rank`, as also used
by Marin's existing serving-parity harness. A 24-completion diagnostic replayed
two fixed prompts three times on each of ranks 0, 1, 4, and 8. Every within-rank
repeat was exactly equal. Different ranks produced different results:
for the same A continuation, rank means ranged from -2.7556984 to -2.4324777.
See `rank-stability.json`. This directly implicates rank selection in the
previous apparent sequential nondeterminism; it does not prove different weights.

A broader check pinned rank 0 for two sequential passes and one four-client
pass over 36 identical targets (108 completions). Targets cover two autosomal
variants, both strands, null/healthy likelihood, zero-/two-shot primary-map
classification and the selected explicit-pair format check. One variant was
chosen because its earlier DNA discrepancy was largest. All 36 targets matched
exactly across the sequential passes. Concurrent results differed by up to
5.6548 DNA log-likelihood units and 0.6612 single-answer log-probability units.
See `rank0-matrix.json` (columns: sequential 1, sequential 2, concurrent) and
`rank0-matrix-summary.json`. No chromosome-X or reserved validation inputs were used.

These interventions establish dependence on both selected rank and batching.
The particular kernel/reduction responsible is not localized. BF16 numerical
order, MoE routing/reductions and rank-local kernel choices are hypotheses;
we have not compared parameter hashes or intermediate activations across ranks.
The stable system fingerprint identifies the build, not a specific execution
rank or batch. Tokenization and client-side summation are not the source.

The immediately testable workaround is one fixed rank plus sequential calls,
with repeated sentinels before/during/after scoring. This achieved exact repeat
agreement in the bounded check but is not guaranteed on a shared service:
other traffic and expert-parallel activity may change the effective batch.
Do not average arbitrary unpinned repeats and call the result corrected.

The durable serving experiment is to enable `serve.vllm_batch_invariant: true`
(the existing Marin configuration maps this to `VLLM_BATCH_INVARIANT=1`) on an
approved scoring deployment, then rerun rank, batch-size, order, and latency
checks before the pilot. The deployed version's batch-invariance documentation
does not list Hero/GrugMoE among validated models, so the flag is a candidate
fix, not proof. Preserve compilation and CUDA graphs. If it fails for this
custom model, compare replicated parameter hashes and per-layer activations
across ranks, then localize attention, router, expert and reduction kernels.
A reserved fixed-rank/fixed-batch scoring route is the fallback, subject to
resource approval. No new deployment or restart is authorized by these findings.

The newer release `0ac9dcb157ee` adds Snowball schema-1 BF16 RMSNorm rounding,
FP8 padding-scale and skipped ShortConv decode-row fixes. The running service
is unquantized Hero and this task scores prefill tokens, so none is established
as the cause or sufficient fix. Track adoption in Marin #11241; do not equate
upgrading with validated repeatability. Sources:
- https://github.com/marin-community/vllm/compare/3004cd8f163299116b96d1b324cfff418c387388...0ac9dcb157ee
- https://github.com/marin-community/vllm/blob/3004cd8f163299116b96d1b324cfff418c387388/docs/features/batch_invariance.md

For the earlier ten-variant comparison, pairwise ordering flips out of 45 were
3 (null), 1 (healthy), 13 (zero-shot), and 7 (two-shot). The single pathogenic
variant's ranks were respectively 1→1, 3→2, 7→5, and 9→8. These describe only
one matched group; no method-quality conclusion follows.

A final 36-call sequential pass after the concurrent intervention recovered
the original rank-0 baseline exactly (maximum absolute difference 0.0).
See `rank0-recovery.json`. This supports batch-dependent numerics rather than
persistent contamination by preceding requests in this diagnostic. All
diagnostic processes exited; full pilot and validation remain unrun.


## 2026-11-13: Approved paced ten-variant trial

The user approved a paced trial before considering the full pilot. Run exactly
ten existing pilot variants, one client, pinned rank 0, with a minimum one-second
gap after each completion before the next completion. Cap all HTTP requests at
1,000 and elapsed time at 900 seconds. No retries or deployment changes.
The six forward-strand sentinels from the first pilot variant cover null/healthy
reference DNA and A/B under primary zero-/two-shot classification. Score them
before the panel, every 48 panel targets, and after the panel. Stop immediately
on absolute drift above 1e-6, any request/protocol failure, any completion over
five seconds, or the median of the latest five completions exceeding both 1.5
seconds and three times the first eight completions' median. Pause time is
excluded from measured request latency. These are client-side load limits,
not proof of no impact on other users.

`paced.py` buffers the small panel score stream until its final check succeeds;
on failure it preserves the partial stream with an invalid marker. Raw wire
and sentinel records flush throughout. No full pilot is launched by this trial.


The approved paced trial completed successfully at source
`e452dc2bb72a63479124e9bb108f01e91ece9a6f`: 458.00 seconds (7m38s),
706 HTTP requests, all HTTP 200. There were 272 panel targets and 42 sentinel
calls (six baselines plus 36 repeats), for 314 completion calls in total.
All sentinel repeats matched exactly; all 36 overlapping targets from the
earlier fixed-rank sequential diagnostic also matched exactly. Full-matrix
aggregation passed for all ten variants. See `paced-trial-summary.json`,
`paced-trial-sentinels.json`, and `paced-trial-variants.json`.

Completion latency excluding pacing was 0.408 seconds median, 0.649 seconds
p95 and 0.941 seconds maximum; no stop threshold triggered. Mean completion
rate including pauses and setup was 0.686 requests/second, with one request
in flight and 105,911 prompt tokens processed. This is comparable to the
previous sequential smoke's 0.400-second median and 0.689-second p95; it is
not a measurement of latency or GPU utilization for other endpoint users.
Expert-parallel work is shared, so rank pinning is not resource isolation.

This passes the agreed ten-variant stability/load gate. The same paced policy
is supported for a bounded continuation, retaining all stop conditions and
sentinel checks. The full 100-variant pilot and reserved validation panel have
not been launched in this trial. No serving settings or deployment changed.
Ten offline tests and required lint passed for the guarded runner; raw HTTP
responses and the private launcher remain in the local private execution record.


## 2026-11-13: Full pilot continuation after passed gate

The active goal resumes completion of #11181. The ten-variant paced gate
passed, permitting the planned full pilot under the accepted load policy.
Run all 100 fixed pilot variants with rank 0, one client, one-second completion
gaps and the unchanged six-sentinel/latency guards. Cap at 10,000 total HTTP
requests and 5,400 seconds (90 minutes); stop on any guard failure without
retry. This consumes the existing endpoint only, with no deployment changes.
Generalize the guarded runner to explicit panel and variant-count arguments;
keep prompts, candidates, scoring formulas and sample selection unchanged.
The reserved 300-variant panel remains unscored until protocol freeze.


The full pilot launched from `4cbc70b7cb` with the stated pacing and limits.
Its first eleven periodic sentinel rounds through 528 panel targets matched
exactly. The process remains live; these partial observations are not results.

The reusable contribution is now [Marin PR #11375](https://github.com/marin-community/marin/pull/11375),
a draft at `b28487e78e2422032ace32b5f48812e610f36fc2`, based on Marin
`246f795749166285a864e891e56facfbcf9127e7`. It contains manifest generation,
reference verification, exact templates, a library likelihood client, a
sequential guarded runner and aggregation. Historical probes and result files
remain in this research branch. The contribution streams partial score records
and writes completion only after its final sentinel check.

The extracted generator reproduces manifest SHA256
`d13451cade17f456327a135f10dceb1c2d189fbfa9605e1ab217cab9e8af69ce`.
Ten focused offline tests passed; the pure likelihood-parser tests omitted
unrelated heavyweight root fixtures. Replaying 706 saved real responses through
the extracted runner reproduced all ten trial aggregates exactly and made no
live endpoint requests (peak RSS 127,552 KiB). All six advisory review sections
ran sequentially; findings about typed results and duplicated constants were
resolved. Repository-wide type checking and the broader affected suite await
CI because the shared VM cannot safely run their full environment. The PR
body and initial feedback were read back, and an event-driven watcher is active.
Archive index setup remains pending; no duplicate checkout operation was started.


## 2026-11-13: Analysis preparation and contribution CI

Marin PR #11375 completed CI with no failed or pending checks at contribution
commit `b28487e78e2422032ace32b5f48812e610f36fc2`. The PR remains a draft
and has no review feedback yet. The archive index entry is now proposed in
[logbooks PR #41](https://github.com/marin-community/marin-logbooks/pull/41).
Neither PR has been merged.

Add `report_sensitivity.py` for offline summaries of forward/reverse score
differences, label-mapping score shifts and matched-pair order flips, the six
format checks, and each positive variant's rank among its nine matched controls.
Ties receive average ranks; strict order flips exclude ties and report their
count separately. These diagnostics complement native grouped-bootstrap AP
metrics and identify representative failures without additional model calls.
The full pilot is still running; no quality conclusions follow from preparation.
