# TSS-proximal DNA-likelihood pilot: 100 variants

Completed November 17, 2026. Raw DNA continuation likelihood only, with no text prefix versus the healthy-individual prefix; no chat template or A/B classification.

## Results

| Method | AUPRC | Match-group bootstrap SE | Within-group positive/control pairwise accuracy |
|---|---:|---:|---:|
| No text prefix | 0.140517 | 0.057157 | 56.67% |
| Healthy-individual prefix | 0.114875 | 0.043701 | 54.44% |

Ten complete matched groups: ten pathogenic positives and ninety matched common-variant controls. Prevalence reference0.10. Healthy minus null AUPRC = **−0.025642**, paired 95% group-bootstrap interval **[−0.105302,+0.010730]**, SE0.030704, two-sided bootstrap p0.247. Native pinned MarinDNA metrics,2000 group resamples,seed11181. Ten groups are below the standard30-group subset threshold, so these results are exploratory.

The healthy prefix does not improve the point estimate on this sample, but the interval includes zero: no reliable benefit or harm is established. Neither score demonstrates strong performance. No signs were flipped or methods selected after seeing the results.

For context only, the earlier missense pilot scored0.212520 (null) and0.301885 (healthy). Different variants/subsets and execution dates prevent a paired cross-subset comparison. Do not interpret the difference as a demonstrated biological-domain effect or combine these panels into one benchmark headline. Reserved missense validation remains unrun.

## What “promoter” means here

The exact pinned dataset category is **tss_proximal**, not a literal promoter annotation. Its curation rule recategorizes original intergenic, intronic, upstream or downstream consequences when the minimum distance to a protein-coding or noncoding transcript's TSS is <=1000 bp. It is not restricted to upstream DNA, protein-coding genes or experimentally validated promoters. Source references and pins are in [the frozen protocol](promoter-protocol.md).

The autosomal development/train pool contains163 complete groups (1630 variants). Selection uses seed11181, SHA256 group ranking in namespace `tss-proximal-group`, and takes ten complete groups without looking at model scores. Chromosomes1–22 only, no chrX/Y. The resulting panel spans chromosomes1,5,7,11,13,17,19; original consequences are31 upstream and69 intronic. Nearest-TSS distance range2–575 bp; median72 bp for both positives and controls. A noncoding TSS is strictly closer for33 variants. These metadata never enter the model prompt.

All100 reference sequences match independent Ensembl GRCh38 requests:98 unique windows because distinct alternate alleles can share a position. Center REF/ALT identity, single-site difference,255-base length and reverse complements were verified. Pinned input files match their original SHA256 values. Exact selected rows and sequences are in `promoter-panels.json`.

## Score definition

Same scorer as the earlier missense DNA evaluation:

- Null input: the uppercase255-base DNA sequence, no trailing newline.
- Healthy input: `The following sequence is from a healthy individual:\n` followed by that sequence, no trailing newline.
- Model-required BOS, no chat headers or assistant prefill. Sum DNA-token logprobs only; exclude natural-language prefix and EOS.
- Score reference and alternate on both strands; variant score is **negative mean strand LLR**, where LLR=logP(alternate)−logP(reference). Higher predicts pathogenicity. No token-count normalization.

All800 unique targets are complete and finite, with correct metadata alignment. Full-input prompt-logprob requests are used; no shared-prefix cache savings are assumed. Raw likelihood scoring is separate from the previously diagnosed answer-letter bias in classification.

## Execution, interruption and recovery

Initial frozen source: `c0b4aeecf50c678ede80c25b9cb1e276c27a00b5`. Recovery amendment and aggregation source: `09253f1f9e4cf2089509d32db64801ff670007ad`. Panel SHA256 `2315f0ce5ab392cb18897520683de16decf4a796efc85193bf3f86fb4f45ba77`.

The initial local runner exited143 after662 saved target scores and64 scheduled repeat checks. Cause is unknown. All1430 recorded HTTP responses were200; its last recorded request was tokenization. No guard failure or completion marker was recorded. Original scores and raw request logs remain unchanged.

Recovery rechecked all eight anchors against the original scores and required the same serving fingerprint/checkpoint, then scored only the missing138 targets and completed the remaining16 scheduled repeats. All80 scheduled repeats and eight recovery checks match exactly (delta0). No target score was replaced. Both attempts and the concatenated complete record stream are preserved with SHA256 hashes in the summary. The complete matrix passed aggregation checks before metrics were computed.

Approximate initial model-discovery timestamp18:34:54 UTC, recovery start18:53:13, final completion18:56:50 UTC on November17. Total elapsed1316.18 seconds (21m56s), including interruption/recovery preparation, within the original45-minute cap. In user-local Buenos Aires time this is approximately15:34:54–15:56:50.

1747 recorded service HTTP requests across both attempts (2 model discovery,857 tokenization,888 completions), all200, plus one separate availability preflight.888 completion calls comprise800 panel targets,80 scheduled repeats and8 recovery checks. API usage121350 prompt tokens and888 internally accounted completion tokens; max_tokens=0 requests use no generated text for scoring, but this build accounts for one internal completion token. Fixed rank, one request in flight, one-second completion gaps; max completion latency0.816 seconds. Same advertised final SFT checkpoint; fingerprint matches the Nov16 chat pilot. No independent weight checksum, batch-invariance guarantee, deployment change or active inference process.

Preparation peak RSS76480KiB,0.48 seconds,exit0. Metrics peak148920KiB,6.04 seconds,exit0. Intensive local work used the shared lock, single-thread limits,nice andionice; resource gates passed. Expected-sign synthetic aggregation fixture passed. Independent reference verification initially hit sandbox DNS restrictions; elevated execution succeeded before inference.

## Evidence and reproduction

- `promoter_likelihood.py`: deterministic preparation, scorer orchestration, bounded recovery, aggregation and native metrics.
- `promoter-protocol.md`, `promoter-panels.json`, `promoter-panel-summary.json`: frozen selection, definitions, exact sequences and metadata.
- `promoter-reference.jsonl`, `promoter-reference.summary.json`: independent reference evidence.
- `promoter-scores.jsonl`, `promoter-resume-scores.jsonl`, `promoter-complete-scores.jsonl`: immutable initial attempt, recovery and concatenation.
- `promoter-variants.json`, `promoter-metrics.json`, `promoter-summary.json`: per-variant scores, uncertainty and sanitized runtime/provenance.

Replay aggregation: `python promoter_likelihood.py aggregate promoter-panels.json promoter-complete-scores.jsonl OUTPUT.json`. Then `python promoter_likelihood.py metrics OUTPUT.json METRICS.json`, with the pinned MarinDNA evals_v2/src on PYTHONPATH and the previous report_metrics.py.lock environment. Metrics source hash is enforced. Raw wire responses, deployment routes and authentication metadata remain private.

## Decision

Record the requested promoter-like evaluation as completed on **TSS-proximal** variants. Neither prefix has an established advantage on this exploratory sample; the missense healthy-prefix point-estimate advantage did not recur here. Preserve both results and uncertainty. No additional panel or inference run is launched. Research branch only: no PR/main merge, training change or reserved-validation use.
