# Expanded train DNA-likelihood experiment: 300 variants per subset

Completed November 17, 2026 on the same advertised Hero SFT step-400407 checkpoint. Raw DNA likelihood only: no text prefix versus the healthy-individual prefix, no chat template or classification. Both panels contain 30 complete matched groups: 30 pathogenic positives and 270 common-variant controls each.

## Results

| Subset | Prefix | AUPRC | Match-group bootstrap SE | Within-group pairwise accuracy |
|---|---|---:|---:|---:|
| Missense | None | 0.178114 | 0.035099 | 61.11% |
| Missense | Healthy individual | 0.188561 | 0.044500 | 60.74% |
| TSS-proximal | None | 0.150101 | 0.049415 | 50.00% |
| TSS-proximal | Healthy individual | 0.140862 | 0.044040 | 49.26% |

Healthy minus null AUPRC:

- **Missense: +0.010447**, paired 95% group-bootstrap interval **[−0.020174, +0.070241]**; SE 0.024013.
- **TSS-proximal: −0.009239**, paired 95% group-bootstrap interval **[−0.043155, +0.007190]**; SE 0.013196.

Neither interval establishes a healthy-prefix benefit or harm. Missense discrimination is modest; TSS-proximal within-group ranking is approximately 50%. Prevalence is 0.10 in each panel. Do not combine these distinct subsets into a single benchmark headline or infer a causal cross-subset effect. Metrics use the pinned native MarinDNA implementation, 2,000 matched-group resamples, seed 11181. [Exact metrics](expanded300-metrics.json), [600 per-variant scores](expanded300-variants.json).

## Selection and interpretation

Train split only, autosomes only, no chromosome X/Y. Each panel retains its original 100 variants and adds 20 complete groups, chosen deterministically without using scores. Overlapping windows were not excluded, as requested. The missense additions include 130 rows from the formerly reserved development panel: this is an expanded exploratory experiment, not independent validation. The test split remains unused. Thirty matched groups per subset now meet the native evaluator's standard subset-size threshold, but do not change the exploratory status.

The exact promoter-like category is `tss_proximal`: original non-exonic consequences within 1,000 bp of the nearest protein-coding or noncoding transcript TSS. It is not an upstream-only promoter annotation. This expanded panel contains 109 original upstream and 191 intronic variants.

All 400 added references match Ensembl GRCh38 (398 unique windows); the original 200 variants retain their prior verification. Center alleles, 255-base windows, single-site differences and reverse complements were checked. [Frozen panel and target matrix](expanded300-plan.json), [protocol](expanded300-protocol.md), [reference checks](expanded300-reference.summary.json).

## Scoring and historical compatibility

Exact prefixes are empty text or `The following sequence is from a healthy individual:\n`. Append the uppercase 255-base sequence without a trailing newline. Use model-required BOS and sum DNA-token log probabilities only, excluding prefix/EOS. Score reference and alternate on both strands:

`pathogenic_score = -mean_strand(logP(alternate) - logP(reference))`

The sign was fixed before execution; no post hoc sign flips, token-count normalization, prompt selection or group replacement. Full-input scoring is used; no cache savings are assumed.

The gate rescored 32 historical contexts and repeated eight anchors. All current repeats were exact. Sampled historical missense log-likelihoods differed by 0.022347–2.687962, so **all 300 missense variants were freshly scored**. The recent TSS-proximal scores matched exactly, allowing its original 100 variants to be reused and only the 200 additions to be scored. This is a sampled compatibility check, not proof of weight identity or exhaustive equality for every historical input.

Earlier 100-variant AUPRCs remain archived: missense 0.212520 / 0.301885 and TSS-proximal 0.140517 / 0.114875 (null / healthy). The old missense serving configuration differs; the change to the new 300-variant result cannot be attributed solely to sample size. [Gate scores](expanded300-compatibility-scores.jsonl).

## Execution and recovery

Selection/scorer freeze: `be97821029ef40c009716809c8484ce37a773350`; verification/gate source `4b4118f36ba40710aae9468b8eddc0716f46aefe`. Metrics frozen in `df5bb05cf0`; recovery amendment `371923e91a`. Exact per-attempt revisions and timings are in [the accounting summary](expanded300-summary.json).

Ten sequential 400-target chunks completed. Chunk 3 stopped after 150 saved targets on one empty HTTP 502 completion response. No automatic retry occurred. A separate model-discovery check returned the expected checkpoint. One bounded recovery verified the same serving fingerprint and anchors, preserved the original 150 targets, and filled only the 250 missing targets. Original, recovery and combined streams remain separate. No further serving failure occurred.

- All **4,800 panel targets** validated: **4,000 fresh + 800 reused**.
- **252 repeat checks**, all exactly equal (maximum delta 0).
- **4,284 successful completion calls**, including panel targets, 32 historical comparisons and repeats; one failed completion call.
- **8,523 service HTTP requests**: 8,522 HTTP 200 and one HTTP 502; two additional successful availability checks. No new deployment.
- 590,364 prompt tokens accounted by the API; max_tokens=0 generated no text used for scoring, while this serving build accounted for 4,284 completion tokens internally.
- Maximum successful completion latency **1.613 seconds**; one-second completion gaps, fixed rank, unchanged fingerprint across successful calls.
- **19:12:55–20:54:35 UTC**, 101m40s including recovery, within the 150-minute / 14,000-HTTP cap. Shared-service impact on other users was not measured.
- Metrics completed in 11.89 seconds, peak RSS 148,640 KiB, exit 0, under the shared heavy-work lock with one thread and lowered CPU/I/O priority.

## Analysis replay

Use the public files in this directory. The original interrupted chunk is excluded from the aggregate in favor of its validated combined stream; it remains preserved for audit.

```python
import json
from pathlib import Path
from expand_likelihood import aggregate
p = Path('.')
chunks = [p / (f'expanded300-chunk-{i:02d}-complete-scores.jsonl' if i == 2
               else f'expanded300-chunk-{i:02d}-scores.jsonl') for i in range(10)]
rows = aggregate(json.loads((p / 'expanded300-plan.json').read_text()),
                 p / 'expanded300-compatibility-scores.jsonl', chunks)
(p / 'replayed-variants.json').write_text(json.dumps(rows, indent=2) + '\n')
```

With the pinned native `marin_dna_evals` package on PYTHONPATH and NumPy/Pandas/SciPy/scikit-learn installed, run `python expanded300_metrics.py replayed-variants.json replayed-metrics.json`. The metrics script verifies native source SHA256 `bee720324d61ea8b88a4644b6bd036569e9bab6f8e931d6335d165fb2e2b2395`.

## Decision

The requested expansion is complete. There is no established advantage to healthy-individual conditioning on either expanded panel. Preserve both methods and uncertainty; no additional inference is running. Endpoint URLs, access details and raw wire responses remain private. Research branch only, no PR or merge into main.
