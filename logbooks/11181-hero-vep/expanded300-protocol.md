# Expanded train panels: 300 missense and 300 TSS-proximal variants

User-approved scope November17: expand each existing100-variant pilot to300 by adding20 complete groups. Retain the original10groups, do not filter overlapping windows, autosomes only, train split only. No chat/classification, only null and healthy-individual raw DNA likelihood. These are expanded exploratory panels, not independent validation.

## Selection

Use the same pinned variant/sequence parquet files and checksum checks as prepare.py. For each subset, preserve the100old rows unchanged; rank remaining autosomal groups by SHA256 `11181:{namespace}:{group_id}`, select20, and sort rows within groups by variant identity. Namespace `query-group` for missense and `tss-proximal-group` for TSS-proximal. Each final panel has30groups,30pathogenic positives and270controls. No overlap/window filters or chromosome1 exclusion. Test split untouched. 130added missense rows are from the earlier reserved development panel; that original reservation is therefore no longer wholly independent or untouched. This is intentional under the user's simplified expansion, and no claim of fresh validation will be made.

TSS-proximal retains its existing definition (non-exonic consequence, nearest protein-coding/noncoding TSS <=1000bp), not an upstream-only promoter annotation. All added references must match Ensembl GRCh38 before live scoring. Exact rows/sequences/targets and source-file hashes are in expanded300-plan.json. The400added rows are separately represented in the reference-verification input.

## Score and reuse gate

Unchanged255-base windows, both strands, both reference/alternate alleles. Prefixes: empty or `The following sequence is from a healthy individual:\n`. Sum DNA-token likelihoods, exclude prefix/EOS; pathogenic score is negative mean strand(logPalt−logPref). No chat, normalization, sign fitting or new prompt.

Historical score provenance is checked for all1600old targets; missense input hashes must match, and TSS rows/sequences come from the exact saved promoter plan. Compatibility gate: rescore eight contexts for each of the first/last old variants in both subsets (32requests), then repeat eight references covering both prefixes/strands/subsets. Repeats must match within1e-6. Reuse a subset's old scores only if every historical comparison for that subset matches within1e-6 and the same advertised checkpoint is present. This is a sampled numerical check, not an independent weight checksum or a proof for all inputs. If compatibility fails, uniformly rescore that subset's full300variants rather than mix inconsistent scores. Log the decision and deltas before the panel run.

## Bounded chunk execution

One request at a time on fixed rank, one-second completion gap; existing token/echo and fingerprint checks, max5-second completion latency and sustained-latency guard. Four anchors covering both prefixes and both subsets repeat at each chunk start and every80new targets. Each chunk contains at most400targets (50variants), saves scores immediately, and has its own completion marker. All chunk outputs must be complete before aggregation; interrupted partial chunks are preserved and resumed only after anchor verification, without silently replacing completed scores.

If both historical subsets pass reuse,3200new targets (eight chunks) plus40compatibility checks and192chunk repeats =3432completion calls. If neither passes,4800targets (twelve chunks) plus40gate+288chunk repeats =5128calls. At most14000service HTTP requests and150minutes aggregate wall time from the first gate request. Each process <=15minutes, within overall budget. No automatic request retries, parallel serving requests, new deployment, capacity changes or paid baseline. Raw wire/auth/route metadata stay private. One separate model-discovery preflight precedes the gate.

## Analysis and reporting

Validate all4800target likelihoods and score reuse provenance; report AUPRC, group-bootstrap SE, paired healthy-minus-null95%interval and within-group pairwise accuracy for each300-row subset separately. Same native MarinDNA metrics hash `bee720324d61ea8b88a4644b6bd036569e9bab6f8e931d6335d165fb2e2b2395`,2000resamples,seed11181. Existing100-row results remain archived; do not call the expansion independent confirmation or select groups based on outcomes. Keep issue logs and commit-pinned evidence. Research branch only, no PR or main merge.
