# TSS-proximal 100-variant DNA-likelihood evaluation

User authorized 100 promoter-like variants, then requested verification of the actual subset. Use the pinned dataset's **tss_proximal** category, not a literal promoter annotation. No A/B classification or chat template.

## Subset definition and selection

Pinned variant dataset revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f`; materialized sequence revision `7b92f047f9a36f90e9ac47886afa2a99264ee35c`. Input SHA256 values are enforced by prepare.py. The data card names consequence-group subsets; inspected curation source at `d2326f46f9bc10561d6a753239b3f6cef8308531` classifies original intergenic/intronic/upstream/downstream consequences as tss_proximal when the minimum distance to a protein-coding or noncoding transcript TSS is <=1000 bp. Config uses threshold1000. This is not limited to upstream DNA or protein-coding promoters, and is not an experimental promoter activity label. Sources:

- https://huggingface.co/datasets/bolinas-dna/evals_mendelian_traits/blob/4aed58e50c5dea0b878a665007af2ef9e5108e9f/README.md
- https://github.com/Open-Athena/marin-dna/blob/d2326f46f9bc10561d6a753239b3f6cef8308531/src/bolinas/pipelines/evals/trait_intervals.py
- https://github.com/Open-Athena/marin-dna/blob/d2326f46f9bc10561d6a753239b3f6cef8308531/snakemake/evals/config/config.yaml

The pinned train split contains 163 complete autosomal tss_proximal groups (1,630 variants). Filter to chromosomes1–22 before grouping; exclude chrX/Y. Rank group IDs by SHA256 of `11181:tss-proximal-group:{group_id}` and take ten complete groups, sorting variants by identity inside each. Exactly ten positives and ninety matched common-variant controls. This is an exploratory subset of the development/train split, not final validation. Existing reserved missense validation stays untouched.

Reuse centered 255-base windows (127 each side), verify central REF/ALT and reverse complements and independently verify every forward reference against Ensembl GRCh38. Retain nearest-TSS distances and original consequences. No gene coordinates or disease annotations are added to prompts.

## Exact scoring inputs

Null: `{sequence}`

Healthy:
```text
The following sequence is from a healthy individual:
{sequence}
```

Uppercase DNA, no trailing newline. Tokenizer supplies one BOS; no chat or assistant prefill. Only DNA token logprobs are summed; exclude conditioning prefix and EOS. Use the existing client.py full-input echo/prompt-logprob scorer with max_tokens=0, fixed rank and no forced vocabulary. Score reference and alternate on both strands for both prefixes: 100 x 2 x 2 x 2 = 800 target likelihoods. Each method's variant score is negative mean over strands of logP(alternate)-logP(reference), exactly matching earlier missense DNA semantics. No normalization by token count and no post-hoc sign changes. Cache savings are not assumed; full-input path preserves the established baseline.

## Execution limits

800 scored targets plus eight sentinel targets repeated after each block of80 targets =80 repeat requests (880 completion calls total). First block is ten variants; continuation is automatic only if the repeated-score and latency gates pass. Sentinels are all eight contexts for the first selected variant, baseline collected in the main stream. One request in flight, fixed rank, one-second completion gaps. Existing PacedClient latency guard: stop >5 seconds or sustained recent median >max(1.5 seconds,3xinitial); repeats must match within1e-6. Tokenization/echo/decoded-target boundaries and model/fingerprint checked throughout.

At most3000 inference-service HTTP requests including tokenizations/model discovery, 45 minutes. No automatic inference retries, deployment, restart or capacity increase. Expected approximately20–30 minutes. Endpoint preflight is a separate single model-discovery request. No generated text is used; max_tokens=0 may account for one internal token per completion as in the existing scorer. Authentication, route and wire metadata remain private.

## Analysis

All800 targets and80 repeat checks must be complete, finite and metadata-aligned before reporting. Native pinned MarinDNA metrics hash `bee720324d61ea8b88a4644b6bd036569e9bab6f8e931d6335d165fb2e2b2395`: AUPRC + group-bootstrap SE,2000 replicates,seed11181; paired healthy-minus-null bootstrap interval; within-group positive/control ranking. Prevalence0.10. Ten groups are below the standard30-group reporting threshold and remain exploratory. Compare missense descriptively only: different variants and potentially serving builds prevent a paired claim across subsets.

Research branch only; log source revision, inputs, reference verification, runtime, incomplete attempts and results before issue milestones. No PR/main merge or broader team outreach.
