# Chat-prefill 100-variant pilot condition

Authorized 2026-11-16 after the four-variant diagnostic. This is a new experimental condition; original v3 raw scores remain unchanged.

Use the same frozen 100 autosomal missense variants and two chromosome-1 demonstrations. Preserve seed 11181 and one original label-independent mapping per variant, shared across zero/two-shot and both strands. Chat template and assistant Answer: prefill exactly match the diagnostic; no thinking-mode override, no tools, demonstrations remain inline in the user message. Canonical single-token targets are leading-space A/B.

400 one-token completions, each returning both requested answer probabilities. Repeat the first variant's four contexts after requests 200 and 400, for eight guard completions. Compare raw A/B logprobs against their first-pass values and stop on drift above 1e-5 or fingerprint/tokenizer mismatch. Sequential, fixed serving rank, one-second gap; stop on completion latency above 10 seconds. Maximum 1609 HTTP requests, 1800 seconds; 45-second request timeout. Every prefix and prefix-plus-answer is checked against the live tokenizer (1200 tokenization checks). One model-discovery request verifies the advertised checkpoint export. No validation-panel inference or new deployment.

Score pathogenic-minus-benign logprob using the saved mapping, then average strands. No post-hoc sign flipping. Report native matched-group AUPRC with 2000 group bootstrap replicates, seed 11181, within-group pairwise ranking, literal A/B preference and descriptive performance by mapping. Compare with historical v3 raw results, explicitly noting the changed serving build and absence of a full contemporaneous raw rerun. The small raw/chat diagnostics on this build are separate evidence. Ten groups remain exploratory; mapping slices may have different prevalences and incomplete groups.

Preserve all prompts, candidate scores, repeat checks and run metadata. Public artifacts omit endpoint/access details. Research branch only; no PR or main merge. Stop and preserve partial evidence on a guard failure rather than silently changing serving settings or rerunning.
