# Review and VEP instruction-order diagnostic

November 16, 2026. User authorized autonomous bounded investigation with logging. Research branch only; no new deployment, training, PR/main merge or reserved validation.

Six authored reviews: two clearly positive, two clearly negative, two ambiguous (unopened item and mixed feelings). Cross two semantic mappings, two displayed option orders, two response-instruction mention orders (A or B / B or A), and short versus neutral structured metadata before the review: 96 contexts. Metadata is explicitly marked unrelated to sentiment; it contains 12 identifier rows, identical across items. Context length and structured presentation change together, so this is a combined distraction/presentation intervention, not a pure length experiment. Ambiguous items are ungraded. Clear sentiment keys are authored, not external labels.

VEP: same four prior variants, original DNA, forward strand, A/B only. Cross both semantic mappings, both displayed list orders and both instruction mention orders: 32 contexts. No VEP accuracy claim.

All contexts use pinned Marin chat template and assistant Answer: prefill. Canonical single-token leading-space A/B candidates. Exact prompts and token IDs are frozen in cross-domain-inputs.json. Execution order randomized seed 11181; four planned repeats (two review, two VEP) after 128 scores. Runner reuses vep_factor_diagnostic.run with a corrected dynamic completion marker; source and dependency are committed together in this branch.

Budget: 132 one-token completions, 384 tokenization checks and one model discovery = 517 HTTP maximum, 600 seconds, sequential fixed rank, one-second completion gap. Stop on request errors, tokenizer mismatch, changed serving fingerprint, completion >10 seconds, or repeat logprob delta >1e-5. No automatic retries. Same advertised checkpoint required. Raw wire/access metadata stay private.

Analyze conditional semantic probability, literal/last-mentioned choices, exact clear-review accuracy and paired semantic flips independently for mapping, list order, mention order and added context. Paired conditions share items and are not independent samples. Swapped content mappings test semantic invariance; no label sign fitting. No free reasoning or tools. Use intuitive counterexamples only when they actually occur; do not present ambiguous sentiment as a model error. If clear reviews resist the effect, report the negative control faithfully and narrow the remaining hypothesis.
