# Grouped-Gram diagnostic outcome

Keep matrix/default as the training baseline. Grouping Gram banks did not
establish a consistent optimizer speedup in this diagnostic, so matrix_grouped
is not promoted. Here matrix/default means the matrix layout with the default
eigensolver. The candidate combines equal-size Gram matrices across weight
banks into one eigensolver batch per size, then splits the eigenpairs back.

Four JAX processes compare seven weight-group direction tensors in two cases:
directions from fixed saved optimizer state, and refreshed Gram/momentum state
plus directions from a fixed synthetic gradient. Each timed replay starts from
the same inputs; returned states are not fed into the next replay, and no model
parameters are updated.

All four tasks succeeded on attempt 0 without failures or preemptions. All four
host logging checks and four numerical W&B runs finished successfully. Final
regional reports passed exact generation-pinned readback, source/checkpoint
identity, finite-output and input-immutability checks. Three repeated outputs
matched bitwise within every case/layout/process. The summary checks that repeated copies of each output shard have identical
hashes and error statistics before counting each shard once: 112 direction
shards, or 133 records when the 21 state outputs are included. The adapted aggregator passed 24 corruption fixtures,
known-answer norm/timing fixtures and actual report-order invariance.

For directions from unchanged saved state, per-process means were
3.307–3.332 seconds for matrix and 3.843–3.872 for matrix_grouped; baseline over
candidate ratios were 0.857–0.861. For a full synthetic-gradient update, means
were 3.569–5.529 and 3.597–5.337 seconds respectively, with same-process ratios
0.810–1.129. All three samples per case/layout remain included, including the
slower first grouped call. Host durations are never summed. Fixed baseline-first
order, only three samples and local completion synchronization limit timing
interpretation. There is no training throughput or MFU result.

All seven direction comparisons fail rtol=atol=1e-5 in both cases. Relative
Frobenius errors are 0.554–0.662% for fixed state and 0.587–0.701% for the full
update; maximum absolute errors are 0.025151 and 0.016699. All 21 state outputs
pass tolerance: 15 have matching hashes, while six Gram outputs have tiny
nonzero errors, at most 2.8422e-14. Directions are not applied to model
parameters in this diagnostic. These differences do not establish loss
degradation, and tolerance failure alone is not the decision against adoption.

The user reiterated that any suitable machine is authorized and then clarified
that TPU should remain the default. The v4-32 restriction was our submission
choice. Future experiments will first inspect suitable ready TPU capacity and
choose a compatible topology; GPU remains an option when justified, not the
new default. A November 4 08:01 UTC snapshot advertised one available reserved
v4-32 slice, 164 free H100s at cw-rno2a, 248 at cw-us-east-02a, and GB200
capacity. These snapshots are not allocations or start guarantees. No new job
has been submitted. GPU work, if selected later, requires allocation retention
through success and failure. Large saved checkpoints remain regional.

An initial local completion observer looked for a nonexistent W&B summary key
`completed`; the harness actually logs `grouped_gram/success`. The observer was
corrected, and all four runs passed the actual success-key check. This was an
observer error, not an execution failure.

[Completion](grouped-gram-completed.json) · [Summary](grouped-gram-summary.json) ·
[Validation](grouped-gram-final-summary-validation.json).
