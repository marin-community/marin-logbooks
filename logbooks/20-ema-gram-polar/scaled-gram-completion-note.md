# Grouped Gram scaling on 32 JAX devices

Keep the independent-bank matrix layout with the default eigensolver as the
training baseline. On 32 JAX v4 devices, the grouped layout had lower measured local mean times
for state refresh plus direction computation on all eight processes, with same-process baseline/grouped ratios of 1.130–1.728. Fixed-state
direction timing was mixed: grouping was faster on three processes and slower
on five. This diagnostic establishes neither training speed nor loss quality, so the
user’s adoption criterion of comparable loss and faster training is not yet met.

The Iris resource name v4-64 corresponds here to 32 JAX devices across eight
host processes. The baseline solves each weight group’s Gram banks separately;
the grouped layout combines banks of equal matrix size and dtype before
distributing whole matrices across devices. Seven weight groups provide seven
directions and 21 saved arrays (momentum, left Gram, right Gram); three Gram
arrays each contain eleven large matrices, giving 33 large solves. The full
case refreshes these states from fixed synthetic gradients and computes
directions. It excludes the model forward/backward pass, parameter application,
data loading and training hooks.

| Case | Matrix mean per process (s) | Grouped mean per process (s) | Same-process matrix/grouped ratio |
| --- | ---: | ---: | ---: |
| Directions from saved state | 3.314–3.399 | 2.710–3.677 | 0.910–1.242 |
| Refresh with fixed synthetic gradients, then directions | 3.729–6.264 | 3.234–4.835 | 1.130–1.728 |

These are ranges across eight processes, each with three warmed samples.
Compilation/first execution is separate, all samples are retained, and host
times are never summed. Each timed call blocks until its local returned outputs are ready.
The baseline always ran first; local output completion
does not align timing entry across hosts. The eight processes are not independent
experimental seeds.

This trial uses twice the devices of the completed 16-device experiment.
Multiplying each process's grouped mean by 32 gives 86.72–117.65 device-seconds
for saved-state directions and 103.49–154.73 for the synthetic full update.
The prior 16-device matrix baseline gives 52.91–53.32 and 57.10–88.46 by the same
calculation. These are local-time cost proxies across separate trials, not
end-to-end cost measurements or paired cross-trial estimates. They provide no
evidence of improved device efficiency.

All eight tasks succeeded on attempt 0 with exit 0, zero failures and zero
preemptions; Iris finished at 17:11:31.968 UTC on November 4, 2026. Source is
`588a02243e29f78e5a7e8522542682cb47acacab`; job is
`/kaiyuew@stanford.edu/ema-gram-20-scaled-gram-v4-64-20261104`. The sole follower
87880 ended naturally with exit 0. All sixteen host-check/worker W&B runs
finished successfully and match the recorded source identities. Regional
report/config generations, checksums and exact readbacks were verified.

Every process and layout is finite, preserves state/gradient inputs and
repeats each output bitwise three times. Replica hashes and statistics agree
before deduplicating 224 direction records and 245 full-update records.
The grouped outputs are compared with independent-bank matrix outputs from
the same saved state and, for refresh, the same synthetic gradients. Relative
Frobenius error is ||grouped − matrix||_F / ||matrix||_F; ranges span the seven
weight-group directions after global shard aggregation.
Direction relative Frobenius errors are 0.557–0.664% for saved-state directions
and 0.586–0.698% for the full update; all seven directions fail the unchanged
elementwise criterion |grouped − matrix| <= 1e-5 + 1e-5 |matrix| in each case. Maximum absolute errors are 0.01817089
and 0.02316278. All 21 state outputs pass tolerance: 15 have identical hashes
and six have tiny differences, at most 5.68434e-14. Directions were not applied
to model parameters, so these differences do not establish loss degradation.
Under the user’s loss/speed criterion, this diagnostic tolerance alone does
not decide training adoption.

The original regional checkpoint supplied 21 Gram/momentum arrays; this is
not a full trainer restore. Full geometry, every-update refresh, source precision
and default solver remain unchanged. The static join passed actual report/row
order reversal and reused the existing 24 mutation fixtures per topology.
A local validation typo (`comparison.records` instead of `comparison.shards`)
was corrected before the order check passed; no numerical worker failed.

The 64-JAX-device topology (Iris v4-128) remains an untested acceleration hypothesis: assigning
33 large matrices would reduce the grouped maximum from two matrices per shard
to one. This is a source assignment count, not a runtime-kernel or speed claim.
Any next trial needs a distinct identity and published launch plan; preserve
the completed job and its guard. No further job or training continuation has been
submitted, and the completed 924-update training baseline is not extended.

[Completion](scaled-gram-completed.json) · [Summary](scaled-gram-summary.json) ·
[Validation](scaled-gram-final-summary-validation.json) · [Design](scaled-gram-design.md).
