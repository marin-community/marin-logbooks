# Candidate correction: wait for all finalization-record reads

The local four-rank teardown simulation exposed a race in candidate
`f6ef3108962a06bd3b39db50970a4d4a434ad372`. Its first rendezvous waited for all
ranks to write their finalization records, then let each rank read the records
and call shutdown. A fast rank zero could close the coordinator service while
a slower rank was still fetching those records.

A four-thread fake coordinator reproduced this ordering with the unmodified Python core from that candidate. The delayed rank received a connection error after rank
zero called its simulated shutdown. The modified core adds a second rendezvous
after every rank reads and validates the records. In the same simulation, all
four ranks read the records before any shutdown callback, and all four return
without error. This ordering also appears in the existing Iris
`lib/iris/src/iris/jax/compile_guard.py:check_distributed_compile_fingerprint`
check, which waits for all reads before deleting coordinator records.

The [new regression](shutdown-candidate/read-ack/regression.json) and
[nine affected exit-hook checks](shutdown-candidate/read-ack/exit-hook-checks.json)
pass. The other
27 [previously passing local checks](shutdown-candidate/integration/summary.json) were unchanged and were not rerun. Lint
and type checks passed for the correction. This is a local simulation with
fake coordinator lifetime and shutdown callbacks; it does not establish the
actual JAX service lifetime or prove the alpha0.25 or alpha0.5 historical training failures'
cause. The [earlier patch](shutdown-candidate/integration/candidate.patch) remains
archived. Apply the [correction patch](shutdown-candidate/read-ack/correction.patch)
after it before considering future qualification. The corrected candidate is
`689fe209f8102a588ebac1bc59cf04289137a3ad`; [source bindings and the controlled
comparison](shutdown-candidate/read-ack/source-audit.json) preserve provenance.

The candidate remains undeployed. No numerical kernels, live services,
experiment reads or cluster jobs ran. Missing-peer cleanup, live multi-host qualification with W&B in online mode and deployed controller behavior remain unverified;
new-job authorization is still required before such execution.
