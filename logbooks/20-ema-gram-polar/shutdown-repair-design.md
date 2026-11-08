# Candidate shutdown repair after the seed0 runs

The source and cached logs support a candidate that records telemetry drain
outcomes and waits for all ranks before JAX shutdown. Nineteen local prototype
checks pass. The candidate is isolated under [shutdown-candidate](shutdown-candidate/);
it is not installed in a training entry point. The incident initiating cause
and the candidate's behavior on real multi-host JAX remain unverified.
The [seed0 findings](single-seed-findings.md) and historical completion gates
are unchanged. Those gates require scheduler success with validated task
lineage, a finished native ONLINE W&B run, and a separate post-run audit of all
37,335 finite five-metric rows and 38 C4 evaluations. The local finalizer does
not perform or replace that post-run audit.

## Evidence and limits

The failed alpha0.25 run's cached process-zero log records W&B finalization,
a 120-second background-thread abandonment, and a shutdown-barrier abort.
The main thread's fatal stack is in Finelog close; another thread reports
`HandleAbandonedError`. Alpha0.5 also records a three-of-four shutdown barrier
and a main thread in Finelog close. These stacks locate operations at the dump
instant. They do not establish which operation caused the initial delay or
that Finelog occupied the full interval. See the
[alpha0.25 report](parallel-long-runs-r01-alpha025-failed.json) and
[alpha0.5 report](parallel-long-runs-r01-alpha05-failed.json).

The accepted source, revision `923d79510475df3ed0fef53d40b3290ae4c90fce`, calls
member trackers' finish methods sequentially. The primary process can spend
120 seconds in background W&B finish before reaching Finelog close. The
Iris JAX shutdown-health timeout is also 120 seconds; its comment assumes
60 seconds of primary-process tracker finalization. This supports asymmetric
teardown as a mechanism to test: siblings can begin their shutdown wait while
the primary process is still draining trackers. The proposed exchange delays
that entry until every rank reports its local drain outcome. The precise
incident timing and native timeout trigger remain unverified. Increasing a timeout alone would not establish
that the underlying telemetry problem was fixed. Communication-health timeouts
are separate from queue, training and scheduler-preemption limits.

`BackgroundTracker.finish()` can return normally after a sink exception, while
a worker remains alive, or without enqueuing the wrapped finish because the
queue is full. The existing behavior is explicitly best-effort. A normal
return cannot certify successful telemetry delivery. The receipt prototype
makes these outcomes visible without changing the existing exception policy.
A receipt is an immutable local status snapshot; it is not a remote history
certificate. Producers must have stopped before it is collected. In this prototype, local
drain success requires the wrapped finish to return successfully, the worker
to exit, and zero recorded failed calls or rejected updates. The lifecycle
prototype receives `required_complete` from a fake sink; wiring that field to
real W&B evidence remains production-integration work.

Finelog already exposes `FlushResult` through `LevanterMetricsWriter.flush`.
`SUCCEEDED`, `DROPPED` and `TIMEOUT` distinguish delivery outcomes for that
flush. The drop count is consumed by a flush, so a final successful flush
cannot establish losslessness over the entire run. Optional Finelog diagnostics
must remain distinct from required ONLINE W&B tracking and the existing history
audit. Integration and aggregation of these receipts remain unfinished.

The EMA-Gram outer wrapper reads device memory with JAX in its `finally` block,
after `train_lm` returns. Therefore shutting JAX down inside `train_lm` before
that wrapper completes is not a supported placement. The proposed rank exchange
uses host coordinator key/value and barrier operations already used by Iris's
compile guard. It has no array collective or numerical kernel. Real coordinator
lifetime and heartbeat behavior during this exchange have not been tested.

The accepted checkout's `complete_job` implementation explicitly marks
unfinished tasks successful and stops them. It is a force-completion operation.
The same checkout derives natural job success when all task states are
successful. The candidate makes no `complete_job` call: each rank exits with
its own result for controller reconciliation. The deployed controller revision
was not established in this investigation. This source observation does not
reclassify historical runs or establish a cause of either failure.
[Source hashes and scope](shutdown-candidate/source-audit.json) bind these
claims to the inspected checkout.

## Candidate behavior

1. Finish all training, evaluation, checkpoint work and outer-wrapper JAX uses.
2. Attempt local tracker drain and retain required-sink completion evidence,
   rejected-update counts, errors and outstanding-worker status. Preserve
   auxiliary diagnostics separately.
3. Exchange all ranks' receipts before any rank enters JAX shutdown. A failed
   receipt remains failed; a missing peer or coordinator error cannot release
   successful completion.
4. After every rank has supplied a receipt, enter shutdown on every live rank.
   Propagate required-telemetry failure after shutdown. Preserve a nonzero exit
   on shutdown failure. The prototype's exit hook uses an explicit nonzero
   process exit because an uncaught Python `atexit` exception alone is
   insufficient.
5. Let natural task outcomes determine the job state. Preserve canonical
   success, finished-ONLINE and full-history checks as separate scientific
   requirements.

This is a proposed lifecycle, not a deployed replacement hook. Receipt
production across the actual tracker composition, treatment of outstanding
workers, preservation of application exceptions, and replacement of the
existing hook still require integration review. Registering both hooks would
be incorrect. Failure after a coordinator outage also requires review against
JAX's own cleanup behavior. On a missing-peer or coordinator-error path, the
isolated exit hook exits nonzero without calling the proposed shutdown
callback. Surviving ranks' real JAX cleanup on that path is unresolved and must
be qualified before adopting the hook.

## Local checks and remaining qualification

| Component | Passing checks | Scope |
| --- | ---: | --- |
| Background receipt | 8 | Successful drain; idempotence; sink and preparation failures; outstanding worker; full queue; missing artifact; post-finish update |
| Rank exchange | 5 | Slow peer; failed receipt; missing peer; coordinator error; stale extra rank |
| Finalizer | 6 | Required versus auxiliary policy; drain exception; required failure; shutdown exception; real local Python success/failure exit codes |

The checks use fake sinks and an in-memory coordinator. They do not import JAX
or W&B, initialize numerical processes, contact a service, or launch a cluster
job. Reproduce from the repository root:

```bash
.venv/bin/python logbooks/20-ema-gram-polar/shutdown-candidate/test_receipt_prototype.py
.venv/bin/python logbooks/20-ema-gram-polar/shutdown-candidate/test_rank_finalization_prototype.py
.venv/bin/python logbooks/20-ema-gram-polar/shutdown-candidate/test_finalize_rank_prototype.py
```

Next, integrate required W&B receipt production and auxiliary Finelog outcomes
into an isolated production candidate, then test application-error and hook
replacement paths locally. A future authorized multi-host ONLINE qualification
must cover delayed and failed telemetry, missing ranks, post-training JAX uses,
shutdown health, natural task aggregation, and complete native history on the
successful path. Failed paths must remain visibly failed. No new training or
qualification job is authorized or launched by this report; frozen execution
source, launch guards and terminal caches remain unchanged.
