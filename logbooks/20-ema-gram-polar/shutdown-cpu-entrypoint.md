# Prepared CPU shutdown diagnostic entrypoint

The shutdown candidate coordinates tracker completion and JAX distributed
shutdown across processes, addressing a suspected asymmetric teardown path.
The initiating cause of the historical failure remains unknown. A non-training
worker entrypoint is prepared for this isolated shutdown candidate
`2674b5d3975e5c1ecac05fb4310814c90ec736e1`. Local checks cover source selection,
ONLINE tracking policy, task identity, cleanup and failure propagation using fake
external services. The worker has not run against real W&B, JAX or Iris services.
No job is authorized or submitted. The candidate library and frozen training
source remain unchanged.

The [archive](shutdown-candidate/cpu-entrypoint/source-audit.json) contains the
worker, manifest validator, checks, source hashes and
[deployment mapping](shutdown-candidate/cpu-entrypoint/cpu-deployment-plan.json).
This extends the [lifecycle observation preparation](shutdown-lifecycle-observation.md).
The historical shutdown cause and scientific classifications are unchanged.

## Proposed first stage

Two separate fresh CPU jobs are proposed: clean exit and an injected application
`SystemExit(23)` on rank zero. Each would contain four Iris tasks, one process per
task, four CPUs and 8 GiB each of memory and disk per task. Each process would
publish four scalar diagnostic records to a fresh ONLINE W&B run. There is no
training, array construction or numerical kernel in the diagnostic itself.
Real package initialization remains untested. Four tasks do not establish four
physical hosts; worker placement must be verified independently.

The diagnostic initializes ONLINE tracking before importing Levanter or JAX,
rejects an existing native run or distributed world, and verifies that Iris
job/task metadata matches the process environment and that JAX process rank
equals the Iris task index. Iris is the job orchestration system; Levanter is
the training library providing the tracker and finalizer. It drains its tracker before installing the
candidate exit hook. That hook exchanges rank receipts, waits for all ranks to
finish reading them, then requests distributed shutdown. The diagnostic records
startup and finalization in separate local JSONL files. Flush does not establish
fsync or remote durability, and synchronous writes can block.

A local tracker receipt records whether queued logging calls and tracker finish
completed, with no failed or rejected updates and no outstanding worker. The
shutdown protocol exchanges rank receipts derived from that result and any
application error. Training requires W&B on rank zero; this diagnostic requires
it on every rank.
Review caught that policy difference. An incomplete local W&B receipt on any
rank now becomes an application failure passed to the candidate finalizer.
An original application error, including exit code 23, retains precedence over
cleanup failures. Local completion still does not certify remote history.

The proposed communication-health budget is 300,000 ms per finalization
rendezvous: the existing tracker finish wait is 120 seconds, with an additional
180 seconds of unmeasured scheduling and diagnostic-write margin. This is an
engineering proposal for these two scenarios, not a measured bound or proven
repair. It imposes no queue, task, training or wrapper wall-clock limit. The
submission mapping omits retry and preemption overrides to preserve Iris
defaults. Existing initialization, heartbeat and shutdown health settings are
unchanged. Delayed-rank and missing-peer scenarios need separate review.

## Local evidence and usage boundary

There are 27 distinct checks with scoped passing evidence: 12 entrypoint checks,
11 manifest checks and four worker-bootstrap checks. These were developed in
stages; unchanged checks were reused. Two portability checks were also run.
After the all-rank receipt correction, two new controls and two affected checks
passed. This is not a freshly rerun full suite or a runtime qualification.
The source-linked check loads the actual installer/core with fake boundaries;
the other checks use fake imports, trackers, JAX and coordinator behavior.
Draft source lint passes. The evidence files identify each invocation's scope.

The worker CLI defaults to read-only preflight. Its separate `--execute` option
can initialize real services and requires new explicit authorization; an option
or a valid manifest is not authorization. No submission client is supplied.
The template leaves tracking entity, region, controller revision and runtime
lock unresolved, and the worker rejects those unresolved bindings before
execution. Filled declarations are not independent deployment verification.

Preflight checks eight source-file hashes against both working files and the
declared Git commit. The proposed runtime image must therefore include a
self-contained candidate Git checkout and Git itself, not the local worktree's
`.git` pointer or shared `.venv`. The bootstrap selects candidate Levanter/Iris
source roots without importing those packages and rejects preloaded runtime
modules. Later module-origin checks do not attest all transitive dependencies
or custom import hooks. The CPU runtime image and dependency lock have not been
built or qualified.

## Remaining qualification

Before requesting execution, bind the CPU image and dependency lock, tracking
entity, region and CPU placement, exact deployed controller revision, and
packaged worker command. Preserve fresh job, task-attempt, native run and
coordinator identities. The initializer names its coordinator endpoint using
task zero's attempt number; that source observation is not evidence of actual
registry freshness or cleanup.

Future results require independent native-history, task-attempt, source and
controller evidence. In the injected-failure case, a nonzero exit is expected;
a clean-success verdict would be a failure of the diagnostic contract. The
controller must be observed deriving the overall job failure from natural
task exits, including code 23, without a forced job-completion call; its handling
of peer exits must also be recorded. This behavior, missing-peer cleanup and
partial initialization remain unqualified. The CPU proposal changes no existing
accelerator allocation and
does not resolve retention ownership or qualify TPU/GPU shutdown behavior.
