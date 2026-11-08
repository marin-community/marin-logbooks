# Local observation of the shutdown candidate

The isolated shutdown candidate now records exit-hook and coordinator-protocol
boundaries through an optional callback that appends local diagnostic events. A failed observer write leaves
normal cleanup in progress and forces the local process to exit nonzero. An
original nonzero application `SystemExit` code determines the exit code even
if protocol cleanup also fails. The core re-raises protocol errors without
replacing them with observer errors. This is
local source validation; no runtime qualification job has run.

Candidate `2674b5d3975e5c1ecac05fb4310814c90ec736e1` follows the
[read-completion correction](shutdown-read-ack-correction.md). Apply the
[integration patch](shutdown-candidate/integration/candidate.patch), then the
[read-completion patch](shutdown-candidate/read-ack/correction.patch), then the
[observation patch](shutdown-candidate/observation/observation.patch).
The [source audit](shutdown-candidate/observation/source-audit.json) binds the
source hashes and evidence scopes. The accepted training checkout remains
frozen; no launch recipe selects this candidate.

The observer records local events around receipt publication, receipt reading
and JAX shutdown, plus exit-hook entry and return. The first rendezvous waits
for all ranks to publish their receipts; the second waits for all ranks to
read them before shutdown.
A receipt is the local report of required telemetry and application outcomes.
The observer receives the real installer and protocol boundaries without
replacing JAX's global coordinator client. It adds no service calls or barriers.
The first observer exception disables further observation; only its type is
retained, so the observer's exception message and traceback are not re-emitted.
Observation failure is checked after normal protocol completion. It does not
add a failure field to the exchanged receipts: only the affected process is
forced nonzero, and real controller aggregation still needs qualification. That aggregation
converts the scheduler tasks' terminal outcomes into the overall job status;
local process exit checks cannot establish it.

The installer now registers its replacement before removing the default exit
handler. A registration failure propagates while retaining the original handler,
and a later retry can install one functioning replacement. Local checks cover
this case. Arbitrary partial failure inside handler removal remains unqualified.

Seventeen source/AST exit-hook checks pass, including four actual Python
`atexit` subprocesses covering clean exit, observer failure, combined application
and observer failure, and the original application exit path. Three checks use
persisted local logs from the actual candidate installer/core with four
independent fake rank fixtures. Nine log-contract checks reject incorrect
ordering, identity, missing ranks and truncated records. These fixtures do not
constitute a distributed runtime. External JAX, coordinator and tracking services
are simulated; no numerical kernels or external services ran. Required source
lint and type checking pass.

The previous four-thread read-completion simulation is reused for unchanged
rendezvous behavior; the earlier 27 tracker/backend/Trainer/wrapper checks were
not rerun. These are separately scoped results, not a freshly rerun full suite.
The archived [hook result](shutdown-candidate/observation/hook-checks.json),
[lifecycle-log result](shutdown-candidate/observation/lifecycle-trace-checks.json)
and [log-contract result](shutdown-candidate/observation/trace-contract-checks.json)
retain those distinctions.

The local log adapter and checks are archived beside the source audit. Set
`EMA_GRAM_CANDIDATE_ROOT` to the isolated checkout and
`EMA_GRAM_CHECK_RESULT_DIR` to an existing separate output directory before running either
`check_trace_contract.py` or `check_observed_lifecycle.py` with Python. The logs' execution, task, attempt, worker and W&B identifiers are synthetic.
The checks never certify native online W&B history or the scheduler's job status. Stream flushing is not an fsync or a
remote durability guarantee. A synchronous local write can block; this work
introduces no wrapper watchdog or training wall-clock limit.

The [qualification design](shutdown-candidate/observation/qualification-design.json)
defines eight future diagnostic scenarios and independent source, process,
tracker-history and controller evidence joins. It remains an unexecuted design.
The training wrapper does not expose the optional observer, and a concrete
non-training runtime entrypoint still needs preparation. Real package
initialization, online multi-host tracking, missing-peer cleanup, deployed
controller revision and natural task aggregation remain unqualified. Accelerator
allocation retention is unresolved; an explicitly CPU-only first stage would
have narrower coverage and still requires new-job authorization. No new job is
authorized by this publication. Historical experiment classifications and the
published scientific findings are unchanged.
