# Shutdown repair: isolated source integration

The candidate now connects local tracker finalization reports through Levanter, Marin
and `experiments.ema_gram.train.run_training_with_coordinated_finalization`. Thirty-
six local checks pass: seven tracker/composition checks, five backend checks, nine
exit-hook checks, six Trainer cleanup checks and nine outer-wrapper checks. These
checks use actual source modules or extracted method bodies with fake external
boundaries. One exit-hook check uses a real local Python subprocess. They do not
initialize numerical kernels, contact tracking services or run cluster jobs.

The [candidate patch](shutdown-candidate/integration/candidate.patch) is an
unqualified development checkpoint, commit `f6ef3108962a06bd3b39db50970a4d4a434ad372`
on `codex/20-ema-gram-shutdown-repair`, based on archive commit
`73e35b11131e0f31c9409d7454854e4a003bca8f`. The accepted execution checkout remains
at `923d79510475df3ed0fef53d40b3290ae4c90fce`. No accepted execution source changed,
and no launch recipe selects the new entrypoint. Library changes in the patch affect
their callers if someone chooses to install the patch; keeping the entrypoint opt-in
does not make the whole patch behavior-neutral. The [seed0 findings](single-seed-
findings.md) and failed-run classifications remain unchanged.

Background tracker reports retain failed calls, rejected updates, finish outcome and
worker liveness. Exactly one locally complete W&B report is required on rank zero.
For a background-wrapped sink this requires successful finish, an exited worker, zero
failed calls and zero rejected updates. An unwrapped W&B report only records
successful local finish; it does not count lifetime losses. In both cases remote
history still requires an independent audit. All ranks report application outcomes
before entering JAX shutdown. Auxiliary Finelog results remain visible without
becoming the scientific completion gate. A local successful report is not
certification of remote metric history. Finelog's final flush cannot recover drop
counts consumed by earlier flushes.

The outer wrapper replaces `_finalize_iris_jax_after_clean_exit` with one coordinated
Python atexit hook after its final device memory query and artifact writes. The hook
exchanges rank outcomes and leaves job status to natural task exits; it never invokes
the force-completion API. It retains application failures and nonzero SystemExit
codes. If a peer or coordinator is unavailable, the proposed hook exits nonzero
without calling its JAX shutdown callback. Actual cleanup and controller aggregation
on that path remain untested.

Source review found an admission race: a logging producer could pass a stopped check
and then enqueue behind the worker's shutdown sentinel. The candidate serializes
admission with the stop transition. A deterministic local check covers that ordering.
The caller must still stop producers before taking its final immutable tracker
report: an earlier report cannot detect later rejected updates. Trainer normally
waits for checkpoint writers before tracker cleanup; custom producers need their own
lifecycle proof. This race has not been shown to cause the failed alpha0.25 or
alpha0.5 shutdowns described in the [earlier design](shutdown-repair-design.md).

Trainer cleanup now preserves the original application exception when checkpoint,
context or final-event cleanup also fails. Cleanup failures are attached as exception
notes and logged. Without an application error, cleanup still fails the process. The
outer wrapper skips device-memory queries after an application or initialization
error, because querying devices can initialize a JAX backend before the required
ONLINE tracking setup has completed. Failed cleanup can prevent evidence files from
being written; missing evidence must not be interpreted as success.

The portable checks run in separate Python processes to isolate their fake modules.
From the candidate checkout, use an existing compatible Python environment with
draccus installed:

```sh
python experiments/ema_gram/shutdown_checks/run_checks.py --output-dir /tmp/ema-gram-checks-new
```

The output directory must not exist. The runner saves each process's output, exit
status and check-source hash. Full package imports, numerical suites, live W&B
delivery and multi-host JAX behavior are outside this validation. The required
`infra/pre-commit.py --fix` checks, including Pyrefly for source files, passed.
[Source lint](shutdown-candidate/integration/candidate-final-lint.log), [check-file
lint](shutdown-candidate/integration/portable-checks-lint.log) and [source
hashes](shutdown-candidate/integration/source-integration-audit.json) are archived
with the [check results](shutdown-candidate/integration/summary.json).

Before deployment, qualification must establish real package initialization, ONLINE
telemetry delivery and complete history, normal and delayed-rank shutdown, required-
sink failure, missing peers, partial initialization, new coordinator state and
rendezvous identity for each process attempt, and controller aggregation of natural
nonzero exits. The exact deployed controller revision must be recorded. A justified
communication-health budget must be explicit; no training or scheduler wall-clock
limit is introduced. No new training or qualification job is authorized. This patch
supports testing coordinated teardown; it does not establish the initiating cause of
the historical failures or demonstrate that they are repaired.
