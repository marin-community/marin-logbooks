# Tiny callback compatibility gate

This new research preparation descends from published source A
`32f42f9a3601258123d316cc03c25b05add7a9c2`, whose instrumented legacy TPU
lowering failed with `Partitioned callback not implemented on tpu backend`.
The untouched 1.531B anchor reproduced the historical 255-gradient and loss
metrics. The failed instrumented attempt produced no valid X/G comparison.
Source A and scientific source `2b17db78aedc0a603664238fbfa0bc0c68542d2b`
remain unchanged in their original checkouts.

The candidate changes only the JAX transport argument of the inner callback
in `capture.emit` from `partitioned=True` to `partitioned=False`. Formatting
wraps that call. `Collector.receive(..., True, ...)`, the outside global
callback, complete manual shard map, mesh/specs, device-id input, dynamic
index-zero conditional, collector checks, and identity custom VJP are unchanged.
The installed JAX 0.11.1 source supports this candidate because full-manual
shard maps already provide local operands and per-device callback lowering.
CPU execution does not establish TPU compatibility.

## Fixture and checks

`callback_compatibility.py` imports the actual candidate `capture` module and
uses its existing `Collector`, `emit`, and `make_tap`. It creates no model or
Trainer. Each Auto/Auto and Explicit/Explicit 2x2 data/model mesh runs:

1. One compiled transport fixture with two actual scan indices `[0, 1]` and
   two identical emissions at each index. Nonzero-index payloads differ from
   the oracle, so incorrect filtering cannot silently pass.
2. One compiled untapped differentiated projection reference.
3. One compiled instrumented differentiated projection, executed twice with
   identical arguments. The two-step dynamic scan uses the same simple
   `equinox.filter_checkpoint(..., prevent_cse=False)` mode selected by the
   experiment's enabled checkpointing. It differentiates all three floating
   inputs: canonical X, projection W, and loss cotangent G.

There are six explicit lower/compile operations and eight compiled executions
in total. Every execution waits for results and `jax.effects_barrier()` before
records are interpreted. Every transport/composition execution has a fresh
actual collector, whose finish runs immediately after that execution's barriers.
Transport requires at least two X and G observations for each identity. Each
composition execution requires at least two X and one G observations per
identity, preserving the aggregate minima while preventing duplicates in one
execution from masking an omission in another. All ten capture byte identities
are explicitly compared across composition executions. Any exception ends the
gate; there is no automatic recovery or full-model escalation.

The required stdlib supervisor starts the worker in an owned process group and
waits with a 300-second external wall deadline. Its clock starts before worker
creation, covering worker imports, native compilation, execution, and effects
barriers. On timeout it sends SIGKILL to that owned process group and reaps the
worker, then exits 124 without recovery. The worker's own SIGALRM remains only
a diagnostic aid and does not guarantee a wall deadline. Root must also pin the
reviewed Iris task timeout; bypassing the supervisor does not satisfy this
prepared command.

X is finite actual BF16 `[4,4,3,2]`, the row-major integers 0 through 95 divided
by 64. G is finite actual BF16 `[4,3,8]`, the row-major integers 0 through 95
divided by 128. All entries are exactly representable. W is BF16 identity
`[8,8]`. Index-one X differs by one; its projection contributes to the loss but
must produce no capture. The independent index-zero X/G oracles are created on
host and checked by exact shape/dtype/byte hashes. Canonical local X is
`[2,2,3,2]`; local G is `[2,3,8]`, including independently received records on
both model replicas. The existing collector checks singleton device ids,
zero dynamic index, finite BF16, every canonical slice and identical repeated
bytes. The probe additionally requires exactly all ten label/identity keys in every
execution, explicit independently computed rectangular slices, and repeated
observations.
Instrumented loss and every gradient must match untapped values byte for byte.

The target requires one process, exactly four local/global TPU devices and the
actual prior v5p device kind `TPU v5`. Scheduling must separately pin the same
v5p-8 target resource. The pinned target Python/JAX/JAXlib/NumPy/Equinox/Optax
versions and the three scientific environment variables are copied exactly
from the old frozen config. The default Shardy partitioner and default matmul
precision are retained. Actual imports must come from the executing source
root; all frozen scientific and candidate file hashes are checked.

## Evidence and release boundary

The command manifest is preparation only. Root must obtain fresh scientific
review, freeze and publicly publish the exact new source/config/command, and
record a dispatch release before deciding whether to submit this tiny gate.
Only root owns any allocation or job operation. No commit, push, submission,
job query, storage access, dependency/environment change, full-model attempt,
or public update is performed by this preparation.

Only compact complete JSON and complete tiny StableHLO/optimized HLO are
written. There are no array, checkpoint, profile, or model files. Hard caps are
2 MiB per artifact and 16 MiB total; oversized artifacts fail, without
truncation. The callback capture view may induce resharding. Successful
capture proves canonical logical shard bytes, not original operand buffer
layout, uninstrumented HLO equivalence, scientific cause, full-model parity,
trajectory, performance, or adoption.

New-fixture development exposed a wrong X element-count literal, an
out-sharding argument inappropriate for Auto axes, and an execution-count
assertion. These fixture setup/bookkeeping mistakes were corrected before
freezing; they are not production defects or TPU results. The original source
A failure remains independently preserved by the parent.

## Targeted review correction

The first preparation and CPU receipts are preserved separately. Fresh review
identified two source-level release blockers: aggregate callback counts could
mask a missing execution, and an internal Python alarm was insufficient for a
hard worker deadline. No observed numerical failure was inferred from those
findings. This revision records complete per-execution captures and requires
the external supervisor in both prepared CPU and target commands.

The CPU-only negative fixture uses the same actual worker, compiled handles,
collector implementation, runtime, and source guards. It deliberately sends
one identical G/ordinal-3 duplicate in the first composition execution and
omits that record from the second. The second execution must fail its own
collector finish despite the earlier duplicates. The separate short supervisor
timeout control kills a sleeping Python child and requires timeout exit 124
and worker termination by SIGKILL. These controls establish gate behavior on
CPU; they do not establish TPU callback support.
