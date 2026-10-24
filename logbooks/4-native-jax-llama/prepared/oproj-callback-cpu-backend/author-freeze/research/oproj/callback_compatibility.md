# Tiny callback compatibility gate

This new isolated research preparation starts from published callback-control
source `ed17de3657fc779e5ed01df4348f7a60fcd813ef`. Its only proposed target
scientific environment change is `JAX_PLATFORMS=tpu,cpu`, previously `tpu`.
The hypothesis is that explicitly initializing a host CPU backend permits the
existing TPU host callbacks to execute. It is not a claim of TPU compatibility
or fidelity to the original ambient backend list.

The parent-owned terminal evidence for the prior tiny job reports that its
first TPU transport compiled, then execution failed with
`jax.debug.callback failed to find a local CPU device to place the inputs on`.
The task failed after 32.59 seconds with exit 1 and no preemptions, before the
first transport completed. No complete X/G compatibility or custom-VJP result
was established. This preparation does not operate or repeat that job.
Its failure, source, configuration, commands and receipts remain preserved.

Source A `32f42f9a3601258123d316cc03c25b05add7a9c2` remains the earlier
full-model transport failure. Its untouched 1.531B anchor reproduced the
historical 255-gradient and loss metrics, while its instrumented legacy TPU
lowering failed with `Partitioned callback not implemented on tpu backend`.
The inherited inner `partitioned=False` callback adaptation from ed17 is
unchanged here. `capture.py`, its actual Collector/emit/make_tap, global and
per-device callbacks, full-manual shard map, ids, dynamic index-zero filter,
and identity custom VJP are byte-identical to ed17. All production scientific
source remains at the frozen hashes from
`2b17db78aedc0a603664238fbfa0bc0c68542d2b`.

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
transport and composition execution, explicit independently computed rectangular
slices, and repeated observations.
Instrumented loss and every gradient must match untapped values byte for byte.

The target requires one process, exactly four default local/global devices,
default backend `tpu`, and actual prior v5p device kind `TPU v5`. After explicitly
initializing the CPU backend, the default global/local device lists must equal
the explicitly requested TPU device lists. Actual local CPU devices must also
exist. `jax_default_device`, deprecated `jax_platform_name`, and deprecated
`jax_xla_backend` overrides are forbidden. The actual JAX platform configuration
must equal the frozen `tpu,cpu` ambient list. The report records these values,
the explicit TPU devices and available host CPU devices. Scheduling must
separately pin the same v5p-8 target resource. These guards reject CPU fallback;
the CPU devices do not contribute to the four default TPU-device count.

The pinned target Python/JAX/JAXlib/NumPy/Equinox/Optax versions, XLA_FLAGS and
LIBTPU_INIT_ARGS remain unchanged from ed17. CPU mode retains `JAX_PLATFORMS=cpu`
and exactly four CPU devices, with the same finite BF16 fixture and checks.
Default Shardy, default matmul precision, source/runtime/import guards and all
numerical/completeness criteria remain unchanged. Actual imports must come
from this source root; all frozen scientific and research file hashes are
checked. No dependencies, locks, tolerances, or markers change.

## Evidence and release boundary

The command manifest is preparation only. Root must obtain fresh scientific
review, freeze and publicly publish the exact new source/config/command, and
record a dispatch release before deciding whether to submit this tiny gate.
Only root owns any allocation or job operation. No commit, push, submission,
job query, storage access, dependency change, full-model attempt,
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

## Scientific environment boundary

Adding the CPU backend changes the ambient backend list and requires its own
source/configuration identity and review. CPU preflight only checks that the
unchanged tiny controls still execute on CPU; it does not simulate the two
backend TPU runtime. A later successful tiny `tpu,cpu` gate would establish
compatibility only for that tested environment and composition. It would not
retroactively validate ed17 or the earlier full-model attempt.

Before any later full-model instrumentation under `tpu,cpu`, a separately
authorized untouched legacy/native run must reproduce the complete historical
255-gradient and loss metric comparison and the cross-branch layout control in
this new environment. The original full-model fixture, source, runtime,
completeness and decision requirements remain in force. This preparation does
not authorize that full-model run or its instrumented successor. Root alone
owns any new job name, bootstrap, publication, release and single-dispatch
wrapper after fresh review.
