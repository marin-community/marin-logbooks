# Full Trainer Stage A under an explicit host CPU backend

This preparation descends from the tiny TPU callback control source
`0be2ab8574125f932b18ab8ec5d04a564b432be5` (S2). The parent-owned terminal audit
records a successful tiny transport/custom-VJP/scan/remat compatibility gate
with actual `tpu,cpu`, four default TPU v5 devices, a host CPU backend, both mesh
modes, six compilations and eight executions. That evidence does not establish
full-model historical replay, instrumentation identity, numerical cause,
trajectory, performance or adoption.

The existing full Trainer Stage A child now calls S2's actual
`callback_compatibility.provenance(config, mode)`. A narrow dictionary bridge
maps the old full-diagnostic field names to the validated helper's exact
runtime/environment/source fields. The child verifies the helper import path
and file hash and records its actual provenance. The old sixteen scientific
source hashes, imported model/benchmark paths and hashes, and research-file
hash checks remain required. The helper is byte-identical to S2 and requires
actual CPU availability, effective backend configuration matching the frozen
list, no default-device/deprecated backend override, default TPU and explicit
TPU device-list equality, four local/global TPU v5 devices, one process/index
zero, default matmul precision and Shardy. CPU mode retains four CPUs.

The target remains the original BF16 1,531,037,696-parameter full Trainer
fixture: batch 8, sequence 2048, 28 layers, width 2048, the original heads,
optimizer, parameter/batch hashes, seed, attention backend and 2x2 data/model
mesh. The new external config copies the complete original 255-tensor
historical gradient comparison and loss metrics unchanged. The only target
scientific environment delta from source A is `JAX_PLATFORMS=tpu,cpu`; other
runtime/environment fields and all numerical criteria are unchanged.

The controller launches a fresh untouched child first. Its legacy/native run
must reproduce the original fixture identities, all historical gradient/loss
metrics and cross-branch physical layout controls under the new environment.
Failure stops before launching the instrumented child. Both target variants
must retain all 255 canonical gradient tensors and all 12 raw gradient leaves.
The instrumented child uses the same ordinary contraction and full Trainer,
with the actual S2 capture module unchanged: global False transport, existing
full-manual local False transport, per-device collector True identity, actual
ids, dynamic scan-index-zero filtering and unchanged identity custom VJP.

Before interpreting X/G equality or selecting its research branch, the existing
controller compares untouched and instrumented loss, every canonical gradient
hash, all raw gradient byte identities, and all physical manifests for each
variant. Any mismatch is fatal. Only then are canonical global/local X/G
bytes and capture-view placement compared. Every actual device record,
including both model replicas of G, finite actual BF16, canonical shard-byte
checks and identical repeated/rematerialized records remain mandatory. These
views can induce resharding and do not prove original operand-buffer tiling
or a contraction/reduction cause.

Local validation runs the unchanged small full-controller CPU fixture: batch
4, sequence 8, three layers, width 32, BF16, vanilla attention, both mesh modes
and the same numerical controls. Existing deliberate gradient-perturbation,
duplicate-disagreement and nonfinite probes check meaningful failure paths.
The existing full-target abstract constructor/Trainer trace checks shapes only,
without lowering, compiling, executing or allocating the full model.

This is preparation only. All old sources, frozen configs and failed-attempt
records remain unchanged. No production math, dependency, lock, tolerance,
marker, original host runner or hardcoded backup URI changes. Root owns any
new bootstrap, public config/launcher/source pin, external process-group
wall deadline, single-attempt dispatch and bounded archive. The legacy
host_runner.py is retained and hash-checked but must not be used for this new
control. No storage/client/catalog/job action is performed here. Target work
requires separate fresh scientific review and root release before commit,
publication or any dispatch. CPU success and abstract tracing are not target
numerical acceptance evidence.
